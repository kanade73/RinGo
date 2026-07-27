import Foundation
import MLX
import RinGoCore
import RinGoEngine
import RinGoModel

struct BenchmarkOptions {
    var model = ""
    var batchSizes: [Int] = [1, 8, 32]
    var precision: KataGoNetwork.Precision = .float16
    var secondsPerPoint: Double = 5
    var nnLen = 19
    var compareSingle = false
    var gpuCacheLimitMB: Int?
}

/// `ringo benchmark`: measures steady-state `NNEvaluator` throughput per batch size,
/// reporting compile warmup cost separately (see design.md "MLX discipline" point 3) and,
/// with `-compare-single`, contrasting against a direct (non-batched, non-actor, uncompiled)
/// `KataGoNetwork.forward` loop so the evaluator's own overhead is visible.
enum Benchmark {
    static func main(_ arguments: [String]) async throws {
        try await run(parse(arguments))
    }

    static func run(_ options: BenchmarkOptions) async throws {
        status("Loading model \(options.model)...")
        let loadStart = Date()
        let desc = try ModelDesc.loadFromFileMaybeGZipped(options.model)
        status("Loaded \(desc.name) (\(desc.shortInfoString)) in \(formatSeconds(Date().timeIntervalSince(loadStart)))")

        let maxBucket = nextPowerOfTwo(max(options.batchSizes.max() ?? 1, 1))
        let evaluator = try NNEvaluator(
            desc: desc,
            nnXLen: options.nnLen,
            nnYLen: options.nnLen,
            precision: options.precision,
            maxBatchSize: maxBucket,
            gpuCacheLimitMB: options.gpuCacheLimitMB
        )
        let cacheLimitBytes = MLX.Memory.cacheLimit

        let poolSize = max(256, maxBucket * 4)
        status("Building a pool of \(poolSize) benchmark positions (nnLen \(options.nnLen))...")
        let positions = try positionPool(nnLen: options.nnLen, targetSize: poolSize)
        guard !positions.isEmpty else {
            throw CLIError(description: "Could not build any benchmark positions")
        }
        // Prefill each pool position's V7 feature planes once, up front, via NNRequest.prefilled
        // -- exactly the seam NNEvaluator exposes so search worker threads can pay `fillRowV7`'s
        // cost off the hot path (see NNEvaluator.swift's doc comment). Without this, a benchmark
        // loop that keeps redrawing from a few hundred cached positions would spend most of its
        // time recomputing *identical* feature planes rather than measuring NN throughput --
        // `fillRowV7` briefly dominated an early version of this benchmark's numbers, which is
        // the reason this seam exists in the first place.
        let pool = positions.map { position -> NNRequest in
            let row = NNInputs.fillRowV7(
                position.board,
                position.history,
                position.nextPlayer,
                nnXLen: options.nnLen,
                nnYLen: options.nnLen
            )
            return NNRequest(
                board: position.board,
                history: position.history,
                nextPlayer: position.nextPlayer,
                prefilled: .init(spatial: row.spatial, global: row.global)
            )
        }

        print(header(desc: desc, options: options, maxBucket: maxBucket, cacheLimitBytes: cacheLimitBytes))
        print(row(title: "batch", evalsPerSecond: "evals/s", msPerBatch: "ms/batch", warmupMs: "warmup_ms"))

        var cursor = PoolCursor(pool: pool)
        for batchSize in options.batchSizes {
            let warmupStart = Date()
            _ = try await evaluator.evaluate(cursor.next(batchSize))
            let warmupMs = Date().timeIntervalSince(warmupStart) * 1000
            status("  batch \(batchSize): compile+warmup call took \(String(format: "%.1f", warmupMs)) ms")

            let point = try await measure(seconds: options.secondsPerPoint) {
                try await evaluator.evaluate(cursor.next(batchSize)).count
            }
            let evalsPerSecond = point.elapsedSeconds > 0 ? Double(point.totalUnits) / point.elapsedSeconds : 0
            let msPerBatch = point.iterations > 0 ? (point.elapsedSeconds * 1000) / Double(point.iterations) : 0
            print(row(
                title: String(batchSize),
                evalsPerSecond: String(format: "%.1f", evalsPerSecond),
                msPerBatch: String(format: "%.3f", msPerBatch),
                warmupMs: String(format: "%.1f", warmupMs)
            ))
        }

        if options.compareSingle {
            status("Measuring direct (non-batched, non-evaluator) single-position forward for contrast...")
            let network = try KataGoNetwork(
                desc: desc,
                nnXLen: options.nnLen,
                nnYLen: options.nnLen,
                precision: options.precision
            )
            let point = try measureSync(seconds: options.secondsPerPoint) {
                let position = cursor.next(1)[0]
                try directForwardOnce(network: network, desc: desc, request: position, nnLen: options.nnLen)
            }
            let evalsPerSecond = point.elapsedSeconds > 0 ? Double(point.totalUnits) / point.elapsedSeconds : 0
            print("")
            print(
                "compare-single (direct KataGoNetwork.forward, batch=1, no evaluator/compile): "
                    + "\(String(format: "%.1f", evalsPerSecond)) evals/s"
            )
        }
    }

    // MARK: - Timing

    private struct BenchmarkPoint {
        let iterations: Int
        let totalUnits: Int
        let elapsedSeconds: Double
    }

    private static func measure(
        seconds: Double,
        _ body: () async throws -> Int
    ) async rethrows -> BenchmarkPoint {
        var iterations = 0
        var totalUnits = 0
        let deadline = Date().addingTimeInterval(seconds)
        let start = Date()
        while Date() < deadline {
            totalUnits += try await body()
            iterations += 1
        }
        return BenchmarkPoint(
            iterations: iterations,
            totalUnits: totalUnits,
            elapsedSeconds: Date().timeIntervalSince(start)
        )
    }

    private static func measureSync(
        seconds: Double,
        _ body: () throws -> Void
    ) rethrows -> BenchmarkPoint {
        var iterations = 0
        let deadline = Date().addingTimeInterval(seconds)
        let start = Date()
        while Date() < deadline {
            try body()
            iterations += 1
        }
        return BenchmarkPoint(
            iterations: iterations,
            totalUnits: iterations,
            elapsedSeconds: Date().timeIntervalSince(start)
        )
    }

    /// Direct single-position forward, mirroring `ringo rawnn`'s inner logic but without
    /// postprocessing (the benchmark only needs the network's raw compute cost forced to
    /// materialize once, matching design.md's "one eval per batch" discipline even here). Takes
    /// an already-`prefilled` request (see `run`) so this is an apples-to-apples contrast against
    /// the batched/compiled path: both measure NN compute, neither re-pays `fillRowV7`.
    private static func directForwardOnce(
        network: KataGoNetwork,
        desc: ModelDesc,
        request: NNRequest,
        nnLen: Int
    ) throws {
        guard let row = request.prefilled else {
            throw CLIError(description: "Internal error: benchmark pool request was not prefilled")
        }
        let spatial = MLXArray(row.spatial, [1, nnLen, nnLen, desc.numInputChannels])
        let global = MLXArray(row.global, [1, desc.numInputGlobalChannels])
        let outputs = network.forward(spatial: spatial, global: global)
        eval([
            outputs.policySpatialLogits,
            outputs.policyPassLogits,
            outputs.valueLogits,
            outputs.scoreValues,
            outputs.ownership,
        ])
    }

    // MARK: - Position pool

    private struct PoolCursor {
        let pool: [NNRequest]
        var offset = 0

        mutating func next(_ count: Int) -> [NNRequest] {
            var result = [NNRequest]()
            result.reserveCapacity(count)
            for _ in 0 ..< count {
                result.append(pool[offset % pool.count])
                offset += 1
            }
            return result
        }
    }

    /// A mix of real positions (replaying the repo's fixture SGFs, gracefully skipped if not
    /// found — the benchmark is expected to run from the repo root) and deterministic random
    /// legal playouts (for volume/diversity beyond what the small fixture set provides).
    private static func positionPool(
        nnLen: Int,
        targetSize: Int
    ) throws -> [(board: Board, history: BoardHistory, nextPlayer: Player)] {
        var pool = [(board: Board, history: BoardHistory, nextPlayer: Player)]()
        for directory in ["Tests/RinGoCoreTests/Fixtures", "Tests/RinGoModelTests/Fixtures"] {
            guard let files = try? FileManager.default.contentsOfDirectory(atPath: directory) else { continue }
            for file in files.sorted() where file.hasSuffix(".sgf") {
                let url = URL(fileURLWithPath: directory).appendingPathComponent(file)
                guard let game = try? SGFReader.load(url), game.boardXSize <= nnLen,
                      game.boardYSize <= nnLen else { continue }
                for moveNumber in 0 ... game.moves.count {
                    guard let position = try? game.position(at: moveNumber) else { continue }
                    pool.append(position)
                }
            }
        }
        status("  \(pool.count) positions replayed from fixture SGFs")

        var seed: UInt64 = 0xB16B_00B5_1234_5678
        while pool.count < targetSize {
            pool.append(contentsOf: randomPlayoutPositions(
                nnLen: nnLen,
                count: min(64, targetSize - pool.count),
                seed: seed
            ))
            seed &+= 1
        }
        return pool
    }

    /// Never voluntarily passes (only when no legal non-pass move exists), so a single walk
    /// comfortably yields dozens of distinct, increasing-move-number legal positions. Prefers
    /// moves that don't immediately self-atari (a cheap O(1) neighbor check, not a full replay):
    /// pure uniform-random legal play piles up an unrealistic density of 1-2-liberty chains as
    /// the game progresses, which is not representative of real search-leaf positions and makes
    /// `fillRowV7`'s ladder search (a real, necessary cost — see `NNInputs.iterLadders`)
    /// disproportionately expensive for these specific synthetic positions.
    private static func randomPlayoutPositions(
        nnLen: Int,
        count: Int,
        seed: UInt64
    ) -> [(board: Board, history: BoardHistory, nextPlayer: Player)] {
        var rng = SplitMix64(seed: seed)
        let board = Board(nnLen, nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        var nextPlayer = Player.black
        var positions = [(board: Board, history: BoardHistory, nextPlayer: Player)]()
        positions.reserveCapacity(count)
        var safety = 0
        while positions.count < count, safety < count * 200 {
            safety += 1
            positions.append((board.copy(), history.copy(), nextPlayer))
            let candidates = board.playableLocations().filter { history.isLegal(board, $0, nextPlayer) }
            let safeCandidates = candidates.filter { board.getBoundNumLibertiesAfterPlay($0, nextPlayer).lower >= 2 }
            let pickFrom = safeCandidates.isEmpty ? candidates : safeCandidates
            var moved = false
            if !pickFrom.isEmpty {
                let index = Int(rng.next() % UInt64(pickFrom.count))
                moved = history.makeBoardMoveTolerant(board, pickFrom[index], nextPlayer)
            }
            if !moved {
                _ = history.makeBoardMoveTolerant(board, Board.passLoc, nextPlayer)
            }
            nextPlayer = nextPlayer.opponent
        }
        return positions
    }

    private struct SplitMix64 {
        private var state: UInt64
        init(seed: UInt64) {
            state = seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    // MARK: - Formatting

    private static func header(
        desc: ModelDesc,
        options: BenchmarkOptions,
        maxBucket: Int,
        cacheLimitBytes: Int
    ) -> String {
        let precisionLabel = options.precision == .float16 ? "fp16" : "fp32"
        let cacheLabel = options.gpuCacheLimitMB != nil
            ? "\(formatBytes(cacheLimitBytes)) (user-set)"
            : "\(formatBytes(cacheLimitBytes)) (MLX default)"
        let nnLen = options.nnLen
        return """

        Model: \(desc.name) (\(desc.shortInfoString), modelVersion=\(desc.modelVersion))
        Precision: \(precisionLabel)   nnLen: \(nnLen)   bucket top (maxBatchSize): \(maxBucket)   GPU cache limit: \(
            cacheLabel
        )
        """
    }

    private static func row(title: String, evalsPerSecond: String, msPerBatch: String, warmupMs: String) -> String {
        pad(title, 8) + pad(evalsPerSecond, 14) + pad(msPerBatch, 12) + pad(warmupMs, 12)
    }

    private static func pad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text + " " : String(repeating: " ", count: width - text.count) + text
    }

    private static func formatBytes(_ bytes: Int) -> String {
        String(format: "%.0f MB", Double(bytes) / (1024 * 1024))
    }

    private static func formatSeconds(_ seconds: Double) -> String {
        String(format: "%.2fs", seconds)
    }

    private static func nextPowerOfTwo(_ value: Int) -> Int {
        var result = 1
        while result < value {
            result <<= 1
        }
        return result
    }

    private static func status(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    // MARK: - Argument parsing

    static func parse(_ arguments: [String]) throws -> BenchmarkOptions {
        var options = BenchmarkOptions()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            if flag == "-compare-single" {
                options.compareSingle = true
                index += 1
                continue
            }
            guard index + 1 < arguments.count else {
                throw CLIError(description: "Missing value for \(flag)")
            }
            let value = arguments[index + 1]
            switch flag {
            case "-model": options.model = value
            case "-batch-sizes": options.batchSizes = try parseBatchSizes(value)
            case "-precision":
                switch value {
                case "fp32": options.precision = .float32
                case "fp16": options.precision = .float16
                default: throw CLIError(description: "Precision must be fp32 or fp16")
                }
            case "-seconds-per-point":
                guard let seconds = Double(value), seconds > 0, seconds.isFinite else {
                    throw CLIError(description: "Invalid -seconds-per-point '\(value)'")
                }
                options.secondsPerPoint = seconds
            case "-nnlen":
                guard let length = Int(value), length > 0, length <= Board.maxLen else {
                    throw CLIError(description: "Invalid -nnlen '\(value)'")
                }
                options.nnLen = length
            case "-gpu-cache-limit-mb":
                guard let megabytes = Int(value), megabytes >= 0 else {
                    throw CLIError(description: "Invalid -gpu-cache-limit-mb '\(value)'")
                }
                options.gpuCacheLimitMB = megabytes
            default: throw CLIError(description: "Unknown option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard !options.model.isEmpty else {
            throw CLIError(description: usage)
        }
        return options
    }

    private static func parseBatchSizes(_ value: String) throws -> [Int] {
        let parsed = value.split(separator: ",").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard !parsed.isEmpty, parsed.allSatisfy({ ($0 ?? 0) > 0 }) else {
            throw CLIError(description: "Invalid -batch-sizes list '\(value)' (expected e.g. 1,8,32,64)")
        }
        return parsed.map { $0! }
    }

    static let usage = """
    Usage: ringo benchmark -model <file> [-batch-sizes 1,8,32,64] [-precision fp16|fp32] \
      [-seconds-per-point 5] [-nnlen 19] [-compare-single] [-gpu-cache-limit-mb <n>]
    """
}
