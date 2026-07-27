import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// WP-GS0 — FROZEN GOLDEN-TRACE regression net around the current (pristine-HEAD) search behavior.
///
/// Purpose: later work packages will modify `Sources/RinGoEngine/Search.swift` behind a default-OFF
/// flag (graph search / transpositions). This suite pins the exact, observable behavior of the
/// CURRENT default (OFF) search on a set of fixed positions/visit levels so that any later change to
/// the OFF path is caught. It is a pure regression baseline: it asserts "identical to HEAD", never
/// "correct go".
///
/// ## What is captured per trace (see `GoldenTrace`)
/// - `bestMove` — the chosen root move (GTP coordinate string). EXACT.
/// - `visitsByMove` — the full root-child visit distribution, canonicalized to (visits desc, move
///   asc) so the comparison is independent of `Array.sort` tie-order. EXACT.
/// - `rootVisits`, `visitsAchieved`, `playouts` — playout accounting (`playouts == visitsAchieved`,
///   the visits completed by this call). EXACT.
/// - `whiteWinProb`, `whiteScoreMean` — root value heads (white perspective), rounded and compared
///   within a documented tolerance (see below).
/// - `evalRequestHashes`, `evalRequestSymmetries` — the exact ordered stream of NN-evaluator
///   requests (position hash via `NNInputs.getHash`, plus each request's `params.symmetry`),
///   captured test-side by wrapping the evaluator (`RecordingEvaluator`). This reflects the precise
///   leaf-expansion/descent order the search took. EXACT.
/// - `rootBoardHash`, `nextPlayer`, and the config knobs — integrity fields: if the position factory
///   or a config knob changes without re-recording, COMPARE fails loudly and tells you to re-record.
///
/// ## Tolerances
/// - `bestMove`, `visitsByMove`, playout counts, and the whole evaluator-request stream are EXACT
///   (fixed seed, fp32, single sequential search actor). The determinism of these fields at these
///   settings is verified directly by RECORD mode (each config is searched twice and the exact
///   fields must match before a fixture is written).
/// - `whiteWinProb`: stored rounded to `winProbDecimals` (6) places; compared within
///   `winProbTolerance` (1e-6).
/// - `whiteScoreMean`: stored rounded to `scoreDecimals` (4) places; compared within `scoreTolerance`
///   (1e-4). Score is in board points, so 1e-4 is a tight bound.
///
/// ## RECORD vs COMPARE
/// - COMPARE (default, part of `make check`): each config is searched once and every field is
///   asserted against the committed `GoldenTraces/<name>.json`.
/// - RECORD (`KATAGO_GOLDEN_RECORD=1`): regenerates every fixture in the source tree. Before writing,
///   each config is searched TWICE and the exact fields must agree (an in-run reproducibility gate);
///   the floats must agree within tolerance. Regenerate with, e.g.:
///     `KATAGO_GOLDEN_RECORD=1 swift test --filter GoldenTraceTests`
///   Fixtures are read/written relative to THIS source file (`#filePath`), so RECORD updates the
///   committed copies directly (the same pattern `NNInputsV7GoldenTests` uses for its `Goldens/`).
///
/// ## Signals NOT captured (would require editing `Sources/`, which WP-GS0 forbids)
/// - Per-visit internal PUCT selection scores and per-node running utility/variance stats
///   (`SearchNode` is file-private). The observable end-state (`visitsByMove` + root value) plus the
///   ordered evaluator-request stream pin the descent/selection behavior tightly without them.
/// - The raw `SearchRandom` stream positions (root-noise / final-sampling / per-node symmetry). In
///   the OFF path traced here, root noise and selection temperature are off and symmetry mode is
///   `.off`, so those streams are not consumed in a way that affects output; each request's resolved
///   `params.symmetry` (always the unset sentinel here) IS captured. The `.random`-mode symmetry
///   stream is out of scope for this OFF baseline.
///
/// Uses the b6c96 test model at 9x9 fp32 (same seam as `SearchTests`/`RootSymmetryTests`; set
/// `KATAGO_MODELS_DIR`). Skips cleanly when the model fixture is absent.
final class GoldenTraceTests: XCTestCase {
    // MARK: - Constants / configuration

    private static let nnLen = 9
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"
    private static let recordEnvVar = "KATAGO_GOLDEN_RECORD"

    private static let winProbDecimals = 6
    private static let scoreDecimals = 4
    private static let winProbTolerance = 1e-6
    private static let scoreTolerance = 1e-4

    private static var isRecording: Bool {
        ProcessInfo.processInfo.environment[recordEnvVar] == "1"
    }

    // MARK: - Trace configs (the RECORD inputs — the golden JSONs hold the recorded outputs)

    private struct TraceConfig {
        var name: String
        var positionID: String
        var note: String
        var maxVisits: Int64
        var numSearchWorkers: Int
        var leafBatchSize: Int
    }

    /// N = 8 diverse fixed configurations: 5 distinct positions (empty, opening, tactical, two
    /// midgames), visit levels 64/128/200/256/400 (all <= 400 per the WP-GS0 GPU-contention rule),
    /// and both worker regimes — single-leaf MCTS (`numSearchWorkers == 1`, `batchTarget == 1`) and
    /// the production batched regime (`numSearchWorkers == 8`, `leafBatchSize == 16`,
    /// `batchTarget == 8`). Tracing both guards the OFF path at the batch width real genmove uses.
    private static let configs: [TraceConfig] = [
        TraceConfig(
            name: "empty-v64-w1",
            positionID: "empty",
            note: "Empty 9x9, black to move, 64 visits, single-leaf MCTS.",
            maxVisits: 64,
            numSearchWorkers: 1,
            leafBatchSize: 8
        ),
        TraceConfig(
            name: "empty-v256-w8",
            positionID: "empty",
            note: "Empty 9x9, black to move, 256 visits, production batched regime (batchTarget 8).",
            maxVisits: 256,
            numSearchWorkers: 8,
            leafBatchSize: 16
        ),
        TraceConfig(
            name: "opening-v64-w1",
            positionID: "opening7",
            note: "7-move opening, 64 visits, single-leaf MCTS.",
            maxVisits: 64,
            numSearchWorkers: 1,
            leafBatchSize: 8
        ),
        TraceConfig(
            name: "opening-v200-w8",
            positionID: "opening7",
            note: "7-move opening, 200 visits, production batched regime (batchTarget 8).",
            maxVisits: 200,
            numSearchWorkers: 8,
            leafBatchSize: 16
        ),
        TraceConfig(
            name: "opening-v400-w1",
            positionID: "opening7",
            note: "7-move opening, 400 visits, single-leaf MCTS (deepest single-leaf trace).",
            maxVisits: 400,
            numSearchWorkers: 1,
            leafBatchSize: 8
        ),
        TraceConfig(
            name: "tactical-v200-w1",
            positionID: "tactical",
            note: "Two-stone central group in atari, black to move, 200 visits, single-leaf MCTS.",
            maxVisits: 200,
            numSearchWorkers: 1,
            leafBatchSize: 8
        ),
        TraceConfig(
            name: "midgame-v200-w8",
            positionID: "midgame",
            note: "Asymmetric 10-move midgame, 200 visits, production batched regime (batchTarget 8).",
            maxVisits: 200,
            numSearchWorkers: 8,
            leafBatchSize: 16
        ),
        TraceConfig(
            name: "endgame-v128-w1",
            positionID: "endgame",
            note: "Fuller 14-move board, 128 visits, single-leaf MCTS.",
            maxVisits: 128,
            numSearchWorkers: 1,
            leafBatchSize: 8
        ),
    ]

    // MARK: - Golden record schema (Codable JSON)

    private struct VisitEntry: Codable, Equatable {
        var move: String
        var visits: Int64
    }

    private struct GoldenTrace: Codable, Equatable {
        // Identity / configuration.
        var name: String
        var note: String
        var positionID: String
        var nnLen: Int
        var precision: String
        var symmetryMode: String
        var randomSeed: UInt64
        var numSearchWorkers: Int
        var leafBatchSize: Int
        var maxVisits: Int64
        // Position identity (integrity cross-check).
        var rootBoardHash: String
        var nextPlayer: String
        // Recorded search outputs.
        var bestMove: String
        var rootVisits: Int64
        var visitsAchieved: Int64
        var playouts: Int64
        var visitsByMove: [VisitEntry]
        var whiteWinProb: Double
        var whiteScoreMean: Double
        // Evaluator request stream (test-side capture).
        var evalRequestCount: Int
        var evalRequestHashes: [String]
        var evalRequestSymmetries: [Int]
    }

    // MARK: - Per-config test entry points (thin wrappers, so a failure names one config)

    func testGoldenTraceEmptyV64W1() async throws {
        try await runOrRecord("empty-v64-w1")
    }

    func testGoldenTraceEmptyV256W8() async throws {
        try await runOrRecord("empty-v256-w8")
    }

    func testGoldenTraceOpeningV64W1() async throws {
        try await runOrRecord("opening-v64-w1")
    }

    func testGoldenTraceOpeningV200W8() async throws {
        try await runOrRecord("opening-v200-w8")
    }

    func testGoldenTraceOpeningV400W1() async throws {
        try await runOrRecord("opening-v400-w1")
    }

    func testGoldenTraceTacticalV200W1() async throws {
        try await runOrRecord("tactical-v200-w1")
    }

    func testGoldenTraceMidgameV200W8() async throws {
        try await runOrRecord("midgame-v200-w8")
    }

    func testGoldenTraceEndgameV128W1() async throws {
        try await runOrRecord("endgame-v128-w1")
    }

    // MARK: - Drivers

    private func runOrRecord(
        _ configName: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        try skipIfModelMissing()
        let config = try Self.config(named: configName)
        if Self.isRecording {
            try await record(config, file: file, line: line)
        } else {
            try await compare(config, file: file, line: line)
        }
    }

    private func compare(_ config: TraceConfig, file: StaticString, line: UInt) async throws {
        let golden = try loadGolden(config.name)
        let fresh = try await runSearch(config)
        assertMatches(fresh: fresh, golden: golden, file: file, line: line)
    }

    private func record(_ config: TraceConfig, file: StaticString, line: UInt) async throws {
        // Reproducibility gate: search twice and require the exact fields to agree before writing.
        let first = try await runSearch(config)
        let second = try await runSearch(config)
        assertDeterministic(first, second, config: config, file: file, line: line)
        try writeGolden(first)
    }

    // MARK: - Running one search under a recording evaluator

    private func runSearch(_ config: TraceConfig) async throws -> GoldenTrace {
        let (desc, base) = try Self.sharedEvaluator()
        let recorder = RecordingEvaluator(base: base)
        let (board, history, nextPlayer) = try Self.makePosition(config.positionID)

        var settings = SearchSettings()
        settings.numSearchWorkers = config.numSearchWorkers
        settings.leafBatchSize = config.leafBatchSize
        // Everything else stays at the GTP defaults for the OFF exploration path: symmetryMode .off,
        // randomSeed 0, selectionTemperature 0, rootDirichletAlpha 0, rootNoiseWeight 0,
        // rootNumSymmetries 1. (treeReuseEnabled is irrelevant: a single search never re-roots.)

        let search = Search(
            evaluator: recorder,
            modelDesc: desc,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            settings: settings,
            board: board,
            history: history,
            nextPlayer: nextPlayer
        )
        let result = try await search.runSearch(maxVisits: config.maxVisits)
        let stream = await recorder.snapshot()
        return Self.makeTrace(
            config: config,
            board: board,
            nextPlayer: nextPlayer,
            result: result,
            stream: stream
        )
    }

    private static func makeTrace(
        config: TraceConfig,
        board: Board,
        nextPlayer: Player,
        result: SearchResult,
        stream: (hashes: [String], symmetries: [Int])
    ) -> GoldenTrace {
        let xSize = board.xSize
        let ySize = board.ySize
        let visitsByMove = canonicalVisits(result.visitsByMove, xSize: xSize, ySize: ySize)
        return GoldenTrace(
            name: config.name,
            note: config.note,
            positionID: config.positionID,
            nnLen: nnLen,
            precision: "float32",
            symmetryMode: "off",
            randomSeed: 0,
            numSearchWorkers: config.numSearchWorkers,
            leafBatchSize: config.leafBatchSize,
            maxVisits: config.maxVisits,
            rootBoardHash: board.posHash.description,
            nextPlayer: playerName(nextPlayer),
            bestMove: Location.toString(result.bestMove, xSize, ySize),
            rootVisits: result.rootVisits,
            visitsAchieved: result.visitsAchieved,
            playouts: result.visitsAchieved,
            visitsByMove: visitsByMove,
            whiteWinProb: rounded(result.whiteWinProb, winProbDecimals),
            whiteScoreMean: rounded(result.whiteScoreMean, scoreDecimals),
            evalRequestCount: stream.hashes.count,
            evalRequestHashes: stream.hashes,
            evalRequestSymmetries: stream.symmetries
        )
    }

    /// (visits desc, move asc) — a total order independent of `Array.sort`'s tie handling and of
    /// child creation order, so the stored/compared distribution is canonical.
    private static func canonicalVisits(
        _ raw: [(loc: Loc, visits: Int64)],
        xSize: Int,
        ySize: Int
    ) -> [VisitEntry] {
        raw.map { VisitEntry(move: Location.toString($0.loc, xSize, ySize), visits: $0.visits) }
            .sorted { lhs, rhs in
                lhs.visits == rhs.visits ? lhs.move < rhs.move : lhs.visits > rhs.visits
            }
    }

    // MARK: - Assertions

    private func assertMatches(
        fresh: GoldenTrace,
        golden: GoldenTrace,
        file: StaticString,
        line: UInt
    ) {
        // Integrity: fixture and factory must describe the same experiment.
        XCTAssertEqual(fresh.positionID, golden.positionID, "positionID", file: file, line: line)
        XCTAssertEqual(
            fresh.rootBoardHash,
            golden.rootBoardHash,
            "root position changed — the position factory drifted; re-record with \(Self.recordEnvVar)=1",
            file: file,
            line: line
        )
        XCTAssertEqual(fresh.nextPlayer, golden.nextPlayer, "nextPlayer", file: file, line: line)
        XCTAssertEqual(fresh.numSearchWorkers, golden.numSearchWorkers, "numSearchWorkers", file: file, line: line)
        XCTAssertEqual(fresh.leafBatchSize, golden.leafBatchSize, "leafBatchSize", file: file, line: line)
        XCTAssertEqual(fresh.maxVisits, golden.maxVisits, "maxVisits", file: file, line: line)

        // EXACT behavioral surface.
        XCTAssertEqual(fresh.bestMove, golden.bestMove, "bestMove drift", file: file, line: line)
        XCTAssertEqual(fresh.rootVisits, golden.rootVisits, "rootVisits drift", file: file, line: line)
        XCTAssertEqual(fresh.visitsAchieved, golden.visitsAchieved, "visitsAchieved drift", file: file, line: line)
        XCTAssertEqual(fresh.playouts, golden.playouts, "playouts drift", file: file, line: line)
        XCTAssertEqual(fresh.visitsByMove, golden.visitsByMove, "visit distribution drift", file: file, line: line)

        // EXACT evaluator request stream.
        XCTAssertEqual(
            fresh.evalRequestCount,
            golden.evalRequestCount,
            "evaluator request count drift",
            file: file,
            line: line
        )
        XCTAssertEqual(
            fresh.evalRequestHashes,
            golden.evalRequestHashes,
            "evaluator request hash-stream drift (leaf expansion order changed)",
            file: file,
            line: line
        )
        XCTAssertEqual(
            fresh.evalRequestSymmetries,
            golden.evalRequestSymmetries,
            "evaluator request symmetry-stream drift",
            file: file,
            line: line
        )

        // Root value heads within documented tolerance.
        XCTAssertEqual(
            fresh.whiteWinProb,
            golden.whiteWinProb,
            accuracy: Self.winProbTolerance,
            "whiteWinProb drift",
            file: file,
            line: line
        )
        XCTAssertEqual(
            fresh.whiteScoreMean,
            golden.whiteScoreMean,
            accuracy: Self.scoreTolerance,
            "whiteScoreMean drift",
            file: file,
            line: line
        )
    }

    /// RECORD-mode reproducibility gate: two independent searches of the same config must agree
    /// exactly on every EXACT field (and within tolerance on the value heads) before a fixture is
    /// trusted. A failure here is the important WP-GS0 finding "these settings are not deterministic".
    private func assertDeterministic(
        _ first: GoldenTrace,
        _ second: GoldenTrace,
        config: TraceConfig,
        file: StaticString,
        line: UInt
    ) {
        let tag = "non-deterministic at config \(config.name)"
        XCTAssertEqual(first.bestMove, second.bestMove, "\(tag): bestMove", file: file, line: line)
        XCTAssertEqual(first.rootVisits, second.rootVisits, "\(tag): rootVisits", file: file, line: line)
        XCTAssertEqual(first.visitsAchieved, second.visitsAchieved, "\(tag): visitsAchieved", file: file, line: line)
        XCTAssertEqual(first.visitsByMove, second.visitsByMove, "\(tag): visitsByMove", file: file, line: line)
        XCTAssertEqual(first.evalRequestHashes, second.evalRequestHashes, "\(tag): eval hashes", file: file, line: line)
        XCTAssertEqual(
            first.evalRequestSymmetries,
            second.evalRequestSymmetries,
            "\(tag): eval symmetries",
            file: file,
            line: line
        )
        XCTAssertEqual(
            first.whiteWinProb,
            second.whiteWinProb,
            accuracy: Self.winProbTolerance,
            "\(tag): whiteWinProb",
            file: file,
            line: line
        )
        XCTAssertEqual(
            first.whiteScoreMean,
            second.whiteScoreMean,
            accuracy: Self.scoreTolerance,
            "\(tag): whiteScoreMean",
            file: file,
            line: line
        )
    }

    // MARK: - Fixture IO (read/write relative to this source file so RECORD updates the tree)

    private static func goldenDirectory() -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("GoldenTraces")
    }

    private func loadGolden(_ name: String) throws -> GoldenTrace {
        let url = Self.goldenDirectory().appendingPathComponent("\(name).json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GoldenTraceError(
                "Missing golden fixture \(url.path); regenerate with \(Self.recordEnvVar)=1 swift test --filter GoldenTraceTests"
            )
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(GoldenTrace.self, from: data)
    }

    private func writeGolden(_ trace: GoldenTrace) throws {
        let directory = Self.goldenDirectory()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(trace)
        try data.write(to: directory.appendingPathComponent("\(trace.name).json"))
        FileHandle.standardError.write(Data("Recorded golden trace: \(trace.name)\n".utf8))
    }

    // MARK: - Positions

    private static func makePosition(_ id: String) throws -> (Board, BoardHistory, Player) {
        switch id {
        case "empty":
            let board = Board(nnLen, nnLen)
            let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
            return (board, history, history.presumedNextMovePla)
        case "opening7":
            return try played([
                (2, 2, .black), (6, 6, .white), (6, 2, .black), (2, 6, .white),
                (4, 4, .black), (4, 2, .white), (3, 3, .black),
            ])
        case "midgame":
            return try played([
                (2, 2, .black), (6, 6, .white), (2, 6, .black), (6, 2, .white),
                (4, 4, .black), (3, 6, .white), (6, 4, .black), (2, 4, .white),
                (4, 6, .black), (4, 2, .white),
            ])
        case "endgame":
            return try played([
                (2, 2, .black), (6, 6, .white), (2, 6, .black), (6, 2, .white),
                (4, 4, .black), (4, 2, .white), (2, 4, .black), (6, 4, .white),
                (4, 6, .black), (3, 3, .white), (5, 5, .black), (3, 5, .white),
                (5, 3, .black), (3, 4, .white),
            ])
        case "tactical":
            // Two-stone central white group in atari, black to move (SearchTests puzzle 3 shape).
            let board = Board(nnLen, nnLen)
            let placements = [
                Move(loc: loc9(4, 4), pla: .white),
                Move(loc: loc9(4, 5), pla: .white),
                Move(loc: loc9(3, 4), pla: .black),
                Move(loc: loc9(5, 4), pla: .black),
                Move(loc: loc9(3, 5), pla: .black),
                Move(loc: loc9(5, 5), pla: .black),
                Move(loc: loc9(4, 6), pla: .black),
            ]
            guard board.setStonesFailIfNoLibs(placements) else {
                throw GoldenTraceError("tactical position setup failed (no-liberty placement)")
            }
            let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
            return (board, history, .black)
        default:
            throw GoldenTraceError("unknown position id \(id)")
        }
    }

    /// Plays a legal move sequence from the empty board via `makeBoardMoveTolerant`, throwing on any
    /// illegal move so a mis-typed fixture setup fails loudly rather than silently skipping a move.
    private static func played(_ moves: [(Int, Int, Player)]) throws -> (Board, BoardHistory, Player) {
        let board = Board(nnLen, nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        for (x, y, pla) in moves {
            guard history.makeBoardMoveTolerant(board, loc9(x, y), pla) else {
                throw GoldenTraceError("illegal setup move (\(x),\(y)) for \(pla)")
            }
        }
        return (board, history, history.presumedNextMovePla)
    }

    private static func loc9(_ x: Int, _ y: Int) -> Loc {
        Location.getLoc(x, y, nnLen)
    }

    // MARK: - Small helpers

    private static func config(named name: String) throws -> TraceConfig {
        guard let config = configs.first(where: { $0.name == name }) else {
            throw GoldenTraceError("no TraceConfig named \(name)")
        }
        return config
    }

    private static func playerName(_ player: Player) -> String {
        player == .black ? "black" : "white"
    }

    private static func rounded(_ value: Double, _ decimals: Int) -> Double {
        let factor = pow(10.0, Double(decimals))
        return (value * factor).rounded() / factor
    }

    // MARK: - Shared evaluator (built once — fp32 warmup is expensive, and this shares GPU nicely)

    private static let sharedEvaluatorResult: Result<(ModelDesc, NNEvaluator), Error> = Result {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"] ?? defaultModelsDirectory
        let path = URL(fileURLWithPath: directory).appendingPathComponent(modelFile).path
        let desc = try ModelDesc.loadFromFileMaybeGZipped(path)
        // maxBatchSize 16 -> physical buckets [1,2,4,8], covering batchTarget 1 (w1) and 8 (w8).
        let evaluator = try NNEvaluator(
            desc: desc,
            nnXLen: nnLen,
            nnYLen: nnLen,
            precision: .float32,
            maxBatchSize: 16
        )
        return (desc, evaluator)
    }

    private static func sharedEvaluator() throws -> (ModelDesc, NNEvaluator) {
        try sharedEvaluatorResult.get()
    }

    private func skipIfModelMissing() throws {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"] ?? Self.defaultModelsDirectory
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw XCTSkip("KataGo model fixtures are missing; set KATAGO_MODELS_DIR (checked \(directory))")
        }
        let path = URL(fileURLWithPath: directory).appendingPathComponent(Self.modelFile).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("KataGo model fixture is missing: \(path)")
        }
    }
}

private struct GoldenTraceError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) {
        self.description = description
    }
}

/// Test-side decorator over any `NNEvaluating`: records the ordered stream of requests (position
/// hash + carried symmetry) each `evaluate` receives, then delegates unchanged. Recording cannot
/// perturb search numerics (the wrapped evaluator sees identical requests and returns identical
/// outputs), so it captures the golden's evaluator-request signal without any `Sources/` change.
private actor RecordingEvaluator: NNEvaluating {
    private let base: any NNEvaluating
    private var hashes: [String] = []
    private var symmetries: [Int] = []

    init(base: any NNEvaluating) {
        self.base = base
    }

    func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput] {
        for request in requests {
            let hash = NNInputs.getHash(request.board, request.history, request.nextPlayer, request.params)
            hashes.append(hash.description)
            symmetries.append(request.params.symmetry)
        }
        return try await base.evaluate(requests)
    }

    func preWarm(_ requests: [NNRequest]) async throws {
        // Warmup is not part of the golden stream; delegate without recording (Search never calls it).
        try await base.preWarm(requests)
    }

    func snapshot() -> (hashes: [String], symmetries: [Int]) {
        (hashes, symmetries)
    }
}
