import Foundation
import RinGoCore
import RinGoEngine
import RinGoModel

struct SelfPlayOptions: Equatable {
    var model = ""
    var games = 0
    var outputDirectory = ""
    var visits: Int64 = 200
    var precision: KataGoNetwork.Precision = .float16
    var temperature = 0.9
    var temperatureMoves = 8
    var dirichletAlpha = 0.15
    var noiseWeight = 0.25
    var seed: UInt64 = 1
    var komi: Float = 7
    var size = 9
    var maxBatchSize = 16
    /// WP-9: when set, ALSO emit RinGoData v2 training shards into this directory (AlphaZero
    /// targets straight from self-play). SGF output continues regardless.
    var writeTrainingDirectory: String?
    var shardSize = 4096
}

enum SelfPlayCommand {
    static let usage = """
    Usage: ringo selfplay -model <file> -games N -out <sgfdir> [-visits 200] \
      [-precision fp16|fp32] [-temp 0.9] [-temp-moves 8] [-dirichlet 0.15] \
      [-noise-weight 0.25] [-seed 1] [-komi 7] [-size 9] [-max-batch-size 16] \
      [-write-training <shard-dir>] [-shard-size 4096]
    """

    static func main(_ arguments: [String]) async throws {
        let date = ISO8601DateFormatter().string(from: Date()).prefix(10)
        _ = try await run(parse(arguments), date: String(date))
    }

    @discardableResult
    static func run(_ options: SelfPlayOptions, date: String) async throws -> [URL] {
        let start = ProcessInfo.processInfo.systemUptime
        let description = try ModelDesc.loadFromFileMaybeGZipped(options.model)
        let evaluator = try NNEvaluator(
            desc: description,
            nnXLen: options.size,
            nnYLen: options.size,
            precision: options.precision,
            maxBatchSize: options.maxBatchSize
        )
        let rules = try Rules.parseRulesWithoutKomi("Chinese", komi: options.komi)
        try await preWarm(
            evaluator: evaluator,
            rules: rules,
            boardSize: options.size,
            maxBatchSize: options.maxBatchSize
        )

        let outputDirectory = URL(fileURLWithPath: options.outputDirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        // WP-9: optional RinGoData v2 training-shard sink. When active, each game's positions are
        // converted to AlphaZero samples and buffered incrementally into shard files (memory-bounded
        // -- at most one shard's worth of samples resident). A provenance JSON is written up front.
        var shardWriter: SelfPlayShardWriter?
        if let trainingPath = options.writeTrainingDirectory {
            let trainingDirectory = URL(fileURLWithPath: trainingPath, isDirectory: true)
            try FileManager.default.createDirectory(at: trainingDirectory, withIntermediateDirectories: true)
            try writeTrainingConfig(to: trainingDirectory, options: options, modelName: description.name)
            shardWriter = SelfPlayShardWriter(
                directory: trainingDirectory, nnLen: options.size,
                shardSize: options.shardSize, seed: options.seed
            )
        }

        var files = [URL]()
        var totalMoves = 0
        var blackWins = 0
        var whiteWins = 0
        var draws = 0
        for gameIndex in 0 ..< options.games {
            var settings = searchSettings(options: options, gameIndex: gameIndex)
            settings.resignEnabled = false
            let configuration = SelfPlayGameConfiguration(
                boardSize: options.size,
                visits: options.visits,
                temperature: options.temperature,
                temperatureMoves: options.temperatureMoves,
                rules: rules,
                searchSettings: settings
            )
            let result = try await SelfPlayGameGenerator.playRecording(
                evaluator: evaluator,
                nnXLen: options.size,
                nnYLen: options.size,
                precision: options.precision,
                configuration: configuration,
                recordTrainingData: shardWriter != nil
            )
            let history = result.history
            totalMoves += history.moveHistory.count
            switch history.winner {
            case .black?: blackWins += 1
            case .white?: whiteWins += 1
            case nil: draws += 1
            }
            if shardWriter != nil {
                let samples = SelfPlayGameGenerator.trainingSamples(from: result, nnLen: options.size)
                try shardWriter?.append(samples)
            }
            let name = "ringo \(description.name)"
            let sgf = try SGFWriter.write(
                history: history,
                blackName: name,
                whiteName: name,
                date: date
            )
            let file = outputDirectory.appendingPathComponent("game-\(options.seed)-\(gameIndex).sgf")
            try Data(sgf.utf8).write(to: file, options: .atomic)
            files.append(file)
        }
        try shardWriter?.finish()

        let elapsed = ProcessInfo.processInfo.systemUptime - start
        let averageLength = Double(totalMoves) / Double(options.games)
        let gamesPerSecond = elapsed > 0 ? Double(options.games) / elapsed : 0
        let averageLengthText = String(format: "%.1f", averageLength)
        print(
            "selfplay: games=\(options.games) avgLength=\(averageLengthText) "
                + "blackWins=\(blackWins) whiteWins=\(whiteWins) draws=\(draws) "
                + String(format: "elapsed=%.3fs games/sec=%.3f", elapsed, gamesPerSecond)
        )
        if let shardWriter {
            print(
                "selfplay: training samples=\(shardWriter.totalSamples) "
                    + "shards=\(shardWriter.totalShards) -> \(shardWriter.directory.path)"
            )
        }
        return files
    }

    /// Writes a self-describing provenance JSON alongside the training shards (mirrors makedata's
    /// `makedata-config.json`): which net generated the games, and the search/noise/seed settings
    /// that shaped the recorded visit distributions. Named per-seed so parallel/append runs into a
    /// shared corpus directory each keep their own provenance.
    private static func writeTrainingConfig(to directory: URL, options: SelfPlayOptions, modelName: String) throws {
        let dictionary: [String: Any] = [
            "source": "selfplay",
            "model": modelName,
            "model_path": options.model,
            "visits": Int(options.visits),
            "precision": options.precision == .float16 ? "fp16" : "fp32",
            "temperature": Double(options.temperature),
            "temperature_moves": options.temperatureMoves,
            "dirichlet_alpha": Double(options.dirichletAlpha),
            "noise_weight": Double(options.noiseWeight),
            "seed": String(options.seed),
            "komi": Double(options.komi),
            "board_size": options.size,
            "games": options.games,
            "shard_size": options.shardSize,
            "created_at": ISO8601DateFormatter().string(from: Date()),
        ]
        let data = try JSONSerialization.data(withJSONObject: dictionary, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("selfplay-config-\(options.seed).json"), options: .atomic)
    }

    static func parse(_ arguments: [String]) throws -> SelfPlayOptions {
        var options = SelfPlayOptions()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else {
                throw CLIError(description: "Missing value for \(flag)")
            }
            let value = arguments[index + 1]
            switch flag {
            case "-model": options.model = value
            case "-games": options.games = try positiveInt(value, flag: flag)
            case "-out": options.outputDirectory = value
            case "-visits":
                guard let visits = Int64(value), visits > 0 else {
                    throw CLIError(description: "Invalid -visits '\(value)'")
                }
                options.visits = visits
            case "-precision":
                switch value {
                case "fp16": options.precision = .float16
                case "fp32": options.precision = .float32
                default: throw CLIError(description: "Precision must be fp32 or fp16")
                }
            case "-temp": options.temperature = try nonnegativeDouble(value, flag: flag)
            case "-temp-moves": options.temperatureMoves = try nonnegativeInt(value, flag: flag)
            case "-dirichlet": options.dirichletAlpha = try nonnegativeDouble(value, flag: flag)
            case "-noise-weight":
                let weight = try nonnegativeDouble(value, flag: flag)
                guard weight <= 1 else { throw CLIError(description: "Invalid -noise-weight '\(value)'") }
                options.noiseWeight = weight
            case "-seed":
                guard let seed = UInt64(value) else { throw CLIError(description: "Invalid -seed '\(value)'") }
                options.seed = seed
            case "-komi":
                guard let komi = Float(value), Rules.komiIsIntOrHalfInt(komi) else {
                    throw CLIError(description: "Invalid -komi '\(value)'; expected an integer or half-integer")
                }
                options.komi = komi
            case "-size":
                let size = try positiveInt(value, flag: flag)
                guard size <= Board.maxLen else { throw CLIError(description: "Invalid -size '\(value)'") }
                options.size = size
            case "-max-batch-size": options.maxBatchSize = try positiveInt(value, flag: flag)
            case "-write-training": options.writeTrainingDirectory = value
            case "-shard-size": options.shardSize = try positiveInt(value, flag: flag)
            default: throw CLIError(description: "Unknown option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard !options.model.isEmpty, options.games > 0, !options.outputDirectory.isEmpty else {
            throw CLIError(description: usage)
        }
        guard options.dirichletAlpha > 0 || options.noiseWeight == 0 else {
            throw CLIError(description: "-dirichlet must be positive when -noise-weight is nonzero")
        }
        return options
    }

    private static func searchSettings(options: SelfPlayOptions, gameIndex: Int) -> SearchSettings {
        var settings = SearchSettings()
        settings.maxVisits = options.visits
        settings.leafBatchSize = options.maxBatchSize
        settings.numSearchWorkers = min(settings.numSearchWorkers, options.maxBatchSize)
        settings.rootDirichletAlpha = options.dirichletAlpha
        settings.rootNoiseWeight = options.noiseWeight
        settings.selectionTemperature = 0
        settings.randomSeed = options.seed &+ UInt64(gameIndex)
        return settings
    }

    private static func preWarm(
        evaluator: NNEvaluator,
        rules: Rules,
        boardSize: Int,
        maxBatchSize: Int
    ) async throws {
        let board = Board(boardSize, boardSize)
        let history = BoardHistory(board, pla: .black, rules: rules)
        var size = 1
        while size <= maxBatchSize {
            let requests = (0 ..< size).map { _ in
                NNRequest(board: board, history: history, nextPlayer: .black)
            }
            try await evaluator.preWarm(requests)
            size <<= 1
        }
    }

    private static func positiveInt(_ value: String, flag: String) throws -> Int {
        guard let result = Int(value), result > 0 else {
            throw CLIError(description: "Invalid \(flag) '\(value)'")
        }
        return result
    }

    private static func nonnegativeInt(_ value: String, flag: String) throws -> Int {
        guard let result = Int(value), result >= 0 else {
            throw CLIError(description: "Invalid \(flag) '\(value)'")
        }
        return result
    }

    private static func nonnegativeDouble(_ value: String, flag: String) throws -> Double {
        guard let result = Double(value), result >= 0, result.isFinite else {
            throw CLIError(description: "Invalid \(flag) '\(value)'")
        }
        return result
    }
}

/// Memory-bounded RinGoData v2 sink for `selfplay -write-training` (WP-9). Buffers samples across
/// games and flushes a full shard (`selfplay-<seed>-%05d.nngd`) whenever the buffer reaches
/// `shardSize`, then `finish()` flushes any remainder -- so at most one shard's worth of samples is
/// ever resident, mirroring `makedata`'s incremental shard writer. The `<seed>` prefix lets several
/// self-play runs append into one corpus directory without colliding; `train`'s `RinGoDataset.load`
/// globs every `*.nngd` in the directory, so any non-`val.nngd` name is picked up.
private struct SelfPlayShardWriter {
    let directory: URL
    let nnLen: Int
    let shardSize: Int
    let seed: UInt64
    private var buffer = [TrainingSample]()
    private var shardIndex = 0
    private(set) var totalSamples = 0
    private(set) var totalShards = 0

    init(directory: URL, nnLen: Int, shardSize: Int, seed: UInt64) {
        self.directory = directory
        self.nnLen = nnLen
        self.shardSize = shardSize
        self.seed = seed
    }

    mutating func append(_ samples: [TrainingSample]) throws {
        for sample in samples {
            buffer.append(sample)
            totalSamples += 1
            if buffer.count >= shardSize { try flush() }
        }
    }

    mutating func finish() throws {
        try flush()
    }

    private mutating func flush() throws {
        guard !buffer.isEmpty else { return }
        let name = "selfplay-\(seed)-" + String(format: "%05d", shardIndex) + ".nngd"
        try TrainingShardWriter.write(
            samples: buffer, nnLen: nnLen, to: directory.appendingPathComponent(name)
        )
        shardIndex += 1
        totalShards += 1
        buffer.removeAll(keepingCapacity: true)
    }
}
