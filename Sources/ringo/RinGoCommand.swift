import Foundation
import MLX
import RinGoCore
import RinGoEngine
import RinGoModel

struct CLIError: Error, CustomStringConvertible {
    let description: String
}

private struct RawNNOptions {
    var model = ""
    var sgf = ""
    var moveNumber: Int?
    var rules: String?
    var optimism: Double = 0
    var precision: KataGoNetwork.Precision = .float32
    var nnLength: Int?
}

@main
private enum RinGoCommand {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            switch arguments.first {
            case "-h", "--help", "help":
                // An explicit help request is a success, so it goes to stdout and exits 0;
                // only a missing/unknown subcommand is an error (usage on stderr, exit 1).
                print(usage)
                return
            case "rawnn":
                try runRawNN(parseRawNN(Array(arguments.dropFirst())))
            case "benchmark":
                try await Benchmark.main(Array(arguments.dropFirst()))
            case "gtp":
                try await GTPCommand.main(Array(arguments.dropFirst()))
            case "makedata":
                try await MakeDataCommand.main(Array(arguments.dropFirst()))
            case "analysisexport":
                try AnalysisExportCommand.main(Array(arguments.dropFirst()))
            case "analysisingest":
                try AnalysisIngestCommand.main(Array(arguments.dropFirst()))
            case "selfplay":
                try await SelfPlayCommand.main(Array(arguments.dropFirst()))
            case "train": try TrainCommand.main(Array(arguments.dropFirst()))
            case "evaluate":
                try EvaluateCommand.main(Array(arguments.dropFirst()))
            case "rankgames":
                try RankGamesCommand.main(Array(arguments.dropFirst()))
            case "bookconvert":
                try BookConvertCommand.main(Array(arguments.dropFirst()))
            case "averagemodels":
                try AverageModelsCommand.main(Array(arguments.dropFirst()))
            case "bookcheck":
                try BookCheckCommand.main(Array(arguments.dropFirst()))
            default:
                throw CLIError(description: usage)
            }
        } catch {
            FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
            exit(1)
        }
    }

    private static func parseRawNN(_ arguments: [String]) throws -> RawNNOptions {
        var options = RawNNOptions()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else {
                throw CLIError(description: "Missing value for \(flag)")
            }
            let value = arguments[index + 1]
            switch flag {
            case "-model": options.model = value
            case "-sgf": options.sgf = value
            case "-move-num":
                guard let number = Int(value) else { throw CLIError(description: "Invalid move number") }
                options.moveNumber = number
            case "-rules": options.rules = value
            case "-optimism":
                guard let optimism = Double(value), optimism.isFinite else {
                    throw CLIError(description: "Invalid optimism")
                }
                options.optimism = optimism
            case "-precision":
                switch value {
                case "fp32": options.precision = .float32
                case "fp16": options.precision = .float16
                default: throw CLIError(description: "Precision must be fp32 or fp16")
                }
            case "-nnlen":
                guard let length = Int(value), length > 0, length <= Board.maxLen else {
                    throw CLIError(description: "Invalid neural-network length")
                }
                options.nnLength = length
            default: throw CLIError(description: "Unknown option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard !options.model.isEmpty, !options.sgf.isEmpty, options.moveNumber != nil else {
            throw CLIError(description: usage)
        }
        return options
    }

    private static func runRawNN(_ options: RawNNOptions) throws {
        let game = try SGFReader.load(URL(fileURLWithPath: options.sgf))
        let rulesOverride: Rules? = if let ruleString = options.rules {
            try Rules.parseRules(ruleString)
        } else {
            nil
        }
        let position = try game.position(at: options.moveNumber!, rules: rulesOverride)
        let nnLength = options.nnLength ?? max(game.boardXSize, game.boardYSize)
        guard position.board.xSize <= nnLength, position.board.ySize <= nnLength else {
            throw CLIError(description: "-nnlen is smaller than the board")
        }

        let model = try ModelDesc.loadFromFileMaybeGZipped(options.model)
        var params = MiscNNInputParams(policyOptimism: options.optimism)
        var spatial = [Float]()
        var global = [Float]()
        NNInputs.fillRowV7(
            position.board,
            position.history,
            position.nextPlayer,
            params,
            nnXLen: nnLength,
            nnYLen: nnLength,
            rowBin: &spatial,
            rowGlobal: &global
        )

        let network = try KataGoNetwork(
            desc: model,
            nnXLen: nnLength,
            nnYLen: nnLength,
            precision: options.precision
        )
        let outputs = network.forward(
            spatial: MLXArray(spatial, [1, nnLength, nnLength, model.numInputChannels]),
            global: MLXArray(global, [1, model.numInputGlobalChannels])
        )
        eval([
            outputs.policySpatialLogits,
            outputs.policyPassLogits,
            outputs.valueLogits,
            outputs.scoreValues,
            outputs.ownership,
        ])
        let policyChannels = outputs.policySpatialLogits.dim(-1)
        let spatialPolicy = outputs.policySpatialLogits.asArray(Float.self)
        let passPolicy = outputs.policyPassLogits.asArray(Float.self)
        let optimism = Float(options.optimism)
        let optimisticChannel = min(1, policyChannels - 1)
        var policy = (0 ..< nnLength * nnLength).map { index in
            let offset = index * policyChannels
            let normal = spatialPolicy[offset]
            return normal + (spatialPolicy[offset + optimisticChannel] - normal) * optimism
        }
        policy.append(passPolicy[0] + (passPolicy[optimisticChannel] - passPolicy[0]) * optimism)
        let raw = NNRawResult(
            policyLogits: policy,
            valueLogits: outputs.valueLogits.asArray(Float.self),
            scoreValues: outputs.scoreValues.asArray(Float.self),
            ownership: outputs.ownership.asArray(Float.self)
        )
        params.policyOptimism = options.optimism
        let output = try NNOutputPostprocessor.process(
            raw: raw,
            board: position.board,
            history: position.history,
            nextPlayer: position.nextPlayer,
            params: params,
            postProcessParams: model.postProcessParams,
            modelVersion: model.modelVersion,
            nnXLen: nnLength,
            nnYLen: nnLength
        )
        print(RawNNFormatter.format(output: output, board: position.board, history: position.history), terminator: "")
    }

    private static let usage = """
    Usage: ringo rawnn -model <file> -sgf <file> -move-num <n> \
      [-rules <string>] [-optimism <p>] [-precision fp32|fp16] [-nnlen <n>]
       \(Benchmark.usage)
       \(GTPCommand.usage)
       \(MakeDataCommand.usage)
       \(AnalysisExportCommand.usage)
       \(AnalysisIngestCommand.usage)
       \(SelfPlayCommand.usage)
       \(TrainCommand.usage)
       \(EvaluateCommand.usage)
       \(RankGamesCommand.usage)
       \(BookConvertCommand.usage)
       \(AverageModelsCommand.usage)
       \(BookCheckCommand.usage)
    """
}
