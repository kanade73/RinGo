import Foundation
import RinGoCore
import RinGoModel

/// Postprocessed neural-network output. Values and ownership are always from white's perspective.
public struct NNOutput: Sendable {
    public var nnHash: Hash128
    public var whiteWinProb: Float
    public var whiteLossProb: Float
    public var whiteNoResultProb: Float
    public var whiteScoreMean: Float
    public var whiteScoreMeanSq: Float
    public var whiteLead: Float
    public var varTimeLeft: Float
    public var shorttermWinlossError: Float
    public var shorttermScoreError: Float
    /// Indexed by NN position, with pass last. Illegal and padded moves use the C++ sentinel -1.
    public var policyProbs: [Float]
    public var policyOptimismUsed: Float
    public var nnXLen: Int
    public var nnYLen: Int
    public var whiteOwnerMap: [Float]?
}

public extension NNOutput {
    /// Element-wise average of several NN outputs that describe the SAME position under different
    /// input symmetries — each already inverse-permuted back to canonical orientation by
    /// `NNEvaluator` — used for KataGo's `rootNumSymmetriesToSample` root averaging. Ports
    /// `NNOutput::NNOutput(const vector<shared_ptr<NNOutput>>&)`
    /// (`$KG/cpp/neuralnet/nninputs.cpp:324-434`) exactly:
    /// - win/loss/noResult probs, scoreMean, scoreMeanSq, lead, varTimeLeft, the two shortterm
    ///   error fields: simple arithmetic mean.
    /// - ownership: mean over the outputs that HAVE an ownership map (divided by that count); `nil`
    ///   if none do. In this port every output in a set shares the same net/postprocessing, so this
    ///   is either all-present or all-absent.
    /// - policy: per-position mean that PRESERVES the illegal (`< 0`) sentinel (all-`-1` averages to
    ///   `-1`), EXCEPT it FALLS BACK to the first output's policy verbatim on ANY legality-sign
    ///   mismatch between outputs (the impossible-in-practice hash-collision guard; legality here is
    ///   board-derived and identical across symmetries).
    /// - policyOptimismUsed: the first output's value when all agree, else their mean.
    ///
    /// `nnHash`/`nnXLen`/`nnYLen` are taken from the first output; they are identical across the set
    /// because the input symmetry is not part of the situational hash. Requires a non-empty input; a
    /// single-element input returns that output unchanged (matching a `k == 1` non-averaged eval).
    static func averaged(_ outputs: [NNOutput]) -> NNOutput {
        precondition(!outputs.isEmpty, "NNOutput.averaged requires at least one output")
        let first = outputs[0]
        guard outputs.count > 1 else { return first }
        let count = Float(outputs.count)

        var result = first
        result.whiteWinProb = outputs.reduce(0) { $0 + $1.whiteWinProb } / count
        result.whiteLossProb = outputs.reduce(0) { $0 + $1.whiteLossProb } / count
        result.whiteNoResultProb = outputs.reduce(0) { $0 + $1.whiteNoResultProb } / count
        result.whiteScoreMean = outputs.reduce(0) { $0 + $1.whiteScoreMean } / count
        result.whiteScoreMeanSq = outputs.reduce(0) { $0 + $1.whiteScoreMeanSq } / count
        result.whiteLead = outputs.reduce(0) { $0 + $1.whiteLead } / count
        result.varTimeLeft = outputs.reduce(0) { $0 + $1.varTimeLeft } / count
        result.shorttermWinlossError = outputs.reduce(0) { $0 + $1.shorttermWinlossError } / count
        result.shorttermScoreError = outputs.reduce(0) { $0 + $1.shorttermScoreError } / count

        // Ownership: average only the maps that are present, divide by that count (mirrors the C++
        // whiteOwnerMapCount accumulation), or leave it nil when none carry one.
        if let ownerLen = first.whiteOwnerMap?.count {
            var summed = [Float](repeating: 0, count: ownerLen)
            var mapCount: Float = 0
            for output in outputs {
                guard let map = output.whiteOwnerMap, map.count == ownerLen else { continue }
                mapCount += 1
                for position in 0 ..< ownerLen {
                    summed[position] += map[position]
                }
            }
            result.whiteOwnerMap = mapCount > 0 ? summed.map { $0 / mapCount } : nil
        }

        // Policy: mean per position, preserving the `< 0` illegal sentinel; but if any output
        // disagrees with the first on which positions are legal, fall back to the first policy
        // verbatim (C++'s hash-collision safety valve).
        let policySize = first.policyProbs.count
        let firstIsIllegal = first.policyProbs.map { $0 < 0 }
        let mismatch = outputs.dropFirst().contains { output in
            zip(firstIsIllegal, output.policyProbs).contains { wasIllegal, prob in wasIllegal != (prob < 0) }
        }
        if mismatch {
            result.policyProbs = first.policyProbs
        } else {
            var summed = [Float](repeating: 0, count: policySize)
            for output in outputs {
                for position in 0 ..< policySize {
                    summed[position] += output.policyProbs[position]
                }
            }
            result.policyProbs = summed.map { $0 / count }
        }

        // Policy optimism: the common value when all agree, else the mean.
        if outputs.dropFirst().allSatisfy({ $0.policyOptimismUsed == first.policyOptimismUsed }) {
            result.policyOptimismUsed = first.policyOptimismUsed
        } else {
            result.policyOptimismUsed = outputs.reduce(0) { $0 + $1.policyOptimismUsed } / count
        }

        return result
    }
}

public struct NNRawResult: Sendable {
    public var policyLogits: [Float]
    public var valueLogits: [Float]
    public var scoreValues: [Float]
    public var ownership: [Float]?

    public init(
        policyLogits: [Float],
        valueLogits: [Float],
        scoreValues: [Float],
        ownership: [Float]?
    ) {
        self.policyLogits = policyLogits
        self.valueLogits = valueLogits
        self.scoreValues = scoreValues
        self.ownership = ownership
    }
}

public enum NNOutputPostprocessorError: Error, CustomStringConvertible, Equatable {
    case invalidInput(String)
    case unsupported(String)

    public var description: String {
        switch self {
        case let .invalidInput(message), let .unsupported(message): message
        }
    }
}

public enum NNOutputPostprocessor {
    public static func process(
        raw: NNRawResult,
        board: Board,
        history: BoardHistory,
        nextPlayer: Player,
        params: MiscNNInputParams = MiscNNInputParams(),
        postProcessParams: ModelPostProcessParams,
        modelVersion: Int,
        nnXLen: Int,
        nnYLen: Int
    ) throws -> NNOutput {
        let policySize = nnXLen * nnYLen + 1
        guard board.xSize <= nnXLen, board.ySize <= nnYLen else {
            throw NNOutputPostprocessorError.invalidInput("Board exceeds neural-network dimensions")
        }
        guard history.presumedNextMovePla == nextPlayer else {
            throw NNOutputPostprocessorError.invalidInput("nextPlayer does not match BoardHistory")
        }
        guard raw.policyLogits.count == policySize else {
            throw NNOutputPostprocessorError.invalidInput(
                "Expected \(policySize) policy logits, got \(raw.policyLogits.count)"
            )
        }
        guard raw.valueLogits.count == 3 else {
            throw NNOutputPostprocessorError.invalidInput("Expected 3 value logits")
        }
        let expectedScores = modelVersion >= 9 ? 6 : (modelVersion >= 8 ? 4 : 2)
        guard modelVersion >= 4, raw.scoreValues.count == expectedScores else {
            throw NNOutputPostprocessorError.invalidInput(
                "Model v\(modelVersion) requires \(expectedScores) score values"
            )
        }
        guard params.nnPolicyTemperature > 0, params.nnPolicyTemperature.isFinite else {
            throw NNOutputPostprocessorError.invalidInput("nnPolicyTemperature must be finite and positive")
        }
        if params.avoidMYTDaggerHack {
            throw NNOutputPostprocessorError.unsupported("avoidMYTDaggerHack postprocessing is not supported")
        }

        let scale = Double(postProcessParams.outputScaleMultiplier)
        var legal = [Bool](repeating: false, count: policySize)
        for position in 0 ..< policySize {
            let loc = NNPos.posToLoc(position, board.xSize, board.ySize, nnXLen, nnYLen)
            legal[position] = loc != Board.nullLoc && history.isLegal(board, loc, nextPlayer)
        }
        let legalCount = legal.count(where: { $0 })
        guard legalCount > 0 else {
            throw NNOutputPostprocessorError.invalidInput("Position has no legal moves")
        }

        let policyScaling = scale / Double(params.nnPolicyTemperature)
        let scaled = zip(raw.policyLogits, legal).map { logit, isLegal in
            isLegal ? Double(logit) * policyScaling : -1e30
        }
        let maxPolicy = scaled.max() ?? -1e30
        var exponentials = zip(scaled, legal).map { value, isLegal in
            isLegal ? Foundation.exp(value - maxPolicy) : 0
        }
        if params.enablePassingHacks {
            let pass = policySize - 1
            let nonPassSum = exponentials.dropLast().reduce(0, +)
            exponentials[pass] = max(1e-20, min(exponentials[pass], nonPassSum * 19))
        }
        let sum = exponentials.reduce(0, +)
        guard sum.isFinite else {
            throw NNOutputPostprocessorError.invalidInput("Non-finite policy sum")
        }
        let policy: [Float] = if sum <= 0 {
            legal.map { $0 ? Float(1 / Double(legalCount)) : -1 }
        } else {
            zip(exponentials, legal).map { value, isLegal in isLegal ? Float(value / sum) : -1 }
        }

        var logits = raw.valueLogits.map { Double($0) * scale }
        if history.rules.koRule != .simple, history.rules.scoringRule != .territory {
            logits[2] -= 100_000
        }
        let maxValue = logits.max()!
        var probabilities = logits.map { Foundation.exp($0 - maxValue) }
        if history.rules.koRule != .simple, history.rules.scoringRule != .territory {
            probabilities[2] = 0
        }
        let probabilitySum = probabilities.reduce(0, +)
        guard probabilitySum.isFinite, probabilitySum > 0 else {
            throw NNOutputPostprocessorError.invalidInput("Non-finite value probability sum")
        }
        probabilities = probabilities.map { $0 / probabilitySum }

        let scores = raw.scoreValues.map { Double($0) * scale }
        var scoreMean = scores[0] * postProcessParams.scoreMeanMultiplier
        let scoreStdev = softPlus(scores[1]) * postProcessParams.scoreStdevMultiplier
        var scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev
        var lead = (modelVersion >= 8 ? scores[2] : scores[0]) * postProcessParams.leadMultiplier
        let varTime = softPlus(modelVersion >= 8 ? scores[3] : 0) * postProcessParams.varianceTimeMultiplier
        let resultProbability = 1 - probabilities[2]
        scoreMean *= resultProbability
        scoreMeanSq *= resultProbability
        lead *= resultProbability

        let shorttermWinlossError: Double
        let shorttermScoreError: Double
        if modelVersion >= 14 {
            let winlossSoftPlus = softPlus(scores[4] * 0.5)
            shorttermWinlossError = Foundation.sqrt(
                winlossSoftPlus * winlossSoftPlus * postProcessParams.shorttermValueErrorMultiplier
            )
            let scoreSoftPlus = softPlus(scores[5] * 0.5)
            shorttermScoreError = Foundation.sqrt(
                scoreSoftPlus * scoreSoftPlus * postProcessParams.shorttermScoreErrorMultiplier
            )
        } else if modelVersion >= 10 {
            shorttermWinlossError = Foundation.sqrt(
                softPlus(scores[4]) * postProcessParams.shorttermValueErrorMultiplier
            )
            shorttermScoreError = Foundation.sqrt(
                softPlus(scores[5]) * postProcessParams.shorttermScoreErrorMultiplier
            )
        } else {
            let rawWinlossError = modelVersion >= 9 ? scores[4] : 0
            let rawScoreError = modelVersion >= 9 ? scores[5] : 0
            shorttermWinlossError = softPlus(rawWinlossError)
            shorttermScoreError = softPlus(rawScoreError) * 10
        }
        guard [scoreMean, scoreMeanSq, lead, varTime, shorttermWinlossError, shorttermScoreError]
            .allSatisfy(\.isFinite)
        else {
            throw NNOutputPostprocessorError.invalidInput("Non-finite score postprocessing result")
        }

        let whiteToMove = nextPlayer == .white
        let ownerMap = try raw.ownership.map { ownership throws -> [Float] in
            guard ownership.count == nnXLen * nnYLen else {
                throw NNOutputPostprocessorError.invalidInput("Ownership map has the wrong size")
            }
            return ownership.enumerated().map { position, value in
                let x = position % nnXLen
                let y = position / nnXLen
                guard x < board.xSize, y < board.ySize else { return 0 }
                let owner = Foundation.tanh(Double(value) * scale)
                return Float(whiteToMove ? owner : -owner)
            }
        }

        return NNOutput(
            nnHash: NNInputs.getHash(board, history, nextPlayer, params),
            whiteWinProb: Float(whiteToMove ? probabilities[0] : probabilities[1]),
            whiteLossProb: Float(whiteToMove ? probabilities[1] : probabilities[0]),
            whiteNoResultProb: Float(probabilities[2]),
            whiteScoreMean: Float(whiteToMove ? scoreMean : -scoreMean),
            whiteScoreMeanSq: Float(scoreMeanSq),
            whiteLead: Float(whiteToMove ? lead : -lead),
            varTimeLeft: modelVersion >= 9 ? Float(varTime) : -1,
            shorttermWinlossError: modelVersion >= 9 ? Float(shorttermWinlossError) : -1,
            shorttermScoreError: modelVersion >= 9 ? Float(shorttermScoreError) : -1,
            policyProbs: policy,
            policyOptimismUsed: Float(params.policyOptimism),
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            whiteOwnerMap: ownerMap
        )
    }

    private static func softPlus(_ value: Double) -> Double {
        if value > 40 { return value }
        return Foundation.log(1 + Foundation.exp(value))
    }
}
