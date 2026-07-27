import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

final class NNOutputPostprocessorTests: XCTestCase {
    private struct Golden: Decodable {
        let modelVersion: Int
        let nnLen: Int
        let policy: [Float]
        let valueLogits: [Float]
        let scoreValues: [Float]
        let ownership: [Float]
    }

    func testV8FixtureLegalMaskTemperatureAndBlackPerspective() throws {
        let golden = try loadGolden("b6c96")
        let game = try SGFReader.load(coreFixtures.appendingPathComponent("capture-just-made.sgf"))
        let position = try game.position(at: game.moves.count)
        XCTAssertEqual(position.nextPlayer, .black)
        var params = MiscNNInputParams(nnPolicyTemperature: 1.37)
        params.policyOptimism = 0.25
        let post = customPostProcessParams()
        let raw = NNRawResult(
            policyLogits: golden.policy,
            valueLogits: golden.valueLogits,
            scoreValues: golden.scoreValues,
            ownership: golden.ownership
        )
        let actual = try NNOutputPostprocessor.process(
            raw: raw,
            board: position.board,
            history: position.history,
            nextPlayer: position.nextPlayer,
            params: params,
            postProcessParams: post,
            modelVersion: golden.modelVersion,
            nnXLen: golden.nnLen,
            nnYLen: golden.nnLen
        )
        let expected = independent(raw, position, params, post, golden.modelVersion, golden.nnLen)
        assertOutput(actual, expected)
        XCTAssertEqual(actual.policyProbs.count(where: { $0 >= 0 }), expected.legalCount)
        XCTAssertEqual(actual.policyProbs.filter { $0 >= 0 }.reduce(0, +), 1, accuracy: 1e-6)
        XCTAssertEqual(actual.varTimeLeft, -1)
        XCTAssertEqual(actual.shorttermWinlossError, -1)
        XCTAssertEqual(actual.shorttermScoreError, -1)
    }

    func testV17FixtureWhitePerspectiveAndSquaredSoftplusErrors() throws {
        let golden = try loadGolden("b4c256-transformer")
        let game = try SGFReader.load(coreFixtures.appendingPathComponent("simple-ko.sgf"))
        let position = try game.position(at: game.moves.count)
        XCTAssertEqual(position.nextPlayer, .white)
        let params = MiscNNInputParams(nnPolicyTemperature: 0.83)
        let post = customPostProcessParams()
        let raw = NNRawResult(
            policyLogits: golden.policy,
            valueLogits: golden.valueLogits,
            scoreValues: golden.scoreValues,
            ownership: golden.ownership
        )
        let actual = try NNOutputPostprocessor.process(
            raw: raw,
            board: position.board,
            history: position.history,
            nextPlayer: position.nextPlayer,
            params: params,
            postProcessParams: post,
            modelVersion: golden.modelVersion,
            nnXLen: golden.nnLen,
            nnYLen: golden.nnLen
        )
        let expected = independent(raw, position, params, post, golden.modelVersion, golden.nnLen)
        assertOutput(actual, expected)
        XCTAssertGreaterThan(actual.varTimeLeft, 0)
        XCTAssertGreaterThan(actual.shorttermWinlossError, 0)
        XCTAssertGreaterThan(actual.shorttermScoreError, 0)
    }

    func testV9AndV13ErrorTransformBranches() throws {
        let golden = try loadGolden("b4c256-transformer")
        let game = try SGFReader.load(coreFixtures.appendingPathComponent("capture-just-made.sgf"))
        let position = try game.position(at: game.moves.count)
        let params = MiscNNInputParams()
        let post = customPostProcessParams()
        let raw = NNRawResult(
            policyLogits: golden.policy,
            valueLogits: golden.valueLogits,
            scoreValues: golden.scoreValues,
            ownership: golden.ownership
        )
        for version in [9, 13] {
            let actual = try NNOutputPostprocessor.process(
                raw: raw,
                board: position.board,
                history: position.history,
                nextPlayer: position.nextPlayer,
                params: params,
                postProcessParams: post,
                modelVersion: version,
                nnXLen: golden.nnLen,
                nnYLen: golden.nnLen
            )
            let expected = independent(raw, position, params, post, version, golden.nnLen)
            assertOutput(actual, expected)
        }
    }

    private struct Expected {
        let policy: [Double]
        let value: [Double]
        let scoreMean: Double
        let scoreMeanSq: Double
        let lead: Double
        let varTime: Double
        let winlossError: Double
        let scoreError: Double
        let owner: [Double]
        let legalCount: Int
    }

    private func independent(
        _ raw: NNRawResult,
        _ position: (board: Board, history: BoardHistory, nextPlayer: Player),
        _ params: MiscNNInputParams,
        _ post: ModelPostProcessParams,
        _ version: Int,
        _ nnLen: Int
    ) -> Expected {
        let scale = Double(post.outputScaleMultiplier)
        let legal = raw.policyLogits.indices.map { index in
            let loc = NNPos.posToLoc(index, position.board.xSize, position.board.ySize, nnLen, nnLen)
            return loc != Board.nullLoc && position.history.isLegal(position.board, loc, position.nextPlayer)
        }
        let scaled = zip(raw.policyLogits, legal).map {
            $1 ? Double($0) * scale / Double(params.nnPolicyTemperature) : -Double.infinity
        }
        let maximum = zip(scaled, legal).filter(\.1).map(\.0).max()!
        let weights = zip(scaled, legal).map { $1 ? exp($0 - maximum) : 0 }
        let weightSum = weights.reduce(0, +)
        let policy = zip(weights, legal).map { $1 ? $0 / weightSum : -1 }

        var valueLogits = raw.valueLogits.map { Double($0) * scale }
        if position.history.rules.koRule != .simple, position.history.rules.scoringRule != .territory {
            valueLogits[2] = -Double.infinity
        }
        let valueMax = valueLogits.max()!
        let valueWeights = valueLogits.map { exp($0 - valueMax) }
        let valueSum = valueWeights.reduce(0, +)
        var value = valueWeights.map { $0 / valueSum }

        let scores = raw.scoreValues.map { Double($0) * scale }
        var mean = scores[0] * post.scoreMeanMultiplier
        let stdev = softplus(scores[1]) * post.scoreStdevMultiplier
        var meanSq = mean * mean + stdev * stdev
        var lead = scores[2] * post.leadMultiplier
        let resultProbability = 1 - value[2]
        mean *= resultProbability
        meanSq *= resultProbability
        lead *= resultProbability
        let varTime = version >= 9 ? softplus(scores[3]) * post.varianceTimeMultiplier : -1
        let winlossError: Double
        let scoreError: Double
        if version >= 14 {
            winlossError = softplus(scores[4] * 0.5) * sqrt(post.shorttermValueErrorMultiplier)
            scoreError = softplus(scores[5] * 0.5) * sqrt(post.shorttermScoreErrorMultiplier)
        } else if version >= 10 {
            winlossError = sqrt(softplus(scores[4]) * post.shorttermValueErrorMultiplier)
            scoreError = sqrt(softplus(scores[5]) * post.shorttermScoreErrorMultiplier)
        } else if version >= 9 {
            winlossError = softplus(scores[4])
            scoreError = softplus(scores[5]) * 10
        } else {
            winlossError = -1
            scoreError = -1
        }
        let white = position.nextPlayer == .white
        if !white { value.swapAt(0, 1) }
        let owner = raw.ownership!.map { (white ? 1.0 : -1.0) * tanh(Double($0) * scale) }
        return Expected(
            policy: policy,
            value: value,
            scoreMean: white ? mean : -mean,
            scoreMeanSq: meanSq,
            lead: white ? lead : -lead,
            varTime: varTime,
            winlossError: winlossError,
            scoreError: scoreError,
            owner: owner,
            legalCount: legal.count(where: { $0 })
        )
    }

    private func assertOutput(_ actual: NNOutput, _ expected: Expected, line: UInt = #line) {
        XCTAssertLessThan(maxDiff(actual.policyProbs, expected.policy), 1e-6, line: line)
        XCTAssertEqual(actual.whiteWinProb, Float(expected.value[0]), accuracy: 1e-6, line: line)
        XCTAssertEqual(actual.whiteLossProb, Float(expected.value[1]), accuracy: 1e-6, line: line)
        XCTAssertEqual(actual.whiteNoResultProb, Float(expected.value[2]), accuracy: 1e-6, line: line)
        XCTAssertEqual(actual.whiteScoreMean, Float(expected.scoreMean), accuracy: 1e-6, line: line)
        XCTAssertEqual(actual.whiteScoreMeanSq, Float(expected.scoreMeanSq), accuracy: 1e-6, line: line)
        XCTAssertEqual(actual.whiteLead, Float(expected.lead), accuracy: 1e-6, line: line)
        XCTAssertEqual(actual.varTimeLeft, Float(expected.varTime), accuracy: 1e-6, line: line)
        XCTAssertEqual(actual.shorttermWinlossError, Float(expected.winlossError), accuracy: 1e-6, line: line)
        XCTAssertEqual(actual.shorttermScoreError, Float(expected.scoreError), accuracy: 1e-6, line: line)
        XCTAssertLessThan(maxDiff(actual.whiteOwnerMap!, expected.owner), 1e-6, line: line)
    }

    private func customPostProcessParams() -> ModelPostProcessParams {
        var params = ModelPostProcessParams()
        params.scoreMeanMultiplier = 17
        params.scoreStdevMultiplier = 13
        params.leadMultiplier = 19
        params.varianceTimeMultiplier = 31
        params.shorttermValueErrorMultiplier = 0.36
        params.shorttermScoreErrorMultiplier = 25
        params.outputScaleMultiplier = 1.25
        return params
    }

    private func softplus(_ value: Double) -> Double {
        log1p(exp(value))
    }

    private func maxDiff(_ actual: [Float], _ expected: [Double]) -> Double {
        zip(actual, expected).map { abs(Double($0) - $1) }.max() ?? 0
    }

    private func loadGolden(_ name: String) throws -> Golden {
        let url = repoRoot.appendingPathComponent("Tests/RinGoModelTests/Fixtures/\(name).json")
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

    private var coreFixtures: URL {
        repoRoot.appendingPathComponent("Tests/RinGoCoreTests/Fixtures")
    }

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
