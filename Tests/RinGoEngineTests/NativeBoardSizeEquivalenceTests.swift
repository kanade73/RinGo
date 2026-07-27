import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

final class NativeBoardSizeEquivalenceTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"

    /// HOST-ONLY / MLX-dependent: constructs two fp32 GPU evaluators and executes both graphs.
    func testHostOnlyNative9MatchesMasked19() async throws {
        let game = try SGFReader.load(
            repoRoot.appendingPathComponent("Tests/RinGoCoreTests/Fixtures/chinese-9-nn19.sgf")
        )
        let position = try game.position(at: game.moves.count)
        let desc = try ModelDesc.loadFromFileMaybeGZipped(modelPath())
        let native = try NNEvaluator(
            desc: desc,
            nnXLen: 9,
            nnYLen: 9,
            precision: .float32,
            maxBatchSize: 1
        )
        let masked = try NNEvaluator(
            desc: desc,
            nnXLen: 19,
            nnYLen: 19,
            precision: .float32,
            maxBatchSize: 1
        )
        let request = NNRequest(
            board: position.board,
            history: position.history,
            nextPlayer: position.nextPlayer
        )

        let nativeOutput = try await native.evaluate(request)
        let maskedOutput = try await masked.evaluate(request)
        let tolerance: Float = 1e-4

        for y in 0 ..< 9 {
            for x in 0 ..< 9 {
                XCTAssertEqual(
                    nativeOutput.policyProbs[y * 9 + x],
                    maskedOutput.policyProbs[y * 19 + x],
                    accuracy: tolerance,
                    "policy at (\(x), \(y))"
                )
            }
        }
        XCTAssertEqual(nativeOutput.policyProbs[81], maskedOutput.policyProbs[361], accuracy: tolerance, "pass")

        XCTAssertEqual(nativeOutput.whiteWinProb, maskedOutput.whiteWinProb, accuracy: tolerance)
        XCTAssertEqual(nativeOutput.whiteLossProb, maskedOutput.whiteLossProb, accuracy: tolerance)
        XCTAssertEqual(nativeOutput.whiteNoResultProb, maskedOutput.whiteNoResultProb, accuracy: tolerance)

        let nativeScores = scoreValues(nativeOutput)
        let maskedScores = scoreValues(maskedOutput)
        for (index, pair) in zip(nativeScores, maskedScores).enumerated() {
            XCTAssertEqual(pair.0, pair.1, accuracy: tolerance, "score value \(index)")
        }

        let nativeOwnership = try XCTUnwrap(nativeOutput.whiteOwnerMap)
        let maskedOwnership = try XCTUnwrap(maskedOutput.whiteOwnerMap)
        for y in 0 ..< 9 {
            for x in 0 ..< 9 {
                XCTAssertEqual(
                    nativeOwnership[y * 9 + x],
                    maskedOwnership[y * 19 + x],
                    accuracy: tolerance,
                    "ownership at (\(x), \(y))"
                )
            }
        }
    }

    private func scoreValues(_ output: NNOutput) -> [Float] {
        [
            output.whiteScoreMean,
            output.whiteScoreMeanSq,
            output.whiteLead,
            output.varTimeLeft,
            output.shorttermWinlossError,
            output.shorttermScoreError,
        ]
    }

    private func modelPath() throws -> String {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"] ?? Self.defaultModelsDirectory
        let path = URL(fileURLWithPath: directory).appendingPathComponent(Self.modelFile).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("KataGo model fixture is missing; set KATAGO_MODELS_DIR (checked \(path))")
        }
        return path
    }

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
