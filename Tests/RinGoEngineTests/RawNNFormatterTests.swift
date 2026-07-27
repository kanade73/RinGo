import RinGoCore
@testable import RinGoEngine
import XCTest

final class RawNNFormatterTests: XCTestCase {
    func testExactReferenceLayout() throws {
        let game = try SGFReader.parse("(;SZ[3:2]KM[7.5]RU[Chinese];B[aa];W[bb])")
        let position = try game.position(at: 2)
        let output = NNOutput(
            nnHash: Hash128(),
            whiteWinProb: 0.625,
            whiteLossProb: 0.25,
            whiteNoResultProb: 0.125,
            whiteScoreMean: 3.25,
            whiteScoreMeanSq: 15.75,
            whiteLead: -1.5,
            varTimeLeft: 20.25,
            shorttermWinlossError: 0.03125,
            shorttermScoreError: 2.5,
            policyProbs: [-1, 0.1, 0, 0.2, -1, 0.45, 0.25],
            policyOptimismUsed: 0.5,
            nnXLen: 3,
            nnYLen: 2,
            whiteOwnerMap: [-1, -0.5, 0, 0.25, 0.5, 1]
        )
        let expected = [
            "Rules: \(position.history.rules)",
            "Encore phase 0",
            "MoveNum: 2 HASH: \(position.board.posHash)",
            "   A B C",
            " 2 X1. .",
            " 1 . O2.",
            "",
            "Win 62.50c",
            "Loss 25.00c",
            "NoResult 12.50c",
            "ScoreMean 3.25",
            "ScoreMeanSq 15.8",
            "Lead -1.50",
            "VarTimeLeft 20.2",
            "STWinlossError 3.12c",
            "STScoreError 2.50",
            "OptimismUsed 0.50",
            "Policy",
            "Pass 250 ",
            "   -  100    0 ",
            " 200    -  450 ",
            "-1000  -500     0 ",
            "  250   500  1000 ",
            "",
            "",
        ].joined(separator: "\n")
        XCTAssertEqual(
            RawNNFormatter.format(output: output, board: position.board, history: position.history),
            expected
        )
    }
}
