import Foundation
import RinGoCore
@testable import RinGoEngine
import XCTest

final class SGFWriterTests: XCTestCase {
    func testRealisticGameRoundTripsMovesMetadataAndResult() throws {
        let rules = try chineseRules(komi: 7)
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: rules)
        let played: [(Int, Int, Player)] = [
            (2, 2, .black), (6, 6, .white), (6, 2, .black), (2, 6, .white),
            (4, 4, .black), (3, 4, .white), (5, 5, .black), (4, 5, .white),
        ]
        for (x, y, player) in played {
            XCTAssertTrue(history.makeBoardMoveTolerant(board, Location.getLoc(x, y, 9), player))
        }
        finishByPasses(board: board, history: history)

        let source = try SGFWriter.write(
            history: history,
            blackName: "Black \\ player",
            whiteName: "White ] player",
            date: "2026-07-06"
        )
        let parsed = try SGFReader.parse(source)

        assertRoundTrip(parsed, history: history)
        XCTAssertTrue(source.contains("PB[Black \\\\ player]"))
        XCTAssertTrue(source.contains("PW[White \\] player]"))
        XCTAssertTrue(source.contains("DT[2026-07-06]"))
        XCTAssertTrue(source.contains(";B[];W[]"), "passes use empty SGF coordinates")
        XCTAssertEqual(parsed.result, expectedResult(history.finalWhiteMinusBlackScore))
    }

    func testAllPassGameRoundTrips() throws {
        let rules = try chineseRules(komi: 0.5)
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: rules)
        finishByPasses(board: board, history: history)

        let source = try SGFWriter.write(
            history: history,
            blackName: "black",
            whiteName: "white",
            date: "2000-01-01"
        )
        let parsed = try SGFReader.parse(source)

        assertRoundTrip(parsed, history: history)
        XCTAssertEqual(parsed.moves.map(\.loc), [Board.passLoc, Board.passLoc])
        XCTAssertEqual(parsed.result, "W+0.5")
    }

    func testRoundTripsDifferentSizesAndKomi() throws {
        for (xSize, ySize, komi) in [(5, 5, Float(0)), (13, 9, Float(5.5)), (19, 19, Float(7.5))] {
            let rules = try chineseRules(komi: komi)
            let board = Board(xSize, ySize)
            let history = BoardHistory(board, pla: .black, rules: rules)
            let first = Location.getLoc(xSize - 1, ySize - 1, xSize)
            XCTAssertTrue(history.makeBoardMoveTolerant(board, first, .black))
            XCTAssertTrue(history.makeBoardMoveTolerant(board, Board.passLoc, .white))
            XCTAssertTrue(history.makeBoardMoveTolerant(board, Board.passLoc, .black))

            let source = try SGFWriter.write(
                history: history,
                blackName: "B",
                whiteName: "W",
                date: "1999-12-31"
            )
            try assertRoundTrip(SGFReader.parse(source), history: history)
        }
    }

    func testRejectsUnfinishedAndUnscoredGames() throws {
        let board = Board(9, 9)
        let history = try BoardHistory(board, pla: .black, rules: chineseRules(komi: 7))
        XCTAssertThrowsError(try SGFWriter.write(
            history: history,
            blackName: "B",
            whiteName: "W",
            date: "2026-07-06"
        )) { XCTAssertEqual($0 as? SGFWriterError, .unfinishedGame) }

        history.setWinnerByResignation(.black)
        XCTAssertThrowsError(try SGFWriter.write(
            history: history,
            blackName: "B",
            whiteName: "W",
            date: "2026-07-06"
        )) { XCTAssertEqual($0 as? SGFWriterError, .unscoredGame) }
    }

    private func chineseRules(komi: Float) throws -> Rules {
        try Rules.parseRulesWithoutKomi("Chinese", komi: komi)
    }

    private func finishByPasses(board: Board, history: BoardHistory) {
        let first = history.presumedNextMovePla
        XCTAssertTrue(history.makeBoardMoveTolerant(board, Board.passLoc, first))
        XCTAssertTrue(history.makeBoardMoveTolerant(board, Board.passLoc, first.opponent))
        XCTAssertTrue(history.isGameFinished)
        XCTAssertTrue(history.isScored)
    }

    private func assertRoundTrip(_ parsed: SGFGame, history: BoardHistory) {
        XCTAssertEqual(parsed.boardXSize, history.initialBoard.xSize)
        XCTAssertEqual(parsed.boardYSize, history.initialBoard.ySize)
        XCTAssertEqual(parsed.komi, history.rules.komi)
        XCTAssertTrue(parsed.rules.equalsIgnoringKomi(history.rules))
        XCTAssertEqual(parsed.moves, history.moveHistory)
    }

    private func expectedResult(_ score: Float) -> String {
        if score == 0 { return "0" }
        let prefix = score > 0 ? "W+" : "B+"
        let magnitude = abs(score)
        let number = magnitude == magnitude.rounded(.towardZero)
            ? String(format: "%.0f", magnitude) : String(format: "%.1f", magnitude)
        return prefix + number
    }
}
