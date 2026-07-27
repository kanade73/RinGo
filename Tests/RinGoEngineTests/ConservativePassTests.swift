import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

final class ConservativePassTests: XCTestCase {
    private static let size = 9

    func testDeadOpponentStonesAreCleanedBeforePass() async throws {
        let board = try deadWhiteStonePosition()
        let history = BoardHistory(board, pla: .black, rules: tournamentRules())
        let deadStone = board.getLoc(7, 7)
        XCTAssertEqual(independentArea(board)[deadStone], .black)

        let search = makeSearch(board: board, history: history)
        let first = try await search.runSearch(maxVisits: 1).bestMove
        XCTAssertNotEqual(first, Board.passLoc)
        XCTAssertFalse(board.wouldBeCapture(first, .black), "the first move should approach the two-liberty group")
        play(first, on: board, history: history)
        try await search.makeMove(first)

        play(Board.passLoc, on: board, history: history)
        try await search.makeMove(Board.passLoc)

        let capture = try await search.runSearch(maxVisits: 1).bestMove
        XCTAssertNotEqual(capture, Board.passLoc)
        XCTAssertTrue(board.wouldBeCapture(capture, .black))
        play(capture, on: board, history: history)
        try await search.makeMove(capture)
        XCTAssertEqual(board.colors[deadStone], .empty)

        play(Board.passLoc, on: board, history: history)
        try await search.makeMove(Board.passLoc)
        let cleanResult = try await search.runSearch(maxVisits: 1)
        XCTAssertEqual(cleanResult.bestMove, Board.passLoc)
    }

    func testFinishedCleanPositionAllowsImmediatePass() async throws {
        let board = try cleanFinishedPosition()
        let history = BoardHistory(board, pla: .black, rules: tournamentRules())
        XCTAssertFalse(hasDeadStone(in: board))

        let result = try await makeSearch(board: board, history: history).runSearch(maxVisits: 1)

        XCTAssertEqual(result.bestMove, Board.passLoc)
    }

    func testSekiStonesAreNotTreatedAsRemovable() async throws {
        let board = try sekiPosition()
        let history = BoardHistory(board, pla: .black, rules: tournamentRules())
        let area = independentArea(board)
        XCTAssertEqual(area[board.getLoc(0, 0)], .empty)
        XCTAssertEqual(area[board.getLoc(8, 0)], .empty)
        XCTAssertFalse(hasDeadStone(in: board), "verified independent-life classification must retain seki")

        let result = try await makeSearch(board: board, history: history).runSearch(maxVisits: 1)

        XCTAssertEqual(result.bestMove, Board.passLoc)
    }

    func testPassingHackToggleOffRetainsOriginalPassChoice() async throws {
        let board = try deadWhiteStonePosition()
        let history = BoardHistory(board, pla: .black, rules: tournamentRules())
        var settings = testSettings()
        settings.enablePassingHacks = false

        let result = try await makeSearch(board: board, history: history, settings: settings)
            .runSearch(maxVisits: 1)

        XCTAssertEqual(result.bestMove, Board.passLoc)
    }

    func testGTPFinalScoreMatchesBoardHistoryForCleanPosition() async throws {
        var settings = testSettings()
        settings.maxVisits = 1
        let rules = tournamentRules()
        let evaluator = TestPolicyEvaluator(
            nnXLen: Self.size,
            nnYLen: Self.size,
            preferredLoc: Board.passLoc
        )
        let engine = GTPEngine(
            evaluator: evaluator,
            nnXLen: Self.size,
            nnYLen: Self.size,
            precision: .float32,
            settings: settings,
            rules: rules,
            nnLength: .fixed(Self.size),
            warmupBucketSizes: [1],
            evaluatorFactory: { _, _ in evaluator }
        )
        let referenceBoard = Board(Self.size, Self.size)
        let referenceHistory = BoardHistory(referenceBoard, pla: .black, rules: rules)
        let moves = [
            Move(loc: referenceBoard.getLoc(4, 4), pla: .black),
            Move(loc: Board.passLoc, pla: .white),
            Move(loc: Board.passLoc, pla: .black),
        ]
        for move in moves {
            let vertex = Location.toString(move.loc, Self.size, Self.size)
            let color = move.pla == .black ? "B" : "W"
            let result = await engine.handle("play \(color) \(vertex)")
            XCTAssertEqual(result.response, "= \n\n")
            referenceHistory.makeBoardMoveAssumeLegal(referenceBoard, move.loc, move.pla)
        }
        XCTAssertTrue(referenceHistory.isScored)

        let scoreResult = await engine.handle("final_score")
        let response = try XCTUnwrap(scoreResult.response)

        XCTAssertEqual(responseText(response), formattedScore(referenceHistory.finalWhiteMinusBlackScore))
    }

    private func makeSearch(
        board: Board,
        history: BoardHistory,
        settings: SearchSettings? = nil
    ) -> Search {
        Search(
            evaluator: TestPolicyEvaluator(
                nnXLen: Self.size,
                nnYLen: Self.size,
                preferredLoc: Board.passLoc
            ),
            nnXLen: Self.size,
            nnYLen: Self.size,
            precision: .float32,
            settings: settings ?? testSettings(),
            board: board,
            history: history,
            nextPlayer: history.presumedNextMovePla,
            fp32FallbackFactory: nil
        )
    }

    private func testSettings() -> SearchSettings {
        var settings = SearchSettings()
        settings.leafBatchSize = 1
        settings.numSearchWorkers = 1
        return settings
    }

    private func tournamentRules() -> Rules {
        Rules(
            koRule: .positional,
            scoringRule: .area,
            taxRule: .none,
            multiStoneSuicideLegal: true,
            komi: 7
        )
    }

    private func independentArea(_ board: Board) -> [Color] {
        board.calculateIndependentLifeArea(
            keepTerritories: false,
            keepStones: false,
            isMultiStoneSuicideLegal: true
        ).area
    }

    private func hasDeadStone(in board: Board) -> Bool {
        let area = independentArea(board)
        return board.playableLocations().contains { loc in
            guard let player = board.colors[loc].player else { return false }
            return area[loc] == player.opponent.color
        }
    }

    private func play(_ loc: Loc, on board: Board, history: BoardHistory) {
        let player = history.presumedNextMovePla
        XCTAssertTrue(history.isLegal(board, loc, player))
        history.makeBoardMoveAssumeLegal(board, loc, player)
    }

    private func deadWhiteStonePosition() throws -> Board {
        try Board.parseBoard(Self.size, Self.size, #"""
        xxxxxxxxx
        x.xxxxxxx
        xxxxxxxxx
        xxxxxxxxx
        xxxxxxxxx
        xxxxxxxxx
        xxxxxx..x
        xxxxxx.ox
        xxxxxxxxx
        """#)
    }

    private func cleanFinishedPosition() throws -> Board {
        try Board.parseBoard(Self.size, Self.size, #"""
        xxxxxxxxx
        x.xxxxxxx
        xxxxxxxxx
        xxxxxxxxx
        xxxxxxxxx
        xxxxxxxxx
        xxxxxx..x
        xxxxxx..x
        xxxxxxxxx
        """#)
    }

    private func sekiPosition() throws -> Board {
        try Board.parseBoard(Self.size, Self.size, #"""
        xxxxxoooo
        x.xxxoo.o
        xxxxxoooo
        xxxxxoooo
        xxxx.oooo
        xxxxxoooo
        xxxxxoooo
        xxxxxoooo
        xxxxxoooo
        """#)
    }

    private func responseText(_ response: String) -> String {
        String(response.dropFirst(2)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func formattedScore(_ whiteMinusBlack: Float) -> String {
        if whiteMinusBlack == 0 { return "0" }
        let prefix = whiteMinusBlack > 0 ? "W+" : "B+"
        let magnitude = abs(whiteMinusBlack)
        let number = magnitude == magnitude.rounded(.towardZero)
            ? String(format: "%.0f", magnitude) : String(format: "%.1f", magnitude)
        return prefix + number
    }
}
