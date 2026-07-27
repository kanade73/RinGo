import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

final class RepeatPositionDefenseTests: XCTestCase {
    func testGTPNineByNineUsesChineseAreaRulesWithPositionalSuperko() async throws {
        let chinese = try Rules.parseRulesWithoutKomi("chinese", komi: 7.0)
        XCTAssertEqual(chinese.koRule, .simple, "fixture proves the GTP tournament override is necessary")
        let engine = makeEngine(rules: chinese)

        let response = await engine.handle("boardsize 9")
        XCTAssertEqual(response.response, "= \n\n")
        let active = await engine.currentRules()
        XCTAssertEqual(active.koRule, .positional)
        XCTAssertEqual(active.scoringRule, .area)
        XCTAssertEqual(active.komi, 7.0)
    }

    func testSearchNeverSelectsImmediateKoRetake() async throws {
        let board = try Board.parseBoard(6, 5, ".o.xxo\noxxxo.\no.x.oo\nxx.oo.\noooo.o")
        let rules = Rules(
            koRule: .positional,
            scoringRule: .territory,
            taxRule: .seki,
            multiStoneSuicideLegal: false,
            komi: 0.5
        )
        let history = BoardHistory(board, pla: .black, rules: rules)
        play(history, board, x: 5, y: 1, player: .black)
        let retake = board.getLoc(5, 0)
        XCTAssertFalse(history.isLegal(board, retake, .white))

        let result = try await search(board: board, history: history, preferredLoc: retake)

        XCTAssertNotEqual(result.bestMove, retake)
        XCTAssertTrue(history.isLegal(board, result.bestMove, .white))
    }

    /// A small artificial two-ko cycle exercises the same whole-board-history rejection needed
    /// for triple ko, without depending on the simple-ko marker alone.
    func testSearchNeverSelectsTripleKoLikeLongCycleRepetition() async throws {
        let board = try Board.parseBoard(6, 5, ".o.xxo\noxxxo.\no.x.oo\nxx.oo.\noooo.o")
        let rules = Rules(
            koRule: .positional,
            scoringRule: .territory,
            taxRule: .seki,
            multiStoneSuicideLegal: false,
            komi: 0.5
        )
        let history = BoardHistory(board, pla: .black, rules: rules)
        let sequence: [(x: Int?, y: Int?, player: Player)] = [
            (5, 1, .black), (nil, nil, .white), (3, 2, .black), (2, 0, .white),
            (0, 0, .black), (5, 0, .white), (nil, nil, .black), (1, 0, .white),
            (nil, nil, .black), (2, 0, .white),
        ]
        for move in sequence {
            if let x = move.x, let y = move.y {
                play(history, board, x: x, y: y, player: move.player)
            } else {
                play(history, board, loc: Board.passLoc, player: move.player)
            }
        }
        let repeatingMove = board.getLoc(0, 0)
        XCTAssertEqual(history.presumedNextMovePla, .black)
        XCTAssertFalse(history.isLegal(board, repeatingMove, .black))
        XCTAssertTrue(history.koHashHistory.contains(board.getPosHashAfterMove(repeatingMove, .black)))

        let result = try await search(board: board, history: history, preferredLoc: repeatingMove)

        XCTAssertNotEqual(result.bestMove, repeatingMove)
        XCTAssertTrue(history.isLegal(board, result.bestMove, .black))
    }

    private func search(
        board: Board,
        history: BoardHistory,
        preferredLoc: Loc
    ) async throws -> SearchResult {
        var settings = SearchSettings()
        settings.leafBatchSize = 1
        settings.numSearchWorkers = 1
        let evaluator = TestPolicyEvaluator(
            nnXLen: board.xSize,
            nnYLen: board.ySize,
            preferredLoc: preferredLoc
        )
        let search = Search(
            evaluator: evaluator,
            nnXLen: board.xSize,
            nnYLen: board.ySize,
            precision: .float32,
            settings: settings,
            board: board,
            history: history,
            nextPlayer: history.presumedNextMovePla,
            fp32FallbackFactory: nil
        )
        return try await search.runSearch(maxVisits: 1)
    }

    private func makeEngine(rules: Rules) -> GTPEngine {
        GTPEngine(
            evaluator: TestPolicyEvaluator(nnXLen: 9, nnYLen: 9),
            nnXLen: 9,
            nnYLen: 9,
            precision: .float32,
            settings: SearchSettings(),
            rules: rules,
            nnLength: .fixed(9),
            warmupBucketSizes: [1],
            evaluatorFactory: { _, _ in TestPolicyEvaluator(nnXLen: 9, nnYLen: 9) }
        )
    }

    private func play(
        _ history: BoardHistory,
        _ board: Board,
        x: Int,
        y: Int,
        player: Player
    ) {
        play(history, board, loc: board.getLoc(x, y), player: player)
    }

    private func play(
        _ history: BoardHistory,
        _ board: Board,
        loc: Loc,
        player: Player
    ) {
        XCTAssertTrue(history.isLegal(board, loc, player))
        history.makeBoardMoveAssumeLegal(board, loc, player)
    }
}
