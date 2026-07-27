import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// Tests for WP-6a search-tree reuse across `Search.makeMove` (re-rooting onto the played child's
/// retained subtree). Uses the deterministic uniform-policy stub evaluator so every case is
/// host-only and needs no MLX / model fixture. Covers the four correctness invariants:
///   (a) re-rooting preserves the played child's stats,
///   (b) the reuse-off flag / `setPosition` force a fresh tree,
///   (c) a superko-divergent history (same board, different `koHashHistory`) is rejected,
///   (d) root Dirichlet noise interacts correctly with reuse (retains subtree, no duplicate child),
///   (e) the visit budget counts NEW visits, layering them on top of reused ones (invariant #3).
final class TreeReuseTests: XCTestCase {
    private static let size = 9

    private func makeSearch(_ settings: SearchSettings) -> Search {
        let board = Board(Self.size, Self.size)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        return Search(
            evaluator: TreeReuseStubEvaluator(nnLen: Self.size),
            nnXLen: Self.size,
            nnYLen: Self.size,
            precision: .float32,
            settings: settings,
            board: board,
            history: history,
            nextPlayer: .black,
            fp32FallbackFactory: nil
        )
    }

    // MARK: - (a) re-rooting preserves the played child's stats

    func testReRootingPreservesPlayedChildVisitCount() async throws {
        let search = makeSearch(SearchSettings())
        let result = try await search.runSearch(maxVisits: 200)
        let best = result.bestMove
        let childVisits = try XCTUnwrap(result.visitsByMove.first { $0.loc == best }?.visits)
        XCTAssertGreaterThan(childVisits, 0)

        try await search.makeMove(best)
        let rootVisits = await search.currentRootVisits()
        XCTAssertEqual(
            rootVisits, childVisits,
            "re-rooting must retain the played child's subtree, so the new root's visits equal the child's"
        )
    }

    // MARK: - (b) reuse-off / setPosition force a fresh tree

    func testReuseDisabledInstallsFreshRoot() async throws {
        var settings = SearchSettings()
        settings.treeReuseEnabled = false
        let search = makeSearch(settings)

        let result = try await search.runSearch(maxVisits: 120)
        try await search.makeMove(result.bestMove)
        let rootVisits = await search.currentRootVisits()
        XCTAssertEqual(rootVisits, 0, "with reuse disabled, makeMove must install a fresh 0-visit root")
    }

    func testReuseEnabledRetainsButSetPositionDiscards() async throws {
        let search = makeSearch(SearchSettings())
        let result = try await search.runSearch(maxVisits: 120)
        try await search.makeMove(result.bestMove)
        let retainedVisits = await search.currentRootVisits()
        XCTAssertGreaterThan(
            retainedVisits, 0,
            "with reuse enabled (default), makeMove must retain the played child's subtree"
        )

        // The GTP undo / clear_board / komi paths all route through setPosition, which must always
        // discard the tree regardless of the reuse setting.
        let board = Board(Self.size, Self.size)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        await search.setPosition(board: board, history: history, nextPlayer: .black)
        let afterSetPosition = await search.currentRootVisits()
        XCTAssertEqual(afterSetPosition, 0, "setPosition must discard the tree")
    }

    // MARK: - (c) superko-sensitive: same board via different histories must NOT reuse

    func testCanReuseRejectsDivergentSuperkoHistoryButAcceptsIdentical() {
        // Two transposed move orders reach the SAME 4-stone board with the SAME side to move, so
        // the simple-ko position hash matches — but the intermediate positions (and therefore the
        // positional-superko history `koHashHistory`) differ, which changes which future repeat
        // positions are superko-banned. Reuse must be rejected on the position hash alone matching.
        let bp1 = (2, 2, Player.black), wq1 = (6, 6, Player.white)
        let bp2 = (6, 2, Player.black), wq2 = (2, 6, Player.white)
        let (boardA, histA) = buildPosition([bp1, wq1, bp2, wq2]) // subtree assumed this history
        let (boardB, histB) = buildPosition([bp2, wq2, bp1, wq1]) // real game reached it differently

        // Precondition: identical final position + side to move => identical position hash.
        XCTAssertEqual(
            boardA.getSitHashWithSimpleKo(.black),
            boardB.getSitHashWithSimpleKo(.black),
            "the transposition must produce an identical board position + side to move"
        )
        // ...but the positional-superko histories differ (this is what the guard must catch).
        XCTAssertNotEqual(
            histA.koHashHistory, histB.koHashHistory,
            "the transposition must produce different positional-superko histories"
        )

        XCTAssertFalse(
            Search.canReuseSubtree(
                childBoard: boardA, childHistory: histA, childNextPla: .black,
                board: boardB, history: histB, nextPla: .black
            ),
            "reuse must be rejected when koHashHistory differs even though the position hash matches"
        )

        // The genuine continuation (byte-identical history) must still be accepted, so the guard
        // never spuriously discards a reusable subtree in the normal move flow.
        let (boardA2, histA2) = buildPosition([bp1, wq1, bp2, wq2])
        XCTAssertTrue(
            Search.canReuseSubtree(
                childBoard: boardA, childHistory: histA, childNextPla: .black,
                board: boardA2, history: histA2, nextPla: .black
            ),
            "reuse must be accepted for an identical continuation history"
        )
    }

    func testCanReuseRejectsWrongSideToMove() {
        let (board, history) = buildPosition([(2, 2, .black)])
        XCTAssertFalse(
            Search.canReuseSubtree(
                childBoard: board, childHistory: history, childNextPla: .white,
                board: board, history: history, nextPla: .black
            ),
            "reuse must be rejected when the side to move differs"
        )
    }

    // MARK: - (d) root Dirichlet noise interacts correctly with reuse

    func testNoiseWithReuseRetainsSubtreeWithoutDuplicatingChildren() async throws {
        var settings = SearchSettings()
        settings.rootDirichletAlpha = 0.15
        settings.rootNoiseWeight = 0.25
        settings.randomSeed = 11
        let search = makeSearch(settings)

        let first = try await search.runSearch(maxVisits: 200)
        let childVisits = try XCTUnwrap(first.visitsByMove.first { $0.loc == first.bestMove }?.visits)
        try await search.makeMove(first.bestMove)
        let retainedVisits = await search.currentRootVisits()
        XCTAssertEqual(
            retainedVisits, childVisits,
            "reuse must retain the subtree even with root noise on (noise only re-derives root priors)"
        )

        // Re-entering prepareRootNoiseIfNeeded on a reused, already-expanded root must not re-add
        // already-expanded locations as untried moves (regression guard: duplicate root child).
        let second = try await search.runSearch(maxVisits: 100)
        let locs = second.visitsByMove.map(\.loc)
        XCTAssertEqual(locs.count, Set(locs).count, "root.children must never contain a duplicate loc")
    }

    func testNoiseWithReuseOffStaysFreshAndLegal() async throws {
        var settings = SearchSettings()
        settings.rootDirichletAlpha = 0.15
        settings.rootNoiseWeight = 0.25
        settings.randomSeed = 11
        settings.treeReuseEnabled = false
        let search = makeSearch(settings)

        let first = try await search.runSearch(maxVisits: 120)
        try await search.makeMove(first.bestMove)
        let afterMove = await search.currentRootVisits()
        XCTAssertEqual(afterMove, 0, "reuse-off must start each move from a fresh root")
        let second = try await search.runSearch(maxVisits: 120) // must not crash on a fresh noisy root
        let locs = second.visitsByMove.map(\.loc)
        XCTAssertEqual(locs.count, Set(locs).count)
    }

    // MARK: - (e) visit budget counts NEW visits (invariant #3)

    func testVisitBudgetCountsNewVisitsOnTopOfReusedOnes() async throws {
        let search = makeSearch(SearchSettings())
        let first = try await search.runSearch(maxVisits: 200)
        try await search.makeMove(first.bestMove)
        let reused = await search.currentRootVisits()
        XCTAssertGreaterThan(reused, 0)

        let second = try await search.runSearch(maxVisits: 150)
        XCTAssertEqual(
            second.visitsAchieved, 150,
            "the per-turn budget must count NEW visits, unaffected by however many were reused"
        )
        XCTAssertEqual(
            second.rootVisits, reused + 150,
            "reused visits are retained; the search layers the new budget on top of them"
        )
    }

    // MARK: - Helpers

    private func buildPosition(_ moves: [(Int, Int, Player)]) -> (Board, BoardHistory) {
        let board = Board(Self.size, Self.size)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        for (x, y, pla) in moves {
            let loc = Location.getLoc(x, y, Self.size)
            XCTAssertTrue(history.makeBoardMoveTolerant(board, loc, pla), "test setup move must be legal")
        }
        return (board, history)
    }
}

/// Deterministic uniform-legal-policy evaluator (a value-neutral 0.5 win prob, flat legal policy),
/// mirroring `SearchExplorationTests`' stub so tree-reuse behavior is exercised with zero MLX/model
/// dependency.
private actor TreeReuseStubEvaluator: NNEvaluating {
    let nnLen: Int

    init(nnLen: Int) {
        self.nnLen = nnLen
    }

    func evaluate(_ requests: [NNRequest]) -> [NNOutput] {
        requests.map(output)
    }

    private func output(_ request: NNRequest) -> NNOutput {
        let policySize = nnLen * nnLen + 1
        var policy = [Float](repeating: -1, count: policySize)
        var legalPositions = [Int]()
        for position in 0 ..< policySize {
            let loc = NNPos.posToLoc(position, request.board.xSize, request.board.ySize, nnLen, nnLen)
            if loc != Board.nullLoc, request.history.isLegal(request.board, loc, request.nextPlayer) {
                legalPositions.append(position)
            }
        }
        for position in legalPositions {
            policy[position] = 1 / Float(legalPositions.count)
        }
        return NNOutput(
            nnHash: Hash128(),
            whiteWinProb: 0.5,
            whiteLossProb: 0.5,
            whiteNoResultProb: 0,
            whiteScoreMean: 0,
            whiteScoreMeanSq: 1,
            whiteLead: 0,
            varTimeLeft: 0,
            shorttermWinlossError: 0,
            shorttermScoreError: 0,
            policyProbs: policy,
            policyOptimismUsed: 0,
            nnXLen: nnLen,
            nnYLen: nnLen,
            whiteOwnerMap: nil
        )
    }
}
