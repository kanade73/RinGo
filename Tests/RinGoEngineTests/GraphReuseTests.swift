import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// Graph-safe tree reuse / GC across `Search.makeMove` under `SearchSettings.useGraphSearch`,
/// including a genuinely transposed reply and a long-chain memory-sanity check. Deterministic STUB
/// evaluators (shared with `GraphSearchTests`) so every case runs everywhere with no model fixture.
///
/// These pin the reuse behavior that replaced an earlier conservative clear-on-makeMove: makeMove now
/// PROMOTES the played child's subtree onto a fresh private root built on the real advanced history,
/// re-filters the root edges for legality, and mark-sweeps everything the new root can't reach. The
/// pure-tree (OFF) path stays pinned by `TreeReuseTests`/`GoldenTraceTests` and is untouched here.
///
/// Actor note: `Search`/`GTPEngine` methods are actor-isolated, and XCTest's `XCTAssert*`/`XCTUnwrap`
/// take SYNC autoclosures, so every actor call is hoisted into a `let` before the assertion (the same
/// pattern `GraphSearchTests`/`PonderTests` use).
final class GraphReuseTests: XCTestCase {
    // MARK: - T-R1: promotion preserves the played child's subtree

    func testPromotionPreservesPlayedChildStats() async throws {
        // The graph-ON analog of TreeReuseTests.testReRootingPreservesPlayedChildVisitCount: after
        // playing the best move, the re-rooted subtree's visits carry over (reuse, not a fresh drop).
        let search = makeGraphSearch(hotSet: corners7())
        let result = try await search.runSearch(maxVisits: 300)
        let best = result.bestMove
        let edgeVisits = try XCTUnwrap(result.visitsByMove.first { $0.loc == best }?.visits)
        XCTAssertGreaterThan(edgeVisits, 0)

        try await search.makeMove(best)
        let rootVisits = await search.currentRootVisits()
        let tableCount = await search.nodeTableCountForTests()
        // The promoted child's own NODE visits become the new root's visits — at least the parent edge
        // that led into it (they differ only if the child is a shared node; a root child is not, absent
        // capture-transpositions). A conservative clear would have made this 0.
        XCTAssertGreaterThanOrEqual(rootVisits, edgeVisits, "re-rooting must retain the played child's subtree")
        XCTAssertGreaterThan(tableCount, 0, "the promoted subtree stays in the table")
    }

    // MARK: - T-R2: mark-sweep keeps reachable, removes unreachable, no ARC leak

    func testSweepKeepsReachableRemovesUnreachableNoLeak() async throws {
        let search = makeGraphSearch(hotSet: corners7() + [loc7(3, 3)])
        let result = try await search.runSearch(maxVisits: 400)
        let tableBefore = await search.nodeTableCountForTests()
        XCTAssertGreaterThan(tableBefore, 1)
        // Weakly capture every current table node; a leak-free sweep frees each removed one.
        await search.armTableLivenessProbeForTests()

        try await search.makeMove(result.bestMove)

        let tableAfter = await search.nodeTableCountForTests()
        let reachable = await search.reachableNodeCountForTests()
        let alive = await search.tableLivenessAliveCountForTests()
        XCTAssertEqual(tableAfter, reachable - 1, "table == reachable minus the out-of-table root")
        XCTAssertLessThan(tableAfter, tableBefore, "the non-played lines must be swept away")
        // ARC-leak gate: the number of originally-captured nodes still alive equals exactly the
        // surviving table set — every swept node was freed (its cycle links were broken), none leaked.
        XCTAssertEqual(alive, tableAfter, "swept nodes must be freed — no ARC reference-cycle leak")
    }

    // MARK: - T-R3: strict root-edge legality re-filter + stats-only recompute (no double-increment)

    func testRootEdgeLegalityFilterDropsIllegalAndRecomputes() async throws {
        let search = makeGraphSearch(hotSet: corners7())
        let result = try await search.runSearch(maxVisits: 300)
        let reply = result.bestMove
        // Inject an edge into the played child whose move (the reply loc, now occupied) is illegal on
        // the real advanced position, so the strict root-edge re-filter must drop it on promotion.
        let injectedOptional = await search.injectIllegalPromotedEdgeForTests(replyLoc: reply)
        let injected = try XCTUnwrap(injectedOptional)
        XCTAssertGreaterThan(injected.visits, 1, "the played child must carry real subtree visits")

        try await search.makeMove(reply)

        let facts = await search.graphRootChildFactsForTests()
        let rootVisits = await search.currentRootVisits()
        XCTAssertFalse(facts.contains { $0.loc == reply }, "the illegal root edge must be filtered out")
        // Filtered visits recompute to exactly 1 + Σ(kept edge visits) == the child's pre-injection
        // visits, with NO double-increment (stats-only recompute is visitsToAdd == 0).
        let keptEdgeSum = facts.reduce(Int64(0)) { $0 + $1.edgeVisits }
        XCTAssertEqual(rootVisits, 1 + keptEdgeSum, "root visits == 1 + Σ kept edge visits (no double count)")
        XCTAssertEqual(rootVisits, injected.visits, "filtered visits recompute back to the child's own visits")
    }

    // MARK: - T-R4: setPosition full-table clear + no leak

    func testSetPositionFullClearNoLeak() async throws {
        let search = makeGraphSearch(hotSet: corners7())
        _ = try await search.runSearch(maxVisits: 300)
        let tableBefore = await search.nodeTableCountForTests()
        XCTAssertGreaterThan(tableBefore, 0)
        await search.armTableLivenessProbeForTests()

        let board = Board(7, 7)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        await search.setPosition(board: board, history: history, nextPlayer: .black)

        let tableAfter = await search.nodeTableCountForTests()
        let rootVisits = await search.currentRootVisits()
        let alive = await search.tableLivenessAliveCountForTests()
        XCTAssertEqual(tableAfter, 0, "setPosition must clear the whole table")
        XCTAssertEqual(rootVisits, 0, "setPosition installs a fresh 0-visit root")
        XCTAssertEqual(alive, 0, "every table node must be freed (no leak)")
    }

    // MARK: - T-R5: integrated reuse (old table copy removed + edge filtered + private root + no leak)

    func testReuseIntegrationOldCopyRemovedEdgeFilteredPrivateRootNoLeak() async throws {
        // Review §B10: root-out-of-table is not sufficient by itself — the same position can still live
        // as a table node. Exercise promotion + sweep + root-edge legality + old-copy removal together.
        let search = makeGraphSearch(hotSet: corners7())
        let result = try await search.runSearch(maxVisits: 320)
        let reply = result.bestMove

        let preBoard = await search.currentRootBoard()
        let preHistory = await search.currentRootHistory()
        let mover = preHistory.presumedNextMovePla
        let oldKeyOptional = await search.graphHashOfRootChildForTests(loc: reply)
        let oldKey = try XCTUnwrap(oldKeyOptional)
        let containedBefore = await search.nodeTableContainsForTests(oldKey)
        XCTAssertTrue(containedBefore, "the played child is an in-table node pre-move")
        let injectedOptional = await search.injectIllegalPromotedEdgeForTests(replyLoc: reply)
        let injected = try XCTUnwrap(injectedOptional)
        await search.armTableLivenessProbeForTests()

        try await search.makeMove(reply)

        // (a) the new root stands on the REAL advanced position (pre-move board + reply).
        let expectedBoard = preBoard.copy()
        let expectedHistory = preHistory.copy()
        expectedHistory.makeBoardMoveAssumeLegal(expectedBoard, reply, mover)
        let rootBoard = await search.currentRootBoard()
        let rootHistory = await search.currentRootHistory()
        XCTAssertEqual(rootBoard.posHash, expectedBoard.posHash, "new root sits on the real advanced position")
        XCTAssertEqual(rootHistory.moveHistory.last?.loc, reply, "new root's history ends on the played reply")
        // (b) the old in-table copy of the promoted position is gone (no surviving duplicate identity).
        let containedAfter = await search.nodeTableContainsForTests(oldKey)
        XCTAssertFalse(containedAfter, "old table copy of the promoted position removed")
        // (c) the illegal root edge was filtered, visits recomputed to the child's own visits.
        let facts = await search.graphRootChildFactsForTests()
        let rootVisits = await search.currentRootVisits()
        XCTAssertFalse(facts.contains { $0.loc == reply }, "illegal root edge filtered on promotion")
        XCTAssertEqual(rootVisits, injected.visits, "visits recompute to the child's own visits")
        // (d) no ARC leak: survivors == the new table set.
        let alive = await search.tableLivenessAliveCountForTests()
        let tableAfter = await search.nodeTableCountForTests()
        XCTAssertEqual(alive, tableAfter, "no ARC reference-cycle leak after the integrated re-root")
    }

    // MARK: - T-G6 extension (§B4): a genuinely transposed reply is promoted, not rejected

    func testTransposedReplyPromotedDespiteDivergentKoHistory() async throws {
        // Root at [b(2,2), w(6,6), b(6,2)] on 9x9 — WHITE to move. The reply w(2,6) reaches the 4-stone
        // board that ALSO arises via the transposed order [b(6,2), w(2,6), b(2,2), w(6,6)]. A shared node
        // for that position (created via the transposed path) carries a DIFFERENT koHashHistory than the
        // real advanced history — the exact case the old koHashHistory-equality gate wrongly rejects but
        // graph-safe promotion (fingerprint, not history equality) must accept. A root-child transposition
        // needs captures to occur naturally, so it is constructed via a seam here, mirroring the injected
        // T-G4/T-G5 scenarios documented in GraphSearchTests.
        let nnLen = 9
        let reply = Location.getLoc(2, 6, nnLen)
        let (rootBoard, rootHistory) = buildOrdered(nnLen, [(2, 2, .black), (6, 6, .white), (6, 2, .black)])
        let (transBoard, transHistory) = buildOrdered(
            nnLen, [(6, 2, .black), (2, 6, .white), (2, 2, .black), (6, 6, .white)]
        )
        // Real advanced position = root + reply.
        let realBoard = rootBoard.copy()
        let realHistory = rootHistory.copy()
        realHistory.makeBoardMoveAssumeLegal(realBoard, reply, .white)

        // Precondition: same final board + side to move, but different superko history.
        XCTAssertEqual(realBoard.posHash, transBoard.posHash, "the transposition must reach the same board")
        XCTAssertNotEqual(realHistory.koHashHistory, transHistory.koHashHistory, "…via a different order")
        // The OLD gate rejects; the graph-safe fingerprint accepts.
        XCTAssertFalse(
            Search.canReuseSubtree(
                childBoard: transBoard, childHistory: transHistory, childNextPla: .black,
                board: realBoard, history: realHistory, nextPla: .black
            ),
            "the koHashHistory-equality gate would WRONGLY reject this valid transposition"
        )
        XCTAssertTrue(
            Search.graphFingerprintMatches(
                lhsBoard: transBoard, lhsHistory: transHistory, lhsNextPla: .black,
                rhsBoard: realBoard, rhsHistory: realHistory, rhsNextPla: .black
            ),
            "graph-safe promotion's fingerprint accepts the genuine transposition"
        )

        let search = makeGraphSearch(
            nnLen: nnLen, hotSet: [Location.getLoc(4, 4, nnLen)],
            board: rootBoard, history: rootHistory, nextPlayer: .white
        )
        let transposedVisits: Int64 = 37
        let key = await search.injectTransposedReplyChildForTests(
            replyLoc: reply, board: transBoard, history: transHistory, visits: transposedVisits
        )

        try await search.makeMove(reply)

        let rootVisits = await search.currentRootVisits()
        let stillContainsOldCopy = await search.nodeTableContainsForTests(key)
        // Promotion accepted the transposed reply: its subtree visits survive on the new private root.
        XCTAssertEqual(
            rootVisits,
            transposedVisits,
            "the transposed reply's pondered subtree must be promoted, not dropped"
        )
        // The old in-table copy of that position is swept (single identity; root is out of table).
        XCTAssertFalse(stillContainsOldCopy, "old table copy of the transposed position removed")
    }

    // MARK: - Long makeMove chain stays bounded (memory sanity)

    func testRepeatedMakeMoveChainKeepsTableBounded() async throws {
        let search = makeGraphSearch(hotSet: corners7() + [loc7(3, 3)])
        var maxTable = 0
        for _ in 0 ..< 10 {
            let result = try await search.runSearch(maxVisits: 120)
            try await search.makeMove(result.bestMove)
            let table = await search.nodeTableCountForTests()
            let reachable = await search.reachableNodeCountForTests()
            XCTAssertEqual(table, reachable - 1, "each re-root keeps table == reachable minus the root")
            maxTable = max(maxTable, table)
        }
        // 120 fresh visits per move on top of a reused head start: the retained subtree is bounded by a
        // single move's worth of new work plus what stays valid, never the cumulative 10*120. A sweep
        // leak (cycles never broken) would let the table grow without bound instead.
        XCTAssertLessThan(maxTable, 500, "the node table stays bounded across a long makeMove chain")
    }

    // MARK: - Ponder × graph coherence (end-to-end): a pondered subtree is harvested on the reply

    func testGraphPonderedSubtreeIsHarvestedWhenOpponentPlaysIt() async throws {
        // The graph-ON analog of PonderTests.testPonderedSubtreeIsHarvestedWhenOpponentPlaysIt. With
        // graph search on, the search→ponder→play flow must reuse the graph exactly like consecutive
        // searches: pondering grows the DAG on the same actor, stopPondering() quiesces it, then the
        // played reply promotes the pondered subtree (graph-safe) instead of clearing it.
        let nnLen = 9
        let ponderCap: Int64 = 200
        let engine = makeGraphEngine(nnLen: nnLen, ponderEnabled: true, ponderMaxVisits: ponderCap, maxVisits: 32)
        _ = await engine.handle("boardsize \(nnLen)")
        _ = await engine.handle("komi 7.0")
        _ = await engine.handle("genmove B") // black plays; white (opponent) to move

        let afterGenmove = await engine.searchRootVisits()
        XCTAssertGreaterThan(afterGenmove, 0, "the white-to-move root carries visits harvested from genmove")

        await engine.startPonderingIfEnabled()
        let ponderHandle = await engine.ponderTaskHandle()
        let ponderTask = try XCTUnwrap(ponderHandle)
        await ponderTask.value
        let afterPonder = await engine.searchRootVisits()
        XCTAssertEqual(afterPonder, afterGenmove + ponderCap, "pondering adds exactly its capped budget of NEW visits")

        let topChildOptional = await engine.topSearchRootChild()
        let topChild = try XCTUnwrap(topChildOptional)
        XCTAssertGreaterThan(topChild.visits, 0)
        await engine.stopPondering() // mirror the GTP loop: quiesce before the tree-mutating command

        let vertex = Location.toString(topChild.loc, nnLen, nnLen)
        let playResult = await engine.handle("play W \(vertex)")
        XCTAssertEqual(playResult.response, "= \n\n", "the pondered opponent move must be legal and accepted")

        let harvested = await engine.searchRootVisits()
        XCTAssertGreaterThanOrEqual(harvested, topChild.visits, "graph re-rooting must harvest the pondered subtree")
        XCTAssertGreaterThan(harvested, 0, "the harvested head start must be non-empty")
        // The next genmove (black to move again after B, W) completes normally on the harvested graph.
        let next = await engine.handle("genmove B")
        XCTAssertTrue((next.response ?? "").hasPrefix("= "), "the next genmove must succeed on the harvested graph")
    }

    // MARK: - Helpers

    private func loc7(_ x: Int, _ y: Int) -> Loc {
        Location.getLoc(x, y, 7)
    }

    private func corners7() -> [Loc] {
        [loc7(1, 1), loc7(5, 5), loc7(1, 5), loc7(5, 1)]
    }

    private func makeGraphSearch(
        nnLen: Int = 7,
        hotSet: [Loc],
        board: Board? = nil,
        history: BoardHistory? = nil,
        nextPlayer: Player = .black
    ) -> Search {
        let realBoard = board ?? Board(nnLen, nnLen)
        let realHistory = history ?? BoardHistory(
            realBoard,
            pla: nextPlayer,
            rules: .getTrompTaylorish(),
            encorePhase: 0
        )
        var settings = SearchSettings()
        settings.numSearchWorkers = 1
        settings.leafBatchSize = 8
        settings.useGraphSearch = true
        return Search(
            evaluator: HotSetStub(nnLen: nnLen, hotSet: hotSet),
            nnXLen: nnLen,
            nnYLen: nnLen,
            precision: .float32,
            settings: settings,
            board: realBoard,
            history: realHistory,
            nextPlayer: nextPlayer,
            fp32FallbackFactory: nil
        )
    }

    private func makeGraphEngine(
        nnLen: Int,
        ponderEnabled: Bool,
        ponderMaxVisits: Int64,
        maxVisits: Int64
    ) -> GTPEngine {
        let hot = [
            Location.getLoc(4, 4, nnLen), Location.getLoc(2, 2, nnLen), Location.getLoc(6, 6, nnLen),
            Location.getLoc(6, 2, nnLen), Location.getLoc(2, 6, nnLen),
        ]
        var settings = SearchSettings()
        settings.maxVisits = maxVisits
        settings.numSearchWorkers = 4
        settings.leafBatchSize = 8
        settings.useGraphSearch = true
        return GTPEngine(
            evaluator: HotSetStub(nnLen: nnLen, hotSet: hot),
            nnXLen: nnLen,
            nnYLen: nnLen,
            precision: .float32,
            settings: settings,
            rules: .getTrompTaylorish(),
            nnLength: .fixed(nnLen),
            warmupBucketSizes: [1, 2, 4, 8],
            evaluatorFactory: { _, _ in HotSetStub(nnLen: nnLen, hotSet: hot) },
            initialTimeSeconds: nil,
            ponderEnabled: ponderEnabled,
            ponderMaxVisits: ponderMaxVisits
        )
    }

    private func buildOrdered(_ nnLen: Int, _ moves: [(Int, Int, Player)]) -> (Board, BoardHistory) {
        let board = Board(nnLen, nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        for (x, y, pla) in moves {
            XCTAssertTrue(
                history.makeBoardMoveTolerant(board, Location.getLoc(x, y, nnLen), pla),
                "test setup move (\(x),\(y)) must be legal"
            )
        }
        return (board, history)
    }
}
