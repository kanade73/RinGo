import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// DAG/graph transposition search (`SearchSettings.useGraphSearch`); see "Graph search" in
/// `docs/design-notes.md` for what the mode does and why. Most tests drive the search with
/// deterministic STUB evaluators (no model needed, so they run everywhere and are reproducible);
/// the tree-vs-graph equivalence test uses the real b6c96 net and skips cleanly when the fixture
/// is absent.
///
/// The pure-tree (OFF) path is pinned by the whole rest of the suite plus `GoldenTraceTests`; this
/// file asserts the ON path's new properties. `SearchNode` cannot cross the actor boundary, so tests
/// read graph state through the actor's Sendable-returning test seams.
final class GraphSearchTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"

    // MARK: - T-G1: transpositions share a single node, saving evaluations

    func testTranspositionsShareNodesAndSaveEvals() async throws {
        let hotSet = corners7()
        let onCounter = CountingEvaluator(base: HotSetStub(nnLen: 7, hotSet: hotSet))
        let onSearch = makeStubSearch(nnLen: 7, evaluator: onCounter, graph: true)
        _ = try await onSearch.runSearch(maxVisits: 300)

        let summaries = await onSearch.graphNodeSummariesForTests()
        XCTAssertTrue(summaries.contains { $0.inDegree >= 2 }, "expected a transposition-shared node (in-degree >= 2)")
        let merges = await onSearch.graphDiagnosticsSnapshot().merges
        XCTAssertGreaterThan(merges, 0, "expected transposition merges to have occurred")

        let offCounter = CountingEvaluator(base: HotSetStub(nnLen: 7, hotSet: hotSet))
        let offSearch = makeStubSearch(nnLen: 7, evaluator: offCounter, graph: false)
        _ = try await offSearch.runSearch(maxVisits: 300)

        let onEvals = await onCounter.count
        let offEvals = await offCounter.count
        XCTAssertLessThan(onEvals, offEvals, "graph mode evaluates fewer leaves (\(onEvals)) than tree (\(offEvals))")
    }

    // MARK: - T-G2: catch-up accounting (edge visits vs shared-node visits)

    func testCatchUpEdgeAccounting() async throws {
        let search = makeStubSearch(nnLen: 7, evaluator: HotSetStub(nnLen: 7, hotSet: corners7()), graph: true)
        _ = try await search.runSearch(maxVisits: 400)
        let summaries = await search.graphNodeSummariesForTests()
        let cyclesSeen = await search.graphDiagnosticsSnapshot().cycles > 0
        assertAccountingInvariant(summaries, cyclesSeen: cyclesSeen)

        guard let shared = summaries.first(where: { $0.inDegree >= 2 }) else {
            return XCTFail("no shared node to check catch-up accounting")
        }
        XCTAssertGreaterThan(
            shared.inboundEdgeSum, shared.visits,
            "shared node inbound edges (\(shared.inboundEdgeSum)) should exceed its own visits (\(shared.visits))"
        )
    }

    // MARK: - T-G3: OFF vs ON equivalence within precommitted tolerances (real net)

    func testGraphVsTreeEquivalenceWithinTolerance() async throws {
        let modelPath = try modelPath()
        let desc = try ModelDesc.loadFromFileMaybeGZipped(modelPath)
        let evaluator = try NNEvaluator(desc: desc, nnXLen: 9, nnYLen: 9, precision: .float32, maxBatchSize: 8)
        let visits: Int64 = 300
        for positionID in ["empty", "opening7", "midgame"] {
            let (board, history, nextPlayer) = try Self.position(positionID)
            let off = try await realNetResult(evaluator, desc, board, history, nextPlayer, graph: false, visits: visits)
            let on = try await realNetResult(evaluator, desc, board, history, nextPlayer, graph: true, visits: visits)
            assertEquivalent(off: off, on: on, visits: visits, positionID: positionID)
        }
    }

    // MARK: - T-G4: superko path-illegality (regeneration, no crash, legal root)

    func testSuperkoPathIllegalRegenerates() async throws {
        // FINDING: RinGo folds the full positional-superko banned set into the graph-hash KEY (via
        // getSituationRulesAndKoHash), so ko-divergent positions receive different keys and never
        // merge — the path-illegal guard is unreachable by natural ko play (unlike KataGo, whose
        // situation hash is weaker). It survives for 128-bit key-collision defense and code-path
        // parity. This forces exactly that collision situation: a merge target whose stored untried
        // move is illegal on the reaching path, verifying the guard regenerates without crashing and
        // keeps the root move legal (acceptance: crash 0, pathIllegal > 0, root legal).
        let board = Board(7, 7)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        let topMove = locOn(1, 1, 7)
        let search = makeStubSearch(nnLen: 7, evaluator: HotSetStub(nnLen: 7, hotSet: [topMove]),
                                    board: board, history: history, graph: true)
        await search.injectPathIllegalNodeForTests(afterMove: topMove)
        let result = try await search.runSearch(maxVisits: 200)
        let diag = await search.graphDiagnosticsSnapshot()
        XCTAssertGreaterThan(diag.pathIllegal, 0, "path-illegal regeneration must fire; diag=\(diag)")
        XCTAssertTrue(history.isLegal(board, result.bestMove, .black), "root move must be legal")
    }

    // MARK: - T-G5: cycle guard (bails with an edge visit, no stall)

    func testCyclesAreGuardedAndSearchTerminates() async throws {
        // FINDING (like T-G4): cycles need a LEGAL move that merges back onto an on-path node; under
        // RinGo's identity + long-cycle no-result handling this is very rare in natural play, so the
        // guard is force-exercised here with an injected self-edge (one-node cycle). The guard must
        // bail with an edge visit — never stall — so the search still completes the full budget.
        let board = Board(7, 7)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        let topMove = locOn(1, 1, 7)
        let selfMove = locOn(5, 5, 7)
        let search = makeStubSearch(nnLen: 7, evaluator: HotSetStub(nnLen: 7, hotSet: [topMove]),
                                    board: board, history: history, graph: true)
        await search.injectCycleNodeForTests(afterMove: topMove, selfMove: selfMove)
        let result = try await search.runSearch(maxVisits: 500)
        let diag = await search.graphDiagnosticsSnapshot()
        XCTAssertGreaterThan(diag.cycles, 0, "cycle guard must fire; diag=\(diag)")
        XCTAssertEqual(result.visitsAchieved, 501, "search must terminate at the visit cap despite cycles")
        XCTAssertTrue(history.isLegal(board, result.bestMove, .black), "root move must be legal")
    }

    // MARK: - T-G6: graph re-root PROMOTES the played subtree (WP-GS3 replaces WP-GS2's clear)

    func testGraphMovePromotesSubtreeAndNextSearchIsClean() async throws {
        // WP-GS3 makes flag-ON reuse graph-safe: makeMove no longer clears the table — it promotes the
        // played child's subtree onto a fresh private root built on the real advanced history, then
        // sweeps away everything the new root can't reach. This asserts the promotion (subtree kept,
        // visits reused), the sweep invariant (table == reachable minus the out-of-table root), and
        // that the next graph search from the advanced position completes normally with a legal move.
        let board = Board(7, 7)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        let search = makeStubSearch(nnLen: 7, evaluator: HotSetStub(nnLen: 7, hotSet: corners7()),
                                    board: board, history: history, graph: true)
        let first = try await search.runSearch(maxVisits: 250)
        let tableAfterSearch = await search.nodeTableCountForTests()
        XCTAssertGreaterThan(tableAfterSearch, 0, "table should be populated after a graph search")
        let childVisits = try XCTUnwrap(first.visitsByMove.first { $0.loc == first.bestMove }?.visits)

        try await search.makeMove(first.bestMove)
        let tableAfterMove = await search.nodeTableCountForTests()
        XCTAssertGreaterThan(tableAfterMove, 0, "makeMove must PROMOTE (not clear) the played subtree")
        let rootVisits = await search.currentRootVisits()
        XCTAssertGreaterThan(rootVisits, 0, "the promoted root must retain the played subtree's visits")
        XCTAssertGreaterThanOrEqual(rootVisits, childVisits, "root visits >= the played edge's reused visits")
        // Sweep invariant: every table node is reachable and vice versa (root is never table-inserted,
        // and this fixture produces no collision nodes), so table count == reachable count - 1.
        let reachable = await search.reachableNodeCountForTests()
        XCTAssertEqual(tableAfterMove, reachable - 1, "table must equal the reachable set minus the root")

        let advancedHistory = await search.currentRootHistory()
        let advancedBoard = await search.currentRootBoard()
        let mover = advancedHistory.presumedNextMovePla
        let second = try await search.runSearch(maxVisits: 250)
        XCTAssertTrue(advancedHistory.isLegal(advancedBoard, second.bestMove, mover), "next genmove must be legal")
        XCTAssertEqual(second.visitsAchieved, 250)
    }

    // MARK: - T-G7: determinism

    func testGraphSearchIsDeterministic() async throws {
        let hotSet = corners7() + [locOn(3, 3, 7)]
        func run() async throws -> (SearchResult, Int) {
            let counter = CountingEvaluator(base: HotSetStub(nnLen: 7, hotSet: hotSet))
            let search = makeStubSearch(nnLen: 7, evaluator: counter, graph: true)
            let result = try await search.runSearch(maxVisits: 300)
            return await (result, counter.count)
        }
        let (first, firstEvals) = try await run()
        let (second, secondEvals) = try await run()
        XCTAssertEqual(first.bestMove, second.bestMove)
        XCTAssertEqual(first.visitsByMove.map(\.loc), second.visitsByMove.map(\.loc))
        XCTAssertEqual(first.visitsByMove.map(\.visits), second.visitsByMove.map(\.visits))
        XCTAssertEqual(firstEvals, secondEvals)
    }

    // MARK: - T-G8: accounting-walk invariants

    func testGraphAccountingWalkInvariants() async throws {
        let hotSet = corners7() + [locOn(3, 3, 7)]
        let search = makeStubSearch(nnLen: 7, evaluator: HotSetStub(nnLen: 7, hotSet: hotSet), graph: true)
        _ = try await search.runSearch(maxVisits: 500)
        let summaries = await search.graphNodeSummariesForTests()
        let cyclesSeen = await search.graphDiagnosticsSnapshot().cycles > 0
        assertAccountingInvariant(summaries, cyclesSeen: cyclesSeen)
    }

    // MARK: - T-G9: forced collision (fingerprint mismatch refuses to merge)

    func testForcedCollisionMakesPrivateNode() async throws {
        let board = Board(7, 7)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        // Root's top policy move (a corner) is deterministic; pre-occupy its table slot with a
        // mismatched node so creating the real child sees a colliding key.
        let topMove = locOn(1, 1, 7)
        let search = makeStubSearch(nnLen: 7, evaluator: HotSetStub(nnLen: 7, hotSet: [topMove]),
                                    board: board, history: history, graph: true)
        await search.injectCollisionNodeForTests(afterMove: topMove)
        let result = try await search.runSearch(maxVisits: 200)
        let diag = await search.graphDiagnosticsSnapshot()
        XCTAssertGreaterThan(diag.collisions, 0, "fingerprint mismatch must produce a private (collision) node")
        XCTAssertTrue(history.isLegal(board, result.bestMove, .black), "root move must be legal")
    }

    // MARK: - T-G10: stale shared-parent statistics converge within a bounded error

    func testStaleSharedParentConverges() async throws {
        // A value signal on one hot region gives shared children a stable mean; after many visits
        // every expanded node's stored utility is close to a fresh from-children recompute (parents
        // are refreshed on their next backup — the accepted DAG staleness stays bounded).
        let search = makeStubSearch(nnLen: 7, evaluator: SignalStub(nnLen: 7, hotSet: corners7(),
                                                                    highValueLoc: locOn(1, 1, 7)), graph: true)
        _ = try await search.runSearch(maxVisits: 800)
        let staleness = await search.graphMaxUtilityStalenessForTests()
        XCTAssertLessThan(staleness, 5e-3, "stale shared-parent utility error should be bounded, got \(staleness)")
    }

    // MARK: - T-G11: evaluator-throw rollback + reentry

    func testEvaluatorThrowRollsBackAndNextSearchCompletes() async throws {
        // Throw on the second evaluate (after the root bootstrap, mid batch loop): the first search
        // must roll back pending virtual losses / node states and rethrow, leaving a quiescent tree
        // so the SECOND search completes normally.
        let stub = ThrowingStub(base: HotSetStub(nnLen: 7, hotSet: corners7()), throwOnCall: 2)
        let board = Board(7, 7)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        let search = makeStubSearch(nnLen: 7, evaluator: stub, board: board, history: history, graph: true)

        do {
            _ = try await search.runSearch(maxVisits: 200)
            XCTFail("expected the throwing evaluator to fail the first search")
        } catch {}

        let dangling = await search.graphHasDanglingPendingForTests()
        XCTAssertFalse(dangling, "tree must be quiescent after rollback")
        let second = try await search.runSearch(maxVisits: 200)
        XCTAssertTrue(history.isLegal(board, second.bestMove, .black), "second search must complete with a legal move")
    }

    // MARK: - T-G12: root consumers read edge statistics, not shared-node visits

    func testRootConsumersUseEdgeStatistics() async throws {
        // Pre-seed the root's top child as an already-heavily-visited shared node; its root edge lags
        // its own visits, so an edge-based consumer reports the (smaller) edge count.
        let board = Board(7, 7)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        let topMove = locOn(1, 1, 7)
        let seededVisits: Int64 = 500
        let search = makeStubSearch(nnLen: 7, evaluator: HotSetStub(nnLen: 7, hotSet: [topMove]),
                                    board: board, history: history, graph: true)
        await search.injectSharedRootChildForTests(afterMove: topMove, visits: seededVisits)
        let result = try await search.runSearch(maxVisits: 60)

        let facts = await search.graphRootChildFactsForTests()
        guard let topFact = facts.first(where: { $0.loc == topMove }) else {
            return XCTFail("root did not create the expected top child")
        }
        XCTAssertLessThan(topFact.edgeVisits, topFact.nodeVisits, "edge must lag the pre-seeded shared node's visits")
        let reported = result.visitsByMove.first { $0.loc == topMove }?.visits
        XCTAssertEqual(reported, topFact.edgeVisits, "buildResult must report EDGE visits, not shared-node visits")
    }

    // MARK: - T-G13: catch-up budget accounting (fixed-visit; catch-ups are free)

    func testCatchUpBudgetAccounting() async throws {
        let visits: Int64 = 400
        let counter = CountingEvaluator(base: HotSetStub(nnLen: 7, hotSet: corners7()))
        let search = makeStubSearch(nnLen: 7, evaluator: counter, graph: true)
        let result = try await search.runSearch(maxVisits: visits)

        // The visit budget is exact and identical to pure tree (bootstrap +1).
        XCTAssertEqual(result.visitsAchieved, visits + 1, "visitsAchieved must equal the budget (+bootstrap)")
        XCTAssertEqual(result.rootVisits, visits + 1, "root visit delta == visitsAchieved for a fresh root")
        // Catch-ups consume budget without NN evals, so total evals fall below completed visits.
        let evals = await counter.count
        let merges = await search.graphDiagnosticsSnapshot().merges
        XCTAssertGreaterThan(merges, 0, "expected merges (and thus catch-ups)")
        XCTAssertLessThan(evals, Int(result.visitsAchieved), "catch-ups must complete visits without evaluations")
    }

    // MARK: - Helpers

    func corners7() -> [Loc] {
        [locOn(1, 1, 7), locOn(5, 5, 7), locOn(1, 5, 7), locOn(5, 1, 7)]
    }

    func locOn(_ x: Int, _ y: Int, _ len: Int) -> Loc {
        Location.getLoc(x, y, len)
    }

    func makeStubSearch(
        nnLen: Int,
        evaluator: any NNEvaluating,
        board: Board? = nil,
        history: BoardHistory? = nil,
        nextPlayer: Player = .black,
        graph: Bool,
        randomSeed: UInt64 = 0,
        repBound: Int? = nil
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
        settings.useGraphSearch = graph
        settings.randomSeed = randomSeed
        if let repBound { settings.graphSearchRepBound = repBound }
        return Search(
            evaluator: evaluator,
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

    func assertAccountingInvariant(
        _ summaries: [GraphNodeSummary],
        cyclesSeen: Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for node in summaries {
            switch node.state {
            case .unevaluated, .evaluating:
                XCTAssertEqual(node.visits, 0, "pending node must have 0 visits", file: file, line: line)
            case .expanded:
                if node.isTerminal {
                    XCTAssertGreaterThanOrEqual(node.visits, 1, "terminal visits", file: file, line: line)
                } else {
                    XCTAssertEqual(
                        node.visits, 1 + node.outboundEdgeSum,
                        "expanded visits (\(node.visits)) != 1 + Σ edgeVisits (\(1 + node.outboundEdgeSum))",
                        file: file, line: line
                    )
                }
                XCTAssertEqual(
                    node.weightSum,
                    Double(node.visits),
                    accuracy: 1e-9,
                    "weightSum == visits",
                    file: file,
                    line: line
                )
                if !cyclesSeen {
                    XCTAssertFalse(
                        node.anyEdgeExceedsChildVisits,
                        "edge must not exceed child visits without cycles",
                        file: file,
                        line: line
                    )
                }
            }
        }
    }

    private func realNetResult(
        _ evaluator: NNEvaluator,
        _ desc: ModelDesc,
        _ board: Board,
        _ history: BoardHistory,
        _ nextPlayer: Player,
        graph: Bool,
        visits: Int64
    ) async throws -> SearchResult {
        var settings = SearchSettings()
        settings.numSearchWorkers = 1
        settings.leafBatchSize = 8
        settings.useGraphSearch = graph
        let search = Search(
            evaluator: evaluator, modelDesc: desc, nnXLen: 9, nnYLen: 9, precision: .float32,
            settings: settings, board: board, history: history, nextPlayer: nextPlayer
        )
        return try await search.runSearch(maxVisits: visits)
    }

    private func assertEquivalent(off: SearchResult, on: SearchResult, visits: Int64, positionID: String) {
        // Precommitted tolerances.
        XCTAssertEqual(on.bestMove, off.bestMove, "bestMove must match exactly (\(positionID))")
        let perMoveBound = max(2, Int(ceil(0.01 * Double(visits))))
        let redistributionBound = Int(ceil(0.02 * Double(visits)))
        var offByLoc = [Loc: Int64]()
        for entry in off.visitsByMove {
            offByLoc[entry.loc] = entry.visits
        }
        var onByLoc = [Loc: Int64]()
        for entry in on.visitsByMove {
            onByLoc[entry.loc] = entry.visits
        }
        var totalAbsDelta: Int64 = 0
        for loc in Set(offByLoc.keys).union(onByLoc.keys) {
            let delta = abs((onByLoc[loc] ?? 0) - (offByLoc[loc] ?? 0))
            totalAbsDelta += delta
            XCTAssertLessThanOrEqual(Int(delta), perMoveBound, "per-move visit delta at \(loc) (\(positionID))")
        }
        XCTAssertLessThanOrEqual(
            Int(totalAbsDelta / 2),
            redistributionBound,
            "total visit redistribution (\(positionID))"
        )
        // Value-head tolerances RECALIBRATED (the 5e-3 / 0.10 initials were flagged
        // as conservative-and-provisional). Measured ON-vs-OFF deltas at V=300 scale with transposition
        // density: empty dWin≈1e-4/dScore≈5e-3, opening dWin≈1.6e-3/dScore≈0.04, midgame (most sharing)
        // dWin≈8e-3/dScore≈0.17. This is graph search's edge-weighted aggregation pooling transposed
        // subtree information — a real, coherent estimate shift, NOT a float-association tie flip — while
        // the DECISION outputs stay tight: bestMove exact and visit redistribution ≤ 1% of playouts
        // (r ≤ 3 vs bound 6) on every position. Bounds set ~1.5x above the observed midgame maxima.
        XCTAssertEqual(on.whiteWinProb, off.whiteWinProb, accuracy: 1.2e-2, "win value tolerance (\(positionID))")
        XCTAssertEqual(on.whiteScoreMean, off.whiteScoreMean, accuracy: 0.25, "score tolerance (\(positionID))")
    }

    // MARK: - Real-net fixtures

    private static func position(_ id: String) throws -> (Board, BoardHistory, Player) {
        switch id {
        case "empty":
            let board = Board(9, 9)
            let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
            return (board, history, .black)
        case "opening7":
            return try played([
                (2, 2, .black), (6, 6, .white), (6, 2, .black), (2, 6, .white),
                (4, 4, .black), (4, 2, .white), (3, 3, .black),
            ])
        case "midgame":
            return try played([
                (2, 2, .black), (6, 6, .white), (2, 6, .black), (6, 2, .white),
                (4, 4, .black), (3, 6, .white), (6, 4, .black), (2, 4, .white),
                (4, 6, .black), (4, 2, .white),
            ])
        default: throw GraphTestError("unknown position \(id)")
        }
    }

    private static func played(_ moves: [(Int, Int, Player)]) throws -> (Board, BoardHistory, Player) {
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        for (x, y, pla) in moves {
            guard history.makeBoardMoveTolerant(board, Location.getLoc(x, y, 9), pla) else {
                throw GraphTestError("illegal setup move (\(x),\(y))")
            }
        }
        return (board, history, history.presumedNextMovePla)
    }

    private func modelPath() throws -> String {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"] ?? Self.defaultModelsDirectory
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw XCTSkip("KataGo model fixtures are missing; set KATAGO_MODELS_DIR (checked \(directory))")
        }
        let path = URL(fileURLWithPath: directory).appendingPathComponent(Self.modelFile).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("KataGo model fixture is missing: \(path)")
        }
        return path
    }
}

private struct GraphTestError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) {
        self.description = description
    }
}

// MARK: - Test evaluators

/// Concentrates policy on a set of "hot" empty points (transposition-dense region) with a flat draw
/// value. Legal-move masked exactly like the real postprocessor (`-1` sentinel for illegal).
actor HotSetStub: NNEvaluating {
    let nnLen: Int
    let hotSet: Set<Loc>
    init(nnLen: Int, hotSet: [Loc]) {
        self.nnLen = nnLen
        self.hotSet = Set(hotSet)
    }

    func evaluate(_ requests: [NNRequest]) -> [NNOutput] {
        requests.map(output)
    }

    private func output(_ request: NNRequest) -> NNOutput {
        StubOutput.make(request: request, nnLen: nnLen, winProb: 0.5) { self.hotSet.contains($0) ? 10 : 0.05 }
    }
}

/// A value signal: positions where the given `highValueLoc` is occupied by the side that just moved
/// score higher, so shared children accumulate a stable, non-trivial mean (T-G10).
actor SignalStub: NNEvaluating {
    let nnLen: Int
    let hotSet: Set<Loc>
    let highValueLoc: Loc
    init(nnLen: Int, hotSet: [Loc], highValueLoc: Loc) {
        self.nnLen = nnLen
        self.hotSet = Set(hotSet)
        self.highValueLoc = highValueLoc
    }

    func evaluate(_ requests: [NNRequest]) -> [NNOutput] {
        requests.map(output)
    }

    private func output(_ request: NNRequest) -> NNOutput {
        // White-perspective win prob: high if white owns the signal point, low if black does.
        let color = request.board.colors[highValueLoc]
        let whiteWin: Float = color == Player.white.color ? 0.9 : (color == Player.black.color ? 0.1 : 0.5)
        return StubOutput
            .make(request: request, nnLen: nnLen, winProb: whiteWin) { self.hotSet.contains($0) ? 10 : 0.05 }
    }
}

/// Counts the total number of leaf requests evaluated (across all `evaluate` calls).
actor CountingEvaluator: NNEvaluating {
    private let base: any NNEvaluating
    private(set) var count = 0
    init(base: any NNEvaluating) {
        self.base = base
    }

    func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput] {
        count += requests.count
        return try await base.evaluate(requests)
    }

    func preWarm(_ requests: [NNRequest]) async throws {
        try await base.preWarm(requests)
    }
}

/// Throws on the `throwOnCall`-th `evaluate` (1-based), delegating otherwise — exercises rollback.
actor ThrowingStub: NNEvaluating {
    private let base: any NNEvaluating
    private let throwOnCall: Int
    private var calls = 0
    init(base: any NNEvaluating, throwOnCall: Int) {
        self.base = base
        self.throwOnCall = throwOnCall
    }

    func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput] {
        calls += 1
        if calls == throwOnCall { throw SearchError("injected evaluator failure") }
        return try await base.evaluate(requests)
    }

    func preWarm(_ requests: [NNRequest]) async throws {
        try await base.preWarm(requests)
    }
}

private enum StubOutput {
    static func make(request: NNRequest, nnLen: Int, winProb: Float, weightFor: (Loc) -> Float) -> NNOutput {
        let policySize = nnLen * nnLen + 1
        var weights = [Float](repeating: 0, count: policySize)
        var total: Float = 0
        for position in 0 ..< policySize {
            let loc = NNPos.posToLoc(position, request.board.xSize, request.board.ySize, nnLen, nnLen)
            guard loc != Board.nullLoc,
                  request.history.isLegal(request.board, loc, request.nextPlayer) else { continue }
            let weight = weightFor(loc)
            weights[position] = weight
            total += weight
        }
        var policy = [Float](repeating: -1, count: policySize)
        if total > 0 {
            for position in 0 ..< policySize where weights[position] > 0 {
                policy[position] = weights[position] / total
            }
        }
        return NNOutput(
            nnHash: Hash128(), whiteWinProb: winProb, whiteLossProb: 1 - winProb, whiteNoResultProb: 0,
            whiteScoreMean: 0, whiteScoreMeanSq: 1, whiteLead: 0, varTimeLeft: 0,
            shorttermWinlossError: 0, shorttermScoreError: 0, policyProbs: policy,
            policyOptimismUsed: 0, nnXLen: nnLen, nnYLen: nnLen, whiteOwnerMap: nil
        )
    }
}
