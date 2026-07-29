import Foundation
@testable import ringo
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// Tests for WP-10 pondering ("think on the opponent's time"). All host-only, driven by the
/// deterministic uniform-policy stub evaluator so no MLX / model fixture is needed. Covers:
///   (A) `Search.runSearch` cooperatively honors task cancellation (the mechanism the GTP loop
///       uses to stop a ponder the instant the next command arrives);
///   (B) the `-ponder`/`-ponder-max-visits` CLI flags parse;
///   (C) the GTPEngine ponder lifecycle: it starts only on the opponent-to-move root after a
///       genmove, never before our color is known, never on our own turn, and never with ponder
///       off; and `stopPondering` cancels-and-awaits cleanly;
///   (D) the pondered subtree is harvested by tree reuse when the opponent actually plays the
///       move that was pondered (root visits after `play` == the pondered child's visits).
final class PonderTests: XCTestCase {
    private static let nnLen = 9

    // MARK: - (A) cooperative cancellation of runSearch

    func testRunSearchStopsPromptlyWhenTaskCancelled() async throws {
        let search = makeSearch()
        // A budget far larger than anything reachable in the sleep window below, so a completed
        // (non-cancelled) search would have to run for seconds — proving the early return is
        // cancellation, not budget exhaustion.
        let budget: Int64 = 5_000_000
        let task = Task { try await search.runSearch(maxVisits: budget) }
        try await Task.sleep(nanoseconds: 20_000_000) // 20 ms of churn
        task.cancel()
        let result = try await task.value
        XCTAssertGreaterThan(result.visitsAchieved, 0, "the search should make some progress before cancellation")
        XCTAssertLessThan(
            result.visitsAchieved, budget,
            "cancellation must break the visit loop well before the full budget is spent"
        )
    }

    func testUncancelledRunSearchStillReachesFullBudget() async throws {
        // Guards the flip side of (A): the added cancellation check is a no-op when the task is not
        // cancelled, so an ordinary search still spends its whole budget (bit-identical path). A
        // fresh root reports maxVisits + 1 (the one-time root-bootstrap eval is not counted against
        // the loop budget); the point here is only that the loop was not truncated early.
        let search = makeSearch()
        let result = try await search.runSearch(maxVisits: 200)
        XCTAssertGreaterThanOrEqual(
            result.visitsAchieved, 200,
            "an uncancelled search must spend its full visit budget"
        )
    }

    // MARK: - (B) CLI flag parsing

    func testPonderFlagParsing() throws {
        XCTAssertFalse(try GTPCommand.parse(["-model", "m.bin.gz"]).ponder)
        XCTAssertEqual(try GTPCommand.parse(["-model", "m.bin.gz"]).ponderMaxVisits, 0)
        XCTAssertTrue(try GTPCommand.parse(["-model", "m.bin.gz", "-ponder", "on"]).ponder)
        XCTAssertTrue(try GTPCommand.parse(["-model", "m.bin.gz", "-ponder", "true"]).ponder)
        XCTAssertFalse(try GTPCommand.parse(["-model", "m.bin.gz", "-ponder", "off"]).ponder)
        XCTAssertEqual(
            try GTPCommand.parse(["-model", "m.bin.gz", "-ponder-max-visits", "1234"]).ponderMaxVisits,
            1234
        )
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "m.bin.gz", "-ponder", "bogus"]))
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "m.bin.gz", "-ponder-max-visits", "0"]))
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "m.bin.gz", "-ponder-max-visits", "notanumber"]))
    }

    func testGPUCacheLimitFlagParsing() throws {
        XCTAssertNil(try GTPCommand.parse(["-model", "m.bin.gz"]).gpuCacheLimitMB)
        XCTAssertEqual(
            try GTPCommand.parse(["-model", "m.bin.gz", "-gpu-cache-limit-mb", "512"]).gpuCacheLimitMB,
            512
        )
        XCTAssertEqual(
            try GTPCommand.parse(["-model", "m.bin.gz", "-gpu-cache-limit-mb", "0"]).gpuCacheLimitMB,
            0
        )
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "m.bin.gz", "-gpu-cache-limit-mb", "-1"]))
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "m.bin.gz", "-gpu-cache-limit-mb", "notanumber"]))
    }

    // MARK: - Absolute ponder-visits ceiling

    /// The pure cap computation `GTPEngine.effectivePonderMaxVisits` used to be an inline
    /// `max(1, 10 * settings.maxVisits)` with no ceiling: at a tournament-scale `-visits 50000`
    /// that is 500,000 NEW nodes a single unattended background ponder could grow the tree by.
    /// Now capped at `GTPEngine.ponderMaxVisitsAbsoluteCap` (20,000) unless the operator passes an
    /// explicit `-ponder-max-visits`, which always wins outright.
    func testEffectivePonderMaxVisitsAppliesAbsoluteCapOnlyWhenNotExplicitlyRequested() {
        // No explicit request (0): 10x heuristic below the cap passes through unchanged.
        XCTAssertEqual(GTPEngine.effectivePonderMaxVisits(requested: 0, searchMaxVisits: 400), 4000)
        // No explicit request: 10x heuristic above the cap is clamped to the absolute ceiling.
        XCTAssertEqual(GTPEngine.effectivePonderMaxVisits(requested: 0, searchMaxVisits: 50000), 20000)
        XCTAssertEqual(
            GTPEngine.effectivePonderMaxVisits(requested: 0, searchMaxVisits: 50000),
            GTPEngine.ponderMaxVisitsAbsoluteCap
        )
        // An explicit, positive -ponder-max-visits always wins, even past the absolute cap --
        // the operator asked for it outright.
        XCTAssertEqual(GTPEngine.effectivePonderMaxVisits(requested: 999_999, searchMaxVisits: 50000), 999_999)
        // A tiny -visits still yields at least 1 (never zero/negative).
        XCTAssertEqual(GTPEngine.effectivePonderMaxVisits(requested: 0, searchMaxVisits: 0), 1)
    }

    // MARK: - (C) GTPEngine ponder lifecycle & gating

    func testPonderOffNeverStartsATask() async {
        let engine = makeEngine(ponderEnabled: false)
        _ = await engine.handle("boardsize 9")
        _ = await engine.handle("komi 7.0")
        _ = await engine.handle("genmove B")
        await engine.startPonderingIfEnabled()
        let pondering = await engine.isPondering()
        XCTAssertFalse(pondering, "with -ponder off, startPonderingIfEnabled must never create a task")
        await engine.stopPondering() // must be a safe no-op
    }

    func testPonderStartsAfterGenmoveAndStopsCleanly() async {
        let engine = makeEngine(ponderEnabled: true, ponderMaxVisits: 64)
        _ = await engine.handle("boardsize 9")
        _ = await engine.handle("komi 7.0")

        // Before any genmove our color is unknown, so pondering must not start even though enabled.
        await engine.startPonderingIfEnabled()
        var pondering = await engine.isPondering()
        XCTAssertFalse(pondering, "ponder must not start before our color is established by a genmove")

        _ = await engine.handle("genmove B") // we are black; the opponent (white) is now to move
        await engine.startPonderingIfEnabled()
        pondering = await engine.isPondering()
        XCTAssertTrue(pondering, "ponder must start on the opponent-to-move root after genmove")

        await engine.stopPondering()
        pondering = await engine.isPondering()
        XCTAssertFalse(pondering, "stopPondering must cancel-and-await and clear the task handle")
    }

    func testPonderDoesNotStartOnOurOwnTurn() async throws {
        let engine = makeEngine(ponderEnabled: true, ponderMaxVisits: 64)
        _ = await engine.handle("boardsize 9")
        _ = await engine.handle("komi 7.0")
        _ = await engine.handle("genmove B") // black played; white to move
        // A legal white reply that the just-finished search actually explored.
        let whiteReplyOptional = await engine.topSearchRootChild()
        let whiteReply = try XCTUnwrap(whiteReplyOptional)
        _ = await engine.handle("play W \(Location.toString(whiteReply.loc, Self.nnLen, Self.nnLen))")

        await engine.startPonderingIfEnabled()
        let pondering = await engine.isPondering()
        XCTAssertFalse(pondering, "ponder must not start when it is our turn (a genmove is imminent)")
    }

    // MARK: - (D) tree reuse harvests the pondered subtree

    func testPonderedSubtreeIsHarvestedWhenOpponentPlaysIt() async throws {
        let ponderCap: Int64 = 300
        let engine = makeEngine(ponderEnabled: true, ponderMaxVisits: ponderCap, maxVisits: 32)
        _ = await engine.handle("boardsize 9")
        _ = await engine.handle("komi 7.0")
        _ = await engine.handle("genmove B") // black played; white (opponent) to move

        let afterGenmove = await engine.searchRootVisits()
        XCTAssertGreaterThan(afterGenmove, 0, "the white-to-move root should carry visits harvested from genmove")

        // Ponder the opponent-to-move root to its natural completion (await the handle, do NOT
        // cancel), so exactly `ponderCap` new visits are added on top of the reused ones.
        await engine.startPonderingIfEnabled()
        let ponderHandle = await engine.ponderTaskHandle()
        let ponderTask = try XCTUnwrap(ponderHandle)
        await ponderTask.value
        let afterPonder = await engine.searchRootVisits()
        XCTAssertEqual(
            afterPonder, afterGenmove + ponderCap,
            "pondering must add exactly its capped budget of NEW visits to the opponent-to-move root"
        )

        // The opponent plays the move the ponder explored most; tree reuse must re-root onto that
        // pondered subtree so its visits become the new root's head start.
        let topChildOptional = await engine.topSearchRootChild()
        let topChild = try XCTUnwrap(topChildOptional)
        XCTAssertGreaterThan(topChild.visits, 0)
        await engine.stopPondering() // mirror the GTP loop: quiesce before the tree-mutating command
        let stillPondering = await engine.isPondering()
        XCTAssertFalse(stillPondering)

        let vertex = Location.toString(topChild.loc, Self.nnLen, Self.nnLen)
        let playResult = await engine.handle("play W \(vertex)")
        XCTAssertEqual(playResult.response, "= \n\n", "the pondered opponent move must be legal and accepted")

        let harvested = await engine.searchRootVisits()
        XCTAssertEqual(
            harvested, topChild.visits,
            "re-rooting must harvest the pondered subtree: the new root's visits equal the pondered child's"
        )
        XCTAssertGreaterThan(harvested, 0, "the harvested head start must be non-empty")
    }

    // MARK: - Fixtures

    private func makeSearch(nnLen: Int = PonderTests.nnLen) -> Search {
        let board = Board(nnLen, nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        return Search(
            evaluator: PonderStubEvaluator(nnLen: nnLen),
            nnXLen: nnLen,
            nnYLen: nnLen,
            precision: .float32,
            settings: SearchSettings(),
            board: board,
            history: history,
            nextPlayer: .black,
            fp32FallbackFactory: nil
        )
    }

    private func makeEngine(
        ponderEnabled: Bool,
        ponderMaxVisits: Int64 = 0,
        maxVisits: Int64 = 32
    ) -> GTPEngine {
        let nnLen = Self.nnLen
        var settings = SearchSettings()
        settings.maxVisits = maxVisits
        settings.numSearchWorkers = 4
        settings.leafBatchSize = 8
        return GTPEngine(
            evaluator: PonderStubEvaluator(nnLen: nnLen),
            nnXLen: nnLen,
            nnYLen: nnLen,
            precision: .float32,
            settings: settings,
            rules: .getTrompTaylorish(),
            nnLength: .fixed(nnLen),
            warmupBucketSizes: [1, 2, 4, 8],
            evaluatorFactory: { _, _ in PonderStubEvaluator(nnLen: nnLen) },
            initialTimeSeconds: nil,
            ponderEnabled: ponderEnabled,
            ponderMaxVisits: ponderMaxVisits
        )
    }
}

/// Deterministic uniform-legal-policy evaluator (value-neutral 0.5 win prob, flat legal policy),
/// mirroring `SearchExplorationTests`/`TreeReuseTests` so pondering is exercised with zero MLX or
/// model dependency.
private actor PonderStubEvaluator: NNEvaluating {
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
