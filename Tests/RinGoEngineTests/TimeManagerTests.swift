import Foundation
@testable import ringo
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

final class TimeManagerTests: XCTestCase {
    func testDocumentedNineByNineBudgetCurve() {
        let cases: [(timeLeft: Double, moves: Int, expected: Double)] = [
            (600, 0, 540.0 / 58.0),
            (300, 30, 270.0 / 28.0),
            (60, 50, 5.0),
            (10, 58, 0.5),
        ]
        for item in cases {
            var manager = TimeManager(mainTimeSeconds: 600)
            manager.updateTimeLeft(item.timeLeft)
            manager.setMovesPlayed(item.moves)
            XCTAssertEqual(manager.budgetSeconds(boardSize: 9), item.expected, accuracy: 1e-9)
        }
    }

    /// `-time-expected-moves` override: `budgetSeconds(baselineMovesOverride:)` must divide the clock
    /// by the override instead of the hardcoded 9x9 baseline (58), so the 9x9 clock isn't budgeted as
    /// if every game lasts 58 own moves (typical games are ~25). `nil` reproduces the default; the
    /// reserve/hardCap/minimumMoves — and thus the no-timeout envelope (budget <= timeLeft) — are
    /// unchanged at any override.
    func testExpectedMovesOverrideChangesNineByNineBudget() {
        var manager = TimeManager(mainTimeSeconds: 600)
        manager.updateTimeLeft(600)
        manager.setMovesPlayed(0)
        // Default (nil) keeps the documented 540/58 curve.
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9), 540.0 / 58.0, accuracy: 1e-9)
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9, baselineMovesOverride: nil), 540.0 / 58.0, accuracy: 1e-9)
        // A smaller override divides by fewer expected moves => a larger per-move budget.
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9, baselineMovesOverride: 30), 540.0 / 30.0, accuracy: 1e-9)
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9, baselineMovesOverride: 34), 540.0 / 34.0, accuracy: 1e-9)
        // Even an extreme override stays within the no-timeout envelope (floors at minimumMoves=10).
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9, baselineMovesOverride: 1), 540.0 / 10.0, accuracy: 1e-9)
        XCTAssertLessThanOrEqual(manager.budgetSeconds(boardSize: 9, baselineMovesOverride: 1), manager.timeLeftSeconds)
    }

    // `-time-safety-reserve`: a fixed reserve subtracted from the server-reported clock so a
    // wall-clock margin survives the controller's time_settings/time_left OVERRIDE of `-time`. The
    // budget runs on `usable = timeLeft - reserve` and clamps to `usable`, so once the report enters
    // the reserve the budget is EXACTLY 0 (emergency pass), not the 0.15s floor leaking into it.
    func testSafetyReserveSubtractsFromReportedClockAndClampsToZero() {
        var manager = TimeManager(mainTimeSeconds: 600)
        manager.updateTimeLeft(600)
        manager.setMovesPlayed(0)
        let baseline = manager.budgetSeconds(boardSize: 9) // no reserve: 540/58
        manager.safetyReserveSeconds = 90
        // usable = 510; dynamic reserve = 51; available = 459; /58.
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9), 459.0 / 58.0, accuracy: 1e-9)
        XCTAssertLessThan(manager.budgetSeconds(boardSize: 9), baseline)
        // At/below the reserve boundary the budget is exactly 0 (not the 0.15s floor).
        manager.updateTimeLeft(90)
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9), 0)
        manager.updateTimeLeft(80)
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9), 0)
        // The reserve is independent of raw clock state: a server time_settings reconfigure keeps it.
        manager.configure(mainTimeSeconds: 600, byoyomiTimeSeconds: 0, byoyomiStones: 0)
        XCTAssertEqual(manager.safetyReserveSeconds, 90)
        XCTAssertEqual(manager.budgetSeconds(boardSize: 9), 459.0 / 58.0, accuracy: 1e-9)
    }

    func testNineteenByNineteenProfileAndMinimumBudget() {
        var manager = TimeManager(mainTimeSeconds: 600)
        XCTAssertEqual(manager.budgetSeconds(boardSize: 19), 540.0 / 200.0, accuracy: 1e-9)

        manager.updateTimeLeft(1)
        XCTAssertEqual(manager.budgetSeconds(boardSize: 19), 0.15, accuracy: 1e-9)
        manager.updateTimeLeft(0.1)
        XCTAssertEqual(manager.budgetSeconds(boardSize: 19), 0.1, accuracy: 1e-9)
    }

    // WP-21: `updateTimeLeft` on an inactive manager (no prior `time_settings`/`configure`) must
    // lazily arm sudden-death from the reported time, instead of silently discarding it and
    // leaving `isActive` false forever (see `TimeManager.armFromTimeLeft`).
    func testUpdateTimeLeftArmsInactiveManagerAsSuddenDeath() {
        var manager = TimeManager()
        XCTAssertFalse(manager.isActive)
        manager.setMovesPlayed(3)

        manager.updateTimeLeft(45)

        XCTAssertTrue(manager.isActive)
        XCTAssertEqual(manager.mainTimeSeconds, 45)
        XCTAssertEqual(manager.timeLeftSeconds, 45)
        XCTAssertEqual(manager.byoyomiTimeSeconds, 0)
        XCTAssertEqual(manager.byoyomiStones, 0)
        XCTAssertEqual(manager.movesPlayed, 3, "arming from time_left must not clobber moves already recorded")
        let budget = manager.budgetSeconds(boardSize: 9)
        XCTAssertTrue(budget.isFinite)
        XCTAssertGreaterThan(budget, 0)
    }

    func testUpdateTimeLeftAfterLazyArmBehavesAsPlainUpdate() {
        var manager = TimeManager()
        manager.updateTimeLeft(45) // lazily arms
        manager.setMovesPlayed(5)

        manager.updateTimeLeft(30) // already active: must be a plain time-left update, not a re-arm

        XCTAssertEqual(manager.timeLeftSeconds, 30)
        XCTAssertEqual(
            manager.mainTimeSeconds,
            45,
            "mainTimeSeconds is fixed at arm time, exactly like a real time_settings"
        )
        XCTAssertEqual(manager.movesPlayed, 5)
    }

    func testDeadlineStopsAtBatchBoundaryAndFinishesInflightEvaluation() async throws {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9, delay: .milliseconds(80))
        var settings = SearchSettings()
        settings.leafBatchSize = 4
        settings.numSearchWorkers = 4
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: Rules(), encorePhase: 0)
        let search = Search(
            evaluator: evaluator,
            nnXLen: 9,
            nnYLen: 9,
            precision: .float32,
            settings: settings,
            board: board,
            history: history,
            nextPlayer: .black,
            fp32FallbackFactory: nil
        )

        let result = try await search.runSearch(maxVisits: 10000, timeBudget: 0.04)

        let callCount = await evaluator.evaluationCallCount()
        XCTAssertEqual(callCount, 1)
        XCTAssertEqual(result.visitsAchieved, 1)
        XCTAssertGreaterThanOrEqual(result.timeUsed, 0.075)
        XCTAssertLessThan(result.timeUsed, 0.20, "deadline plus one delayed batch should bound runtime")
        XCTAssertTrue(history.isLegal(board, result.bestMove, .black))
    }

    func testGTPFallsBackToVisitLimitWithoutTimeSettings() async {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9)
        var settings = SearchSettings()
        settings.maxVisits = 5
        settings.leafBatchSize = 4
        settings.numSearchWorkers = 4
        let engine = makeEngine(evaluator: evaluator, settings: settings)

        let response = await engine.handle("genmove B")

        XCTAssertTrue(response.response?.hasPrefix("= ") == true)
        let positionCount = await evaluator.evaluatedPositionCount()
        XCTAssertEqual(positionCount, 6, "root bootstrap plus five visit-limited playouts")
        let manager = await engine.currentTimeManager()
        XCTAssertFalse(manager.isActive)
    }

    func testGTPTimeSettingsAndOwnColorTimeLeft() async {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9)
        var settings = SearchSettings()
        settings.maxVisits = 1
        let engine = makeEngine(evaluator: evaluator, settings: settings)

        let settingsResponse = await engine.handle("time_settings 600 30 5")
        XCTAssertEqual(settingsResponse.response, "= \n\n")
        var manager = await engine.currentTimeManager()
        XCTAssertTrue(manager.isActive)
        XCTAssertEqual(manager.mainTimeSeconds, 600)
        XCTAssertEqual(manager.byoyomiTimeSeconds, 30)
        XCTAssertEqual(manager.byoyomiStones, 5)

        let opponentTime = await engine.handle("time_left W 500 0")
        XCTAssertEqual(opponentTime.response, "= \n\n")
        let ownTime = await engine.handle("time_left B 540 0")
        XCTAssertEqual(ownTime.response, "= \n\n")
        _ = await engine.handle("genmove B")
        manager = await engine.currentTimeManager()
        XCTAssertLessThanOrEqual(manager.timeLeftSeconds, 540)
        XCTAssertGreaterThan(manager.timeLeftSeconds, 539)
        XCTAssertEqual(manager.movesPlayed, 1)

        let ignoredOpponentTime = await engine.handle("time_left W 400 0")
        XCTAssertEqual(ignoredOpponentTime.response, "= \n\n")
        let afterOpponentUpdate = await engine.currentTimeManager()
        XCTAssertEqual(afterOpponentUpdate.timeLeftSeconds, manager.timeLeftSeconds, accuracy: 1e-6)
    }

    // WP-21: a controller/referee that only ever sends `time_left` (never `time_settings`) must
    // still arm the clock, or `genmove` gets a `nil` budget and search runs unbounded to the visit
    // cap — a guaranteed timeout loss. Companion to `testGTPFallsBackToVisitLimitWithoutTimeSettings`
    // above, which documents the (still-correct) case where NEITHER command is ever sent.
    func testGTPTimeLeftOnlyArmsManagerWithoutTimeSettings() async {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9)
        var settings = SearchSettings()
        settings.maxVisits = 5
        let engine = makeEngine(evaluator: evaluator, settings: settings)

        let timeLeftResponse = await engine.handle("time_left B 5 0")
        XCTAssertEqual(timeLeftResponse.response, "= \n\n")
        let beforeGenmove = await engine.currentTimeManager()
        XCTAssertFalse(beforeGenmove.isActive, "time_left before engineColor is known is buffered, not yet applied")

        _ = await engine.handle("genmove B")

        let manager = await engine.currentTimeManager()
        XCTAssertTrue(
            manager.isActive,
            "a bare time_left report, with time_settings never sent, must still arm the clock"
        )
        XCTAssertEqual(manager.mainTimeSeconds, 5)
        let budget = manager.budgetSeconds(boardSize: 9)
        XCTAssertTrue(budget.isFinite)
        XCTAssertGreaterThan(budget, 0)
    }

    /// Byo-yomi-style reports (stones > 0) must still arm, treating the reported time as sudden-death
    /// main time only — `stones` is not modeled (mirrors TimeManager.budgetSeconds's documented
    /// "main time only" design), so this is conservative: the engine never budgets more than what a
    /// single reported `time_left` carries.
    func testGTPTimeLeftOnlyWithStonesArmsConservativelyOnMainTimeOnly() async {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9)
        var settings = SearchSettings()
        settings.maxVisits = 5
        let engine = makeEngine(evaluator: evaluator, settings: settings)

        let timeLeftResponse = await engine.handle("time_left B 12 3")
        XCTAssertEqual(timeLeftResponse.response, "= \n\n")

        _ = await engine.handle("genmove B")

        let manager = await engine.currentTimeManager()
        XCTAssertTrue(manager.isActive)
        XCTAssertEqual(
            manager.mainTimeSeconds, 12,
            "stones are not modeled; the reported time is treated as sudden-death main time only"
        )
        XCTAssertEqual(manager.byoyomiStones, 0)
        XCTAssertEqual(manager.byoyomiTimeSeconds, 0)
        let budget = manager.budgetSeconds(boardSize: 9)
        XCTAssertTrue(budget.isFinite)
        XCTAssertGreaterThan(budget, 0)
        XCTAssertLessThanOrEqual(
            budget,
            manager.timeLeftSeconds,
            "conservative: budget must stay within the reported remaining time"
        )
    }

    // Regression: a later genuine time_settings must still fully reconfigure the clock (fresh main
    // time, byoyomi fields, resynced movesPlayed) even after an earlier lazy time_left-only arm.
    func testGTPTimeSettingsStillFullyReconfiguresAfterLazyTimeLeftArm() async {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9)
        var settings = SearchSettings()
        settings.maxVisits = 1
        let engine = makeEngine(evaluator: evaluator, settings: settings)

        _ = await engine.handle("time_left B 5 0")
        _ = await engine.handle("genmove B")
        let lazilyArmed = await engine.currentTimeManager()
        XCTAssertTrue(lazilyArmed.isActive)
        XCTAssertEqual(lazilyArmed.mainTimeSeconds, 5)

        let settingsResponse = await engine.handle("time_settings 600 30 5")
        XCTAssertEqual(settingsResponse.response, "= \n\n")

        let reconfigured = await engine.currentTimeManager()
        XCTAssertTrue(reconfigured.isActive)
        XCTAssertEqual(reconfigured.mainTimeSeconds, 600)
        XCTAssertEqual(reconfigured.timeLeftSeconds, 600)
        XCTAssertEqual(reconfigured.byoyomiTimeSeconds, 30)
        XCTAssertEqual(reconfigured.byoyomiStones, 5)
        XCTAssertEqual(
            reconfigured.movesPlayed, 1,
            "engineColor is already known post-genmove, so time_settings resyncs movesPlayed from the move list"
        )
    }

    /// WP-24 (docs/reviews/CODEX_DEBATE_0712.md R1 finding #3): `Search.runSearch`'s deadline is only
    /// checked between playout batches, so the root-bootstrap NN evaluation (the very first batch)
    /// always runs regardless of `timeBudget` — even a budget of exactly 0 still pays for one full
    /// evaluation. Under sudden death at (near-)zero remaining time this can burn the entire
    /// remainder and lose the game on time, so `genmove` must skip search entirely (zero evaluator
    /// calls) and answer with an emergency pass instead.
    func testGTPEmergencyPassOnZeroTimeLeftSkipsSearchEntirely() async throws {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9)
        var settings = SearchSettings()
        settings.maxVisits = 5
        let engine = makeEngine(evaluator: evaluator, settings: settings)

        _ = await engine.handle("time_settings 600 30 5")
        _ = await engine.handle("time_left B 0 0")

        let response = try await responseText(engine.handle("genmove B"))

        XCTAssertEqual(
            response, "pass",
            "at (near-)zero remaining time, genmove must emergency-pass rather than search"
        )
        let callCount = await evaluator.evaluationCallCount()
        XCTAssertEqual(callCount, 0, "the emergency pass must skip search entirely, including the root-bootstrap eval")
    }

    // Regression: a small-but-playable budget (well above the emergency threshold) must still
    // search normally — the emergency pass must fire only on genuinely near-zero remaining time.
    func testGTPSmallNonzeroTimeLeftStillSearchesNormally() async {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9)
        var settings = SearchSettings()
        settings.maxVisits = 5
        let engine = makeEngine(evaluator: evaluator, settings: settings)

        _ = await engine.handle("time_settings 600 30 5")
        _ = await engine.handle("time_left B 1.0 0")

        let response = await engine.handle("genmove B")

        XCTAssertTrue(response.response?.hasPrefix("= ") == true)
        let positionCount = await evaluator.evaluatedPositionCount()
        XCTAssertEqual(
            positionCount, 6,
            "root bootstrap plus five visit-limited playouts: 1.0s remaining must still search normally"
        )
    }

    /// Companion to `testGTPTimeLeftOnlyArmsManagerWithoutTimeSettings` (WP-21): the emergency pass
    /// must also fire when the clock was armed lazily from a bare `time_left`, never a
    /// `time_settings` — the WP-21 arming path must not bypass the WP-24 clock-safety check.
    func testGTPEmergencyPassFiresWhenArmedLazilyViaTimeLeftOnly() async throws {
        let evaluator = TestPolicyEvaluator(nnXLen: 9, nnYLen: 9)
        var settings = SearchSettings()
        settings.maxVisits = 5
        let engine = makeEngine(evaluator: evaluator, settings: settings)

        let timeLeftResponse = await engine.handle("time_left B 0 0")
        XCTAssertEqual(timeLeftResponse.response, "= \n\n")
        let beforeGenmove = await engine.currentTimeManager()
        XCTAssertFalse(beforeGenmove.isActive, "time_left before engineColor is known is buffered, not yet applied")

        let response = try await responseText(engine.handle("genmove B"))

        XCTAssertEqual(response, "pass")
        let manager = await engine.currentTimeManager()
        XCTAssertTrue(manager.isActive, "a bare time_left report must still lazily arm the clock")
        let callCount = await evaluator.evaluationCallCount()
        XCTAssertEqual(callCount, 0)
    }

    func testGTPTimeCommandsRejectMalformedInput() async {
        let engine = makeEngine(evaluator: TestPolicyEvaluator(nnXLen: 9, nnYLen: 9), settings: SearchSettings())
        let malformed = [
            "time_settings",
            "time_settings 600 nope 0",
            "time_settings -1 0 0",
            "time_settings 600 0 -1",
            "time_settings 600 0 0 extra",
            "time_left",
            "time_left green 10 0",
            "time_left B nope 0",
            "time_left B -1 0",
            "time_left B 10 -1",
            "time_left B 10 0 extra",
        ]
        for command in malformed {
            let response = await engine.handle(command).response
            XCTAssertTrue(response?.hasPrefix("? ") == true, "expected error for: \(command)")
        }
    }

    func testCLIParsesLocalTimeOverrideAndTournamentKomiDefault() throws {
        let defaults = try GTPCommand.parse(["-model", "model.bin.gz"])
        XCTAssertEqual(defaults.komi, 7.0)
        XCTAssertNil(defaults.timeSeconds)
        XCTAssertTrue(defaults.resignEnabled)

        let overridden = try GTPCommand.parse(["-model", "model.bin.gz", "-time", "60"])
        XCTAssertEqual(overridden.timeSeconds, 60)
        let resignDisabled = try GTPCommand.parse(["-model", "model.bin.gz", "-resign-enabled", "false"])
        XCTAssertFalse(resignDisabled.resignEnabled)
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "model.bin.gz", "-time", "0"]))
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "model.bin.gz", "-time", "nan"]))
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "model.bin.gz", "-resign-enabled", "no"]))
    }

    func testResignEnabledDefaultsOnAndCanBeDisabled() async throws {
        func makeLosingBlackEngine(resignEnabled: Bool) -> GTPEngine {
            var settings = SearchSettings()
            settings.maxVisits = 1
            settings.leafBatchSize = 1
            settings.numSearchWorkers = 1
            settings.resignEnabled = resignEnabled
            return makeEngine(
                evaluator: TestPolicyEvaluator(
                    nnXLen: 9,
                    nnYLen: 9,
                    preferredLoc: Board.passLoc,
                    whiteWinProb: 1,
                    whiteLossProb: 0
                ),
                settings: settings
            )
        }

        let enabled = makeLosingBlackEngine(resignEnabled: true)
        let enabledResult = try await playThreeLosingTurns(enabled)
        XCTAssertEqual(enabledResult, "resign")

        let disabled = makeLosingBlackEngine(resignEnabled: false)
        let disabledResult = try await playThreeLosingTurns(disabled)
        XCTAssertEqual(disabledResult, "pass")
    }

    private func makeEngine(
        evaluator: any NNEvaluating,
        settings: SearchSettings
    ) -> GTPEngine {
        GTPEngine(
            evaluator: evaluator,
            nnXLen: 9,
            nnYLen: 9,
            precision: .float32,
            settings: settings,
            rules: Rules(),
            nnLength: .fixed(9),
            warmupBucketSizes: [1],
            evaluatorFactory: { _, _ in TestPolicyEvaluator(nnXLen: 9, nnYLen: 9) }
        )
    }

    private func playThreeLosingTurns(_ engine: GTPEngine) async throws -> String {
        let first = try await responseText(engine.handle("genmove B"))
        XCTAssertEqual(first, "pass")
        let firstReply = await engine.handle("play W A1")
        XCTAssertEqual(firstReply.response, "= \n\n")
        let second = try await responseText(engine.handle("genmove B"))
        XCTAssertEqual(second, "pass")
        let secondReply = await engine.handle("play W B1")
        XCTAssertEqual(secondReply.response, "= \n\n")
        return try await responseText(engine.handle("genmove B"))
    }

    private func responseText(_ result: (response: String?, shouldQuit: Bool)) throws -> String {
        let response = try XCTUnwrap(result.response)
        XCTAssertFalse(result.shouldQuit)
        return String(response.dropFirst(2)).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

actor TestPolicyEvaluator: NNEvaluating {
    private let nnXLen: Int
    private let nnYLen: Int
    private let preferredLoc: Loc?
    private let delay: Duration?
    private let whiteWinProb: Float
    private let whiteLossProb: Float
    private var calls = 0
    private var positions = 0

    init(
        nnXLen: Int,
        nnYLen: Int,
        preferredLoc: Loc? = nil,
        delay: Duration? = nil,
        whiteWinProb: Float = 0.5,
        whiteLossProb: Float = 0.5
    ) {
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.preferredLoc = preferredLoc
        self.delay = delay
        self.whiteWinProb = whiteWinProb
        self.whiteLossProb = whiteLossProb
    }

    func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput] {
        calls += 1
        positions += requests.count
        if let delay {
            try await Task.sleep(for: delay)
        }
        return requests.map(output)
    }

    func evaluationCallCount() -> Int {
        calls
    }

    func evaluatedPositionCount() -> Int {
        positions
    }

    private func output(for request: NNRequest) -> NNOutput {
        let policySize = nnXLen * nnYLen + 1
        var policy = [Float](repeating: 0.001, count: policySize)
        if let preferredLoc {
            let position = NNPos.locToPos(preferredLoc, request.board.xSize, nnXLen, nnYLen)
            if policy.indices.contains(position) {
                policy[position] = 1
            }
        }
        return NNOutput(
            nnHash: Hash128(),
            whiteWinProb: whiteWinProb,
            whiteLossProb: whiteLossProb,
            whiteNoResultProb: 0,
            whiteScoreMean: 0,
            whiteScoreMeanSq: 1,
            whiteLead: 0,
            varTimeLeft: 0,
            shorttermWinlossError: 0,
            shorttermScoreError: 0,
            policyProbs: policy,
            policyOptimismUsed: 0,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            whiteOwnerMap: nil
        )
    }
}
