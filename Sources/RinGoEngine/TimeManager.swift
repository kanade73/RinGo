import Foundation

/// Sudden-death clock allocation for tournament GTP play.
public struct TimeManager: Sendable {
    public static let hardCapFraction = 0.20
    public static let reserveFraction = 0.10
    public static let reserveMinimumSeconds = 10.0
    public static let reserveMaximumFraction = 0.50
    public static let minimumBudgetSeconds = 0.15

    public private(set) var isActive = false
    public private(set) var mainTimeSeconds = 0.0
    public private(set) var timeLeftSeconds = 0.0
    public private(set) var movesPlayed = 0
    public private(set) var byoyomiTimeSeconds = 0.0
    public private(set) var byoyomiStones = 0
    /// Fixed wall-clock reserve (seconds) subtracted from every authoritative remaining-time report
    /// in `budgetSeconds`, so a margin survives the controller's `time_settings`/`time_left` OVERRIDE
    /// of the CLI clock (CGOS/tournament send both, replacing `-time`). `min(reported, cap)` cannot
    /// bound cumulative use — repeated reports restore the capped clock — so the reserve is
    /// subtractive. `0` = off (behavior-identical). Set once (from `-time-safety-reserve`); NOT
    /// touched by `configure`/`updateTimeLeft`, so it persists across server reconfiguration.
    public var safetyReserveSeconds = 0.0

    public init() {}

    public init(mainTimeSeconds: Double, byoyomiTimeSeconds: Double = 0, byoyomiStones: Int = 0) {
        configure(
            mainTimeSeconds: mainTimeSeconds,
            byoyomiTimeSeconds: byoyomiTimeSeconds,
            byoyomiStones: byoyomiStones
        )
    }

    public mutating func configure(
        mainTimeSeconds: Double,
        byoyomiTimeSeconds: Double,
        byoyomiStones: Int
    ) {
        precondition(mainTimeSeconds >= 0 && mainTimeSeconds.isFinite)
        precondition(byoyomiTimeSeconds >= 0 && byoyomiTimeSeconds.isFinite)
        precondition(byoyomiStones >= 0)
        isActive = true
        self.mainTimeSeconds = mainTimeSeconds
        timeLeftSeconds = mainTimeSeconds
        movesPlayed = 0
        self.byoyomiTimeSeconds = byoyomiTimeSeconds
        self.byoyomiStones = byoyomiStones
    }

    public mutating func resetForNewGame() {
        guard isActive else { return }
        timeLeftSeconds = mainTimeSeconds
        movesPlayed = 0
    }

    public mutating func updateTimeLeft(_ seconds: Double) {
        precondition(seconds >= 0 && seconds.isFinite)
        guard isActive else {
            armFromTimeLeft(seconds)
            return
        }
        timeLeftSeconds = seconds
    }

    /// Lazily arms the manager from a GTP `time_left` report when no `time_settings` was ever
    /// received (WP-21). Some tournament controllers/referees only ever send `time_left`, never
    /// `time_settings`; without this, `isActive` stays false forever, `genmove` gets a `nil`
    /// budget, and search runs unbounded to the visit cap — a guaranteed timeout loss.
    ///
    /// Treats the reported remaining time as sudden-death main time: equivalent to a late
    /// `time_settings <seconds> 0 0` immediately followed by this same update. This is
    /// deliberately conservative even when the report carries byo-yomi stones (`time_left <color>
    /// <time> <stones>` with `stones > 0`) — `stones` is not modeled here (mirrors
    /// `budgetSeconds`'s existing "main time only" design above), so a byo-yomi-style report still
    /// arms with only `seconds` of budget to work with, never more.
    ///
    /// `movesPlayed` is intentionally left untouched here (unlike `configure`, which always starts
    /// a fresh clock at move 0): the caller (`GTPEngine`) syncs `movesPlayed` from the move list
    /// before ever reporting time left, and clobbering it back to 0 mid-game would understate
    /// moves already played and over-tighten the very first post-arm budget.
    private mutating func armFromTimeLeft(_ seconds: Double) {
        isActive = true
        mainTimeSeconds = seconds
        timeLeftSeconds = seconds
        byoyomiTimeSeconds = 0
        byoyomiStones = 0
    }

    public mutating func setMovesPlayed(_ count: Int) {
        precondition(count >= 0)
        movesPlayed = count
    }

    public mutating func recordMove(timeUsed: TimeInterval) {
        precondition(timeUsed >= 0 && timeUsed.isFinite)
        timeLeftSeconds = max(0, timeLeftSeconds - timeUsed)
        movesPlayed += 1
    }

    /// Uses only main time even if GTP reports byoyomi: CGF Open is sudden death, so byoyomi is
    /// retained as diagnostic safety information and never spent by this allocator.
    ///
    /// Typical 9x9 curve (seconds), using the move count expected at each remaining-time point:
    ///
    /// | time left | own moves | expected left | reserve | budget |
    /// |----------:|----------:|--------------:|--------:|-------:|
    /// | 600       | 0         | 58            | 60      | 9.31   |
    /// | 300       | 30        | 28            | 30      | 9.64   |
    /// | 60        | 50        | 10            | 10      | 5.00   |
    /// | 10        | 58        | 10            | 5       | 0.50   |
    public func budgetSeconds(boardSize: Int, baselineMovesOverride: Int? = nil) -> TimeInterval {
        guard isActive, timeLeftSeconds > 0 else { return 0 }
        // Controller-resistant safety reserve: never budget against the last `safetyReserveSeconds`
        // of the server-reported clock. Everything below runs on `usable`, and the final clamp is to
        // `usable` (NOT `timeLeftSeconds`), so once the report falls into the reserve `usable <= 0`
        // returns exactly 0 — the caller's emergency-pass fires instead of the 0.15s floor leaking a
        // budget into the reserve. Sudden-death/main time only (this port budgets main time; CGOS
        // byo-yomi = `0 0`). `0` reserve reproduces the prior `timeLeftSeconds`-based behavior.
        let usable = max(0, timeLeftSeconds - safetyReserveSeconds)
        guard usable > 0 else { return 0 }
        let profile = boardSize <= 9
            ? (baselineMoves: 58, minimumMoves: 10)
            : (baselineMoves: 200, minimumMoves: 15)
        // `-time-expected-moves` overrides the hardcoded expected own-move count (9x9 default 58,
        // vs ~25 typical => ~40-60% of the clock left unused). `nil` keeps the per-size profile;
        // minimumMoves/reserve/hardCap are unchanged, preserving the no-timeout structure.
        let baselineMoves = baselineMovesOverride ?? profile.baselineMoves
        let expectedRemainingMoves = max(profile.minimumMoves, baselineMoves - movesPlayed)
        let reserve = min(
            max(Self.reserveFraction * usable, Self.reserveMinimumSeconds),
            Self.reserveMaximumFraction * usable
        )
        let hardCap = Self.hardCapFraction * usable
        let available = max(0, usable - reserve)
        let divided = available / Double(max(expectedRemainingMoves, profile.minimumMoves))
        let allocated = min(hardCap, divided)
        return min(usable, max(Self.minimumBudgetSeconds, allocated))
    }
}
