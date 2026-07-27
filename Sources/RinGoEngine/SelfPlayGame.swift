import RinGoCore
import RinGoModel

public struct SelfPlayGameConfiguration: Sendable {
    public var boardSize: Int
    public var visits: Int64
    public var temperature: Double
    public var temperatureMoves: Int
    public var maximumMoves: Int
    public var rules: Rules
    public var searchSettings: SearchSettings

    public init(
        boardSize: Int,
        visits: Int64,
        temperature: Double,
        temperatureMoves: Int,
        maximumMoves: Int = 400,
        rules: Rules,
        searchSettings: SearchSettings
    ) {
        self.boardSize = boardSize
        self.visits = visits
        self.temperature = temperature
        self.temperatureMoves = temperatureMoves
        self.maximumMoves = maximumMoves
        self.rules = rules
        self.searchSettings = searchSettings
    }
}

/// One self-play position's captured search evidence, retained until the game ends so the eventual
/// game outcome can be projected onto each position's side-to-move perspective (WP-9, AlphaZero
/// targets). Features are computed with the SAME `fillRowV7` path `makedata` uses, so a self-play
/// shard is bit-consistent with a distilled/teacherless shard.
public struct SelfPlayPositionRecord: Sendable {
    /// `fillRowV7` spatial features (NHWC), from the side-to-move perspective.
    public let spatial: [Float]
    /// `fillRowV7` global features.
    public let global: [Float]
    /// Dense normalized ROOT VISIT distribution over `nnLen*nnLen+1` (index `nnLen*nnLen` is pass).
    /// This is the PRE-temperature, POST-search AlphaZero policy target: root Dirichlet noise and
    /// the selection temperature influence the SEARCH and which move is PLAYED, but the recorded
    /// distribution is simply the resulting visit counts -- standard AlphaZero practice.
    public let policyTarget: [Float]
    /// Side to move at this position (the perspective of `spatial`/`global`/`policyTarget`).
    public let nextPlayer: Player

    public init(spatial: [Float], global: [Float], policyTarget: [Float], nextPlayer: Player) {
        self.spatial = spatial
        self.global = global
        self.policyTarget = policyTarget
        self.nextPlayer = nextPlayer
    }
}

/// A finished self-play game plus the evidence needed to emit RinGoData v2 training samples.
public struct SelfPlayResult: Sendable {
    public let history: BoardHistory
    /// Per-move captured evidence in play order. Empty unless recording was requested.
    public let records: [SelfPlayPositionRecord]
    /// Final-position area classification (`getAreaNow`) -- the ownership-label source.
    public let finalArea: [Color]
    /// The scored board's x-size, for indexing `finalArea` by `(x, y)`.
    public let boardXSize: Int

    public init(history: BoardHistory, records: [SelfPlayPositionRecord], finalArea: [Color], boardXSize: Int) {
        self.history = history
        self.records = records
        self.finalArea = finalArea
        self.boardXSize = boardXSize
    }
}

/// CPU-testable self-play loop. The production CLI supplies one shared `NNEvaluator`; tests use
/// the existing plain-Swift `NNEvaluating` seam and never construct an MLX value.
public enum SelfPlayGameGenerator {
    public static func play(
        evaluator: any NNEvaluating,
        nnXLen: Int,
        nnYLen: Int,
        precision: KataGoNetwork.Precision,
        configuration: SelfPlayGameConfiguration
    ) async throws -> BoardHistory {
        try await playRecording(
            evaluator: evaluator,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            precision: precision,
            configuration: configuration,
            recordTrainingData: false
        ).history
    }

    /// Plays one self-play game, optionally capturing per-move AlphaZero training evidence (WP-9).
    /// When `recordTrainingData` is false this is exactly the historical `play` loop (no extra
    /// per-move feature fills); when true, each position's `fillRowV7` features and normalized root
    /// visit distribution are recorded before the move is applied, and the final-position area is
    /// captured after scoring. The returned `SelfPlayResult` carries enough to project the eventual
    /// outcome onto every position via `trainingSamples(from:nnLen:)`.
    public static func playRecording(
        evaluator: any NNEvaluating,
        nnXLen: Int,
        nnYLen: Int,
        precision: KataGoNetwork.Precision,
        configuration: SelfPlayGameConfiguration,
        recordTrainingData: Bool
    ) async throws -> SelfPlayResult {
        guard configuration.boardSize >= 2,
              configuration.boardSize <= nnXLen,
              configuration.boardSize <= nnYLen,
              configuration.visits > 0,
              configuration.temperature >= 0,
              configuration.temperature.isFinite,
              configuration.temperatureMoves >= 0,
              configuration.maximumMoves > 0
        else {
            throw SearchError("Invalid self-play game configuration")
        }
        // Recording assumes a square NN grid (the RinGoData shard format is square-`nnLen`-only);
        // the production CLI always passes `nnXLen == nnYLen == boardSize`.
        if recordTrainingData, nnXLen != nnYLen {
            throw SearchError("Self-play training capture requires a square NN grid (nnXLen == nnYLen)")
        }

        let board = Board(configuration.boardSize, configuration.boardSize)
        let history = BoardHistory(board, pla: .black, rules: configuration.rules)
        let search = Search(
            evaluator: evaluator,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            precision: precision,
            settings: configuration.searchSettings,
            board: board,
            history: history,
            nextPlayer: .black,
            fp32FallbackFactory: nil
        )

        var records = [SelfPlayPositionRecord]()
        while !history.isGameFinished, history.moveHistory.count < configuration.maximumMoves {
            let temperature = history.moveHistory.count < configuration.temperatureMoves
                ? configuration.temperature : 0
            let result = try await search.runSearch(
                maxVisits: configuration.visits,
                selectionTemperature: temperature
            )
            let player = history.presumedNextMovePla
            if recordTrainingData {
                // Capture the position BEFORE the move is applied: features + the search's own root
                // visit distribution are the training target for this side-to-move position.
                let features = NNInputs.fillRowV7(
                    board, history, player, MiscNNInputParams(),
                    nnXLen: nnXLen, nnYLen: nnYLen, useNHWC: true
                )
                let policy = visitDistribution(result.visitsByMove, boardXSize: board.xSize, nnLen: nnXLen)
                records.append(SelfPlayPositionRecord(
                    spatial: features.spatial, global: features.global,
                    policyTarget: policy, nextPlayer: player
                ))
            }
            guard history.isLegal(board, result.bestMove, player) else {
                throw SearchError("Search returned an illegal self-play move")
            }
            history.makeBoardMoveAssumeLegal(board, result.bestMove, player)
            try await search.makeMove(result.bestMove)
        }
        // The tournament's move limit and rare no-result cycles are adjudicated from the current
        // legal board so SGFWriter always receives a scored target. This also means a recorded
        // self-play game is ALWAYS scored (never a bare no-result), so its score/ownership are
        // genuine labels -- see `trainingSamples(from:nnLen:)`.
        if !history.isScored { history.endAndScoreGameNow(board) }
        let finalArea = history.getAreaNow(board)
        return SelfPlayResult(history: history, records: records, finalArea: finalArea, boardXSize: board.xSize)
    }

    /// Normalizes a search's root visit counts into a dense `nnLen*nnLen+1` probability distribution
    /// (pass at index `nnLen*nnLen`). `visitsByMove` holds `(loc, visits)` for every EXPANDED root
    /// child, and search only ever expands legal moves, so the result is already legal-masked and
    /// (when any child was visited) sums to 1 over the searched moves. Mirrors
    /// `TeacherLabeler.searchTargets`' policy normalization exactly. Returns an all-zero vector only
    /// in the degenerate no-visits case (cannot happen for `visits >= 1` on a non-terminal root).
    static func visitDistribution(
        _ visitsByMove: [(loc: Loc, visits: Int64)],
        boardXSize: Int,
        nnLen: Int
    ) -> [Float] {
        let area = nnLen * nnLen
        var policy = [Float](repeating: 0, count: area + 1)
        let total = visitsByMove.reduce(Int64(0)) { $0 + $1.visits }
        guard total > 0 else { return policy }
        for (loc, visits) in visitsByMove where visits > 0 {
            let position = NNPos.locToPos(loc, boardXSize, nnLen, nnLen)
            guard position >= 0, position < policy.count else { continue }
            policy[position] += Float(Double(visits) / Double(total))
        }
        return policy
    }

    /// Projects a finished self-play game's outcome onto each recorded position to produce
    /// RinGoData v2 training samples (WP-9). Each sample keeps its captured features and root-visit
    /// policy target, and gains a value/score/ownership target from the SIDE-TO-MOVE perspective of
    /// that position -- the same perspective conventions `TeacherLabeler`/teacherless `makedata` use:
    ///
    /// - value `[win, loss, noResult]`: `[1,0,0]` if this position's mover won, `[0,1,0]` if it
    ///   lost, and `[0.5, 0.5, 0]` for a jigo (an exact komi-integer draw, where `history.winner`
    ///   is nil). A soft jigo target is legal since WP-2a, and is preferable to dropping the
    ///   position. `noResult` stays 0 because the driver always force-scores (see `playRecording`).
    /// - score: `finalWhiteMinusBlackScore` negated for a black mover.
    /// - ownership: the final-position area, reprojected to the mover ({+1 self, -1 opponent, 0}).
    ///
    /// Validity: a self-play game is always genuinely scored (the driver never resigns --
    /// `resignEnabled` is off -- and force-scores move-cap/no-result endings via
    /// `endAndScoreGameNow`), so `scoreValid`/`ownershipValid` are true. They are still keyed off the
    /// history flags defensively: a resignation (should it ever occur) leaves both false, exactly as
    /// teacherless `makedata` handles a resigned SGF (TRAINING_PROCESS_AUDIT.md P0-3).
    public static func trainingSamples(from result: SelfPlayResult, nnLen: Int) -> [TrainingSample] {
        let history = result.history
        let scored = history.isScored && !history.isResignation
        let winner = history.winner // nil only for a jigo (exact draw incl. komi)
        let whiteMinusBlack = history.finalWhiteMinusBlackScore
        return result.records.map { record in
            let player = record.nextPlayer
            let value: [Float] = if let winner {
                winner == player ? [1, 0, 0] : [0, 1, 0]
            } else {
                [0.5, 0.5, 0]
            }
            let score = player == .white ? whiteMinusBlack : -whiteMinusBlack
            let ownership = ownershipTarget(
                area: result.finalArea, boardXSize: result.boardXSize, player: player, nnLen: nnLen
            )
            return TrainingSample(
                spatial: record.spatial,
                global: record.global,
                policyTarget: record.policyTarget,
                valueTarget: value,
                scoreTarget: score,
                scoreValid: scored,
                ownershipTarget: ownership,
                ownershipValid: scored
            )
        }
    }

    /// Reprojects a final-position area classification to the given mover's perspective, matching
    /// `TrainingDataPipeline.ownershipTarget` (teacherless `makedata`): +1 mover-owned, -1
    /// opponent-owned, 0 neutral/dame/seki.
    private static func ownershipTarget(
        area: [Color],
        boardXSize: Int,
        player: Player,
        nnLen: Int
    ) -> [Int8] {
        var result = [Int8](repeating: 0, count: nnLen * nnLen)
        for y in 0 ..< nnLen {
            for x in 0 ..< nnLen {
                let owner = area[Location.getLoc(x, y, boardXSize)]
                result[y * nnLen + x] = owner == player.color ? 1 : owner == player.opponent.color ? -1 : 0
            }
        }
        return result
    }
}
