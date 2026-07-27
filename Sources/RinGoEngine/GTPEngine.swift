import Foundation
import RinGoCore
import RinGoModel

public struct GTPError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

public enum GTPNNLength: Equatable, Sendable {
    case auto
    case fixed(Int)

    public init(argument: String) throws {
        if argument == "auto" {
            self = .auto
        } else if let length = Int(argument), length > 0, length <= Board.maxLen {
            self = .fixed(length)
        } else {
            throw GTPError("Invalid -nnlen '\(argument)'; expected auto or an integer from 1 to \(Board.maxLen)")
        }
    }
}

public typealias GTPEvaluatorFactory = @Sendable (_ nnXLen: Int, _ nnYLen: Int) throws -> any NNEvaluating

/// GTP (Go Text Protocol) command loop, factored as a pure `String -> String?` transducer (no
/// stdio) so tests can drive a canned command script directly (`Tests/.../GTPEngineTests.swift`)
/// and the CLI (`Sources/ringo/GTPCommand.swift`) just wires stdin/stdout to `handle`.
///
/// Supports exactly the commands the task spec lists: `protocol_version`, `name`, `version`,
/// `list_commands`, `known_command`, `boardsize` (square only), `clear_board`, `komi`, `play`,
/// `genmove`, `undo`, `time_settings`, `time_left`, `showboard`, `final_score`, `quit`. Anything
/// else gets GTP's standard "unknown command" error.
///
/// An `actor` even though GTP is inherently a one-command-at-a-time protocol: this is just for
/// consistency with the rest of the module (`NNEvaluator`, `Search`) and cheap safety, not because
/// concurrent `handle` calls are an expected use case.
public actor GTPEngine {
    private static let supportedCommands = [
        "protocol_version", "name", "version", "list_commands", "known_command",
        "boardsize", "clear_board", "komi", "play", "genmove", "kata-genmove_analyze",
        "time_settings", "time_left",
        "undo", "showboard", "final_score", "quit",
    ]

    private var evaluator: any NNEvaluating
    private let evaluatorFactory: GTPEvaluatorFactory
    private let modelDesc: ModelDesc?
    private var nnXLen: Int
    private var nnYLen: Int
    private let nnLength: GTPNNLength
    private let precision: KataGoNetwork.Precision
    private var settings: SearchSettings
    private var warmupBucketSizes: [Int]

    private var boardSize: Int
    private var rules: Rules
    private var board: Board
    private var history: BoardHistory
    private var moveList: [Move] = []
    private var search: Search
    private var timeManager: TimeManager
    /// `-time-expected-moves`: overrides `TimeManager.budgetSeconds`'s hardcoded expected own-move
    /// count (9x9 default 58). nil keeps the per-board-size default. Set once at init; stateless in
    /// `TimeManager` so `time_settings`/`time_left` reconfiguration never drops it.
    private let timeExpectedMovesOverride: Int?
    private var engineColor: Player?
    private var pendingBlackTimeLeft: Double?
    private var pendingWhiteTimeLeft: Double?

    /// Optional opening book (WP-12). When present, `genmove` probes it by the current situational
    /// hash BEFORE any search; a hit plays the book move immediately via the normal make-move path
    /// (so tree reuse and pondering stay coherent) and skips search entirely. `nil` (no `-book`
    /// flag) means every path is byte-identical to the pre-book engine.
    private let openingBook: OpeningBook?

    /// Pondering ("think on the opponent's time", WP-10). When enabled, after the engine answers a
    /// `genmove` — leaving the internal position opponent-to-move — `startPonderingIfEnabled()`
    /// launches a background search on that root that the GTP loop cancels-and-awaits the instant
    /// the next command arrives (`stopPondering()`). Tree reuse (`Search.makeMove`) then harvests
    /// the accumulated subtree when the opponent's actual move is played. Defaults OFF: with
    /// `ponderEnabled == false` no task is ever created and every path is bit-identical to the
    /// pre-ponder engine. `ponderMaxVisits` caps the NEW visits a single ponder may add, bounding
    /// the tree's memory growth while the opponent thinks (the clock — `TimeManager` — is never
    /// touched by ponder time).
    private let ponderEnabled: Bool
    private let ponderMaxVisits: Int64
    private var ponderTask: Task<Void, Never>?

    /// Default ponder cap (O-2/I-2, CODEBASE_REVIEW_0711.md): an unset `-ponder-max-visits`
    /// (`requested <= 0`) used to fall back to `10x -visits` uncapped, so a tournament-scale
    /// `-visits 50000` config let a single ponder grow a 500k-node background tree — an
    /// unbounded-in-practice memory grower for a search actor that runs for the ENTIRE duration
    /// of the opponent's clock. `20_000` is an absolute ceiling on top of the `10x` heuristic
    /// (whichever is smaller): generous for any realistic tournament `-visits` setting (the low
    /// hundreds to low thousands) while bounding worst-case ponder-tree memory even at very high
    /// `-visits`. An explicit `-ponder-max-visits` always wins outright (unbounded by this cap —
    /// the operator asked for it).
    public static let ponderMaxVisitsAbsoluteCap: Int64 = 20000

    public static func effectivePonderMaxVisits(requested: Int64, searchMaxVisits: Int64) -> Int64 {
        guard requested <= 0 else { return requested }
        return max(1, min(10 * searchMaxVisits, ponderMaxVisitsAbsoluteCap))
    }

    /// Resign tracking (task spec: "resign if root winrate < 0.05 for 3 consecutive own turns").
    private var lowWinrateStreak = 0
    private var lastGenmoveColor: Player?
    private let resignThreshold = 0.05
    private let resignConsecutiveTurns = 3

    /// The `SearchResult` of the most recent `generateMove` that actually searched (nil after a
    /// book-hit or emergency-pass shortcut, which produce no search stats). Set by both `genmove`
    /// and `kata-genmove_analyze` since they share `generateMove`; read by the analysis formatter
    /// and by `GTPAnalyzeTests`'s genmove-vs-analyze visit-distribution equivalence check.
    private var lastGenmoveResult: SearchResult?

    /// WP-24 (docs/reviews/CODEX_DEBATE_0712.md R1 finding #3): `Search.runSearch`'s deadline is
    /// only checked between playout batches, so the root-bootstrap NN evaluation (the very first
    /// batch) always runs to completion regardless of `timeBudget` — even a budget of exactly 0
    /// still pays for one full evaluation (tens of ms, more under a cold cache/slow batch). Under
    /// sudden death with (near-)zero remaining time (e.g. a referee sending `time_left B 0 0`),
    /// that unconditional spend can burn the entire remainder and lose the game on time.
    ///
    /// `TimeManager.minimumBudgetSeconds` (0.15s) does NOT prevent this: `budgetSeconds` clamps its
    /// return value to `min(timeLeftSeconds, max(minimumBudgetSeconds, allocated))`, so whenever
    /// `timeLeftSeconds` itself is below the floor, the outer `min` picks `timeLeftSeconds` — the
    /// floor never fires and the reported near-zero time passes straight through as the budget
    /// (confirmed via `TimeManager.swift`: `budgetSeconds(boardSize:)` returns exactly 0 when
    /// `timeLeftSeconds == 0`, and returns `timeLeftSeconds` unfloored for any positive value below
    /// 0.15). So the fix point below is intentionally the *budget*, not `timeLeftSeconds` directly.
    ///
    /// Below this threshold, `genmove` skips search entirely and answers with an unconditional
    /// emergency pass instead: a legal, zero-cost move that can never itself lose on time (unlike a
    /// search-selected move, which still requires at least one NN evaluation to compute).
    /// `conservativePass` semantics (see `Search`) do not apply here — this is a clock emergency
    /// escape, not a scoring-safety heuristic.
    private static let emergencyPassBudgetSeconds: TimeInterval = 0.005

    public init(
        evaluator: any NNEvaluating,
        modelDesc: ModelDesc,
        nnXLen: Int,
        nnYLen: Int,
        precision: KataGoNetwork.Precision,
        settings: SearchSettings,
        rules: Rules,
        nnLength: GTPNNLength? = nil,
        maxBatchSize: Int? = nil,
        evaluatorFactory: GTPEvaluatorFactory? = nil,
        initialTimeSeconds: Double? = nil,
        ponderEnabled: Bool = false,
        ponderMaxVisits: Int64 = 0,
        openingBook: OpeningBook? = nil,
        timeExpectedMoves: Int? = nil,
        timeSafetyReserve: Double = 0
    ) {
        self.evaluator = evaluator
        self.modelDesc = modelDesc
        self.openingBook = openingBook
        timeExpectedMovesOverride = timeExpectedMoves
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.precision = precision
        self.settings = settings
        self.ponderEnabled = ponderEnabled
        self.ponderMaxVisits = Self.effectivePonderMaxVisits(
            requested: ponderMaxVisits,
            searchMaxVisits: settings.maxVisits
        )
        var tournamentRules = rules
        tournamentRules.koRule = .positional
        self.rules = tournamentRules
        timeManager = initialTimeSeconds.map { TimeManager(mainTimeSeconds: $0) } ?? TimeManager()
        timeManager.safetyReserveSeconds = timeSafetyReserve
        engineColor = nil
        pendingBlackTimeLeft = nil
        pendingWhiteTimeLeft = nil
        let configuredBatchSize = maxBatchSize ?? settings.leafBatchSize
        warmupBucketSizes = Self.bucketSizes(upTo: configuredBatchSize)
        self.nnLength = nnLength ?? .fixed(min(nnXLen, nnYLen))
        // Evaluators built on board-size changes must carry the SAME symmetry mode as the search;
        // `Search` reads it from `settings`, so the default factory does too (captured by value).
        let factorySymmetry = settings.symmetryMode
        self.evaluatorFactory = evaluatorFactory ?? { newXLen, newYLen in
            try NNEvaluator(
                desc: modelDesc,
                nnXLen: newXLen,
                nnYLen: newYLen,
                precision: precision,
                maxBatchSize: configuredBatchSize,
                symmetry: factorySymmetry
            )
        }
        boardSize = min(Board.defaultLen, nnXLen, nnYLen)
        let initialBoard = Board(boardSize, boardSize)
        let initialHistory = BoardHistory(initialBoard, pla: .black, rules: tournamentRules, encorePhase: 0)
        board = initialBoard
        history = initialHistory
        search = Search(
            evaluator: evaluator,
            modelDesc: modelDesc,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            precision: precision,
            settings: settings,
            board: initialBoard,
            history: initialHistory,
            nextPlayer: .black
        )
    }

    init(
        evaluator: any NNEvaluating,
        nnXLen: Int,
        nnYLen: Int,
        precision: KataGoNetwork.Precision,
        settings: SearchSettings,
        rules: Rules,
        nnLength: GTPNNLength,
        warmupBucketSizes: [Int],
        evaluatorFactory: @escaping GTPEvaluatorFactory,
        initialTimeSeconds: Double? = nil,
        ponderEnabled: Bool = false,
        ponderMaxVisits: Int64 = 0,
        openingBook: OpeningBook? = nil,
        timeExpectedMoves: Int? = nil,
        timeSafetyReserve: Double = 0
    ) {
        self.evaluator = evaluator
        self.evaluatorFactory = evaluatorFactory
        self.openingBook = openingBook
        timeExpectedMovesOverride = timeExpectedMoves
        modelDesc = nil
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.nnLength = nnLength
        self.precision = precision
        self.settings = settings
        self.ponderEnabled = ponderEnabled
        self.ponderMaxVisits = Self.effectivePonderMaxVisits(
            requested: ponderMaxVisits,
            searchMaxVisits: settings.maxVisits
        )
        var tournamentRules = rules
        tournamentRules.koRule = .positional
        self.rules = tournamentRules
        timeManager = initialTimeSeconds.map { TimeManager(mainTimeSeconds: $0) } ?? TimeManager()
        timeManager.safetyReserveSeconds = timeSafetyReserve
        engineColor = nil
        pendingBlackTimeLeft = nil
        pendingWhiteTimeLeft = nil
        self.warmupBucketSizes = warmupBucketSizes
        boardSize = min(Board.defaultLen, nnXLen, nnYLen)
        let initialBoard = Board(boardSize, boardSize)
        let initialHistory = BoardHistory(initialBoard, pla: .black, rules: tournamentRules, encorePhase: 0)
        board = initialBoard
        history = initialHistory
        search = Search(
            evaluator: evaluator,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            precision: precision,
            settings: settings,
            board: initialBoard,
            history: initialHistory,
            nextPlayer: .black,
            fp32FallbackFactory: nil
        )
    }

    // コードリーディング用メモ:
    // これから使うグラフの計算を温めている。ネットに理解が深まってきたら戻ってくる。
    // ここは書き換えたくないので、publicとしたところからprivateを呼び出すようにしている。

    /// Pre-warms every `NNEvaluator` compile bucket (design.md MLX discipline posint 3: fp32
    /// `compile()` warmup can be ~80x slower than fp16's, so long-lived processes should pay it
    /// once at startup, not on the first real `genmove`) with dummy all-empty-board batches.
    public func preWarm(bucketSizes: [Int]) async throws {
        warmupBucketSizes = bucketSizes
        try await preWarm(evaluator, boardSize: boardSize)
    }

    private func preWarm(_ evaluator: any NNEvaluating, boardSize: Int) async throws {
        let dummyBoard = Board(boardSize, boardSize)
        let dummyHistory = BoardHistory(dummyBoard, pla: .black, rules: rules, encorePhase: 0)
        for size in warmupBucketSizes {
            let requests = (0 ..< size).map { _ in
                NNRequest(board: dummyBoard, history: dummyHistory, nextPlayer: .black)
            }
            try await evaluator.preWarm(requests)
        }
    }

    // MARK: - Pondering (think on the opponent's time, WP-10)

    /// Launches a background ponder search on the current root if — and only if — pondering is
    /// enabled, no ponder is already running, our color is known, the game is live, and it is the
    /// *opponent's* turn (i.e. we just answered a `genmove`, so the root is opponent-to-move). The
    /// GTP loop calls this after printing each response; the returned search runs on the `Search`
    /// actor while the loop blocks on `readLine`, and is torn down by `stopPondering()` before the
    /// next command is processed. Cheap and idempotent: the guards make it a no-op after `play`
    /// (our turn — the next `genmove` will search anyway), after `clear_board`/`boardsize` (color
    /// reset to `nil`), on a finished game, and always when `ponderEnabled == false`.
    ///
    /// Actor-safety: the ponder `Task` captures only the `Search` actor (never mutating GTPEngine
    /// state), so it runs entirely on `Search`'s executor and cannot race the GTP loop's own
    /// `Search` calls — those are serialized by the actor, and `stopPondering()` awaits this task
    /// to completion before any tree-mutating command (`play`/`genmove`/`makeMove`) runs, so the
    /// tree is always quiescent and consistent at re-root time. The ponder search itself never
    /// touches `TimeManager`, so ponder time is never charged to our clock.
    public func startPonderingIfEnabled() {
        guard ponderEnabled,
              ponderTask == nil,
              let engineColor,
              !history.isGameFinished,
              history.presumedNextMovePla == engineColor.opponent
        else { return }
        let search = search
        let cap = ponderMaxVisits
        ponderTask = Task { [search] in
            let result = try? await search.runSearch(maxVisits: cap)
            if let result {
                FileHandle.standardError.write(Data(
                    "Ponder: +\(result.visitsAchieved) visits (root=\(result.rootVisits))\n".utf8
                ))
            }
        }
    }

    /// Cancels any in-flight ponder search and awaits its completion, leaving the `Search` actor
    /// quiescent. Must be called before every command so a tree-mutating command never interleaves
    /// with a live ponder search. Awaiting `task.value` (not merely `cancel()`) is load-bearing:
    /// cancellation only *requests* the ponder loop to stop at its next between-batches check, and
    /// we must not re-root or start a new search until that loop has actually returned. No-op when
    /// nothing is pondering.
    public func stopPondering() async {
        guard let task = ponderTask else { return }
        ponderTask = nil
        task.cancel()
        await task.value
    }

    /// Handles one GTP input line, returning the formatted response text (already terminated per
    /// GTP framing: trailing blank line) to print, or `nil` for a blank/comment-only line that
    /// produces no response at all. `shouldQuit` tells the caller's stdio loop to stop after
    /// printing.
    public func handle(_ line: String) async -> (response: String?, shouldQuit: Bool) {
        guard let command = Self.parse(line) else { return (nil, false) }
        do {
            let (text, shouldQuit) = try await execute(command)
            return (Self.format(id: command.id, marker: "=", text: text), shouldQuit)
        } catch {
            return (Self.format(id: command.id, marker: "?", text: "\(error)"), false)
        }
    }

    // MARK: - Command dispatch

    private struct ParsedCommand {
        var id: String?
        var name: String
        var args: [String]
    }

    private func execute(_ command: ParsedCommand) async throws -> (text: String, shouldQuit: Bool) {
        switch command.name {
        case "protocol_version":
            return ("2", false)
        case "name":
            return ("ringo", false)
        case "version":
            return ("0.1", false)
        case "list_commands":
            return (Self.supportedCommands.joined(separator: "\n"), false)
        case "known_command":
            guard let value = command.args.first else {
                throw GTPError("known_command requires a command name argument")
            }
            return (Self.supportedCommands.contains(value) ? "true" : "false", false)
        case "boardsize":
            guard let value = command.args.first, let size = Int(value) else {
                throw GTPError("boardsize requires an integer argument")
            }
            try await setBoardSize(size)
            return ("", false)
        case "clear_board":
            await clearBoard()
            return ("", false)
        case "komi":
            guard let value = command.args.first, let komi = Float(value), komi.isFinite else {
                throw GTPError("komi requires a finite numeric argument")
            }
            await setKomi(komi)
            return ("", false)
        case "play":
            guard command.args.count == 2 else {
                throw GTPError("play requires a color and a vertex")
            }
            try await play(colorString: command.args[0], vertexString: command.args[1])
            return ("", false)
        case "genmove":
            guard let colorString = command.args.first else {
                throw GTPError("genmove requires a color")
            }
            return try await (genmove(colorString: colorString), false)
        case "kata-genmove_analyze":
            guard let colorString = command.args.first else {
                throw GTPError("kata-genmove_analyze requires a color")
            }
            return try await (
                genmoveAnalyze(colorString: colorString, args: Array(command.args.dropFirst())),
                false
            )
        case "time_settings":
            try setTimeSettings(command.args)
            return ("", false)
        case "time_left":
            try setTimeLeft(command.args)
            return ("", false)
        case "undo":
            try await undo()
            return ("", false)
        case "showboard":
            return (history.printBasicInfo(board), false)
        case "final_score":
            return (finalScore(), false)
        case "quit":
            return ("", true)
        default:
            throw GTPError("unknown command")
        }
    }

    // MARK: - Board setup

    private func setBoardSize(_ size: Int) async throws {
        guard size >= 2, size <= Board.maxLen else {
            throw GTPError("unacceptable size")
        }
        if case let .fixed(length) = nnLength, size > length {
            throw GTPError("boardsize \(size) exceeds fixed nnlen \(length)")
        }
        if nnLength == .auto, size != nnXLen || size != nnYLen {
            let newEvaluator = try evaluatorFactory(size, size)
            try await preWarm(newEvaluator, boardSize: size)
            let newBoard = Board(size, size)
            let newHistory = BoardHistory(newBoard, pla: .black, rules: rules, encorePhase: 0)
            let newSearch = if let modelDesc {
                Search(
                    evaluator: newEvaluator,
                    modelDesc: modelDesc,
                    nnXLen: size,
                    nnYLen: size,
                    precision: precision,
                    settings: settings,
                    board: newBoard,
                    history: newHistory,
                    nextPlayer: .black
                )
            } else {
                Search(
                    evaluator: newEvaluator,
                    nnXLen: size,
                    nnYLen: size,
                    precision: precision,
                    settings: settings,
                    board: newBoard,
                    history: newHistory,
                    nextPlayer: .black,
                    fp32FallbackFactory: nil
                )
            }
            evaluator = newEvaluator
            nnXLen = size
            nnYLen = size
            boardSize = size
            board = newBoard
            history = newHistory
            search = newSearch
            moveList = []
            lowWinrateStreak = 0
            lastGenmoveColor = nil
            engineColor = nil
            pendingBlackTimeLeft = nil
            pendingWhiteTimeLeft = nil
            timeManager.resetForNewGame()
            return
        }
        boardSize = size
        await clearBoard()
    }

    private static func bucketSizes(upTo maxBatchSize: Int) -> [Int] {
        var sizes = [Int]()
        var size = 1
        while size <= maxBatchSize {
            sizes.append(size)
            size <<= 1
        }
        return sizes
    }

    private func clearBoard() async {
        moveList = []
        lowWinrateStreak = 0
        lastGenmoveColor = nil
        engineColor = nil
        pendingBlackTimeLeft = nil
        pendingWhiteTimeLeft = nil
        timeManager.resetForNewGame()
        let newBoard = Board(boardSize, boardSize)
        let newHistory = BoardHistory(newBoard, pla: .black, rules: rules, encorePhase: 0)
        board = newBoard
        history = newHistory
        await search.setPosition(board: newBoard, history: newHistory, nextPlayer: .black)
    }

    private func setKomi(_ komi: Float) async {
        rules.komi = komi
        history.setKomi(komi)
        // The cached tree's NN evals baked in the old komi (a direct NN input feature), so it
        // cannot be reused across a komi change.
        await search.setPosition(board: board, history: history, nextPlayer: history.presumedNextMovePla)
    }

    // MARK: - Play / genmove / undo

    private func play(colorString: String, vertexString: String) async throws {
        let color = try Self.parseColor(colorString)
        guard let loc = Location.ofString(vertexString, board.xSize, board.ySize) else {
            throw GTPError("invalid vertex")
        }
        guard color == history.presumedNextMovePla else {
            throw GTPError("move is out of turn")
        }
        guard history.isLegal(board, loc, color) else {
            throw GTPError("illegal move")
        }
        history.makeBoardMoveAssumeLegal(board, loc, color)
        moveList.append(Move(loc: loc, pla: color))
        try await search.makeMove(loc)
    }

    /// The move a `generateMove` produced, plus the `SearchResult` it came from (nil for the
    /// book-hit and emergency-pass shortcuts, which do not search). Shared by `genmove` (which uses
    /// only `.move`) and `kata-genmove_analyze` (which also formats `.result`).
    private struct GenmoveOutcome {
        var move: String
        var result: SearchResult?
    }

    private func genmove(colorString: String) async throws -> String {
        try await generateMove(colorString: colorString).move
    }

    /// `kata-genmove_analyze COLOR [key value ...]`: runs the EXACT same move-generation path as
    /// `genmove` (via `generateMove`), then answers with KataGo's analysis format — one `info` line
    /// covering the searched root children, an optional `ownership` block (when `ownership true` was
    /// requested), and a final `play <move>` line. All board/tree/clock/ponder state effects are
    /// identical to `genmove` because the underlying routine is literally the same call. Unknown
    /// key-value arguments are tolerated and ignored; only `ownership true|false` is honored.
    private func genmoveAnalyze(colorString: String, args: [String]) async throws -> String {
        let includeOwnership = Self.parseOwnershipFlag(args)
        let color = try Self.parseColor(colorString)
        let outcome = try await generateMove(colorString: colorString)

        var lines: [String] = []
        if let result = outcome.result {
            let analysis = GTPAnalysisFormatter.analysisLine(
                result: result,
                mover: color,
                boardXSize: board.xSize,
                boardYSize: board.ySize,
                includeOwnership: includeOwnership
            )
            if !analysis.isEmpty {
                lines.append(analysis)
            }
        }
        lines.append("play \(outcome.move)")
        return lines.joined(separator: "\n")
    }

    /// Scans `key value` argument pairs for `ownership true|false` (default false). Any other token
    /// is tolerated and skipped, per the reference protocol's "be robust to unknown fields".
    private static func parseOwnershipFlag(_ args: [String]) -> Bool {
        var result = false
        var index = 0
        while index < args.count {
            if args[index].lowercased() == "ownership", index + 1 < args.count {
                let value = args[index + 1].lowercased()
                result = value == "true" || value == "1"
                index += 2
            } else {
                index += 1
            }
        }
        return result
    }

    private func generateMove(colorString: String) async throws -> GenmoveOutcome {
        let color = try Self.parseColor(colorString)
        guard color == history.presumedNextMovePla else {
            throw GTPError("genmove color does not match whose turn it is")
        }
        if lastGenmoveColor != color {
            lowWinrateStreak = 0
        }
        lastGenmoveColor = color

        engineColor = color
        lastGenmoveResult = nil

        // WP-12: opening-book probe BEFORE search. On a hit, play the book move via the same
        // make-move path a searched move takes (tree reuse + pondering stay coherent) and answer
        // immediately. Belt-and-braces: re-verify legality in our own rules state; on any mismatch
        // log a warning and fall through to search unchanged.
        if let book = openingBook, let entry = book.lookup(board: board, nextPlayer: color) {
            let loc = entry.loc
            if history.isLegal(board, loc, color) {
                history.makeBoardMoveAssumeLegal(board, loc, color)
                moveList.append(Move(loc: loc, pla: color))
                try await search.makeMove(loc)
                lowWinrateStreak = 0
                let move = loc == Board.passLoc ? "pass" : Location.toString(loc, board.xSize, board.ySize)
                return GenmoveOutcome(move: move, result: nil)
            }
            FileHandle.standardError.write(Data((
                "Warning: opening-book move \(Location.toString(loc, board.xSize, board.ySize)) "
                    + "is illegal in the current position; falling through to search\n"
            ).utf8))
        }

        timeManager.setMovesPlayed(moveList.count(where: { $0.pla == color }))
        if let reported = color == .black ? pendingBlackTimeLeft : pendingWhiteTimeLeft {
            timeManager.updateTimeLeft(reported)
            if color == .black {
                pendingBlackTimeLeft = nil
            } else {
                pendingWhiteTimeLeft = nil
            }
        }

        let budget = timeManager.isActive
            ? timeManager.budgetSeconds(boardSize: boardSize, baselineMovesOverride: timeExpectedMovesOverride)
            : nil
        // WP-24: near-zero time budget — skip search (which would still pay for a root-bootstrap
        // NN evaluation before its first deadline check) and pass immediately to protect the clock.
        // See `emergencyPassBudgetSeconds`'s doc comment for why this checks the budget rather than
        // `timeManager.timeLeftSeconds` directly.
        if let budget, budget <= Self.emergencyPassBudgetSeconds {
            FileHandle.standardError.write(Data((
                "Warning: time budget \(budget)s at or below emergency-pass threshold "
                    + "(\(Self.emergencyPassBudgetSeconds)s); skipping search and passing to protect the clock\n"
            ).utf8))
            history.makeBoardMoveAssumeLegal(board, Board.passLoc, color)
            moveList.append(Move(loc: Board.passLoc, pla: color))
            try await search.makeMove(Board.passLoc)
            timeManager.recordMove(timeUsed: 0)
            return GenmoveOutcome(move: "pass", result: nil)
        }
        // WP-11: LCB final-move selection applies to GTP `genmove` only (self-play/teacher never opt
        // in). It changes only the chosen move; resign still reads `result.whiteWinProb` (the root
        // winrate) below, independent of selection.
        let result = try await search.runSearch(
            maxVisits: settings.maxVisits,
            timeBudget: budget,
            useLcbForSelection: settings.lcbEnabled
        )
        lastGenmoveResult = result
        if timeManager.isActive {
            timeManager.recordMove(timeUsed: result.timeUsed)
        }
        let moverWinProb = color == .white ? result.whiteWinProb : 1.0 - result.whiteWinProb
        lowWinrateStreak = moverWinProb < resignThreshold ? lowWinrateStreak + 1 : 0
        if settings.resignEnabled, lowWinrateStreak >= resignConsecutiveTurns {
            return GenmoveOutcome(move: "resign", result: result)
        }

        var loc = result.bestMove
        if !history.isLegal(board, loc, color) {
            FileHandle.standardError.write(Data(
                "Warning: search selected an illegal or repeating move; playing pass instead\n".utf8
            ))
            loc = Board.passLoc
        }
        history.makeBoardMoveAssumeLegal(board, loc, color)
        moveList.append(Move(loc: loc, pla: color))
        try await search.makeMove(loc)
        let move = loc == Board.passLoc ? "pass" : Location.toString(loc, board.xSize, board.ySize)
        return GenmoveOutcome(move: move, result: result)
    }

    private func undo() async throws {
        guard !moveList.isEmpty else {
            throw GTPError("cannot undo")
        }
        moveList.removeLast()
        let newBoard = Board(boardSize, boardSize)
        let newHistory = BoardHistory(newBoard, pla: .black, rules: rules, encorePhase: 0)
        for move in moveList {
            guard newHistory.makeBoardMoveTolerant(newBoard, move.loc, move.pla) else {
                throw GTPError("internal error replaying move history for undo")
            }
        }
        board = newBoard
        history = newHistory
        lowWinrateStreak = 0
        lastGenmoveColor = nil
        if let engineColor {
            timeManager.setMovesPlayed(moveList.count(where: { $0.pla == engineColor }))
        }
        await search.setPosition(board: newBoard, history: newHistory, nextPlayer: newHistory.presumedNextMovePla)
    }

    // MARK: - Time controls

    private func setTimeSettings(_ args: [String]) throws {
        guard args.count == 3,
              let mainTime = Double(args[0]), mainTime.isFinite, mainTime >= 0,
              let byoyomiTime = Double(args[1]), byoyomiTime.isFinite, byoyomiTime >= 0,
              let byoyomiStones = Int(args[2]), byoyomiStones >= 0
        else {
            throw GTPError("time_settings requires nonnegative <main> <byo_time> <byo_stones>")
        }
        timeManager.configure(
            mainTimeSeconds: mainTime,
            byoyomiTimeSeconds: byoyomiTime,
            byoyomiStones: byoyomiStones
        )
        if let engineColor {
            timeManager.setMovesPlayed(moveList.count(where: { $0.pla == engineColor }))
        }
    }

    private func setTimeLeft(_ args: [String]) throws {
        guard args.count == 3 else {
            throw GTPError("time_left requires <color> <time> <stones>")
        }
        let color = try Self.parseColor(args[0])
        guard let seconds = Double(args[1]), seconds.isFinite, seconds >= 0,
              let stones = Int(args[2]), stones >= 0
        else {
            throw GTPError("time_left requires nonnegative <color> <time> <stones>")
        }
        if color == .black {
            pendingBlackTimeLeft = seconds
        } else {
            pendingWhiteTimeLeft = seconds
        }
        if color == engineColor {
            timeManager.updateTimeLeft(seconds)
        }
    }

    /// Read-only diagnostics used by protocol tests and future analysis output.
    public func currentTimeManager() -> TimeManager {
        timeManager
    }

    /// The exact rules object in the current legality path.
    public func currentRules() -> Rules {
        history.rules
    }

    /// Test-only seams for `PonderTests` (accessed via `@testable import`). `isPondering` reports
    /// whether a ponder task handle is currently held; `ponderTaskHandle` lets a test await the
    /// ponder search's *natural* completion (rather than cancelling it) so it can assert on the
    /// accumulated tree; `searchRootVisits`/`topSearchRootChild` expose the live `Search` root so a
    /// test can verify that a played opponent move harvests the pondered subtree.
    func isPondering() -> Bool {
        ponderTask != nil
    }

    func ponderTaskHandle() -> Task<Void, Never>? {
        ponderTask
    }

    func searchRootVisits() async -> Int64 {
        await search.currentRootVisits()
    }

    func topSearchRootChild() async -> (loc: Loc, visits: Int64)? {
        await search.topRootChild()
    }

    /// Test-only (`GTPAnalyzeTests`): the visit distribution of the last searched `generateMove`, so
    /// a test can assert `genmove` and `kata-genmove_analyze` produce byte-identical root statistics
    /// from the same seeded position. `nil` after a book-hit / emergency-pass shortcut or before any
    /// genmove.
    func lastGenmoveVisitsByMove() -> [(loc: Loc, visits: Int64)]? {
        lastGenmoveResult?.visitsByMove
    }

    // MARK: - Scoring

    /// Real score if the game already ended by double pass; otherwise an area-scoring estimate
    /// of the *current* position (a copy, so this never mutates the live game) assuming no
    /// further moves. `SearchSettings.swift`'s "Skipped" list omits dead-stone-removal-aware
    /// scoring niceties (`rootPruneUselessMoves`, ending bonus points, etc.), so this is exact
    /// Tromp-Taylor-ish scoring, not KataGo's estimation-aware `final_score`.
    private func finalScore() -> String {
        let finalWhiteMinusBlackScore: Float
        if history.isGameFinished, history.isScored {
            finalWhiteMinusBlackScore = history.finalWhiteMinusBlackScore
        } else {
            let boardCopy = board.copy()
            let historyCopy = history.copy()
            historyCopy.endAndScoreGameNow(boardCopy)
            finalWhiteMinusBlackScore = historyCopy.finalWhiteMinusBlackScore
        }
        if finalWhiteMinusBlackScore > 0 { return "W+\(Self.formatScore(finalWhiteMinusBlackScore))" }
        if finalWhiteMinusBlackScore < 0 { return "B+\(Self.formatScore(-finalWhiteMinusBlackScore))" }
        return "0"
    }

    private static func formatScore(_ value: Float) -> String {
        value == value.rounded(.towardZero) ? String(format: "%.0f", value) : String(format: "%.1f", value)
    }

    // MARK: - Parsing / formatting

    private static func parseColor(_ string: String) throws -> Player {
        switch string.lowercased() {
        case "b", "black": return .black
        case "w", "white": return .white
        default: throw GTPError("invalid color")
        }
    }

    private static func parse(_ line: String) -> ParsedCommand? {
        var text = line
        if let hashIndex = text.firstIndex(of: "#") { text = String(text[..<hashIndex]) }
        let tokens = text.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\r" }).map(String.init)
        guard !tokens.isEmpty else { return nil }
        var rest = tokens
        var id: String?
        if let first = rest.first, !first.isEmpty, first.allSatisfy(\.isNumber) {
            id = first
            rest.removeFirst()
        }
        guard let name = rest.first else { return nil }
        return ParsedCommand(id: id, name: name, args: Array(rest.dropFirst()))
    }

    /// Standard GTP response framing: `"="`/`"?"`, the echoed id (if the request had one), a
    /// space, the response text, and a blank-line terminator. Any blank line *within* the text
    /// (e.g. `showboard`'s board-then-metadata output) is replaced with a single space first,
    /// since GTP frames responses by scanning for a blank line — an internal one would terminate
    /// the response early for a naive reader.
    private static func format(id: String?, marker: String, text: String) -> String {
        let sanitized = sanitizeMultiline(text)
        let idPart = id ?? ""
        return "\(marker)\(idPart) \(sanitized)\n\n"
    }

    private static func sanitizeMultiline(_ text: String) -> String {
        var lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        while lines.last?.isEmpty == true {
            lines.removeLast()
        }
        return lines.map { $0.isEmpty ? " " : $0 }.joined(separator: "\n")
    }
}
