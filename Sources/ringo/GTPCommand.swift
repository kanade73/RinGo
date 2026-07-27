import Foundation
import RinGoCore
import RinGoEngine
import RinGoModel

struct GTPOptions {
    var model = ""
    var precision: KataGoNetwork.Precision = .float16
    var visits: Int64 = 400
    var maxBatchSize = 16
    /// Leaf-collection width for search (`SearchSettings.numSearchWorkers`): each NN `evaluate`
    /// gathers up to `min(this, leafBatchSize)` distinct leaves via virtual loss. nil keeps the
    /// legacy behavior `min(8, maxBatchSize)`. A larger value widens the GPU batch — the small 9x9
    /// net underutilizes the GPU at width 8, so wider collection raises visits/s — at a per-eval
    /// tree-freshness (staleness) cost that only a fixed-time/equal-visits A/B can price. Clamped to
    /// `leafBatchSize` (= `maxBatchSize`), so exploiting width >16 also requires a larger
    /// `-max-batch-size`. Search stays bit-reproducible at every fixed value (SearchSettings doc).
    var numSearchWorkers: Int?
    var rules = "chinese"
    var komi: Float = 7.0
    var nnLength: GTPNNLength = .auto
    var timeSeconds: Double?
    /// `-time-expected-moves N`: overrides `TimeManager`'s hardcoded expected own-move count used to
    /// divide the sudden-death clock. The 9x9 default (58) budgets as if games last 58 own moves;
    /// typical 9x9 games are ~25, so the engine left ~40-60% of the clock unused. nil keeps the
    /// per-board-size default. Structurally cannot cause a timeout (reserve/hardCap unchanged).
    var timeExpectedMoves: Int?
    /// `-time-safety-reserve <s>`: fixed wall-clock reserve subtracted from the server-reported clock
    /// in `TimeManager.budgetSeconds`, so a margin survives the CGOS/tournament controller's
    /// `time_settings`/`time_left` OVERRIDE of the `-time` fallback. 0 = off (behavior-identical).
    var timeSafetyReserve = 0.0
    var resignEnabled = true
    var treeReuse = true
    /// WP-GS2: KataGo `useGraphSearch` — DAG transposition search. Default OFF (pure tree, behavior
    /// identical to today). `-graph-search on` shares transposing positions across the search tree
    /// and reports edge statistics. Kept default OFF for the tournament regardless (activation is a
    /// separate, gated decision); exposed here for A/B evaluation.
    var graphSearch = false
    /// WP-11: LCB (Lower Confidence Bound) final-move selection for `genmove`. Default ON (KataGo's
    /// GTP default; community-measured ~+100 Elo at fixed visits). `-lcb off` restores max-visits.
    var lcb = true
    /// Optional override for `SearchSettings.lcbStdevs` (nil = keep the default 5.0).
    var lcbStdevs: Double?
    /// Optional override for `SearchSettings.cpuctExploration` (nil = keep the KataGo GTP default
    /// 1.0). Exposed for 9x9 PUCT tuning A/Bs; a lower value explores less (may suit 9x9's smaller
    /// branching factor). Does not affect numerics/oracle — only the search's move selection.
    var cpuctExploration: Double?
    /// `-symmetry off|hash|random`: KataGo `nnRandomize`-style per-eval NN input symmetry. `off`
    /// (default) = current behavior; `hash` = deterministic hash-selected (the legacy `-symmetry on`);
    /// `random` = true per-node random symmetry drawn in `Search` from `-symmetry-seed`. Closes a
    /// measured high-visit gap vs reference KataGo.
    var symmetryMode: NNSymmetryMode = .off
    /// `-symmetry-seed <n>`: seed for `random`-mode per-node symmetry draws. Fixed default for
    /// reproducible A/Bs (shared with `SearchSettings.defaultSymmetrySeed`).
    var symmetrySeed: UInt64 = SearchSettings.defaultSymmetrySeed
    /// `-root-symmetries <k>`: KataGo `rootNumSymmetriesToSample`. Averages the NN output of the root
    /// position over `k` distinct dihedral symmetries (1 = off, the default = today's behavior),
    /// lowering the variance of the root prior/value for a noisy self-trained net. Reuses
    /// `-symmetry-seed`'s stream for the without-replacement draw. Effective on square boards only.
    var rootNumSymmetries = 1
    var ponder = false
    /// Cap on NEW visits per ponder search (memory bound). 0 = auto: min(10x -visits, 20000) --
    /// see `GTPEngine.effectivePonderMaxVisits` (I-2, CODEBASE_REVIEW_0711.md: uncapped 10x let a
    /// high -visits config grow an unbounded background tree).
    var ponderMaxVisits: Int64 = 0
    /// GPU cache limit in MB for the search NNEvaluator (nil = MLX's default, uncapped). Mirrors
    /// `train`/`makedata`'s `-gpu-cache-limit-mb` (I-2, CODEBASE_REVIEW_0711.md: gtp had no cache
    /// cap at all despite being the process most likely to run for hours at a tournament).
    var gpuCacheLimitMB: Int?
    /// WP-12: path to an opening book produced by `ringo bookconvert`. Off by default (nil = no
    /// book); when set, `genmove` probes it before search and plays book moves instantly on a hit.
    var bookPath: String?
    /// WP-27 (Codex R1 finding #5): per-command wall-clock watchdog, in seconds. A hung MLX/Metal
    /// evaluation (a Metal command buffer that completes with error drops the batch's completion
    /// event, so `array.wait()` blocks forever holding mlx-swift's global `evalLock`) wedges the
    /// search actor with NO in-process recovery — the wait is a synchronous C call inside a lock
    /// that no `Task.cancel()` can interrupt. Such a wedge otherwise loses the current game on time
    /// AND every later round (the process never answers another command). This watchdog bounds each
    /// GTP command's wall clock; on breach it prints a diagnostic and force-exits so the supervisor
    /// restarts the engine (later rounds saved). `0` disables it. Default `180`: impossibly long for
    /// any single time-controlled tournament move (10-minute sudden death), so it can only fire on a
    /// genuine wedge; raise or set `0` for offline very-high-`-visits` analysis where one legitimate
    /// `genmove` might exceed it.
    var genmoveWatchdogSeconds: Double = 180
}

/// `ringo gtp`: a standard GTP loop over stdin/stdout backed by `RinGoEngine.GTPEngine`
/// (all protocol logic lives there, factored so tests can drive it without a process — this file
/// is just the stdio + argument-parsing + startup-warmup wrapper).
enum GTPCommand {
    static func main(_ arguments: [String]) async throws {
        let options = try parse(arguments)
        let rules = try Rules.parseRulesWithoutKomi(options.rules, komi: options.komi)

        var settings = SearchSettings()
        settings.maxVisits = options.visits
        settings.leafBatchSize = options.maxBatchSize
        if let workers = options.numSearchWorkers {
            settings.numSearchWorkers = max(1, min(workers, settings.leafBatchSize))
        } else {
            settings.numSearchWorkers = min(settings.numSearchWorkers, max(1, options.maxBatchSize))
        }
        settings.resignEnabled = options.resignEnabled
        settings.treeReuseEnabled = options.treeReuse
        settings.useGraphSearch = options.graphSearch
        settings.lcbEnabled = options.lcb
        if let lcbStdevs = options.lcbStdevs {
            settings.lcbStdevs = lcbStdevs
        }
        if let cpuctExploration = options.cpuctExploration {
            settings.cpuctExploration = cpuctExploration
        }
        settings.symmetryMode = options.symmetryMode
        settings.symmetrySeed = options.symmetrySeed
        settings.rootNumSymmetries = options.rootNumSymmetries

        var openingBook: OpeningBook?
        if let bookPath = options.bookPath {
            status("Loading opening book \(bookPath)...")
            openingBook = try OpeningBook.load(path: bookPath)
        }

        status("Loading model \(options.model)...")
        let desc = try ModelDesc.loadFromFileMaybeGZipped(options.model)
        let nnLen = switch options.nnLength {
        case .auto: Board.defaultLen
        case let .fixed(length): length
        }
        let evaluator = try NNEvaluator(
            desc: desc,
            nnXLen: nnLen,
            nnYLen: nnLen,
            precision: options.precision,
            maxBatchSize: options.maxBatchSize,
            gpuCacheLimitMB: options.gpuCacheLimitMB,
            symmetry: options.symmetryMode
        )
        let engine = GTPEngine(
            evaluator: evaluator,
            modelDesc: desc,
            nnXLen: nnLen,
            nnYLen: nnLen,
            precision: options.precision,
            settings: settings,
            rules: rules,
            nnLength: options.nnLength,
            maxBatchSize: options.maxBatchSize,
            initialTimeSeconds: options.timeSeconds,
            ponderEnabled: options.ponder,
            ponderMaxVisits: options.ponderMaxVisits,
            openingBook: openingBook,
            timeExpectedMoves: options.timeExpectedMoves,
            timeSafetyReserve: options.timeSafetyReserve
        )

        let effectivePonderMaxVisits = GTPEngine.effectivePonderMaxVisits(
            requested: options.ponderMaxVisits, searchMaxVisits: options.visits
        )
        let ponderBanner = options.ponder ? "on(max-visits=\(effectivePonderMaxVisits))" : "off"
        let lcbBanner = options.lcb ? "on(stdevs=\(settings.lcbStdevs))" : "off"
        let bookBanner = openingBook.map { "on(\(options.bookPath ?? "?"), \($0.count) entries)" } ?? "off"
        let workersBanner = options.numSearchWorkers.map { String($0) } ?? "default"
        let expectedMovesBanner = options.timeExpectedMoves.map { String($0) } ?? "default"
        let cpuctBanner = options.cpuctExploration.map { String($0) } ?? "default"
        let symmetryBanner = options.symmetryMode == .random
            ? "random(seed=\(options.symmetrySeed))"
            : options.symmetryMode.rawValue
        let rootSymmetriesBanner = options.rootNumSymmetries > 1
            ? "\(options.rootNumSymmetries)(seed=\(options.symmetrySeed))"
            : "off"
        status("Config: visits=\(options.visits) precision=\(options.precision == .float16 ? "fp16" : "fp32") "
            + "tree-reuse=\(options.treeReuse ? "on" : "off") graph-search=\(options.graphSearch ? "on" : "off") "
            + "resign=\(options.resignEnabled) "
            + "lcb=\(lcbBanner) ponder=\(ponderBanner) book=\(bookBanner) "
            + "search-workers=\(workersBanner) time-expected-moves=\(expectedMovesBanner) cpuct=\(cpuctBanner) "
            + "symmetry=\(symmetryBanner) root-symmetries=\(rootSymmetriesBanner)")
        status("Pre-warming compile buckets...")
        let warmupStart = Date()
        try await engine.preWarm(bucketSizes: bucketSizes(upTo: options.maxBatchSize))
        status("Ready (warmup took \(String(format: "%.1f", Date().timeIntervalSince(warmupStart) * 1000)) ms).")

        // Pondering runs a background search on the opponent-to-move root while this loop blocks on
        // `readLine`. Every command first tears down any in-flight ponder (cancel-and-await, so the
        // Search actor is quiescent before the command mutates the tree), and — unless we're
        // quitting — a fresh ponder is started after the response is printed. With `-ponder off`
        // both calls are no-ops, so this is byte-identical to the pre-ponder loop.
        let watchdogSeconds = options.genmoveWatchdogSeconds
        while let line = readLine(strippingNewline: true) {
            // WP-27: bound the stopPondering+handle wall clock. A wedged MLX eval cannot be
            // cancelled in-process, so the watchdog's only recovery is a loud force-exit; the
            // supervisor then restarts. `startPonderingIfEnabled` is outside the guard: it only
            // spawns a background task and returns immediately (a wedge in the ponder it starts is
            // caught by the NEXT command's stopPondering, which is inside the guard).
            let (response, shouldQuit) = await withWatchdog(
                seconds: watchdogSeconds,
                onTimeout: { watchdogExpired(seconds: watchdogSeconds, command: line) },
                body: {
                    await engine.stopPondering()
                    return await engine.handle(line)
                }
            )
            if let response {
                print(response, terminator: "")
                fflush(stdout)
            }
            if shouldQuit { break }
            await engine.startPonderingIfEnabled()
        }
        await engine.stopPondering()
    }

    /// Runs `body`, force-invoking `onTimeout` exactly once if it has not finished within `seconds`
    /// wall clock (a detached watchdog task; `seconds <= 0` disables it). Race-free via
    /// `WatchdogLatch`: whichever of {body finished, deadline elapsed} happens first claims the
    /// latch; the loser is a no-op, so `onTimeout` never fires for a body that beat the deadline and
    /// the watchdog task is always cancelled once the body returns. Factored out (and left
    /// non-private) so `GTPCommandWatchdogTests` can drive it with a synthetic slow/fast body without
    /// a real MLX evaluator or a process exit.
    static func withWatchdog<T>(
        seconds: Double,
        onTimeout: @escaping @Sendable () -> Void,
        body: () async -> T
    ) async -> T {
        guard seconds > 0 else { return await body() }
        let latch = WatchdogLatch()
        let watchdog = Task {
            try? await Task.sleep(nanoseconds: UInt64((seconds * 1_000_000_000).rounded()))
            if latch.claim() { onTimeout() }
        }
        let result = await body()
        if latch.claim() { watchdog.cancel() }
        return result
    }

    /// Emits the wedge diagnostic and terminates immediately. Uses `_exit` (not `exit`) on purpose:
    /// a normal `exit` runs atexit handlers / static destructors that may try to tear down the MLX
    /// scheduler and re-acquire the very `evalLock` the wedged eval still holds, which would hang the
    /// exit itself. `FileHandle.write` is an unbuffered `write(2)`, so the diagnostic is flushed
    /// before the process dies. Exit code 70 (`EX_SOFTWARE`) signals an internal failure to the
    /// supervisor.
    private static func watchdogExpired(seconds: Double, command: String) -> Never {
        FileHandle.standardError.write(Data((
            "FATAL: GTP command \(command.split(separator: " ").first.map(String.init) ?? command) "
                + "exceeded the \(Int(seconds.rounded()))s wall-clock watchdog — the MLX/Metal "
                + "evaluator is wedged (an unrecoverable hung GPU eval; Codex R1 finding #5). "
                + "Force-exiting so the supervisor can restart: this game is lost but later rounds "
                + "are saved.\n"
        ).utf8))
        _exit(70)
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

    private static func status(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    static func parse(_ arguments: [String]) throws -> GTPOptions {
        var options = GTPOptions()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else {
                throw CLIError(description: "Missing value for \(flag)")
            }
            let value = arguments[index + 1]
            switch flag {
            case "-model": options.model = value
            case "-precision":
                switch value {
                case "fp32": options.precision = .float32
                case "fp16": options.precision = .float16
                default: throw CLIError(description: "Precision must be fp32 or fp16")
                }
            case "-visits":
                guard let visits = Int64(value), visits > 0 else {
                    throw CLIError(description: "Invalid -visits '\(value)'")
                }
                options.visits = visits
            case "-max-batch-size":
                guard let maxBatchSize = Int(value), maxBatchSize > 0 else {
                    throw CLIError(description: "Invalid -max-batch-size '\(value)'")
                }
                options.maxBatchSize = maxBatchSize
            case "-rules": options.rules = value
            case "-komi":
                guard let komi = Float(value), komi.isFinite else {
                    throw CLIError(description: "Invalid -komi '\(value)'")
                }
                options.komi = komi
            case "-nnlen": options.nnLength = try GTPNNLength(argument: value)
            case "-time":
                guard let time = Double(value), time.isFinite, time > 0 else {
                    throw CLIError(description: "Invalid -time '\(value)'")
                }
                options.timeSeconds = time
            case "-time-safety-reserve":
                guard let reserve = Double(value), reserve.isFinite, reserve >= 0 else {
                    throw CLIError(
                        description: "Invalid -time-safety-reserve '\(value)'; expected a nonnegative number"
                    )
                }
                options.timeSafetyReserve = reserve
            case "-resign-enabled":
                switch value {
                case "true": options.resignEnabled = true
                case "false": options.resignEnabled = false
                default: throw CLIError(description: "Invalid -resign-enabled '\(value)'; expected true or false")
                }
            case "-tree-reuse":
                switch value {
                case "on", "true": options.treeReuse = true
                case "off", "false": options.treeReuse = false
                default: throw CLIError(description: "Invalid -tree-reuse '\(value)'; expected on or off")
                }
            case "-graph-search":
                switch value {
                case "on", "true": options.graphSearch = true
                case "off", "false": options.graphSearch = false
                default: throw CLIError(description: "Invalid -graph-search '\(value)'; expected on or off")
                }
            case "-lcb":
                switch value {
                case "on", "true": options.lcb = true
                case "off", "false": options.lcb = false
                default: throw CLIError(description: "Invalid -lcb '\(value)'; expected on or off")
                }
            case "-lcb-stdevs":
                guard let stdevs = Double(value), stdevs.isFinite, stdevs >= 0 else {
                    throw CLIError(description: "Invalid -lcb-stdevs '\(value)'; expected a non-negative number")
                }
                options.lcbStdevs = stdevs
            case "-cpuct-exploration":
                guard let cpuct = Double(value), cpuct.isFinite, cpuct > 0 else {
                    throw CLIError(description: "Invalid -cpuct-exploration '\(value)'; expected a positive number")
                }
                options.cpuctExploration = cpuct
            case "-symmetry":
                switch value {
                case "off", "false": options.symmetryMode = .off
                // Legacy alias: `-symmetry on` (and `true`) selected the deterministic hash mode.
                case "hash", "on", "true": options.symmetryMode = .hash
                case "random": options.symmetryMode = .random
                default: throw CLIError(description: "Invalid -symmetry '\(value)'; expected off, hash, or random")
                }
            case "-symmetry-seed":
                guard let seed = UInt64(value) else {
                    throw CLIError(description: "Invalid -symmetry-seed '\(value)'; expected a non-negative integer")
                }
                options.symmetrySeed = seed
            case "-root-symmetries":
                guard let parsed = Int(value), parsed >= 1, parsed <= Symmetry.count else {
                    throw CLIError(
                        description: "Invalid -root-symmetries '\(value)'; expected an integer in 1...\(Symmetry.count)"
                    )
                }
                options.rootNumSymmetries = parsed
            case "-ponder":
                switch value {
                case "on", "true": options.ponder = true
                case "off", "false": options.ponder = false
                default: throw CLIError(description: "Invalid -ponder '\(value)'; expected on or off")
                }
            case "-ponder-max-visits":
                guard let cap = Int64(value), cap > 0 else {
                    throw CLIError(description: "Invalid -ponder-max-visits '\(value)'; expected a positive integer")
                }
                options.ponderMaxVisits = cap
            case "-gpu-cache-limit-mb":
                guard let parsed = Int(value), parsed >= 0 else {
                    throw CLIError(description: "Invalid -gpu-cache-limit-mb '\(value)'")
                }
                options.gpuCacheLimitMB = parsed
            case "-num-search-workers":
                guard let parsed = Int(value), parsed >= 1 else {
                    throw CLIError(description: "Invalid -num-search-workers '\(value)'; expected a positive integer")
                }
                options.numSearchWorkers = parsed
            case "-time-expected-moves":
                guard let parsed = Int(value), parsed >= 1 else {
                    throw CLIError(description: "Invalid -time-expected-moves '\(value)'; expected a positive integer")
                }
                options.timeExpectedMoves = parsed
            case "-book": options.bookPath = value
            case "-genmove-watchdog-seconds":
                guard let parsed = Double(value), parsed.isFinite, parsed >= 0 else {
                    throw CLIError(
                        description: "Invalid -genmove-watchdog-seconds '\(value)'; expected a non-negative number (0 disables)"
                    )
                }
                options.genmoveWatchdogSeconds = parsed
            default: throw CLIError(description: "Unknown option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard !options.model.isEmpty else {
            throw CLIError(description: usage)
        }
        return options
    }

    static let usage = """
    Usage: ringo gtp -model <file> [-precision fp16|fp32] [-visits N=400] \
      [-max-batch-size 16] [-num-search-workers N] [-rules <str>=chinese] [-komi f=7.0] \
      [-nnlen <auto|N>=auto] [-time seconds] [-time-expected-moves N] \
      [-resign-enabled true|false] [-tree-reuse on|off] [-graph-search on|off] [-lcb on|off] [-lcb-stdevs f=5.0] \
      [-cpuct-exploration f=1.0] [-symmetry off|hash|random] [-symmetry-seed N] \
      [-root-symmetries k=1] \
      [-ponder on|off] [-ponder-max-visits N] [-gpu-cache-limit-mb <N>] [-book <file.nngb>] \
      [-genmove-watchdog-seconds f=180]
    """
}

/// Race-free "who finished first" latch shared between a GTP command coroutine and its watchdog
/// task (WP-27). `claim()` returns `true` to exactly one caller — whichever of {the command
/// returned, the watchdog deadline elapsed} reaches it first — so the watchdog's force-exit fires
/// at most once and never for a command that beat the deadline. `@unchecked Sendable`: the single
/// `Bool` is guarded by the lock, the only shared state.
final class WatchdogLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var settled = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if settled { return false }
        settled = true
        return true
    }
}
