import Foundation
import RinGoCore
import RinGoModel

public struct SearchError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// Graph-search event counters (design §2.4). Exposed through `Search.graphDiagnosticsSnapshot()`
/// for the adversarial tests (path-illegal / collision / cycle coverage) and self-play smoke logs.
struct GraphDiagnostics: Equatable {
    /// Playouts where the PUCT-selected move was illegal on the actual path history, forcing a
    /// path-specific NN regeneration (KataGo `search.cpp:1239-1286`).
    var pathIllegal = 0
    /// Node-table hits whose full-state fingerprint did NOT match, i.e. a genuine hash collision of
    /// unrelated positions, forced to a private (non-merged) node (review §B2).
    var collisions = 0
    /// Playouts that would have descended into a node already on the current path (a cycle),
    /// bailed with an edge visit (KataGo `search.cpp:1393-1414`).
    var cycles = 0
    /// Merges: node-table hits whose fingerprint matched, so a transposing edge linked an existing
    /// shared node instead of creating a new one. Not an error — the whole point of graph search.
    var merges = 0
}

/// Sendable per-node facts for graph-search accounting tests (the `SearchNode` itself cannot cross
/// the actor boundary, so the walk happens inside the actor and returns these).
struct GraphNodeSummary {
    enum StateKind { case unevaluated, evaluating, expanded }
    var state: StateKind
    var isTerminal: Bool
    var visits: Int64
    var weightSum: Double
    var outboundEdgeSum: Int64
    var inDegree: Int
    var inboundEdgeSum: Int64
    var anyEdgeExceedsChildVisits: Bool
    var utilityAvg: Double
}

/// Sendable per-root-child facts for the edge-statistics consumer test (T-G12).
struct GraphRootChildFact {
    var loc: Loc
    var edgeVisits: Int64
    var nodeVisits: Int64
    var inDegree: Int
}

/// Holds WEAK references to a set of `SearchNode`s so a test can detect ARC leaks after a graph
/// re-root/clear (T-R2/T-R4). Because it never strong-references a node, it cannot keep a
/// mark-swept node alive; `aliveCount` therefore reports exactly how many survive — which for a
/// leak-free, cycle-broken sweep equals the retained (reachable) set, never more.
final class WeakNodeBox {
    private struct Weak {
        weak var node: SearchNode?
    }

    private let refs: [Weak]

    init(_ nodes: some Sequence<SearchNode>) {
        refs = nodes.map { Weak(node: $0) }
    }

    var aliveCount: Int {
        refs.reduce(0) { $0 + ($1.node == nil ? 0 : 1) }
    }
}

/// One node of the search tree. A plain reference type mutated only from within `Search`'s actor
/// isolation (never shared across actors), so it does not need `Sendable` conformance.
///
/// Fields directly mirror `SearchParams`'s pared-down needs per `SearchSettings.swift`'s doc
/// comment: `visits`/`weightSum` (kept as two fields, though this port always has `weight == 1`
/// per leaf since `useUncertainty` is not implemented — see that file's "Skipped" list — so
/// `weightSum == Double(visits)` always; kept separate to mirror KataGo's `NodeStats` and to
/// leave a seam if weighted evals are ever added), `utilityAvg`/`utilitySqAvg` (for PUCT's
/// parent-utility-stdev factor), `scoreMeanAvg`/`scoreMeanSqAvg` (for score utility),
/// `winValueAvg` (white win-minus-loss average, reporting only — winrate/resign), and
/// `virtualLosses`. `children` is an append-only array (not a `[Loc: SearchNode]` dictionary) so
/// that iteration order is deterministic (creation order) rather than hash-seed dependent, which
/// matters for the determinism test: dictionary iteration order could otherwise silently flip an
/// exact PUCT-value tie's winner between runs.
final class SearchNode {
    enum State {
        case unevaluated
        case evaluating
        case expanded
    }

    let board: Board
    let history: BoardHistory
    let nextPla: Player

    /// Graph-search node identity (`GraphHash`), set at creation when `useGraphSearch` is on and the
    /// node-table key this node is stored under (root excepted — the root is never table-inserted).
    /// Left `Hash128()` in pure-tree mode, where it is never read. Mirrors KataGo's `graphHash`.
    var graphHash = Hash128()

    var state = State.unevaluated
    var nnOutput: NNOutput?
    var visits: Int64 = 0
    var weightSum = 0.0
    var utilityAvg = 0.0
    var utilitySqAvg = 0.0
    var winValueAvg = 0.0
    var scoreMeanAvg = 0.0
    var scoreMeanSqAvg = 0.0
    var virtualLosses = 0

    /// Graph-mode backup micro-cache for `recomputeFromChildren`'s "own NN value at weight 1" term.
    /// `cachedSelfLeafValue` is a pure function of `nnOutput` (constant once expanded); `cachedSelfUtility`
    /// additionally depends on the search's `recentScoreCenter`, which is constant across a single
    /// `runSearch` (set once by `computeRecentScoreCenter`), so it is tagged with the center it was
    /// computed under and reused only on an exact match. Because graph backup recomputes a node from its
    /// children on EVERY playout through it, this turns a per-ancestor-per-playout `scoreUtility`
    /// evaluation (~200 `atan` calls via `expectedWhiteScoreValue`) into a once-per-node cost. Invalidated
    /// whenever `nnOutput` is overwritten (path-illegal reeval). Never read in pure-tree mode.
    var cachedSelfLeafValue: LeafValue?
    var cachedSelfUtility = 0.0
    var cachedSelfUtilityCenter = Double.nan

    /// Legal moves not yet expanded into a child, sorted by descending NN policy probability.
    /// Because an unexpanded child's explore-selection value is monotonic in its policy prob (all
    /// unexpanded children share the same FPU utility and zero weight — see
    /// `Search.selectChild`), the best not-yet-tried move is always the head of this list, so a
    /// simple pop-the-front pointer reproduces KataGo's "try the best new child" logic exactly
    /// without re-scanning every remaining legal move on every visit.
    var untriedMoves: [(loc: Loc, prob: Float)] = []
    /// `(loc, node, edgeVisits)`. `edgeVisits` is this parent's own investment in the edge, tracked
    /// separately from the (possibly shared) child node's `visits` only in graph mode; in pure-tree
    /// mode it stays 0 and is never read (edge == child visits there by construction).
    private(set) var children: [(loc: Loc, node: SearchNode, edgeVisits: Int64)] = []

    init(board: Board, history: BoardHistory, nextPla: Player) {
        self.board = board
        self.history = history
        self.nextPla = nextPla
    }

    func childNode(for loc: Loc) -> SearchNode? {
        children.first { $0.loc == loc }?.node
    }

    func childIndex(for loc: Loc) -> Int? {
        children.firstIndex { $0.loc == loc }
    }

    func addChild(loc: Loc, node: SearchNode, edgeVisits: Int64 = 0) {
        children.append((loc, node, edgeVisits))
    }

    /// Graph mode only: bump one outgoing edge's own visit count (KataGo `addEdgeVisits(1)`).
    func incrementEdgeVisits(at index: Int) {
        children[index].edgeVisits += 1
    }

    /// Graph-mode root promotion: drop every outgoing edge whose move fails `isRetained` (the strict
    /// legality re-filter run once a promoted subtree becomes the new root — KataGo
    /// `beginSearch`'s root-child filter, `search.cpp:744-770`). Only ever called on the new private
    /// root, which is never in the node table, so a dropped edge needs no table bookkeeping here (the
    /// caller's mark-sweep reclaims any now-unreachable child). Returns whether anything was dropped,
    /// so the caller only recomputes visits/stats when the edge set actually changed (matching
    /// KataGo's `anyFiltered` guard — a promoted child with all-legal edges keeps its exact stats).
    @discardableResult
    func retainChildren(where isRetained: (Loc) -> Bool) -> Bool {
        let before = children.count
        children.removeAll { !isRetained($0.loc) }
        return children.count != before
    }

    /// Breaks this node's outgoing references so a doomed node in a DAG (which may sit on a cycle of
    /// strong references once transpositions link it both ways) can be freed by ARC. Used by graph
    /// mode's table teardown; `unowned`/`weak` were rejected (design §2.5-5 / review §A6) in favor of
    /// strong edges plus this explicit teardown.
    func clearLinks() {
        children.removeAll()
        untriedMoves.removeAll()
    }

    /// Applies one weight-1 leaf observation to this node's running means: mean utility and its
    /// second moment (`utilityAvg`/`utilitySqAvg`, whose difference `utilitySqAvg - utilityAvg^2`
    /// is the value variance LCB reads — WP-11), plus the score/winrate reporting stats, advancing
    /// `weightSum`/`visits`. Extracted verbatim from `Search.applyBackup`'s per-node loop (same
    /// operations, same order, using the pre-update `weightSum`) so backup arithmetic — and thus
    /// determinism — is unchanged; kept as the single source of truth for the mean/second-moment
    /// update so the variance LCB consumes is exactly what search accumulated.
    func accumulateLeaf(utility: Double, utilitySq: Double, scoreMean: Double, scoreMeanSq: Double, winLoss: Double) {
        let newWeightSum = weightSum + 1.0
        utilityAvg = (utilityAvg * weightSum + utility) / newWeightSum
        utilitySqAvg = (utilitySqAvg * weightSum + utilitySq) / newWeightSum
        scoreMeanAvg = (scoreMeanAvg * weightSum + scoreMean) / newWeightSum
        scoreMeanSqAvg = (scoreMeanSqAvg * weightSum + scoreMeanSq) / newWeightSum
        winValueAvg = (winValueAvg * weightSum + winLoss) / newWeightSum
        weightSum = newWeightSum
        visits += 1
    }

    /// Drops the graph-backup self-value cache; call whenever `nnOutput` is replaced so the next
    /// `recomputeFromChildren` recomputes the own-value term from the new output (path-illegal reeval).
    func invalidateSelfValueCache() {
        cachedSelfLeafValue = nil
        cachedSelfUtilityCenter = Double.nan
    }
}

/// A leaf observation in KataGo's white-perspective units, before combining into scalar utility
/// (`Search.utility(of:)`). Mirrors the parameters of `Search::addLeafValue` in
/// `$KG/cpp/search/searchupdatehelpers.cpp`.
struct LeafValue {
    var winLossValue: Double
    var noResultValue: Double
    var scoreMean: Double
    var scoreMeanSq: Double
}

/// Result of a completed `Search.runSearch` call: enough for GTP `genmove`/resign-tracking and
/// for tests, without exposing tree internals (`SearchNode` stays file-private to the module).
public struct SearchResult: Sendable {
    public var bestMove: Loc
    public var rootVisits: Int64
    /// Visits completed by this call, independent of any visits retained through tree reuse.
    public var visitsAchieved: Int64
    /// Monotonic wall-clock time spent in this call, including root bootstrap evaluation.
    public var timeUsed: TimeInterval
    /// White win-probability treating no-result as a half win (`drawEquivalentWinsForWhite`
    /// convention, matching `ScoreValueUtils.whiteWinsOfWinner`), derived from the root's
    /// backed-up `winValueAvg` (`whiteWinProb - whiteLossProb`, averaged): `(winValueAvg+1)/2`.
    public var whiteWinProb: Double
    public var whiteScoreMean: Double
    /// `(move, visits)` for every expanded root child, in descending visit order — used by tests
    /// and GTP `showboard`/analysis-style debugging; not required by any GTP command in scope.
    public var visitsByMove: [(loc: Loc, visits: Int64)]
    /// The root's raw (unsearched) postprocessed NN output, as computed while bootstrapping the
    /// root. Exposed for teacher-search distillation (`TeacherLabeler.searchTargets`), which pairs
    /// the search's visit/value distribution with the root net's own score/ownership heads. Always
    /// non-nil after a completed `runSearch` (the root is always evaluated), but optional so the
    /// type stays honest about the pre-bootstrap state.
    public var rootNNOutput: NNOutput?
    /// Backed-up win-minus-loss value (`SearchNode.winValueAvg`, WHITE perspective, in [-1, 1]) for
    /// every EXPANDED root child, keyed by move loc (pass included). Same white-perspective
    /// convention as `whiteWinProb`/`whiteScoreMean`; a consumer reprojects to the side to move.
    /// Exposed for the Gumbel completed-Q teacher policy target (WP-13,
    /// `TeacherLabeler.searchTargets(target: .completedQ)`), which needs each visited child's Q — not
    /// just its visit count — to complete the improved policy. A loc is present here iff it is a root
    /// child, so `rootChildWinValues[loc]` and the same loc's entry in `visitsByMove` describe the
    /// same child. Empty until a completed `runSearch`.
    public var rootChildWinValues: [Loc: Double]
}

/// PUCT MCTS over `RinGoCore`/`RinGoEngine` primitives. Design, against the task's spec:
///
/// - **Concurrency**: `Search` is an actor — its own isolation *is* the "lightweight lock" the
///   task calls for; every `SearchNode` mutation happens on this actor, so no separate lock type
///   is needed. `runSearch` drives one deterministic playout loop (no worker tasks): it collects
///   a batch of leaves that each need a fresh NN eval, issues a single awaited
///   `NNEvaluator.evaluate` for the whole batch, then backs the results up — repeating until the
///   visit budget is spent. An earlier revision spawned `numSearchWorkers` concurrent worker
///   tasks feeding a "flush when full or when all workers are parked" batcher; that protocol
///   deadlocked (a worker finishing its loop lowered the active-worker count below the parked
///   count with no code path left to re-check the flush predicate, so parked workers'
///   continuations were never resumed), and its multi-worker interleaving made results
///   non-reproducible against the non-associative running-mean backup. The sequential collector is
///   simpler, always terminates, and is bit-reproducible at every worker count. The GPU/CPU
///   overlap the old design chased is minor here (tree descent is microseconds against a
///   multi-millisecond batched GPU eval), and `NNEvaluator` still pipelines the physical
///   sub-batches within each `evaluate` internally.
/// - **Batching**: one collection pass walks the tree up to `batchTarget`
///   (`min(leafBatchSize, numSearchWorkers)` — the old design's effective batch width) times,
///   accumulating distinct leaves that need evaluation; virtual loss applied to each collected
///   leaf's path diversifies the next walk in the same pass so the batch spans different leaves
///   (leaf-parallel MCTS, run sequentially). Terminal playouts encountered during collection are
///   backed up immediately and counted; the pass stops early when a walk re-selects a leaf already
///   in the batch (`.abandoned`) or the remaining visit budget is reached. `numSearchWorkers == 1`
///   collapses this to `batchTarget == 1` — textbook one-leaf-at-a-time MCTS.
/// - **Virtual loss**: applied to every node on a path from (but not including) the root down to a
///   collected leaf once that leaf is confirmed pending an NN eval, and removed when its eval
///   result is backed up. Because collection and backup run on the same serial actor context with
///   no suspension between a walk and its virtual-loss bookkeeping, this is observably identical to
///   KataGo's per-step add/remove around each recursive call, just simpler to implement.
/// - **Node reached mid-eval**: if a walk reaches a node already `.evaluating` (a leaf collected
///   earlier in the same batch, re-selected via virtual-loss-adjusted descent), collection stops
///   and the batch is flushed as-is, matching `Search::playoutDescend`'s
///   `STATE_EVALUATING -> shouldCountPlayout = false` branch (here: don't double-collect it).
public actor Search {
    private let evaluator: any NNEvaluating
    private let primaryPrecision: KataGoNetwork.Precision
    private let nnXLen: Int
    private let nnYLen: Int
    private let settings: SearchSettings
    private let fp32FallbackFactory: (@Sendable () throws -> any NNEvaluating)?
    private var fp32Fallback: (any NNEvaluating)?
    private var random: SearchRandom
    /// Per-eval input symmetry mode (KataGo `nnRandomize`). Only `.random` engages the draw below.
    private let symmetryMode: NNSymmetryMode
    /// Dedicated seeded stream for `.random`-mode per-node symmetry draws, kept SEPARATE from
    /// `random` so a symmetry draw never shifts the root-noise / final-move-sampling sequence.
    private var symmetryRandom: SearchRandom

    private var root: SearchNode
    private var rootPolicyProbs: [Float]?
    private var recentScoreCenter = 0.0
    private var sqrtBoardArea = 0.0

    /// Graph-search node table (`useGraphSearch` only): `GraphHash` → the single shared `SearchNode`
    /// for that position. Always empty in pure-tree mode. The root is deliberately never inserted
    /// (KataGo `search.cpp:725-729`). A single dictionary replaces KataGo's sharded map + mutex pool
    /// because this port's search is a serial actor.
    private var nodeTable: [Hash128: SearchNode] = [:]
    /// Adversarial-test / smoke-log counters for graph-mode events (design §2.4 diagnostics seam).
    private var graphDiagnostics = GraphDiagnostics()
    /// Warn-once-per-search set for path-illegal regenerations, keyed by the regenerated output's
    /// `nnHash` (KataGo `thread.illegalMoveHashes`, `search.cpp:1268`). Reset each `runSearch`.
    private var warnedIllegalHashes: Set<Hash128> = []
    /// WP-GS3 ARC-leak probe (T-R2/T-R4): a test arms this with WEAK references to the current table
    /// nodes just before a re-root/clear, then reads back how many survived. Because it never
    /// strong-references a node it cannot itself keep a swept node alive, so a residual count above
    /// the reachable set would expose a mark-sweep cycle leak. Always `nil` outside those tests.
    private var livenessProbe: WeakNodeBox?

    public init(
        evaluator: any NNEvaluating,
        modelDesc: ModelDesc,
        nnXLen: Int,
        nnYLen: Int,
        precision: KataGoNetwork.Precision,
        settings: SearchSettings,
        board: Board,
        history: BoardHistory,
        nextPlayer: Player
    ) {
        self.evaluator = evaluator
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        primaryPrecision = precision
        self.settings = settings
        random = SearchRandom(seed: settings.randomSeed)
        symmetryMode = settings.symmetryMode
        symmetryRandom = SearchRandom(seed: settings.symmetrySeed)
        // The NaN-retry fp32 evaluator must carry the SAME symmetry mode as the primary, or a
        // single fp16 fallback would silently evaluate one leaf under a different mode. The carried
        // per-request `params.symmetry` (set below) additionally guarantees an identical drawn
        // symmetry across primary and fallback of the SAME request. `fallbackSymmetry` is captured
        // by value into the @Sendable factory closure.
        let fallbackSymmetry = settings.symmetryMode
        fp32FallbackFactory = {
            try NNEvaluator(
                desc: modelDesc,
                nnXLen: nnXLen,
                nnYLen: nnYLen,
                precision: .float32,
                maxBatchSize: 1,
                symmetry: fallbackSymmetry
            )
        }
        root = SearchNode(board: board.copy(), history: history.copy(), nextPla: nextPlayer)
    }

    init(
        evaluator: any NNEvaluating,
        nnXLen: Int,
        nnYLen: Int,
        precision: KataGoNetwork.Precision,
        settings: SearchSettings,
        board: Board,
        history: BoardHistory,
        nextPlayer: Player,
        fp32FallbackFactory: (@Sendable () throws -> any NNEvaluating)?
    ) {
        self.evaluator = evaluator
        primaryPrecision = precision
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        self.settings = settings
        random = SearchRandom(seed: settings.randomSeed)
        symmetryMode = settings.symmetryMode
        symmetryRandom = SearchRandom(seed: settings.symmetrySeed)
        self.fp32FallbackFactory = fp32FallbackFactory
        root = SearchNode(board: board.copy(), history: history.copy(), nextPla: nextPlayer)
    }

    /// Discards the tree entirely and starts fresh at `board`/`history`/`nextPlayer` (GTP
    /// `clear_board`/`boardsize`/`komi`/`undo` — anything that doesn't correspond to "the
    /// opponent or we played one legal move forward").
    public func setPosition(board: Board, history: BoardHistory, nextPlayer: Player) {
        clearGraphTable()
        graphDiagnostics = GraphDiagnostics()
        root = SearchNode(board: board.copy(), history: history.copy(), nextPla: nextPlayer)
        rootPolicyProbs = nil
    }

    public func currentRootBoard() -> Board {
        root.board
    }

    public func currentRootHistory() -> BoardHistory {
        root.history
    }

    /// The current root's accumulated visit count. After `makeMove` this equals the reused
    /// subtree's visits (0 for a fresh/non-reused root); exposed for tree-reuse tests to assert
    /// re-rooting preserved (or, when disabled/rejected, discarded) the played child's stats.
    public func currentRootVisits() -> Int64 {
        root.visits
    }

    /// The current root's most-visited child, as `(loc, visits)`, or `nil` if the root has no
    /// expanded children yet. Test-only seam for the pondering harvest test (`PonderTests`): it
    /// lets a test pick the exact opponent reply the ponder search explored most, `play` it, and
    /// assert the re-rooted subtree's visits match — proving pondered work survives the re-root.
    func topRootChild() -> (loc: Loc, visits: Int64)? {
        if settings.useGraphSearch {
            // Edge visits are the per-parent investment the ponder-harvest test wants to compare.
            return root.children.max { $0.edgeVisits < $1.edgeVisits }.map { ($0.loc, $0.edgeVisits) }
        }
        return root.children.max { $0.node.visits < $1.node.visits }.map { ($0.loc, $0.node.visits) }
    }

    // MARK: - Graph-search test seams (internal; invisible to the public API, OFF-path unaffected)

    //
    // `SearchNode` cannot cross the actor boundary (non-Sendable), so every walk happens here and
    // returns Sendable value summaries.

    /// Snapshot of the graph-mode event counters accumulated across every `runSearch` since the last
    /// `setPosition` (a re-root via `makeMove` deliberately does NOT reset them, so smoke logging sees
    /// per-game cumulative path-illegal / collision / cycle / merge rates). Used by the T-G/T-R tests.
    func graphDiagnosticsSnapshot() -> GraphDiagnostics {
        graphDiagnostics
    }

    /// Number of shared nodes currently in the graph node table (0 in pure-tree mode).
    func nodeTableCountForTests() -> Int {
        nodeTable.count
    }

    /// Walks the DAG from the root (dedup by identity) and returns per-node accounting facts for the
    /// invariant tests (T-G1/T-G2/T-G8).
    func graphNodeSummariesForTests() -> [GraphNodeSummary] {
        let nodes = reachableNodes()
        var inDegree: [ObjectIdentifier: Int] = [:]
        var inboundEdgeSum: [ObjectIdentifier: Int64] = [:]
        for node in nodes {
            for child in node.children {
                let key = ObjectIdentifier(child.node)
                inDegree[key, default: 0] += 1
                inboundEdgeSum[key, default: 0] += child.edgeVisits
            }
        }
        return nodes.map { node in
            let key = ObjectIdentifier(node)
            let outbound = node.children.reduce(Int64(0)) { $0 + $1.edgeVisits }
            let stateKind: GraphNodeSummary.StateKind = switch node.state {
            case .unevaluated: .unevaluated
            case .evaluating: .evaluating
            case .expanded: .expanded
            }
            return GraphNodeSummary(
                state: stateKind,
                isTerminal: node.history.isGameFinished,
                visits: node.visits,
                weightSum: node.weightSum,
                outboundEdgeSum: outbound,
                inDegree: inDegree[key] ?? 0,
                inboundEdgeSum: inboundEdgeSum[key] ?? 0,
                anyEdgeExceedsChildVisits: node.children.contains { $0.edgeVisits > $0.node.visits },
                utilityAvg: node.utilityAvg
            )
        }
    }

    /// Per-root-child facts (edge visits vs shared node visits + in-degree) for the edge-statistics
    /// consumer test (T-G12).
    func graphRootChildFactsForTests() -> [GraphRootChildFact] {
        let nodes = reachableNodes()
        var inDegree: [ObjectIdentifier: Int] = [:]
        for node in nodes {
            for child in node.children {
                inDegree[ObjectIdentifier(child.node), default: 0] += 1
            }
        }
        return root.children.map {
            GraphRootChildFact(
                loc: $0.loc,
                edgeVisits: $0.edgeVisits,
                nodeVisits: $0.node.visits,
                inDegree: inDegree[ObjectIdentifier($0.node)] ?? 0
            )
        }
    }

    /// True if any reachable node is still `.evaluating` or carries a virtual loss — a dangling
    /// pending state (T-G11 quiescence check).
    func graphHasDanglingPendingForTests() -> Bool {
        reachableNodes().contains { $0.state == .evaluating || $0.virtualLosses != 0 }
    }

    /// Maximum absolute difference, over expanded non-terminal nodes, between a node's stored
    /// `utilityAvg` and the value a fresh from-children recompute would give — i.e. the worst stale
    /// shared-parent error (T-G10). Small once heavily-visited shared children have stabilized.
    func graphMaxUtilityStalenessForTests() -> Double {
        var worst = 0.0
        for node in reachableNodes() where node.state == .expanded && !node.history.isGameFinished {
            worst = max(worst, abs(node.utilityAvg - recomputedUtilityWithoutMutation(node)))
        }
        return worst
    }

    /// Forces a hash COLLISION (T-G9): inserts a private node under the graph hash the child reached
    /// by `loc` will compute, but with a DIFFERENT board/history so the fingerprint cannot match —
    /// the search must then refuse to merge and make a private node (`collisions` diag increments).
    func injectCollisionNodeForTests(afterMove loc: Loc) {
        let key = childGraphHashFromRoot(loc: loc)
        // A deliberately mismatched occupant: the current root position (fingerprint will differ from
        // the child reached by `loc`).
        let mismatched = SearchNode(board: root.board.copy(), history: root.history.copy(), nextPla: root.nextPla)
        mismatched.graphHash = key
        nodeTable[key] = mismatched
    }

    /// Pre-seeds a shared node under the graph hash the root child at `loc` will compute, with a
    /// MATCHING fingerprint and `visits` already accumulated (T-G12), so that child merges onto it
    /// and its root edge lags the shared node's visits.
    func injectSharedRootChildForTests(afterMove loc: Loc, visits: Int64) {
        let key = childGraphHashFromRoot(loc: loc)
        let childBoard = root.board.copy()
        let childHistory = root.history.copy()
        childHistory.makeBoardMoveAssumeLegal(childBoard, loc, root.nextPla)
        let node = SearchNode(board: childBoard, history: childHistory, nextPla: root.nextPla.opponent)
        node.graphHash = key
        node.state = .expanded
        node.nnOutput = NNOutput(
            nnHash: Hash128(), whiteWinProb: 0.5, whiteLossProb: 0.5, whiteNoResultProb: 0,
            whiteScoreMean: 0, whiteScoreMeanSq: 1, whiteLead: 0, varTimeLeft: 0,
            shorttermWinlossError: 0, shorttermScoreError: 0,
            policyProbs: [Float](repeating: -1, count: nnXLen * nnYLen + 1),
            policyOptimismUsed: 0, nnXLen: nnXLen, nnYLen: nnYLen, whiteOwnerMap: nil
        )
        node.visits = visits
        node.weightSum = Double(visits)
        nodeTable[key] = node
    }

    /// Forces the path-illegal regeneration branch (T-G4). RinGo's node identity folds the full
    /// positional-superko banned set into the graph-hash key, so ko-divergent positions get DIFFERENT
    /// keys and never merge — the path-illegal guard is therefore practically unreachable by natural
    /// ko play (a finding: it survives as defense against 128-bit key collisions and for KataGo
    /// code-path parity). This injects the exact merge-collision situation the guard exists for: a
    /// merge target whose stored untried move (`loc`, already occupied on its own board) is illegal on
    /// the reaching path, so the first descent into it must regenerate rather than crash.
    func injectPathIllegalNodeForTests(afterMove loc: Loc) {
        let key = childGraphHashFromRoot(loc: loc)
        let childBoard = root.board.copy()
        let childHistory = root.history.copy()
        childHistory.makeBoardMoveAssumeLegal(childBoard, loc, root.nextPla)
        let node = SearchNode(board: childBoard, history: childHistory, nextPla: root.nextPla.opponent)
        node.graphHash = key
        node.state = .expanded
        node.nnOutput = syntheticOutput()
        // `loc` is now occupied on this board, so it is illegal — but it sits in untriedMoves as if a
        // different-legality path had recorded it. visits 0 forces descent (not catch-up) into it.
        node.untriedMoves = [(loc, 1.0)]
        nodeTable[key] = node
    }

    /// Forces the cycle guard (T-G5). Cycles need a legal move that merges back to a node already on
    /// the descent path; under RinGo's identity + long-cycle no-result handling this is very rare in
    /// natural play, so this injects a merge target carrying a SELF edge (a one-node cycle): once the
    /// edge catches up and the search descends into the node, selecting the self edge would revisit it
    /// — the guard bails with an edge visit instead of looping.
    func injectCycleNodeForTests(afterMove loc: Loc, selfMove: Loc) {
        let key = childGraphHashFromRoot(loc: loc)
        let childBoard = root.board.copy()
        let childHistory = root.history.copy()
        childHistory.makeBoardMoveAssumeLegal(childBoard, loc, root.nextPla)
        let node = SearchNode(board: childBoard, history: childHistory, nextPla: root.nextPla.opponent)
        node.graphHash = key
        node.state = .expanded
        node.nnOutput = syntheticOutput(legalPolicyAt: policyPos(selfMove, board: childBoard))
        node.visits = 1
        node.weightSum = 1
        // Self edge with edgeVisits kept above `visits` so selection descends (not catch-up) into it.
        node.addChild(loc: selfMove, node: node, edgeVisits: 5)
        nodeTable[key] = node
    }

    // MARK: - WP-GS3 reuse/GC test seams (design §3.3 T-R1..T-R5)

    /// Whether the graph node table currently holds an entry under `hash` — used by T-R5 to assert the
    /// old in-table copy of a promoted position is swept away after a re-root.
    func nodeTableContainsForTests(_ hash: Hash128) -> Bool {
        nodeTable[hash] != nil
    }

    /// The graph hash the root's child at `loc` is stored under (nil if `loc` is not a root child).
    /// Captured before a re-root so a test can look the old position up in the table afterward (T-R5).
    func graphHashOfRootChildForTests(loc: Loc) -> Hash128? {
        root.childNode(for: loc)?.graphHash
    }

    /// Arms the ARC-leak probe with WEAK references to every current table node (T-R2/T-R4). Read back
    /// with `tableLivenessAliveCountForTests()` after a re-root/clear: a leak-free sweep frees every
    /// removed node, so the alive count drops to exactly the surviving (reachable) set.
    func armTableLivenessProbeForTests() {
        livenessProbe = WeakNodeBox(nodeTable.values)
    }

    /// How many of the nodes captured by `armTableLivenessProbeForTests()` are still alive.
    func tableLivenessAliveCountForTests() -> Int {
        livenessProbe?.aliveCount ?? 0
    }

    /// Number of distinct nodes reachable from the current root (root included), deduped by identity.
    /// For a collision-free fixture this equals `nodeTableCountForTests() + 1` (the root is never
    /// table-inserted), which T-R2 asserts to pin "table == reachable" after a sweep.
    func reachableNodeCountForTests() -> Int {
        reachableNodes().count
    }

    /// T-R3/T-R5 seam: after a search, inject into the root's played-reply child (`replyLoc`) one
    /// extra child EDGE whose move is illegal on the REAL advanced position, so the strict root-edge
    /// re-filter must drop it on promotion. `replyLoc` itself is the illegal grandchild move (it is
    /// occupied once the reply is played), pointing at a throwaway expanded node. Returns the child's
    /// pre-injection `visits` (what the filtered new-root visits must recompute back to) plus its graph
    /// hash, or nil if `replyLoc` is not a root child.
    func injectIllegalPromotedEdgeForTests(
        replyLoc: Loc,
        edgeVisits: Int64 = 3
    ) -> (visits: Int64, graphHash: Hash128)? {
        guard let child = root.childNode(for: replyLoc) else { return nil }
        let dummy = SearchNode(
            board: child.board.copy(), history: child.history.copy(), nextPla: child.nextPla.opponent
        )
        dummy.state = .expanded
        dummy.nnOutput = syntheticOutput()
        dummy.visits = edgeVisits
        dummy.weightSum = Double(edgeVisits)
        child.addChild(loc: replyLoc, node: dummy, edgeVisits: edgeVisits)
        return (child.visits, child.graphHash)
    }

    /// T-G6-ext (§B4) seam: plant a root child at `replyLoc` whose stored board/history come from a
    /// TRANSPOSED move order — identical final position + fingerprint to the real advanced history but
    /// a DIFFERENT `koHashHistory` — carrying `visits` already accumulated. This is exactly the case
    /// where the old `koHashHistory`-equality gate would WRONGLY reject a valid transposition; graph-safe
    /// promotion must accept it via `graphFingerprintMatches`. Returns the child's graph hash (its
    /// table key) so a test can assert the old copy is swept after the re-root.
    @discardableResult
    func injectTransposedReplyChildForTests(
        replyLoc: Loc,
        board: Board,
        history: BoardHistory,
        visits: Int64
    ) -> Hash128 {
        let child = SearchNode(
            board: board.copy(), history: history.copy(), nextPla: history.presumedNextMovePla
        )
        child.state = .expanded
        child.nnOutput = syntheticOutput()
        child.visits = visits
        child.weightSum = Double(visits)
        let hash = childGraphHashFromRoot(loc: replyLoc)
        child.graphHash = hash
        root.addChild(loc: replyLoc, node: child, edgeVisits: visits)
        nodeTable[hash] = child
        return hash
    }

    /// A minimal expanded-node NN output for injected test nodes: flat draw value, all-illegal policy
    /// except an optional single legal position.
    private func syntheticOutput(legalPolicyAt position: Int? = nil) -> NNOutput {
        var policy = [Float](repeating: -1, count: nnXLen * nnYLen + 1)
        if let position { policy[position] = 1.0 }
        return NNOutput(
            nnHash: Hash128(), whiteWinProb: 0.5, whiteLossProb: 0.5, whiteNoResultProb: 0,
            whiteScoreMean: 0, whiteScoreMeanSq: 1, whiteLead: 0, varTimeLeft: 0,
            shorttermWinlossError: 0, shorttermScoreError: 0, policyProbs: policy,
            policyOptimismUsed: 0, nnXLen: nnXLen, nnYLen: nnYLen, whiteOwnerMap: nil
        )
    }

    /// The graph hash the root's child at `loc` will be keyed under, computed exactly as the search
    /// does (root hash from scratch, then one chained move).
    private func childGraphHashFromRoot(loc: Loc) -> Hash128 {
        let rootHash = GraphHash.fromScratch(
            hist: root.history, board: root.board, nextPla: root.nextPla,
            repBound: settings.graphSearchRepBound, drawEquivWinsForWhite: settings.drawEquivalentWinsForWhite
        )
        let childBoard = root.board.copy()
        let childHistory = root.history.copy()
        childHistory.makeBoardMoveAssumeLegal(childBoard, loc, root.nextPla)
        return GraphHash.graphHash(
            prev: rootHash, hist: childHistory, board: childBoard, prevMoveLoc: loc,
            nextPla: root.nextPla.opponent, repBound: settings.graphSearchRepBound,
            drawEquivWinsForWhite: settings.drawEquivalentWinsForWhite
        )
    }

    /// All nodes reachable from the root, deduped by identity (cycle-safe).
    private func reachableNodes() -> [SearchNode] {
        var seen: Set<ObjectIdentifier> = [ObjectIdentifier(root)]
        var stack = [root]
        var result = [SearchNode]()
        while let node = stack.popLast() {
            result.append(node)
            for child in node.children where seen.insert(ObjectIdentifier(child.node)).inserted {
                stack.append(child.node)
            }
        }
        return result
    }

    /// Non-mutating twin of `recomputeFromChildren`'s utility output, for staleness measurement.
    private func recomputedUtilityWithoutMutation(_ node: SearchNode) -> Double {
        guard let nnOutput = node.nnOutput else { return node.utilityAvg }
        let ownUtility = utility(of: leafValue(from: nnOutput))
        var weightSum = 1.0
        var utilityAcc = ownUtility
        for (_, child, edgeVisits) in node.children {
            guard edgeVisits > 0, child.visits > 0, child.weightSum > 0 else { continue }
            let weight = child.weightSum * Double(edgeVisits) / Double(max(child.visits, 1))
            weightSum += weight
            utilityAcc += child.utilityAvg * weight
        }
        return utilityAcc / weightSum
    }

    /// Advances the root by one legal move, reusing the played child's retained subtree (visit
    /// counts, values, expanded children — "tree reuse" across `play`/`genmove`) when three
    /// conditions all hold: reuse is enabled (`settings.treeReuseEnabled`), the move does not land
    /// on an already-game-ending position, and the retained child's full legality-relevant context
    /// matches the freshly-advanced position (`canReuseSubtree`). Otherwise the tree is dropped and
    /// a fresh unevaluated root is installed.
    ///
    /// - Game-ending drop: a child explored during search is a plain terminal node (scored via
    ///   `BoardHistory` directly, no NN expansion), not a root with `forceNonTerminal` semantics,
    ///   so it cannot correctly serve as the next search's root.
    /// - Superko-safe key (correctness invariant #1): the retained subtree's search assumed a
    ///   specific ko/superko history. In this port every tree node carries the full `Board`/
    ///   `BoardHistory` it was built from (via `copy()` + `makeBoardMoveAssumeLegal`), so in the
    ///   normal move flow the child's history is byte-identical to the freshly-advanced one and the
    ///   guard always passes — reuse stays bit-identical to the pre-flag behavior. The guard is the
    ///   verified safety net that rejects any position whose position-hash matches but whose
    ///   positional-superko history (`koHashHistory`) differs, so a re-rooted subtree can never
    ///   inherit legality assumptions (ko/superko bans) that don't hold for the real game history.
    /// - Memory (invariant #4): re-rooting drops the old root, which is the sole owner of every
    ///   non-played sibling subtree, so ARC frees them immediately. `SearchNode` holds only
    ///   downward child references (no parent pointer), so the node graph is a strict tree with no
    ///   retain cycles; the retained subtree is bounded by the game's move count times per-move
    ///   new visits.
    public func makeMove(_ loc: Loc) throws {
        let movePla = root.nextPla
        guard root.history.isLegal(root.board, loc, movePla) else {
            throw SearchError("Illegal move \(Location.toString(loc, root.board.xSize, root.board.ySize))")
        }
        let newBoard = root.board.copy()
        let newHistory = root.history.copy()
        newHistory.makeBoardMoveAssumeLegal(newBoard, loc, movePla)
        let nextPla = movePla.opponent
        if settings.useGraphSearch {
            promoteGraphRoot(loc: loc, newBoard: newBoard, newHistory: newHistory, nextPla: nextPla)
            return
        }
        let reusableChild = reusableChild(for: loc, board: newBoard, history: newHistory, nextPla: nextPla)
        root = reusableChild ?? SearchNode(board: newBoard, history: newHistory, nextPla: nextPla)
        rootPolicyProbs = nil
    }

    /// The played child's subtree if it may serve as the next search's root (reuse enabled, the
    /// move does not end the game, and its full legality-relevant context matches the freshly
    /// advanced position — see `canReuseSubtree`); otherwise `nil`, forcing a fresh root.
    private func reusableChild(
        for loc: Loc,
        board: Board,
        history: BoardHistory,
        nextPla: Player
    ) -> SearchNode? {
        guard settings.treeReuseEnabled,
              !history.isGameFinished,
              let child = root.childNode(for: loc),
              Self.canReuseSubtree(
                  childBoard: child.board, childHistory: child.history, childNextPla: child.nextPla,
                  board: board, history: history, nextPla: nextPla
              )
        else { return nil }
        return child
    }

    /// Superko-safe subtree-reuse predicate (correctness invariant #1). A retained child may serve
    /// as the next search's root only if its *entire* legality-relevant context matches the
    /// freshly-advanced position — not merely the stones on the board. Move-sequence identity alone
    /// is insufficient: the same board can arise via different histories that differ in
    /// positional-superko state, and a re-rooted subtree was searched assuming the child's specific
    /// history, so reusing it under a different history could honor (or omit) ko/superko bans that
    /// don't apply to the real game.
    ///
    /// The key compared here is Zobrist/`BoardHistory`-hash based, per the review bar:
    /// - `getSitHashWithSimpleKo` — position (`posHash`) + simple-ko point + side to move.
    /// - `koHashHistory` — the full positional-superko history (element-wise); this is the field
    ///   that distinguishes same-board-different-history positions and is why the check cannot be
    ///   reduced to a single position hash.
    /// - `koRecapBlockHash` + `encorePhase` — encore/territory ko-recap-block state.
    /// - `consecutiveEndingPasses` — pass-termination state (affects whether a child pass is a
    ///   terminal node).
    ///
    /// Stricter than strictly necessary for the current GTP/self-play flow (where it always passes),
    /// but the comparison is O(history length) once per move — negligible against the batched NN
    /// evaluations a single search performs.
    static func canReuseSubtree(
        childBoard: Board,
        childHistory: BoardHistory,
        childNextPla: Player,
        board: Board,
        history: BoardHistory,
        nextPla: Player
    ) -> Bool {
        childNextPla == nextPla
            && childBoard.getSitHashWithSimpleKo(childNextPla) == board.getSitHashWithSimpleKo(nextPla)
            && childHistory.koHashHistory == history.koHashHistory
            && childHistory.koRecapBlockHash == history.koRecapBlockHash
            && childHistory.encorePhase == history.encorePhase
            && childHistory.consecutiveEndingPasses == history.consecutiveEndingPasses
    }

    /// Graph-merge fingerprint (`useGraphSearch`): decides whether a node-table hit is a GENUINE
    /// transposition (merge) or an (astronomically rare) 128-bit hash collision of unrelated
    /// positions (make a private node). Crucially this is NOT `canReuseSubtree`: it deliberately
    /// OMITS the `koHashHistory` comparison, because two positions reached by different move orders
    /// ALWAYS differ in their past-position history even when they are the same node — comparing the
    /// full history would reject every transposition and defeat graph search.
    ///
    /// It is sound to omit history here precisely because RinGo folds the FULL current
    /// positional-superko banned set (`superKoBanned`) into the graph-hash KEY itself (via
    /// `BoardHistory.getSituationRulesAndKoHash`), so any two positions with differing current
    /// legality already receive DIFFERENT keys and never reach this comparison. What remains to
    /// verify is only that the current situation is structurally identical (board + side + simple ko
    /// + ko-recap-block + encore + pass state) — guarding the 128-bit collision. Any legality
    /// divergence that could survive a key match is the (practically unreachable) collision case the
    /// descent-time path-legality guard exists to catch.
    static func graphFingerprintMatches(
        lhsBoard: Board,
        lhsHistory: BoardHistory,
        lhsNextPla: Player,
        rhsBoard: Board,
        rhsHistory: BoardHistory,
        rhsNextPla: Player
    ) -> Bool {
        lhsNextPla == rhsNextPla
            && lhsBoard.getSitHashWithSimpleKo(lhsNextPla) == rhsBoard.getSitHashWithSimpleKo(rhsNextPla)
            && lhsHistory.koRecapBlockHash == rhsHistory.koRecapBlockHash
            && lhsHistory.encorePhase == rhsHistory.encorePhase
            && lhsHistory.consecutiveEndingPasses == rhsHistory.consecutiveEndingPasses
    }

    /// Runs PUCT search from the current root for `maxVisits` playouts, then samples the final root
    /// move using `selectionTemperature` (or the setting when omitted). Both root noise and final
    /// sampling default off, retaining the original stable argmax-visits GTP behavior.
    ///
    /// The loop is fully sequential (see the type doc comment): collect a batch of NN-eval leaves,
    /// await one `evaluate`, back the results up, repeat. `completedVisits` counts both terminal
    /// playouts (resolved during collection) and NN-eval playouts (resolved after backup). The
    /// visit limit is exact; an optional deadline is checked only before collecting another batch.
    /// - Parameter useLcbForSelection: WP-11. When `true`, the FINAL root move is chosen by Lower
    ///   Confidence Bound (`SearchLCB`) instead of the visit distribution — used only by GTP
    ///   `genmove`. Defaults `false`, so self-play, teacher labeling, pondering, and every existing
    ///   test keep bit-identical visit-count selection AND an unchanged recorded `visitsByMove`
    ///   (LCB in self-play degrades training data — Leela Zero #2411).
    public func runSearch(
        maxVisits: Int64,
        timeBudget: TimeInterval? = nil,
        selectionTemperature: Double? = nil,
        useLcbForSelection: Bool = false
    ) async throws -> SearchResult {
        guard maxVisits >= 1 else {
            throw SearchError("maxVisits must be >= 1, got \(maxVisits)")
        }
        if let timeBudget, !timeBudget.isFinite || timeBudget < 0 {
            throw SearchError("timeBudget must be finite and >= 0, got \(timeBudget)")
        }
        let finalTemperature = selectionTemperature ?? settings.selectionTemperature
        guard finalTemperature.isFinite, finalTemperature >= 0 else {
            throw SearchError("selectionTemperature must be finite and >= 0, got \(finalTemperature)")
        }
        let startTime = ProcessInfo.processInfo.systemUptime
        let rootVisitsBefore = root.visits
        sqrtBoardArea = root.board.sqrtBoardArea()
        try await computeRecentScoreCenter()
        prepareRootNoiseIfNeeded()

        let batchTarget = max(1, min(settings.leafBatchSize, settings.numSearchWorkers))
        if settings.useGraphSearch {
            // Bootstrap the root's graph hash from scratch (KataGo `search.cpp:712-714`); the root is
            // never table-inserted but its hash seeds every child's chained hash. `warnedIllegalHashes`
            // is per-search. Then run the DAG playout loop (with evaluator-throw rollback).
            root.graphHash = GraphHash.fromScratch(
                hist: root.history,
                board: root.board,
                nextPla: root.nextPla,
                repBound: settings.graphSearchRepBound,
                drawEquivWinsForWhite: settings.drawEquivalentWinsForWhite
            )
            warnedIllegalHashes.removeAll(keepingCapacity: true)
            try await runGraphPlayoutLoop(
                batchTarget: batchTarget,
                maxVisits: maxVisits,
                timeBudget: timeBudget,
                startTime: startTime
            )
        } else {
            var completedVisits: Int64 = 0
            while completedVisits < maxVisits {
                // Cooperative cancellation for background pondering (WP-10): a ponder search runs on an
                // unstructured `Task` that the GTP loop cancels the instant the next command arrives.
                // Checked only between batches (visit-loop granularity), identical to the deadline
                // below — an in-flight batch always finishes collecting, evaluating, and backing up
                // before this fires, so the tree is never left with dangling `.evaluating` nodes and a
                // subsequent `makeMove` re-root can safely harvest the pondered subtree. For every
                // non-pondering caller (`genmove`, self-play, teacher labeling, tests) the enclosing
                // task is never cancelled mid-search, so `Task.isCancelled` is always false here and the
                // loop stays bit-identical to the pre-ponder behavior.
                if Task.isCancelled { break }
                // The deadline is deliberately checked only between batches. Once collection starts,
                // every leaf in that batch is evaluated and backed up before search returns.
                if let timeBudget, ProcessInfo.processInfo.systemUptime - startTime >= timeBudget {
                    break
                }
                let batch = collectBatch(target: batchTarget, budget: maxVisits - completedVisits)
                completedVisits += batch.terminalsApplied
                guard !batch.requests.isEmpty else {
                    // No leaf needed evaluation this pass. If no terminal was applied either, the tree
                    // is wedged (only reachable if a previous `runSearch` threw mid-batch and left nodes
                    // `.evaluating`); break rather than spin forever. A healthy tree always makes
                    // progress on the first walk of a pass, so this never fires during a normal search.
                    if batch.terminalsApplied == 0 { break }
                    continue
                }
                let outputs = try await evaluateWithFallback(batch.requests)
                for (path, output) in zip(batch.paths, outputs) {
                    applyEvalResult(output, path: path)
                }
                completedVisits += Int64(batch.requests.count)
            }
        }
        let timeUsed = ProcessInfo.processInfo.systemUptime - startTime
        return buildResult(
            visitsAchieved: root.visits - rootVisitsBefore,
            timeUsed: timeUsed,
            selectionTemperature: finalTemperature,
            useLcbForSelection: useLcbForSelection
        )
    }

    // MARK: - Leaf batch collection

    private struct CollectedBatch {
        var paths: [[SearchNode]] = []
        var requests: [NNRequest] = []
        var terminalsApplied: Int64 = 0
    }

    /// Walks the tree, backing up terminal playouts immediately and accumulating up to `target`
    /// distinct leaves that need a fresh NN eval (each gets virtual loss along its path so the next
    /// walk in this pass diverges toward a different leaf). Stops at `target` leaves, when the
    /// per-pass `budget` of playouts (terminals + collected leaves) is exhausted, or when a walk
    /// re-selects a leaf already in this batch (`.abandoned`). Guaranteed to make progress on a
    /// healthy tree: the first walk of a pass never returns `.abandoned`, because no node is
    /// `.evaluating` at pass start (every leaf collected in the previous pass was backed up to
    /// `.expanded`).
    private func collectBatch(target: Int, budget: Int64) -> CollectedBatch {
        var batch = CollectedBatch()
        while batch.requests.count < target, batch.terminalsApplied + Int64(batch.requests.count) < budget {
            switch descend() {
            case .terminalApplied:
                batch.terminalsApplied += 1
            case .abandoned:
                return batch
            case let .needsEval(path, request):
                for node in path.dropFirst() {
                    node.virtualLosses += 1
                }
                batch.paths.append(path)
                batch.requests.append(request)
            }
        }
        return batch
    }

    /// fp16 resilience (conventions.md): a non-finite policy sum only shows up on synthetic
    /// test-only checkpoints, never on real trained nets, but `NNOutputPostprocessor` throws
    /// rather than crashing when it happens — retry once at fp32, scoped to just the offending
    /// position, and log one warning. A batch-level throw loses every result in that physical
    /// batch (postprocessing failure isn't isolated per-row inside `NNEvaluator.evaluate`), so on
    /// failure this re-issues the batch as individual single-item calls against the primary
    /// (fp16) evaluator to isolate which position is bad, then retries only that one at fp32.
    private func evaluateWithFallback(_ requests: [NNRequest]) async throws -> [NNOutput] {
        do {
            return try await evaluator.evaluate(requests)
        } catch {
            // Already fp32 — retrying at the same precision would just fail identically.
            guard primaryPrecision != .float32 else { throw error }
            var outputs = [NNOutput]()
            outputs.reserveCapacity(requests.count)
            for request in requests {
                try await outputs.append(singleWithFallback(request))
            }
            return outputs
        }
    }

    private func singleWithFallback(_ request: NNRequest) async throws -> NNOutput {
        do {
            return try await evaluator.evaluate(request)
        } catch {
            FileHandle.standardError.write(Data(
                "Warning: NN eval produced a non-finite policy at fp16; retrying this position at fp32 (\(error))\n"
                    .utf8
            ))
            let fallback = try fp32FallbackEvaluator()
            return try await fallback.evaluate(request)
        }
    }

    private func fp32FallbackEvaluator() throws -> any NNEvaluating {
        if let fp32Fallback { return fp32Fallback }
        guard let fp32FallbackFactory else {
            throw SearchError("No fp32 fallback evaluator factory is configured")
        }
        let created = try fp32FallbackFactory()
        fp32Fallback = created
        return created
    }

    /// Builds the `MiscNNInputParams` for one leaf/root NN request. In `.random` symmetry mode this
    /// draws exactly one dihedral symmetry (0...7) from the Search-owned `symmetryRandom` stream and
    /// carries it on `params.symmetry`, so the evaluator applies that orientation to the fp16 primary
    /// AND any fp32 fallback of the SAME request — one draw per node, then persisted with the node's
    /// nnOutput (and across tree reuse). `.off`/`.hash` leave `symmetry` unset
    /// (`symmetryNotSpecified`) so the evaluator's own mode logic runs. Non-square NN tensors are not
    /// drawn (the evaluator would ignore the carried value), keeping the stream position independent
    /// of board squareness. `preWarm` requests are built outside `Search` and never route through
    /// here, so warmup consumes no draws and cannot shift the sequence.
    private func leafInputParams() -> MiscNNInputParams {
        var params = MiscNNInputParams(
            drawEquivalentWinsForWhite: settings.drawEquivalentWinsForWhite,
            policyOptimism: settings.policyOptimism
        )
        if symmetryMode == .random, nnXLen == nnYLen {
            params.symmetry = Int(symmetryRandom.next() & 7)
        }
        return params
    }

    /// KataGo `rootNumSymmetriesToSample > 1`, gated to a square NN tensor (the only shape the
    /// dihedral transforms are defined for). When false, every root eval takes the ordinary
    /// single-symmetry path and the root behaves byte-identically to today.
    private var averagesRootSymmetries: Bool {
        settings.rootNumSymmetries > 1 && nnXLen == nnYLen
    }

    /// The root NN output averaged over `k = min(settings.rootNumSymmetries, 8)` DISTINCT dihedral
    /// symmetries drawn WITHOUT replacement from the Search-owned `symmetryRandom` stream (partial
    /// Fisher-Yates over `0..<8`, mirroring `$KG/cpp/neuralnet/nneval.cpp:811-816`). Each symmetry is
    /// evaluated as its own NN request carrying that orientation on `params.symmetry`; the evaluator
    /// inverse-permutes every result back to canonical orientation, so the `k` outputs are directly
    /// averaged by `NNOutput.averaged` (`$KG/cpp/neuralnet/nninputs.cpp:324-434`). RinGo's
    /// `NNEvaluator` has no eval cache, so `k` distinct requests cannot collapse onto one cached
    /// orientation — the KataGo `skipCacheThisIteration` guard is unnecessary here. Only called when
    /// `averagesRootSymmetries` is true. The per-request params otherwise match `leafInputParams`
    /// (root optimism/drawEquivalent), but the symmetry is set explicitly here rather than drawn by
    /// `leafInputParams`, so `.random` mode's per-node draw never fires at an averaged root (KataGo's
    /// root averaging and `nnRandomize` are mutually exclusive at the root).
    private func averagedRootNNOutput() async throws -> NNOutput {
        let k = min(settings.rootNumSymmetries, Symmetry.count)
        var indexes = Array(0 ..< Symmetry.count)
        var requests = [NNRequest]()
        requests.reserveCapacity(k)
        for i in 0 ..< k {
            let j = i + Int(symmetryRandom.next() % UInt64(Symmetry.count - i))
            indexes.swapAt(i, j)
            var params = MiscNNInputParams(
                drawEquivalentWinsForWhite: settings.drawEquivalentWinsForWhite,
                policyOptimism: settings.policyOptimism
            )
            params.symmetry = indexes[i]
            requests.append(NNRequest(
                board: root.board,
                history: root.history,
                nextPlayer: root.nextPla,
                params: params
            ))
        }
        let outputs = try await evaluateWithFallback(requests)
        return NNOutput.averaged(outputs)
    }

    // MARK: - Tree descent (synchronous, actor-isolated)

    private enum DescendOutcome {
        case terminalApplied
        case abandoned
        case needsEval(path: [SearchNode], request: NNRequest)
    }

    /// Walks from the root to a leaf, applying (and immediately backing up) terminal values
    /// along the way, and returns either a completed terminal resolution or the leaf/path that
    /// needs a fresh NN eval. The root is always treated as non-terminal ("forceNonTerminal"),
    /// matching KataGo's rule that a position handed to search must remain searchable even if two
    /// passes would otherwise end the game right there (`Search::playoutDescend`'s
    /// `!node.forceNonTerminal` guard) — `makeMove` additionally never reuses a subtree that
    /// already landed on a finished position as a new root, so no non-root node should ever need
    /// this rule applied. Friendly-pass/conservative-pass forceNonTerminal variants (deeper than
    /// the root) are out of scope (see SearchSettings.swift's "Skipped" list).
    private func descend() -> DescendOutcome {
        var path = [root]
        var current = root
        while true {
            let isCurrentRoot = current === root
            if current.history.isGameFinished, !isCurrentRoot {
                applyBackup(path: path, value: terminalLeafValue(current.history))
                return .terminalApplied
            }
            switch current.state {
            case .unevaluated:
                current.state = .evaluating
                let request = NNRequest(
                    board: current.board,
                    history: current.history,
                    nextPlayer: current.nextPla,
                    params: leafInputParams()
                )
                return .needsEval(path: path, request: request)
            case .evaluating:
                return .abandoned
            case .expanded:
                guard let child = selectChild(current, isRoot: isCurrentRoot) else {
                    // No legal move at all was ever recorded (shouldn't happen: pass is always a
                    // legal candidate). Fall back to the node's own NN eval as its value, mirroring
                    // `Search::playoutDescend`'s `bestChildIdx <= -1` branch.
                    applyBackup(path: path, value: leafValue(from: current.nnOutput!))
                    return .terminalApplied
                }
                path.append(child)
                current = child
            }
        }
    }

    /// PUCT child selection: `Search::selectBestChildToDescend` /
    /// `searchexplorehelpers.cpp`, reduced to this port's scope (pure tree — no graph search, no
    /// eval cache, no wide-root-noise/human-SL/hint-loc/futile-visits/anti-mirror hacks; see
    /// SearchSettings.swift's "Skipped" list). Returns the existing or newly-created child to
    /// descend into.
    private func selectChild(_ node: SearchNode, isRoot: Bool) -> SearchNode? {
        guard let nnOutput = node.nnOutput else { return nil }
        let policyProbs = isRoot ? rootPolicyProbs ?? nnOutput.policyProbs : nnOutput.policyProbs

        var policyProbMassVisited = 0.0
        var totalChildWeight = 0.0
        for (loc, child, _) in node.children {
            let prob = policyProbs[policyPos(loc, board: node.board)]
            guard prob >= 0 else { continue }
            policyProbMassVisited += Double(prob)
            totalChildWeight += Double(child.visits)
        }

        let (fpuValue, stdevFactor) = fpuValueAndStdevFactor(
            node: node,
            isRoot: isRoot,
            policyProbMassVisited: policyProbMassVisited
        )
        let exploreScaling = cpuctExploration(totalChildWeight) * (totalChildWeight + 0.01).squareRoot() * stdevFactor
        let utilityRadius = settings.winLossUtilityFactor + settings.staticScoreUtilityFactor
            + settings.dynamicScoreUtilityFactor

        var bestScore = -Double.infinity
        var bestChild: SearchNode?
        for (loc, child, _) in node.children {
            let prob = Double(policyProbs[policyPos(loc, board: node.board)])
            guard prob >= 0 else { continue }
            var childWeight = Double(child.visits)
            var childUtility = child.visits <= 0 ? fpuValue : child.utilityAvg
            if child.virtualLosses > 0 {
                let virtualLossWeight = Double(child.virtualLosses) * settings.numVirtualLossesPerThread
                let virtualLossUtility = node.nextPla == .white ? -utilityRadius : utilityRadius
                let frac = virtualLossWeight / (virtualLossWeight + max(0.25, childWeight))
                childUtility += (virtualLossUtility - childUtility) * frac
                childWeight += virtualLossWeight
            }
            let explore = exploreScaling * prob / (1 + childWeight)
            let value = node.nextPla == .white ? childUtility : -childUtility
            let score = explore + value
            if score > bestScore {
                bestScore = score
                bestChild = child
            }
        }

        if let (loc, prob) = node.untriedMoves.first {
            let value = node.nextPla == .white ? fpuValue : -fpuValue
            let score = exploreScaling * Double(prob) + value
            if score > bestScore {
                node.untriedMoves.removeFirst()
                let childBoard = node.board.copy()
                let childHistory = node.history.copy()
                childHistory.makeBoardMoveAssumeLegal(childBoard, loc, node.nextPla)
                let child = SearchNode(board: childBoard, history: childHistory, nextPla: node.nextPla.opponent)
                node.addChild(loc: loc, node: child)
                return child
            }
        }
        return bestChild
    }

    /// `Search::getFpuValueForChildrenAssumeVisited` (`searchexplorehelpers.cpp`), minus the
    /// `fpuParentWeight`/`fpuLossProp` blends (both default to 0 / KataGo's GTP config leaves
    /// them at the constructor default, so this port omits the fields entirely — "keep simple"
    /// per the task spec).
    private func fpuValueAndStdevFactor(
        node: SearchNode,
        isRoot: Bool,
        policyProbMassVisited: Double
    ) -> (fpuValue: Double, stdevFactor: Double) {
        let visits = node.visits
        let weightSum = node.weightSum
        let parentUtility = node.utilityAvg
        let variancePrior = settings.cpuctUtilityStdevPrior * settings.cpuctUtilityStdevPrior
        let variancePriorWeight = settings.cpuctUtilityStdevPriorWeight

        let parentUtilityStdev: Double
        if visits <= 0 || weightSum <= 1 {
            parentUtilityStdev = settings.cpuctUtilityStdevPrior
        } else {
            let utilitySq = parentUtility * parentUtility
            let utilitySqAvg = max(node.utilitySqAvg, utilitySq)
            parentUtilityStdev = (
                ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum)
                    / (variancePriorWeight + weightSum - 1.0) - utilitySq
            ).clampedNonNegative().squareRoot()
        }
        let stdevFactor = 1.0
            + settings.cpuctUtilityStdevScale * (parentUtilityStdev / settings.cpuctUtilityStdevPrior - 1.0)

        var parentUtilityForFPU = parentUtility
        if settings.fpuParentWeightByVisitedPolicy {
            let avgWeight = min(1.0, pow(policyProbMassVisited, settings.fpuParentWeightByVisitedPolicyPow))
            parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * utility(ofNNOutput: node.nnOutput!)
        }
        let fpuReductionMax = isRoot ? settings.rootFpuReductionMax : settings.fpuReductionMax
        let reduction = fpuReductionMax * policyProbMassVisited.squareRoot()
        let fpuValue = node.nextPla == .white ? parentUtilityForFPU - reduction : parentUtilityForFPU + reduction
        return (fpuValue, stdevFactor)
    }

    private func cpuctExploration(_ totalChildWeight: Double) -> Double {
        settings.cpuctExploration
            + settings.cpuctExplorationLog
            * log((totalChildWeight + settings.cpuctExplorationBase) / settings.cpuctExplorationBase)
    }

    private func policyPos(_ loc: Loc, board: Board) -> Int {
        NNPos.locToPos(loc, board.xSize, nnXLen, nnYLen)
    }

    // MARK: - Utility (Search::getResultUtility / getScoreUtility, searchhelpers.cpp)

    private func resultUtility(winLossValue: Double, noResultValue: Double) -> Double {
        winLossValue * settings.winLossUtilityFactor + noResultValue * settings.noResultUtilityForWhite
    }

    private func scoreUtility(scoreMean: Double, scoreMeanSq: Double) -> Double {
        let scoreStdev = ScoreValueUtils.getScoreStdev(scoreMean, scoreMeanSq)
        let staticValue = ScoreValueUtils.expectedWhiteScoreValue(
            whiteScoreMean: scoreMean,
            whiteScoreStdev: scoreStdev,
            center: 0.0,
            scale: 2.0,
            sqrtBoardArea: sqrtBoardArea
        )
        let dynamicValue = ScoreValueUtils.expectedWhiteScoreValue(
            whiteScoreMean: scoreMean,
            whiteScoreStdev: scoreStdev,
            center: recentScoreCenter,
            scale: settings.dynamicScoreCenterScale,
            sqrtBoardArea: sqrtBoardArea
        )
        return staticValue * settings.staticScoreUtilityFactor + dynamicValue * settings.dynamicScoreUtilityFactor
    }

    private func utility(of value: LeafValue) -> Double {
        resultUtility(winLossValue: value.winLossValue, noResultValue: value.noResultValue)
            + scoreUtility(scoreMean: value.scoreMean, scoreMeanSq: value.scoreMeanSq)
    }

    private func utility(ofNNOutput output: NNOutput) -> Double {
        let result = (Double(output.whiteWinProb) - Double(output.whiteLossProb)) * settings.winLossUtilityFactor
            + Double(output.whiteNoResultProb) * settings.noResultUtilityForWhite
        return result + scoreUtility(
            scoreMean: Double(output.whiteScoreMean),
            scoreMeanSq: Double(output.whiteScoreMeanSq)
        )
    }

    private func leafValue(from output: NNOutput) -> LeafValue {
        LeafValue(
            winLossValue: Double(output.whiteWinProb) - Double(output.whiteLossProb),
            noResultValue: Double(output.whiteNoResultProb),
            scoreMean: Double(output.whiteScoreMean),
            scoreMeanSq: Double(output.whiteScoreMeanSq)
        )
    }

    /// Terminal leaf value: `Search::playoutDescend`'s `thread.history.isGameFinished` branch.
    /// `isResignation` never arises inside the tree (resignation is a GTP-level decision applied
    /// to the real game, never a property intrinsic to a searched position), so only the
    /// no-result and scored cases are handled, matching the reference.
    private func terminalLeafValue(_ history: BoardHistory) -> LeafValue {
        if history.isNoResult {
            return LeafValue(winLossValue: 0, noResultValue: 1, scoreMean: 0, scoreMeanSq: 0)
        }
        let winLossValue = 2 * ScoreValueUtils
            .whiteWinsOfWinner(history.winner, settings.drawEquivalentWinsForWhite) - 1
        let finalScore = Double(history.finalWhiteMinusBlackScore)
        let scoreMean = finalScore + Double(history.whiteKomiAdjustmentForDraws(settings.drawEquivalentWinsForWhite))
        let scoreMeanSq = ScoreValueUtils.whiteScoreMeanSqOfScoreGridded(
            finalScore,
            settings.drawEquivalentWinsForWhite
        )
        return LeafValue(winLossValue: winLossValue, noResultValue: 0, scoreMean: scoreMean, scoreMeanSq: scoreMeanSq)
    }

    // MARK: - Backup

    /// Incremental running-mean backup along `path` (root...leaf inclusive). Because this port
    /// always uses leaf weight 1 (no uncertainty weighting — see SearchSettings.swift's "Skipped"
    /// list) and never downweights/prunes/biases children, this reduces to a plain visit-weighted
    /// mean at every node, which is exactly what KataGo's from-scratch
    /// `Search::recomputeNodeStats` recomputes on every visit in the general (graph-search) case
    /// — the incremental update here is algebraically identical, just cheaper, per
    /// SearchSettings.swift's doc comment.
    private func applyBackup(path: [SearchNode], value: LeafValue) {
        let utilityValue = utility(of: value)
        let utilitySq = utilityValue * utilityValue
        for node in path {
            node.accumulateLeaf(
                utility: utilityValue,
                utilitySq: utilitySq,
                scoreMean: value.scoreMean,
                scoreMeanSq: value.scoreMeanSq,
                winLoss: value.winLossValue
            )
        }
    }

    private func applyEvalResult(_ output: NNOutput, path: [SearchNode]) {
        let leaf = path[path.count - 1]
        leaf.nnOutput = output
        leaf.untriedMoves = sortedLegalMoves(
            from: output,
            board: leaf.board,
            history: leaf.history,
            nextPlayer: leaf.nextPla
        )
        leaf.state = .expanded
        applyBackup(path: path, value: leafValue(from: output))
        for node in path.dropFirst() {
            node.virtualLosses -= 1
        }
    }

    private func sortedLegalMoves(
        from output: NNOutput,
        board: Board,
        history: BoardHistory,
        nextPlayer: Player
    ) -> [(loc: Loc, prob: Float)] {
        sortedLegalMoves(
            policyProbs: output.policyProbs,
            board: board,
            history: history,
            nextPlayer: nextPlayer
        )
    }

    private func sortedLegalMoves(
        policyProbs: [Float],
        board: Board,
        history: BoardHistory,
        nextPlayer: Player
    ) -> [(loc: Loc, prob: Float)] {
        let policySize = nnXLen * nnYLen + 1
        var moves = [(loc: Loc, prob: Float)]()
        moves.reserveCapacity(policySize)
        for pos in 0 ..< policySize {
            let prob = policyProbs[pos]
            guard prob >= 0 else { continue }
            let loc = NNPos.posToLoc(pos, board.xSize, board.ySize, nnXLen, nnYLen)
            guard loc != Board.nullLoc, history.isLegal(board, loc, nextPlayer) else { continue }
            moves.append((loc, prob))
        }
        moves.sort { $0.prob > $1.prob }
        return moves
    }

    // MARK: - Graph search (DAG core, useGraphSearch only)

    //
    // Everything below runs ONLY when `settings.useGraphSearch` is true; the pure-tree path above is
    // untouched. Design: docs/strategy/GRAPH_SEARCH_DESIGN.md §2.4-2.7. Reference: KataGo
    // `$KG/cpp/search/search.cpp` / `searchupdatehelpers.cpp`.

    /// One step along a graph-mode descent path: a node plus the index of the edge INTO it within its
    /// parent's `children` array (so backup can bump exactly that edge). `-1` at the root.
    private struct GraphPathStep {
        var node: SearchNode
        var childIndex: Int
    }

    /// A collected item that still needs an NN evaluation round-trip.
    private enum GraphEvalKind {
        /// An unevaluated node to expand.
        case freshLeaf
        /// An expanded node whose policy is stale for this path — regenerate its NN output for the
        /// path's legality context (KataGo `search.cpp:1259-1260`), then bail counting a visit.
        case reeval
    }

    private struct GraphBatch {
        var paths: [[GraphPathStep]] = []
        var requests: [NNRequest] = []
        var kinds: [GraphEvalKind] = []
        /// Playouts already completed during collection (terminal / catch-up / cycle) — no NN eval.
        var immediateApplied: Int64 = 0
    }

    private enum GraphDescendOutcome {
        case immediatelyApplied
        case abandoned
        case needsEval(path: [GraphPathStep], request: NNRequest, kind: GraphEvalKind)
    }

    /// PUCT selection result at a graph-mode node.
    private enum GraphSelection {
        /// No selectable legal move (defensive; pass is always a candidate). Back up own NN value.
        case none
        /// The PUCT-selected move is illegal on the path — regenerate this node's NN output.
        case pathIllegal
        /// The selected existing edge lags the shared child's visits — import one unit (no descent).
        case catchUp(childIndex: Int)
        /// Descend into the selected/created child (which may be a fresh, merged, or private node).
        /// `pathAdvanced` is true when the selection already played `loc` onto the path board/history
        /// (child creation does this to build the child), so the descent must NOT replay it — a
        /// bit-exact removal of the duplicate `makeBoardMoveAssumeLegal` that child creation + descent
        /// otherwise both do for a newly-created child (WP-GS3.5).
        case descend(child: SearchNode, childIndex: Int, loc: Loc, pathAdvanced: Bool)
    }

    /// The graph-mode playout loop, mirroring `runSearch`'s pure-tree loop but collecting/backing up
    /// through the DAG core, and rolling back pending virtual losses / node states on an evaluator
    /// throw (review §B8) so the tree is left quiescent for the next search.
    private func runGraphPlayoutLoop(
        batchTarget: Int,
        maxVisits: Int64,
        timeBudget: TimeInterval?,
        startTime: TimeInterval
    ) async throws {
        var completedVisits: Int64 = 0
        while completedVisits < maxVisits {
            if Task.isCancelled { break }
            if let timeBudget, ProcessInfo.processInfo.systemUptime - startTime >= timeBudget { break }
            let batch = collectGraphBatch(target: batchTarget, budget: maxVisits - completedVisits)
            completedVisits += batch.immediateApplied
            guard !batch.requests.isEmpty else {
                if batch.immediateApplied == 0 { break }
                continue
            }
            let outputs: [NNOutput]
            do {
                outputs = try await evaluateWithFallback(batch.requests)
            } catch {
                // Roll back this batch's pending virtual losses + node states so a subsequent search
                // starts from a quiescent tree (the pure-tree path leaves a wedged tree here; graph
                // sharing broadens the blast radius, so graph mode must undo it — T-G11).
                rollbackGraphBatch(batch)
                throw error
            }
            for index in outputs.indices {
                applyGraphEvalResult(
                    outputs[index],
                    path: batch.paths[index],
                    kind: batch.kinds[index],
                    request: batch.requests[index]
                )
            }
            completedVisits += Int64(batch.requests.count)
        }
    }

    private func collectGraphBatch(target: Int, budget: Int64) -> GraphBatch {
        var batch = GraphBatch()
        while batch.requests.count < target, batch.immediateApplied + Int64(batch.requests.count) < budget {
            switch descendGraph() {
            case .immediatelyApplied:
                batch.immediateApplied += 1
            case .abandoned:
                return batch
            case let .needsEval(path, request, kind):
                for step in path.dropFirst() {
                    step.node.virtualLosses += 1
                }
                batch.paths.append(path)
                batch.requests.append(request)
                batch.kinds.append(kind)
            }
        }
        return batch
    }

    /// One graph-mode playout descent. Carries a per-playout copy of the root board/history and
    /// advances it move by move (KataGo's `SearchThread` board/history/graphPath), so legality and
    /// terminal detection are always against the ACTUAL path — node identity is hash-approximate but
    /// legality is path-exact. Applies terminal/catch-up/cycle playouts in-walk; returns the
    /// leaf/path that needs an NN eval otherwise.
    private func descendGraph() -> GraphDescendOutcome {
        let pathBoard = root.board.copy()
        let pathHistory = root.history.copy()
        var path = [GraphPathStep(node: root, childIndex: -1)]
        var visited: Set<ObjectIdentifier> = [ObjectIdentifier(root)]
        var current = root
        while true {
            let isCurrentRoot = current === root
            if pathHistory.isGameFinished, !isCurrentRoot {
                applyGraphLeafBackup(path: path, value: terminalLeafValue(pathHistory))
                return .immediatelyApplied
            }
            switch current.state {
            case .unevaluated:
                current.state = .evaluating
                let request = NNRequest(
                    board: current.board,
                    history: current.history,
                    nextPlayer: current.nextPla,
                    params: leafInputParams()
                )
                return .needsEval(path: path, request: request, kind: .freshLeaf)
            case .evaluating:
                return .abandoned
            case .expanded:
                let selection = selectChildGraph(
                    current, isRoot: isCurrentRoot, pathBoard: pathBoard, pathHistory: pathHistory
                )
                switch selection {
                case .none:
                    applyGraphLeafBackup(path: path, value: leafValue(from: current.nnOutput!))
                    return .immediatelyApplied
                case .pathIllegal:
                    // Regenerate `current`'s NN output for THIS path (carried on the request), then
                    // bail counting a visit — needs an eval round-trip. Copy the path state so later
                    // walks reusing `pathBoard`/`pathHistory` can't mutate the in-flight request.
                    current.state = .evaluating
                    let request = NNRequest(
                        board: pathBoard.copy(),
                        history: pathHistory.copy(),
                        nextPlayer: current.nextPla,
                        params: leafInputParams()
                    )
                    return .needsEval(path: path, request: request, kind: .reeval)
                case let .catchUp(childIndex):
                    current.incrementEdgeVisits(at: childIndex)
                    applyGraphEdgeBackup(path: path)
                    return .immediatelyApplied
                case let .descend(child, childIndex, loc, pathAdvanced):
                    if visited.contains(ObjectIdentifier(child)) {
                        // Descending would revisit a node already on this path (a cycle) — bail with
                        // an edge visit (KataGo `search.cpp:1393-1414`); the shared node's stats stay.
                        // (A brand-new child from creation can never be on the path, so this only ever
                        // fires for an existing/merged child, whose selection leaves `pathAdvanced` false.)
                        current.incrementEdgeVisits(at: childIndex)
                        graphDiagnostics.cycles += 1
                        applyGraphEdgeBackup(path: path)
                        return .immediatelyApplied
                    }
                    if !pathAdvanced {
                        // Existing-child descent: advance the path by replaying the move. When child
                        // creation already advanced the path (pathAdvanced), replaying would be a
                        // duplicate `makeBoardMoveAssumeLegal` of the identical move — skip it.
                        pathHistory.makeBoardMoveAssumeLegal(pathBoard, loc, current.nextPla)
                    }
                    path.append(GraphPathStep(node: child, childIndex: childIndex))
                    visited.insert(ObjectIdentifier(child))
                    current = child
                }
            }
        }
    }

    /// Edge-weighted PUCT (design §2.4: only `childWeight` and `totalChildWeight` change vs pure
    /// tree — everything else, including the FPU/stdev math, is shared). Also decides child creation
    /// with the collision contract, catch-up, and path-illegal regeneration.
    private func selectChildGraph(
        _ node: SearchNode,
        isRoot: Bool,
        pathBoard: Board,
        pathHistory: BoardHistory
    ) -> GraphSelection {
        guard let nnOutput = node.nnOutput else { return .none }
        let policyProbs = isRoot ? rootPolicyProbs ?? nnOutput.policyProbs : nnOutput.policyProbs

        var policyProbMassVisited = 0.0
        var totalChildWeight = 0.0
        for (loc, child, edgeVisits) in node.children {
            let prob = policyProbs[policyPos(loc, board: node.board)]
            guard prob >= 0 else { continue }
            policyProbMassVisited += Double(prob)
            totalChildWeight += edgeAdjustedWeight(child: child, edgeVisits: edgeVisits)
        }

        let (fpuValue, stdevFactor) = fpuValueAndStdevFactor(
            node: node,
            isRoot: isRoot,
            policyProbMassVisited: policyProbMassVisited
        )
        let exploreScaling = cpuctExploration(totalChildWeight) * (totalChildWeight + 0.01).squareRoot() * stdevFactor
        let utilityRadius = settings.winLossUtilityFactor + settings.staticScoreUtilityFactor
            + settings.dynamicScoreUtilityFactor

        var bestScore = -Double.infinity
        var bestChildIndex = -1
        for index in node.children.indices {
            let (loc, child, edgeVisits) = node.children[index]
            let prob = Double(policyProbs[policyPos(loc, board: node.board)])
            guard prob >= 0 else { continue }
            var childWeight = edgeAdjustedWeight(child: child, edgeVisits: edgeVisits)
            var childUtility = child.visits <= 0 ? fpuValue : child.utilityAvg
            if child.virtualLosses > 0 {
                let virtualLossWeight = Double(child.virtualLosses) * settings.numVirtualLossesPerThread
                let virtualLossUtility = node.nextPla == .white ? -utilityRadius : utilityRadius
                let frac = virtualLossWeight / (virtualLossWeight + max(0.25, childWeight))
                childUtility += (virtualLossUtility - childUtility) * frac
                childWeight += virtualLossWeight
            }
            let explore = exploreScaling * prob / (1 + childWeight)
            let value = node.nextPla == .white ? childUtility : -childUtility
            let score = explore + value
            if score > bestScore {
                bestScore = score
                bestChildIndex = index
            }
        }

        var untriedWins = false
        if let (_, prob) = node.untriedMoves.first {
            let value = node.nextPla == .white ? fpuValue : -fpuValue
            let score = exploreScaling * Double(prob) + value
            if score > bestScore { untriedWins = true }
        }

        if untriedWins {
            let loc = node.untriedMoves[0].loc
            guard pathHistory.isLegal(pathBoard, loc, node.nextPla) else { return .pathIllegal }
            return createChildGraph(node: node, loc: loc, pathBoard: pathBoard, pathHistory: pathHistory)
        }

        guard bestChildIndex >= 0 else { return .none }
        let (loc, child, edgeVisits) = node.children[bestChildIndex]
        guard pathHistory.isLegal(pathBoard, loc, node.nextPla) else { return .pathIllegal }
        if edgeVisits < child.visits {
            return .catchUp(childIndex: bestChildIndex)
        }
        // Existing child: the path has NOT been advanced past `loc` yet, so the descent must replay it.
        return .descend(child: child, childIndex: bestChildIndex, loc: loc, pathAdvanced: false)
    }

    /// Creates (or merges to) the child reached by playing `loc` from `node` on the path, computing
    /// the child's chained `GraphHash` and consulting the node table with a full-state fingerprint:
    /// - table hit + fingerprint match → MERGE: link the shared node and catch-up its lead;
    /// - table hit + fingerprint mismatch → COLLISION: a private, never-merging node (review §B2);
    /// - table miss → a fresh node, inserted under its hash.
    private func createChildGraph(
        node: SearchNode,
        loc: Loc,
        pathBoard: Board,
        pathHistory: BoardHistory
    ) -> GraphSelection {
        node.untriedMoves.removeFirst()
        // Advance the path board/history IN PLACE (was: copy the path, then play `loc` onto the copy,
        // then have the descent replay the identical move onto the path — two `makeBoardMoveAssumeLegal`
        // for one edge). Advancing once and copying the resulting post-move state for a freshly-created
        // child is byte-identical (the move is deterministic) and lets the descent skip its replay via
        // `pathAdvanced: true`. For a merge/collision the post-move path state is exactly what the hash
        // and fingerprint need, so no separate child-state copy is made until a private node is created.
        pathHistory.makeBoardMoveAssumeLegal(pathBoard, loc, node.nextPla)
        let childPla = node.nextPla.opponent
        let childHash = GraphHash.graphHash(
            prev: node.graphHash,
            hist: pathHistory,
            board: pathBoard,
            prevMoveLoc: loc,
            nextPla: childPla,
            repBound: settings.graphSearchRepBound,
            drawEquivWinsForWhite: settings.drawEquivalentWinsForWhite
        )

        if let existing = nodeTable[childHash] {
            if Self.graphFingerprintMatches(
                lhsBoard: existing.board, lhsHistory: existing.history, lhsNextPla: existing.nextPla,
                rhsBoard: pathBoard, rhsHistory: pathHistory, rhsNextPla: childPla
            ) {
                // Real transposition: link the shared node. A brand-new edge (edgeVisits 0) to a node
                // with visits > 0 immediately catches up — importing the shared subtree one visit at
                // a time (KataGo's mechanism). If the shared node has 0 visits (unevaluated), descend.
                node.addChild(loc: loc, node: existing, edgeVisits: 0)
                graphDiagnostics.merges += 1
                let index = node.children.count - 1
                if existing.visits > 0 {
                    return .catchUp(childIndex: index)
                }
                return .descend(child: existing, childIndex: index, loc: loc, pathAdvanced: true)
            }
            let collided = SearchNode(board: pathBoard.copy(), history: pathHistory.copy(), nextPla: childPla)
            collided.graphHash = childHash
            node.addChild(loc: loc, node: collided, edgeVisits: 0)
            graphDiagnostics.collisions += 1
            return .descend(child: collided, childIndex: node.children.count - 1, loc: loc, pathAdvanced: true)
        }

        let fresh = SearchNode(board: pathBoard.copy(), history: pathHistory.copy(), nextPla: childPla)
        fresh.graphHash = childHash
        nodeTable[childHash] = fresh
        node.addChild(loc: loc, node: fresh, edgeVisits: 0)
        return .descend(child: fresh, childIndex: node.children.count - 1, loc: loc, pathAdvanced: true)
    }

    /// The parent's view of a (possibly shared) child's weight: the child's own `weightSum` rescaled
    /// by this edge's share `edgeVisits / max(childVisits, 1)` (KataGo `NodeStats::getChildWeight`,
    /// searchnode.h:64-72). In this port's weight-1 world it equals `edgeVisits`.
    private func edgeAdjustedWeight(child: SearchNode, edgeVisits: Int64) -> Double {
        guard edgeVisits > 0 else { return 0 }
        return child.weightSum * Double(edgeVisits) / Double(max(child.visits, 1))
    }

    /// Recomputes `node`'s averaged stats from scratch over its children (each weighted by its edge
    /// share) plus the node's own NN eval at weight 1, then adds `visitsToAdd` to `visits` (KataGo
    /// `recomputeNodeStats`, searchupdatehelpers.cpp:167-360; `visitsToAdd == 0` is the stats-only
    /// form used by root filtering). The uncertainty/noise-pruning/subtree-bias/pattern/eval-cache
    /// terms are all skipped in this port, so the second-moment consistency correction is identically
    /// zero and omitted. In the weight-1 world `weightSum` stays exactly `Double(visits)`.
    private func recomputeFromChildren(_ node: SearchNode, visitsToAdd: Int64) {
        guard let nnOutput = node.nnOutput else {
            node.visits += visitsToAdd
            return
        }
        // The own-value term is a pure function of `nnOutput` (constant once expanded); its utility also
        // depends on `recentScoreCenter`, constant across this search. Cache both so a node visited by
        // many playouts pays the (~200-atan) scoreUtility exactly once, not once per backup (WP-GS3.5).
        let ownValue: LeafValue
        if let cached = node.cachedSelfLeafValue {
            ownValue = cached
        } else {
            ownValue = leafValue(from: nnOutput)
            node.cachedSelfLeafValue = ownValue
        }
        let ownUtility: Double
        if node.cachedSelfUtilityCenter == recentScoreCenter {
            ownUtility = node.cachedSelfUtility
        } else {
            ownUtility = utility(of: ownValue)
            node.cachedSelfUtility = ownUtility
            node.cachedSelfUtilityCenter = recentScoreCenter
        }
        var weightSum = 1.0
        var winLossAcc = ownValue.winLossValue
        var scoreMeanAcc = ownValue.scoreMean
        var scoreMeanSqAcc = ownValue.scoreMeanSq
        var utilityAcc = ownUtility
        var utilitySqAcc = ownUtility * ownUtility
        for (_, child, edgeVisits) in node.children {
            guard edgeVisits > 0, child.visits > 0, child.weightSum > 0 else { continue }
            let weight = child.weightSum * Double(edgeVisits) / Double(max(child.visits, 1))
            weightSum += weight
            winLossAcc += child.winValueAvg * weight
            scoreMeanAcc += child.scoreMeanAvg * weight
            scoreMeanSqAcc += child.scoreMeanSqAvg * weight
            utilityAcc += child.utilityAvg * weight
            utilitySqAcc += child.utilitySqAvg * weight
        }
        node.winValueAvg = winLossAcc / weightSum
        node.scoreMeanAvg = scoreMeanAcc / weightSum
        node.scoreMeanSqAvg = scoreMeanSqAcc / weightSum
        node.utilityAvg = utilityAcc / weightSum
        node.utilitySqAvg = utilitySqAcc / weightSum
        node.weightSum = weightSum
        node.visits += visitsToAdd
    }

    /// Backup for a leaf/terminal playout: the last node accumulates the leaf observation (its own NN
    /// or the terminal value), then each ancestor bumps the traversed edge and recomputes, leaf→root.
    private func applyGraphLeafBackup(path: [GraphPathStep], value: LeafValue) {
        let utilityValue = utility(of: value)
        let utilitySq = utilityValue * utilityValue
        path[path.count - 1].node.accumulateLeaf(
            utility: utilityValue,
            utilitySq: utilitySq,
            scoreMean: value.scoreMean,
            scoreMeanSq: value.scoreMeanSq,
            winLoss: value.winLossValue
        )
        recomputeAncestors(path: path)
    }

    /// Backup for a catch-up/cycle playout: the last node's catch-up edge was already bumped by the
    /// caller, so recompute it (its edge-sum grew by 1), then the ancestors.
    private func applyGraphEdgeBackup(path: [GraphPathStep]) {
        recomputeFromChildren(path[path.count - 1].node, visitsToAdd: 1)
        recomputeAncestors(path: path)
    }

    /// Bumps every traversed edge (parent→child) and recomputes every ancestor of the path's last
    /// node, leaf→root, adding one visit to each — the "return true" propagation up KataGo's
    /// recursion. The last node itself is handled by the caller.
    private func recomputeAncestors(path: [GraphPathStep]) {
        var index = path.count - 1
        while index >= 1 {
            let childStep = path[index]
            let parent = path[index - 1].node
            parent.incrementEdgeVisits(at: childStep.childIndex)
            recomputeFromChildren(parent, visitsToAdd: 1)
            index -= 1
        }
    }

    private func applyGraphEvalResult(
        _ output: NNOutput,
        path: [GraphPathStep],
        kind: GraphEvalKind,
        request: NNRequest
    ) {
        let leaf = path[path.count - 1].node
        switch kind {
        case .freshLeaf:
            leaf.nnOutput = output
            leaf.untriedMoves = sortedLegalMoves(
                from: output, board: leaf.board, history: leaf.history, nextPlayer: leaf.nextPla
            )
            leaf.state = .expanded
            applyGraphLeafBackup(path: path, value: leafValue(from: output))
        case .reeval:
            // §8.D-1 decision = OVERWRITE the shared node's nnOutput (KataGo-faithful, graph-history
            // interaction approximation). Rebuild untriedMoves against the PATH state (from the
            // request) so only path-legal moves remain. Bail counting a visit: the node's own visits
            // stay put (KataGo's return-true bail does not run updateStatsAfterPlayout on it), only
            // the edge INTO it and the ancestors advance — so the `visits == 1 + Σedge` invariant is
            // preserved everywhere.
            leaf.nnOutput = output
            leaf.invalidateSelfValueCache()
            leaf.untriedMoves = sortedLegalMoves(
                from: output, board: request.board, history: request.history, nextPlayer: leaf.nextPla
            )
            leaf.state = .expanded
            graphDiagnostics.pathIllegal += 1
            if warnedIllegalHashes.insert(output.nnHash).inserted {
                FileHandle.standardError.write(Data(
                    "Warning: graph-search path-illegal selection; regenerated NN output for path legality (nnHash \(output.nnHash))\n"
                        .utf8
                ))
            }
            recomputeAncestors(path: path)
        }
        for step in path.dropFirst() {
            step.node.virtualLosses -= 1
        }
    }

    /// Undoes a batch's pending virtual losses and node states after an evaluator throw (T-G11), so
    /// the next search finds a quiescent tree. Freshly-collected nodes revert to `.unevaluated` (they
    /// get re-collected); regenerated (reeval) nodes revert to `.expanded`.
    private func rollbackGraphBatch(_ batch: GraphBatch) {
        for index in batch.paths.indices {
            let path = batch.paths[index]
            for step in path.dropFirst() {
                step.node.virtualLosses -= 1
            }
            let leaf = path[path.count - 1].node
            switch batch.kinds[index] {
            case .freshLeaf: leaf.state = .unevaluated
            case .reeval: leaf.state = .expanded
            }
        }
    }

    /// Tears down the graph node table, breaking every node's outgoing edges first so a DAG that has
    /// linked transpositions into reference cycles can be freed by ARC (design §2.5-5/6). A no-op in
    /// pure-tree mode (the table is always empty there).
    private func clearGraphTable() {
        for (_, node) in nodeTable {
            node.clearLinks()
        }
        nodeTable.removeAll()
    }

    // MARK: - Graph-safe re-root (WP-GS3, design §2.5, review §B4/§B10)

    /// Re-roots the graph after one legal move (`makeMove` under `useGraphSearch`). Faithful to
    /// KataGo `search.cpp:318-399` (root promotion) + `beginSearch`'s root-child legality filter
    /// (`:744-802`), adapted to this port's private-root-object model:
    ///
    /// 1. The new root is ALWAYS a fresh private `SearchNode` built from the REAL advanced board/
    ///    history — never a shared table node adopted as root (a table node's stored history came
    ///    from whichever path first reached it, so its superko context may diverge from the real
    ///    game; KataGo copies the child out of the table for exactly this reason, `:376-379`).
    /// 2. When reuse is enabled, the move doesn't end the game, and the played child is an expanded
    ///    node whose fingerprint matches the real position, the child's compatible stats + child
    ///    edges + untried moves are transplanted onto the new root. The fingerprint is
    ///    `graphFingerprintMatches` (NOT `canReuseSubtree`): a genuine transposition reached by a
    ///    different order ALWAYS differs in `koHashHistory`, so requiring history equality would
    ///    wrongly reject valid reuse (review §B4). Superko correctness is instead upheld by the
    ///    descent-time path-legality guard plus step 3's strict root-edge re-filter.
    /// 3. The new root's edges are re-checked for legality on the real history; any illegal edge is
    ///    dropped and the root's visits/stats are rebuilt stats-only (`recomputeFromChildren(...,
    ///    visitsToAdd: 0)`, no double-increment — review §B5).
    /// 4. A cycle-aware mark-sweep keeps only nodes reachable from the new root and tears down +
    ///    removes the rest from the table — including the old in-table copy of the promoted position
    ///    whenever no surviving transposition reaches it (review §B10: root-out-of-table alone does
    ///    not guarantee a single identity). When reuse doesn't apply the new root is childless, so
    ///    the sweep reclaims the entire table (equivalent to `clearGraphTable`).
    ///
    /// This is the same code path a search→ponder→next-command sequence takes: pondering grows the
    /// graph on this same actor and `stopPondering()` quiesces it before `makeMove`, so a pondered
    /// (possibly transposed) subtree is promoted here exactly like one grown by consecutive searches.
    private func promoteGraphRoot(loc: Loc, newBoard: Board, newHistory: BoardHistory, nextPla: Player) {
        let newRoot = SearchNode(board: newBoard, history: newHistory, nextPla: nextPla)
        if let child = promotableGraphChild(for: loc, board: newBoard, history: newHistory, nextPla: nextPla) {
            transplantGraphStats(from: child, to: newRoot)
            // Strict root-edge legality re-filter (KataGo "once we are the root, we are strict on
            // legality", `search.cpp:759`). Only recompute when an edge was actually dropped, so an
            // all-legal promoted child keeps its exact transplanted stats (KataGo's `anyFiltered`).
            if newRoot.retainChildren(where: { newHistory.isLegal(newBoard, $0, nextPla) }) {
                var filteredVisits: Int64 = 1
                for (_, _, edgeVisits) in newRoot.children {
                    filteredVisits += edgeVisits
                }
                newRoot.visits = filteredVisits
                recomputeFromChildren(newRoot, visitsToAdd: 0)
            }
        }
        root = newRoot
        rootPolicyProbs = nil
        sweepUnreachableGraphNodes()
    }

    /// The played child's shared node if it may seed the new graph root — reuse enabled, the move
    /// doesn't end the game, and the child is an expanded node whose graph fingerprint matches the
    /// real advanced position. Uses `graphFingerprintMatches` (NOT `canReuseSubtree`): a genuine
    /// transposition reached by another order always differs in `koHashHistory`, so history equality
    /// would wrongly reject valid reuse (review §B4). Otherwise `nil`, forcing a fresh root.
    private func promotableGraphChild(
        for loc: Loc,
        board: Board,
        history: BoardHistory,
        nextPla: Player
    ) -> SearchNode? {
        guard settings.treeReuseEnabled,
              !history.isGameFinished,
              let child = root.childNode(for: loc),
              child.state == .expanded,
              child.nnOutput != nil,
              Self.graphFingerprintMatches(
                  lhsBoard: child.board, lhsHistory: child.history, lhsNextPla: child.nextPla,
                  rhsBoard: board, rhsHistory: history, rhsNextPla: nextPla
              )
        else { return nil }
        return child
    }

    /// Copies the promoted child's graph-compatible state onto the new private root: its backed-up
    /// means, NN output, expanded state, untried moves, and — crucially — its child EDGES (shared
    /// grandchild nodes with their per-edge visit counts). The child node object itself is left
    /// behind (the new root is a separate object built on the real history), to be reclaimed by the
    /// sweep unless a surviving transposition still reaches it. `virtualLosses` is not copied: a
    /// completed/quiesced search leaves the subtree with none. `graphHash` is copied for consistency
    /// but is authoritatively recomputed from scratch at the next `runSearch` (KataGo `:712-714`);
    /// that recompute equals the child's stored key, so the transplanted grandchildren stay findable
    /// under their existing table keys (chain-hash associativity — design §2.5).
    private func transplantGraphStats(from child: SearchNode, to newRoot: SearchNode) {
        newRoot.graphHash = child.graphHash
        newRoot.state = .expanded
        newRoot.nnOutput = child.nnOutput
        newRoot.visits = child.visits
        newRoot.weightSum = child.weightSum
        newRoot.utilityAvg = child.utilityAvg
        newRoot.utilitySqAvg = child.utilitySqAvg
        newRoot.winValueAvg = child.winValueAvg
        newRoot.scoreMeanAvg = child.scoreMeanAvg
        newRoot.scoreMeanSqAvg = child.scoreMeanSqAvg
        newRoot.untriedMoves = child.untriedMoves
        for (childLoc, grandchild, edgeVisits) in child.children {
            newRoot.addChild(loc: childLoc, node: grandchild, edgeVisits: edgeVisits)
        }
    }

    /// Incremental mark-sweep GC (design §2.5-5, review §A6; KataGo `search.cpp:382-384`): with the
    /// new root already installed, mark everything reachable from it (cycle-safe identity DFS), then
    /// break the outgoing edges of every UNREACHABLE table node BEFORE dropping it from the table —
    /// the explicit teardown is what lets ARC reclaim a doomed node that a transposition has linked
    /// into a reference cycle (`unowned`/`weak` were rejected — review §A6). No reachable node ever
    /// points at a doomed node (reachability is closed under the child relation), so clearing the
    /// doomed set's links can never sever a surviving node's subtree. Cost is bounded by the reused
    /// subtree's size, not the whole (now-discarded) previous tree.
    private func sweepUnreachableGraphNodes() {
        let reachable = Set(reachableNodes().map(ObjectIdentifier.init))
        var doomedKeys: [Hash128] = []
        var doomedNodes: [SearchNode] = []
        for (key, node) in nodeTable where !reachable.contains(ObjectIdentifier(node)) {
            doomedKeys.append(key)
            doomedNodes.append(node)
        }
        for node in doomedNodes {
            node.clearLinks()
        }
        for key in doomedKeys {
            nodeTable[key] = nil
        }
    }

    // MARK: - Root bootstrap / result

    /// `Search::computeRootValues` (`search.cpp`), minus mirroring detection (out of scope).
    /// Bootstraps the root's own NN eval first if it has never been visited (fresh tree, or a
    /// reused child that was created but never actually completed a playout before its parent's
    /// search ended — see the type doc comment) by reusing the normal expansion path rather than
    /// a separate throwaway eval call.
    private func computeRecentScoreCenter() async throws {
        if root.visits <= 0 {
            root.state = .evaluating
            // Root averaging (`rootNumSymmetriesToSample > 1`) replaces this first root eval with the
            // k-symmetry average; k=1 / non-square takes the ordinary single-symmetry leaf path and
            // stays byte-identical to today (including the `symmetryRandom` stream position).
            let output: NNOutput
            if averagesRootSymmetries {
                output = try await averagedRootNNOutput()
            } else {
                let request = NNRequest(
                    board: root.board,
                    history: root.history,
                    nextPlayer: root.nextPla,
                    params: leafInputParams()
                )
                output = try await evaluateWithFallback([request])[0]
            }
            applyEvalResult(output, path: [root])
        } else if averagesRootSymmetries {
            try await reaverageReusedRoot()
        }
        let expectedScore = root.scoreMeanAvg
        var center = expectedScore * (1.0 - settings.dynamicScoreCenterZeroWeight)
        let cap = sqrtBoardArea * settings.dynamicScoreCenterScale
        center = min(center, expectedScore + cap)
        center = max(center, expectedScore - cap)
        recentScoreCenter = center
    }

    /// Reused-root re-average (KataGo `$KG/cpp/search/searchnnhelpers.cpp:191-206`): a root retained
    /// across `makeMove` was first evaluated as an ordinary SINGLE-symmetry leaf during the previous
    /// search, so under `rootNumSymmetriesToSample > 1` its stored prior must be recomputed as the
    /// k-symmetry average every genmove. Replaces only the root's `nnOutput` (the prior source for
    /// root child selection) and refreshes its untried-move ordering from the averaged policy,
    /// excluding already-expanded children (the duplicate-`addChild` guard `prepareRootNoiseIfNeeded`
    /// documents). The root's accumulated visit statistics and every descendant node are left
    /// untouched — only the root prior changes, exactly as KataGo's recompute does.
    private func reaverageReusedRoot() async throws {
        let output = try await averagedRootNNOutput()
        root.nnOutput = output
        let alreadyExpanded = Set(root.children.map(\.loc))
        root.untriedMoves = sortedLegalMoves(
            from: output,
            board: root.board,
            history: root.history,
            nextPlayer: root.nextPla
        ).filter { !alreadyExpanded.contains($0.loc) }
    }

    private func prepareRootNoiseIfNeeded() {
        guard rootPolicyProbs == nil,
              settings.rootDirichletAlpha > 0,
              settings.rootDirichletAlpha.isFinite,
              settings.rootNoiseWeight > 0,
              settings.rootNoiseWeight.isFinite,
              let output = root.nnOutput
        else { return }

        let legalPositions = output.policyProbs.indices.filter { output.policyProbs[$0] >= 0 }
        guard !legalPositions.isEmpty else { return }
        let priors = legalPositions.map { Double(output.policyProbs[$0]) }
        let mixed = SearchSampling.mixDirichlet(
            priors: priors,
            alpha: settings.rootDirichletAlpha,
            weight: min(1, settings.rootNoiseWeight),
            random: &random
        )
        var policy = output.policyProbs
        for (position, probability) in zip(legalPositions, mixed) {
            policy[position] = Float(probability)
        }
        rootPolicyProbs = policy
        // `root` may already have children here: with tree reuse (`makeMove`), the new root is a
        // former child that was itself expanded (and expanded ITS OWN children) while being
        // explored as part of a previous move's search, and `rootPolicyProbs` is reset to `nil`
        // on every `makeMove` so this function's guard re-passes on the next `runSearch` call.
        // `selectChild` already looks up EXISTING children's priors from `rootPolicyProbs` fresh
        // on every call (see its `isRoot` branch), so noise is correctly reflected for them
        // without touching `untriedMoves` at all. `untriedMoves` only needs to be refreshed (for
        // ordering/exploration-priority of NOT-yet-expanded moves) and must exclude anything
        // already in `root.children` — recomputing the full legal-move list unconditionally would
        // re-add already-expanded locations, producing a second `addChild` for the same `loc` and
        // a fatal duplicate-key crash the first time something (e.g. conservativePass's cleanup
        // scan) builds a `[Loc: _]` dictionary from `root.children`.
        let alreadyExpanded = Set(root.children.map(\.loc))
        root.untriedMoves = sortedLegalMoves(
            policyProbs: policy,
            board: root.board,
            history: root.history,
            nextPlayer: root.nextPla
        ).filter { !alreadyExpanded.contains($0.loc) }
    }

    private func buildResult(
        visitsAchieved: Int64,
        timeUsed: TimeInterval,
        selectionTemperature: Double,
        useLcbForSelection: Bool
    ) -> SearchResult {
        var bestLoc = Board.passLoc
        var visitsByMove = [(loc: Loc, visits: Int64)]()
        visitsByMove.reserveCapacity(root.children.count)
        // Parallel per-child value map (WP-13). Keyed by loc so a consumer joins it to `visitsByMove`
        // without depending on either collection's order; white-perspective, matching `winValueAvg`.
        var rootChildWinValues = [Loc: Double](minimumCapacity: root.children.count)
        for (loc, child, edgeVisits) in root.children {
            // Graph mode reports each move's EDGE visits (this root's own investment); a shared
            // child's `node.visits` includes other inbound edges (design §2.7 / review §B3). The
            // per-child value stays the shared node's averaged `winValueAvg`. Pure-tree mode reads
            // `child.visits` unchanged (edge == node visits there).
            visitsByMove.append((loc, settings.useGraphSearch ? edgeVisits : child.visits))
            rootChildWinValues[loc] = child.winValueAvg
        }
        // WP-11: LCB final-move selection is a GTP-`genmove`-only override of the CHOSEN move; it
        // never rewrites `visitsByMove`, which stays the pure visit distribution that self-play and
        // teacher labeling record as their training target. LCB is applied only when the caller opts
        // in (`useLcbForSelection`) — self-play/teacher/ponder never do, so their selection and this
        // whole `else if` visit-sampling path stay bit-identical to the pre-WP-11 behavior.
        if useLcbForSelection, let lcbLoc = lcbSelectedMove() {
            bestLoc = lcbLoc
        } else if let index = SearchSampling.sampleVisitIndex(
            visits: visitsByMove.map(\.visits),
            temperature: selectionTemperature,
            random: &random
        ) {
            bestLoc = visitsByMove[index].loc
        }
        visitsByMove.sort { $0.visits > $1.visits }
        if bestLoc == Board.passLoc, let cleanupLoc = conservativePassCleanupMove() {
            bestLoc = cleanupLoc
        }
        return SearchResult(
            bestMove: bestLoc,
            rootVisits: root.visits,
            visitsAchieved: visitsAchieved,
            timeUsed: timeUsed,
            whiteWinProb: (root.winValueAvg + 1.0) / 2.0,
            whiteScoreMean: root.scoreMeanAvg,
            visitsByMove: visitsByMove,
            rootNNOutput: root.nnOutput,
            rootChildWinValues: rootChildWinValues
        )
    }

    /// WP-11 LCB final selection: the loc of the root child with the best lower-confidence-bounded
    /// value among children with enough visits, or `nil` (fall back to visit selection) if the root
    /// has no expanded children yet or no NN output. Ports KataGo's `getPlaySelectionValues` LCB
    /// branch (`useNonBuggyLcb` semantics) via the pure `SearchLCB` helper; play-selection weight
    /// equals a child's visit count here (this port omits `getReducedPlaySelectionWeight` root
    /// pruning — see SearchSettings.swift). `rootPolicyProbs` (Dirichlet-mixed priors, if any) is
    /// used for the reference-weight goodness tie-break, matching `selectChild`'s root branch.
    private func lcbSelectedMove() -> Loc? {
        guard let nnOutput = root.nnOutput else { return nil }
        let policyProbs = rootPolicyProbs ?? nnOutput.policyProbs
        let children = root.children.map { loc, child, edgeVisits in
            // Graph mode: play-selection weight / sample size is the EDGE visit count; the utility
            // moments stay the shared node's averages (design §2.7). Pure-tree: `child.visits`.
            SearchLCB.ChildStat(
                loc: loc,
                visits: settings.useGraphSearch ? edgeVisits : child.visits,
                utilityAvg: child.utilityAvg,
                utilitySqAvg: child.utilitySqAvg,
                policyProb: Double(policyProbs[policyPos(loc, board: root.board)])
            )
        }
        let utilityRangeRadius = settings.winLossUtilityFactor
            + settings.staticScoreUtilityFactor + settings.dynamicScoreUtilityFactor
        return SearchLCB.selectedMove(
            children: children,
            moverIsWhite: root.nextPla == .white,
            utilityRangeRadius: utilityRangeRadius,
            lcbStdevs: settings.lcbStdevs,
            minVisitPropForLCB: settings.minVisitPropForLCB
        )
    }

    /// Scoped port of `PlayUtils::maybeCleanupBeforePass`, using the verified independent-life
    /// classifier in place of NN ownership. `Search::shouldSuppressPass` is territory-scoring and
    /// visit-quality based; this tournament needs the analogous area-scoring cleanup behavior.
    ///
    /// Independent-life regions intentionally omit seki. An opponent stone classified as our
    /// color is therefore pass-dead inside our independently alive area, while a seki stone is
    /// unclassified rather than mistaken for removable. Candidate liberties mirror KataGo's
    /// `safeArea[loc] == pla && isAdjacentToPla(loc, opp)` scan. Expanded candidates rank by
    /// visits, with root policy as the deterministic tiebreak/fallback if a short search did not
    /// expand one. This final-choice fallback is what makes the safety property independent of a
    /// tight visit or wall-clock cutoff without perturbing search numerics.
    private func conservativePassCleanupMove() -> Loc? {
        guard settings.enablePassingHacks,
              root.history.rules.scoringRule == .area,
              root.history.isFinalPhase(),
              !root.history.hasButton,
              let policy = root.nnOutput?.policyProbs
        else { return nil }

        let board = root.board
        let player = root.nextPla
        let opponent = player.opponent
        let independentArea = board.calculateIndependentLifeArea(
            keepTerritories: false,
            keepStones: false,
            isMultiStoneSuicideLegal: root.history.rules.multiStoneSuicideLegal
        ).area
        let deadOpponentHeads = Set(board.playableLocations().compactMap { loc -> Loc? in
            board.colors[loc] == opponent.color && independentArea[loc] == player.color
                ? board.chainHead[loc] : nil
        })
        guard !deadOpponentHeads.isEmpty else { return nil }

        let visitsByLoc = Dictionary(uniqueKeysWithValues: root.children.map {
            ($0.loc, settings.useGraphSearch ? $0.edgeVisits : $0.node.visits)
        })
        var bestLoc: Loc?
        var bestVisits: Int64 = -1
        var bestPolicy = -Float.infinity
        for loc in board.playableLocations() {
            guard board.colors[loc] == .empty,
                  independentArea[loc] == player.color,
                  root.history.isLegal(board, loc, player),
                  board.adjOffsets.prefix(4).contains(where: {
                      let adjacent = loc + $0
                      return board.colors[adjacent] == opponent.color
                          && deadOpponentHeads.contains(board.chainHead[adjacent])
                  }),
                  board.wouldBeCapture(loc, player) || board.getNumLibertiesAfterPlay(loc, player, 2) > 1
            else { continue }

            let visits = visitsByLoc[loc] ?? 0
            let policyProb = policy[policyPos(loc, board: board)]
            if visits > bestVisits || visits == bestVisits && policyProb > bestPolicy {
                bestLoc = loc
                bestVisits = visits
                bestPolicy = policyProb
            }
        }
        return bestLoc
    }
}

private extension Double {
    func clampedNonNegative() -> Double {
        Swift.max(0.0, self)
    }
}

/// Pure, model-free LCB (Lower Confidence Bound) final-move selection (WP-11), factored out of the
/// `Search` actor so the numerics are unit-testable on synthetic child stats. Faithful port of
/// KataGo's `Search::getSelfUtilityLCBAndRadius` (`searchhelpers.cpp`) and the LCB branch of
/// `Search::getPlaySelectionValues` (`searchresults.cpp`, `useNonBuggyLcb = true`), reduced to this
/// port's scope:
///
/// - **weight == 1 per leaf** (no uncertainty weighting — SearchSettings.swift's "Skipped" list), so
///   for a child `weightSum == weightSqSum == visits`, and the effective sample size before the
///   variance prior is exactly `visits`.
/// - **`endingScoreBonus == 0`** always (`rootEndingBonusPoints` is skipped), so KataGo's
///   `getScoreUtilityDiff` term is identically 0 and the child's self-utility is `utilityAvg`
///   (perspective-adjusted) with no score-bonus correction.
/// - **No `getReducedPlaySelectionWeight` root pruning** (skipped), so a child's play-selection
///   weight is exactly its visit count.
///
/// The variance estimate is `utilitySqAvg - utilityAvg^2` — the second moment `Search` already
/// accumulates in backup (`SearchNode.accumulateLeaf`) — with KataGo's small-sample prior that
/// nudges the variance toward the maximum possible (`utilityRangeRadius^2`) at low visit counts.
enum SearchLCB {
    /// Per-root-child inputs for LCB selection. `policyProb < 0` marks a move illegal in the policy
    /// (its play-selection weight is 0), matching KataGo's `policyProbs[pos] < 0` sentinel.
    struct ChildStat {
        var loc: Loc
        var visits: Int64
        var utilityAvg: Double
        var utilitySqAvg: Double
        var policyProb: Double
    }

    /// `Search::getSelfUtilityLCBAndRadius`, our-scope. Returns the lower confidence bound (signed
    /// from the mover's perspective — higher is better for the mover) and the radius. A child with
    /// no visits returns the maximum radius / minimum LCB (`-2 * utilityRangeRadius * lcbStdevs`),
    /// so it can never win selection, exactly as the reference's zero-visit fallback.
    static func selfUtilityLCBAndRadius(
        visits: Int64,
        utilityAvg: Double,
        utilitySqAvg rawUtilitySqAvg: Double,
        moverIsWhite: Bool,
        utilityRangeRadius: Double,
        lcbStdevs: Double
    ) -> (lcb: Double, radius: Double) {
        var radius = 2.0 * utilityRangeRadius * lcbStdevs
        var lcb = -radius
        // weight == 1 per leaf => weightSum == weightSqSum == visits.
        let weightSum0 = Double(visits)
        guard visits > 0, weightSum0 > 0 else { return (lcb, radius) }

        var weightSum = weightSum0
        var weightSqSum = weightSum0
        // Small-weight prior that the variance is the largest it can be, so the bound behaves at
        // very low playouts without a T-distribution approximation (KataGo's exact scheme).
        let ess0 = weightSum * weightSum / weightSqSum
        let priorWeight = weightSum / (ess0 * ess0 * ess0)
        var utilitySqAvg = Swift.max(rawUtilitySqAvg, utilityAvg * utilityAvg + 1e-8)
        utilitySqAvg = (utilitySqAvg * weightSum
            + (utilitySqAvg + utilityRangeRadius * utilityRangeRadius) * priorWeight)
            / (weightSum + priorWeight)
        weightSum += priorWeight
        weightSqSum += priorWeight * priorWeight
        let ess = weightSum * weightSum / weightSqSum

        // endingScoreBonus / getScoreUtilityDiff are 0 in this port, so selfUtility == utilityAvg.
        let selfUtility = moverIsWhite ? utilityAvg : -utilityAvg
        let utilityVariance = utilitySqAvg - utilityAvg * utilityAvg
        let estimateStdev = (utilityVariance / ess).squareRoot()
        radius = estimateStdev * lcbStdevs
        lcb = selfUtility - radius
        return (lcb, radius)
    }

    /// The LCB-preferred final move, or `nil` to fall back to visit-count selection. Ports the LCB
    /// branch of `getPlaySelectionValues`: among children whose play-selection weight (visits) is
    /// `> 0` and at least `minVisitPropForLCB * referenceWeight`, choose the highest LCB (ties
    /// broken by child creation order, matching KataGo's strict-`>` scan over the children array).
    /// `referenceWeight` is the "most stably explored" child's weight per KataGo's goodness metric
    /// (`weight * max(0, visits-1)/max(1, visits) + 2 * policyProb`) — at realistic visit counts
    /// this is simply the max-visits child. The max-visits child is always eligible, so a non-`nil`
    /// result is returned whenever any legal child exists.
    static func selectedMove(
        children: [ChildStat],
        moverIsWhite: Bool,
        utilityRangeRadius: Double,
        lcbStdevs: Double,
        minVisitPropForLCB: Double
    ) -> Loc? {
        guard !children.isEmpty else { return nil }

        var referenceWeight = 0.0
        var maxGoodness = -Double.infinity
        for child in children {
            let selValue = child.policyProb < 0 ? 0.0 : Double(child.visits)
            let ev = Double(child.visits)
            let goodness = selValue * Swift.max(0.0, ev - 1.0) / Swift.max(1.0, ev) + 2.0 * child.policyProb
            if goodness > maxGoodness {
                maxGoodness = goodness
                referenceWeight = selValue
            }
        }

        let threshold = minVisitPropForLCB * referenceWeight
        var bestLcb = -Double.infinity
        var bestLoc: Loc?
        for child in children {
            let selValue = child.policyProb < 0 ? 0.0 : Double(child.visits)
            guard selValue > 0, selValue >= threshold else { continue }
            let (lcb, _) = selfUtilityLCBAndRadius(
                visits: child.visits,
                utilityAvg: child.utilityAvg,
                utilitySqAvg: child.utilitySqAvg,
                moverIsWhite: moverIsWhite,
                utilityRangeRadius: utilityRangeRadius,
                lcbStdevs: lcbStdevs
            )
            if lcb > bestLcb {
                bestLcb = lcb
                bestLoc = child.loc
            }
        }
        return bestLoc
    }
}
