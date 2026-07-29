import Foundation

/// The subset of KataGo's `SearchParams` this port implements, with the defaults KataGo's GTP
/// engine actually plays with: the `SearchParams::SearchParams()` constructor defaults from
/// `$KG/cpp/search/searchparams.cpp`, overlaid with the GTP-context defaults that
/// `Setup::loadParams` (`$KG/cpp/program/setup.cpp`, non-distributed path) applies when the
/// config leaves them unset — e.g. `staticScoreUtilityFactor = 0.1` / `dynamicScoreUtilityFactor
/// = 0.3` (not the constructor's 0.3/0.0), `cpuctExplorationLog = 0.45`, `policyOptimism = 1.0`,
/// `rootPolicyOptimism = 0.2`, `fpuParentWeightByVisitedPolicyPow = 2.0`,
/// `cpuctUtilityStdevScale = 0.85`.
///
/// What this port's search implements vs. skips (docs/design-notes.md scoping; skips are also noted at the
/// matching code sites in `Search.swift`):
///
/// Ported:
/// - PUCT selection: `cpuctExploration`/`cpuctExplorationLog` scaling with the
///   `sqrt(totalChildWeight + 0.01)` term and the parent-utility-stdev exploration factor
///   (`cpuctUtilityStdevPrior/PriorWeight/Scale`), per `searchexplorehelpers.cpp`.
/// - First-play urgency: `fpuReductionMax`/`rootFpuReductionMax` with
///   `fpuParentWeightByVisitedPolicy` blending of parent average vs. parent NN utility.
/// - Utility: win/loss + no-result utility (`getResultUtility`) plus static and dynamic score
///   utility (`getScoreUtility`) with the per-search dynamic score center
///   (`dynamicScoreCenterZeroWeight`/`dynamicScoreCenterScale`), per `searchhelpers.cpp` /
///   `search.cpp::computeRootValues`.
/// - Virtual loss with KataGo's exact utility/weight adjustment (`numVirtualLossesPerThread`).
/// - Terminal-node values (win/loss/no-result/score from `BoardHistory`), including the
///   root `forceNonTerminal` rule; the tree is dropped instead of reused when a move lands on a
///   game-ending position, matching KataGo's "clear search when a pass from the root would be
///   terminal" behavior.
/// - Root policy optimism vs. tree policy optimism (`rootPolicyOptimism`/`policyOptimism`),
///   with the root NN output re-fetched at root optimism after tree reuse
///   (KataGo's `maybeRecomputeExistingNNOutput`).
/// - Chosen-move temperature schedule: `chosenMoveTemperature`/`Early`/`Halflife` with
///   `interpolateEarly` board-size scaling and the exact `chooseIndexWithTemperature` transform.
/// - Tree reuse across `makeMove` (gated by `treeReuseEnabled`, verified by a superko-safe
///   Zobrist/`koHashHistory` position key — see `Search.canReuseSubtree`).
/// - LCB (Lower Confidence Bound) final-move selection (WP-11), matching KataGo's
///   `getSelfUtilityLCBAndRadius`/`getPlaySelectionValues` `useNonBuggyLcb` semantics
///   (`lcbEnabled`/`lcbStdevs`/`minVisitPropForLCB`). Applied ONLY to GTP `genmove` final choice
///   (`Search.runSearch(useLcbForSelection:)`); self-play/training keep pure visit-count selection.
///
/// Skipped (documented non-goals; values that would control them are omitted here):
/// - Subtree value bias, pattern bonus. In pure-tree mode (`useGraphSearch == false`, the
///   default) edge visits always equal child visits and are not tracked separately; graph/DAG
///   transposition search is implemented behind `useGraphSearch` (default OFF — see below).
/// - `getReducedPlaySelectionWeight` root overexploration pruning (so a child's play-selection
///   weight is exactly its visit count; the LCB final selection above is implemented on top of it).
/// - Uncertainty weighting (`useUncertainty`): all leaf evals have weight 1, so node weight ==
///   visits. (The b6c96/b10c128-era nets don't emit shortterm error estimates anyway.)
/// - `valueWeightExponent` bad-child downweighting and `useNoisePruning` (weights are exact
///   visit-weighted means; this is what makes incremental backup equal KataGo's
///   recompute-from-children).
/// - Wide root noise (`rootDirichletAlpha`/`rootNoiseWeight` below implement only standard
///   AlphaZero-style symmetric Dirichlet noise),
/// - `friendlyPass`, `rootEndingBonusPoints`, `rootPruneUselessMoves`, `fillDameBeforePass`:
///   pass remains a normal legal search move. `enablePassingHacks` only filters the final root
///   choice when area scoring would otherwise leave removable opponent stones on the board.
/// - `futileVisitsThreshold`, wide root noise, anti-mirror, human SL, root symmetry sampling,
///   analysis output.
///
/// Not in this struct because it is an orchestration concern, not a search-numeric — pondering
/// ("think on the opponent's time", WP-10) is implemented in `GTPEngine` as a background search on
/// the opponent-to-move root, torn down by cooperative `Task` cancellation (`Search.runSearch`
/// checks `Task.isCancelled` between batches) the instant the next GTP command arrives; tree reuse
/// then harvests the pondered subtree. Enabled via `gtp -ponder on` (default off).
public struct SearchSettings: Sendable {
    /// Whether GTP `genmove` may resign after its existing low-winrate streak threshold.
    public var resignEnabled = true

    /// Whether `Search.makeMove` re-roots onto the played child's retained subtree (visit counts,
    /// values, expanded children) instead of discarding the tree and starting fresh. Defaults ON:
    /// under sudden death this recycles the 30-60% of visits that stay valid after a move, a large
    /// effective-Elo gain. Reuse is additionally gated per-move by a superko-safe position key
    /// (`Search.canReuseSubtree`), so setting this `false` only forces the always-fresh path (used
    /// for A/B evaluation via `gtp -tree-reuse off`); the default ON path is bit-identical to the
    /// pre-flag behavior on any position where the key matches (which, in the GTP/self-play move
    /// flow, is every position — the guard is defense-in-depth, never a behavior change there).
    public var treeReuseEnabled = true

    /// Graph/DAG transposition search (KataGo `useGraphSearch`, `searchparams.h:51`). Default OFF:
    /// with this `false` the search is a pure tree and every code path is byte-identical to the
    /// pre-flag behavior (the whole DAG core in `Search.swift` is gated behind this flag). When
    /// `true`, positions reached by different move orders share a single `SearchNode` via a
    /// `GraphHash`-keyed node table, parent→child edges carry their own `edgeVisits` (separate from
    /// the shared node's `visits`), backup recomputes each node from its children, and every root
    /// consumer reports edge statistics. Enabled only via GTP `-graph-search on`; self-play,
    /// teacher labeling, and every existing test default OFF. The doc (§2.3) referred to this as
    /// `graphSearchEnabled`; named `useGraphSearch` here to match KataGo and the WP-GS2 spec.
    public var useGraphSearch = false

    /// KataGo `graphSearchRepBound` (`searchparams.cpp:42`, default 11): the cycle-size bound the
    /// `GraphHash` chaining uses. A move whose disturbed region exceeds this bound proves no cycle
    /// of `<= repBound` moves could return to the same position, so the graph hash resets to the
    /// pure state hash (transpositions merge); otherwise it chains a path-dependent hash so
    /// superko-relevant short cycles do NOT merge. Only read when `useGraphSearch` is `true`.
    public var graphSearchRepBound = 11

    // Utility scaling (setup.cpp GTP defaults).
    public var winLossUtilityFactor = 1.0
    public var staticScoreUtilityFactor = 0.1
    public var dynamicScoreUtilityFactor = 0.3
    public var dynamicScoreCenterZeroWeight = 0.2
    public var dynamicScoreCenterScale = 0.75
    public var noResultUtilityForWhite = 0.0
    public var drawEquivalentWinsForWhite = 0.5

    // PUCT exploration.
    public var cpuctExploration = 1.0
    public var cpuctExplorationLog = 0.45
    public var cpuctExplorationBase = 500.0
    public var cpuctUtilityStdevPrior = 0.40
    public var cpuctUtilityStdevPriorWeight = 2.0
    public var cpuctUtilityStdevScale = 0.85

    // First-play urgency.
    public var fpuReductionMax = 0.2
    public var rootFpuReductionMax = 0.1
    public var fpuParentWeightByVisitedPolicy = true
    public var fpuParentWeightByVisitedPolicyPow = 2.0

    // Policy optimism (blended inside NN postprocessing via `MiscNNInputParams.policyOptimism`).
    public var policyOptimism = 1.0
    public var rootPolicyOptimism = 0.2

    /// Root-only conservative cleanup before passing. This is the GTP-facing equivalent of
    /// KataGo's passing hacks for area rules and defaults on, as in KataGo's GTP setup. It does
    /// not alter PUCT or NN postprocessing: if pass wins the visit count but verified independent
    /// area marks opponent stones as removable inside our area, final move selection substitutes
    /// the best safe cleanup liberty. Set false to retain unfiltered argmax-visits behavior.
    public var enablePassingHacks = true

    /// LCB (Lower Confidence Bound) final-move selection (WP-11). When applied, the final root move
    /// is the child with the best lower-confidence-bounded value estimate among children with
    /// enough visits, instead of the plain max-visits child — community-measured ~+100 Elo at fixed
    /// visits (Leela Zero #2411). Defaults mirror KataGo's GTP setup (`searchparams.cpp`:
    /// `useLcbForSelection=true`, `lcbStdevs=5`, `minVisitPropForLCB=0.15`, `useNonBuggyLcb=true`).
    ///
    /// `lcbEnabled` is read ONLY by the GTP `genmove` path (which passes
    /// `Search.runSearch(useLcbForSelection: settings.lcbEnabled)`); self-play and teacher searches
    /// never pass `useLcbForSelection: true`, so LCB never touches the recorded visit distribution
    /// or their move selection (LCB in self-play degrades training data — the Leela report).
    public var lcbEnabled = true
    /// Stddevs of the value estimate subtracted for the confidence bound (KataGo `lcbStdevs`).
    public var lcbStdevs = 5.0
    /// A child may win LCB selection only if its visits are at least this fraction of the reference
    /// (most-stably-explored) child's visits (KataGo `minVisitPropForLCB`).
    public var minVisitPropForLCB = 0.15

    /// Virtual loss, batching, limits.
    public var numVirtualLossesPerThread = 3.0
    /// Leaves collected per NN batch via virtual loss — the analog of KataGo's
    /// `numSearchThreads` (GTP example config uses 6; we default to a GPU-friendlier 16).
    /// Also the natural `NNEvaluator` batch size to configure.
    public var leafBatchSize = 16
    public var maxVisits: Int64 = 400
    /// Leaf-batch collection width: each NN `evaluate` gathers up to
    /// `min(leafBatchSize, numSearchWorkers)` distinct leaves (diversified by virtual loss) in one
    /// sequential pass. Named for KataGo's `numSearchThreads`, and originally the count of
    /// concurrent descent tasks feeding the batcher — but that concurrent "flush when full or when
    /// all workers are parked" protocol deadlocked (see `Search`'s doc comment), so the batcher is
    /// now a single deterministic loop that spawns no tasks. Consequently search is
    /// bit-reproducible at *every* value here (not just `1`); `1` is textbook one-leaf-at-a-time
    /// MCTS, larger values widen the GPU batch at a small cost to per-eval tree freshness. Default
    /// `min(leafBatchSize, 8)` once configured by the caller (`GTPCommand`).
    public var numSearchWorkers = 8

    /// Self-play exploration. All three behavioral controls default off, preserving GTP search.
    /// Symmetric Dirichlet concentration mixed into legal root priors. Zero disables root noise.
    /// Self-play uses 0.15 on 9x9, close to KataGo's area-scaled `0.03 * 361 / boardArea`
    /// heuristic (0.134 on 9x9), rounded to the conventional value requested by the CLI.
    public var rootDirichletAlpha = 0.0
    /// Fraction of the normalized root policy replaced by Dirichlet noise. Zero disables mixing.
    public var rootNoiseWeight = 0.0
    /// Temperature for the final root choice. Zero retains stable argmax-visits selection.
    public var selectionTemperature = 0.0
    /// Seed for the deterministic SplitMix64 stream used by root noise and final-move sampling.
    public var randomSeed: UInt64 = 0

    /// Per-eval NN input symmetry mode (KataGo `nnRandomize`), applied only by the search path.
    /// `.off` (default) is behavior-identical to no symmetry — oracle parity and every existing gate
    /// are preserved. `.hash` reproduces the previous deterministic `-symmetry on`. `.random` draws
    /// one dihedral symmetry per node from `symmetrySeed`'s stream and carries it on
    /// `MiscNNInputParams.symmetry`, so the evaluator applies it consistently across the fp16 primary
    /// and any fp32 fallback of the SAME request. Self-play, teacher labeling, and every test default
    /// to `.off`; only the GTP command surfaces `.hash`/`.random`.
    public var symmetryMode: NNSymmetryMode = .off
    /// Seed for the deterministic SplitMix64 stream `Search` draws per-node symmetries from in
    /// `.random` mode. Deliberately SEPARATE from `randomSeed` so drawing symmetries never shifts the
    /// root-noise / final-move-sampling stream (and vice versa). Fixed default for reproducible A/Bs.
    public var symmetrySeed: UInt64 = SearchSettings.defaultSymmetrySeed

    /// Default `symmetrySeed` (also the CLI `-symmetry-seed` default): a fixed constant so `.random`
    /// runs are reproducible by default. Shared with `GTPCommand` so both defaults stay in lockstep.
    public static let defaultSymmetrySeed: UInt64 = 20_260_722

    /// KataGo `rootNumSymmetriesToSample`: how many DISTINCT dihedral symmetries' NN outputs are
    /// averaged at the ROOT node only (leaves are never touched). `1` (default) is off and
    /// byte-identical to today — no draws, no averaging, no extra evals. `k > 1` draws `k` symmetries
    /// without replacement (from `symmetrySeed`'s stream) each genmove, evaluates the root position
    /// under each as a separate NN request, and averages the (already canonical-orientation) outputs
    /// into a lower-variance root prior + root value — well suited to a noisy self-trained net. Costs
    /// `k - 1` extra root evals per genmove (negligible vs total playouts). Engaged only on a square
    /// NN tensor (`nnXLen == nnYLen`, the only shape the dihedral transforms are defined for);
    /// `Search` clamps the effective count to at most `Symmetry.count` (8) at the draw site. GTP
    /// surfaces it via `-root-symmetries`; self-play, teacher labeling, and every test default to `1`.
    public var rootNumSymmetries = 1

    // Chosen-move temperature schedule (setup.cpp GTP defaults).
    public var chosenMoveTemperature = 0.1
    public var chosenMoveTemperatureEarly = 0.5
    public var chosenMoveTemperatureHalflife = 19.0
    public var chosenMoveTemperatureOnlyBelowProb = 1.0

    public init() {}
}
