import Foundation
import RinGoCore

/// Converts a teacher network's postprocessed `NNOutput` into RinGoData v2 training targets for
/// distillation (`makedata -teacher-model ...`, TRAINING_FIX_PLAN.md WP-2).
///
/// The single non-obvious job here is perspective/encoding reconciliation between the PRODUCER
/// (`NNOutputPostprocessor`, which always emits values/ownership from WHITE's perspective and a
/// policy with the C++ `-1` sentinel on illegal moves) and the CONSUMER (`LossComputer` +
/// `Trainer.batch`, which read the v2 targets from the SIDE-TO-MOVE perspective). Every mapping
/// below is chosen to match what training consumes; see the per-field comments for the exact
/// producer/consumer code that pins it.
public enum TeacherLabeler {
    /// Which improved-policy target a teacher SEARCH emits (`makedata -teacher-target`, WP-13).
    /// The raw string values are the CLI spellings and the `makedata-config.json` provenance values.
    public enum PolicyTargetKind: String, Sendable {
        /// Normalized root VISIT distribution (AlphaZero-style; the historical default).
        case visits
        /// Gumbel completed-Q improved policy (see `completedQPolicy`).
        case completedQ = "completed-q"
    }

    public struct Targets: Equatable, Sendable {
        /// Dense probability distribution over `nnLen*nnLen+1` moves, illegal moves masked to 0.
        public var policy: [Float]
        /// `[win, loss, noResult]` from the side-to-move perspective (matches v2 `valueTarget`).
        public var value: [Float]
        /// Side-to-move final-score proxy in points (the loss divides by 20 to match
        /// `scoreMeanMultiplier`; see `LossComputer`).
        public var score: Float
        /// Per-point `{-1, 0, +1}` from the side-to-move perspective (matches v2 `ownershipTarget`).
        public var ownership: [Int8]

        public init(policy: [Float], value: [Float], score: Float, ownership: [Int8]) {
            self.policy = policy
            self.value = value
            self.score = score
            self.ownership = ownership
        }
    }

    /// Builds v2 targets from a teacher `NNOutput` evaluated at this position.
    ///
    /// - policy: `NNOutput.policyProbs` is already the legal-move-masked softmax the engine's own
    ///   search consumes (illegal/padded entries are the `-1` sentinel). We copy the legal
    ///   probabilities through and set the illegal ones to exactly 0, so the result is a dense
    ///   distribution that sums to ~1 over the legal moves -- exactly the soft policy target the
    ///   dense-policy `crossEntropy` branch in `LossComputer` expects.
    /// - value: `NNOutput` exposes win/loss/noResult from WHITE's perspective
    ///   (`whiteWinProb`/`whiteLossProb`/`whiteNoResultProb`), but v2 `valueTarget` is
    ///   `[win, loss, noResult]` from the SIDE-TO-MOVE perspective. When black is to move, white's
    ///   win probability is black's loss probability, so win/loss swap; noResult is
    ///   perspective-independent. (Producer: `NNOutputPostprocessor` sets
    ///   `whiteWinProb = whiteToMove ? probabilities[0] : probabilities[1]`, i.e. `probabilities[0]`
    ///   is P(win for side-to-move). Consumer: `LossComputer` runs
    ///   `crossEntropy(logits: outputs.valueLogits, targets: batch.valueTarget)`, and teacherless
    ///   `makedata` writes `winner == nextPlayer ? [1,0,0] : [0,1,0]` -- i.e. index 0 is
    ///   side-to-move win. Both agree that index 0 = side-to-move win, index 1 = side-to-move loss,
    ///   index 2 = noResult.)
    /// - score: `whiteScoreMean` is white-perspective points; negate for a black side-to-move.
    /// - ownership: `whiteOwnerMap` is white-perspective tanh in [-1, 1]; negate for black, then
    ///   quantize to the `{-1, 0, +1}` Int8 alphabet v2 uses. This is REQUIRED, not merely
    ///   convenient: `Trainer.batch` reads each stored Int8 straight into a Float and
    ///   `LossComputer` maps it via `(v + 1) / 2` into a [0, 1] BCE target, so only Int8 values in
    ///   `{-1, 0, +1}` decode to a valid probability `{0, 0.5, 1}`. Rounding to nearest keeps the
    ///   decoded probability within 0.25 of the teacher's continuous value.
    public static func targets(from output: NNOutput, nextPlayer: Player, nnLen: Int) -> Targets {
        let area = nnLen * nnLen
        let whiteToMove = nextPlayer == .white

        var policy = [Float](repeating: 0, count: area + 1)
        let probs = output.policyProbs
        for index in 0 ..< min(policy.count, probs.count) {
            let probability = probs[index]
            policy[index] = probability > 0 ? probability : 0
        }

        let win = whiteToMove ? output.whiteWinProb : output.whiteLossProb
        let loss = whiteToMove ? output.whiteLossProb : output.whiteWinProb
        let value: [Float] = [win, loss, output.whiteNoResultProb]

        let score = whiteToMove ? output.whiteScoreMean : -output.whiteScoreMean

        var ownership = [Int8](repeating: 0, count: area)
        if let map = output.whiteOwnerMap {
            for index in 0 ..< min(area, map.count) {
                let sideToMoveOwner = whiteToMove ? map[index] : -map[index]
                let quantized = sideToMoveOwner.rounded() // -1, 0, or +1 for input in [-1, 1]
                ownership[index] = Int8(max(-1, min(1, quantized)))
            }
        }

        return Targets(policy: policy, value: value, score: score, ownership: ownership)
    }

    /// Builds v2 targets from a teacher PUCT SEARCH at this position (Phase P, `makedata
    /// -teacher-model <net> -teacher-visits N`, N >= 2). Search-amplified targets are strictly
    /// stronger supervision than the raw net (AlphaZero-style): the policy is the search's own
    /// action distribution and the value is the search's backed-up winrate.
    ///
    /// - policy: the normalized root visit distribution. `visitsByMove` is `(loc, visits)` for every
    ///   expanded root child (pass is a normal legal child, so it appears here when it was searched);
    ///   each loc maps to its dense-policy position via `NNPos.locToPos` (pass -> `area`). Only legal
    ///   moves are ever expanded, so the result is already legal-masked and sums to 1 over the
    ///   searched moves. With few visits the distribution is deliberately coarse -- that is expected.
    ///   If no child was visited at all (defensive; cannot happen for N >= 2 on a non-terminal root),
    ///   the raw net's masked policy is kept as a fallback.
    /// - value: the root's search-aggregated winrate reprojected to the side to move. `whiteWinProb`
    ///   is white's draw-equivalent win probability from the backed-up root value; when black is to
    ///   move it is `1 - whiteWinProb`. noResult is set to 0: it is negligible on 9x9 with positive
    ///   komi and is already folded into the draw-equivalent winrate, so re-adding the raw net's
    ///   noResult would double-count it. The stored `[win, loss, noResult]` still sums to 1.
    /// - score and ownership: taken verbatim from the raw net output AT THE ROOT (`rootOutput`), via
    ///   the same perspective/quantization logic as `targets(from:)`. Search does not materially
    ///   improve the score/ownership heads (a few dozen visits mostly sharpen policy and value), so
    ///   spending the search budget to re-estimate them is not worth it; the raw root estimate is
    ///   already what the raw-distillation path uses and is a genuine label (both valid flags true).
    ///
    /// `target` selects the POLICY construction (WP-13). `.visits` (default) is the historical
    /// normalized visit distribution and is byte-identical to the pre-WP-13 behavior; `.completedQ`
    /// builds the Gumbel improved policy from the root prior + per-child Q values (see
    /// `completedQPolicy`), for which `childWinValues` (from `SearchResult.rootChildWinValues`) must
    /// be supplied. Value/score/ownership are identical across both modes.
    public static func searchTargets(
        visitsByMove: [(loc: Loc, visits: Int64)],
        whiteWinProb: Double,
        rootOutput: NNOutput,
        boardXSize: Int,
        nextPlayer: Player,
        nnLen: Int,
        target: PolicyTargetKind = .visits,
        childWinValues: [Loc: Double] = [:]
    ) -> Targets {
        let area = nnLen * nnLen
        // Ownership + score (and the raw-policy fallback) come from the root's raw net evaluation,
        // which the search already computed while bootstrapping the root -- no extra teacher eval.
        var result = targets(from: rootOutput, nextPlayer: nextPlayer, nnLen: nnLen)

        // Policy: normalized root visit counts, legal-masked (only legal moves are expanded), pass
        // included. Keep the raw masked policy from `targets` when nothing was visited.
        let totalVisits = visitsByMove.reduce(Int64(0)) { $0 + $1.visits }
        if totalVisits > 0 {
            switch target {
            case .visits:
                var policy = [Float](repeating: 0, count: area + 1)
                for (loc, visits) in visitsByMove where visits > 0 {
                    let position = NNPos.locToPos(loc, boardXSize, nnLen, nnLen)
                    guard position >= 0, position < policy.count else { continue }
                    policy[position] += Float(Double(visits) / Double(totalVisits))
                }
                result.policy = policy
            case .completedQ:
                result.policy = completedQPolicy(
                    visitsByMove: visitsByMove,
                    childWinValues: childWinValues,
                    rootOutput: rootOutput,
                    boardXSize: boardXSize,
                    nextPlayer: nextPlayer,
                    nnLen: nnLen,
                    fallback: result.policy
                )
            }
        }

        // Value: search winrate reprojected to the side to move; noResult folded into the winrate.
        let whiteToMove = nextPlayer == .white
        let sideToMoveWin = Float(whiteToMove ? whiteWinProb : 1 - whiteWinProb)
        result.value = [sideToMoveWin, 1 - sideToMoveWin, 0]

        return result
    }

    // MARK: - Gumbel completed-Q policy target (WP-13)

    /// Gumbel MuZero sigma calibration. From the DeepMind `mctx` reference implementation's
    /// `qtransform_completed_by_mix_value` defaults (`maxvisit_init = 50.0`, `value_scale = 0.1`,
    /// `rescale_values = True`), NOT the paper TEXT's `c_scale = 1.0`. Deviation: the paper writes
    /// `sigma(q) = (c_visit + max_b N(b)) * c_scale * q`; with our completed-Q rescaled to [0, 1]
    /// (below), `c_scale = 1.0` gives `sigma` up to `50 + maxN` (~tens of nats), which after the
    /// softmax collapses the improved policy to a near-hard argmax over Q and erases the prior's
    /// contribution — the opposite of the intended *soft* improvement. The released `mctx` default
    /// (`value_scale = 0.1` with values rescaled to [0, 1]) keeps `sigma` in a range comparable to
    /// the prior logits, so both terms matter; we follow the reference implementation.
    private static let cVisit = 50.0
    private static let cScale = 0.1

    /// Builds the Gumbel MuZero "completed-Q" improved policy as a dense `area+1` distribution.
    ///
    /// Reference: Danihelka, Guez, Schrittwieser & Silver, "Policy improvement by planning with
    /// Gumbel" (ICLR 2022), and the DeepMind `mctx` implementation (`qtransforms.py`
    /// `qtransform_completed_by_mix_value` + `_compute_mixed_value`, and `policies.py`
    /// `pi' = softmax(prior_logits + sigma(completedQ))`).
    ///
    /// Formula (all Q/value quantities in SIDE-TO-MOVE win-minus-loss units, [-1, 1]):
    ///   - `pi(a)`  = root prior policy  = `rootOutput.policyProbs[pos]` (a legal-masked softmax,
    ///                already summing to ~1 over legal moves; the `-1` sentinel marks illegal).
    ///   - `q(a)`   = the visited child's backed-up value `sign * childWinValues[loc]`, where
    ///                `sign = +1` if white to move else `-1` (reproject WHITE-perspective `winValueAvg`
    ///                to the side to move). Defined only for children with `N(a) > 0`.
    ///   - `v_hat`  = raw NETWORK root value `sign * (whiteWinProb - whiteLossProb)` from `rootOutput`
    ///                (the paper's `v_pi`: the value PREDICTION, not the backed-up search value —
    ///                matching `mctx`, which passes `tree.raw_values`).
    ///   - `v_mix`  = (v_hat + (sum_b N(b)) * weighted_q) / (1 + sum_b N(b)), where
    ///                `weighted_q = (sum_{a:N(a)>0} pi(a) q(a)) / (sum_{a:N(a)>0} pi(a))`.
    ///   - `completedQ(a)` = `q(a)` if `N(a) > 0` else `v_mix`, over EVERY legal action.
    ///   - rescale `completedQ` to [0, 1] via min-max over the legal actions (`mctx rescale_values`):
    ///     `(q - min) / max(max - min, 1e-8)`. This min-max is affine, so whether the inputs are in
    ///     [-1, 1] (our win-minus-loss) or [0, 1] (winrate) is irrelevant to the result — only the
    ///     side-to-move SIGN of `q`/`v_hat` matters, and that is applied above. (This is the honest
    ///     answer to the "[-1,1] vs [0,1]" scale question: the rescale cancels the range difference.)
    ///   - `sigma(a)` = `(cVisit + max_b N(b)) * cScale * completedQ_rescaled(a)`.
    ///   - `pi'(a)`   = softmax over legal actions of `log(pi(a)) + sigma(a)`; illegal actions -> 0.
    ///     (`log(pi(a))` recovers the prior logits up to a constant the softmax absorbs; `pi(a)` is
    ///     floored at 1e-8 before the log so a legal move the net rounded to 0 does not become `-inf`.)
    ///
    /// Deviations from the paper, beyond the `cScale` note on the constants above:
    ///   - Legal set: we treat `policyProbs[pos] >= 0` as the legal mask (the same convention
    ///     `targets(from:)` and the visits path use). Completed-Q therefore spreads prior mass onto
    ///     ALL net-legal actions (including any the engine would separately reject, e.g. a rare
    ///     positional-superko move), whereas the visits path only ever has mass on engine-EXPANDED
    ///     children. Negligible on 9x9; kept for consistency with the existing raw/visits paths.
    ///   - `fallback` (the raw masked policy) is returned unchanged if nothing was visited or the
    ///     softmax degenerates; the caller only reaches here when `totalVisits > 0`.
    private static func completedQPolicy(
        visitsByMove: [(loc: Loc, visits: Int64)],
        childWinValues: [Loc: Double],
        rootOutput: NNOutput,
        boardXSize: Int,
        nextPlayer: Player,
        nnLen: Int,
        fallback: [Float]
    ) -> [Float] {
        let area = nnLen * nnLen
        let probs = rootOutput.policyProbs
        let sign = nextPlayer == .white ? 1.0 : -1.0

        // Per-position visit counts and (for visited children) side-to-move Q values.
        var visitsByPos = [Int: Int64]()
        var qByPos = [Int: Double]()
        var totalVisits: Int64 = 0
        var maxVisits: Int64 = 0
        for (loc, visits) in visitsByMove where visits > 0 {
            let pos = NNPos.locToPos(loc, boardXSize, nnLen, nnLen)
            guard pos >= 0, pos <= area else { continue }
            visitsByPos[pos, default: 0] += visits
            totalVisits += visits
            maxVisits = max(maxVisits, visitsByPos[pos]!)
            if let winValue = childWinValues[loc] { qByPos[pos] = sign * winValue }
        }
        guard totalVisits > 0 else { return fallback }

        // v_hat: raw NETWORK root value, reprojected to the side to move (the paper's v_pi).
        let vHat = sign * (Double(rootOutput.whiteWinProb) - Double(rootOutput.whiteLossProb))

        // weighted_q over VISITED children only: prior-weighted mean of their Q values.
        var visitedPriorSum = 0.0
        var weightedQNumerator = 0.0
        for (pos, q) in qByPos {
            let prior = pos < probs.count ? Double(max(0, probs[pos])) : 0
            visitedPriorSum += prior
            weightedQNumerator += prior * q
        }
        let weightedQ = visitedPriorSum > 0 ? weightedQNumerator / visitedPriorSum : vHat

        // v_mix: mctx `_compute_mixed_value`.
        let sumN = Double(totalVisits)
        let vMix = (vHat + sumN * weightedQ) / (sumN + 1.0)

        // Completed Q over every legal action (pos with policyProbs >= 0), pass at index `area`.
        var legalPositions = [Int]()
        var completedQ = [Double]()
        legalPositions.reserveCapacity(area + 1)
        completedQ.reserveCapacity(area + 1)
        for pos in 0 ... area where pos < probs.count && probs[pos] >= 0 {
            legalPositions.append(pos)
            completedQ.append(qByPos[pos] ?? vMix)
        }
        guard !legalPositions.isEmpty else { return fallback }

        // Rescale completed Q to [0, 1] (affine min-max; see doc comment on scale invariance).
        let minQ = completedQ.min() ?? 0
        let maxQ = completedQ.max() ?? 0
        let denominator = max(maxQ - minQ, 1e-8)
        let sigmaScale = (cVisit + Double(maxVisits)) * cScale

        // Improved logits: log(prior) + sigma(completedQ). Track the max for a stable softmax.
        var logits = [Double](repeating: 0, count: legalPositions.count)
        var maxLogit = -Double.infinity
        for index in legalPositions.indices {
            let pos = legalPositions[index]
            let prior = Double(max(0, probs[pos]))
            let rescaledQ = (completedQ[index] - minQ) / denominator
            let logit = Foundation.log(max(prior, 1e-8)) + sigmaScale * rescaledQ
            logits[index] = logit
            maxLogit = max(maxLogit, logit)
        }

        var expSum = 0.0
        for index in logits.indices {
            let value = Foundation.exp(logits[index] - maxLogit)
            logits[index] = value
            expSum += value
        }
        guard expSum > 0, expSum.isFinite else { return fallback }

        var policy = [Float](repeating: 0, count: area + 1)
        for index in legalPositions.indices {
            policy[legalPositions[index]] = Float(logits[index] / expSum)
        }
        return policy
    }
}
