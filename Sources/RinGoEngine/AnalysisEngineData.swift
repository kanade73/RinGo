import Foundation
import RinGoCore

// WP-14: ingest official KataGo **analysis-engine** output as RinGoData v2 training labels.
//
// Two halves, both CPU-only and MLX-free:
//   1. `AnalysisExporter` -- turns a local SGF corpus into a JSON-lines file of analysis-engine
//      *queries* (one query per game, all its positions batched via `analyzeTurns`) plus a sidecar
//      `manifest.json` binding each query `id` back to its (game file, turn range).
//   2. `AnalysisIngestor` -- reads the analysis engine's JSON-lines *responses*, matches them to the
//      manifest + original SGFs by `id`+`turnNumber`, rebuilds the V7 features locally (the same
//      `fillRowV7` path as `makedata`), and writes standard v2 shards + a provenance JSON.
//
// The one genuinely subtle part is PERSPECTIVE. The analysis engine reports `winrate`, `scoreLead`
// and `ownership` from the perspective selected by the engine config's `reportAnalysisWinratesAs`
// (`BLACK` | `WHITE` | `SIDETOMOVE`); the shipped `analysis_example.cfg` default is **BLACK**
// (verified against KataGo master). RinGoData v2 wants everything from the SIDE-TO-MOVE
// perspective. `AnalysisIngestor` therefore takes an explicit `-winrates-as` flag that MUST match
// the engine config, and `AnalysisLabeler` reprojects every field to side-to-move accordingly. See
// `AnalysisLabeler.targets` for the exact per-field mapping and doc citations.

// MARK: - Response schema (subset of the Analysis_Engine.md response object)

/// One entry of the analysis response `moveInfos` array. Only the fields WP-14 consumes are decoded;
/// unknown fields (pv, lcb, order, utility, ...) are ignored.
public struct AnalysisMoveInfo: Decodable, Sendable {
    /// GTP-style move string ("C4", "pass"). Parsed via `Location.ofString`.
    public let move: String
    /// Search visit count for this move (the raw material for the visit-distribution policy target).
    public let visits: Int
    /// Move winrate in [0, 1], from the config's `reportAnalysisWinratesAs` perspective. Used as the
    /// per-child Q for the completed-Q policy target.
    public let winrate: Double
    public let prior: Double?
    public let scoreLead: Double?
}

/// The analysis response `rootInfo` object (position-level statistics). Only WP-14's fields decode.
public struct AnalysisRootInfo: Decodable, Sendable {
    /// Search-backed root winrate in [0, 1], perspective = `reportAnalysisWinratesAs`. This is the
    /// value target source (a deep search's backed-up winrate is a far stronger label than the raw
    /// net's).
    public let winrate: Double
    /// Points the perspective player leads by (perspective = `reportAnalysisWinratesAs`).
    public let scoreLead: Double
    /// Raw-net (searchless) winrate in [0, 1]; the completed-Q `v_hat`. Optional -- only needed for
    /// the completed-Q policy target.
    public let rawWinrate: Double?
    /// "B" or "W": the actual side to move at this turn (perspective-independent). Used as a turn
    /// alignment cross-check against our own SGF replay.
    public let currentPlayer: String?
    public let visits: Int?
}

/// One line of the analysis engine's JSON-lines output. Every field is optional so that error /
/// warning / interim lines decode without throwing and can be classified and skipped.
public struct AnalysisResponse: Decodable, Sendable {
    public let id: String?
    public let turnNumber: Int?
    /// True on interim reports emitted mid-search (`reportDuringSearchEvery`); such lines carry
    /// preliminary numbers and MUST be skipped -- only the final report per turn is a label.
    public let isDuringSearch: Bool?
    public let moveInfos: [AnalysisMoveInfo]?
    public let rootInfo: AnalysisRootInfo?
    /// Length `boardYSize*boardXSize`, values in [-1, 1], row-major top-left->bottom-right, +1 = the
    /// `reportAnalysisWinratesAs` player owns. Present iff the query set `includeOwnership`.
    public let ownership: [Double]?
    /// Length `boardYSize*boardXSize+1`, raw-net policy, positive entries summing to 1, `-1` for
    /// illegal, pass last. Present iff the query set `includePolicy`.
    public let policy: [Double]?
    public let error: String?
    public let warning: String?
}

/// Perspective the analysis engine used for `winrate`/`scoreLead`/`ownership`. MUST match the engine
/// config's `reportAnalysisWinratesAs`. The raw values are the `-winrates-as` CLI spellings.
public enum AnalysisWinratePerspective: String, Sendable, CaseIterable {
    case black
    case white
    case sidetomove

    /// The concrete player whose perspective the reported values are in, given the side to move at
    /// the analyzed turn. For `sidetomove` this is exactly the side to move (so no flip is ever
    /// needed); for `black`/`white` it is that fixed color.
    public func reportPlayer(sideToMove: Player) -> Player {
        switch self {
        case .black: .black
        case .white: .white
        case .sidetomove: sideToMove
        }
    }
}

// MARK: - Label conversion (pure, unit-testable)

/// Converts one analysis response into RinGoData v2 targets from the SIDE-TO-MOVE perspective.
public enum AnalysisLabeler {
    /// Builds v2 targets from an analysis response evaluated at a position whose side to move is
    /// `nextPlayer`. Returns `nil` when there is nothing to label (no `moveInfos`), which the caller
    /// counts and skips.
    ///
    /// Perspective reprojection (all citations from KataGo `docs/Analysis_Engine.md`):
    /// - The doc states "All values will be from the perspective of `reportAnalysisWinratesAs`".
    ///   `perspective.reportPlayer(sideToMove:)` resolves that to a concrete player `P`; we flip a
    ///   field iff `P != nextPlayer` (i.e. the engine reported from the opponent's side).
    /// - **value**: `rootInfo.winrate` is P(P wins) in [0, 1] (doc: "winrate ... a float in [0,1]").
    ///   Side-to-move win = winrate if `P == nextPlayer` else `1 - winrate`. Stored as
    ///   `[win, 1-win, 0]`: noResult is folded into the draw-adjusted winrate and set to 0, exactly
    ///   as `TeacherLabeler.searchTargets` does (negligible on 9x9 with positive komi; re-adding the
    ///   raw net's noResult would double-count it).
    /// - **score**: `rootInfo.scoreLead` is "the predicted average number of points that the
    ///   [perspective] side is leading by"; negate for a flip. Stored in points (the loss divides by
    ///   20; matches the teacherless / teacher paths).
    /// - **ownership**: `ownership[i]` in [-1, 1] with +1 = P owns (doc: length
    ///   `boardYSize*boardXSize`, row-major top-left->bottom-right); negate for a flip, then quantize
    ///   to `{-1, 0, +1}` by rounding -- the same `{-1,0,1}` alphabet v2 requires (`LossComputer`
    ///   decodes each Int8 via `(v+1)/2` into a BCE target). Row-major index `y*boardXSize + x` maps
    ///   to dense position `y*nnLen + x` (identical for our square nnLen==boardXSize case).
    /// - **policy**: `.visits` (default) is the normalized `moveInfos` visit distribution, legal by
    ///   construction (only legal moves are searched), pass included, summing to 1 -- the AlphaZero
    ///   target and what a deep (1000+ visit) analysis is for. `.completedQ` reuses WP-13's
    ///   `TeacherLabeler.completedQPolicy` via `searchTargets` (see below). Policy is
    ///   perspective-independent (indexed by board position), so no flip.
    public static func targets(
        response: AnalysisResponse,
        nextPlayer: Player,
        boardXSize: Int,
        boardYSize: Int,
        nnLen: Int,
        perspective: AnalysisWinratePerspective,
        policyKind: TeacherLabeler.PolicyTargetKind = .visits
    ) -> TeacherLabeler.Targets? {
        guard let root = response.rootInfo, let moveInfos = response.moveInfos, !moveInfos.isEmpty else {
            return nil
        }
        let area = nnLen * nnLen
        let reportPlayer = perspective.reportPlayer(sideToMove: nextPlayer)
        let flip = reportPlayer != nextPlayer

        // (loc, visits) for every searched move; also used as the completed-Q visit source.
        var visitsByMove = [(loc: Loc, visits: Int64)]()
        visitsByMove.reserveCapacity(moveInfos.count)
        for info in moveInfos {
            guard let loc = Location.ofString(info.move, boardXSize, boardYSize), info.visits > 0 else { continue }
            visitsByMove.append((loc, Int64(info.visits)))
        }

        // value: side-to-move win/loss, noResult folded to 0.
        let sideToMoveWin = Float(flip ? 1 - root.winrate : root.winrate)
        let value: [Float] = [sideToMoveWin, 1 - sideToMoveWin, 0]

        // score: side-to-move points lead.
        let score = Float(flip ? -root.scoreLead : root.scoreLead)

        // ownership: reproject + quantize to {-1, 0, +1}, mapped row-major -> dense y*nnLen+x.
        var ownership = [Int8](repeating: 0, count: area)
        if let map = response.ownership, map.count == boardYSize * boardXSize {
            for y in 0 ..< min(nnLen, boardYSize) {
                for x in 0 ..< min(nnLen, boardXSize) {
                    let reported = map[y * boardXSize + x]
                    let sideToMove = flip ? -reported : reported
                    ownership[y * nnLen + x] = quantizeOwnership(sideToMove)
                }
            }
        }

        // policy.
        let policy: [Float] = switch policyKind {
        case .visits:
            visitsPolicy(visitsByMove: visitsByMove, boardXSize: boardXSize, nnLen: nnLen)
        case .completedQ:
            completedQPolicy(
                response: response, moveInfos: moveInfos, visitsByMove: visitsByMove,
                nextPlayer: nextPlayer, reportPlayer: reportPlayer, boardXSize: boardXSize,
                boardYSize: boardYSize, nnLen: nnLen
            )
        }
        // A position whose searched moves all failed to parse (defensive) has no usable policy.
        guard policy.contains(where: { $0 > 0 }) else { return nil }

        return TeacherLabeler.Targets(policy: policy, value: value, score: score, ownership: ownership)
    }

    /// Nearest-integer quantization of a continuous [-1, 1] ownership value into the v2 `{-1, 0, +1}`
    /// Int8 alphabet, clamped (mirrors `TeacherLabeler.targets`).
    private static func quantizeOwnership(_ value: Double) -> Int8 {
        Int8(max(-1.0, min(1.0, value.rounded())))
    }

    /// Normalized root visit distribution as a dense `area+1` policy (pass at index `area`). Byte-for
    /// byte the same construction as `TeacherLabeler.searchTargets`'s visits branch.
    private static func visitsPolicy(
        visitsByMove: [(loc: Loc, visits: Int64)],
        boardXSize: Int,
        nnLen: Int
    ) -> [Float] {
        let area = nnLen * nnLen
        var policy = [Float](repeating: 0, count: area + 1)
        let total = visitsByMove.reduce(Int64(0)) { $0 + $1.visits }
        guard total > 0 else { return policy }
        for (loc, visits) in visitsByMove where visits > 0 {
            let pos = NNPos.locToPos(loc, boardXSize, nnLen, nnLen)
            guard pos >= 0, pos < policy.count else { continue }
            policy[pos] += Float(Double(visits) / Double(total))
        }
        return policy
    }

    /// Gumbel completed-Q improved policy (WP-13), reusing `TeacherLabeler.searchTargets` verbatim.
    ///
    /// `completedQPolicy` operates in WHITE-perspective inputs and reprojects to the side to move
    /// itself via `nextPlayer`, so we feed it a synthetic WHITE-perspective root:
    /// - `policyProbs` = the raw-net `policy` array (the prior; `-1` = illegal), mapped row-major ->
    ///   dense NN position. This IS the legal mask completed-Q uses.
    /// - `whiteWinProb`/`whiteLossProb` = raw-net root winrate reprojected to white (the `v_hat`).
    /// - `childWinValues[loc]` = each move's white-perspective win-minus-loss (`2*winrate_white - 1`).
    /// We take ONLY the returned `.policy`; value/score/ownership are computed directly above.
    /// Falls back to the visit distribution when `includePolicy` was off (no prior to improve).
    private static func completedQPolicy(
        response: AnalysisResponse,
        moveInfos: [AnalysisMoveInfo],
        visitsByMove: [(loc: Loc, visits: Int64)],
        nextPlayer: Player,
        reportPlayer: Player,
        boardXSize: Int,
        boardYSize: Int,
        nnLen: Int
    ) -> [Float] {
        let fallback = visitsPolicy(visitsByMove: visitsByMove, boardXSize: boardXSize, nnLen: nnLen)
        let area = nnLen * nnLen
        guard let rawPolicy = response.policy, rawPolicy.count == boardYSize * boardXSize + 1 else {
            return fallback
        }
        let reportIsWhite = reportPlayer == .white

        // Prior in NN-position order (illegal stays -1). Row-major analysis idx -> dense y*nnLen+x.
        var priors = [Float](repeating: -1, count: area + 1)
        for y in 0 ..< min(nnLen, boardYSize) {
            for x in 0 ..< min(nnLen, boardXSize) {
                priors[y * nnLen + x] = Float(rawPolicy[y * boardXSize + x])
            }
        }
        priors[area] = Float(rawPolicy[boardYSize * boardXSize]) // pass

        // v_hat: raw-net root winrate reprojected to white (win-minus-loss handled inside searchTargets).
        let rawReport = response.rootInfo?.rawWinrate ?? 0.5
        let rawWhiteWin = Float(reportIsWhite ? rawReport : 1 - rawReport)
        // Per-child Q in white-perspective win-minus-loss units.
        var childWinValues = [Loc: Double]()
        for info in moveInfos {
            guard let loc = Location.ofString(info.move, boardXSize, boardYSize) else { continue }
            let whiteWin = reportIsWhite ? info.winrate : 1 - info.winrate
            childWinValues[loc] = 2 * whiteWin - 1
        }

        let synthetic = NNOutput(
            nnHash: Hash128(0, 0),
            whiteWinProb: rawWhiteWin,
            whiteLossProb: 1 - rawWhiteWin,
            whiteNoResultProb: 0,
            whiteScoreMean: 0,
            whiteScoreMeanSq: 0,
            whiteLead: 0,
            varTimeLeft: -1,
            shorttermWinlossError: -1,
            shorttermScoreError: -1,
            policyProbs: priors,
            policyOptimismUsed: 0,
            nnXLen: nnLen,
            nnYLen: nnLen,
            whiteOwnerMap: nil
        )
        let reused = TeacherLabeler.searchTargets(
            visitsByMove: visitsByMove,
            whiteWinProb: 0.5, // unused for the policy we extract
            rootOutput: synthetic,
            boardXSize: boardXSize,
            nextPlayer: nextPlayer,
            nnLen: nnLen,
            target: .completedQ,
            childWinValues: childWinValues
        )
        return reused.policy
    }
}
