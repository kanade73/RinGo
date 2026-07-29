import Foundation
import RinGoCore

/// Formats a `SearchResult` into the single-line `kata-analyze`/`kata-genmove_analyze` payload that
/// KataGo's GTP extensions define (`docs/GTP_Extensions.md` in the KataGo tree). One `info` block per
/// expanded root child — all concatenated on ONE line, KataGo-style — optionally followed by an
/// `ownership` block. This is the analysis LINE only; the caller appends the `play <move>` line.
///
/// Field conventions follow the reference doc: `winrate`/`prior` as floats in [0,1] from the *mover's*
/// perspective, `scoreLead` in points (mover's perspective), `ownership` as boardArea floats in
/// [-1,1] row-major from the top-left (mover's perspective). The CGOS viewer's `AnalyzeResultParser`
/// (`.tools/cgos-client/bin/gtpengine.py`) reads exactly the `move/visits/winrate/scoreLead/prior/pv`
/// tokens plus the trailing `ownership` array; unknown tokens are skipped, so the format is a strict
/// superset-safe subset.
///
/// Limitations kept deliberate to satisfy the "GTP layer + read-only accessors only" containment
/// gate (no `Search.swift` behavior changes): per-move `scoreLead` reports the searched *root* score
/// (`SearchResult.whiteScoreMean`) reprojected to the mover — the same value for every move — because
/// per-child searched score is not exposed on `SearchResult`; `ownership` is the root net's own
/// ownership head (`rootNNOutput.whiteOwnerMap`, the position actually analyzed) rather than a
/// search-averaged map; and each `pv` is just the move itself (deeper PV extraction would require a
/// tree accessor). All three are honest, in-range values suitable for the tournament viewer.
public enum GTPAnalysisFormatter {
    private static let posix = Locale(identifier: "en_US_POSIX")

    /// Builds the analysis line for `result`, from `mover`'s perspective, on a `boardXSize` ×
    /// `boardYSize` board. Emits an `info` block for every expanded root child (descending visit
    /// order, matching `SearchResult.visitsByMove`) and, when `includeOwnership` is true and a root
    /// ownership map exists, a trailing `ownership` block of `boardXSize*boardYSize` floats. Returns
    /// an empty string only if there are no expanded children (e.g. an immediate terminal), in which
    /// case the caller emits just the `play` line.
    public static func analysisLine(
        result: SearchResult,
        mover: Player,
        boardXSize: Int,
        boardYSize: Int,
        includeOwnership: Bool
    ) -> String {
        let nnXLen = result.rootNNOutput?.nnXLen ?? boardXSize
        let nnYLen = result.rootNNOutput?.nnYLen ?? boardYSize
        let moverScore = mover == .white ? result.whiteScoreMean : -result.whiteScoreMean

        var tokens: [String] = []
        tokens.reserveCapacity(result.visitsByMove.count * 14 + boardXSize * boardYSize + 1)

        for (order, entry) in result.visitsByMove.enumerated() {
            let loc = entry.loc
            let moveString = Location.toString(loc, boardXSize, boardYSize)

            // Child value is WHITE-perspective win-minus-loss in [-1,1]; fall back to the root value
            // for any child missing from the map (should not happen for an expanded child).
            let whiteValue = result.rootChildWinValues[loc] ?? (2 * result.whiteWinProb - 1)
            let whiteWinProb = (whiteValue + 1) / 2
            let moverWinrate = mover == .white ? whiteWinProb : 1 - whiteWinProb

            let priorRaw: Float
            if let output = result.rootNNOutput {
                let pos = NNPos.locToPos(loc, boardXSize, nnXLen, nnYLen)
                priorRaw = (pos >= 0 && pos < output.policyProbs.count) ? output.policyProbs[pos] : 0
            } else {
                priorRaw = 0
            }
            let prior = max(0, Double(priorRaw))

            tokens.append("info")
            tokens.append("move"); tokens.append(moveString)
            tokens.append("visits"); tokens.append(String(entry.visits))
            tokens.append("winrate"); tokens.append(number(clamp01(moverWinrate)))
            tokens.append("scoreLead"); tokens.append(number(moverScore))
            tokens.append("prior"); tokens.append(number(clamp01(prior)))
            tokens.append("order"); tokens.append(String(order))
            tokens.append("pv"); tokens.append(moveString)
        }

        if includeOwnership, let ownerMap = result.rootNNOutput?.whiteOwnerMap {
            tokens.append("ownership")
            for y in 0 ..< boardYSize {
                for x in 0 ..< boardXSize {
                    let pos = NNPos.xyToPos(x, y, nnXLen)
                    let whiteOwner = pos < ownerMap.count ? ownerMap[pos] : 0
                    let moverOwner = mover == .white ? whiteOwner : -whiteOwner
                    tokens.append(ownershipNumber(moverOwner))
                }
            }
        }

        return tokens.joined(separator: " ")
    }

    private static func clamp01(_ value: Double) -> Double {
        min(1, max(0, value))
    }

    private static func number(_ value: Double) -> String {
        String(format: "%.6f", locale: posix, value)
    }

    private static func ownershipNumber(_ value: Float) -> String {
        String(format: "%.4f", locale: posix, Double(min(1, max(-1, value))))
    }
}
