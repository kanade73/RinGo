import Foundation
import RinGoCore

/// Score-utility math ported from `$KG/cpp/neuralnet/nninputs.cpp` (`namespace ScoreValue`).
///
/// One deliberate deviation, documented per design.md: `expectedWhiteScoreValue` evaluates the
/// same Gaussian-weighted integral of the atan score-value curve that KataGo tabulates
/// (`ScoreValue::initTables`, stepsPerUnit=10, boundStdevs=5), but computes it directly at the
/// query point instead of building KataGo's ~350k-entry table at startup and bilinearly
/// interpolating grid-rounded (mean, stdev). Direct evaluation agrees with the table to ~1e-3
/// (it is the table's own integrand, minus the grid rounding), which is far inside search
/// noise, and avoids a multi-second table build in debug test runs.
enum ScoreValueUtils {
    static let twoOverPi = 0.63661977236758134308

    /// `ScoreValue::whiteWinsOfWinner`. `winner == nil` means a draw.
    static func whiteWinsOfWinner(_ winner: Player?, _ drawEquivalentWinsForWhite: Double) -> Double {
        switch winner {
        case .white: 1.0
        case .black: 0.0
        case nil: drawEquivalentWinsForWhite
        }
    }

    /// `ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust`.
    static func whiteScoreValueOfScoreSmoothNoDrawAdjust(
        _ finalWhiteMinusBlackScore: Double,
        center: Double,
        scale: Double,
        sqrtBoardArea: Double
    ) -> Double {
        let adjustedScore = finalWhiteMinusBlackScore - center
        return atan(adjustedScore / (scale * sqrtBoardArea)) * twoOverPi
    }

    /// Normal-PDF weights at 0.1-stdev steps over ±5 stdevs, matching KataGo's integration grid
    /// (`stepsPerUnit = 10`, `boundStdevs = 5` in `ScoreValue::initTables`).
    private static let normalWeights: (weights: [Double], sum: Double) = {
        var weights = [Double]()
        weights.reserveCapacity(101)
        var sum = 0.0
        for i in -50 ... 50 {
            let xInStdevs = Double(i) / 10.0
            let weight = exp(-0.5 * xInStdevs * xInStdevs)
            weights.append(weight)
            sum += weight
        }
        return (weights, sum)
    }()

    /// `ScoreValue::expectedWhiteScoreValue` (see the type doc comment for the one deviation).
    static func expectedWhiteScoreValue(
        whiteScoreMean: Double,
        whiteScoreStdev: Double,
        center: Double,
        scale: Double,
        sqrtBoardArea: Double
    ) -> Double {
        guard whiteScoreStdev > 0 else {
            return whiteScoreValueOfScoreSmoothNoDrawAdjust(
                whiteScoreMean,
                center: center,
                scale: scale,
                sqrtBoardArea: sqrtBoardArea
            )
        }
        let (weights, weightSum) = normalWeights
        var weightedSum = 0.0
        for (index, weight) in weights.enumerated() {
            let xInStdevs = Double(index - 50) / 10.0
            let score = whiteScoreMean + whiteScoreStdev * xInStdevs
            weightedSum += weight * whiteScoreValueOfScoreSmoothNoDrawAdjust(
                score,
                center: center,
                scale: scale,
                sqrtBoardArea: sqrtBoardArea
            )
        }
        return weightedSum / weightSum
    }

    /// `ScoreValue::getScoreStdev`.
    static func getScoreStdev(_ scoreMean: Double, _ scoreMeanSq: Double) -> Double {
        let variance = scoreMeanSq - scoreMean * scoreMean
        guard variance > 0 else { return 0 }
        return variance.squareRoot()
    }

    /// `ScoreValue::whiteScoreMeanSqOfScoreGridded`. `finalWhiteMinusBlackScore` is always an
    /// integer or half-integer for scored games.
    static func whiteScoreMeanSqOfScoreGridded(
        _ finalWhiteMinusBlackScore: Double,
        _ drawEquivalentWinsForWhite: Double
    ) -> Double {
        let isInteger = finalWhiteMinusBlackScore == finalWhiteMinusBlackScore.rounded(.towardZero)
        guard isInteger else {
            return finalWhiteMinusBlackScore * finalWhiteMinusBlackScore
        }
        let lower = finalWhiteMinusBlackScore - 0.5
        let upper = finalWhiteMinusBlackScore + 0.5
        let lowerSq = lower * lower
        let upperSq = upper * upper
        return lowerSq + (upperSq - lowerSq) * drawEquivalentWinsForWhite
    }
}
