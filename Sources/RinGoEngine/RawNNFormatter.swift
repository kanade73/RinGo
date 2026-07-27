import Foundation
import RinGoCore

public enum RawNNFormatter {
    public static func format(
        output: NNOutput,
        board: Board,
        history: BoardHistory
    ) -> String {
        var result = "Rules: \(history.rules)\n"
        result += "Encore phase \(history.encorePhase)\n"
        result += formatBoard(board, history: history.moveHistory)
        result += line("Win %.2fc", Double(output.whiteWinProb) * 100)
        result += line("Loss %.2fc", Double(output.whiteLossProb) * 100)
        result += line("NoResult %.2fc", Double(output.whiteNoResultProb) * 100)
        result += line("ScoreMean %.2f", Double(output.whiteScoreMean))
        result += line("ScoreMeanSq %.1f", Double(output.whiteScoreMeanSq))
        result += line("Lead %.2f", Double(output.whiteLead))
        result += line("VarTimeLeft %.1f", Double(output.varTimeLeft))
        result += line("STWinlossError %.2fc", Double(output.shorttermWinlossError) * 100)
        result += line("STScoreError %.2f", Double(output.shorttermScoreError))
        result += line("OptimismUsed %.2f", Double(output.policyOptimismUsed))
        result += "Policy\n"
        let pass = NNPos.getPassPos(output.nnXLen, output.nnYLen)
        result += String(format: "Pass%4d \n", locale: posix, perMille(output.policyProbs[pass]))
        for y in 0 ..< board.ySize {
            for x in 0 ..< board.xSize {
                let probability = output.policyProbs[NNPos.xyToPos(x, y, output.nnXLen)]
                result += probability < 0
                    ? "   - "
                    : String(format: "%4d ", locale: posix, perMille(probability))
            }
            result += "\n"
        }
        if let ownership = output.whiteOwnerMap {
            for y in 0 ..< board.ySize {
                for x in 0 ..< board.xSize {
                    let value = ownership[NNPos.xyToPos(x, y, output.nnXLen)]
                    result += String(format: "%5d ", locale: posix, perMille(value))
                }
                result += "\n"
            }
            result += "\n"
        }
        return result
    }

    private static let posix = Locale(identifier: "en_US_POSIX")

    private static func line(_ format: String, _ value: Double) -> String {
        String(format: format + "\n", locale: posix, value)
    }

    private static func perMille(_ value: Float) -> Int {
        Int((Double(value) * 1000).rounded(.toNearestOrAwayFromZero))
    }

    private static func formatBoard(_ board: Board, history: [Move]) -> String {
        var result = "MoveNum: \(history.count) HASH: \(board.posHash)\n"
        let letters = Array("ABCDEFGHJKLMNOPQRSTUVWXYZ")
        result += "  "
        for x in 0 ..< board.xSize {
            result += x <= 24 ? " \(letters[x])" : "A\(letters[x - 25])"
        }
        result += "\n"
        let recentStart = max(0, history.count - 3)
        for y in 0 ..< board.ySize {
            result += String(format: "%2d ", locale: posix, board.ySize - y)
            for x in 0 ..< board.xSize {
                let loc = board.getLoc(x, y)
                let stone: Character = switch board.colors[loc] {
                case .black: "X"
                case .white: "O"
                case .empty: "."
                case .wall: "#"
                }
                result.append(stone)
                var marked = false
                for index in recentStart ..< history.count where history[index].loc == loc {
                    result += String(index - recentStart + 1)
                    marked = true
                    break
                }
                if x < board.xSize - 1, !marked { result += " " }
            }
            result += "\n"
        }
        result += "\n"
        return result
    }
}
