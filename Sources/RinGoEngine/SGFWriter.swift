import Foundation
import RinGoCore

public enum SGFWriterError: Error, CustomStringConvertible, Equatable {
    case unfinishedGame
    case unscoredGame
    case unsupportedInitialPosition

    public var description: String {
        switch self {
        case .unfinishedGame: "SGFWriter requires a finished game"
        case .unscoredGame: "SGFWriter requires a scored, non-resignation result"
        case .unsupportedInitialPosition: "SGFWriter supports games that start from an empty board"
        }
    }
}

/// Serializes a completed played game as one FF[4] SGF main line. The date is injected so tests
/// and seeded self-play runs can be byte-reproducible; this type never reads the system clock.
public enum SGFWriter {
    public static func write(
        history: BoardHistory,
        blackName: String,
        whiteName: String,
        date: String
    ) throws -> String {
        guard history.isGameFinished else { throw SGFWriterError.unfinishedGame }
        guard history.isScored, !history.isResignation, !history.isNoResult else {
            throw SGFWriterError.unscoredGame
        }
        guard history.initialBoard.isEmpty() else { throw SGFWriterError.unsupportedInitialPosition }

        let board = history.initialBoard
        let size = board.xSize == board.ySize ? "\(board.xSize)" : "\(board.xSize):\(board.ySize)"
        var result = "(;GM[1]FF[4]CA[UTF-8]"
        result += "SZ[\(size)]KM[\(formatNumber(history.rules.komi))]RU[Chinese]"
        result += "PB[\(escape(blackName))]PW[\(escape(whiteName))]DT[\(escape(date))]"
        result += "RE[\(formatResult(history.finalWhiteMinusBlackScore))]"
        for move in history.moveHistory {
            let color = move.pla == .black ? "B" : "W"
            result += ";\(color)[\(coordinate(move.loc, board: board))]"
        }
        return result + ")\n"
    }

    private static func coordinate(_ loc: Loc, board: Board) -> String {
        guard loc != Board.passLoc else { return "" }
        let x = Location.getX(loc, board.xSize)
        let y = Location.getY(loc, board.xSize)
        return String(sgfCharacter(x)) + String(sgfCharacter(y))
    }

    private static func sgfCharacter(_ value: Int) -> Character {
        let base = value < 26 ? UInt8(ascii: "a") : UInt8(ascii: "A") - 26
        return Character(UnicodeScalar(base + UInt8(value)))
    }

    private static func formatResult(_ whiteMinusBlack: Float) -> String {
        if whiteMinusBlack == 0 { return "0" }
        let prefix = whiteMinusBlack > 0 ? "W+" : "B+"
        return prefix + formatNumber(abs(whiteMinusBlack))
    }

    private static func formatNumber(_ value: Float) -> String {
        value == value.rounded(.towardZero)
            ? String(format: "%.0f", locale: Locale(identifier: "en_US_POSIX"), value)
            : String(format: "%.1f", locale: Locale(identifier: "en_US_POSIX"), value)
    }

    private static func escape(_ value: String) -> String {
        value.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "]", with: "\\]")
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
    }
}
