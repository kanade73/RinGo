import RinGoCore
@testable import RinGoEngine
import XCTest

/// Round-trip + probe tests for the runtime `OpeningBook` table (format serialize/parse and
/// hash-keyed lookup against a live board position).
final class OpeningBookTests: XCTestCase {
    private func situationHash(_ moves: [(Int, Int, Player)]) -> (Hash128, Board, Player) {
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: Rules(), encorePhase: 0)
        for (x, y, pla) in moves {
            let loc = Location.getLoc(x, y, 9)
            history.makeBoardMoveAssumeLegal(board, loc, pla)
        }
        let next = history.presumedNextMovePla
        return (BoardHistory.getSituationAndSimpleKoHash(board, nextPlayer: next), board, next)
    }

    func testSerializeParseRoundTrip() throws {
        let (h1, _, _) = situationHash([])
        let (h2, _, _) = situationHash([(4, 4, .black)])
        let table: [Hash128: OpeningBook.Entry] = [
            h1: OpeningBook.Entry(loc: Location.getLoc(4, 4, 9), wl: 0.033, visits: 1.5e14),
            h2: OpeningBook.Entry(loc: Location.getLoc(2, 2, 9), wl: -0.12, visits: 4.2e5),
        ]
        let book = OpeningBook(boardXSize: 9, boardYSize: 9, minVisits: 1e4, table: table)
        let parsed = try OpeningBook.parse(book.serialized())

        XCTAssertEqual(parsed.count, 2)
        XCTAssertEqual(parsed.boardXSize, 9)
        XCTAssertEqual(parsed.boardYSize, 9)
        XCTAssertEqual(parsed.minVisits, 1e4)
        XCTAssertEqual(parsed.lookup(hash: h1)?.loc, Location.getLoc(4, 4, 9))
        XCTAssertEqual(parsed.lookup(hash: h1)?.wl ?? 0, 0.033, accuracy: 1e-6)
        XCTAssertEqual(parsed.lookup(hash: h2)?.loc, Location.getLoc(2, 2, 9))
        XCTAssertNil(parsed.lookup(hash: Hash128(1, 2)))
    }

    func testLookupByLiveBoardMatchesConversionKey() throws {
        // The table is keyed exactly as the engine keys at runtime, so a board reached by real moves
        // must hit the entry stored under its situational hash.
        let (hash, _, _) = situationHash([(4, 4, .black), (2, 6, .white)])
        let table = [hash: OpeningBook.Entry(loc: Location.getLoc(6, 2, 9), wl: 0.1, visits: 1e5)]
        let book = try OpeningBook.parse(
            OpeningBook(boardXSize: 9, boardYSize: 9, minVisits: 0, table: table).serialized()
        )

        // Rebuild the same position independently and probe with the live board.
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: Rules(), encorePhase: 0)
        history.makeBoardMoveAssumeLegal(board, Location.getLoc(4, 4, 9), .black)
        history.makeBoardMoveAssumeLegal(board, Location.getLoc(2, 6, 9), .white)
        let entry = book.lookup(board: board, nextPlayer: history.presumedNextMovePla)
        XCTAssertEqual(entry?.loc, Location.getLoc(6, 2, 9))
    }

    func testParseRejectsBadMagic() {
        var bytes = [UInt8]("XXXXXXXX".utf8)
        bytes.append(contentsOf: [UInt8](repeating: 0, count: 24))
        XCTAssertThrowsError(try OpeningBook.parse(Data(bytes)))
    }

    func testParseRejectsTruncated() {
        XCTAssertThrowsError(try OpeningBook.parse(Data([0, 1, 2])))
    }
}
