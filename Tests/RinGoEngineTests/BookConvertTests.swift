@testable import ringo
import RinGoCore
import RinGoEngine
import XCTest

/// Tests for `bookconvert`: the page parser (mechanics) and the converter's symmetry-composition
/// walk (semantics). The walk test uses a small SYNTHETIC book — not katagobooks.org content — so
/// it is self-contained and license-clean, while still exercising the real `A' = compose(A,
/// invert(linkSym))` update and the board-array validation against a non-trivial symmetry (5/6, the
/// discriminating rotate-90 pair: a wrong update formula fails the validation loudly).
final class BookConvertTests: XCTestCase {
    // MARK: - Page format helpers (KataGo layout, fabricated data)

    private func boardString(_ cells: [Int]) -> String {
        "[" + cells.map(String.init).joined(separator: ",") + ",]"
    }

    private func page(pla: Int, board: [Int], links: [Int: String], syms: [Int: Int], moves: String) -> String {
        let linksStr = "{" + links.sorted { $0.key < $1.key }.map { "\($0.key):'\($0.value)'," }.joined() + "}"
        let symsStr = "{" + syms.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)," }.joined() + "}"
        return """
        <html><header><script>
        const nextPla = \(pla);
        const pLink = '';
        const pSym = 0;
        const board = \(boardString(board));
        const links = \(linksStr);
        const linkSyms = \(symsStr);
        const moves = \(moves);
        </script></header><body></body></html>
        """
    }

    // MARK: - Parser mechanics

    func testParseXYMove() throws {
        let moves = "[{'xy':[[5,3],[5,5],[3,3],[3,5],],'p':0.187,'wl':0.033,'ssM':0.04,'wlRad':0.2,'sRad':0.9,'v':153385547426788,'av':22997034,},]"
        let html = page(pla: 1, board: [Int](repeating: 0, count: 81), links: [:], syms: [:], moves: moves)
        let page = try parsePageContent(html, path: "test")
        XCTAssertEqual(page.nextPla, 1)
        XCTAssertEqual(page.board.count, 81)
        let best = try XCTUnwrap(page.bestMove)
        XCTAssertFalse(best.isPass)
        XCTAssertEqual(best.x, 5)
        XCTAssertEqual(best.y, 3)
        XCTAssertEqual(best.wl, 0.033, accuracy: 1e-9)
        XCTAssertEqual(best.visits, 153_385_547_426_788, accuracy: 1)
    }

    func testParseLinksAndLinkSyms() throws {
        let moves = "[{'xy':[[4,4],],'p':0.1,'wl':0.0,'v':100,},]"
        let html = page(
            pla: 2,
            board: [Int](repeating: 0, count: 81),
            links: [13: "../F2/abc.html", 81: "../EF/pass.html"],
            syms: [13: 3, 81: 0],
            moves: moves
        )
        let page = try parsePageContent(html, path: "test")
        XCTAssertEqual(page.nextPla, 2)
        XCTAssertEqual(page.links[13], "../F2/abc.html")
        XCTAssertEqual(page.links[81], "../EF/pass.html")
        XCTAssertEqual(page.linkSyms[13], 3)
        XCTAssertEqual(page.linkSyms[81], 0)
    }

    func testParsePassMove() throws {
        let moves = "[{'move':'pass','p':0.0,'wl':0.99,'ssM':13.7,'v':8557294,},]"
        let html = page(pla: 1, board: [Int](repeating: 0, count: 81), links: [:], syms: [:], moves: moves)
        let page = try parsePageContent(html, path: "test")
        let best = try XCTUnwrap(page.bestMove)
        XCTAssertTrue(best.isPass)
        XCTAssertEqual(best.wl, 0.99, accuracy: 1e-9)
    }

    func testParseOtherAggregateHasNoBestMove() throws {
        let moves = "[{'move':'other','p':0.14,'wl':0.0,'v':10019,},]"
        let html = page(pla: 1, board: [Int](repeating: 0, count: 81), links: [:], syms: [:], moves: moves)
        let page = try parsePageContent(html, path: "test")
        XCTAssertNil(page.bestMove)
    }

    // MARK: - Converter walk (symmetry composition end-to-end)

    func testConverterWalkValidatesAndMapsThroughSymmetry() throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        // Root: empty board, Black to move. One link at index 1 (the point (1,0)) to the child,
        // recorded under linkSym 5 (rotate-90 — invert is 6, so a wrong walk update would diverge).
        // moves[0] recommends that same (1,0).
        let rootHTML = page(
            pla: 1,
            board: [Int](repeating: 0, count: 81),
            links: [1: "../S/child.html"],
            syms: [1: 5],
            moves: "[{'xy':[[1,0],],'p':0.5,'wl':0.03,'v':1000000000,},]"
        )
        try write(rootHTML, to: tmp.appendingPathComponent("root/root.html"))

        // Child canonical board = getSymBoard(black@(1,0), accumSym) where accumSym = compose(0,
        // invert(5)) = 6. apply((1,0), 6) = (0,7) -> index 63.
        var childBoard = [Int](repeating: 0, count: 81)
        childBoard[63] = 1
        // moves[0] recommends white at display (0,6); its concrete image is apply((0,6), invert(6)=5)
        // = (2,0).
        let childHTML = page(
            pla: 2,
            board: childBoard,
            links: [:],
            syms: [:],
            moves: "[{'xy':[[0,6],],'p':0.9,'wl':-0.4,'v':500000,},]"
        )
        try write(childHTML, to: tmp.appendingPathComponent("S/child.html"))

        let converter = BookConvertCommand.Converter(
            bookDir: tmp.path, minVisits: 0, maxPositions: .max, rules: Rules(), quiet: true
        )
        try converter.run()

        XCTAssertEqual(converter.validationChecks, 2, "both pages must pass board validation")
        XCTAssertEqual(converter.visited.count, 2)
        XCTAssertEqual(converter.entries.count, 2)

        // Root recommendation, keyed by the empty-board situational hash, maps to (1,0).
        let rootHash = BoardHistory.getSituationAndSimpleKoHash(Board(9, 9), nextPlayer: .black)
        XCTAssertEqual(converter.entries[rootHash]?.loc, Location.getLoc(1, 0, 9))

        // Child recommendation, keyed by the real post-move position, maps to concrete (2,0).
        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: Rules(), encorePhase: 0)
        history.makeBoardMoveAssumeLegal(board, Location.getLoc(1, 0, 9), .black)
        let childHash = BoardHistory.getSituationAndSimpleKoHash(board, nextPlayer: .white)
        XCTAssertEqual(converter.entries[childHash]?.loc, Location.getLoc(2, 0, 9))
        XCTAssertEqual(converter.entries[childHash]?.wl ?? 0, -0.4, accuracy: 1e-6)
    }

    func testConverterAbortsOnBoardMismatch() throws {
        let tmp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: tmp) }

        // Deliberately WRONG child board (uses the un-inverted symmetry): with linkSym 5 the child
        // canonical stone belongs at index 63; place it at the wrong cell so validation must abort.
        let rootHTML = page(
            pla: 1,
            board: [Int](repeating: 0, count: 81),
            links: [1: "../S/child.html"],
            syms: [1: 5],
            moves: "[{'xy':[[1,0],],'p':0.5,'wl':0.03,'v':1000000000,},]"
        )
        try write(rootHTML, to: tmp.appendingPathComponent("root/root.html"))
        var wrongBoard = [Int](repeating: 0, count: 81)
        wrongBoard[1] = 1 // stone left in the root frame instead of the canonical (0,7).
        let childHTML = page(
            pla: 2, board: wrongBoard, links: [:], syms: [:],
            moves: "[{'xy':[[0,6],],'p':0.9,'wl':-0.4,'v':500000,},]"
        )
        try write(childHTML, to: tmp.appendingPathComponent("S/child.html"))

        let converter = BookConvertCommand.Converter(
            bookDir: tmp.path, minVisits: 0, maxPositions: .max, rules: Rules(), quiet: true
        )
        XCTAssertThrowsError(try converter.run())
    }

    // MARK: - Real-archive smoke (skips when the local book export is absent)

    /// Walks the actual katagobooks.org export (local, gitignored `.tools/` — never vendored) for a
    /// few hundred positions and asserts: every visited page passed board validation, and the root's
    /// recommended move is the book's own top opening (the near-tengen orbit KataGo prefers on 9x9).
    /// Skipped in CI / anywhere the export is missing.
    func testRealArchiveRootMapsToKnownOpening() throws {
        let bookDir = ProcessInfo.processInfo.environment["RINGO_BOOK_DIR"]
            ?? ".tools/opening-book/html"
        guard FileManager.default.fileExists(atPath: (bookDir as NSString).appendingPathComponent("root/root.html"))
        else {
            throw XCTSkip("opening-book export not present at \(bookDir)")
        }

        let converter = BookConvertCommand.Converter(
            bookDir: bookDir, minVisits: 10000, maxPositions: 300, rules: Rules(), quiet: true
        )
        try converter.run()

        // The walk-validation held for every page it read (the core correctness gate) and nothing
        // was skipped for being unreadable.
        XCTAssertEqual(converter.validationChecks, converter.pagesRead)
        XCTAssertEqual(converter.unreadablePages, 0)
        XCTAssertGreaterThan(converter.entries.count, 0)

        // Root's recommendation must be one of the four near-tengen points the book ranks first
        // (moves[0] orbit {(3,3),(3,5),(5,3),(5,5)} — the 9x9 "4-4/tengen-adjacent" family).
        let rootHash = BoardHistory.getSituationAndSimpleKoHash(Board(9, 9), nextPlayer: .black)
        let rootMove = try XCTUnwrap(converter.entries[rootHash]?.loc)
        let orbit = [(3, 3), (3, 5), (5, 3), (5, 5)].map { Location.getLoc($0.0, $0.1, 9) }
        XCTAssertTrue(
            orbit.contains(rootMove),
            "root move \(Location.toString(rootMove, 9, 9)) not in the top-opening orbit"
        )
    }

    // MARK: - temp helpers

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bookconvert-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ contents: String, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }
}
