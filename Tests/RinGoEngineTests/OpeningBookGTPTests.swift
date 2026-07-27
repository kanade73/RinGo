import Foundation
import RinGoCore
@testable import RinGoEngine
import XCTest

/// End-to-end `genmove` probe test (WP-12). The evaluator here THROWS on any `evaluate`, so a
/// `genmove` that returns a real move proves the book answered before search ran; a `genmove` that
/// errors proves it fell through to search. This drives the exact `GTPEngine.handle` path a
/// tournament game uses, with no model fixture required.
final class OpeningBookGTPTests: XCTestCase {
    private let nnLen = 9

    /// Build the book for a scripted line where the engine is Black: empty board -> tengen, then the
    /// position after {B tengen, W c3} -> the a9 corner. Off-book positions are absent.
    private func makeBook(whiteReplyLoc: Loc) -> (OpeningBook, firstMove: Loc, secondMove: Loc) {
        let firstMove = Location.getLoc(4, 4, 9) // tengen (E5)
        let secondMove = Location.getLoc(0, 0, 9) // A9 corner

        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: Rules(), encorePhase: 0)
        let emptyHash = BoardHistory.getSituationAndSimpleKoHash(board, nextPlayer: .black)

        history.makeBoardMoveAssumeLegal(board, firstMove, .black)
        history.makeBoardMoveAssumeLegal(board, whiteReplyLoc, .white)
        let afterReplyHash = BoardHistory.getSituationAndSimpleKoHash(board, nextPlayer: .black)

        let table: [Hash128: OpeningBook.Entry] = [
            emptyHash: OpeningBook.Entry(loc: firstMove, wl: 0.03, visits: 1e9),
            afterReplyHash: OpeningBook.Entry(loc: secondMove, wl: 0.05, visits: 1e6),
        ]
        return (
            OpeningBook(boardXSize: 9, boardYSize: 9, minVisits: 0, table: table),
            firstMove: firstMove,
            secondMove: secondMove
        )
    }

    private func makeEngine(book: OpeningBook?) -> GTPEngine {
        var settings = SearchSettings()
        settings.maxVisits = 8
        return GTPEngine(
            evaluator: ThrowingEvaluator(),
            nnXLen: nnLen,
            nnYLen: nnLen,
            precision: .float32,
            settings: settings,
            rules: .getTrompTaylorish(),
            nnLength: .fixed(nnLen),
            warmupBucketSizes: [1],
            evaluatorFactory: { _, _ in ThrowingEvaluator() },
            openingBook: book
        )
    }

    func testGenmoveFollowsBookThenFallsThroughOffBook() async throws {
        // Resolve the white reply vertex to the exact loc the engine will parse, so the book key
        // matches the engine's post-move board.
        let whiteReply = try XCTUnwrap(Location.ofString("C3", 9, 9))
        let (book, firstMove, secondMove) = makeBook(whiteReplyLoc: whiteReply)
        let engine = makeEngine(book: book)

        _ = await engine.handle("boardsize 9")
        _ = await engine.handle("clear_board")

        // Ply 1: book hit on the empty board -> tengen, WITHOUT touching the (throwing) evaluator.
        let move1 = await engine.handle("genmove B")
        XCTAssertEqual(move1.response, "= \(Location.toString(firstMove, 9, 9))\n\n")

        // Opponent plays the in-book reply.
        _ = await engine.handle("play W C3")

        // Ply 2: still on-book -> a9.
        let move2 = await engine.handle("genmove B")
        XCTAssertEqual(move2.response, "= \(Location.toString(secondMove, 9, 9))\n\n")

        // Opponent now plays an OFF-book move; the next genmove must fall through to search, which
        // hits the throwing evaluator and surfaces as a GTP error ("? ..."), proving no book hit.
        _ = await engine.handle("play W A1")
        let move3 = await engine.handle("genmove B")
        let response3 = try XCTUnwrap(move3.response)
        XCTAssertTrue(response3.hasPrefix("?"), "off-book genmove should fall through to search, got: \(response3)")
    }

    func testNoBookAlwaysSearches() async throws {
        let engine = makeEngine(book: nil)
        _ = await engine.handle("boardsize 9")
        _ = await engine.handle("clear_board")
        // With no book, the very first genmove must go to search -> throwing evaluator -> "? ...".
        let move = await engine.handle("genmove B")
        let response = try XCTUnwrap(move.response)
        XCTAssertTrue(response.hasPrefix("?"), "no-book genmove should search, got: \(response)")
    }
}

/// An evaluator that refuses to evaluate — used to prove book hits never reach the network.
private actor ThrowingEvaluator: NNEvaluating {
    func evaluate(_: [NNRequest]) async throws -> [NNOutput] {
        throw GTPError("evaluator intentionally unavailable (book hit expected)")
    }

    func preWarm(_: [NNRequest]) {}
}
