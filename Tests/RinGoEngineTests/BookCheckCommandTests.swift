import Foundation
@testable import ringo
import RinGoCore
@testable import RinGoEngine
import XCTest

/// Tests for the `ringo bookcheck` subcommand (WP-30, "does the net already know the book?").
/// The BFS walk itself runs MLX (`KataGoNetwork.forward`) and stays a host-only smoke; the flag
/// parsing, policy-ranking, candidate-move dedup, per-position comparison, and TSV writer are pure
/// Swift and covered here (no GPU).
final class BookCheckCommandTests: XCTestCase {
    // MARK: - Flag parsing

    func testParseRequiresBookAndModel() {
        XCTAssertThrowsError(try BookCheckCommand.parse(["-model", "m.bin.gz"]))
        XCTAssertThrowsError(try BookCheckCommand.parse(["-book", "b.nngb"]))
        XCTAssertThrowsError(try BookCheckCommand.parse([]))
    }

    func testParseDefaults() throws {
        let options = try BookCheckCommand.parse(["-book", "b.nngb", "-model", "m.bin.gz"])
        XCTAssertEqual(options.bookPath, "b.nngb")
        XCTAssertEqual(options.modelPath, "m.bin.gz")
        XCTAssertEqual(options.precision, .float16)
        XCTAssertEqual(options.maxDepth, 20)
        XCTAssertEqual(options.maxPositions, 2000)
        XCTAssertEqual(options.branch, 3)
        XCTAssertEqual(options.nnLen, 9)
        XCTAssertEqual(options.rulesName, "chinese")
        XCTAssertEqual(options.komi, 7.0)
        XCTAssertEqual(options.outPath, "bookcheck.tsv")
    }

    func testParseOverrides() throws {
        let options = try BookCheckCommand.parse([
            "-book", "b.nngb", "-model", "m.bin.gz", "-precision", "fp32",
            "-max-depth", "10", "-max-positions", "500", "-branch", "5",
            "-nnlen", "13", "-rules", "tromp-taylor", "-komi", "6.5", "-out", "out.tsv",
        ])
        XCTAssertEqual(options.precision, .float32)
        XCTAssertEqual(options.maxDepth, 10)
        XCTAssertEqual(options.maxPositions, 500)
        XCTAssertEqual(options.branch, 5)
        XCTAssertEqual(options.nnLen, 13)
        XCTAssertEqual(options.rulesName, "tromp-taylor")
        XCTAssertEqual(options.komi, 6.5, accuracy: 1e-6)
        XCTAssertEqual(options.outPath, "out.tsv")
    }

    func testParseRejectsInvalidValues() {
        XCTAssertThrowsError(try BookCheckCommand.parse(["-book", "b", "-model", "m", "-precision", "int8"]))
        XCTAssertThrowsError(try BookCheckCommand.parse(["-book", "b", "-model", "m", "-max-depth", "0"]))
        XCTAssertThrowsError(try BookCheckCommand.parse(["-book", "b", "-model", "m", "-max-positions", "-1"]))
        XCTAssertThrowsError(try BookCheckCommand.parse(["-book", "b", "-model", "m", "-branch", "0"]))
        XCTAssertThrowsError(try BookCheckCommand.parse(["-book", "b", "-model", "m", "-nnlen", "1"]))
        XCTAssertThrowsError(try BookCheckCommand.parse(["-book", "b", "-model", "m", "-komi", "nope"]))
    }

    // MARK: - rankedLegalPolicy

    func testRankedLegalPolicyFiltersIllegalAndSortsDescending() {
        // nnLen 3 -> 10-slot policy (9 spatial + pass). -1 marks illegal/padded per the C++ sentinel.
        let output = makeOutput(policyProbs: [0.1, -1, 0.5, 0.05, -1, 0.05, 0.1, 0.1, 0.1, -1], nnLen: 3)
        let ranked = BookCheckCommand.rankedLegalPolicy(output)
        XCTAssertEqual(ranked.map(\.pos), [2, 0, 6, 7, 8, 3, 5])
        XCTAssertEqual(ranked.first?.prob ?? 0, 0.5, accuracy: 1e-6)
    }

    func testRankedLegalPolicyBreaksTiesByIndex() {
        let output = makeOutput(policyProbs: [0.2, 0.2, 0.2, -1], nnLen: 1)
        // Board is 1x1 here just to keep the policy vector short; ties should resolve by ascending pos.
        let ranked = BookCheckCommand.rankedLegalPolicy(output)
        XCTAssertEqual(ranked.map(\.pos), [0, 1, 2])
    }

    // MARK: - candidateMoves

    func testCandidateMovesDedupsBookMoveAgainstNetTopBranch() {
        let board = Board(3, 3)
        let bookLoc = Location.getLoc(0, 0, 3)
        let entry = OpeningBook.Entry(loc: bookLoc, wl: 0.1, visits: 100)
        // Net's top-2 legal moves happen to be pos 0 (== the book move, (0,0)) and pos 4 ((1,1)).
        let ranked: [(pos: Int, prob: Float)] = [(0, 0.6), (4, 0.3), (8, 0.1)]
        let candidates = BookCheckCommand.candidateMoves(
            entry: entry,
            ranked: ranked,
            board: board,
            branch: 2,
            nnLen: 3
        )
        // Book move + net's top-2, deduplicated (pos 0 counted once) -> book move, then (1,1).
        XCTAssertEqual(candidates, [bookLoc, Location.getLoc(1, 1, 3)])
    }

    func testCandidateMovesRespectsBranchCount() {
        let board = Board(3, 3)
        let entry = OpeningBook.Entry(loc: Board.passLoc, wl: 0, visits: 1)
        let ranked: [(pos: Int, prob: Float)] = [(0, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)]
        let candidates = BookCheckCommand.candidateMoves(
            entry: entry,
            ranked: ranked,
            board: board,
            branch: 3,
            nnLen: 3
        )
        XCTAssertEqual(candidates, [
            Board.passLoc, Location.getLoc(0, 0, 3), Location.getLoc(1, 0, 3), Location.getLoc(2, 0, 3),
        ])
    }

    // MARK: - makeRow

    func testMakeRowAgreesWhenNetTop1IsBookMove() {
        let board = Board(3, 3)
        let history = BoardHistory(board, pla: .black, rules: Rules(), encorePhase: 0)
        let bookLoc = Location.getLoc(1, 1, 3) // center -> spatial pos 4 for nnLen 3
        let entry = OpeningBook.Entry(loc: bookLoc, wl: 0.25, visits: 5000)
        let node = BookCheckCommand.Node(board: board, history: history, nextPlayer: .black, ply: 1, entry: entry)
        let output = makeOutput(policyProbs: [0.05, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05, 0.05, 0.05, -1], nnLen: 3)
        let ranked = BookCheckCommand.rankedLegalPolicy(output)

        let (row, bookMoveLegal) = BookCheckCommand.makeRow(node: node, output: output, ranked: ranked, nnLen: 3)
        XCTAssertTrue(bookMoveLegal)
        XCTAssertTrue(row.agree)
        XCTAssertEqual(row.bookMoveRank, 1)
        XCTAssertEqual(row.netTopProb, 0.6, accuracy: 1e-6)
        XCTAssertEqual(row.netBookProb, 0.6, accuracy: 1e-6)
        XCTAssertEqual(row.bookMove, Location.toString(bookLoc, 3, 3))
        XCTAssertEqual(row.netTopMove, row.bookMove)
        XCTAssertEqual(row.ply, 1)
        XCTAssertEqual(row.bookVisits, 5000)
        XCTAssertEqual(row.bookWLBlack, 0.25, accuracy: 1e-6)
    }

    func testMakeRowDisagreesAndRanksBookMoveCorrectly() {
        let board = Board(3, 3)
        let history = BoardHistory(board, pla: .black, rules: Rules(), encorePhase: 0)
        let bookLoc = Location.getLoc(2, 2, 3) // pos 8, the net's 3rd choice below
        let entry = OpeningBook.Entry(loc: bookLoc, wl: -0.1, visits: 10)
        let node = BookCheckCommand.Node(board: board, history: history, nextPlayer: .black, ply: 3, entry: entry)
        // pos 0 = 0.5 (top), pos 4 = 0.3 (2nd), pos 8 (book move) = 0.05 (should rank behind both).
        let output = makeOutput(
            policyProbs: [0.5, 0.05, 0.05, 0.05, 0.3, 0, 0, 0, 0.05, -1], nnLen: 3
        )
        let ranked = BookCheckCommand.rankedLegalPolicy(output)

        let (row, bookMoveLegal) = BookCheckCommand.makeRow(node: node, output: output, ranked: ranked, nnLen: 3)
        XCTAssertTrue(bookMoveLegal)
        XCTAssertFalse(row.agree)
        XCTAssertEqual(row.netTopMove, Location.toString(Location.getLoc(0, 0, 3), 3, 3))
        XCTAssertGreaterThan(row.bookMoveRank, 1)
        XCTAssertEqual(row.netBookProb, 0.05, accuracy: 1e-6)
    }

    func testMakeRowHandlesIllegalBookMoveGracefully() {
        let board = Board(3, 3)
        let history = BoardHistory(board, pla: .black, rules: Rules(), encorePhase: 0)
        let bookLoc = Location.getLoc(1, 0, 3) // pos 1, marked illegal (-1) in the policy below
        let entry = OpeningBook.Entry(loc: bookLoc, wl: 0, visits: 1)
        let node = BookCheckCommand.Node(board: board, history: history, nextPlayer: .black, ply: 5, entry: entry)
        let output = makeOutput(
            policyProbs: [0.5, -1, 0.05, 0.05, 0.3, 0.05, 0.03, 0.01, 0.01, -1], nnLen: 3
        )
        let ranked = BookCheckCommand.rankedLegalPolicy(output)

        let (row, bookMoveLegal) = BookCheckCommand.makeRow(node: node, output: output, ranked: ranked, nnLen: 3)
        XCTAssertFalse(bookMoveLegal)
        XCTAssertFalse(row.agree)
        XCTAssertEqual(row.netBookProb, 0, accuracy: 1e-6)
        XCTAssertEqual(row.bookMoveRank, ranked.count + 1)
    }

    // MARK: - writeTSV

    func testWriteTSVRoundTrips() throws {
        let rows = [
            BookCheckCommand.Row(
                ply: 1, bookMove: "E5", bookVisits: 12345, bookWLBlack: 0.05,
                netTopMove: "E5", netTopProb: 0.42, netBookProb: 0.42, bookMoveRank: 1, agree: true
            ),
            BookCheckCommand.Row(
                ply: 2, bookMove: "pass", bookVisits: 10, bookWLBlack: -0.3,
                netTopMove: "D4", netTopProb: 0.3, netBookProb: 0.01, bookMoveRank: 7, agree: false
            ),
        ]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bookcheck-test-\(UUID()).tsv")
        defer { try? FileManager.default.removeItem(at: url) }
        try BookCheckCommand.writeTSV(rows, to: url)

        let text = try String(contentsOf: url, encoding: .utf8)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        XCTAssertEqual(
            lines[0],
            "ply\tbook_move\tbook_visits\tbook_wl_black\tnet_top1_move\tnet_top1_prob\tnet_book_prob\tbook_move_rank\tagree"
        )
        XCTAssertTrue(lines[1].hasPrefix("1\tE5\t12345.0\t0.0500\tE5\t0.4200\t0.4200\t1\t1"))
        XCTAssertTrue(lines[2].hasPrefix("2\tpass\t10.0\t-0.3000\tD4\t0.3000\t0.0100\t7\t0"))
    }

    // MARK: - Fixtures

    private func makeOutput(policyProbs: [Float], nnLen: Int) -> NNOutput {
        NNOutput(
            nnHash: Hash128(),
            whiteWinProb: 0.5,
            whiteLossProb: 0.5,
            whiteNoResultProb: 0,
            whiteScoreMean: 0,
            whiteScoreMeanSq: 0,
            whiteLead: 0,
            varTimeLeft: -1,
            shorttermWinlossError: -1,
            shorttermScoreError: -1,
            policyProbs: policyProbs,
            policyOptimismUsed: 0,
            nnXLen: nnLen,
            nnYLen: nnLen,
            whiteOwnerMap: nil
        )
    }
}
