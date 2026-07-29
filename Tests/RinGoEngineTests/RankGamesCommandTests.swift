import Foundation
@testable import ringo
import XCTest

/// Tests for the `ringo rankgames` subcommand. The forward
/// pass runs MLX, so it stays a host-only smoke; the flag parsing, KL(teacher||student) direction,
/// per-game aggregation, and TSV writer are pure Swift and covered here (no GPU).
final class RankGamesCommandTests: XCTestCase {
    // MARK: - Flag parsing

    func testParseRequiresSgfDirStudentAndTeacher() {
        XCTAssertThrowsError(try RankGamesCommand.parse(["-student", "s.bin.gz", "-teacher", "t.bin.gz"]))
        XCTAssertThrowsError(try RankGamesCommand.parse(["-sgf-dir", "d", "-teacher", "t.bin.gz"]))
        XCTAssertThrowsError(try RankGamesCommand.parse(["-sgf-dir", "d", "-student", "s.bin.gz"]))
        XCTAssertThrowsError(try RankGamesCommand.parse([]))
    }

    func testParseDefaults() throws {
        let options = try RankGamesCommand.parse([
            "-sgf-dir", "corpus", "-student", "s.bin.gz", "-teacher", "t.bin.gz",
        ])
        XCTAssertEqual(options.sgfDirectory, "corpus")
        XCTAssertEqual(options.studentModel, "s.bin.gz")
        XCTAssertEqual(options.teacherModel, "t.bin.gz")
        XCTAssertEqual(options.studentPrecision, .float32)
        XCTAssertEqual(options.teacherPrecision, .float16)
        XCTAssertEqual(options.batchSize, 256)
        XCTAssertEqual(options.nnLen, 9)
        XCTAssertEqual(options.rulesName, "chinese")
        XCTAssertEqual(options.minimumMoves, 8)
        XCTAssertEqual(options.top, 0)
        XCTAssertNil(options.linkDirectory)
        XCTAssertEqual(options.outputPath, "ranking.tsv")
        XCTAssertNil(options.positionsOutputPath)
    }

    func testParseOverrides() throws {
        let options = try RankGamesCommand.parse([
            "-sgf-dir", "c", "-student", "s", "-teacher", "t",
            "-student-precision", "fp16", "-teacher-precision", "fp32",
            "-batch", "64", "-nnlen", "9", "-rules", "tromp-taylor", "-min-moves", "4",
            "-top", "50", "-link-dir", "picks", "-out", "r.tsv",
            "-positions-out", "positions.tsv",
        ])
        XCTAssertEqual(options.studentPrecision, .float16)
        XCTAssertEqual(options.teacherPrecision, .float32)
        XCTAssertEqual(options.batchSize, 64)
        XCTAssertEqual(options.rulesName, "tromp-taylor")
        XCTAssertEqual(options.minimumMoves, 4)
        XCTAssertEqual(options.top, 50)
        XCTAssertEqual(options.linkDirectory, "picks")
        XCTAssertEqual(options.outputPath, "r.tsv")
        XCTAssertEqual(options.positionsOutputPath, "positions.tsv")
    }

    func testParseRejectsBadValues() {
        let base = ["-sgf-dir", "c", "-student", "s", "-teacher", "t"]
        XCTAssertThrowsError(try RankGamesCommand.parse(base + ["-student-precision", "fp8"]))
        XCTAssertThrowsError(try RankGamesCommand.parse(base + ["-batch", "0"]))
        XCTAssertThrowsError(try RankGamesCommand.parse(base + ["-nnlen", "1"]))
        XCTAssertThrowsError(try RankGamesCommand.parse(base + ["-top", "-1"]))
    }

    // MARK: - KL direction

    func testKLIsZeroForIdenticalDistributions() {
        let logits: [Float] = [1.0, 2.0, 3.0, 0.5]
        let kl = RankGamesCommand.klTeacherStudent(
            student: logits, teacher: logits, offset: 0, legal: [0, 1, 2, 3]
        )
        XCTAssertEqual(kl, 0, accuracy: 1e-5)
    }

    func testKLPenalizesStudentSurpriseAtTeacherFavoredMove() {
        // Teacher strongly favors move 0; the student that gives move 0 near-zero probability must
        // score a much higher KL(teacher||student) than a student that agrees with the teacher.
        let teacher: [Float] = [10, 0, 0]
        let studentDisagrees: [Float] = [0, 0, 10] // mass on move 2, ~none on the teacher's move 0
        let studentAgrees: [Float] = [10, 0, 0]
        let legal = [0, 1, 2]
        let klDisagree = RankGamesCommand.klTeacherStudent(
            student: studentDisagrees, teacher: teacher, offset: 0, legal: legal
        )
        let klAgree = RankGamesCommand.klTeacherStudent(
            student: studentAgrees, teacher: teacher, offset: 0, legal: legal
        )
        XCTAssertGreaterThan(klDisagree, 5) // dominated by p_t(0)*(-log p_s(0)), p_s(0) ~ e^-10
        XCTAssertEqual(klAgree, 0, accuracy: 1e-5)
        XCTAssertGreaterThan(klDisagree, klAgree)
    }

    func testKLRespectsOffsetAndLegalMask() {
        // Row 1 (offset = stride) should be scored independently of row 0, and illegal moves must be
        // excluded from the softmax normalization entirely.
        let stride = 3
        let student: [Float] = [0, 0, 0, /* row1 */ 5, 0, 0]
        let teacher: [Float] = [0, 0, 0, /* row1 */ 0, 0, 5]
        // Only moves 0 and 2 of row 1 are legal (move 1 masked out).
        let kl = RankGamesCommand.klTeacherStudent(
            student: student, teacher: teacher, offset: stride, legal: [0, 2]
        )
        XCTAssertGreaterThan(kl, 0)
        XCTAssertTrue(kl.isFinite)
    }

    // MARK: - Per-game aggregation

    func testScoreGameAveragesTopFive() {
        let (score, top) = RankGamesCommand.scoreGame([1, 2, 3, 4, 5, 6, 7])
        XCTAssertEqual(score, 5, accuracy: 1e-6) // mean of {7,6,5,4,3}
        XCTAssertEqual(top, [6, 5, 4, 3, 2]) // turn indices, highest KL first
    }

    func testScoreGameHandlesFewerThanFivePositions() {
        let (score, top) = RankGamesCommand.scoreGame([3, 1])
        XCTAssertEqual(score, 2, accuracy: 1e-6)
        XCTAssertEqual(top, [0, 1])
    }

    func testScoreGameEmpty() {
        let (score, top) = RankGamesCommand.scoreGame([])
        XCTAssertEqual(score, 0)
        XCTAssertTrue(top.isEmpty)
    }

    // MARK: - TSV writer

    func testWriteRankingEmitsHeaderAndRows() throws {
        let ranked = [
            RankGamesCommand.RankedGame(path: "/a/g1.sgf", score: 1.25, moveCount: 40, topPositions: [3, 7, 1]),
            RankGamesCommand.RankedGame(path: "/a/g2.sgf", score: 0.5, moveCount: 12, topPositions: [0]),
        ]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rankgames-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("ranking.tsv")

        try RankGamesCommand.writeRanking(ranked, to: url)

        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        XCTAssertEqual(lines[0], "path\tscore\tmove_count\ttop_positions")
        XCTAssertEqual(lines[1], "/a/g1.sgf\t1.250000\t40\t3,7,1")
        XCTAssertEqual(lines[2], "/a/g2.sgf\t0.500000\t12\t0")
    }

    // MARK: - Positions TSV writer (-positions-out)

    func testWritePositionsEmitsHeaderAndRows() throws {
        let rows = [
            RankGamesCommand.PositionRow(
                sgfPath: "/a/g1.sgf", turnNumber: 0, kl: 1.25, studentTopMove: "E5", teacherTopMove: "C3"
            ),
            RankGamesCommand.PositionRow(
                sgfPath: "/a/g1.sgf", turnNumber: 7, kl: 0.0, studentTopMove: "pass", teacherTopMove: "pass"
            ),
        ]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rankgames-pos-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("positions.tsv")

        try RankGamesCommand.writePositions(rows, to: url)

        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        XCTAssertEqual(lines[0], "sgf_path\tturn_number\tkl\tstudent_top_move\tteacher_top_move")
        XCTAssertEqual(lines[1], "/a/g1.sgf\t0\t1.250000\tE5\tC3")
        XCTAssertEqual(lines[2], "/a/g1.sgf\t7\t0.000000\tpass\tpass")
    }

    // MARK: - Policy-index -> GTP vertex + argmax

    func testGtpMoveStringMatchesEngineVertexFormat() {
        // On 9x9, pos = y*9 + x maps to the engine's canonical GTP vertex (rows numbered from the
        // bottom, column letters skip "I"); the pass slot (area == 81) formats as "pass".
        XCTAssertEqual(RankGamesCommand.gtpMoveString(pos: 0, nnLen: 9), "A9") // x=0, y=0 (top row)
        XCTAssertEqual(RankGamesCommand.gtpMoveString(pos: 8, nnLen: 9), "J9") // x=8 -> "J" (no "I")
        XCTAssertEqual(RankGamesCommand.gtpMoveString(pos: 40, nnLen: 9), "E5") // center of 9x9
        XCTAssertEqual(RankGamesCommand.gtpMoveString(pos: 81, nnLen: 9), "pass")
    }

    func testArgmaxLegalPicksHighestLogitAmongLegalOnly() {
        // Illegal index 2 has the largest logit but is excluded; among legal {0,1,3} index 1 wins.
        let logits: [Float] = [0.5, 2.0, 9.0, 1.0]
        XCTAssertEqual(RankGamesCommand.argmaxLegal(logits, offset: 0, legal: [0, 1, 3]), 1)
        // Offset addressing: row 1 begins at index 3; among legal {0,2} index 2 (value 4) beats 0.
        let row: [Float] = [0, 0, 0, /* row1 */ 1, 5, 4]
        XCTAssertEqual(RankGamesCommand.argmaxLegal(row, offset: 3, legal: [0, 2]), 2)
    }
}
