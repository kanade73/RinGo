import Foundation
@testable import ringo
import XCTest

/// Tests for `ringo makedata`'s pure (I/O-free) helpers. The full pipeline (SGF parsing,
/// shard writing, optional teacher labeling) needs real fixtures/MLX and is exercised elsewhere
/// (`TrainingDataFormatTests`, host-only teacher tests); this file covers O-4
/// (CODEBASE_REVIEW_0711.md): `MakeDataCommand.highSkipRateWarning` is factored out specifically
/// so this threshold logic is testable without a corpus.
final class MakeDataCommandTests: XCTestCase {
    func testNoWarningWhenSkipRateIsAtOrBelowThirtyPercent() {
        XCTAssertNil(MakeDataCommand.highSkipRateWarning(gamesUsed: 100, gamesSkipped: 0))
        XCTAssertNil(MakeDataCommand.highSkipRateWarning(gamesUsed: 70, gamesSkipped: 30))
    }

    func testWarningWhenSkipRateExceedsThirtyPercent() throws {
        let warning = MakeDataCommand.highSkipRateWarning(gamesUsed: 69, gamesSkipped: 31)
        XCTAssertNotNil(warning)
        XCTAssertTrue(try XCTUnwrap(warning?.contains("WARNING")))
        XCTAssertTrue(try XCTUnwrap(warning?.contains("31.0%")))
        XCTAssertTrue(try XCTUnwrap(warning?.contains("31/100")))
    }

    func testWarningWhenEverythingIsSkipped() throws {
        let warning = MakeDataCommand.highSkipRateWarning(gamesUsed: 0, gamesSkipped: 5)
        XCTAssertNotNil(warning)
        XCTAssertTrue(try XCTUnwrap(warning?.contains("100.0%")))
    }

    func testNoWarningWhenNoGamesAtAll() {
        XCTAssertNil(MakeDataCommand.highSkipRateWarning(gamesUsed: 0, gamesSkipped: 0))
    }

    func testWarningTextReportsPercentageAndCounts() throws {
        let warning = try XCTUnwrap(MakeDataCommand.highSkipRateWarning(gamesUsed: 1, gamesSkipped: 9))
        XCTAssertEqual(
            warning,
            "makedata: WARNING skipped 90.0% of games (9/10) -- check -sgf-dir input quality "
                + "(parse/board-size/min-moves/jigo/result/illegal-move filters)"
        )
    }
}
