import Foundation
@testable import ringo
import XCTest

/// Flag parsing for `ringo averagemodels` (SWA, BREAKTHROUGH_IDEAS E4). Pure Swift — the
/// model load / average / write path runs elsewhere; here we pin the CLI contract.
final class AverageModelsCommandTests: XCTestCase {
    func testParsesCommaSeparatedModelsAndOutput() throws {
        let options = try AverageModelsCommand.parse([
            "-models", "a.bin.gz,b.bin.gz,c.bin.gz", "-out", "avg.bin.gz",
        ])
        XCTAssertEqual(options.models, ["a.bin.gz", "b.bin.gz", "c.bin.gz"])
        XCTAssertEqual(options.output, "avg.bin.gz")
    }

    func testTrimsWhitespaceAndDropsEmptyModelEntries() throws {
        let options = try AverageModelsCommand.parse([
            "-models", " a.bin.gz , b.bin.gz ,", "-out", "avg.bin.gz",
        ])
        XCTAssertEqual(options.models, ["a.bin.gz", "b.bin.gz"])
    }

    func testFewerThanTwoModelsIsRejected() {
        XCTAssertThrowsError(try AverageModelsCommand.parse(["-models", "only.bin.gz", "-out", "avg.bin.gz"]))
    }

    func testMissingOutputIsRejected() {
        XCTAssertThrowsError(try AverageModelsCommand.parse(["-models", "a.bin.gz,b.bin.gz"]))
    }

    func testUnknownFlagIsRejected() {
        XCTAssertThrowsError(
            try AverageModelsCommand.parse(["-models", "a.bin.gz,b.bin.gz", "-out", "avg.bin.gz", "-bogus", "x"])
        )
    }

    func testMissingValueForFlagIsRejected() {
        XCTAssertThrowsError(try AverageModelsCommand.parse(["-models"]))
    }
}
