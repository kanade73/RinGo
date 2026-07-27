@testable import ringo
import RinGoModel
import XCTest

final class SelfPlayCommandTests: XCTestCase {
    func testParsesDefaultsAndAllFlagsWithoutConstructingEvaluator() throws {
        let defaults = try SelfPlayCommand.parse([
            "-model", "net.bin.gz", "-games", "3", "-out", "sgfs",
        ])
        XCTAssertEqual(defaults.visits, 200)
        XCTAssertEqual(defaults.precision, .float16)
        XCTAssertEqual(defaults.temperature, 0.9)
        XCTAssertEqual(defaults.temperatureMoves, 8)
        XCTAssertEqual(defaults.dirichletAlpha, 0.15)
        XCTAssertEqual(defaults.noiseWeight, 0.25)
        XCTAssertEqual(defaults.seed, 1)
        XCTAssertEqual(defaults.komi, 7)
        XCTAssertEqual(defaults.size, 9)
        XCTAssertEqual(defaults.maxBatchSize, 16)

        let all = try SelfPlayCommand.parse([
            "-model", "net.bin.gz", "-games", "2", "-out", "out", "-visits", "12",
            "-precision", "fp32", "-temp", "1.2", "-temp-moves", "10", "-dirichlet", "0.2",
            "-noise-weight", "0.4", "-seed", "99", "-komi", "7.5", "-size", "13",
            "-max-batch-size", "8",
        ])
        XCTAssertEqual(all.games, 2)
        XCTAssertEqual(all.visits, 12)
        XCTAssertEqual(all.precision, KataGoNetwork.Precision.float32)
        XCTAssertEqual(all.temperature, 1.2)
        XCTAssertEqual(all.temperatureMoves, 10)
        XCTAssertEqual(all.dirichletAlpha, 0.2)
        XCTAssertEqual(all.noiseWeight, 0.4)
        XCTAssertEqual(all.seed, 99)
        XCTAssertEqual(all.komi, 7.5)
        XCTAssertEqual(all.size, 13)
        XCTAssertEqual(all.maxBatchSize, 8)
    }

    func testRejectsMissingAndInvalidArguments() {
        XCTAssertThrowsError(try SelfPlayCommand.parse([]))
        XCTAssertThrowsError(try SelfPlayCommand.parse([
            "-model", "net", "-games", "0", "-out", "out",
        ]))
        XCTAssertThrowsError(try SelfPlayCommand.parse([
            "-model", "net", "-games", "1", "-out", "out", "-noise-weight", "1.1",
        ]))
        XCTAssertThrowsError(try SelfPlayCommand.parse([
            "-model", "net", "-games", "1", "-out", "out", "-komi", "7.25",
        ]))
    }
}
