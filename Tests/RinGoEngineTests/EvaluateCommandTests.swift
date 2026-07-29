import Foundation
@testable import ringo
import RinGoTrain
import XCTest

/// Tests for the `ringo evaluate` subcommand. The CLI's model-loading path
/// runs MLX, so it stays host-only (exercised by `RUN_HOST_ONLY_MLX_TESTS=1`); the CSV writer
/// and flag parsing are pure Swift and are covered here.
final class EvaluateCommandTests: XCTestCase {
    // MARK: - Loss-weight flags/defaults must match TrainCommand's,

    // not LossWeights()'s bare struct defaults, or total_loss can't be overlaid with training's
    // metrics.csv despite this subcommand's docstring claiming it can.

    func testDefaultLossWeightsMatchTrainCommandDefaultsNotLossWeightsBareDefaults() throws {
        let options = try EvaluateCommand.parse(["-model", "m.bin.gz", "-data", "shard.nngd"])
        XCTAssertEqual(options.valueWeight, TrainCommand.defaultValueWeight)
        XCTAssertEqual(options.ownershipWeight, TrainCommand.defaultOwnershipWeight)
        XCTAssertEqual(options.scoreWeight, TrainCommand.defaultScoreWeight)
        // Pin the actual numbers too so a change to either default is caught here, not just via
        // the (looser) "matches TrainCommand" assertions above.
        XCTAssertEqual(options.valueWeight, 3)
        XCTAssertEqual(options.ownershipWeight, 0.5)
        XCTAssertEqual(options.scoreWeight, 0.02)
    }

    func testLossWeightFlagsOverrideDefaults() throws {
        let options = try EvaluateCommand.parse([
            "-model", "m.bin.gz", "-data", "shard.nngd",
            "-value-weight", "1.5", "-ownership-weight", "0.06", "-score-weight", "0.01",
        ])
        XCTAssertEqual(options.valueWeight, 1.5)
        XCTAssertEqual(options.ownershipWeight, 0.06)
        XCTAssertEqual(options.scoreWeight, 0.01)
    }

    func testInvalidLossWeightFlagsThrow() {
        XCTAssertThrowsError(
            try EvaluateCommand.parse(["-model", "m.bin.gz", "-data", "d", "-value-weight", "-1"])
        )
        XCTAssertThrowsError(
            try EvaluateCommand.parse(["-model", "m.bin.gz", "-data", "d", "-ownership-weight", "nope"])
        )
        XCTAssertThrowsError(
            try EvaluateCommand.parse(["-model", "m.bin.gz", "-data", "d", "-score-weight", "nan"])
        )
    }

    func testParseRequiresModelAndData() {
        XCTAssertThrowsError(try EvaluateCommand.parse(["-model", "m.bin.gz"]))
        XCTAssertThrowsError(try EvaluateCommand.parse(["-data", "d"]))
        XCTAssertThrowsError(try EvaluateCommand.parse([]))
    }

    func testWriteCSVEmitsHeaderAndOneRowInSpecOrder() throws {
        let metrics = ValidationMetrics(
            step: 0,
            policyLoss: 1.5,
            valueLoss: 0.7,
            scoreLoss: 0.3,
            ownershipLoss: 0.5,
            totalLoss: 3.0,
            sampleCount: 8192
        )
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("eval-cmd-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("val_metrics.csv")

        try EvaluateCommand.writeCSV(result: metrics, to: url)

        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        XCTAssertEqual(lines[0], "policy_loss,value_loss,score_loss,ownership_loss,total_loss,sample_count")
        XCTAssertEqual(lines[1], "1.5,0.7,0.3,0.5,3.0,8192")
    }
}
