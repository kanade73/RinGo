import Foundation
@testable import RinGoTrain
import XCTest

final class RinGoShardTests: XCTestCase {
    func testReadsExactV2LittleEndianLayout() throws {
        let data = shardData(nnLen: 2, policyIndex: 4)
        let shard = try RinGoShard.read(data: data)
        XCTAssertEqual(shard.nnLen, 2)
        XCTAssertEqual(shard.samples.count, 1)
        let sample = try XCTUnwrap(shard.samples.first)
        XCTAssertEqual(sample.spatial.count, 88)
        // Spatial planes are binary 0/1 (see `shardData`); the u8 backing store round-trips them to
        // exactly 0.0/1.0. Little-endian fp32 layout parsing is still fully exercised by the global /
        // policy / value / score fields below, which carry arbitrary floats.
        XCTAssertEqual(sample.spatial[2], 0)
        XCTAssertEqual(sample.spatial[3], 1)
        XCTAssertEqual(sample.global, (0 ..< 19).map { Float($0) - 9 })
        XCTAssertEqual(sample.policyTarget, [0, 0, 0, 0, 1])
        XCTAssertEqual(sample.valueTarget, [0, 1, 0])
        XCTAssertEqual(sample.scoreTarget, -3.5)
        XCTAssertTrue(sample.scoreValid)
        XCTAssertEqual(sample.ownershipTarget, [1, -1, 0, 1])
        XCTAssertTrue(sample.ownershipValid)
    }

    /// WP-2 (TRAINING_FIX_PLAN.md): RinGoData v2 is NOT backward compatible with v1. A v1 shard
    /// (one-hot policy index, no validity flags) must be rejected with a clear, actionable error --
    /// not silently misread as a truncated/malformed v2 shard.
    func testRejectsV1Shards() {
        var data = shardData(nnLen: 2, policyIndex: 4)
        data[4] = 1 // version is the first little-endian u32 right after the 4-byte magic.
        XCTAssertThrowsError(try RinGoShard.read(data: data)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("Unsupported RinGoData version 1"), message)
            XCTAssertTrue(message.contains("expected 2"), message)
            XCTAssertTrue(message.contains("regenerate"), message)
        }
    }

    /// Audit P1-2: the u8 spatial packing only holds if every 9x9 V7 plane is binary 0/1. A
    /// non-binary spatial value must fail loudly with the offending sample/position/plane -- NEVER
    /// be silently quantized. Spatial starts right after the 24-byte header; flat index 5 is written
    /// as 1.0 (little-endian bytes 00 00 80 3F). Clearing the 0x80 byte turns it into 0.5
    /// (00 00 00 3F), a non-binary plane value the loader must reject.
    func testRejectsNonBinarySpatialValue() {
        var data = shardData(nnLen: 2, policyIndex: 4)
        data[24 + 5 * 4 + 2] = 0
        XCTAssertThrowsError(try RinGoShard.read(data: data)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("binary 0/1"), message)
            XCTAssertTrue(message.contains("plane 5"), message)
            XCTAssertTrue(message.contains("0.5"), message)
        }
    }

    func testRejectsTruncatedAndTrailingData() {
        let valid = shardData(nnLen: 2, policyIndex: 4)
        XCTAssertThrowsError(try RinGoShard.read(data: Data(valid.dropLast()))) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("ownership validity flag for sample 0"), message)
            XCTAssertTrue(message.contains("only 0 remain"), message)
        }
        XCTAssertThrowsError(try RinGoShard.read(data: Data(valid.prefix(34)))) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("spatial features for sample 0"), message)
            XCTAssertTrue(message.contains("expected 352 bytes, only 10 remain"), message)
        }
        XCTAssertThrowsError(try RinGoShard.read(data: valid + Data([0])))
    }

    func testBulkReadTimingSanity() throws {
        let sampleCount: UInt32 = 750
        let data = shardData(nnLen: 9, policyIndex: 81, sampleCount: sampleCount)

        let start = ProcessInfo.processInfo.systemUptime
        let shard = try RinGoShard.read(data: data)
        let elapsed = ProcessInfo.processInfo.systemUptime - start

        XCTAssertEqual(shard.samples.count, Int(sampleCount))
        XCTAssertEqual(shard.samples.last?.spatial.count, 22 * 81)
        XCTAssertLessThan(elapsed, 1.5, "Parsing took \(elapsed) seconds")
    }

    // MARK: - P0-1: RinGoDataset.load(directory:excluding:) must not leak val.nngd into train

    /// Regression for TRAINING_PROCESS_AUDIT.md P0-1: `TrainCommand` reads training shards and the
    /// held-out `val.nngd` from the same directory (the exact layout `makedata -val-ratio`
    /// produces). Without an exclusion mechanism, `RinGoDataset.load(directory:)` reads every
    /// `.nngd` file it finds — including `val.nngd` — so the "held-out" validation samples end up
    /// counted in train too. This proves both sides: excluding `val.nngd`'s URL yields a train set
    /// with exactly the non-val samples, while loading without exclusion (the pre-fix behavior)
    /// still combines everything, confirming the test would have caught the original bug.
    func testLoadDirectoryExcludingValNngdYieldsOnlyTrainSamples() throws {
        let directory = try makeTempDirectory()
        let trainURL = directory.appendingPathComponent("shard-00000.nngd")
        let valURL = directory.appendingPathComponent("val.nngd")
        try shardData(nnLen: 2, policyIndex: 4, sampleCount: 3).write(to: trainURL)
        try shardData(nnLen: 2, policyIndex: 4, sampleCount: 2).write(to: valURL)

        // Pre-fix behavior: loading the directory with no exclusion combines train + val.
        let combined = try RinGoDataset.load(directory: directory)
        XCTAssertEqual(combined.samples.count, 5)

        // Fixed behavior: excluding val.nngd's URL leaves only the training samples.
        let train = try RinGoDataset.load(directory: directory, excluding: [valURL])
        XCTAssertEqual(train.samples.count, 3)

        // The validation side, read directly, has exactly val.nngd's own samples.
        let validation = try RinGoShard.read(url: valURL)
        XCTAssertEqual(validation.samples.count, 2)
    }

    /// Excluding a URL that doesn't match any file in the directory is a no-op — this guards
    /// against a future refactor accidentally excluding by filename substring or basename alone
    /// (which could silently drop unrelated shards sharing a name pattern).
    func testLoadDirectoryExcludingAnUnrelatedURLIsANoOp() throws {
        let directory = try makeTempDirectory()
        let trainURL = directory.appendingPathComponent("shard-00000.nngd")
        try shardData(nnLen: 2, policyIndex: 4, sampleCount: 3).write(to: trainURL)
        let unrelated = directory.appendingPathComponent("val.nngd")

        let train = try RinGoDataset.load(directory: directory, excluding: [unrelated])

        XCTAssertEqual(train.samples.count, 3)
    }

    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ringo-shard-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    /// Builds one RinGoData v2 sample, repeated `sampleCount` times: a one-hot dense policy with
    /// its mass at `policyIndex`, a fixed value/score/ownership payload, and both validity flags
    /// set true (matching what a genuinely scored teacherless `makedata` game writes).
    private func shardData(nnLen: UInt32, policyIndex: Int, sampleCount: UInt32 = 1) -> Data {
        var data = Data("NNGD".utf8)
        append(UInt32(2), to: &data)
        append(nnLen, to: &data)
        append(UInt32(22), to: &data)
        append(UInt32(19), to: &data)
        append(sampleCount, to: &data)

        let boardArea = Int(nnLen * nnLen)
        var sample = Data()
        // V7 9x9 spatial planes are binary 0/1 indicators (every `NNInputs.fillRowV7` `set` writes
        // the default value 1). Write an alternating 0/1 pattern so the loader's binary validation
        // accepts it and the u8 packing round-trips exactly.
        for index in 0 ..< 22 * boardArea {
            append(Float(index % 2), to: &sample)
        }
        for index in 0 ..< 19 {
            append(Float(index) - 9, to: &sample)
        }
        for index in 0 ... boardArea {
            append(index == policyIndex ? Float(1) : Float(0), to: &sample)
        }
        [Float(0), 1, 0].forEach { append($0, to: &sample) }
        append(Float(-3.5), to: &sample)
        sample.append(1) // scoreValid = true
        for index in 0 ..< boardArea {
            let ownership: Int8 = switch index % 3 {
            case 0: 1
            case 1: -1
            default: 0
            }
            sample.append(UInt8(bitPattern: ownership))
        }
        sample.append(1) // ownershipValid = true
        for _ in 0 ..< sampleCount {
            data.append(sample)
        }
        return data
    }

    private func append(_ value: UInt32, to data: inout Data) {
        for shift in stride(from: 0, to: 32, by: 8) {
            data.append(UInt8(truncatingIfNeeded: value >> UInt32(shift)))
        }
    }

    private func append(_ value: Float, to data: inout Data) {
        append(value.bitPattern, to: &data)
    }
}
