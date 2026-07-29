import Darwin
import Foundation
@testable import RinGoTrain
import XCTest

/// Opt-in performance measurements, not correctness gates. Both are skipped unless their
/// environment variable is set, so `swift test` on a fresh checkout never runs them; they exist
/// so the two claims they cover can be re-measured on any machine rather than taken on trust.
final class TrainingPerformanceProbes: XCTestCase {
    private func residentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Resident-memory cost of holding a dataset two ways, measured on the same shards in one
    /// process so the comparison is apples-to-apples: the columnar `u8` representation
    /// `RinGoDataset` actually stores, versus the `[RinGoSample]` array (`[Float]` spatial planes)
    /// that materializing `shard.samples` produces. The gap is why the columnar form exists —
    /// it decides whether a multi-million-sample shard set fits in unified memory at all.
    ///
    /// Point `RINGO_PROBE_SHARD_DIR` at a directory of `.nngd` shards to run it, and set
    /// `RINGO_PROBE_EXTRAPOLATE_TO` to a sample count to also print the projection to a full
    /// dataset of that size.
    func testMeasureResident() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let dir = environment["RINGO_PROBE_SHARD_DIR"] else {
            throw XCTSkip("set RINGO_PROBE_SHARD_DIR to a directory of .nngd shards")
        }
        let gb = { (b: UInt64) in String(format: "%.3f GB", Double(b) / 1_073_741_824) }
        let base = residentBytes()
        let shard = try RinGoDataset.load(directory: URL(fileURLWithPath: dir))
        let afterLoad = residentBytes()
        let samples = Array(shard.samples) // materializes the [RinGoSample] representation
        let afterMaterialize = residentBytes()
        let n = shard.sampleCount
        let columnarResident = afterLoad - base
        let materializedResident = afterMaterialize - afterLoad
        print("[probe rss] samples=\(n)")
        print(
            "[probe rss] base=\(gb(base)) afterLoad(columnar)=\(gb(afterLoad)) afterMaterialize=\(gb(afterMaterialize))"
        )
        print("[probe rss] columnar u8 resident = \(gb(columnarResident)) (\(columnarResident / UInt64(n)) B/sample)")
        print(
            "[probe rss] [RinGoSample] resident = \(gb(materializedResident)) "
                + "(\(materializedResident / UInt64(n)) B/sample)"
        )
        if let target = environment["RINGO_PROBE_EXTRAPOLATE_TO"].flatMap(Double.init) {
            print(String(format: "[probe rss] extrapolated to %.0f samples: columnar=%.2f GB  materialized=%.2f GB",
                         target,
                         Double(columnarResident) / Double(n) * target / 1_073_741_824,
                         Double(materializedResident) / Double(n) * target / 1_073_741_824))
        }
        withExtendedLifetime((shard, samples)) {}
        XCTAssertGreaterThan(columnarResident, 0)
    }

    /// Wall time of one validation pass over a synthetic 50k-sample shard, run two ways:
    /// `evaluateEager` (uncompiled, one host sync per microbatch) versus `evaluate` (one compiled
    /// forward reused across microbatches, one host sync for the pass). The compiled path is timed
    /// both cold — first call, paying the one-time trace and compile — and warm, which is the
    /// steady-state cost a training run actually pays when it validates.
    ///
    /// Set `RINGO_PROBE_TIMING=1` on a Metal-capable host to run it. Absolute numbers move with
    /// whatever else is using the GPU; the eager-to-warm-compiled ratio is the figure that holds.
    func testValidationWallTime() throws {
        guard ProcessInfo.processInfo.environment["RINGO_PROBE_TIMING"] == "1" else {
            throw XCTSkip("set RINGO_PROBE_TIMING=1 (Metal-capable host) to time the validation pass")
        }
        let sampleCount = 50000
        let shard = try RinGoShard(nnLen: 9, samples: (0 ..< sampleCount).map(Self.syntheticSample(seed:)))
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: 256, totalSteps: 10, learningRate: 1e-3, warmupSteps: 0,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        // Move params off init so the compiled trace reads live (post-step) params, mirroring how
        // validation runs mid-training.
        _ = trainer.trainOneStep(shard.batch(indices: Array(0 ..< 256)))

        func time(_ label: String, _ body: () throws -> ValidationMetrics) rethrows {
            let start = Date()
            let m = try body()
            let elapsed = Date().timeIntervalSince(start)
            print(String(format: "[probe time] %-18@ %6.3fs  (total=%.5f, n=%d)",
                         label as NSString, elapsed, m.totalLoss, m.sampleCount))
        }

        try time("eager #1") { try trainer.evaluateEager(validation: shard) }
        try time("eager #2") { try trainer.evaluateEager(validation: shard) }
        try time("compiled cold") { try trainer.evaluate(validation: shard) }
        try time("compiled warm #1") { try trainer.evaluate(validation: shard) }
        try time("compiled warm #2") { try trainer.evaluate(validation: shard) }
    }

    /// Deterministic synthetic sample with BINARY spatial planes (the u8-packing invariant), mixed
    /// validity flags, and a normalized 2-hot policy — enough to exercise every loss head.
    private static func syntheticSample(seed: Int) -> RinGoSample {
        var spatial = [Float](repeating: 0, count: 9 * 9 * 22)
        for i in spatial.indices where (i * 31 + seed * 17) % 3 == 0 {
            spatial[i] = 1
        }
        var policy = [Float](repeating: 0, count: 82)
        policy[(seed * 7) % 82] = 0.6
        policy[(seed * 13 + 1) % 82] = 0.4
        var value = [Float](repeating: 0, count: 3)
        value[seed % 3] = 1
        var ownership = [Int8](repeating: 0, count: 81)
        for i in ownership.indices {
            ownership[i] = Int8((i + seed) % 3) - 1
        }
        var global = [Float](repeating: 0, count: 19)
        for i in global.indices {
            global[i] = Float(seed) * 0.1 + Float(i) * 0.01
        }
        return RinGoSample(
            spatial: spatial, global: global, policyTarget: policy, valueTarget: value,
            scoreTarget: Float(seed) * 0.5 - 2, scoreValid: seed % 2 == 0,
            ownershipTarget: ownership, ownershipValid: seed % 3 != 0
        )
    }
}
