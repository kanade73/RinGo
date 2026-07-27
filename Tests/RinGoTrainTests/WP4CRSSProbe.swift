import Darwin
import Foundation
@testable import RinGoTrain
import XCTest

/// THROWAWAY WP-4c measurement (gated by WP4C_PROBE_DIR). Loads a shard subset and reports resident
/// memory for the NEW columnar u8 representation vs the OLD materialized [RinGoSample] ([Float]
/// spatial) representation, on the SAME data, so before/after is apples-to-apples. Not part of any
/// gate; deleted after measurement.
final class WP4CRSSProbe: XCTestCase {
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

    func testMeasureResident() throws {
        guard let dir = ProcessInfo.processInfo.environment["WP4C_PROBE_DIR"] else {
            throw XCTSkip("set WP4C_PROBE_DIR to a shard subset directory")
        }
        let gb = { (b: UInt64) in String(format: "%.3f GB", Double(b) / 1_073_741_824) }
        let base = residentBytes()
        let shard = try RinGoDataset.load(directory: URL(fileURLWithPath: dir))
        let afterLoad = residentBytes()
        let samples = Array(shard.samples) // materializes the OLD [RinGoSample] representation
        let afterMaterialize = residentBytes()
        let n = shard.sampleCount
        let newResident = afterLoad - base
        let oldResident = afterMaterialize - afterLoad
        let full = 3_949_477.0
        print("[WP4C RSS] samples=\(n)")
        print(
            "[WP4C RSS] base=\(gb(base)) afterLoad(new columnar)=\(gb(afterLoad)) afterMaterialize=\(gb(afterMaterialize))"
        )
        print("[WP4C RSS] NEW columnar resident = \(gb(newResident)) (\(newResident / UInt64(n)) B/sample)")
        print("[WP4C RSS] OLD [RinGoSample] resident = \(gb(oldResident)) (\(oldResident / UInt64(n)) B/sample)")
        print(String(format: "[WP4C RSS] extrapolated to full %.0f samples: NEW=%.2f GB  OLD=%.2f GB",
                     full,
                     Double(newResident) / Double(n) * full / 1_073_741_824,
                     Double(oldResident) / Double(n) * full / 1_073_741_824))
        withExtendedLifetime((shard, samples)) {}
        XCTAssertGreaterThan(newResident, 0)
    }

    /// THROWAWAY WP-4c wall-time measurement (gated by WP4C_TIMING=1). Times a full validation pass
    /// over a synthetic ~50k-sample shard the OLD way (eager, uncompiled, per-microbatch host syncs:
    /// `evaluateEager`) vs the NEW way (compiled forward reused across microbatches, one host sync:
    /// `evaluate`). Compiled is timed both COLD (first call, includes the one-time trace/compile) and
    /// WARM (reused trace — the steady-state cost during a training run, which is what P1-3 targets).
    /// GPU-contention caveat: production D1 training may share the GPU, inflating absolute numbers;
    /// the eager-vs-warm-compiled RATIO is the meaningful figure.
    func testValidationWallTime() throws {
        guard ProcessInfo.processInfo.environment["WP4C_TIMING"] == "1" else {
            throw XCTSkip("set WP4C_TIMING=1 (Metal-capable host) to time the validation pass")
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
            print(String(format: "[WP4C TIME] %-18@ %6.3fs  (total=%.5f, n=%d)",
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
