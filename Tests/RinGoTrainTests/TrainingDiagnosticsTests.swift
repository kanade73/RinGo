import Foundation
import MLX
import MLXNN
import MLXOptimizers
import RinGoModel
@testable import RinGoTrain
import XCTest

/// THROWAWAY DIAGNOSTIC (gated behind RUN_TRAIN_DIAGNOSTICS=1). Investigates why value_loss and
/// ownership_loss plateau at their ln(2) marginal baseline during real training. Not part of
/// `make check`; reads a real shard from RINGO_DIAG_SHARD (default: the step-a-run shard 0).
final class TrainingDiagnosticsTests: XCTestCase {
    private func requireDiag() throws -> RinGoShard {
        guard ProcessInfo.processInfo.environment["RUN_TRAIN_DIAGNOSTICS"] == "1" else {
            throw XCTSkip("DIAGNOSTIC: set RUN_TRAIN_DIAGNOSTICS=1 on a Metal-capable host")
        }
        let path = ProcessInfo.processInfo.environment["RINGO_DIAG_SHARD"]
            ?? ".tools/step-a-run/shards/shard-00000.nngd"
        return try RinGoShard.read(url: URL(fileURLWithPath: path))
    }

    /// Classify a store key (without the "values." nesting prefix) into a coarse group.
    private func group(for key: String) -> String {
        let name = key.drop { $0 != ":" }.dropFirst() // drop "conv:"/"matmul:"/... prefix
        if name.hasPrefix("vownership/") { return "ownership" }
        if name.hasPrefix("sv3/") { return "score-final" }
        if name.hasPrefix("v3/") { return "value-final" }
        if name.hasPrefix("v1/") || name.hasPrefix("v2/") { return "value-shared(v1,v2)" }
        if name.hasPrefix("p1/") || name.hasPrefix("g1/") || name.hasPrefix("p2/")
            || name.hasPrefix("matmulg2w") || name.hasPrefix("matmulpass") { return "policy" }
        return "trunk"
    }

    private func flatGrads(_ grads: ModuleParameters) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (k, v) in grads.flattened() {
            // keys look like "values.conv:v1/w"
            let stripped = k.hasPrefix("values.") ? String(k.dropFirst("values.".count)) : k
            out[stripped] = v
        }
        return out
    }

    private func realBatch(_ shard: RinGoShard, count: Int, seedOffset: Int = 0) -> TrainingBatch {
        let n = min(count, shard.samples.count)
        let picked = (0 ..< n).map { shard.samples[(seedOffset + $0) % shard.samples.count] }
        return Trainer.batch(samples: picked, nnLen: shard.nnLen)
    }

    /// 1) Confirm the data carries a learnable signal for value and ownership.
    func testDataTargetStatistics() throws {
        let shard = try requireDiag()
        let n = shard.samples.count
        var wins = 0
        var losses = 0
        var noResult = 0
        var ownPos = 0
        var ownNeg = 0
        var ownNeutral = 0
        var scoreSum: Float = 0
        var scoreAbsSum: Float = 0
        for s in shard.samples {
            if s.valueTarget[0] > 0.5 { wins += 1 }
            if s.valueTarget[1] > 0.5 { losses += 1 }
            if s.valueTarget[2] > 0.5 { noResult += 1 }
            for o in s.ownershipTarget {
                if o > 0 { ownPos += 1 } else if o < 0 { ownNeg += 1 } else { ownNeutral += 1 }
            }
            scoreSum += s.scoreTarget
            scoreAbsSum += abs(s.scoreTarget)
        }
        let ownTotal = ownPos + ownNeg + ownNeutral
        print("[DIAG data] samples=\(n)")
        print("[DIAG data] value win=\(wins) loss=\(losses) noResult=\(noResult) " +
            "winFrac=\(Float(wins) / Float(n))")
        print("[DIAG data] ownership +1=\(Float(ownPos) / Float(ownTotal)) " +
            "-1=\(Float(ownNeg) / Float(ownTotal)) 0=\(Float(ownNeutral) / Float(ownTotal))")
        print("[DIAG data] score mean=\(scoreSum / Float(n)) meanAbs=\(scoreAbsSum / Float(n))")
        // Sanity: not degenerate (both wins and losses present, ownership not all neutral).
        XCTAssertGreaterThan(wins, 0)
        XCTAssertGreaterThan(losses, 0)
        XCTAssertGreaterThan(ownPos + ownNeg, ownNeutral / 2)
    }

    /// 2) Per-group gradient norms (pre-clip) + global norm + clip factor at init and after N joint steps.
    func testPerGroupGradientNorms() throws {
        let shard = try requireDiag()
        let batch = realBatch(shard, count: 256)
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let weights = LossWeights()
        let lc = LossComputer(weights: weights)
        let optimizer = AdamW(learningRate: 1e-3, weightDecay: 0.01, biasCorrection: true)
        let maxNorm: Float = 1

        func gradFn() -> (Losses, ModuleParameters) {
            let gf = valueAndGrad(model: network.store) { [network] _, arrays in
                let inp = TrainingBatch(
                    spatial: arrays[0], global: arrays[1], policyTarget: arrays[2],
                    valueTarget: arrays[3], scoreTarget: arrays[4], scoreValidTarget: arrays[5],
                    ownershipTarget: arrays[6], ownershipValidTarget: arrays[7]
                )
                let l = lc(outputs: network.forward(spatial: arrays[0], global: arrays[1]), batch: inp)
                return [l.total, l.policy, l.value, l.score, l.ownership]
            }
            let arrays = [
                batch.spatial, batch.global, batch.policyTarget, batch.valueTarget,
                batch.scoreTarget, batch.scoreValidTarget, batch.ownershipTarget, batch.ownershipValidTarget,
            ]
            let (lv, g) = gf(network.store, arrays)
            let losses = Losses(policy: lv[1], value: lv[2], score: lv[3], ownership: lv[4], total: lv[0])
            return (losses, g)
        }

        func report(_ label: String) {
            let (losses, grads) = gradFn()
            let flat = flatGrads(grads)
            var perGroupSq = [String: MLXArray]()
            for (k, g) in flat {
                let grp = group(for: k)
                let sq = g.square().sum()
                perGroupSq[grp] = (perGroupSq[grp].map { $0 + sq }) ?? sq
            }
            var globalSq = MLXArray(Float(0))
            for (_, g) in flat {
                globalSq += g.square().sum()
            }
            let globalNorm = sqrt(globalSq)
            eval(losses.value, losses.ownership, losses.policy, globalNorm)
            eval(Array(perGroupSq.values))
            let gn = globalNorm.item(Float.self)
            let clip = gn < maxNorm ? 1.0 : Double(maxNorm) / Double(gn)
            print("[DIAG grad \(label)] valLoss=\(losses.value.item(Float.self)) " +
                "ownLoss=\(losses.ownership.item(Float.self)) polLoss=\(losses.policy.item(Float.self))")
            print("[DIAG grad \(label)] globalNorm=\(gn) clipActive=\(gn >= maxNorm) clipFactor=\(clip)")
            for grp in ["trunk", "policy", "value-shared(v1,v2)", "value-final", "score-final", "ownership"] {
                if let sq = perGroupSq[grp] {
                    print("[DIAG grad \(label)]   \(grp) preClipNorm=\(sqrt(sq).item(Float.self))")
                }
            }
            // Weight L2 of the two zero-init heads we care about (do they move at all?).
            let v3w = network.parameters["matmul:v3/w"].map { sqrt($0.square().sum()).item(Float.self) } ?? -1
            let ow = network.parameters["conv:vownership/w"].map { sqrt($0.square().sum()).item(Float.self) } ?? -1
            print("[DIAG grad \(label)]   ||v3/w||=\(v3w) ||vownership/w||=\(ow)")
        }

        report("step0")
        // Run joint steps manually (mirrors Trainer: clip global norm, AdamW update).
        for i in 1 ... 400 {
            let (_, grads) = gradFn()
            let flat = flatGrads(grads)
            var globalSq = MLXArray(Float(0))
            for (_, g) in flat {
                globalSq += g.square().sum()
            }
            let totalNorm = sqrt(globalSq)
            let normalizer = maxNorm / (totalNorm + 1e-6)
            let clipped = grads.mapValues { which(totalNorm .< maxNorm, $0, $0 * normalizer) }
            optimizer.update(model: network.store, gradients: clipped)
            eval(Array(network.parameters.values) + optimizer.innerState())
            if i == 50 || i == 200 || i == 400 { report("step\(i)") }
        }
    }

    /// 3) Isolation: train ONLY value_loss (then ONLY ownership_loss), NO clipping, on a fixed real
    ///    batch. If the head can reduce its own loss here, the head/loss math is fine and the joint
    ///    stall is a training-dynamics problem.
    func testIsolationValueOnly() throws {
        let shard = try requireDiag()
        try runIsolation(shard: shard, label: "VALUE",
                         weights: LossWeights(policy: 0, value: 1, score: 0, ownership: 0),
                         read: { $0.value })
    }

    func testIsolationOwnershipOnly() throws {
        let shard = try requireDiag()
        try runIsolation(shard: shard, label: "OWNERSHIP",
                         weights: LossWeights(policy: 0, value: 0, score: 0, ownership: 1),
                         read: { $0.ownership })
    }

    private func runIsolation(
        shard: RinGoShard, label: String, weights: LossWeights, read: (Losses) -> MLXArray
    ) throws {
        let batch = realBatch(shard, count: 256)
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let lc = LossComputer(weights: weights)
        let optimizer = AdamW(learningRate: 1e-3, weightDecay: 0, biasCorrection: true)
        let arrays = [
            batch.spatial, batch.global, batch.policyTarget, batch.valueTarget,
            batch.scoreTarget, batch.scoreValidTarget, batch.ownershipTarget, batch.ownershipValidTarget,
        ]
        let gf = valueAndGrad(model: network.store) { [network] _, a in
            let inp = TrainingBatch(spatial: a[0], global: a[1], policyTarget: a[2],
                                    valueTarget: a[3], scoreTarget: a[4], scoreValidTarget: a[5],
                                    ownershipTarget: a[6], ownershipValidTarget: a[7])
            let l = lc(outputs: network.forward(spatial: a[0], global: a[1]), batch: inp)
            return [l.total, l.policy, l.value, l.score, l.ownership]
        }
        for i in 0 ... 400 {
            let (lv, grads) = gf(network.store, arrays)
            optimizer.update(model: network.store, gradients: grads) // NO clipping
            eval(Array(network.parameters.values) + optimizer.innerState() + lv)
            if i % 50 == 0 {
                let losses = Losses(policy: lv[1], value: lv[2], score: lv[3], ownership: lv[4], total: lv[0])
                print("[DIAG isolation \(label)] step \(i) loss=\(read(losses).item(Float.self))")
            }
        }
    }

    /// 4) Joint training, clip vs no-clip, on streamed real batches -- does removing the global-norm
    ///    clip let value/ownership escape the ln(2) plateau?
    func testJointClipVsNoClip() throws {
        let shard = try requireDiag()
        try runJoint(shard: shard, maxNorm: 1, label: "clip=1")
        try runJoint(shard: shard, maxNorm: 1e9, label: "noClip")
    }

    private func runJoint(shard: RinGoShard, maxNorm: Float, label: String) throws {
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let config = TrainerConfiguration(
            batchSize: 256, totalSteps: 600, learningRate: 1e-3, warmupSteps: 50,
            weightDecay: 0.01, maximumGradientNorm: maxNorm, checkpointInterval: 0,
            snapshotInterval: 0, metricsInterval: 0
        )
        let trainer = try Trainer(network: network, configuration: config)
        var idx = Array(shard.samples.indices)
        idx.shuffle()
        var off = 0
        var last: TrainingMetrics?
        for step in 1 ... 600 {
            if off + 256 > idx.count { idx.shuffle(); off = 0 }
            let sel = (0 ..< 256).map { idx[off + $0] }
            off += 256
            let batch = Trainer.batch(samples: sel.map { shard.samples[$0] }, nnLen: shard.nnLen)
            let m = trainer.trainOneStep(batch)
            last = m
            if step % 100 == 0 || step == 1 {
                print("[DIAG joint \(label)] step \(step) val=\(m.valueLoss) own=\(m.ownershipLoss) " +
                    "pol=\(m.policyLoss) score=\(m.scoreLoss)")
            }
        }
        if let m = last {
            print("[DIAG joint \(label)] FINAL val=\(m.valueLoss) own=\(m.ownershipLoss) pol=\(m.policyLoss)")
        }
    }

    /// 5) ISOLATE compiled-Trainer-bug vs streaming: run the real compiled Trainer on a SINGLE fixed
    ///    real batch. If value/ownership learn here, the compiled step is fine and the streaming stall
    ///    is a generalization/SNR problem (not a Trainer bug).
    func testTrainerCompiledFixedBatch() throws {
        let shard = try requireDiag()
        let batch = realBatch(shard, count: 256)
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: 256, totalSteps: 400, learningRate: 1e-3, warmupSteps: 0,
            weightDecay: 0.01, maximumGradientNorm: 1, checkpointInterval: 0,
            snapshotInterval: 0, metricsInterval: 0
        ))
        for step in 1 ... 400 {
            let m = trainer.trainOneStep(batch)
            if step % 50 == 0 || step == 1 {
                print("[DIAG trainerFixed] step \(step) val=\(m.valueLoss) own=\(m.ownershipLoss) pol=\(m.policyLoss)")
            }
        }
    }

    /// Overwrite the zero-initialized value/score/ownership head OUTPUT layers with a small nonzero
    /// truncated-normal init (std = scale / sqrt(fanIn) / 0.8796, matching TrainableNetwork's own
    /// `initialized` for identity activation). This is the candidate fix.
    private func perturbHeadOutputs(_ network: TrainableNetwork, scale: Float) {
        func small(_ shape: [Int], _ fanIn: Int) -> MLXArray {
            let std = scale / Foundation.sqrt(Float(fanIn)) / 0.87962566103423978
            return MLXRandom.truncatedNormal(Float(-2) ..< Float(2), shape) * std
        }
        var params = network.parameters
        params["matmul:v3/w"] = small([64, 3], 64)
        params["matmul:sv3/w"] = small([64, 4], 64)
        params["conv:vownership/w"] = small([1, 1, 1, 32], 32)
        try? network.replaceParameters(params)
        eval(Array(network.parameters.values))
    }

    /// 6) TEST THE FIX: same streaming joint run as (4), but with the value/score/ownership head
    ///    output layers given a small nonzero init instead of zero. If value/ownership now drop below
    ///    the ln(2) plateau on STREAMING data, the zero-init bootstrap starvation is confirmed as the
    ///    cause and this init is the fix.
    func testStreamingSmallHeadInit() throws {
        let shard = try requireDiag()
        for scale in [Float(1.0)] {
            let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
            perturbHeadOutputs(network, scale: scale)
            let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
                batchSize: 256, totalSteps: 600, learningRate: 1e-3, warmupSteps: 50,
                weightDecay: 0.01, maximumGradientNorm: 1, checkpointInterval: 0,
                snapshotInterval: 0, metricsInterval: 0
            ))
            var idx = Array(shard.samples.indices)
            idx.shuffle()
            var off = 0
            for step in 1 ... 600 {
                if off + 256 > idx.count { idx.shuffle(); off = 0 }
                let sel = (0 ..< 256).map { idx[off + $0] }
                off += 256
                let batch = Trainer.batch(samples: sel.map { shard.samples[$0] }, nnLen: shard.nnLen)
                let m = trainer.trainOneStep(batch)
                if step % 100 == 0 || step == 1 {
                    print("[DIAG fixInit scale=\(scale)] step \(step) val=\(m.valueLoss) " +
                        "own=\(m.ownershipLoss) pol=\(m.policyLoss)")
                }
            }
        }
    }

    private func wnorm(_ network: TrainableNetwork, _ key: String) -> Float {
        guard let p = network.parameters[key] else { return -1 }
        let n = sqrt(p.square().sum())
        eval(n)
        return n.item(Float.self)
    }

    /// 7) PINNED vs SLOW: longer (2000-step) stock streaming run, logging the value/ownership head
    ///    output-weight norms AND the value feature-extractor (v1) conv norm, to see whether those
    ///    weights move at all over a realistic horizon.
    func testStreamingLongProbe() throws {
        let shard = try requireDiag()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: 256, totalSteps: 2000, learningRate: 3e-4, warmupSteps: 200,
            weightDecay: 0.01, maximumGradientNorm: 1, checkpointInterval: 0,
            snapshotInterval: 0, metricsInterval: 0
        ))
        var idx = Array(shard.samples.indices)
        idx.shuffle()
        var off = 0
        for step in 1 ... 2000 {
            if off + 256 > idx.count { idx.shuffle(); off = 0 }
            let sel = (0 ..< 256).map { idx[off + $0] }
            off += 256
            let batch = Trainer.batch(samples: sel.map { shard.samples[$0] }, nnLen: shard.nnLen)
            let m = trainer.trainOneStep(batch)
            if step % 200 == 0 || step == 1 {
                print("[DIAG longProbe] step \(step) val=\(m.valueLoss) own=\(m.ownershipLoss) " +
                    "pol=\(m.policyLoss) ||v3/w||=\(wnorm(network, "matmul:v3/w")) " +
                    "||vown/w||=\(wnorm(network, "conv:vownership/w")) " +
                    "||v1/w||=\(wnorm(network, "conv:v1/w"))")
            }
        }
    }
}
