import Foundation
import MLX
import MLXNN
import RinGoModel
@testable import RinGoTrain
import XCTest

/// HOST-ONLY: these tests construct and execute MLX graphs. They must remain skipped in CPU CI.
final class TrainingHostTests: XCTestCase {
    func testExportDescRoundtripPreservesStructure() throws {
        try requireHostMLX()
        let source = RinGoArchitecture.b6c96()
        let network = try TrainableNetwork(desc: source, nnXLen: 9, nnYLen: 9)
        let exported = network.exportDesc()
        XCTAssertEqual(exported.modelVersion, source.modelVersion)
        XCTAssertEqual(exported.trunk.blocks.map(\.kind), source.trunk.blocks.map(\.kind))
        XCTAssertEqual(exported.trunk.initialConv.weights.count, source.trunk.initialConv.weights.count)
        XCTAssertEqual(exported.policyHead.p2Conv.weights.count, source.policyHead.p2Conv.weights.count)
        XCTAssertEqual(exported.valueHead.v3Mul.weights.count, source.valueHead.v3Mul.weights.count)
    }

    /// The value / score / ownership output layers used to be zero-initialized. On RinGo's small
    /// Step A that trapped them at ln(2) plateaus, so they must start with small-nonzero weights.
    /// Policy output layers stay zero-initialized (they already learn quickly).
    func testValueHeadOutputLayersAreNonzeroAtInit() throws {
        try requireHostMLX()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let v3 = try XCTUnwrap(network.parameters["matmul:v3/w"])
        let sv3 = try XCTUnwrap(network.parameters["matmul:sv3/w"])
        let ownership = try XCTUnwrap(network.parameters["conv:vownership/w"])
        XCTAssertGreaterThan(sqrt(v3.square().sum()).item(Float.self), 0)
        XCTAssertGreaterThan(sqrt(sv3.square().sum()).item(Float.self), 0)
        XCTAssertGreaterThan(sqrt(ownership.square().sum()).item(Float.self), 0)
        // Policy head intentionally unchanged.
        let p2 = try XCTUnwrap(network.parameters["conv:p2/w"])
        let pass = try XCTUnwrap(network.parameters["matmul:matmulpass"])
        XCTAssertEqual(sqrt(p2.square().sum()).item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(sqrt(pass.square().sum()).item(Float.self), 0, accuracy: 1e-6)
    }

    func testOneOptimizerStepDecreasesTotalLoss() throws {
        try requireHostMLX()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let batch = Trainer.batch(samples: [sample(policy: 81)], nnLen: 9)
        let computer = LossComputer()
        let before = computer(outputs: network.forward(spatial: batch.spatial, global: batch.global), batch: batch)
            .total
        eval(before)
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: 1, totalSteps: 10, learningRate: 1e-2, warmupSteps: 0,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        _ = trainer.trainOneStep(batch)
        let after = computer(outputs: network.forward(spatial: batch.spatial, global: batch.global), batch: batch).total
        eval(after)
        XCTAssertLessThan(after.item(Float.self), before.item(Float.self))
        XCTAssertTrue(after.item(Float.self).isFinite)
    }

    func testOverfitsTenSamplesToAboveNinetyPercentPolicyAccuracy() throws {
        try requireHostMLX()
        let samples = (0 ..< 10).map { _ in sample(policy: 81) }
        let batch = Trainer.batch(samples: samples, nnLen: 9)
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: 10, totalSteps: 200, learningRate: 3e-3, warmupSteps: 10,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        for _ in 0 ..< 200 {
            _ = trainer.trainOneStep(batch)
        }
        let output = network.forward(spatial: batch.spatial, global: batch.global)
        let logits = concatenated([
            output.policySpatialLogits.reshaped(10, -1), output.policyPassLogits.reshaped(10, -1),
        ], axis: 1)
        // batch.policyTarget is a dense one-hot distribution (v2), not a class index -- take its
        // argmax to recover the played-move index for the accuracy comparison.
        let targetIndex = argMax(batch.policyTarget, axis: 1)
        let accuracy = (argMax(logits, axis: 1) .== targetIndex).asType(.float32).mean()
        eval(accuracy)
        XCTAssertGreaterThan(accuracy.item(Float.self), 0.9)
    }

    /// The ownership loss switched from `tanh -> affine -> withLogits:false` to the fused
    /// `withLogits:true` path on `2 * rawOwnership`. Since sigmoid(2x) == (tanh(x)+1)/2 the two are
    /// mathematically identical; this pins that they agree numerically so the switch stays a pure
    /// efficiency change. Both branches replicate MLXNN's exact `binaryCrossEntropy` formulas
    /// (Losses.swift: withLogits:false clips log-space at -100; withLogits:true uses logAddExp),
    /// using only MLX primitives so the test target needs no extra module dependency.
    func testOwnershipLossLogitsFormulationMatchesTanhAffine() throws {
        try requireHostMLX()
        // A fixed spread of raw ownership logits and {0,1} targets -- no RNG, so this is stable.
        let raw = MLXArray([Float(-6), -3, -1, -0.2, 0, 0.4, 1.5, 3, 7])
        let target = MLXArray([Float(0), 1, 0, 1, 0, 1, 1, 0, 1])

        // Old: tanh -> affine to [0, 1] -> withLogits:false BCE (with the -100 log clip).
        let probability = (tanh(raw) + 1) / 2
        let oldLoss = -(target * clip(log(probability), min: -100)
            + (1 - target) * clip(log(1 - probability), min: -100))
        // New: withLogits:true BCE on 2 * raw (what LossComputer now uses).
        let newLoss = logAddExp(0, raw * 2) - target * (raw * 2)

        eval(oldLoss, newLoss)
        let oldValues = oldLoss.asArray(Float.self)
        let newValues = newLoss.asArray(Float.self)
        XCTAssertEqual(oldValues.count, newValues.count)
        for (old, new) in zip(oldValues, newValues) {
            XCTAssertEqual(old, new, accuracy: 1e-4)
        }
    }

    /// WP-2 (TRAINING_FIX_PLAN.md): RinGoData v2 replaced the one-hot policy INDEX with a dense
    /// probability distribution. For a teacherless (one-hot) shard this must be numerically
    /// IDENTICAL to the old index-based cross entropy over the same logits -- this pins that
    /// equivalence through the actual `LossComputer` policy-loss path, not just MLXNN's primitive.
    func testDensePolicyOneHotCrossEntropyMatchesOldIndexBasedCrossEntropy() throws {
        try requireHostMLX()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let playedMoves = [3, 40, 81, 0]
        let batch = Trainer.batch(samples: playedMoves.map { sample(policy: $0) }, nnLen: 9)
        let outputs = network.forward(spatial: batch.spatial, global: batch.global)
        let denseLoss = LossComputer()(outputs: outputs, batch: batch).policy

        // The same policy logits LossComputer builds internally, scored against an index target
        // the way the OLD (pre-WP-2) format would have stored `policyTarget`.
        let batchSize = batch.spatial.dim(0)
        let policyLogits = concatenated([
            outputs.policySpatialLogits.reshaped(batchSize, -1),
            outputs.policyPassLogits.reshaped(batchSize, -1),
        ], axis: 1)
        let indexTargets = MLXArray(playedMoves.map { Int32($0) })
        let indexLoss = crossEntropy(logits: policyLogits, targets: indexTargets, reduction: .mean)

        eval(denseLoss, indexLoss)
        XCTAssertEqual(denseLoss.item(Float.self), indexLoss.item(Float.self), accuracy: 1e-5)
    }

    /// TRAINING_PROCESS_AUDIT.md P0-3 / WP-2: a sample with `ownershipValid == false` must
    /// contribute exactly zero to BOTH the ownership loss and its gradient, and the loss must be
    /// normalized by the COUNT OF VALID samples rather than the batch size -- so tacking an invalid
    /// (e.g. resigned) sample onto a batch must not perturb training on the valid samples at all.
    /// The invalid sample's ownership is deliberately flipped to the opposite of a real label so a
    /// masking bug (leaked loss/gradient) would show up as an obvious nonzero difference.
    func testOwnershipValidityMaskZeroesLossAndGradientForInvalidSamples() throws {
        try requireHostMLX()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let weights = LossWeights(policy: 0, value: 0, score: 0, ownership: 1)
        let validSamples = (0 ..< 4).map { sample(policy: $0 * 7) }
        let base = sample(policy: 0)
        let invalidSample = RinGoSample(
            spatial: base.spatial, global: base.global, policyTarget: base.policyTarget,
            valueTarget: base.valueTarget, scoreTarget: base.scoreTarget, scoreValid: base.scoreValid,
            ownershipTarget: [Int8](repeating: 1, count: 81), ownershipValid: false
        )

        let (baselineLoss, baselineGrad) = lossAndGradient(
            network: network, weights: weights,
            batch: Trainer.batch(samples: validSamples, nnLen: 9), parameterKey: "conv:vownership/w"
        )
        let (withInvalidLoss, withInvalidGrad) = lossAndGradient(
            network: network, weights: weights,
            batch: Trainer.batch(samples: validSamples + [invalidSample], nnLen: 9), parameterKey: "conv:vownership/w"
        )

        XCTAssertEqual(baselineLoss, withInvalidLoss, accuracy: 1e-6)
        let maxAbsDifference = (baselineGrad - withInvalidGrad).abs().max()
        eval(maxAbsDifference)
        XCTAssertEqual(maxAbsDifference.item(Float.self), 0, accuracy: 1e-6)
    }

    /// Same guarantee as above, for `scoreValid == false`: the huber score loss must be masked out
    /// per-sample AND normalized by the valid count, not the batch size.
    func testScoreValidityMaskZeroesLossAndGradientForInvalidSamples() throws {
        try requireHostMLX()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let weights = LossWeights(policy: 0, value: 0, score: 1, ownership: 0)
        let validSamples = (0 ..< 4).map { sample(policy: $0 * 7) }
        let base = sample(policy: 0)
        let invalidSample = RinGoSample(
            spatial: base.spatial, global: base.global, policyTarget: base.policyTarget,
            valueTarget: base.valueTarget, scoreTarget: 999, scoreValid: false,
            ownershipTarget: base.ownershipTarget, ownershipValid: base.ownershipValid
        )

        let (baselineLoss, baselineGrad) = lossAndGradient(
            network: network, weights: weights,
            batch: Trainer.batch(samples: validSamples, nnLen: 9), parameterKey: "matmul:sv3/w"
        )
        let (withInvalidLoss, withInvalidGrad) = lossAndGradient(
            network: network, weights: weights,
            batch: Trainer.batch(samples: validSamples + [invalidSample], nnLen: 9), parameterKey: "matmul:sv3/w"
        )

        XCTAssertEqual(baselineLoss, withInvalidLoss, accuracy: 1e-6)
        let maxAbsDifference = (baselineGrad - withInvalidGrad).abs().max()
        eval(maxAbsDifference)
        XCTAssertEqual(maxAbsDifference.item(Float.self), 0, accuracy: 1e-6)
    }

    /// Runs one `valueAndGrad` pass of `LossComputer(weights:)` over `batch` and returns the total
    /// loss plus the gradient of a single named parameter (dropping the "values." store-key
    /// prefix `FlatParameterStore` adds -- see `TrainingDiagnosticsTests.flatGrads`).
    private func lossAndGradient(
        network: TrainableNetwork, weights: LossWeights, batch: TrainingBatch, parameterKey: String
    ) -> (Float, MLXArray) {
        let computer = LossComputer(weights: weights)
        let gradientFunction = valueAndGrad(model: network.store) { [network] _, arrays in
            let inputs = TrainingBatch(
                spatial: arrays[0], global: arrays[1], policyTarget: arrays[2],
                valueTarget: arrays[3], scoreTarget: arrays[4], scoreValidTarget: arrays[5],
                ownershipTarget: arrays[6], ownershipValidTarget: arrays[7]
            )
            let losses = computer(outputs: network.forward(spatial: arrays[0], global: arrays[1]), batch: inputs)
            return [losses.total]
        }
        let arrays = [
            batch.spatial, batch.global, batch.policyTarget, batch.valueTarget,
            batch.scoreTarget, batch.scoreValidTarget, batch.ownershipTarget, batch.ownershipValidTarget,
        ]
        let (values, gradients) = gradientFunction(network.store, arrays)
        let flattened = gradients.flattened()
        guard let gradient = flattened.first(where: { $0.0 == "values.\(parameterKey)" })?.1 else {
            preconditionFailure("No gradient found for parameter \(parameterKey)")
        }
        eval(values[0], gradient)
        return (values[0].item(Float.self), gradient)
    }

    /// MLX_TRAINING_AUDIT_0708.md P0-1 regression test: on a CACHE-HIT replay of the compiled
    /// training step (i.e. every call after the first, which is the only one that can possibly
    /// retrace), the current `optimizer.learningRate` must actually reach the compiled kernel --
    /// not the value that happened to be set when the graph was first traced. This is exactly what
    /// `MLXOptimizers.AdamW`'s Swift-`Float` learning rate fails to do (see `RinGoAdamW`'s doc
    /// comment): it gets compiled in as a graph CONSTANT on the one-time retrace, so every later
    /// call replays that same value forever regardless of what the caller sets `learningRate` to
    /// afterward. Six steps at lr = [x, x, x, 0, 0, 0]: the first three must move a probe parameter
    /// every step, and the last three (lr=0, all cache-hit replays) must leave it BIT-IDENTICAL --
    /// `0 * anything` is exactly `0` in IEEE754, so a live schedule freezes the parameter exactly.
    func testCompiledStepHonorsLiveLearningRateIncludingZeroOnCacheHitReplays() throws {
        try requireHostMLX()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let batch = Trainer.batch(samples: (0 ..< 4).map { sample(policy: $0 * 7) }, nnLen: 9)
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: 4, totalSteps: 100, learningRate: 1e-2, warmupSteps: 0,
            weightDecay: 0.01, maximumGradientNorm: 10
        ))
        let probeKey = "conv:vownership/w"

        func probeValues() throws -> [Float] {
            let value = try XCTUnwrap(network.parameters[probeKey])
            eval(value)
            return value.asArray(Float.self)
        }

        let lrSequence: [Float] = [3e-3, 3e-3, 3e-3, 0, 0, 0]
        var snapshots = try [probeValues()]
        for lr in lrSequence {
            _ = trainer.trainOneStep(batch, learningRate: lr)
            try snapshots.append(probeValues())
        }

        // snapshots[0] = before any step; snapshots[i] = after lrSequence[i - 1].
        XCTAssertNotEqual(snapshots[0], snapshots[1], "step 1 (lr>0) must move the parameter")
        XCTAssertNotEqual(snapshots[1], snapshots[2], "step 2 (lr>0) must move the parameter")
        XCTAssertNotEqual(snapshots[2], snapshots[3], "step 3 (lr>0) must move the parameter")
        XCTAssertEqual(snapshots[3], snapshots[4], "step 4 (lr=0, cache-hit replay) must be a no-op")
        XCTAssertEqual(snapshots[4], snapshots[5], "step 5 (lr=0, cache-hit replay) must be a no-op")
        XCTAssertEqual(snapshots[5], snapshots[6], "step 6 (lr=0, cache-hit replay) must be a no-op")
    }

    /// MLX_TRAINING_AUDIT_0708.md P0-1 regression test, cosine-schedule sanity: the compiled step's
    /// live learning-rate argument must actually SCALE the parameter update -- proof against a
    /// stuck "did lr change from its trace-time value" gate that could apply the wrong nonzero
    /// magnitude while still passing a simpler zero-lr test.
    ///
    /// This can NOT be measured by feeding two different lrs sequentially into the SAME optimizer
    /// run (the original design). AdamW's bias-correction factor c1 = lr / (1 - beta1^t) depends
    /// on the step index t, so even a perfectly correct optimizer shows observedRatio != lrRatio
    /// when lrA is probed at step t=4 and lrB at the next step t=5: for lrB/lrA = 3.0, bias
    /// correction alone drags the true ratio down to 3.0 * (1 - 0.9^4) / (1 - 0.9^5) ~= 2.52, and
    /// evolving Adam moment state (m, v) pulls it further, to ~2.28 observed -- not a bug, just
    /// what a correct optimizer does when t differs between the two measurements.
    ///
    /// Instead, probe the SAME step index (t=4) in two separate runs that are seeded identically
    /// (network init draws from the same global MLX RNG, so re-seeding before construction makes
    /// both networks bit-for-bit identical) and driven through identical warmup steps. That keeps
    /// bias correction and Adam moment history identical between the two measurements, so only the
    /// probe step's lr differs and deltaB/deltaA must equal the lr ratio exactly (mod float noise).
    func testCompiledStepScalesParameterDeltaWithLiveLearningRate() throws {
        try requireHostMLX()
        let probeKey = "conv:vownership/w"
        let rngSeed: UInt64 = 20_260_709

        func stepFourDelta(probeLearningRate: Float) throws -> Float {
            MLXRandom.seed(rngSeed)
            let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
            let batch = Trainer.batch(samples: (0 ..< 4).map { sample(policy: $0 * 7) }, nnLen: 9)
            let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
                batchSize: 4, totalSteps: 100, learningRate: 1e-2, warmupSteps: 0,
                weightDecay: 0, maximumGradientNorm: 10
            ))

            func probeValues() throws -> [Float] {
                let value = try XCTUnwrap(network.parameters[probeKey])
                eval(value)
                return value.asArray(Float.self)
            }

            // Three identical fixed-lr warmup steps (cache-hit replays after step 1) so every run
            // reaches step t=4 with identical Adam moment state before the probe step.
            for _ in 0 ..< 3 {
                _ = trainer.trainOneStep(batch, learningRate: 1e-3)
            }

            let before = try probeValues()
            _ = trainer.trainOneStep(batch, learningRate: probeLearningRate)
            let after = try probeValues()
            var sumSquares: Float = 0
            for (a, b) in zip(before, after) {
                sumSquares += (a - b) * (a - b)
            }
            return sqrt(sumSquares)
        }

        let lrA: Float = 2e-3
        let lrB: Float = 6e-3
        let deltaA = try stepFourDelta(probeLearningRate: lrA)
        let deltaB = try stepFourDelta(probeLearningRate: lrB)

        XCTAssertGreaterThan(deltaA, 0)
        let observedRatio = deltaB / deltaA
        let expectedRatio = lrB / lrA
        XCTAssertEqual(observedRatio, expectedRatio, accuracy: expectedRatio * 0.05)
    }

    /// MLX_TRAINING_AUDIT_0708.md P0-2 regression test: `saveCheckpoint` must write via a temp file
    /// + atomic swap, not overwrite `checkpoint.safetensors` in place -- a crash mid-write must
    /// never destroy the only resume point of an hours-long run. Exercises both the first-ever
    /// checkpoint path (no existing file to replace) and the overwrite path (an existing checkpoint
    /// being replaced), and asserts no temp file is left behind in either case.
    func testSaveCheckpointIsAtomicAndLeavesNoTempFiles() throws {
        try requireHostMLX()
        let batch = Trainer.batch(samples: (0 ..< 4).map { sample(policy: $0 % 82) }, nnLen: 9)
        let config = TrainerConfiguration(
            batchSize: 4, totalSteps: 10, learningRate: 1e-3, warmupSteps: 0,
            weightDecay: 0.01, maximumGradientNorm: 10
        )
        let trainer = try Trainer(
            network: TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9),
            configuration: config
        )
        _ = trainer.trainOneStep(batch)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ringo-ckpt-atomic-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("checkpoint.safetensors")

        try trainer.saveCheckpoint(to: url) // first-ever checkpoint: no existing file to replace
        let contentsAfterFirstSave = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertEqual(contentsAfterFirstSave, ["checkpoint.safetensors"])

        _ = trainer.trainOneStep(batch)
        try trainer.saveCheckpoint(to: url) // overwrite: exercises the replaceItemAt path
        let contentsAfterSecondSave = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertEqual(
            contentsAfterSecondSave, ["checkpoint.safetensors"], "no temp files should remain after save"
        )

        let resumed = try Trainer(
            network: TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9),
            configuration: config
        )
        try resumed.loadCheckpoint(from: url)
        XCTAssertEqual(resumed.step, trainer.step)
    }

    /// A checkpoint round-trip must restore params, step, AND optimizer (Adam) state exactly, so a
    /// resumed run continues identically. We verify that by training the resumed trainer one more
    /// step from the same batch and asserting its metrics match the original trainer's next step
    /// bit-for-bit (the next Adam update depends on the restored moments, so a match proves state
    /// was restored, not just the weights).
    func testCheckpointRoundtripResumesIdentically() throws {
        try requireHostMLX()
        let batch = Trainer.batch(samples: (0 ..< 8).map { sample(policy: $0 % 82) }, nnLen: 9)
        let config = TrainerConfiguration(
            batchSize: 8, totalSteps: 100, learningRate: 2e-3, warmupSteps: 5,
            weightDecay: 0.01, maximumGradientNorm: 10
        )
        let original = try Trainer(network: TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9),
                                   configuration: config)
        for _ in 0 ..< 12 {
            _ = original.trainOneStep(batch)
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ringo-ckpt-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try original.saveCheckpoint(to: url)

        let resumed = try Trainer(network: TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9),
                                  configuration: config)
        try resumed.loadCheckpoint(from: url)
        XCTAssertEqual(resumed.step, original.step)

        // Continue both one more step from the same batch; metrics must match if state was restored.
        let a = original.trainOneStep(batch)
        let b = resumed.trainOneStep(batch)
        XCTAssertEqual(a.step, b.step)
        XCTAssertEqual(a.totalLoss, b.totalLoss, accuracy: 1e-5)
        XCTAssertEqual(a.policyLoss, b.policyLoss, accuracy: 1e-5)
        XCTAssertEqual(a.learningRate, b.learningRate, accuracy: 1e-7)
    }

    /// Audit P1-2 identity: assembling a batch from the columnar u8 store (`RinGoShard.batch`) must
    /// be BIT-IDENTICAL to the reference fp32 path (`Trainer.batch` over materialized samples) for all
    /// 8 arrays -- 0/1 u8 maps to Float 0.0/1.0 exactly and every other field is copied verbatim.
    /// Uses a permutation with duplicate indices so a wrong offset / swapped-sample bug would surface.
    func testColumnarBatchMatchesReferenceFp32Path() throws {
        try requireHostMLX()
        let samples = (0 ..< 6).map { variedSample(seed: $0 + 1) }
        let shard = try RinGoShard(nnLen: 9, samples: samples)
        let indices = [4, 1, 5, 0, 3, 2, 2, 0] // permutation + duplicates (training can repeat a sample)
        let reference = Trainer.batch(samples: indices.map { samples[$0] }, nnLen: 9)
        let columnar = shard.batch(indices: indices)
        assertBatchArraysEqual(reference, columnar)
    }

    /// Audit P1-3 identity: the compiled validation pass (`Trainer.evaluate`) must produce the same
    /// numbers as the eager reference (`Trainer.evaluateEager`) -- same weighted-mean averaging and
    /// masking. 70 samples at batch 16 forces a partial final microbatch (4 full + remainder 6), so
    /// the eager-remainder branch of the compiled path is exercised. One training step first moves the
    /// params off init, proving the compiled function reads the CURRENT params (not trace-time ones).
    func testCompiledValidationMatchesEager() throws {
        try requireHostMLX()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let samples = (0 ..< 70).map { variedSample(seed: $0 + 1) }
        let shard = try RinGoShard(nnLen: 9, samples: samples)
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: 16, totalSteps: 10, learningRate: 1e-3, warmupSteps: 0,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        _ = trainer.trainOneStep(Trainer.batch(samples: samples, nnLen: 9))

        let eager = try trainer.evaluateEager(validation: shard)
        let compiled = try trainer.evaluate(validation: shard)

        XCTAssertEqual(compiled.sampleCount, 70)
        XCTAssertEqual(compiled.sampleCount, eager.sampleCount)
        func close(_ lhs: Float, _ rhs: Float, _ name: String) {
            XCTAssertEqual(lhs, rhs, accuracy: max(1e-6, abs(rhs) * 1e-5), name)
        }
        close(compiled.policyLoss, eager.policyLoss, "policy")
        close(compiled.valueLoss, eager.valueLoss, "value")
        close(compiled.scoreLoss, eager.scoreLoss, "score")
        close(compiled.ownershipLoss, eager.ownershipLoss, "ownership")
        close(compiled.totalLoss, eager.totalLoss, "total")
    }

    /// Asserts two `TrainingBatch`es are element-exact equal across all 8 arrays (shape + values).
    private func assertBatchArraysEqual(
        _ lhs: TrainingBatch, _ rhs: TrainingBatch, file: StaticString = #filePath, line: UInt = #line
    ) {
        func check(_ a: MLXArray, _ b: MLXArray, _ name: String) {
            eval(a, b)
            XCTAssertEqual(a.shape, b.shape, "\(name) shape", file: file, line: line)
            XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self), "\(name) values", file: file, line: line)
        }
        check(lhs.spatial, rhs.spatial, "spatial")
        check(lhs.global, rhs.global, "global")
        check(lhs.policyTarget, rhs.policyTarget, "policy")
        check(lhs.valueTarget, rhs.valueTarget, "value")
        check(lhs.scoreTarget, rhs.scoreTarget, "score")
        check(lhs.scoreValidTarget, rhs.scoreValidTarget, "scoreValid")
        check(lhs.ownershipTarget, rhs.ownershipTarget, "ownership")
        check(lhs.ownershipValidTarget, rhs.ownershipValidTarget, "ownershipValid")
    }

    /// A sample whose every field varies deterministically with `seed` (so an indexing/offset bug is
    /// detectable), with BINARY spatial planes (the u8 packing's invariant) and mixed validity flags.
    private func variedSample(seed: Int) -> RinGoSample {
        var spatial = [Float](repeating: 0, count: 9 * 9 * 22)
        for i in spatial.indices {
            spatial[i] = ((i * 31 + seed * 17) % 3 == 0) ? 1 : 0
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

    /// WP-5: every supported `-arch` variant must construct, pass `TrainableNetwork.validate`, build
    /// its MLX graph, and run a forward pass producing finite, correctly-shaped outputs. That is the
    /// end-to-end proof the parameterized topology (varying depth/width) is wired consistently — a
    /// mismatched matmul/conv dimension would crash the forward or produce the wrong policy width.
    func testAllVariantsConstructValidateAndRunForward() throws {
        try requireHostMLX()
        for variant in RinGoArchitecture.Variant.allCases {
            let network = try TrainableNetwork(desc: RinGoArchitecture.make(variant), nnXLen: 9, nnYLen: 9)
            let batch = Trainer.batch(samples: (0 ..< 3).map { sample(policy: $0 * 11) }, nnLen: 9)
            let outputs = network.forward(spatial: batch.spatial, global: batch.global)
            let spatial = outputs.policySpatialLogits.reshaped(3, -1)
            let pass = outputs.policyPassLogits.reshaped(3, -1)
            // 9x9 board => 81 spatial policy logits + 1 pass logit == 82 move logits.
            XCTAssertEqual(spatial.dim(1), 81, "\(variant) spatial policy width")
            XCTAssertEqual(pass.dim(1), 1, "\(variant) pass logit")
            // A NaN/Inf anywhere in an output propagates through the sum, so Float.isFinite on the
            // reduced scalar is a whole-tensor finiteness check using only proven MLX/Swift APIs.
            let policySum = spatial.sum().item(Float.self)
            let valueSum = outputs.valueLogits.sum().item(Float.self)
            XCTAssertTrue(policySum.isFinite, "\(variant) finite policy logits")
            XCTAssertTrue(valueSum.isFinite, "\(variant) finite value logits")
        }
    }

    /// WP-5: the larger student must actually train — a handful of compiled optimizer steps on a
    /// tiny synthetic batch must drive its total loss down (same recipe as b6c96, just b10c128).
    func testB10C128CompiledTrainStepsDecreaseLoss() throws {
        try requireHostMLX()
        let network = try TrainableNetwork(desc: RinGoArchitecture.b10c128(), nnXLen: 9, nnYLen: 9)
        let batch = Trainer.batch(samples: (0 ..< 8).map { sample(policy: $0 % 82) }, nnLen: 9)
        let computer = LossComputer()
        let before = computer(outputs: network.forward(spatial: batch.spatial, global: batch.global), batch: batch)
            .total
        eval(before)
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: 8, totalSteps: 20, learningRate: 3e-3, warmupSteps: 2,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        for _ in 0 ..< 20 {
            _ = trainer.trainOneStep(batch)
        }
        let after = computer(outputs: network.forward(spatial: batch.spatial, global: batch.global), batch: batch)
            .total
        eval(after)
        XCTAssertTrue(after.item(Float.self).isFinite)
        XCTAssertLessThan(after.item(Float.self), before.item(Float.self))
    }

    /// WP-5: the v8 desc format is size-generic — a b10c128 net exported via `ModelWriter` must
    /// re-parse to a structurally identical `ModelDesc` (same param count, block kinds, trunk
    /// widths). This proves `.bin.gz` export/import is not hardwired to b6c96's dimensions.
    func testB10C128ExportRoundtripThroughModelWriterPreservesTopology() throws {
        try requireHostMLX()
        let source = RinGoArchitecture.b10c128()
        let network = try TrainableNetwork(desc: source, nnXLen: 9, nnYLen: 9)
        let exported = network.exportDesc()

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ringo-b10c128-\(UUID().uuidString).bin.gz")
        defer { try? FileManager.default.removeItem(at: url) }
        try ModelWriter.write(desc: exported, to: url)

        let reloaded = try ModelDesc.loadFromFileMaybeGZipped(url.path)
        // Compare against `exported`, not `source`: `exportDesc` emits an explicit scale for every
        // BatchNorm (KataGo folds scale into `mergedScale`), whereas the placeholder `source` desc
        // marks the scale-free BNs `hasScale == false`. Both describe the same network; only the
        // exported form's official `numParameters` counts those scales, and write/read preserves
        // tensor shapes exactly, so reloaded must match exported to the parameter.
        XCTAssertEqual(reloaded.numParameters, exported.numParameters)
        XCTAssertEqual(reloaded.modelVersion, source.modelVersion)
        XCTAssertEqual(reloaded.trunk.numBlocks, source.trunk.numBlocks)
        XCTAssertEqual(reloaded.trunk.trunkNumChannels, source.trunk.trunkNumChannels)
        XCTAssertEqual(reloaded.trunk.regularNumChannels, source.trunk.regularNumChannels)
        XCTAssertEqual(reloaded.trunk.gpoolNumChannels, source.trunk.gpoolNumChannels)
        XCTAssertEqual(reloaded.trunk.blocks.map(\.kind), source.trunk.blocks.map(\.kind))
        XCTAssertEqual(reloaded.trunk.initialConv.weights.count, source.trunk.initialConv.weights.count)
        XCTAssertEqual(reloaded.valueHead.v2Mul.weights.count, source.valueHead.v2Mul.weights.count)
    }

    /// WP-5 resume safety: a checkpoint saved under one architecture must NOT load into another.
    /// `TrainableNetwork.replaceParameters` (via `loadCheckpoint`) must throw rather than misload.
    func testLoadingB6C96CheckpointIntoB10C128Fails() throws {
        try requireHostMLX()
        let saver = try Trainer(
            network: TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9),
            configuration: TrainerConfiguration(
                batchSize: 4, totalSteps: 10, learningRate: 1e-3, warmupSteps: 0,
                weightDecay: 0, maximumGradientNorm: 10
            )
        )
        _ = saver.trainOneStep(Trainer.batch(samples: (0 ..< 4).map { sample(policy: $0 * 7) }, nnLen: 9))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ringo-arch-mismatch-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try saver.saveCheckpoint(to: url)

        let loader = try Trainer(
            network: TrainableNetwork(desc: RinGoArchitecture.b10c128(), nnXLen: 9, nnYLen: 9),
            configuration: TrainerConfiguration(
                batchSize: 4, totalSteps: 10, learningRate: 1e-3, warmupSteps: 0,
                weightDecay: 0, maximumGradientNorm: 10
            )
        )
        XCTAssertThrowsError(try loader.loadCheckpoint(from: url))
    }

    // MARK: - WP-8: -init-model warm start

    /// The acceptance bar for `-init-model`: warm-starting is the EXACT inverse of `exportDesc`.
    /// After `train a few steps -> exportDesc -> loadDonorParameters`, the recipient network's
    /// parameter store must be element-for-element identical to the donor's (all conv/matmul/bias/
    /// bn tensors), and a forward pass must reproduce the donor's raw NN outputs bit-exactly.
    func testInitModelReproducesDonorParametersAndForwardExactly() throws {
        try requireHostMLX()
        // Donor: a b6c96 net trained a few steps so its weights (and folded BN scales) are nontrivial.
        let donor = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let batch = Trainer.batch(samples: (0 ..< 4).map { sample(policy: $0 * 19) }, nnLen: 9)
        let trainer = try Trainer(network: donor, configuration: TrainerConfiguration(
            batchSize: 4, totalSteps: 10, learningRate: 5e-3, warmupSteps: 0,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        for _ in 0 ..< 5 {
            _ = trainer.trainOneStep(batch)
        }

        let donorDesc = donor.exportDesc()
        let recipient = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        try recipient.loadDonorParameters(from: donorDesc)

        // (1) Every store parameter is element-exact identical (loadDonorParameters ∘ exportDesc == id).
        let donorParameters = donor.parameters
        let recipientParameters = recipient.parameters
        XCTAssertEqual(Set(donorParameters.keys), Set(recipientParameters.keys))
        for (key, donorValue) in donorParameters {
            let recipientValue = try XCTUnwrap(recipientParameters[key])
            eval(donorValue, recipientValue)
            XCTAssertEqual(donorValue.shape, recipientValue.shape, "\(key) shape")
            XCTAssertEqual(donorValue.asArray(Float.self), recipientValue.asArray(Float.self), "\(key) values")
        }

        // (2) The raw NN outputs match bit-exactly (same params, same graph).
        let reference = donor.forward(spatial: batch.spatial, global: batch.global)
        let warmStarted = recipient.forward(spatial: batch.spatial, global: batch.global)
        assertRawOutputsEqual(reference, warmStarted, accuracy: 0)
    }

    /// The realistic CLI path: warm-start from an on-disk `.bin.gz` (parse -> loadDonorParameters).
    /// This routes through `ModelWriter`'s activation-reduction transform, so it is only guaranteed
    /// close (not bit-exact) — the tolerance here is the documented ModelWriter round-trip precision,
    /// proving init-from-file faithfully reconstructs the donor's inference behavior.
    func testInitModelFromBinGzFileMatchesDonorWithinRoundtripTolerance() throws {
        try requireHostMLX()
        let donor = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let batch = Trainer.batch(samples: (0 ..< 4).map { sample(policy: $0 * 7 + 1) }, nnLen: 9)
        let trainer = try Trainer(network: donor, configuration: TrainerConfiguration(
            batchSize: 4, totalSteps: 10, learningRate: 5e-3, warmupSteps: 0,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        for _ in 0 ..< 5 {
            _ = trainer.trainOneStep(batch)
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ringo-init-\(UUID().uuidString).bin.gz")
        defer { try? FileManager.default.removeItem(at: url) }
        try ModelWriter.write(desc: donor.exportDesc(), to: url)

        let reloaded = try ModelDesc.loadFromFileMaybeGZipped(url.path)
        let recipient = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        try recipient.loadDonorParameters(from: reloaded)

        let reference = donor.forward(spatial: batch.spatial, global: batch.global)
        let warmStarted = recipient.forward(spatial: batch.spatial, global: batch.global)
        // Value/score logits are O(1); policy logits are O(10). 1e-3 comfortably covers the
        // ModelWriter fold/unfold rounding while still catching a real load bug.
        assertRawOutputsEqual(reference, warmStarted, accuracy: 1e-3)
    }

    /// Arch-mismatch must fail loudly, reusing WP-5's key/shape guard: a b6c96 donor cannot load
    /// into a b10c128 network (different block count AND trunk widths).
    func testLoadDonorParametersRejectsArchitectureMismatch() throws {
        try requireHostMLX()
        let donor = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let donorDesc = donor.exportDesc()
        let recipient = try TrainableNetwork(desc: RinGoArchitecture.b10c128(), nnXLen: 9, nnYLen: 9)
        XCTAssertThrowsError(try recipient.loadDonorParameters(from: donorDesc)) { error in
            XCTAssertTrue(error is TrainableNetworkError, "expected a clear arch-mismatch error, got \(error)")
        }
    }

    /// A donor whose modelVersion is unsupported by the training network must be rejected by
    /// `loadDonorParameters`' `validate` guard rather than misloaded — the same guard that gates
    /// random init, so warm start can never smuggle in an unsupported topology.
    func testLoadDonorParametersRejectsUnsupportedModelVersion() throws {
        try requireHostMLX()
        var donorDesc = RinGoArchitecture.b6c96()
        donorDesc.modelVersion = 9 // RinGoTrain v1 supports only modelVersion 8
        let recipient = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        XCTAssertThrowsError(try recipient.loadDonorParameters(from: donorDesc)) { error in
            XCTAssertTrue(error is TrainableNetworkError, "expected a validate() rejection, got \(error)")
        }
    }

    /// The training payoff: warm-starting from a trained donor begins at a LOW loss (near the
    /// donor's, far below a random-init net) and continued training drives it down further. This is
    /// the end-to-end proof the donor's learned weights actually loaded — random init could not
    /// start this low.
    func testInitModelStartsAtLowLossAndContinuesToDecrease() throws {
        try requireHostMLX()
        // A small fixed synthetic dataset the donor learns to fit.
        let samples = (0 ..< 8).map { variedSample(seed: $0 + 1) }
        let batch = Trainer.batch(samples: samples, nnLen: 9)
        let computer = LossComputer()

        // Train a donor to a moderately low (not fully converged) loss on this batch.
        let donor = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let donorTrainer = try Trainer(network: donor, configuration: TrainerConfiguration(
            batchSize: 8, totalSteps: 80, learningRate: 3e-3, warmupSteps: 5,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        for _ in 0 ..< 80 {
            _ = donorTrainer.trainOneStep(batch)
        }
        let donorLoss = lossValue(computer, donor, batch)

        // A fresh random-init net starts high.
        let randomNet = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        let randomLoss = lossValue(computer, randomNet, batch)

        // Warm-started net starts near the donor's loss (and far below random).
        let warmStarted = try TrainableNetwork(desc: RinGoArchitecture.b6c96(), nnXLen: 9, nnYLen: 9)
        try warmStarted.loadDonorParameters(from: donor.exportDesc())
        let warmStartInitialLoss = lossValue(computer, warmStarted, batch)

        XCTAssertEqual(warmStartInitialLoss, donorLoss, accuracy: max(1e-4, abs(donorLoss) * 1e-3),
                       "warm-started loss must equal the donor's")
        XCTAssertLessThan(warmStartInitialLoss, randomLoss * 0.9,
                          "warm-started loss (\(warmStartInitialLoss)) must be well below random (\(randomLoss))")

        // Continue training the warm-started net at a LOW polish-style LR: from a near-converged
        // donor a large LR overshoots, so realistic warm-start fine-tuning uses a small step. The
        // loss must keep dropping.
        let continueTrainer = try Trainer(network: warmStarted, configuration: TrainerConfiguration(
            batchSize: 8, totalSteps: 40, learningRate: 3e-4, warmupSteps: 0,
            weightDecay: 0, maximumGradientNorm: 10
        ))
        for _ in 0 ..< 40 {
            _ = continueTrainer.trainOneStep(batch)
        }
        let afterLoss = lossValue(computer, warmStarted, batch)
        XCTAssertLessThan(afterLoss, warmStartInitialLoss, "continued training must decrease the loss")
        XCTAssertTrue(afterLoss.isFinite)
    }

    private func lossValue(_ computer: LossComputer, _ network: TrainableNetwork, _ batch: TrainingBatch) -> Float {
        let loss = computer(outputs: network.forward(spatial: batch.spatial, global: batch.global), batch: batch).total
        eval(loss)
        return loss.item(Float.self)
    }

    /// Asserts two `ModelRawOutputs` agree element-wise across all five heads. `accuracy == 0`
    /// demands bit-exact equality (used for the in-memory warm-start path).
    private func assertRawOutputsEqual(
        _ lhs: ModelRawOutputs, _ rhs: ModelRawOutputs, accuracy: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        func check(_ a: MLXArray, _ b: MLXArray, _ name: String) {
            eval(a, b)
            XCTAssertEqual(a.shape, b.shape, "\(name) shape", file: file, line: line)
            let lhsValues = a.asArray(Float.self)
            let rhsValues = b.asArray(Float.self)
            var maxDiff: Float = 0
            for (x, y) in zip(lhsValues, rhsValues) {
                maxDiff = Swift.max(maxDiff, abs(x - y))
            }
            XCTAssertLessThanOrEqual(maxDiff, accuracy, "\(name) max abs diff \(maxDiff)", file: file, line: line)
        }
        check(lhs.policySpatialLogits, rhs.policySpatialLogits, "policySpatial")
        check(lhs.policyPassLogits, rhs.policyPassLogits, "policyPass")
        check(lhs.valueLogits, rhs.valueLogits, "value")
        check(lhs.scoreValues, rhs.scoreValues, "score")
        check(lhs.ownership, rhs.ownership, "ownership")
    }

    private func requireHostMLX() throws {
        guard ProcessInfo.processInfo.environment["RUN_HOST_ONLY_MLX_TESTS"] == "1" else {
            throw XCTSkip("HOST-ONLY: set RUN_HOST_ONLY_MLX_TESTS=1 on a Metal-capable host")
        }
    }

    private func sample(policy: Int) -> RinGoSample {
        var spatial = [Float](repeating: 0, count: 9 * 9 * 22)
        for point in 0 ..< 81 {
            spatial[point * 22] = 1
        }
        var policyTarget = [Float](repeating: 0, count: 82)
        policyTarget[policy] = 1
        return RinGoSample(
            spatial: spatial,
            global: [Float](repeating: 0, count: 19),
            policyTarget: policyTarget,
            valueTarget: [1, 0, 0],
            scoreTarget: 0,
            scoreValid: true,
            ownershipTarget: [Int8](repeating: 0, count: 81),
            ownershipValid: true
        )
    }
}
