import Foundation
import MLX
import MLXNN
import MLXOptimizers
import RinGoModel

public struct TrainerConfiguration: Sendable {
    public var batchSize: Int
    public var totalSteps: Int
    public var learningRate: Float
    public var warmupSteps: Int
    public var weightDecay: Float
    public var maximumGradientNorm: Float
    public var checkpointInterval: Int
    public var snapshotInterval: Int
    public var metricsInterval: Int
    public var lossWeights: LossWeights
    /// Opt-in GPU cache limit in MB (`MLX.Memory.cacheLimit`), applied once at `Trainer` init.
    /// `nil` leaves MLX's default, mirroring `NNEvaluator`'s inference-side knob. Not defaulted to
    /// a nonzero value on purpose: this exists so a long unattended run *can* bound GPU cache, not
    /// to change the default behaviour.
    public var gpuCacheLimitMB: Int?
    /// If non-nil, every `validationInterval` steps the trainer evaluates the loss on the
    /// validation shard passed to `train(dataset:validation:outputDirectory:)` and appends one row
    /// to `validation_metrics.csv`. The validation forward pass is run OUTSIDE the compiled training
    /// step (no grad, no optimizer update), so the training graph is unaffected.
    public var validationInterval: Int?

    public init(
        batchSize: Int = 256, totalSteps: Int, learningRate: Float = 3e-4,
        warmupSteps: Int = 1000, weightDecay: Float = 0.01,
        maximumGradientNorm: Float = 1, checkpointInterval: Int = 1000,
        snapshotInterval: Int = 5000, metricsInterval: Int = 1,
        lossWeights: LossWeights = LossWeights(), gpuCacheLimitMB: Int? = nil,
        validationInterval: Int? = nil
    ) {
        self.batchSize = batchSize
        self.totalSteps = totalSteps
        self.learningRate = learningRate
        self.warmupSteps = warmupSteps
        self.weightDecay = weightDecay
        self.maximumGradientNorm = maximumGradientNorm
        self.checkpointInterval = checkpointInterval
        self.snapshotInterval = snapshotInterval
        self.metricsInterval = metricsInterval
        self.lossWeights = lossWeights
        self.gpuCacheLimitMB = gpuCacheLimitMB
        self.validationInterval = validationInterval
    }
}

public struct TrainingMetrics: Sendable {
    public let step: Int
    public let learningRate: Float
    public let policyLoss: Float
    public let valueLoss: Float
    public let scoreLoss: Float
    public let ownershipLoss: Float
    public let totalLoss: Float
}

/// One row of `validation_metrics.csv`. Weighted mean over a validation shard; produced by
/// `Trainer.evaluate(validation:)` (no grad) and also by `ringo evaluate` (the standalone
/// subcommand reads the same loss formulas via `LossComputer`).
public struct ValidationMetrics: Sendable {
    public let step: Int
    public let policyLoss: Float
    public let valueLoss: Float
    public let scoreLoss: Float
    public let ownershipLoss: Float
    public let totalLoss: Float
    public let sampleCount: Int

    public init(
        step: Int, policyLoss: Float, valueLoss: Float, scoreLoss: Float,
        ownershipLoss: Float, totalLoss: Float, sampleCount: Int
    ) {
        self.step = step
        self.policyLoss = policyLoss
        self.valueLoss = valueLoss
        self.scoreLoss = scoreLoss
        self.ownershipLoss = ownershipLoss
        self.totalLoss = totalLoss
        self.sampleCount = sampleCount
    }
}

public final class Trainer {
    public let network: TrainableNetwork
    public private(set) var step: Int
    private let configuration: TrainerConfiguration
    private let lossComputer: LossComputer
    private let optimizer: RinGoAdamW
    /// The whole forward + backward + optimizer update, `MLX.compile`d once and reused every step
    /// (built lazily on the first step so the batch shape that fixes the trace is known). Rebuilding
    /// it per step would defeat compilation, so it is cached here. See `makeCompiledStep`.
    private var compiledStep: (([MLXArray]) -> [MLXArray])?
    /// The validation forward + loss, `MLX.compile`d once and reused across every microbatch of
    /// every validation pass. Built lazily on the first `evaluate` call. Invalidated
    /// alongside `compiledStep` on `loadCheckpoint` (the restored param state reshapes the trace).
    /// See `makeCompiledEvaluate`.
    private var compiledEvaluate: (([MLXArray]) -> [MLXArray])?

    public init(network: TrainableNetwork, configuration: TrainerConfiguration, step: Int = 0) throws {
        // Apply the optional GPU cache limit once at construction, mirroring
        // `NNEvaluator`'s inference-side pattern. Opt-in only: `nil` leaves MLX's default.
        if let gpuCacheLimitMB = configuration.gpuCacheLimitMB {
            guard gpuCacheLimitMB >= 0 else {
                throw TrainableNetworkError("gpuCacheLimitMB must be >= 0, got \(gpuCacheLimitMB)")
            }
            MLX.Memory.cacheLimit = gpuCacheLimitMB * 1024 * 1024
        }
        self.network = network
        self.configuration = configuration
        self.step = step
        lossComputer = LossComputer(weights: configuration.lossWeights)
        optimizer = RinGoAdamW(
            learningRate: configuration.learningRate,
            weightDecay: configuration.weightDecay
        )
    }

    @discardableResult
    public func trainOneStep(_ batch: TrainingBatch) -> TrainingMetrics {
        trainOneStep(batch, learningRate: scheduledLearningRate(at: step))
    }

    /// Runs one compiled step at a CALLER-SUPPLIED learning rate rather than the schedule computed
    /// from `step`. `internal` (not `private`) so `@testable import` tests can inject an exact lr
    /// sequence -- e.g. to prove lr=0 truly reaches the compiled kernel on a cache-hit replay,
    /// which is exactly what `MLXOptimizers.AdamW`'s Swift-Float lr fails to do (see
    /// `RinGoAdamW`'s doc comment).
    @discardableResult
    func trainOneStep(_ batch: TrainingBatch, learningRate: Float) -> TrainingMetrics {
        // Set the rate BEFORE invoking the compiled step: `RinGoAdamW.learningRate` is backed by
        // an MLXArray that is part of `optimizer.innerState()`, so it flows into the compiled
        // step's REAL graph inputs every call -- including on a cache-hit replay, where the Swift
        // closure that built the trace is never re-invoked. See `RinGoAdamW`'s doc comment.
        optimizer.learningRate = learningRate
        let step_ = compiledStep ?? {
            let built = makeCompiledStep()
            compiledStep = built
            return built
        }()
        let lossValues = step_(Self.arrays(from: batch))
        // The compiled step already materializes results and threads the updated params/optimizer
        // state through its output captures; this eval is the single designated sync point and is
        // effectively free once the graph has run.
        eval(lossValues + Array(network.parameters.values) + optimizer.innerState())
        step += 1
        let losses = Self.unpackLosses(lossValues)
        // The per-step `.item` reads are cheap host copies now that everything above is already
        // materialized. `metricsInterval` gates whether these are *logged* (see `train`), not
        // whether they are computed -- they are a byproduct of the gradient value, no extra work.
        return TrainingMetrics(
            step: step,
            learningRate: learningRate,
            policyLoss: losses.policy.item(Float.self),
            valueLoss: losses.value.item(Float.self),
            scoreLoss: losses.score.item(Float.self),
            ownershipLoss: losses.ownership.item(Float.self),
            totalLoss: losses.total.item(Float.self)
        )
    }

    /// Builds the compiled per-step function following MLX's canonical training-graph pattern
    /// (`compile(inputs: [model, optimizer], outputs: [model, optimizer])` wrapping
    /// valueAndGrad + optimizer.update). Built once and cached in `compiledStep`; the batch shape
    /// is constant across a run (`train` always emits `batchSize`-shaped batches), so the trace is
    /// reused every step and kernels stay fused.
    private func makeCompiledStep() -> ([MLXArray]) -> [MLXArray] {
        let gradientFunction = valueAndGrad(model: network.store) { [network, lossComputer] _, arrays in
            // The bound `model` parameter is intentionally ignored (`_`). valueAndGrad's wrapper
            // calls `model.update(parameters:)` on `network.store` -- the SAME object passed as
            // `model:` -- immediately before invoking this closure (mlx-swift ValueAndGrad.swift
            // ~104-107), so reading the captured `network.store` via `network.forward` observes the
            // freshly substituted tracer parameters and keeps the gradient graph intact. Do NOT
            // "clean this up" to use the ignored parameter or read a different object: `network.store`
            // is exactly what valueAndGrad differentiates, and rewiring it would detach the gradients.
            let inputs = TrainingBatch(
                spatial: arrays[0], global: arrays[1], policyTarget: arrays[2],
                valueTarget: arrays[3], scoreTarget: arrays[4], scoreValidTarget: arrays[5],
                ownershipTarget: arrays[6], ownershipValidTarget: arrays[7]
            )
            // Return ALL loss components, not just `.total`: valueAndGrad materializes its value
            // return alongside the gradients for free, so this is the metrics source -- there is no
            // separate forward pass just to break the loss down for logging.
            let losses = lossComputer(outputs: network.forward(spatial: arrays[0], global: arrays[1]), batch: inputs)
            return Self.packLosses(losses)
        }
        let maximumGradientNorm = configuration.maximumGradientNorm
        let state: [any Updatable] = [network.store, optimizer]
        return compile(inputs: state, outputs: state) { [network, optimizer] arrays in
            let (lossValues, gradients) = gradientFunction(network.store, arrays)
            let (clipped, _) = clipGradNorm(gradients: gradients, maxNorm: maximumGradientNorm)
            optimizer.update(model: network.store, gradients: clipped)
            return lossValues
        }
    }

    /// Packs the loss components into a fixed-order `[MLXArray]` so the `valueAndGrad` closure can
    /// return them as its value (unpacked by `unpackLosses`). Order: total, policy, value, score,
    /// ownership. `total` leads because it is the differentiated scalar.
    private static func packLosses(_ losses: Losses) -> [MLXArray] {
        [losses.total, losses.policy, losses.value, losses.score, losses.ownership]
    }

    /// Inverse of `packLosses`.
    private static func unpackLosses(_ values: [MLXArray]) -> Losses {
        Losses(policy: values[1], value: values[2], score: values[3], ownership: values[4], total: values[0])
    }

    public func train(
        dataset: RinGoShard, outputDirectory: URL, validation: RinGoShard? = nil
    ) throws {
        guard dataset.nnLen > 0, dataset.sampleCount > 0 else {
            throw RinGoShardError("Training dataset is empty")
        }
        if let validation, validation.nnLen != dataset.nnLen || validation.sampleCount == 0 {
            throw RinGoShardError("Validation shard must be non-empty and match the training nnLen")
        }
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let logURL = outputDirectory.appendingPathComponent("training.log")
        var logger = TrainingLogger(url: logURL)
        let metricsURL = outputDirectory.appendingPathComponent("metrics.csv")
        try prepareMetricsFile(metricsURL)
        let validationMetricsURL = outputDirectory.appendingPathComponent("validation_metrics.csv")
        if validation != nil, configuration.validationInterval != nil {
            try prepareValidationMetricsFile(validationMetricsURL)
        }
        try logger.open()
        defer { try? logger.close() }
        // Log both counts explicitly so a run directory is self-auditing without
        // digging into config.json — `dataset` here has already had the validation shard excluded
        // by the caller (`TrainCommand.main`), so `trainSamples` is genuinely train-only.
        try logger.log(
            "info",
            "training started (resumeStep=\(step), target=\(configuration.totalSteps), "
                + "trainSamples=\(dataset.sampleCount), validationSamples=\(validation?.sampleCount ?? 0))"
        )
        var indices = Array(0 ..< dataset.sampleCount)
        indices.shuffle()
        var offset = 0
        while step < configuration.totalSteps {
            if offset + configuration.batchSize > indices.count {
                indices.shuffle()
                offset = 0
            }
            let count = min(configuration.batchSize, indices.count)
            let selected = (0 ..< count).map { indices[(offset + $0) % indices.count] }
            offset += count
            // Columnar hot path: assemble the batch straight from the shard's u8/Float
            // buffers, converting spatial u8 → Float32 inline. No intermediate `RinGoSample`s.
            let batch = dataset.batch(indices: selected)
            let metrics = trainOneStep(batch)
            if configuration.metricsInterval > 0, metrics.step.isMultiple(of: configuration.metricsInterval) {
                try append(metrics, to: metricsURL)
            }
            if configuration.checkpointInterval > 0, metrics.step.isMultiple(of: configuration.checkpointInterval) {
                try saveCheckpoint(to: outputDirectory.appendingPathComponent("checkpoint.safetensors"))
                try logger.log("info", "checkpoint saved at step \(metrics.step)")
            }
            if configuration.snapshotInterval > 0, metrics.step.isMultiple(of: configuration.snapshotInterval) {
                let name = String(format: "model-step-%08d.bin.gz", metrics.step)
                try ModelWriter.write(desc: network.exportDesc(), to: outputDirectory.appendingPathComponent(name))
                try logger.log("info", "snapshot saved at step \(metrics.step) -> \(name)")
            }
            if let validation, configuration.validationInterval != nil {
                let interval = configuration.validationInterval!
                guard interval > 0, metrics.step.isMultiple(of: interval) else { continue }
                let result = try evaluate(validation: validation)
                try appendValidation(result, to: validationMetricsURL)
                try logger.log("info", "validation at step \(metrics.step): total=\(result.totalLoss)")
            }
        }
        try logger.log("info", "training completed at step \(step)")
    }

    /// Pure forward pass over a validation shard (no grad, no optimizer update). Runs OUTSIDE the
    /// compiled training step so the training graph is untouched -- see `validationInterval`'s doc
    /// comment. Losses are microbatch-averaged to a single weighted mean.
    ///
    /// The forward + loss is `MLX.compile`d once (via `makeCompiledEvaluate`) and reused
    /// for every full microbatch, and the per-component weighted sums are accumulated ON DEVICE so
    /// there is exactly ONE host sync at the end instead of five per microbatch. A partial final
    /// microbatch (the remainder) is evaluated eagerly because its shape differs from the compiled
    /// trace; routing it through the compiled function would force a second trace. The averaging and
    /// masking semantics are identical to `evaluateEager` (pinned by a test to ~1e-6 relative).
    public func evaluate(validation: RinGoShard) throws -> ValidationMetrics {
        guard validation.nnLen > 0, validation.sampleCount > 0 else {
            throw RinGoShardError("Validation dataset is empty")
        }
        let batchSize = max(1, configuration.batchSize)
        let total = validation.sampleCount
        let fullBatchCount = total / batchSize
        let evaluateFn = compiledEvaluate ?? {
            let built = makeCompiledEvaluate()
            compiledEvaluate = built
            return built
        }()

        // On-device weighted-sum accumulators. `eval` after each microbatch forces that microbatch's
        // work and bounds the lazy graph, but copies nothing to host; the single host read is the
        // `.item` calls at the very end.
        var policyAccum = MLXArray(Float(0))
        var valueAccum = MLXArray(Float(0))
        var scoreAccum = MLXArray(Float(0))
        var ownershipAccum = MLXArray(Float(0))
        var totalAccum = MLXArray(Float(0))
        var totalSamples = 0
        func accumulate(_ losses: [MLXArray], count: Int) {
            let weight = Float(count)
            policyAccum += losses[0] * weight
            valueAccum += losses[1] * weight
            scoreAccum += losses[2] * weight
            ownershipAccum += losses[3] * weight
            totalAccum += losses[4] * weight
            eval(policyAccum, valueAccum, scoreAccum, ownershipAccum, totalAccum)
            totalSamples += count
        }

        for microbatch in 0 ..< fullBatchCount {
            let start = microbatch * batchSize
            let batch = validation.batch(indices: Array(start ..< start + batchSize))
            accumulate(evaluateFn(Self.arrays(from: batch)), count: batchSize)
        }
        let remainderStart = fullBatchCount * batchSize
        if remainderStart < total {
            let batch = validation.batch(indices: Array(remainderStart ..< total))
            let losses = lossComputer(
                outputs: network.forward(spatial: batch.spatial, global: batch.global),
                batch: batch
            )
            accumulate(
                [losses.policy, losses.value, losses.score, losses.ownership, losses.total],
                count: total - remainderStart
            )
        }

        let n = Float(max(1, totalSamples))
        return ValidationMetrics(
            step: step,
            policyLoss: policyAccum.item(Float.self) / n,
            valueLoss: valueAccum.item(Float.self) / n,
            scoreLoss: scoreAccum.item(Float.self) / n,
            ownershipLoss: ownershipAccum.item(Float.self) / n,
            totalLoss: totalAccum.item(Float.self) / n,
            sampleCount: totalSamples
        )
    }

    /// Eager (uncompiled) reference implementation of `evaluate`, retained as the numerical oracle
    /// the compiled path is pinned against (`testCompiledValidationMatchesEager`) and as a safe
    /// fallback. Same weighted-mean / masking semantics; differs only in that the forward is eager
    /// and the per-microbatch weighted sums accumulate on the host.
    func evaluateEager(validation: RinGoShard) throws -> ValidationMetrics {
        guard validation.nnLen > 0, validation.sampleCount > 0 else {
            throw RinGoShardError("Validation dataset is empty")
        }
        let batchSize = max(1, configuration.batchSize)
        var totalSamples = 0
        var policySum: Float = 0
        var valueSum: Float = 0
        var scoreSum: Float = 0
        var ownershipSum: Float = 0
        var totalSum: Float = 0
        for start in stride(from: 0, to: validation.sampleCount, by: batchSize) {
            let end = min(start + batchSize, validation.sampleCount)
            let batch = validation.batch(indices: Array(start ..< end))
            let outputs = network.forward(spatial: batch.spatial, global: batch.global)
            let losses = lossComputer(outputs: outputs, batch: batch)
            eval(losses.policy, losses.value, losses.score, losses.ownership, losses.total)
            let count = Float(end - start)
            totalSamples += (end - start)
            policySum += losses.policy.item(Float.self) * count
            valueSum += losses.value.item(Float.self) * count
            scoreSum += losses.score.item(Float.self) * count
            ownershipSum += losses.ownership.item(Float.self) * count
            totalSum += losses.total.item(Float.self) * count
        }
        let n = Float(max(1, totalSamples))
        return ValidationMetrics(
            step: step,
            policyLoss: policySum / n,
            valueLoss: valueSum / n,
            scoreLoss: scoreSum / n,
            ownershipLoss: ownershipSum / n,
            totalLoss: totalSum / n,
            sampleCount: totalSamples
        )
    }

    /// Builds the compiled validation forward + loss. `network.store` is declared as a compile INPUT
    /// (not baked as a graph constant) so the CURRENT parameters flow in on every call -- validation
    /// runs mid-training after the params have moved, and a plain `compile()` would replay whatever
    /// params were captured at first trace. `outputs` is empty: validation mutates no state. This
    /// mirrors `makeCompiledStep`'s inputs-as-state discipline.
    private func makeCompiledEvaluate() -> ([MLXArray]) -> [MLXArray] {
        compile(inputs: [network.store], outputs: []) { [network, lossComputer] arrays in
            let batch = TrainingBatch(
                spatial: arrays[0], global: arrays[1], policyTarget: arrays[2],
                valueTarget: arrays[3], scoreTarget: arrays[4], scoreValidTarget: arrays[5],
                ownershipTarget: arrays[6], ownershipValidTarget: arrays[7]
            )
            let losses = lossComputer(outputs: network.forward(spatial: arrays[0], global: arrays[1]), batch: batch)
            return [losses.policy, losses.value, losses.score, losses.ownership, losses.total]
        }
    }

    /// The fixed-order `[MLXArray]` argument vector the compiled train/eval functions expect.
    static func arrays(from batch: TrainingBatch) -> [MLXArray] {
        [
            batch.spatial, batch.global, batch.policyTarget, batch.valueTarget,
            batch.scoreTarget, batch.scoreValidTarget, batch.ownershipTarget, batch.ownershipValidTarget,
        ]
    }

    /// Writes `checkpoint.safetensors` ATOMICALLY: a crash or kill mid-write to the
    /// final path would otherwise destroy an hours-long run's only resume point, since the old
    /// behaviour overwrote it in place. This saves to a temp file in the SAME directory (so the
    /// swap below is a same-volume rename, not a cross-volume copy) and only replaces the real
    /// path once that write has fully succeeded; a crash during the save leaves the previous good
    /// checkpoint untouched and, at worst, an orphaned temp file that the `defer` below cleans up.
    public func saveCheckpoint(to url: URL) throws {
        var arrays = network.parameters.reduce(into: [String: MLXArray]()) {
            $0["parameter.\($1.key)"] = $1.value
        }
        let states = optimizer.innerState()
        for (index, state) in states.enumerated() {
            arrays[String(format: "optimizer.%08d", index)] = state
        }
        // `MLX.save` dispatches on `url.pathExtension`, so the temp file must also end in
        // `.safetensors` -- hence the extension goes last, not on a `.tmp` suffix.
        let tempURL = url.deletingLastPathComponent().appendingPathComponent(
            "\(url.deletingPathExtension().lastPathComponent).\(UUID().uuidString).tmp.safetensors"
        )
        try MLX.save(
            arrays: arrays,
            metadata: ["step": String(step), "optimizerStateCount": String(states.count)],
            url: tempURL
        )
        var tempStillPresent = true
        defer { if tempStillPresent { try? FileManager.default.removeItem(at: tempURL) } }
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tempURL)
        } else {
            try FileManager.default.moveItem(at: tempURL, to: url)
        }
        tempStillPresent = false
    }

    public func loadCheckpoint(from url: URL) throws {
        // Load on the CPU stream (MLX's own default for this call): the safetensors `Load`
        // primitive has no GPU implementation, so forcing `.default` (GPU) crashes with
        // "[Load::eval_gpu] Not implemented" when the loaded arrays are first evaluated.
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url, stream: .cpu)
        guard let stepText = metadata["step"], let restoredStep = Int(stepText), restoredStep >= 0 else {
            throw TrainableNetworkError("Checkpoint is missing a valid step")
        }
        let parameterPrefix = "parameter."
        let parameters = arrays.reduce(into: [String: MLXArray]()) { result, entry in
            guard entry.key.hasPrefix(parameterPrefix) else { return }
            result[String(entry.key.dropFirst(parameterPrefix.count))] = entry.value
        }
        try network.replaceParameters(parameters)

        // `RinGoAdamW` exposes no direct state setter -- `primeState` materializes zeroed (m, v)
        // state for every trainable parameter (a no-op for keys that already have state), giving
        // the state tree its final shape so the loop below can restore each array in place via
        // `MLXArray._updateInternal`.
        optimizer.primeState(for: network.store.trainableParameters())
        let states = optimizer.innerState()
        guard let countText = metadata["optimizerStateCount"], let expectedCount = Int(countText),
              expectedCount == states.count
        else {
            // Deliberately a hard failure, not a partial/silent load: a checkpoint saved before
            // WP-4a's RinGoAdamW (learningRate/step folded into innerState()) has a DIFFERENT
            // state array count than this build expects, and there is no correct way to map old
            // state onto the new layout. Re-start training fresh from this checkpoint's weights
            // (via a from-scratch optimizer) rather than resuming it directly.
            throw TrainableNetworkError(
                "Checkpoint optimizer state count does not match RinGoAdamW's state layout -- "
                    + "this checkpoint predates WP-4a's learning-rate fix and cannot be resumed; "
                    + "start a fresh run (optionally warm-starting the network weights only)"
            )
        }
        for (index, state) in states.enumerated() {
            let key = String(format: "optimizer.%08d", index)
            guard let restored = arrays[key] else {
                throw TrainableNetworkError("Checkpoint is missing optimizer state \(key)")
            }
            state._updateInternal(restored)
        }
        eval(Array(network.parameters.values) + states)
        step = restoredStep
        // The cached compiled step traced the optimizer's previous (freshly-created) state tree.
        // Loading a checkpoint reshapes/restores that state, so drop the cache and let the next
        // step re-trace against the restored state. The validation function traces the param state
        // too, so invalidate it as well.
        compiledStep = nil
        compiledEvaluate = nil
    }

    public func scheduledLearningRate(at step: Int) -> Float {
        if configuration.warmupSteps > 0, step < configuration.warmupSteps {
            return configuration.learningRate * Float(step + 1) / Float(configuration.warmupSteps)
        }
        let decaySteps = max(1, configuration.totalSteps - configuration.warmupSteps)
        let progress = min(1, max(0, Float(step - configuration.warmupSteps) / Float(decaySteps)))
        return configuration.learningRate * 0.5 * (1 + Foundation.cos(Float.pi * progress))
    }

    public static func batch(samples: [RinGoSample], nnLen: Int) -> TrainingBatch {
        let area = nnLen * nnLen
        let spatial = samples.flatMap(\.spatial)
        let global = samples.flatMap(\.global)
        let policy = samples.flatMap(\.policyTarget)
        let value = samples.flatMap(\.valueTarget)
        let score = samples.map(\.scoreTarget)
        let scoreValid = samples.map { $0.scoreValid ? Float(1) : Float(0) }
        let ownership = samples.flatMap { $0.ownershipTarget.map(Float.init) }
        let ownershipValid = samples.map { $0.ownershipValid ? Float(1) : Float(0) }
        return TrainingBatch(
            spatial: MLXArray(spatial, [samples.count, nnLen, nnLen, 22]),
            global: MLXArray(global, [samples.count, 19]),
            policyTarget: MLXArray(policy, [samples.count, area + 1]),
            valueTarget: MLXArray(value, [samples.count, 3]),
            scoreTarget: MLXArray(score),
            scoreValidTarget: MLXArray(scoreValid),
            ownershipTarget: MLXArray(ownership, [samples.count, nnLen, nnLen, 1]),
            ownershipValidTarget: MLXArray(ownershipValid)
        )
    }

    private func prepareMetricsFile(_ url: URL) throws {
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        try Data("step,learning_rate,policy_loss,value_loss,score_loss,ownership_loss,total_loss\n".utf8)
            .write(to: url)
    }

    private func append(_ metrics: TrainingMetrics, to url: URL) throws {
        let line = "\(metrics.step),\(metrics.learningRate),\(metrics.policyLoss),\(metrics.valueLoss),"
            + "\(metrics.scoreLoss),\(metrics.ownershipLoss),\(metrics.totalLoss)\n"
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(line.utf8))
    }

    private func prepareValidationMetricsFile(_ url: URL) throws {
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        try Data(
            "step,policy_loss,value_loss,score_loss,ownership_loss,total_loss,sample_count\n".utf8
        ).write(to: url)
    }

    private func appendValidation(_ metrics: ValidationMetrics, to url: URL) throws {
        let line = "\(metrics.step),\(metrics.policyLoss),\(metrics.valueLoss),"
            + "\(metrics.scoreLoss),\(metrics.ownershipLoss),\(metrics.totalLoss),\(metrics.sampleCount)\n"
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(line.utf8))
    }
}

/// Append-only human-readable progress log written next to `metrics.csv` in the run directory.
/// Format: `[ISO8601] [LEVEL] message`. Kept tiny and dependency-free — all run management code
/// (start time, checkpoint saved, snapshot saved, validation result, completion) writes through
/// here so a long unattended run leaves a single readable timeline.
struct TrainingLogger {
    private let url: URL
    private var handle: FileHandle?

    init(url: URL) {
        self.url = url
    }

    mutating func open() throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            try Data().write(to: url, options: .atomic)
        }
        handle = try FileHandle(forWritingTo: url)
        try handle?.seekToEnd()
    }

    mutating func close() throws {
        try handle?.close()
        handle = nil
    }

    mutating func log(_ level: String, _ message: String) throws {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "[\(timestamp)] [\(level)] \(message)\n"
        try handle?.seekToEnd()
        try handle?.write(contentsOf: Data(line.utf8))
    }
}
