import MLX
import MLXNN
import RinGoModel

public struct LossWeights: Sendable {
    public var policy: Float
    public var value: Float
    public var score: Float
    public var ownership: Float

    public init(policy: Float = 1, value: Float = 1.5, score: Float = 0.02, ownership: Float = 0.06) {
        self.policy = policy
        self.value = value
        self.score = score
        self.ownership = ownership
    }
}

public struct TrainingBatch {
    public let spatial: MLXArray
    public let global: MLXArray
    /// Dense policy distribution, shape `[batch, nnLen*nnLen+1]` -- same rank as the policy logits,
    /// so `MLXNN.crossEntropy` treats it as a probability distribution rather than a class index.
    public let policyTarget: MLXArray
    public let valueTarget: MLXArray
    public let scoreTarget: MLXArray
    /// Shape `[batch]`, 1 for samples whose `scoreTarget` is a genuine label and 0 for samples
    /// whose score must be excluded from the score loss (e.g. the resignation proxy).
    public let scoreValidTarget: MLXArray
    public let ownershipTarget: MLXArray
    /// Shape `[batch]`, 1 for samples whose `ownershipTarget` is a genuine label and 0 for samples
    /// whose ownership must be excluded from the ownership loss (e.g. a resignation).
    public let ownershipValidTarget: MLXArray

    public init(
        spatial: MLXArray, global: MLXArray, policyTarget: MLXArray,
        valueTarget: MLXArray, scoreTarget: MLXArray, scoreValidTarget: MLXArray,
        ownershipTarget: MLXArray, ownershipValidTarget: MLXArray
    ) {
        self.spatial = spatial
        self.global = global
        self.policyTarget = policyTarget
        self.valueTarget = valueTarget
        self.scoreTarget = scoreTarget
        self.scoreValidTarget = scoreValidTarget
        self.ownershipTarget = ownershipTarget
        self.ownershipValidTarget = ownershipValidTarget
    }
}

public struct Losses {
    public let policy: MLXArray
    public let value: MLXArray
    public let score: MLXArray
    public let ownership: MLXArray
    public let total: MLXArray
}

public struct LossComputer: Sendable {
    public var weights: LossWeights

    public init(weights: LossWeights = LossWeights()) {
        self.weights = weights
    }

    public func callAsFunction(outputs: ModelRawOutputs, batch: TrainingBatch) -> Losses {
        let batchSize = batch.spatial.dim(0)
        let policySpatial = outputs.policySpatialLogits.reshaped(batchSize, -1)
        let policyLogits = concatenated([policySpatial, outputs.policyPassLogits.reshaped(batchSize, -1)], axis: 1)
        // batch.policyTarget is dense with the SAME shape as policyLogits (targets.ndim ==
        // logits.ndim), so MLXNN's crossEntropy takes the `sum(logits * targets, axis) -
        // logSumExp(logits, axis)` probability-distribution branch instead of its sparse
        // class-index branch. When the distribution happens to be one-hot (the teacherless
        // makedata case) this is numerically identical to the old index-based cross entropy --
        // see PolicyLossTests.testDensePolicyOneHotMatchesOldIndexBasedCrossEntropy.
        let policy = crossEntropy(logits: policyLogits, targets: batch.policyTarget, reduction: .mean)
        // batch.valueTarget is likewise dense and may be a soft distribution (e.g. a teacher's
        // win-probability estimate) rather than a one-hot outcome; crossEntropy handles both
        // identically since it only depends on `targets.ndim == logits.ndim`.
        let value = crossEntropy(logits: outputs.valueLogits, targets: batch.valueTarget, reduction: .mean)

        // ModelPostProcessParams.scoreMeanMultiplier defaults to 20 and the v8 postprocessor
        // computes scoreMean = raw scoreValues[0] * that multiplier.
        let scorePrediction = outputs.scoreValues[.ellipsis, 0]
        // Per-sample loss (reduction: .none) so invalid samples (scoreValidTarget == 0, e.g. the
        // resignation proxy -- TRAINING_PROCESS_AUDIT.md P0-3) can be zeroed out of both the loss
        // AND its normalization, rather than diluted into a fixed-batchSize mean.
        let scorePointLoss = huberLoss(inputs: scorePrediction, targets: batch.scoreTarget / 20,
                                       delta: 10, reduction: .none)
        let score = (scorePointLoss * batch.scoreValidTarget).sum() / maximum(batch.scoreValidTarget.sum(), 1)

        // KataGo maps the raw ownership output through tanh into [-1, 1]. Because
        // sigmoid(2x) == (tanh(x) + 1) / 2, feeding `2 * rawOwnership` into the fused
        // logits-mode binary cross entropy is numerically identical to the old
        // tanh -> affine -> withLogits:false path, but avoids the extra tanh/log/clip
        // round-trip and uses MLXNN's single logAddExp formulation (Losses.swift ~107).
        let ownershipTarget = (batch.ownershipTarget + 1) / 2
        let ownershipPointLoss = binaryCrossEntropy(
            logits: outputs.ownership * 2,
            targets: ownershipTarget,
            withLogits: true,
            reduction: .none
        )
        let onBoardMask = batch.spatial[.ellipsis, 0 ..< 1]
        // Broadcast the per-sample validity flag over the spatial dims so a resignation's whole
        // board (not just some points) drops out of both the ownership loss and its normalization.
        let validityMask = batch.ownershipValidTarget.reshaped(batchSize, 1, 1, 1)
        let mask = onBoardMask * validityMask
        let ownership = (ownershipPointLoss * mask).sum() / maximum(mask.sum(), 1)
        let total = weights.policy * policy + weights.value * value + weights.score * score
            + weights.ownership * ownership
        return Losses(policy: policy, value: value, score: score, ownership: ownership, total: total)
    }
}
