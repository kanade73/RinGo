import Foundation

public struct ModelAverageError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// Stochastic Weight Averaging (SWA, BREAKTHROUGH_IDEAS E4) over RinGo training snapshots.
///
/// `average` takes N same-architecture `ModelDesc`s (as produced by `TrainableNetwork.exportDesc`
/// and written by `ModelWriter`) and returns a new descriptor whose every weight tensor is the
/// arithmetic mean of the corresponding input tensors. The result is an ordinary `ModelDesc`; the
/// caller serializes it with `ModelWriter.write` so it canonicalizes and loads exactly like any
/// other `.bin.gz`.
///
/// ## Why averaging in the parsed (canonical) space is correct
/// `ModelDesc.parse` applies `transformToReduceActivations`, so every input is in the same
/// canonical BatchNorm form: `mean == 0`, `variance == 1`, `scale == mergedScale`,
/// `bias == mergedBias`, `epsilon == 1e-20`. Element-wise averaging is linear, so it preserves
/// each of those invariants (e.g. averaged `scale` still equals averaged `mergedScale` because
/// the two were bit-identical in every input). The averaged descriptor is therefore itself a
/// valid canonical descriptor, and `ModelWriter.write` re-canonicalizes it on the normal path.
///
/// Scope matches what RinGo actually trains: modelVersion-8-style standard convnet trunks with
/// only ordinary and global-pooling residual blocks. RMSNorm trunks, transformer/nested-bottleneck
/// blocks, and SGF metadata encoders are refused up front (they can never be produced by the
/// training loop, and the writer cannot emit metadata encoders anyway).
public enum ModelAverager {
    /// Returns the element-wise arithmetic mean of every weight tensor across `models`.
    ///
    /// All inputs must be the same architecture; any per-tensor shape/name difference or any
    /// unsupported architecture is refused with an error naming the offending tensor or model.
    /// The mean is accumulated in fp32 (the parsed storage precision) in a fixed model order, so
    /// the result is deterministic; averaging a model with itself reproduces it bit-for-bit.
    public static func average(_ models: [ModelDesc]) throws -> ModelDesc {
        guard let template = models.first else {
            throw ModelAverageError("averagemodels: no input models were supplied")
        }
        for (index, model) in models.enumerated() {
            try validate(model, index: index)
        }
        for index in models.indices.dropFirst() {
            try requireSameArchitecture(template, models[index], index: index)
        }

        let lists = models.map { tensorList($0) }
        let reference = lists[0]
        for (index, list) in lists.enumerated().dropFirst() {
            guard list.count == reference.count else {
                throw ModelAverageError(
                    "averagemodels: model \(index) has \(list.count) weight tensors, "
                        + "but model 0 has \(reference.count) (architecture mismatch)"
                )
            }
            for position in reference.indices {
                guard list[position].name == reference[position].name else {
                    throw ModelAverageError(
                        "averagemodels: tensor #\(position) is '\(list[position].name)' in model \(index) "
                            + "but '\(reference[position].name)' in model 0 (architecture mismatch)"
                    )
                }
                guard list[position].values.count == reference[position].values.count else {
                    throw ModelAverageError(
                        "averagemodels: tensor '\(reference[position].name)' has "
                            + "\(list[position].values.count) floats in model \(index) but "
                            + "\(reference[position].values.count) in model 0 (shape mismatch)"
                    )
                }
            }
        }

        let divisor = Float(models.count)
        var averaged = [[Float]]()
        averaged.reserveCapacity(reference.count)
        for position in reference.indices {
            let length = reference[position].values.count
            var accumulator = [Float](repeating: 0, count: length)
            for list in lists {
                let values = list[position].values
                for element in 0 ..< length {
                    accumulator[element] += values[element]
                }
            }
            for element in 0 ..< length {
                accumulator[element] /= divisor
            }
            averaged.append(accumulator)
        }

        var result = template
        result.sha256 = ""
        applyTensorList(&result, averaged)
        return result
    }

    // MARK: - Architecture gating

    private static func validate(_ desc: ModelDesc, index: Int) throws {
        func reject(_ reason: String) -> ModelAverageError {
            ModelAverageError("averagemodels: model \(index) (\(desc.name)) \(reason)")
        }
        guard desc.metaEncoderVersion == 0, desc.trunk.metaEncoderVersion == 0 else {
            throw reject("has an SGF metadata encoder, which averagemodels does not support")
        }
        guard desc.trunk.trunkNormKind == .standard, desc.trunk.trunkTipRMSNorm == nil,
              desc.trunk.trunkTipBN != nil
        else {
            throw reject("does not use a standard BatchNorm trunk, which averagemodels does not support")
        }
        guard !desc.hasAnyTransformerBlocks else {
            throw reject("has transformer blocks, which averagemodels does not support")
        }
        guard !desc.trunk.blocks.contains(where: { $0.kind == .nestedBottleneck }) else {
            throw reject("has nested-bottleneck blocks, which averagemodels does not support")
        }
    }

    private static func requireSameArchitecture(_ template: ModelDesc, _ other: ModelDesc, index: Int) throws {
        func reject(_ reason: String) -> ModelAverageError {
            ModelAverageError("averagemodels: model \(index) \(reason) (architecture mismatch with model 0)")
        }
        guard template.modelVersion == other.modelVersion else {
            throw reject("is modelVersion \(other.modelVersion), not \(template.modelVersion)")
        }
        guard template.numInputChannels == other.numInputChannels,
              template.numInputGlobalChannels == other.numInputGlobalChannels
        else {
            throw reject("has different input channel counts")
        }
        guard template.trunk.numBlocks == other.trunk.numBlocks else {
            throw reject("has \(other.trunk.numBlocks) trunk blocks, not \(template.trunk.numBlocks)")
        }
        guard template.trunk.trunkNumChannels == other.trunk.trunkNumChannels else {
            throw reject("has \(other.trunk.trunkNumChannels) trunk channels, not \(template.trunk.trunkNumChannels)")
        }
        let templateKinds = template.trunk.blocks.map(\.kind)
        let otherKinds = other.trunk.blocks.map(\.kind)
        guard templateKinds == otherKinds else {
            throw reject("has a different sequence of block kinds")
        }
    }

    // MARK: - Tensor traversal

    //
    // `tensorList` (read) and `applyTensorList` (write) both run the SAME `walk` traversal, so the
    // i-th tensor collected corresponds to the i-th tensor overwritten. Keep them defined only in
    // terms of `walk` so the two orders can never drift apart.

    /// Every weight tensor of `desc` in traversal order, each tagged with a stable structural name
    /// (used for the cross-model shape/architecture checks and for byte-comparison in tests).
    static func tensorList(_ desc: ModelDesc) -> [(name: String, values: [Float])] {
        var copy = desc
        var collected = [(name: String, values: [Float])]()
        walk(&copy) { name, values in collected.append((name, values)) }
        return collected
    }

    private static func applyTensorList(_ desc: inout ModelDesc, _ tensors: [[Float]]) {
        var index = 0
        walk(&desc) { _, values in
            values = tensors[index]
            index += 1
        }
    }

    private static func walk(_ desc: inout ModelDesc, _ visit: (String, inout [Float]) -> Void) {
        walkTrunk(&desc.trunk, visit)
        walkPolicy(&desc.policyHead, visit)
        walkValue(&desc.valueHead, visit)
    }

    private static func walkTrunk(_ trunk: inout TrunkDesc, _ visit: (String, inout [Float]) -> Void) {
        visitConv(&trunk.initialConv, visit)
        visitMatMul(&trunk.initialMatMul, visit)
        for index in trunk.blocks.indices {
            walkBlock(&trunk.blocks[index], visit)
        }
        if var tipBN = trunk.trunkTipBN {
            visitBatchNorm(&tipBN, visit)
            trunk.trunkTipBN = tipBN
        }
    }

    private static func walkBlock(_ block: inout BlockDesc, _ visit: (String, inout [Float]) -> Void) {
        switch block {
        case var .ordinary(residual):
            visitBatchNorm(&residual.preBN, visit)
            visitConv(&residual.regularConv, visit)
            visitBatchNorm(&residual.midBN, visit)
            visitConv(&residual.finalConv, visit)
            block = .ordinary(residual)
        case var .globalPooling(pooling):
            visitBatchNorm(&pooling.preBN, visit)
            visitConv(&pooling.regularConv, visit)
            visitConv(&pooling.gpoolConv, visit)
            visitBatchNorm(&pooling.gpoolBN, visit)
            visitMatMul(&pooling.gpoolToBiasMul, visit)
            visitBatchNorm(&pooling.midBN, visit)
            visitConv(&pooling.finalConv, visit)
            block = .globalPooling(pooling)
        case .nestedBottleneck, .transformerAttention, .transformerFFN:
            // Refused by `validate`; kept exhaustive so a future block kind fails to compile here.
            break
        }
    }

    private static func walkPolicy(_ head: inout PolicyHeadDesc, _ visit: (String, inout [Float]) -> Void) {
        visitConv(&head.p1Conv, visit)
        visitConv(&head.g1Conv, visit)
        visitBatchNorm(&head.g1BN, visit)
        visitMatMul(&head.gpoolToBiasMul, visit)
        visitBatchNorm(&head.p1BN, visit)
        visitConv(&head.p2Conv, visit)
        visitMatMul(&head.gpoolToPassMul, visit)
        if var passBias = head.gpoolToPassBias {
            visitMatBias(&passBias, visit)
            head.gpoolToPassBias = passBias
        }
        if var passMul2 = head.gpoolToPassMul2 {
            visitMatMul(&passMul2, visit)
            head.gpoolToPassMul2 = passMul2
        }
    }

    private static func walkValue(_ head: inout ValueHeadDesc, _ visit: (String, inout [Float]) -> Void) {
        visitConv(&head.v1Conv, visit)
        visitBatchNorm(&head.v1BN, visit)
        visitMatMul(&head.v2Mul, visit)
        visitMatBias(&head.v2Bias, visit)
        visitMatMul(&head.v3Mul, visit)
        visitMatBias(&head.v3Bias, visit)
        visitMatMul(&head.sv3Mul, visit)
        visitMatBias(&head.sv3Bias, visit)
        visitConv(&head.vOwnershipConv, visit)
    }

    private static func visitConv(_ layer: inout ConvLayerDesc, _ visit: (String, inout [Float]) -> Void) {
        visit("conv:\(layer.name)", &layer.weights)
    }

    private static func visitMatMul(_ layer: inout MatMulLayerDesc, _ visit: (String, inout [Float]) -> Void) {
        visit("matmul:\(layer.name)", &layer.weights)
    }

    private static func visitMatBias(_ layer: inout MatBiasLayerDesc, _ visit: (String, inout [Float]) -> Void) {
        visit("bias:\(layer.name)", &layer.weights)
    }

    /// Averages all six BatchNorm arrays independently. For canonical (parsed) descriptors this is
    /// exactly the average of each layer's affine transform: `mean`/`variance` stay 0/1, and
    /// `scale`/`bias` stay equal to `mergedScale`/`mergedBias` because averaging is linear.
    private static func visitBatchNorm(_ layer: inout BatchNormLayerDesc, _ visit: (String, inout [Float]) -> Void) {
        visit("bn:\(layer.name):mean", &layer.mean)
        visit("bn:\(layer.name):variance", &layer.variance)
        visit("bn:\(layer.name):scale", &layer.scale)
        visit("bn:\(layer.name):bias", &layer.bias)
        visit("bn:\(layer.name):mergedScale", &layer.mergedScale)
        visit("bn:\(layer.name):mergedBias", &layer.mergedBias)
    }
}
