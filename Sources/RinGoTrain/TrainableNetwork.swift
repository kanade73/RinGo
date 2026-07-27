import Foundation
import MLX
import MLXNN
@_spi(Training) import RinGoModel

public struct TrainableNetworkError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

final class FlatParameterStore: Module {
    @ParameterInfo(key: "values") var values: [String: MLXArray]

    init(_ values: [String: MLXArray]) {
        _values = ParameterInfo(wrappedValue: values, key: "values")
        super.init()
    }
}

public final class TrainableNetwork {
    private var desc: ModelDesc
    private let graph: KataGoNetwork
    let store: FlatParameterStore

    public var parameters: [String: MLXArray] {
        store.values
    }

    public init(desc: ModelDesc, nnXLen: Int, nnYLen: Int) throws {
        try Self.validate(desc)
        self.desc = desc
        var tensors = [String: MLXArray]()
        Self.initialize(trunk: desc.trunk, into: &tensors)
        Self.initialize(policy: desc.policyHead, into: &tensors)
        Self.initialize(value: desc.valueHead, into: &tensors)
        store = FlatParameterStore(tensors)
        graph = try KataGoNetwork(desc: desc, nnXLen: nnXLen, nnYLen: nnYLen, precision: .float32)
        eval(Array(tensors.values))
    }

    public func forward(spatial: MLXArray, global: MLXArray) -> ModelRawOutputs {
        let values = store.values
        return graph.forward(spatial: spatial, global: global) { key in
            guard let value = values[key] else { preconditionFailure("Missing trainable parameter \(key)") }
            return value
        }
    }

    public func exportDesc() -> ModelDesc {
        // Materialize every parameter (in its exported layout) with a SINGLE batched `eval`, then
        // read each back from host memory. The previous implementation eval'd one tensor per layer
        // -- 40-60 separate GPU sync round-trips per snapshot. This fires only at snapshot/save
        // time, never in the hot loop, but the batched eval is strictly cheaper.
        let floats = exportedParameterFloats()
        var result = desc
        result.sha256 = ""
        result.trunk.initialConv = exported(result.trunk.initialConv, floats)
        result.trunk.initialMatMul = exported(result.trunk.initialMatMul, floats)
        result.trunk.blocks = exported(blocks: result.trunk.blocks, floats)
        if let norm = result.trunk.trunkTipBN { result.trunk.trunkTipBN = exported(norm, floats) }

        result.policyHead.p1Conv = exported(result.policyHead.p1Conv, floats)
        result.policyHead.g1Conv = exported(result.policyHead.g1Conv, floats)
        result.policyHead.g1BN = exported(result.policyHead.g1BN, floats)
        result.policyHead.gpoolToBiasMul = exported(result.policyHead.gpoolToBiasMul, floats)
        result.policyHead.p1BN = exported(result.policyHead.p1BN, floats)
        result.policyHead.p2Conv = exported(result.policyHead.p2Conv, floats)
        result.policyHead.gpoolToPassMul = exported(result.policyHead.gpoolToPassMul, floats)

        result.valueHead.v1Conv = exported(result.valueHead.v1Conv, floats)
        result.valueHead.v1BN = exported(result.valueHead.v1BN, floats)
        result.valueHead.v2Mul = exported(result.valueHead.v2Mul, floats)
        result.valueHead.v2Bias = exported(result.valueHead.v2Bias, floats)
        result.valueHead.v3Mul = exported(result.valueHead.v3Mul, floats)
        result.valueHead.v3Bias = exported(result.valueHead.v3Bias, floats)
        result.valueHead.sv3Mul = exported(result.valueHead.sv3Mul, floats)
        result.valueHead.sv3Bias = exported(result.valueHead.sv3Bias, floats)
        result.valueHead.vOwnershipConv = exported(result.valueHead.vOwnershipConv, floats)
        return result
    }

    /// Every store parameter, materialized in its exported layout with one batched `eval`, keyed
    /// by store key. Conv weights are transposed MLX `[outC, kH, kW, inC]` -> desc
    /// `[outC, inC, kH, kW]`; all others are just cast to float32.
    private func exportedParameterFloats() -> [String: [Float]] {
        let keys = Array(store.values.keys)
        let tensors: [MLXArray] = keys.map { key in
            let value = store.values[key]!
            if key.hasPrefix("conv:") {
                return value.transposed(0, 3, 1, 2).asType(.float32)
            }
            return value.asType(.float32)
        }
        eval(tensors)
        var result = [String: [Float]](minimumCapacity: keys.count)
        for (key, tensor) in zip(keys, tensors) {
            result[key] = tensor.asArray(Float.self)
        }
        return result
    }

    func replaceParameters(_ values: [String: MLXArray]) throws {
        // Resume safety (WP-5): a checkpoint trained under a different `-arch` must never be
        // silently misloaded. The key SET already differs across the b6/b10/b15 variants (they
        // have different block counts), and the per-key shape check below additionally rejects any
        // same-key/different-width mismatch — so a wrong architecture fails loudly here even when
        // config.json's recorded arch is absent (pre-WP-5 checkpoint) or was hand-edited.
        let expected = Set(store.values.keys)
        let provided = Set(values.keys)
        guard expected == provided else {
            let missing = expected.subtracting(provided).sorted().prefix(4)
            let unexpected = provided.subtracting(expected).sorted().prefix(4)
            throw TrainableNetworkError(
                "Checkpoint parameter keys do not match the network (architecture mismatch?). "
                    + "Missing \(missing.count) e.g. \(Array(missing)); "
                    + "unexpected \(unexpected.count) e.g. \(Array(unexpected))."
            )
        }
        for (key, value) in values {
            guard let existing = store.values[key] else { continue }
            guard existing.shape == value.shape else {
                throw TrainableNetworkError(
                    "Checkpoint tensor \(key) has shape \(value.shape) but this network expects "
                        + "\(existing.shape) — the checkpoint was trained with a different -arch."
                )
            }
        }
        for (key, value) in values {
            store.values[key]?._updateInternal(value)
        }
    }

    /// WP-8: warm-start this network's parameters from a donor `ModelDesc` parsed out of an
    /// exported `.bin.gz` (the exact inverse of `exportDesc`). Used by `train -init-model` for
    /// low-LR polish fine-tuning and Step-B RL (continue-training the current champion).
    ///
    /// The donor must describe the SAME architecture this network was built for: the shared store
    /// key set (layer names + block structure) and per-key shapes must match. Both invariants are
    /// enforced by reusing `replaceParameters` (WP-5's resume guard) verbatim, so a wrong `-arch`
    /// for the file — or a structurally different donor — fails loudly instead of misloading.
    ///
    /// ## BatchNorm canonicalization (the exact inverse of `exportDesc`)
    /// The forward graph consumes only a BN layer's fully-folded affine transform: `KataGoNetwork`
    /// registers `bn-scale`/`bn-bias` from `mergedScale`/`mergedBias`, NOT from the raw
    /// `scale`/`bias` (which are 1/0 for a `hasScale==false`/`hasBias==false` layer). The parser
    /// always populates `mergedScale`/`mergedBias` regardless of the `hasScale`/`hasBias` flags
    /// (`mergedScale = scale / sqrt(variance + epsilon)`, `mergedBias = bias - mergedScale*mean`),
    /// so reading those two fields is correct for every donor — including one whose BNs are stored
    /// scale-free — and is exactly what `exportDesc` writes back out. Reading `scale`/`bias` here
    /// instead would silently drop the folded normalization of any scale-free BN.
    public func loadDonorParameters(from donor: ModelDesc) throws {
        try Self.validate(donor)
        var values = [String: MLXArray]()
        Self.load(trunk: donor.trunk, into: &values)
        Self.load(policy: donor.policyHead, into: &values)
        Self.load(value: donor.valueHead, into: &values)
        eval(Array(values.values))
        // Reuse WP-5's key-set + per-key shape checks: throws a clear architecture-mismatch error
        // when the donor's keys/shapes don't match this network.
        try replaceParameters(values)
        eval(Array(store.values.values))
    }

    private static func load(trunk: TrunkDesc, into values: inout [String: MLXArray]) {
        load(trunk.initialConv, into: &values)
        load(trunk.initialMatMul, into: &values)
        for block in trunk.blocks {
            switch block {
            case let .ordinary(block):
                load(block.preBN, into: &values)
                load(block.regularConv, into: &values)
                load(block.midBN, into: &values)
                load(block.finalConv, into: &values)
            case let .globalPooling(block):
                load(block.preBN, into: &values)
                load(block.regularConv, into: &values)
                load(block.gpoolConv, into: &values)
                load(block.gpoolBN, into: &values)
                load(block.gpoolToBiasMul, into: &values)
                load(block.midBN, into: &values)
                load(block.finalConv, into: &values)
            case .nestedBottleneck, .transformerAttention, .transformerFFN:
                preconditionFailure("Unsupported block passed validation")
            }
        }
        load(trunk.trunkTipBN!, into: &values)
    }

    private static func load(policy: PolicyHeadDesc, into values: inout [String: MLXArray]) {
        load(policy.p1Conv, into: &values)
        load(policy.g1Conv, into: &values)
        load(policy.g1BN, into: &values)
        load(policy.gpoolToBiasMul, into: &values)
        load(policy.p1BN, into: &values)
        load(policy.p2Conv, into: &values)
        load(policy.gpoolToPassMul, into: &values)
    }

    private static func load(value: ValueHeadDesc, into values: inout [String: MLXArray]) {
        load(value.v1Conv, into: &values)
        load(value.v1BN, into: &values)
        load(value.v2Mul, into: &values)
        load(value.v2Bias, into: &values)
        load(value.v3Mul, into: &values)
        load(value.v3Bias, into: &values)
        load(value.sv3Mul, into: &values)
        load(value.sv3Bias, into: &values)
        load(value.vOwnershipConv, into: &values)
    }

    /// Conv: desc `weights` are OIHW `[outC, inC, kH, kW]`; the store holds the MLX conv layout
    /// `[outC, kH, kW, inC]`. This transpose mirrors `KataGoNetwork.register(ConvLayerDesc)` and is
    /// the exact inverse of `exportedParameterFloats`' `transposed(0, 3, 1, 2)`.
    private static func load(_ layer: ConvLayerDesc, into values: inout [String: MLXArray]) {
        values[convKey(layer)] = MLXArray(
            layer.weights,
            [layer.outChannels, layer.inChannels, layer.convYSize, layer.convXSize]
        ).transposed(0, 2, 3, 1).asType(.float32)
    }

    private static func load(_ layer: MatMulLayerDesc, into values: inout [String: MLXArray]) {
        values[matmulKey(layer)] = MLXArray(layer.weights, [layer.inChannels, layer.outChannels]).asType(.float32)
    }

    private static func load(_ layer: MatBiasLayerDesc, into values: inout [String: MLXArray]) {
        values[biasKey(layer)] = MLXArray(layer.weights).asType(.float32)
    }

    private static func load(_ layer: BatchNormLayerDesc, into values: inout [String: MLXArray]) {
        values[normScaleKey(layer)] = MLXArray(layer.mergedScale).asType(.float32)
        values[normBiasKey(layer)] = MLXArray(layer.mergedBias).asType(.float32)
    }

    private static func validate(_ desc: ModelDesc) throws {
        guard desc.modelVersion == 8 else {
            throw TrainableNetworkError("RinGoTrain v1 supports only modelVersion 8 (got \(desc.modelVersion))")
        }
        guard desc.trunk.trunkNormKind == .standard else {
            throw TrainableNetworkError("RinGoTrain v1 supports only trunkNormKind standard")
        }
        guard desc.trunk.trunkTipRMSNorm == nil else {
            throw TrainableNetworkError("RinGoTrain v1 does not support RMSNorm")
        }
        guard !desc.hasAnyTransformerBlocks else {
            throw TrainableNetworkError("RinGoTrain v1 does not support transformer blocks")
        }
        guard !desc.trunk.blocks.contains(where: { $0.kind == .nestedBottleneck }) else {
            throw TrainableNetworkError("RinGoTrain v1 does not support nested bottleneck (nbt) blocks")
        }
        guard desc.metaEncoderVersion == 0, desc.trunk.metaEncoderVersion == 0 else {
            throw TrainableNetworkError("RinGoTrain v1 does not support metadata encoders")
        }
    }

    private static func initialize(trunk: TrunkDesc, into values: inout [String: MLXArray]) {
        add(trunk.initialConv, scale: 0.8, activation: .relu, to: &values)
        add(trunk.initialMatMul, scale: 0.6, activation: .relu, to: &values)
        let fixupScale = Float(1 / Foundation.sqrt(Double(trunk.numBlocks)))
        for block in trunk.blocks {
            switch block {
            case let .ordinary(block):
                add(block.preBN, to: &values)
                add(block.regularConv, scale: fixupScale, activation: .relu, to: &values)
                add(block.midBN, to: &values)
                add(block.finalConv, scale: 0, activation: .relu, to: &values)
            case let .globalPooling(block):
                add(block.preBN, to: &values)
                add(block.regularConv, scale: fixupScale * 0.8, activation: .relu, to: &values)
                let pooledScale = Foundation.sqrt(fixupScale) * Foundation.sqrt(Float(0.6))
                add(block.gpoolConv, scale: pooledScale, activation: .relu, to: &values)
                add(block.gpoolBN, to: &values)
                add(block.gpoolToBiasMul, scale: pooledScale, activation: .relu, to: &values)
                add(block.midBN, to: &values)
                add(block.finalConv, scale: 0, activation: .relu, to: &values)
            case .nestedBottleneck, .transformerAttention, .transformerFFN:
                preconditionFailure("Unsupported block passed validation")
            }
        }
        add(trunk.trunkTipBN!, to: &values)
    }

    private static func initialize(policy: PolicyHeadDesc, into values: inout [String: MLXArray]) {
        add(policy.p1Conv, scale: 0.8, activation: .relu, to: &values)
        add(policy.g1Conv, scale: 1, activation: .relu, to: &values)
        add(policy.g1BN, to: &values)
        add(policy.gpoolToBiasMul, scale: 0.6, activation: .relu, to: &values)
        add(policy.p1BN, to: &values)
        add(policy.p2Conv, scale: 0, activation: .identity, to: &values)
        add(policy.gpoolToPassMul, scale: 0, activation: .identity, to: &values)
    }

    private static func initialize(value: ValueHeadDesc, into values: inout [String: MLXArray]) {
        add(value.v1Conv, scale: 1, activation: .relu, to: &values)
        add(value.v1BN, to: &values)
        add(value.v2Mul, scale: 1, activation: .relu, to: &values)
        add(value.v2Bias, to: &values)
        // Value / score / ownership output layers used scale:0 (KataGo-style neutral prior), but on
        // RinGo's small-scale Step A the heads get stuck at ln(2) plateaus. Use small-nonzero init
        // so gradients propagate immediately. Verified by TrainingDiagnosticsTests; scale chosen to
        // give ~0.1-0.2 std on b6c96 output layers without biasing the initial predictions far from
        // zero.
        add(value.v3Mul, scale: 1, activation: .identity, to: &values)
        add(value.v3Bias, to: &values)
        add(value.sv3Mul, scale: 1, activation: .identity, to: &values)
        add(value.sv3Bias, to: &values)
        add(value.vOwnershipConv, scale: 1, activation: .identity, to: &values)
    }

    private static func add(
        _ layer: ConvLayerDesc, scale: Float, activation: ActivationKind,
        to values: inout [String: MLXArray]
    ) {
        values[convKey(layer)] = initialized(
            shape: [layer.outChannels, layer.convYSize, layer.convXSize, layer.inChannels],
            fanIn: layer.inChannels * layer.convYSize * layer.convXSize,
            scale: scale,
            activation: activation
        )
    }

    private static func add(
        _ layer: MatMulLayerDesc, scale: Float, activation: ActivationKind,
        to values: inout [String: MLXArray]
    ) {
        values[matmulKey(layer)] = initialized(shape: [layer.inChannels, layer.outChannels],
                                               fanIn: layer.inChannels, scale: scale,
                                               activation: activation)
    }

    private static func add(_ layer: MatBiasLayerDesc, to values: inout [String: MLXArray]) {
        values[biasKey(layer)] = MLXArray.zeros([layer.numChannels])
    }

    private static func add(_ layer: BatchNormLayerDesc, to values: inout [String: MLXArray]) {
        values[normScaleKey(layer)] = MLXArray.ones([layer.numChannels])
        values[normBiasKey(layer)] = MLXArray.zeros([layer.numChannels])
    }

    /// Mirrors KataGo's `init_weights`: gain/sqrt(fanIn), corrected by 0.87962566 for a
    /// normal truncated at two standard deviations. MLX samples the unit truncated normal,
    /// so scaling it is algebraically the same as PyTorch's bounded call.
    private static func initialized(
        shape: [Int], fanIn: Int, scale: Float,
        activation: ActivationKind
    ) -> MLXArray {
        guard scale > 1e-10 else { return MLXArray.zeros(shape) }
        let gain: Float = activation == .identity ? 1 : Foundation.sqrt(2)
        let standardDeviation = scale * gain / Foundation.sqrt(Float(fanIn)) / 0.87962566103423978
        return MLXRandom.truncatedNormal(Float(-2) ..< Float(2), shape) * standardDeviation
    }

    private func exported(_ layer: ConvLayerDesc, _ floats: [String: [Float]]) -> ConvLayerDesc {
        var layer = layer
        layer.weights = lookup(convKey(layer), floats)
        return layer
    }

    private func exported(_ layer: MatMulLayerDesc, _ floats: [String: [Float]]) -> MatMulLayerDesc {
        var layer = layer
        layer.weights = lookup(matmulKey(layer), floats)
        return layer
    }

    private func exported(_ layer: MatBiasLayerDesc, _ floats: [String: [Float]]) -> MatBiasLayerDesc {
        var layer = layer
        layer.weights = lookup(biasKey(layer), floats)
        return layer
    }

    private func exported(_ layer: BatchNormLayerDesc, _ floats: [String: [Float]]) -> BatchNormLayerDesc {
        var layer = layer
        layer.mergedScale = lookup(normScaleKey(layer), floats)
        layer.mergedBias = lookup(normBiasKey(layer), floats)
        layer.mean = [Float](repeating: 0, count: layer.numChannels)
        layer.variance = [Float](repeating: 1, count: layer.numChannels)
        layer.scale = layer.mergedScale
        layer.bias = layer.mergedBias
        layer.hasScale = true
        layer.hasBias = true
        return layer
    }

    private func exported(blocks: [BlockDesc], _ floats: [String: [Float]]) -> [BlockDesc] {
        blocks.map { block in
            switch block {
            case var .ordinary(value):
                value.preBN = exported(value.preBN, floats)
                value.regularConv = exported(value.regularConv, floats)
                value.midBN = exported(value.midBN, floats)
                value.finalConv = exported(value.finalConv, floats)
                return .ordinary(value)
            case var .globalPooling(value):
                value.preBN = exported(value.preBN, floats)
                value.regularConv = exported(value.regularConv, floats)
                value.gpoolConv = exported(value.gpoolConv, floats)
                value.gpoolBN = exported(value.gpoolBN, floats)
                value.gpoolToBiasMul = exported(value.gpoolToBiasMul, floats)
                value.midBN = exported(value.midBN, floats)
                value.finalConv = exported(value.finalConv, floats)
                return .globalPooling(value)
            case .nestedBottleneck, .transformerAttention, .transformerFFN:
                preconditionFailure("Unsupported block passed validation")
            }
        }
    }

    private func lookup(_ key: String, _ floats: [String: [Float]]) -> [Float] {
        guard let result = floats[key] else { preconditionFailure("Missing parameter \(key)") }
        return result
    }
}

private func convKey(_ layer: ConvLayerDesc) -> String {
    "conv:\(layer.name)"
}

private func matmulKey(_ layer: MatMulLayerDesc) -> String {
    "matmul:\(layer.name)"
}

private func biasKey(_ layer: MatBiasLayerDesc) -> String {
    "bias:\(layer.name)"
}

private func normScaleKey(_ layer: BatchNormLayerDesc) -> String {
    "bn-scale:\(layer.name)"
}

private func normBiasKey(_ layer: BatchNormLayerDesc) -> String {
    "bn-bias:\(layer.name)"
}
