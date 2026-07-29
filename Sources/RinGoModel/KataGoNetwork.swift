import Foundation
import MLX
import MLXFast
import MLXNN

/// Raw, un-postprocessed neural-network head outputs. Spatial tensors use NHWC layout.
public struct ModelRawOutputs {
    public let policySpatialLogits: MLXArray
    public let policyPassLogits: MLXArray
    public let valueLogits: MLXArray
    public let scoreValues: MLXArray
    public let ownership: MLXArray
}

public struct KataGoNetworkError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// Debug-only forward-pass checkpoint hook, off by default (`nil`). When supplied to
/// `forward(spatial:global:debugSink:)`, it is invoked with the (still-lazy) tensor at each
/// bisection checkpoint (pre-block trunk, post-block trunk, transformer attention/FFN
/// internals) plus the matching validity mask, if any. Building a summary line from `tensor`
/// requires `eval()`, which is normally banned mid-graph (see docs/design-notes.md "MLX discipline") —
/// this hook is the sanctioned escape hatch for bisecting numerical parity against the
/// reference Eigen backend's `DEBUG_INTERMEDIATE_VALUES` dumps, never used on the production
/// path since callers must opt in explicitly.
public typealias NetworkDebugSink = (_ label: String, _ tensor: MLXArray, _ mask: MLXArray?) -> Void

/// A shape-specialized KataGo inference graph. Inputs are NHWC spatial features and NC global features.
public final class KataGoNetwork {
    public enum Precision: Sendable {
        case float32
        case float16
    }

    private let desc: ModelDesc
    private let nnXLen: Int
    private let nnYLen: Int
    private let dtype: DType
    private var parameters = [String: MLXArray]()
    private var injectedParameterLookup: ((String) -> MLXArray)?

    /// MLX 0.31.4 defaults `MLX_ENABLE_TF32=1`, which on Apple GPUs with neural accelerators
    /// routes even float32 matmuls and scaled-dot-product attention through TF32-reduced
    /// precision kernels (`mlx/utils.h` `enable_tf32`, gated in `metal/matmul.cpp` and
    /// `metal/scaled_dot_product_attention.cpp`). That costs ~7e-3 relative error per GEMM —
    /// far outside KataGo fp32 parity tolerances (measured directly against the reference
    /// Eigen backend: transformer Q projections drifted ~1e-3 while conv and GEMV paths,
    /// which never hit the TF32 kernels, matched to ~1e-6). MLX latches the flag in a static
    /// on the first GEMM dispatch, so opt out for the whole process before any forward pass
    /// runs. `overwrite: 0` keeps an explicit user-supplied `MLX_ENABLE_TF32` in force.
    /// Float16 inference is unaffected either way (non-float32 dtypes always take the fast
    /// path regardless of the flag).
    private static let disableTF32ReducedPrecision: Void = {
        setenv("MLX_ENABLE_TF32", "0", 0)
        return ()
    }()

    public init(desc: ModelDesc, nnXLen: Int, nnYLen: Int, precision: Precision) throws {
        _ = Self.disableTF32ReducedPrecision
        guard nnXLen > 0, nnYLen > 0 else {
            throw KataGoNetworkError("Neural-network dimensions must be positive")
        }
        guard desc.metaEncoderVersion == 0, desc.trunk.metaEncoderVersion == 0 else {
            throw KataGoNetworkError("SGF metadata encoders are not supported")
        }
        self.desc = desc
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        dtype = precision == .float32 ? .float32 : .float16

        try register(desc.trunk.initialConv)
        register(desc.trunk.initialMatMul)
        try register(blocks: desc.trunk.blocks)
        if let norm = desc.trunk.trunkTipBN {
            register(norm)
        }
        if let norm = desc.trunk.trunkTipRMSNorm {
            register(norm)
        }
        try register(policy: desc.policyHead)
        try register(value: desc.valueHead)
        eval(Array(parameters.values))
    }

    /// Builds one lazy forward graph. This method does not synchronize or inspect tensor values,
    /// unless `debugSink` is supplied (debug-only; see `NetworkDebugSink`, default `nil`).
    public func forward(
        spatial: MLXArray,
        global: MLXArray,
        debugSink: NetworkDebugSink? = nil
    ) -> ModelRawOutputs {
        forwardImpl(spatial: spatial, global: global, debugSink: debugSink)
    }

    /// Training-only entry point for reusing the inference graph with an externally owned
    /// parameter collection. The ordinary inference path above continues to read the eagerly
    /// registered model weights and therefore retains its original graph and numerics.
    @_spi(Training) public func forward(
        spatial: MLXArray,
        global: MLXArray,
        parameterLookup: @escaping (String) -> MLXArray
    ) -> ModelRawOutputs {
        precondition(injectedParameterLookup == nil, "Concurrent injected parameter lookup is not supported")
        injectedParameterLookup = parameterLookup
        defer { injectedParameterLookup = nil }
        return forwardImpl(spatial: spatial, global: global)
    }

    private func forwardImpl(
        spatial: MLXArray,
        global: MLXArray,
        debugSink: NetworkDebugSink? = nil
    ) -> ModelRawOutputs {
        precondition(spatial.ndim == 4 && spatial.dim(1) == nnYLen && spatial.dim(2) == nnXLen)
        precondition(spatial.dim(3) == desc.numInputChannels)
        precondition(global.ndim == 2 && global.dim(0) == spatial.dim(0))
        precondition(global.dim(1) == desc.numInputGlobalChannels)

        let mask = spatial[.ellipsis, 0 ..< 1].asType(dtype)
        let maskSum = mask.asType(.float32).sum(axes: [1, 2], keepDims: true)
        var trunk = convolution(spatial.asType(dtype) * mask, desc.trunk.initialConv)
        let initialBias = matrixMultiply(global.asType(dtype), desc.trunk.initialMatMul)
        trunk += initialBias.reshaped(initialBias.dim(0), 1, 1, initialBias.dim(1))
        debugSink?("pre-blocks (post initialConv+initialMatMul)", trunk, mask)
        trunk = apply(blocks: desc.trunk.blocks, to: trunk, mask: mask, maskSum: maskSum, debugSink: debugSink)

        switch desc.trunk.trunkNormKind {
        case .standard:
            trunk = batchNorm(
                trunk,
                tryParameter(batchNormScaleKey(desc.trunk.trunkTipBN!)),
                tryParameter(batchNormBiasKey(desc.trunk.trunkTipBN!)),
                activation: desc.trunk.trunkTipActivation.activation,
                mask: mask
            )
        case .rmsNorm:
            trunk = rmsNorm(
                trunk,
                desc.trunk.trunkTipRMSNorm!,
                activation: desc.trunk.trunkTipActivation.activation,
                mask: mask,
                maskSum: maskSum
            )
        }

        let policy = policyHead(trunk, mask: mask, maskSum: maskSum)
        let value = valueHead(trunk, mask: mask, maskSum: maskSum)
        return ModelRawOutputs(
            policySpatialLogits: policy.spatial.asType(.float32),
            policyPassLogits: policy.pass.asType(.float32),
            valueLogits: value.value.asType(.float32),
            scoreValues: value.score.asType(.float32),
            ownership: value.ownership.asType(.float32)
        )
    }

    private func apply(
        blocks: [BlockDesc],
        to input: MLXArray,
        mask: MLXArray,
        maskSum: MLXArray,
        debugSink: NetworkDebugSink? = nil
    ) -> MLXArray {
        var trunk = input
        for (index, block) in blocks.enumerated() {
            switch block {
            case let .ordinary(block):
                let mid = convolution(
                    batchNorm(trunk, block.preBN, activation: block.preActivation.activation, mask: mask),
                    block.regularConv
                )
                trunk += convolution(
                    batchNorm(mid, block.midBN, activation: block.midActivation.activation, mask: mask),
                    block.finalConv
                )
            case let .globalPooling(block):
                let pre = batchNorm(trunk, block.preBN, activation: block.preActivation.activation, mask: mask)
                var regular = convolution(pre, block.regularConv)
                let pooledBranch = batchNorm(
                    convolution(pre, block.gpoolConv),
                    block.gpoolBN,
                    activation: block.gpoolActivation.activation,
                    mask: mask
                )
                let bias = matrixMultiply(gpool(pooledBranch, mask: mask, maskSum: maskSum), block.gpoolToBiasMul)
                regular += bias.reshaped(bias.dim(0), 1, 1, bias.dim(1))
                trunk += convolution(
                    batchNorm(regular, block.midBN, activation: block.midActivation.activation, mask: mask),
                    block.finalConv
                )
            case let .nestedBottleneck(block):
                var mid = convolution(
                    batchNorm(trunk, block.preBN, activation: block.preActivation.activation, mask: mask),
                    block.preConv
                )
                mid = apply(blocks: block.blocks, to: mid, mask: mask, maskSum: maskSum, debugSink: debugSink)
                trunk += convolution(
                    batchNorm(mid, block.postBN, activation: block.postActivation.activation, mask: mask),
                    block.postConv
                )
            case let .transformerAttention(block):
                trunk = attention(trunk, block, mask: mask, debugSink: debugSink)
            case let .transformerFFN(block):
                trunk = transformerFFN(trunk, block, mask: mask, debugSink: debugSink)
            }
            debugSink?("block \(index) [\(block.kind.rawValue)]", trunk, mask)
        }
        return trunk
    }

    private func policyHead(
        _ trunk: MLXArray,
        mask: MLXArray,
        maskSum: MLXArray
    ) -> (spatial: MLXArray, pass: MLXArray) {
        let head = desc.policyHead
        var p1 = convolution(trunk, head.p1Conv)
        let g1 = batchNorm(
            convolution(trunk, head.g1Conv),
            head.g1BN,
            activation: head.g1Activation.activation,
            mask: mask
        )
        let pooled = gpool(g1, mask: mask, maskSum: maskSum)
        let bias = matrixMultiply(pooled, head.gpoolToBiasMul)
        p1 += bias.reshaped(bias.dim(0), 1, 1, bias.dim(1))
        p1 = batchNorm(p1, head.p1BN, activation: head.p1Activation.activation, mask: mask)
        let spatial = convolution(p1, head.p2Conv)
        let pass: MLXArray
        if head.modelVersion >= 15 {
            var hidden = matrixMultiply(pooled, head.gpoolToPassMul)
            hidden += tryParameter(biasKey(head.gpoolToPassBias!))
            hidden = kataGoActivation(hidden, head.passActivation!.activation)
            pass = matrixMultiply(hidden, head.gpoolToPassMul2!)
        } else {
            pass = matrixMultiply(pooled, head.gpoolToPassMul)
        }
        return (spatial, pass)
    }

    private func valueHead(
        _ trunk: MLXArray,
        mask: MLXArray,
        maskSum: MLXArray
    ) -> (value: MLXArray, score: MLXArray, ownership: MLXArray) {
        let head = desc.valueHead
        let v1 = batchNorm(
            convolution(trunk, head.v1Conv),
            head.v1BN,
            activation: head.v1Activation.activation,
            mask: mask
        )
        var v2 = matrixMultiply(valuePool(v1, maskSum: maskSum), head.v2Mul)
        v2 = kataGoActivation(v2 + tryParameter(biasKey(head.v2Bias)), head.v2Activation.activation)
        let value = matrixMultiply(v2, head.v3Mul) + tryParameter(biasKey(head.v3Bias))
        let score = matrixMultiply(v2, head.sv3Mul) + tryParameter(biasKey(head.sv3Bias))
        return (value, score, convolution(v1, head.vOwnershipConv))
    }

    private func attention(
        _ trunk: MLXArray,
        _ block: TransformerAttentionDesc,
        mask: MLXArray,
        debugSink: NetworkDebugSink? = nil
    ) -> MLXArray {
        let batch = trunk.dim(0)
        let sequence = nnXLen * nnYLen
        let normalized = transformerRMSNorm(trunk, block.preLN, mask: mask)
        debugSink?("attn:rmsnorm-out", normalized, mask)
        let flat = normalized.reshaped(batch, sequence, block.preLN.numChannels)
        let queryFlat = matrixMultiply(flat, block.qProj).reshaped(batch, sequence, block.numHeads, block.qHeadDim)
        debugSink?("attn:q", queryFlat.reshaped(batch, sequence, block.numHeads * block.qHeadDim), mask)
        var query = queryFlat.transposed(0, 2, 1, 3)
        var key = matrixMultiply(flat, block.kProj)
            .reshaped(batch, sequence, block.numKVHeads, block.qHeadDim)
            .transposed(0, 2, 1, 3)
        let value = matrixMultiply(flat, block.vProj)
            .reshaped(batch, sequence, block.numKVHeads, block.vHeadDim)
            .transposed(0, 2, 1, 3)
        if block.useRope {
            query = kataGoApplyRoPE(
                query,
                cosine: tryParameter(ropeQCosKey(block)),
                sine: tryParameter(ropeQSinKey(block))
            )
            key = kataGoApplyRoPE(
                key,
                cosine: tryParameter(ropeKCosKey(block)),
                sine: tryParameter(ropeKSinKey(block))
            )
        }

        let flatMask = mask.reshaped(batch, sequence)
        let additiveMask = (1 - flatMask).reshaped(batch, 1, 1, sequence).asType(dtype) * -1e9
        var attended = MLX.MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1 / Foundation.sqrt(Float(block.qHeadDim)),
            mask: additiveMask
        )
        attended *= flatMask.reshaped(batch, 1, sequence, 1)
        attended = attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, block.numHeads * block.vHeadDim)
        let projected = matrixMultiply(attended, block.outProj).reshaped(
            batch,
            nnYLen,
            nnXLen,
            block.outProj.outChannels
        )
        debugSink?("attn:outproj", projected, mask)
        let residual = trunk + projected * mask
        debugSink?("attn:residual", residual, mask)
        return residual
    }

    private func transformerFFN(
        _ trunk: MLXArray,
        _ block: TransformerFFNDesc,
        mask: MLXArray,
        debugSink: NetworkDebugSink? = nil
    ) -> MLXArray {
        let normalized = transformerRMSNorm(trunk, block.preLN, mask: mask)
        debugSink?("ffn:rmsnorm-out", normalized, mask)
        var hidden = matrixMultiply(normalized, block.linear1)
        if block.useSwiGLU {
            hidden = kataGoSwiGLU(hidden, gate: matrixMultiply(normalized, block.linearGate!))
        } else {
            hidden = kataGoActivation(hidden, .relu)
        }
        debugSink?("ffn:swiglu", hidden, nil)
        let residual = trunk + matrixMultiply(hidden, block.linear2) * mask
        debugSink?("ffn:residual", residual, mask)
        return residual
    }

    private func convolution(_ input: MLXArray, _ layer: ConvLayerDesc) -> MLXArray {
        conv2d(
            input,
            tryParameter(convKey(layer)),
            padding: [layer.convYSize / 2, layer.convXSize / 2]
        )
    }

    private func matrixMultiply(_ input: MLXArray, _ layer: MatMulLayerDesc) -> MLXArray {
        matmul(input, tryParameter(matMulKey(layer)))
    }

    private func batchNorm(
        _ input: MLXArray,
        _ layer: BatchNormLayerDesc,
        activation: ActivationKind,
        mask: MLXArray
    ) -> MLXArray {
        batchNorm(
            input,
            tryParameter(batchNormScaleKey(layer)),
            tryParameter(batchNormBiasKey(layer)),
            activation: activation,
            mask: mask
        )
    }

    private func batchNorm(
        _ input: MLXArray,
        _ scale: MLXArray,
        _ bias: MLXArray,
        activation: ActivationKind,
        mask: MLXArray
    ) -> MLXArray {
        kataGoBatchNorm(input, scale: scale, bias: bias, activation: activation, mask: mask)
    }

    private func transformerRMSNorm(
        _ input: MLXArray,
        _ layer: TransformerRMSNormDesc,
        mask: MLXArray
    ) -> MLXArray {
        MLX.MLXFast.rmsNorm(
            input,
            weight: tryParameter(transformerNormKey(layer)),
            eps: layer.epsilon
        ) * mask
    }

    private func rmsNorm(
        _ input: MLXArray,
        _ layer: RMSNormLayerDesc,
        activation: ActivationKind,
        mask: MLXArray,
        maskSum: MLXArray
    ) -> MLXArray {
        let gamma = tryParameter(rmsGammaKey(layer))
        let beta = tryParameter(rmsBetaKey(layer))
        let normalized: MLXArray
        if !layer.spatial {
            normalized = input * (input.square().mean(axis: -1, keepDims: true) + layer.epsilon).rsqrt()
        } else if layer.cgroupSize > 0 {
            let groupCount = layer.numChannels / layer.cgroupSize
            let grouped = input.reshaped(input.dim(0), nnYLen, nnXLen, groupCount, layer.cgroupSize)
                .transposed(0, 3, 4, 1, 2)
            let groupedMask = mask.reshaped(input.dim(0), 1, 1, nnYLen, nnXLen)
            let divisor = maskSum.reshaped(input.dim(0), 1, 1, 1, 1) * Float(layer.cgroupSize)
            let inverseRMS = ((grouped.square() * groupedMask).sum(axes: [2, 3, 4], keepDims: true)
                / divisor + layer.epsilon)
                .rsqrt()
            normalized = (grouped * inverseRMS).transposed(0, 3, 4, 1, 2).reshaped(input.shape)
        } else {
            let divisor = maskSum * Float(layer.numChannels)
            let inverseRMS = ((input.square() * mask).sum(axes: [1, 2, 3], keepDims: true)
                / divisor + layer.epsilon)
                .rsqrt()
            normalized = input * inverseRMS.asType(dtype)
        }
        return kataGoActivation(normalized * gamma + beta, activation) * mask
    }

    private func gpool(_ input: MLXArray, mask: MLXArray, maskSum: MLXArray) -> MLXArray {
        kataGoGPool(input, mask: mask, maskSum: maskSum)
    }

    private func valuePool(_ input: MLXArray, maskSum: MLXArray) -> MLXArray {
        let sum = input.sum(axes: [1, 2])
        let divisor = maskSum.reshaped(input.dim(0), 1).asType(dtype)
        let mean = sum / divisor
        let scale = (maskSum.sqrt() * 0.1 - 1.4).reshaped(input.dim(0), 1).asType(dtype)
        return concatenated([mean, mean * scale, mean * (scale.square() - 0.1)], axis: -1)
    }

    private func register(blocks: [BlockDesc]) throws {
        for block in blocks {
            switch block {
            case let .ordinary(block):
                register(block.preBN)
                try register(block.regularConv)
                register(block.midBN)
                try register(block.finalConv)
            case let .globalPooling(block):
                register(block.preBN)
                try register(block.regularConv)
                try register(block.gpoolConv)
                register(block.gpoolBN)
                register(block.gpoolToBiasMul)
                register(block.midBN)
                try register(block.finalConv)
            case let .nestedBottleneck(block):
                register(block.preBN)
                try register(block.preConv)
                try register(blocks: block.blocks)
                register(block.postBN)
                try register(block.postConv)
            case let .transformerAttention(block):
                register(block.preLN)
                register(block.qProj)
                register(block.kProj)
                register(block.vProj)
                register(block.outProj)
                if block.useRope {
                    try registerRope(block)
                }
            case let .transformerFFN(block):
                register(block.preLN)
                register(block.linear1)
                if let gate = block.linearGate {
                    register(gate)
                }
                register(block.linear2)
            }
        }
    }

    private func register(policy head: PolicyHeadDesc) throws {
        try register(head.p1Conv)
        try register(head.g1Conv)
        register(head.g1BN)
        register(head.gpoolToBiasMul)
        register(head.p1BN)
        try register(head.p2Conv)
        register(head.gpoolToPassMul)
        if let bias = head.gpoolToPassBias {
            register(bias)
        }
        if let multiply = head.gpoolToPassMul2 {
            register(multiply)
        }
    }

    private func register(value head: ValueHeadDesc) throws {
        try register(head.v1Conv)
        register(head.v1BN)
        register(head.v2Mul)
        register(head.v2Bias)
        register(head.v3Mul)
        register(head.v3Bias)
        register(head.sv3Mul)
        register(head.sv3Bias)
        try register(head.vOwnershipConv)
    }

    private func register(_ layer: ConvLayerDesc) throws {
        guard layer.dilationY == 1, layer.dilationX == 1 else {
            throw KataGoNetworkError("Dilated convolution is not supported: \(layer.name)")
        }
        parameters[convKey(layer)] = MLXArray(
            layer.weights,
            [layer.outChannels, layer.inChannels, layer.convYSize, layer.convXSize]
        ).transposed(0, 2, 3, 1).asType(dtype)
    }

    private func register(_ layer: MatMulLayerDesc) {
        parameters[matMulKey(layer)] = MLXArray(layer.weights, [layer.inChannels, layer.outChannels]).asType(dtype)
    }

    private func register(_ layer: MatBiasLayerDesc) {
        parameters[biasKey(layer)] = MLXArray(layer.weights).asType(dtype)
    }

    private func register(_ layer: BatchNormLayerDesc) {
        parameters[batchNormScaleKey(layer)] = MLXArray(layer.mergedScale).asType(dtype)
        parameters[batchNormBiasKey(layer)] = MLXArray(layer.mergedBias).asType(dtype)
    }

    private func register(_ layer: RMSNormLayerDesc) {
        parameters[rmsGammaKey(layer)] = MLXArray(layer.gamma).asType(dtype)
        parameters[rmsBetaKey(layer)] = MLXArray(layer.beta).asType(dtype)
    }

    private func register(_ layer: TransformerRMSNormDesc) {
        parameters[transformerNormKey(layer)] = MLXArray(layer.weight).asType(dtype)
    }

    private func registerRope(_ layer: TransformerAttentionDesc) throws {
        let tables = try layer.computeRopeCosSin(nnXLen: nnXLen, nnYLen: nnYLen, paddedNNXYLen: nnXLen * nnYLen)
        let pairs = layer.qHeadDim / 2
        let sequence = nnXLen * nnYLen
        let keyCos: MLXArray
        let keySin: MLXArray
        if layer.learnableRope {
            keyCos = MLXArray(tables.cos, [layer.numKVHeads, pairs, sequence]).transposed(0, 2, 1)
            keySin = MLXArray(tables.sin, [layer.numKVHeads, pairs, sequence]).transposed(0, 2, 1)
        } else {
            keyCos = MLXArray(tables.cos, [pairs, sequence]).transposed(1, 0)[.newAxis]
            keySin = MLXArray(tables.sin, [pairs, sequence]).transposed(1, 0)[.newAxis]
        }
        let keyCosPairs = repeated(keyCos, count: 2, axis: -1)
        let keySinPairs = repeated(keySin, count: 2, axis: -1)
        let queryCos: MLXArray
        let querySin: MLXArray
        if layer.learnableRope {
            let headMap = MLXArray((0 ..< layer.numHeads).map { $0 * layer.numKVHeads / layer.numHeads })
            queryCos = take(keyCosPairs, headMap, axis: 0)
            querySin = take(keySinPairs, headMap, axis: 0)
        } else {
            queryCos = keyCosPairs
            querySin = keySinPairs
        }
        parameters[ropeQCosKey(layer)] = queryCos[.newAxis].asType(dtype)
        parameters[ropeQSinKey(layer)] = querySin[.newAxis].asType(dtype)
        parameters[ropeKCosKey(layer)] = keyCosPairs[.newAxis].asType(dtype)
        parameters[ropeKSinKey(layer)] = keySinPairs[.newAxis].asType(dtype)
    }

    private func tryParameter(_ key: String) -> MLXArray {
        if let injectedParameterLookup {
            return injectedParameterLookup(key)
        }
        guard let parameter = parameters[key] else {
            preconditionFailure("Missing registered parameter \(key)")
        }
        return parameter
    }

    /// Debug-only convenience sink (see `NetworkDebugSink`): prints "DEBUG SWIFT ..." summary
    /// lines to stderr in the same format as the reference Eigen backend's
    /// `DEBUG_INTERMEDIATE_VALUES` dumps (`$KG/cpp/neuralnet/debugprint.cpp`), for bisecting
    /// numerical parity failures against `.tools/dump_nn_debug` output. Never called unless a
    /// caller explicitly opts in by passing it to `forward(debugSink:)`.
    public static func stderrDebugSink() -> NetworkDebugSink {
        { label, tensor, mask in
            let line = formatDebugSummary(label, tensor, mask: mask)
            FileHandle.standardError.write(Data((line + "\n").utf8))
        }
    }
}

/// Debug-only: builds a "DEBUG SWIFT <label> [shape] valid=.. min=.. max=.. mean=.. rms=.. first8=..
/// " line, mirroring `DebugPrint::print3DSummary`/`print2DSummary` in the reference Eigen
/// backend so lines can be diffed directly against `.tools/dump_nn_debug` stderr output.
/// `tensor`'s last axis is treated as channels; all other axes are treated as flattened
/// row-major "positions" (matching this port's NHWC == Eigen's NSC layout, see docs/design-notes.md).
/// When `mask` is supplied its element count must equal the position count; only positions
/// where `mask != 0` count towards valid/min/max/mean/rms (matching the reference). `first8`
/// is always the first 8 raw values (channels 0..7 of position 0), unmasked, matching the
/// reference's behavior of not filtering the raw preview values.
func formatDebugSummary(_ label: String, _ tensor: MLXArray, mask: MLXArray?) -> String {
    let channels = tensor.dim(-1)
    let flat = tensor.asType(.float32).reshaped(-1)
    eval(flat)
    let data = flat.asArray(Float.self)

    var expandedMask: [Float]?
    if let mask {
        let flatMask = mask.asType(.float32).reshaped(-1)
        eval(flatMask)
        let maskData = flatMask.asArray(Float.self)
        precondition(maskData.count * channels == data.count, "debug mask/tensor position count mismatch")
        var expanded = [Float](repeating: 1, count: data.count)
        for position in 0 ..< maskData.count {
            let value = maskData[position]
            let base = position * channels
            for channel in 0 ..< channels {
                expanded[base + channel] = value
            }
        }
        expandedMask = expanded
    }

    var validCount = 0
    var minValue = Float.greatestFiniteMagnitude
    var maxValue = -Float.greatestFiniteMagnitude
    var sum = 0.0
    var sumSquares = 0.0
    for (index, value) in data.enumerated() {
        if let expandedMask, expandedMask[index] == 0 { continue }
        validCount += 1
        minValue = Swift.min(minValue, value)
        maxValue = Swift.max(maxValue, value)
        sum += Double(value)
        sumSquares += Double(value) * Double(value)
    }
    if validCount == 0 {
        minValue = 0
        maxValue = 0
    }
    let mean = validCount > 0 ? sum / Double(validCount) : 0
    let rms = validCount > 0 ? (sumSquares / Double(validCount)).squareRoot() : 0

    let shape = tensor.shape.map(String.init).joined(separator: "x")
    var line = "DEBUG SWIFT \(label) [\(shape)] valid=\(validCount)"
        + " min=\(sixSignificantFigures(Double(minValue))) max=\(sixSignificantFigures(Double(maxValue)))"
        + " mean=\(sixSignificantFigures(mean)) rms=\(sixSignificantFigures(rms))"
    let previewCount = Swift.min(8, data.count)
    if previewCount > 0 {
        let preview = data.prefix(previewCount).map { " " + sixSignificantFigures(Double($0)) }.joined()
        line += " first\(previewCount)=" + preview
    }
    return line
}

private func sixSignificantFigures(_ value: Double) -> String {
    String(format: "%.6g", value)
}

func kataGoActivation(_ input: MLXArray, _ activation: ActivationKind) -> MLXArray {
    switch activation {
    case .identity: input
    case .relu: relu(input)
    case .mish: mish(input)
    case .silu: silu(input)
    case .mishScale8: input * tanh(softplus(input * 8))
    }
}

func kataGoBatchNorm(
    _ input: MLXArray,
    scale: MLXArray,
    bias: MLXArray,
    activation: ActivationKind,
    mask: MLXArray
) -> MLXArray {
    kataGoActivation(input * scale + bias, activation) * mask
}

func kataGoSwiGLU(_ value: MLXArray, gate: MLXArray) -> MLXArray {
    silu(value) * gate
}

func kataGoGPool(_ input: MLXArray, mask: MLXArray, maskSum: MLXArray) -> MLXArray {
    let divisor = maskSum.reshaped(input.dim(0), 1).asType(input.dtype)
    let mean = input.sum(axes: [1, 2]) / divisor
    let scale = (maskSum.sqrt() * 0.1 - 1.4).reshaped(input.dim(0), 1).asType(input.dtype)
    let maskedMaximum = maximum((input + mask - 1).max(axes: [1, 2]), -1)
    return concatenated([mean, mean * scale, maskedMaximum], axis: -1)
}

func kataGoApplyRoPE(_ input: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
    let even = input[.ellipsis, .stride(by: 2)]
    let odd = input[.ellipsis, .stride(from: 1, by: 2)]
    let rotated = stacked([-odd, even], axis: -1).flattened(start: -2, end: -1)
    return input * cosine + rotated * sine
}

private func convKey(_ layer: ConvLayerDesc) -> String {
    "conv:\(layer.name)"
}

private func matMulKey(_ layer: MatMulLayerDesc) -> String {
    "matmul:\(layer.name)"
}

private func biasKey(_ layer: MatBiasLayerDesc) -> String {
    "bias:\(layer.name)"
}

private func batchNormScaleKey(_ layer: BatchNormLayerDesc) -> String {
    "bn-scale:\(layer.name)"
}

private func batchNormBiasKey(_ layer: BatchNormLayerDesc) -> String {
    "bn-bias:\(layer.name)"
}

private func rmsGammaKey(_ layer: RMSNormLayerDesc) -> String {
    "rms-gamma:\(layer.name)"
}

private func rmsBetaKey(_ layer: RMSNormLayerDesc) -> String {
    "rms-beta:\(layer.name)"
}

private func transformerNormKey(_ layer: TransformerRMSNormDesc) -> String {
    "transformer-rms:\(layer.name)"
}

private func ropeQCosKey(_ layer: TransformerAttentionDesc) -> String {
    "rope-q-cos:\(layer.name)"
}

private func ropeQSinKey(_ layer: TransformerAttentionDesc) -> String {
    "rope-q-sin:\(layer.name)"
}

private func ropeKCosKey(_ layer: TransformerAttentionDesc) -> String {
    "rope-k-cos:\(layer.name)"
}

private func ropeKSinKey(_ layer: TransformerAttentionDesc) -> String {
    "rope-k-sin:\(layer.name)"
}
