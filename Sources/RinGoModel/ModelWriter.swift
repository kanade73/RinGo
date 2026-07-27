import CZlib
import Foundation

public struct ModelWriteError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// Serializes inference-ready model descriptors in KataGo's model-file grammar.
///
/// `ModelDesc.parse` applies `transformToReduceActivations` after reading. The writer therefore
/// serializes an inverse-transformed copy. Parsing a file produced here reconstructs the supplied
/// inference-ready descriptor rather than applying the activation transform a second time.
public enum ModelWriter {
    public static func write(desc: ModelDesc, to url: URL, binaryFloats: Bool = true) throws {
        guard desc.metaEncoderVersion == 0, desc.trunk.metaEncoderVersion == 0,
              desc.trunk.sgfMetadataEncoder == nil
        else {
            throw ModelWriteError("Writing metaEncoderVersion > 0 is not supported")
        }
        guard desc.postProcessParams.outputScaleMultiplier == 1 else {
            throw ModelWriteError("outputScaleMultiplier is not represented in the KataGo model-file grammar")
        }

        var diskDesc = desc
        // Canonicalize before inverting. The loader always applies transformToReduceActivations,
        // so any desc that came from parsing is already in canonical form (every BN mergedScale
        // has magnitude >= 1, with the sub-1 factors folded into the preceding conv). For those
        // this call is a no-op, preserving the bit-exact round-trip. But a desc produced directly
        // — e.g. TrainableNetwork.exportDesc() on freshly-trained weights — can carry BN scales
        // below 1, which the inverse below rejects. Applying the (output-preserving) forward
        // transform first puts such a desc into canonical form so it can be written; the reloaded
        // net is functionally identical, though not tensor-identical, to the pre-write desc.
        diskDesc.transformToReduceActivations()
        try diskDesc.inverseTransformToReduceActivationsForWriting()
        var encoder = ModelTextEncoder(binaryFloats: binaryFloats)
        try encoder.writeModel(diskDesc)
        try writeGzip(encoder.data, to: url)
    }
}

private struct ModelTextEncoder {
    var data = Data()
    let binaryFloats: Bool

    mutating func writeModel(_ desc: ModelDesc) throws {
        guard (8 ... 17).contains(desc.modelVersion) else {
            throw ModelWriteError("Model version \(desc.modelVersion) is outside the supported range 8...17")
        }
        try token(desc.name, context: "model name")
        line(desc.modelVersion)
        line(desc.numInputChannels)
        line(desc.numInputGlobalChannels)
        if desc.modelVersion >= 13 {
            line(desc.postProcessParams.tdScoreMultiplier)
            line(desc.postProcessParams.scoreMeanMultiplier)
            line(desc.postProcessParams.scoreStdevMultiplier)
            line(desc.postProcessParams.leadMultiplier)
            line(desc.postProcessParams.varianceTimeMultiplier)
            line(desc.postProcessParams.shorttermValueErrorMultiplier)
            line(desc.postProcessParams.shorttermScoreErrorMultiplier)
        }
        if desc.modelVersion >= 15 {
            line(0)
            repeatLine(0, count: 7)
        }
        try writeTrunk(desc.trunk, modelVersion: desc.modelVersion)
        try writePolicyHead(desc.policyHead, modelVersion: desc.modelVersion)
        try writeValueHead(desc.valueHead, modelVersion: desc.modelVersion)
    }

    mutating func writeTrunk(_ trunk: TrunkDesc, modelVersion: Int) throws {
        try token(trunk.name, context: "trunk name")
        line(trunk.numBlocks)
        line(trunk.trunkNumChannels)
        line(trunk.midNumChannels)
        line(trunk.regularNumChannels)
        line(0)
        line(trunk.gpoolNumChannels)
        if modelVersion >= 15 {
            line(trunk.trunkNormKind.rawValue)
            repeatLine(0, count: 5)
        } else if trunk.trunkNormKind != .standard {
            throw ModelWriteError("\(trunk.name): RMSNorm trunk requires model version 15 or newer")
        }
        try writeConv(trunk.initialConv)
        try writeMatMul(trunk.initialMatMul)
        guard trunk.blocks.count == trunk.numBlocks else {
            throw ModelWriteError("\(trunk.name): numBlocks does not match blocks.count")
        }
        for block in trunk.blocks {
            try writeBlock(block, modelVersion: modelVersion)
        }
        switch trunk.trunkNormKind {
        case .standard:
            guard let norm = trunk.trunkTipBN, trunk.trunkTipRMSNorm == nil else {
                throw ModelWriteError("\(trunk.name): standard trunk must have only trunkTipBN")
            }
            try writeBatchNorm(norm)
        case .rmsNorm:
            guard let norm = trunk.trunkTipRMSNorm, trunk.trunkTipBN == nil else {
                throw ModelWriteError("\(trunk.name): RMSNorm trunk must have only trunkTipRMSNorm")
            }
            try writeRMSNorm(norm)
        }
        try writeActivation(trunk.trunkTipActivation, modelVersion: modelVersion)
    }

    mutating func writeBlock(_ block: BlockDesc, modelVersion: Int) throws {
        switch block {
        case let .ordinary(block):
            line("ordinary_block")
            try token(block.name, context: "ordinary block name")
            try writeBatchNorm(block.preBN)
            try writeActivation(block.preActivation, modelVersion: modelVersion)
            try writeConv(block.regularConv)
            try writeBatchNorm(block.midBN)
            try writeActivation(block.midActivation, modelVersion: modelVersion)
            try writeConv(block.finalConv)
        case let .globalPooling(block):
            line("gpool_block")
            try token(block.name, context: "global-pooling block name")
            try writeBatchNorm(block.preBN)
            try writeActivation(block.preActivation, modelVersion: modelVersion)
            try writeConv(block.regularConv)
            try writeConv(block.gpoolConv)
            try writeBatchNorm(block.gpoolBN)
            try writeActivation(block.gpoolActivation, modelVersion: modelVersion)
            try writeMatMul(block.gpoolToBiasMul)
            try writeBatchNorm(block.midBN)
            try writeActivation(block.midActivation, modelVersion: modelVersion)
            try writeConv(block.finalConv)
        case let .nestedBottleneck(block):
            line("nested_bottleneck_block")
            try token(block.name, context: "nested-bottleneck block name")
            line(block.numBlocks)
            try writeBatchNorm(block.preBN)
            try writeActivation(block.preActivation, modelVersion: modelVersion)
            try writeConv(block.preConv)
            guard block.blocks.count == block.numBlocks else {
                throw ModelWriteError("\(block.name): numBlocks does not match blocks.count")
            }
            for child in block.blocks {
                try writeBlock(child, modelVersion: modelVersion)
            }
            try writeBatchNorm(block.postBN)
            try writeActivation(block.postActivation, modelVersion: modelVersion)
            try writeConv(block.postConv)
        case let .transformerAttention(block):
            line("transformer_attention_block")
            try token(block.name, context: "transformer-attention block name")
            line(block.numHeads)
            line(block.numKVHeads)
            line(block.qHeadDim)
            line(block.vHeadDim)
            line(block.useRope ? 1 : 0)
            line(block.learnableRope ? 1 : 0)
            try writeTransformerRMSNorm(block.preLN)
            try writeMatMul(block.qProj)
            try writeMatMul(block.kProj)
            try writeMatMul(block.vProj)
            try writeMatMul(block.outProj)
            if block.useRope, block.learnableRope {
                try token("\(block.name).rope_freqs", context: "RoPE frequencies name")
                line(block.ropeNumKVHeads)
                line(block.ropeNumPairs)
                line(2)
                try floats(block.ropeFreqs, count: block.ropeNumKVHeads * block.ropeNumPairs * 2, name: block.name)
            } else if block.useRope {
                try token("\(block.name).rope_theta", context: "RoPE theta name")
                line(block.ropeTheta)
            }
        case let .transformerFFN(block):
            line("transformer_ffn_block")
            try token(block.name, context: "transformer-FFN block name")
            line(block.numChannels)
            line(block.ffnChannels)
            line(block.useSwiGLU ? 1 : 0)
            try writeTransformerRMSNorm(block.preLN)
            try writeMatMul(block.linear1)
            if block.useSwiGLU {
                guard let gate = block.linearGate else {
                    throw ModelWriteError("\(block.name): useSwiGLU is set but linearGate is missing")
                }
                try writeMatMul(gate)
            } else if block.linearGate != nil {
                throw ModelWriteError("\(block.name): linearGate is present but useSwiGLU is false")
            }
            try writeMatMul(block.linear2)
        }
    }

    mutating func writePolicyHead(_ head: PolicyHeadDesc, modelVersion: Int) throws {
        try token(head.name, context: "policy head name")
        if modelVersion >= 17 {
            line(head.policyOutChannels)
            repeatLine(0, count: 3)
        }
        try writeConv(head.p1Conv)
        try writeConv(head.g1Conv)
        try writeBatchNorm(head.g1BN)
        try writeActivation(head.g1Activation, modelVersion: modelVersion)
        try writeMatMul(head.gpoolToBiasMul)
        try writeBatchNorm(head.p1BN)
        try writeActivation(head.p1Activation, modelVersion: modelVersion)
        try writeConv(head.p2Conv)
        try writeMatMul(head.gpoolToPassMul)
        if modelVersion >= 15 {
            guard let bias = head.gpoolToPassBias, let activation = head.passActivation,
                  let multiply = head.gpoolToPassMul2
            else {
                throw ModelWriteError("\(head.name): version 15+ pass-head fields are missing")
            }
            try writeMatBias(bias)
            try writeActivation(activation, modelVersion: modelVersion)
            try writeMatMul(multiply)
        }
    }

    mutating func writeValueHead(_ head: ValueHeadDesc, modelVersion: Int) throws {
        try token(head.name, context: "value head name")
        if modelVersion >= 17 {
            repeatLine(0, count: 3)
        }
        try writeConv(head.v1Conv)
        try writeBatchNorm(head.v1BN)
        try writeActivation(head.v1Activation, modelVersion: modelVersion)
        try writeMatMul(head.v2Mul)
        try writeMatBias(head.v2Bias)
        try writeActivation(head.v2Activation, modelVersion: modelVersion)
        try writeMatMul(head.v3Mul)
        try writeMatBias(head.v3Bias)
        try writeMatMul(head.sv3Mul)
        try writeMatBias(head.sv3Bias)
        try writeConv(head.vOwnershipConv)
    }

    mutating func writeConv(_ layer: ConvLayerDesc) throws {
        try token(layer.name, context: "convolution name")
        line(layer.convYSize)
        line(layer.convXSize)
        line(layer.inChannels)
        line(layer.outChannels)
        line(layer.dilationY)
        line(layer.dilationX)
        let count = layer.convYSize * layer.convXSize * layer.inChannels * layer.outChannels
        guard layer.weights.count == count else {
            throw ModelWriteError("\(layer.name): expected \(count) convolution weights, found \(layer.weights.count)")
        }
        let ocStride = layer.inChannels * layer.convYSize * layer.convXSize
        let icStride = layer.convYSize * layer.convXSize
        var fileOrder = [Float]()
        fileOrder.reserveCapacity(count)
        // Invert the parser's y,x,ic,oc -> OIHW rearrangement.
        for y in 0 ..< layer.convYSize {
            for x in 0 ..< layer.convXSize {
                for input in 0 ..< layer.inChannels {
                    for output in 0 ..< layer.outChannels {
                        fileOrder.append(layer.weights[output * ocStride + input * icStride + y * layer.convXSize + x])
                    }
                }
            }
        }
        try floats(fileOrder, count: count, name: layer.name)
    }

    mutating func writeBatchNorm(_ layer: BatchNormLayerDesc) throws {
        try token(layer.name, context: "batch norm name")
        line(layer.numChannels)
        line(layer.epsilon)
        line(layer.hasScale ? 1 : 0)
        line(layer.hasBias ? 1 : 0)
        // These are the descriptor's raw fields. The inverse-transform copy reconstructs them
        // only for norms that need a preimage of the parser's activation-reduction transform.
        try floats(layer.mean, count: layer.numChannels, name: "\(layer.name) mean")
        try floats(layer.variance, count: layer.numChannels, name: "\(layer.name) variance")
        if layer.hasScale {
            try floats(layer.scale, count: layer.numChannels, name: "\(layer.name) scale")
        }
        if layer.hasBias {
            try floats(layer.bias, count: layer.numChannels, name: "\(layer.name) bias")
        }
    }

    mutating func writeActivation(_ layer: ActivationLayerDesc, modelVersion: Int) throws {
        try token(layer.name, context: "activation name")
        if modelVersion < 11 {
            guard layer.activation == .relu else {
                throw ModelWriteError("\(layer.name): model versions below 11 can only encode ReLU")
            }
            return
        }
        switch layer.activation {
        case .identity: line("ACTIVATION_IDENTITY")
        case .relu: line("ACTIVATION_RELU")
        case .mish: line("ACTIVATION_MISH")
        case .silu: line("ACTIVATION_SILU")
        case .mishScale8: throw ModelWriteError("\(layer.name): mishScale8 has no model-file token")
        }
    }

    mutating func writeMatMul(_ layer: MatMulLayerDesc) throws {
        try token(layer.name, context: "matrix multiply name")
        line(layer.inChannels)
        line(layer.outChannels)
        // ModelDesc and the file grammar both use [input][output] order.
        try floats(layer.weights, count: layer.inChannels * layer.outChannels, name: layer.name)
    }

    mutating func writeMatBias(_ layer: MatBiasLayerDesc) throws {
        try token(layer.name, context: "matrix bias name")
        line(layer.numChannels)
        try floats(layer.weights, count: layer.numChannels, name: layer.name)
    }

    mutating func writeRMSNorm(_ layer: RMSNormLayerDesc) throws {
        try token(layer.name, context: "RMSNorm name")
        line(layer.numChannels)
        line(layer.epsilon)
        line(layer.spatial ? 1 : 0)
        line(layer.cgroupSize)
        try floats(layer.gamma, count: layer.numChannels, name: "\(layer.name) gamma")
        try floats(layer.beta, count: layer.numChannels, name: "\(layer.name) beta")
    }

    mutating func writeTransformerRMSNorm(_ layer: TransformerRMSNormDesc) throws {
        try token(layer.name, context: "transformer RMSNorm name")
        line(layer.numChannels)
        line(layer.epsilon)
        try floats(layer.weight, count: layer.numChannels, name: layer.name)
    }

    mutating func floats(_ values: [Float], count: Int, name: String) throws {
        guard values.count == count else {
            throw ModelWriteError("\(name): expected \(count) floats, found \(values.count)")
        }
        guard values.allSatisfy(\.isFinite) else {
            throw ModelWriteError("\(name): weights contain NaN or infinity")
        }
        if binaryFloats {
            data.append(contentsOf: "@BIN@".utf8)
            for value in values {
                let bits = value.bitPattern.littleEndian
                data.append(UInt8(truncatingIfNeeded: bits))
                data.append(UInt8(truncatingIfNeeded: bits >> 8))
                data.append(UInt8(truncatingIfNeeded: bits >> 16))
                data.append(UInt8(truncatingIfNeeded: bits >> 24))
            }
            data.append(10)
        } else {
            for value in values {
                line(String(format: "%.9g", locale: Locale(identifier: "en_US_POSIX"), Double(value)))
            }
        }
    }

    mutating func token(_ value: String, context: String) throws {
        guard !value.isEmpty, value.utf8.allSatisfy({ !$0.isModelWriterWhitespace }) else {
            throw ModelWriteError("\(context) must be a nonempty token without whitespace")
        }
        line(value)
    }

    mutating func line(_ value: String) {
        data.append(contentsOf: value.utf8)
        data.append(10)
    }

    mutating func line(_ value: Int) {
        line(String(value))
    }

    mutating func line(_ value: Float) {
        line(String(format: "%.9g", locale: Locale(identifier: "en_US_POSIX"), Double(value)))
    }

    mutating func line(_ value: Double) {
        line(String(format: "%.17g", locale: Locale(identifier: "en_US_POSIX"), value))
    }

    mutating func repeatLine(_ value: Int, count: Int) {
        for _ in 0 ..< count {
            line(value)
        }
    }
}

private extension UInt8 {
    var isModelWriterWhitespace: Bool {
        self == 9 || self == 10 || self == 11 || self == 12 || self == 13 || self == 32
    }
}

private func writeGzip(_ data: Data, to url: URL) throws {
    guard let file = gzopen(url.path, "wb") else {
        throw ModelWriteError("Could not open gzip output file: \(url.path)")
    }
    var writeError: ModelWriteError?
    data.withUnsafeBytes { bytes in
        var offset = 0
        while offset < bytes.count {
            let count = min(bytes.count - offset, Int(UInt32.max))
            let written = gzwrite(file, bytes.baseAddress?.advanced(by: offset), UInt32(count))
            if written != count {
                var errorNumber: Int32 = 0
                let message = gzerror(file, &errorNumber).map(String.init(cString:)) ?? "unknown zlib error"
                writeError = ModelWriteError("Failed to write gzip model: \(message)")
                break
            }
            offset += Int(written)
        }
    }
    let closeStatus = gzclose(file)
    if let writeError {
        throw writeError
    }
    guard closeStatus == 0 else {
        throw ModelWriteError("Failed to finish gzip model: zlib status \(closeStatus)")
    }
}
