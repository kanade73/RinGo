import CZlib
import Foundation

private final class ModelTokenReader {
    private let data: Data
    private var offset = 0
    let binaryFloats: Bool

    init(data: Data, binaryFloats: Bool) {
        self.data = data
        self.binaryFloats = binaryFloats
    }

    func readToken(_ context: String) throws -> String {
        while offset < data.count, data[offset].isModelWhitespace {
            offset += 1
        }
        let start = offset
        while offset < data.count, !data[offset].isModelWhitespace {
            offset += 1
        }
        guard start < offset, let token = String(data: data[start ..< offset], encoding: .utf8) else {
            throw ModelParseError("\(context): could not read token at byte \(offset)")
        }
        return token
    }

    func readInt(_ context: String) throws -> Int {
        let token = try readToken(context)
        guard let value = Int(token) else {
            throw ModelParseError("\(context): expected integer, found '\(token)'")
        }
        return value
    }

    func readFloat(_ context: String) throws -> Float {
        let token = try readToken(context)
        guard let value = Float(token), value.isFinite else {
            throw ModelParseError("\(context): expected finite float, found '\(token)'")
        }
        return value
    }

    func readDouble(_ context: String) throws -> Double {
        let token = try readToken(context)
        guard let value = Double(token), value.isFinite else {
            throw ModelParseError("\(context): expected finite double, found '\(token)'")
        }
        return value
    }

    func readFloats(count: Int, name: String) throws -> [Float] {
        guard count >= 0 else {
            throw ModelParseError("\(name): negative float count")
        }
        if !binaryFloats {
            return try (0 ..< count).map { _ in try readFloat(name) }
        }

        var charactersBeforeMarker = 0
        while offset < data.count, data[offset] != Character("@").asciiValue {
            offset += 1
            charactersBeforeMarker += 1
            if charactersBeforeMarker > 100 {
                throw ModelParseError(
                    "\(name): could not read float weights; invalid binary model or text/binary format mismatch"
                )
            }
        }
        guard offset < data.count else {
            throw ModelParseError("\(name): could not find @BIN@ float marker")
        }
        offset += 1
        let marker = Data("BIN@".utf8)
        guard offset + marker.count <= data.count,
              data[offset ..< offset + marker.count].elementsEqual(marker)
        else {
            throw ModelParseError("\(name): did not find expected @BIN@ header for binary float block")
        }
        offset += marker.count

        let byteCount = try checkedProduct(count, 4, context: "\(name) binary float byte count")
        guard offset + byteCount <= data.count else {
            throw ModelParseError("\(name): did not find the expected number of floats in binary float block")
        }
        var result = [Float]()
        result.reserveCapacity(count)
        for index in 0 ..< count {
            let base = offset + index * 4
            let bits = UInt32(data[base])
                | UInt32(data[base + 1]) << 8
                | UInt32(data[base + 2]) << 16
                | UInt32(data[base + 3]) << 24
            let value = Float(bitPattern: bits)
            guard value.isFinite else {
                throw ModelParseError("\(name): NaN or infinite neural net weight or parameter")
            }
            result.append(value)
        }
        offset += byteCount
        return result
    }
}

private extension UInt8 {
    var isModelWhitespace: Bool {
        self == 9 || self == 10 || self == 11 || self == 12 || self == 13 || self == 32
    }
}

private extension Character {
    var asciiValue: UInt8 {
        String(self).utf8.first!
    }
}

private func checkedProduct(_ values: Int..., context: String) throws -> Int {
    var result = 1
    for value in values {
        let next = result.multipliedReportingOverflow(by: value)
        guard !next.overflow else {
            throw ModelParseError("\(context): integer overflow")
        }
        result = next.partialValue
    }
    return result
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    guard condition() else { throw ModelParseError(message) }
}

private extension ConvLayerDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("convolution layer name")
        convYSize = try reader.readInt("\(name) convYSize")
        convXSize = try reader.readInt("\(name) convXSize")
        inChannels = try reader.readInt("\(name) inChannels")
        outChannels = try reader.readInt("\(name) outChannels")
        dilationY = try reader.readInt("\(name) dilationY")
        dilationX = try reader.readInt("\(name) dilationX")
        try require(convXSize > 0 && convYSize > 0, "\(name): convolution filter sizes must be positive")
        try require(inChannels > 0 && outChannels > 0, "\(name): in and out channels must be positive")
        try require(dilationX > 0 && dilationY > 0, "\(name): dilation factors must be positive")
        try require(convXSize % 2 == 1 && convYSize % 2 == 1, "\(name): convolution filter sizes must be odd")

        let count = try checkedProduct(convYSize, convXSize, inChannels, outChannels, context: name)
        let fileOrderWeights = try reader.readFloats(count: count, name: name)
        // The model file stores floats in y,x,ic,oc order (oc fastest-changing) — NOT already
        // OIHW. Rearrange into this struct's OIHW row-major order (oc,ic,y,x; x fastest-
        // changing), mirroring $KG/cpp/neuralnet/desc.cpp ConvLayerDesc::ConvLayerDesc exactly
        // (its comment: "Model file order is y,x,ic,oc / Cuda's order is oc,ic,y,x"). Storing
        // `fileOrderWeights` verbatim here (skipping this rearrangement) silently scrambles
        // every convolution in the network — confirmed by bisecting against the reference
        // Eigen backend's DEBUG_INTERMEDIATE_VALUES dump, where the very first trunk checkpoint
        // (right after initialConv) already diverged before this fix.
        let ocStride = convYSize * convXSize * inChannels
        let icStride = convYSize * convXSize
        let yStride = convXSize
        var reordered = [Float](repeating: 0, count: count)
        var index = 0
        for y in 0 ..< convYSize {
            for x in 0 ..< convXSize {
                for ic in 0 ..< inChannels {
                    for oc in 0 ..< outChannels {
                        reordered[oc * ocStride + ic * icStride + y * yStride + x] = fileOrderWeights[index]
                        index += 1
                    }
                }
            }
        }
        weights = reordered
    }
}

private extension BatchNormLayerDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("batch norm layer name")
        numChannels = try reader.readInt("\(name) numChannels")
        epsilon = try reader.readFloat("\(name) epsilon")
        hasScale = try reader.readInt("\(name) hasScale") != 0
        hasBias = try reader.readInt("\(name) hasBias") != 0
        try require(numChannels >= 1, "\(name): numChannels (\(numChannels)) < 1")
        try require(epsilon > 0, "\(name): epsilon (\(epsilon)) <= 0")
        mean = try reader.readFloats(count: numChannels, name: name)
        variance = try reader.readFloats(count: numChannels, name: name)
        scale = hasScale ? try reader.readFloats(count: numChannels, name: name) : [Float](
            repeating: 1,
            count: numChannels
        )
        bias = hasBias ? try reader.readFloats(count: numChannels, name: name) : [Float](
            repeating: 0,
            count: numChannels
        )
        mergedScale = [Float](repeating: 0, count: numChannels)
        mergedBias = mergedScale
        for channel in 0 ..< numChannels {
            mergedScale[channel] = scale[channel] / Foundation.sqrt(variance[channel] + epsilon)
            mergedBias[channel] = bias[channel] - mergedScale[channel] * mean[channel]
        }
    }
}

private extension ActivationLayerDesc {
    init(reader: ModelTokenReader, modelVersion: Int) throws {
        name = try reader.readToken("activation layer name")
        if modelVersion < 11 {
            activation = .relu
            return
        }
        let token = try reader.readToken("\(name) activation kind")
        switch token {
        case "ACTIVATION_IDENTITY": activation = .identity
        case "ACTIVATION_RELU": activation = .relu
        case "ACTIVATION_MISH": activation = .mish
        case "ACTIVATION_SILU": activation = .silu
        default: throw ModelParseError("\(name): unknown activation \(token)")
        }
    }
}

private extension MatMulLayerDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("matrix multiply layer name")
        inChannels = try reader.readInt("\(name) inChannels")
        outChannels = try reader.readInt("\(name) outChannels")
        try require(inChannels > 0 && outChannels > 0, "\(name): in and out channels must be positive")
        weights = try reader.readFloats(
            count: checkedProduct(inChannels, outChannels, context: name),
            name: name
        )
    }

    static var empty: Self {
        Self(name: "", inChannels: 0, outChannels: 0, weights: [])
    }
}

private extension MatBiasLayerDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("matrix bias layer name")
        numChannels = try reader.readInt("\(name) numChannels")
        try require(numChannels > 0, "\(name): number of channels must be positive")
        weights = try reader.readFloats(count: numChannels, name: name)
    }
}

private extension ResidualBlockDesc {
    init(reader: ModelTokenReader, modelVersion: Int) throws {
        name = try reader.readToken("residual block name")
        preBN = try BatchNormLayerDesc(reader: reader)
        preActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        regularConv = try ConvLayerDesc(reader: reader)
        midBN = try BatchNormLayerDesc(reader: reader)
        midActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        finalConv = try ConvLayerDesc(reader: reader)
        try require(
            preBN.numChannels == regularConv.inChannels,
            "\(name): preBN channels do not match regularConv input"
        )
        try require(
            midBN.numChannels == regularConv.outChannels,
            "\(name): midBN channels do not match regularConv output"
        )
        try require(midBN.numChannels == finalConv.inChannels, "\(name): midBN channels do not match finalConv input")
    }
}

private extension GlobalPoolingResidualBlockDesc {
    init(reader: ModelTokenReader, modelVersion: Int) throws {
        name = try reader.readToken("global pooling residual block name")
        self.modelVersion = modelVersion
        preBN = try BatchNormLayerDesc(reader: reader)
        preActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        regularConv = try ConvLayerDesc(reader: reader)
        gpoolConv = try ConvLayerDesc(reader: reader)
        gpoolBN = try BatchNormLayerDesc(reader: reader)
        gpoolActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        gpoolToBiasMul = try MatMulLayerDesc(reader: reader)
        midBN = try BatchNormLayerDesc(reader: reader)
        midActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        finalConv = try ConvLayerDesc(reader: reader)
        try require(
            preBN.numChannels == regularConv.inChannels,
            "\(name): preBN channels do not match regularConv input"
        )
        try require(preBN.numChannels == gpoolConv.inChannels, "\(name): preBN channels do not match gpoolConv input")
        try require(
            gpoolBN.numChannels == gpoolConv.outChannels,
            "\(name): gpoolBN channels do not match gpoolConv output"
        )
        try require(
            gpoolBN.numChannels * 3 == gpoolToBiasMul.inChannels,
            "\(name): gpoolToBiasMul input channels mismatch"
        )
        try require(
            midBN.numChannels == regularConv.outChannels,
            "\(name): midBN channels do not match regularConv output"
        )
        try require(
            midBN.numChannels == gpoolToBiasMul.outChannels,
            "\(name): midBN channels do not match gpoolToBiasMul output"
        )
        try require(midBN.numChannels == finalConv.inChannels, "\(name): midBN channels do not match finalConv input")
    }
}

private extension RMSNormLayerDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("RMSNorm layer name")
        numChannels = try reader.readInt("\(name) numChannels")
        epsilon = try reader.readFloat("\(name) epsilon")
        spatial = try reader.readInt("\(name) spatial") != 0
        cgroupSize = try reader.readInt("\(name) cgroupSize")
        try require(epsilon > 0 && epsilon <= 1, "\(name): RMSNorm epsilon is not positive or is too large")
        try require(numChannels >= 1, "\(name): RMSNorm numChannels < 1")
        try require(cgroupSize == 0, "\(name): grouped spatial RMSNorm is not supported (cgroupSize \(cgroupSize))")
        gamma = try reader.readFloats(count: numChannels, name: name)
        beta = try reader.readFloats(count: numChannels, name: name)
    }
}

private extension TransformerRMSNormDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("transformer RMSNorm layer name")
        numChannels = try reader.readInt("\(name) numChannels")
        epsilon = try reader.readFloat("\(name) epsilon")
        try require(numChannels >= 1, "\(name): transformer RMSNorm numChannels < 1")
        try require(epsilon > 0 && epsilon <= 1, "\(name): transformer RMSNorm epsilon is not positive or is too large")
        weight = try reader.readFloats(count: numChannels, name: name)
    }
}

private extension TransformerAttentionDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("transformer attention block name")
        numHeads = try reader.readInt("\(name) numHeads")
        numKVHeads = try reader.readInt("\(name) numKVHeads")
        qHeadDim = try reader.readInt("\(name) qHeadDim")
        vHeadDim = try reader.readInt("\(name) vHeadDim")
        useRope = try reader.readInt("\(name) useRope") != 0
        learnableRope = try reader.readInt("\(name) learnableRope") != 0
        try require(numHeads >= 1 && numKVHeads >= 1, "\(name): attention head counts must be positive")
        try require(numHeads % numKVHeads == 0, "\(name): numHeads must be divisible by numKVHeads")
        try require(qHeadDim >= 1 && vHeadDim >= 1, "\(name): attention head dimensions must be positive")
        try require(!useRope || qHeadDim % 2 == 0, "\(name): qHeadDim must be even when RoPE is used")

        preLN = try TransformerRMSNormDesc(reader: reader)
        qProj = try MatMulLayerDesc(reader: reader)
        kProj = try MatMulLayerDesc(reader: reader)
        vProj = try MatMulLayerDesc(reader: reader)
        outProj = try MatMulLayerDesc(reader: reader)
        try require(qProj.outChannels == numHeads * qHeadDim, "\(name): qProj output channels mismatch")
        try require(kProj.outChannels == numKVHeads * qHeadDim, "\(name): kProj output channels mismatch")
        try require(vProj.outChannels == numKVHeads * vHeadDim, "\(name): vProj output channels mismatch")
        try require(outProj.inChannels == numHeads * vHeadDim, "\(name): outProj input channels mismatch")

        ropeNumKVHeads = 0
        ropeNumPairs = 0
        ropeFreqs = []
        ropeTheta = 0
        if useRope, learnableRope {
            _ = try reader.readToken("\(name) learnable RoPE frequencies name")
            ropeNumKVHeads = try reader.readInt("\(name) ropeNumKVHeads")
            ropeNumPairs = try reader.readInt("\(name) ropeNumPairs")
            let finalDimension = try reader.readInt("\(name) RoPE final dimension")
            try require(ropeNumKVHeads == numKVHeads, "\(name): ropeNumKVHeads does not match numKVHeads")
            try require(ropeNumPairs == qHeadDim / 2, "\(name): ropeNumPairs does not match qHeadDim / 2")
            try require(finalDimension == 2, "\(name): RoPE frequency final dimension must be 2")
            ropeFreqs = try reader.readFloats(
                count: checkedProduct(ropeNumKVHeads, ropeNumPairs, 2, context: name),
                name: name
            )
        } else if useRope {
            _ = try reader.readToken("\(name) RoPE theta name")
            ropeTheta = try reader.readFloat("\(name) RoPE theta")
            try require(ropeTheta > 0, "\(name): RoPE theta must be positive")
        }
    }
}

private extension TransformerFFNDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("transformer FFN block name")
        numChannels = try reader.readInt("\(name) numChannels")
        ffnChannels = try reader.readInt("\(name) ffnChannels")
        useSwiGLU = try reader.readInt("\(name) useSwiGLU") != 0
        try require(numChannels >= 1 && ffnChannels >= 1, "\(name): transformer FFN channels must be positive")
        preLN = try TransformerRMSNormDesc(reader: reader)
        linear1 = try MatMulLayerDesc(reader: reader)
        linearGate = useSwiGLU ? try MatMulLayerDesc(reader: reader) : nil
        linear2 = try MatMulLayerDesc(reader: reader)
        try require(linear1.inChannels == numChannels, "\(name): linear1 input channels mismatch")
        try require(linear1.outChannels == ffnChannels, "\(name): linear1 output channels mismatch")
        try require(
            linearGate == nil || linearGate?.inChannels == numChannels,
            "\(name): linearGate input channels mismatch"
        )
        try require(
            linearGate == nil || linearGate?.outChannels == ffnChannels,
            "\(name): linearGate output channels mismatch"
        )
        try require(linear2.inChannels == ffnChannels, "\(name): linear2 input channels mismatch")
        try require(linear2.outChannels == numChannels, "\(name): linear2 output channels mismatch")
    }
}

private func parseBlockStack(
    reader: ModelTokenReader,
    modelVersion: Int,
    name: String,
    numBlocks: Int,
    trunkNumChannels: Int
) throws -> [BlockDesc] {
    var blocks = [BlockDesc]()
    blocks.reserveCapacity(numBlocks)
    for _ in 0 ..< numBlocks {
        let kind = try reader.readToken("\(name) block kind")
        switch kind {
        case "ordinary_block":
            let block = try ResidualBlockDesc(reader: reader, modelVersion: modelVersion)
            try require(block.preBN.numChannels == trunkNumChannels, "\(name): \(block.name) preBN channels mismatch")
            try require(
                block.finalConv.outChannels == trunkNumChannels,
                "\(name): \(block.name) finalConv channels mismatch"
            )
            blocks.append(.ordinary(block))
        case "gpool_block":
            let block = try GlobalPoolingResidualBlockDesc(reader: reader, modelVersion: modelVersion)
            try require(block.preBN.numChannels == trunkNumChannels, "\(name): \(block.name) preBN channels mismatch")
            try require(
                block.finalConv.outChannels == trunkNumChannels,
                "\(name): \(block.name) finalConv channels mismatch"
            )
            blocks.append(.globalPooling(block))
        case "nested_bottleneck_block":
            let block = try NestedBottleneckResidualBlockDesc(reader: reader, modelVersion: modelVersion)
            try require(block.preBN.numChannels == trunkNumChannels, "\(name): \(block.name) preBN channels mismatch")
            try require(
                block.postConv.outChannels == trunkNumChannels,
                "\(name): \(block.name) postConv channels mismatch"
            )
            blocks.append(.nestedBottleneck(block))
        case "transformer_attention_block":
            let block = try TransformerAttentionDesc(reader: reader)
            try require(
                block.qProj.inChannels == trunkNumChannels,
                "\(name): \(block.name) qProj input channels mismatch"
            )
            try require(
                block.outProj.outChannels == trunkNumChannels,
                "\(name): \(block.name) outProj output channels mismatch"
            )
            blocks.append(.transformerAttention(block))
        case "transformer_ffn_block":
            let block = try TransformerFFNDesc(reader: reader)
            try require(block.numChannels == trunkNumChannels, "\(name): \(block.name) channels mismatch")
            blocks.append(.transformerFFN(block))
        default:
            throw ModelParseError("\(name): found unknown block kind: \(kind)")
        }
    }
    return blocks
}

private extension NestedBottleneckResidualBlockDesc {
    init(reader: ModelTokenReader, modelVersion: Int) throws {
        name = try reader.readToken("nested bottleneck residual block name")
        numBlocks = try reader.readInt("\(name) numBlocks")
        try require(numBlocks >= 1, "\(name): nested bottleneck numBlocks must be positive")
        preBN = try BatchNormLayerDesc(reader: reader)
        preActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        preConv = try ConvLayerDesc(reader: reader)
        blocks = try parseBlockStack(
            reader: reader,
            modelVersion: modelVersion,
            name: name,
            numBlocks: numBlocks,
            trunkNumChannels: preConv.outChannels
        )
        postBN = try BatchNormLayerDesc(reader: reader)
        postActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        postConv = try ConvLayerDesc(reader: reader)
        try require(preBN.numChannels == preConv.inChannels, "\(name): preBN channels do not match preConv input")
        try require(postBN.numChannels == preConv.outChannels, "\(name): postBN channels do not match preConv output")
        try require(postBN.numChannels == postConv.inChannels, "\(name): postBN channels do not match postConv input")
    }
}

private extension SGFMetadataEncoderDesc {
    init(reader: ModelTokenReader, modelVersion: Int, metaEncoderVersion: Int) throws {
        name = try reader.readToken("SGF metadata encoder name")
        self.metaEncoderVersion = metaEncoderVersion
        numInputMetaChannels = try reader.readInt("\(name) numInputMetaChannels")
        let expectedChannels = metaEncoderVersion == 1 ? 192 : 0
        try require(numInputMetaChannels == expectedChannels, "\(name): metadata input channel count mismatch")
        mul1 = try MatMulLayerDesc(reader: reader)
        bias1 = try MatBiasLayerDesc(reader: reader)
        act1 = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        mul2 = try MatMulLayerDesc(reader: reader)
        bias2 = try MatBiasLayerDesc(reader: reader)
        act2 = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        mul3 = try MatMulLayerDesc(reader: reader)
        try require(mul1.outChannels == bias1.numChannels, "\(name): mul1 and bias1 channels mismatch")
        try require(mul2.inChannels == mul1.outChannels, "\(name): mul2 and mul1 channels mismatch")
        try require(mul2.outChannels == bias2.numChannels, "\(name): mul2 and bias2 channels mismatch")
        try require(mul2.outChannels == mul3.inChannels, "\(name): mul2 and mul3 channels mismatch")
    }
}

private extension TrunkDesc {
    init(reader: ModelTokenReader, modelVersion: Int, metaEncoderVersion: Int) throws {
        name = try reader.readToken("trunk name")
        self.modelVersion = modelVersion
        numBlocks = try reader.readInt("\(name) numBlocks")
        trunkNumChannels = try reader.readInt("\(name) trunkNumChannels")
        midNumChannels = try reader.readInt("\(name) midNumChannels")
        regularNumChannels = try reader.readInt("\(name) regularNumChannels")
        _ = try reader.readInt("\(name) unused dilatedNumChannels")
        gpoolNumChannels = try reader.readInt("\(name) gpoolNumChannels")
        self.metaEncoderVersion = metaEncoderVersion
        trunkNormKind = .standard
        if modelVersion >= 15 {
            let normValue = try reader.readInt("\(name) trunkNormKind")
            guard let parsedKind = TrunkNormKind(rawValue: normValue) else {
                throw ModelParseError("\(name): unknown or unsupported trunk norm kind: \(normValue)")
            }
            trunkNormKind = parsedKind
            for option in ["B", "C", "D", "E", "F"] {
                let unused = try reader.readInt("\(name) trunk option \(option)")
                try require(unused == 0, "\(name): unknown/unsupported trunk option \(option): \(unused)")
            }
        }
        try require(numBlocks >= 1, "\(name): trunk numBlocks must be positive")
        try require(
            trunkNumChannels > 0 && midNumChannels > 0 && regularNumChannels > 0 && gpoolNumChannels > 0,
            "\(name): all numbers of channels must be positive"
        )

        initialConv = try ConvLayerDesc(reader: reader)
        try require(initialConv.outChannels == trunkNumChannels, "\(name): initialConv output channels mismatch")
        initialMatMul = try MatMulLayerDesc(reader: reader)
        try require(initialMatMul.outChannels == trunkNumChannels, "\(name): initialMatMul output channels mismatch")
        if metaEncoderVersion > 0 {
            let encoder = try SGFMetadataEncoderDesc(
                reader: reader,
                modelVersion: modelVersion,
                metaEncoderVersion: metaEncoderVersion
            )
            try require(encoder.mul1.inChannels == 192, "\(name): metadata encoder input channels mismatch")
            try require(
                encoder.mul3.outChannels == trunkNumChannels,
                "\(name): metadata encoder output channels mismatch"
            )
            sgfMetadataEncoder = encoder
        } else {
            sgfMetadataEncoder = nil
        }
        blocks = try parseBlockStack(
            reader: reader,
            modelVersion: modelVersion,
            name: name,
            numBlocks: numBlocks,
            trunkNumChannels: trunkNumChannels
        )
        if trunkNormKind == .standard {
            let norm = try BatchNormLayerDesc(reader: reader)
            try require(norm.numChannels == trunkNumChannels, "\(name): trunk-tip BatchNorm channels mismatch")
            trunkTipBN = norm
            trunkTipRMSNorm = nil
        } else {
            let norm = try RMSNormLayerDesc(reader: reader)
            try require(norm.numChannels == trunkNumChannels, "\(name): trunk-tip RMSNorm channels mismatch")
            trunkTipBN = nil
            trunkTipRMSNorm = norm
        }
        trunkTipActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
    }
}

private extension PolicyHeadDesc {
    init(reader: ModelTokenReader, modelVersion: Int) throws {
        name = try reader.readToken("policy head name")
        self.modelVersion = modelVersion
        if modelVersion >= 17 {
            policyOutChannels = try reader.readInt("\(name) policyOutChannels")
            try require(
                policyOutChannels == 2 || policyOutChannels == 4,
                "\(name): invalid policyOutChannels \(policyOutChannels)"
            )
        } else if modelVersion == 16 {
            policyOutChannels = 4
        } else if modelVersion >= 12 {
            policyOutChannels = 2
        } else {
            policyOutChannels = 1
        }
        if modelVersion >= 17 {
            for option in ["A", "B", "C"] {
                let unused = try reader.readInt("\(name) policy option \(option)")
                try require(unused == 0, "\(name): unknown/unsupported policy option \(option): \(unused)")
            }
        }

        p1Conv = try ConvLayerDesc(reader: reader)
        g1Conv = try ConvLayerDesc(reader: reader)
        g1BN = try BatchNormLayerDesc(reader: reader)
        g1Activation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        gpoolToBiasMul = try MatMulLayerDesc(reader: reader)
        p1BN = try BatchNormLayerDesc(reader: reader)
        p1Activation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        p2Conv = try ConvLayerDesc(reader: reader)
        gpoolToPassMul = try MatMulLayerDesc(reader: reader)
        if modelVersion >= 15 {
            gpoolToPassBias = try MatBiasLayerDesc(reader: reader)
            passActivation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
            gpoolToPassMul2 = try MatMulLayerDesc(reader: reader)
        } else {
            gpoolToPassBias = nil
            passActivation = nil
            gpoolToPassMul2 = nil
        }

        try require(p1Conv.outChannels == p1BN.numChannels, "\(name): p1Conv and p1BN channels mismatch")
        try require(g1Conv.outChannels == g1BN.numChannels, "\(name): g1Conv and g1BN channels mismatch")
        try require(
            gpoolToBiasMul.inChannels == g1BN.numChannels * 3,
            "\(name): gpoolToBiasMul input channels mismatch"
        )
        try require(gpoolToBiasMul.outChannels == p1BN.numChannels, "\(name): gpoolToBiasMul output channels mismatch")
        try require(p2Conv.inChannels == p1BN.numChannels, "\(name): p2Conv input channels mismatch")
        try require(
            gpoolToPassMul.inChannels == g1BN.numChannels * 3,
            "\(name): gpoolToPassMul input channels mismatch"
        )
        try require(p2Conv.outChannels == policyOutChannels, "\(name): p2Conv output channels mismatch")
        if modelVersion >= 15 {
            guard let passBias = gpoolToPassBias, let passMul2 = gpoolToPassMul2 else {
                throw ModelParseError("\(name): missing version 15 pass-head extension")
            }
            try require(gpoolToPassMul.outChannels == passBias.numChannels, "\(name): pass bias channels mismatch")
            try require(
                gpoolToPassMul.outChannels == passMul2.inChannels,
                "\(name): second pass multiply input mismatch"
            )
            try require(gpoolToPassMul.outChannels == p1Conv.outChannels, "\(name): pass hidden channels mismatch")
            try require(passMul2.outChannels == policyOutChannels, "\(name): second pass multiply output mismatch")
        } else {
            try require(gpoolToPassMul.outChannels == policyOutChannels, "\(name): pass output channels mismatch")
        }
    }
}

private extension ValueHeadDesc {
    init(reader: ModelTokenReader, modelVersion: Int) throws {
        name = try reader.readToken("value head name")
        self.modelVersion = modelVersion
        if modelVersion >= 17 {
            for option in ["A", "B", "C"] {
                let unused = try reader.readInt("\(name) value option \(option)")
                try require(unused == 0, "\(name): unknown/unsupported value option \(option): \(unused)")
            }
        }
        v1Conv = try ConvLayerDesc(reader: reader)
        v1BN = try BatchNormLayerDesc(reader: reader)
        v1Activation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        v2Mul = try MatMulLayerDesc(reader: reader)
        v2Bias = try MatBiasLayerDesc(reader: reader)
        v2Activation = try ActivationLayerDesc(reader: reader, modelVersion: modelVersion)
        v3Mul = try MatMulLayerDesc(reader: reader)
        v3Bias = try MatBiasLayerDesc(reader: reader)
        sv3Mul = try MatMulLayerDesc(reader: reader)
        sv3Bias = try MatBiasLayerDesc(reader: reader)
        vOwnershipConv = try ConvLayerDesc(reader: reader)

        try require(v1Conv.outChannels == v1BN.numChannels, "\(name): v1Conv and v1BN channels mismatch")
        try require(v2Mul.inChannels == v1BN.numChannels * 3, "\(name): v2Mul input channels mismatch")
        try require(v2Mul.outChannels == v2Bias.numChannels, "\(name): v2Mul and v2Bias channels mismatch")
        try require(v2Mul.outChannels == v3Mul.inChannels, "\(name): v2Mul and v3Mul channels mismatch")
        try require(v3Mul.outChannels == 3 && v3Bias.numChannels == 3, "\(name): value output must have 3 channels")
        try require(sv3Mul.inChannels == v2Mul.outChannels, "\(name): score-value input channels mismatch")
        let expectedScoreChannels = modelVersion >= 9 ? 6 : 4
        try require(
            sv3Mul.outChannels == expectedScoreChannels && sv3Bias.numChannels == expectedScoreChannels,
            "\(name): score-value output channels mismatch"
        )
        try require(vOwnershipConv.inChannels == v1Conv.outChannels, "\(name): ownership input channels mismatch")
        try require(vOwnershipConv.outChannels == 1, "\(name): ownership output channels must be 1")
    }
}

private extension ModelDesc {
    init(reader: ModelTokenReader) throws {
        name = try reader.readToken("model name")
        sha256 = ""
        modelVersion = try reader.readInt("\(name) modelVersion")
        try require(name.count <= 96, "Model name is too long (\(name.count) chars, max 96): \(name)")
        let allowedName = name.unicodeScalars.allSatisfy { scalar in
            (65 ... 90).contains(scalar.value) || (97 ... 122).contains(scalar.value)
                || (48 ... 57).contains(scalar.value) || scalar.value == 95 || scalar.value == 45
        }
        try require(
            allowedName,
            "Model name must contain only alphanumeric characters, underscores, and hyphens: \(name)"
        )
        try require(
            (8 ... 17).contains(modelVersion),
            "This parser supports KataGo model versions 8 through 17; found version \(modelVersion)"
        )

        numInputChannels = try reader.readInt("\(name) numInputChannels")
        numInputGlobalChannels = try reader.readInt("\(name) numInputGlobalChannels")
        try require(numInputChannels > 0, "\(name): numInputChannels must be positive")
        try require(numInputGlobalChannels > 0, "\(name): numInputGlobalChannels must be positive")

        postProcessParams = ModelPostProcessParams()
        if modelVersion >= 13 {
            postProcessParams.tdScoreMultiplier = try reader.readDouble("\(name) tdScoreMultiplier")
            postProcessParams.scoreMeanMultiplier = try reader.readDouble("\(name) scoreMeanMultiplier")
            postProcessParams.scoreStdevMultiplier = try reader.readDouble("\(name) scoreStdevMultiplier")
            postProcessParams.leadMultiplier = try reader.readDouble("\(name) leadMultiplier")
            postProcessParams.varianceTimeMultiplier = try reader.readDouble("\(name) varianceTimeMultiplier")
            postProcessParams.shorttermValueErrorMultiplier = try reader
                .readDouble("\(name) shorttermValueErrorMultiplier")
            postProcessParams.shorttermScoreErrorMultiplier = try reader
                .readDouble("\(name) shorttermScoreErrorMultiplier")
            let values = [
                postProcessParams.tdScoreMultiplier,
                postProcessParams.scoreMeanMultiplier,
                postProcessParams.scoreStdevMultiplier,
                postProcessParams.leadMultiplier,
                postProcessParams.varianceTimeMultiplier,
                postProcessParams.shorttermValueErrorMultiplier,
                postProcessParams.shorttermScoreErrorMultiplier,
            ]
            try require(values.allSatisfy { $0 > 0 }, "\(name): post-process multipliers must be positive")
        }

        if modelVersion >= 15 {
            metaEncoderVersion = try reader.readInt("\(name) metaEncoderVersion")
            try require(
                (0 ... 1).contains(metaEncoderVersion),
                "\(name): metaEncoderVersion is not implemented: \(metaEncoderVersion)"
            )
            numInputMetaChannels = metaEncoderVersion == 1 ? 192 : 0
            for option in ["B", "C", "D", "E", "F", "G", "H"] {
                let unused = try reader.readInt("\(name) model option \(option)")
                try require(unused == 0, "\(name): unknown/unsupported model option \(option): \(unused)")
            }
        } else {
            metaEncoderVersion = 0
            numInputMetaChannels = 0
        }

        trunk = try TrunkDesc(reader: reader, modelVersion: modelVersion, metaEncoderVersion: metaEncoderVersion)
        policyHead = try PolicyHeadDesc(reader: reader, modelVersion: modelVersion)
        valueHead = try ValueHeadDesc(reader: reader, modelVersion: modelVersion)
        numPolicyChannels = policyHead.policyOutChannels
        numValueChannels = valueHead.v3Mul.outChannels
        numScoreValueChannels = valueHead.sv3Mul.outChannels
        numOwnershipChannels = valueHead.vOwnershipConv.outChannels
        try require(numInputChannels == trunk.initialConv.inChannels, "\(name): spatial input channels mismatch")
        try require(numInputGlobalChannels == trunk.initialMatMul.inChannels, "\(name): global input channels mismatch")
        try require(
            trunk.trunkNumChannels == policyHead.p1Conv.inChannels,
            "\(name): policy p1 input channels mismatch"
        )
        try require(
            trunk.trunkNumChannels == policyHead.g1Conv.inChannels,
            "\(name): policy g1 input channels mismatch"
        )
        try require(trunk.trunkNumChannels == valueHead.v1Conv.inChannels, "\(name): value input channels mismatch")
    }
}

public extension ModelDesc {
    static func parse(data: Data, binaryFloats: Bool) throws -> ModelDesc {
        do {
            var model = try ModelDesc(reader: ModelTokenReader(data: data, binaryFloats: binaryFloats))
            model.transformToReduceActivations()
            return model
        } catch let error as ModelParseError {
            throw error
        } catch {
            throw ModelParseError("Unexpected model parse error: \(error)")
        }
    }

    /// Loads `.txt`, `.bin`, `.txt.gz`, `.bin.gz`, or an extension-ambiguous `.gz` model.
    /// `expectedSha256` is accepted for API parity but intentionally not verified.
    static func loadFromFileMaybeGZipped(
        _ fileName: String,
        expectedSha256 _: String = ""
    ) throws -> ModelDesc {
        do {
            let lower = fileName.lowercased()
            if lower.hasSuffix(".txt") {
                return try parse(data: Data(contentsOf: URL(fileURLWithPath: fileName)), binaryFloats: false)
            }
            if lower.hasSuffix(".bin") {
                return try parse(data: Data(contentsOf: URL(fileURLWithPath: fileName)), binaryFloats: true)
            }
            if lower.hasSuffix(".txt.gz") || lower.hasSuffix(".bin.gz") || lower.hasSuffix(".gz") {
                let data = try readGzipFile(fileName)
                let binaryFloats = !lower.hasSuffix(".txt.gz")
                do {
                    return try parse(data: data, binaryFloats: binaryFloats)
                } catch let binaryError where binaryFloats && !lower.hasSuffix(".bin.gz") {
                    do {
                        return try parse(data: data, binaryFloats: false)
                    } catch let textError {
                        throw ModelParseError(
                            "Could neither parse .gz model as text nor binary; errors were:\n\(textError)\n\(binaryError)"
                        )
                    }
                }
            }
            throw ModelParseError("Model file must end with .txt, .bin, .txt.gz, .bin.gz, or .gz")
        } catch let error as ModelParseError {
            throw ModelParseError("Error loading or parsing model file \(fileName): \(error.description)")
        } catch {
            throw ModelParseError("Error loading or parsing model file \(fileName): \(error)")
        }
    }
}

private func readGzipFile(_ path: String) throws -> Data {
    guard let file = gzopen(path, "rb") else {
        throw ModelParseError("Could not open gzip file")
    }
    defer { gzclose(file) }
    var result = Data()
    var buffer = [UInt8](repeating: 0, count: 1 << 16)
    while true {
        let count = buffer.withUnsafeMutableBytes { bytes in
            gzread(file, bytes.baseAddress, UInt32(bytes.count))
        }
        if count > 0 {
            result.append(buffer, count: Int(count))
        } else if count == 0 {
            break
        } else {
            var errorNumber: Int32 = 0
            let message = gzerror(file, &errorNumber).map(String.init(cString:)) ?? "unknown zlib error"
            throw ModelParseError("Failed to decompress gzip file: \(message)")
        }
    }
    return result
}
