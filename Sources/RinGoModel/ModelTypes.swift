import Foundation

public struct ModelParseError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

public enum ActivationKind: Int, Sendable {
    case identity = 0
    case relu = 1
    case mish = 2
    case silu = 3
    case mishScale8 = 12
}

public enum TrunkNormKind: Int, Sendable {
    case standard = 0
    case rmsNorm = 1
}

public enum BlockKind: String, Sendable {
    case regular
    case globalPooling
    case nestedBottleneck
    case transformerAttention
    case transformerFFN
}

public struct ConvLayerDesc: Sendable {
    public var name: String
    public var convYSize: Int
    public var convXSize: Int
    public var inChannels: Int
    public var outChannels: Int
    public var dilationY: Int
    public var dilationX: Int
    /// Parsed (not raw-file) order: OIHW row-major, i.e. `[outChannels][inChannels][convYSize]
    /// [convXSize]` with `x` fastest-changing. The on-disk model file order is different
    /// (y,x,ic,oc; see `ModelParser.swift`'s `ConvLayerDesc.init(reader:)`) — the parser
    /// rearranges into this OIHW order at load time, mirroring `$KG/cpp/neuralnet/desc.cpp`.
    public var weights: [Float]

    public init(
        name: String, convYSize: Int, convXSize: Int, inChannels: Int, outChannels: Int,
        dilationY: Int = 1, dilationX: Int = 1, weights: [Float]
    ) {
        self.name = name
        self.convYSize = convYSize
        self.convXSize = convXSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.dilationY = dilationY
        self.dilationX = dilationX
        self.weights = weights
    }

    public var spatialConvDepth: Double {
        Double(convYSize + convXSize - 2) / 4.0
    }

    public var numParameters: Int64 {
        Int64(weights.count)
    }
}

public struct BatchNormLayerDesc: Sendable {
    public var name: String
    public var numChannels: Int
    public var epsilon: Float
    public var hasScale: Bool
    public var hasBias: Bool
    public var mean: [Float]
    public var variance: [Float]
    public var scale: [Float]
    public var bias: [Float]
    public var mergedScale: [Float]
    public var mergedBias: [Float]

    public init(
        name: String, numChannels: Int, epsilon: Float, hasScale: Bool, hasBias: Bool,
        mean: [Float], variance: [Float], scale: [Float], bias: [Float],
        mergedScale: [Float], mergedBias: [Float]
    ) {
        self.name = name
        self.numChannels = numChannels
        self.epsilon = epsilon
        self.hasScale = hasScale
        self.hasBias = hasBias
        self.mean = mean
        self.variance = variance
        self.scale = scale
        self.bias = bias
        self.mergedScale = mergedScale
        self.mergedBias = mergedBias
    }

    public var numParameters: Int64 {
        Int64(hasScale ? numChannels : 0) + Int64(hasBias ? numChannels : 0)
    }
}

public struct ActivationLayerDesc: Sendable {
    public var name: String
    public var activation: ActivationKind

    public init(name: String, activation: ActivationKind) {
        self.name = name
        self.activation = activation
    }
}

public struct MatMulLayerDesc: Sendable {
    public var name: String
    public var inChannels: Int
    public var outChannels: Int
    /// Stored in input-channel, output-channel order.
    public var weights: [Float]

    public init(name: String, inChannels: Int, outChannels: Int, weights: [Float]) {
        self.name = name
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.weights = weights
    }

    public var numParameters: Int64 {
        Int64(weights.count)
    }
}

public struct MatBiasLayerDesc: Sendable {
    public var name: String
    public var numChannels: Int
    public var weights: [Float]

    public init(name: String, numChannels: Int, weights: [Float]) {
        self.name = name
        self.numChannels = numChannels
        self.weights = weights
    }

    public var numParameters: Int64 {
        Int64(weights.count)
    }
}

public struct ResidualBlockDesc: Sendable {
    public var name: String
    public var preBN: BatchNormLayerDesc
    public var preActivation: ActivationLayerDesc
    public var regularConv: ConvLayerDesc
    public var midBN: BatchNormLayerDesc
    public var midActivation: ActivationLayerDesc
    public var finalConv: ConvLayerDesc

    public init(
        name: String, preBN: BatchNormLayerDesc, preActivation: ActivationLayerDesc,
        regularConv: ConvLayerDesc, midBN: BatchNormLayerDesc,
        midActivation: ActivationLayerDesc, finalConv: ConvLayerDesc
    ) {
        self.name = name
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }

    public var spatialConvDepth: Double {
        regularConv.spatialConvDepth + finalConv.spatialConvDepth
    }

    public var numParameters: Int64 {
        preBN.numParameters + regularConv.numParameters + midBN.numParameters + finalConv.numParameters
    }
}

public struct GlobalPoolingResidualBlockDesc: Sendable {
    public var name: String
    public var modelVersion: Int
    public var preBN: BatchNormLayerDesc
    public var preActivation: ActivationLayerDesc
    public var regularConv: ConvLayerDesc
    public var gpoolConv: ConvLayerDesc
    public var gpoolBN: BatchNormLayerDesc
    public var gpoolActivation: ActivationLayerDesc
    public var gpoolToBiasMul: MatMulLayerDesc
    public var midBN: BatchNormLayerDesc
    public var midActivation: ActivationLayerDesc
    public var finalConv: ConvLayerDesc

    public init(
        name: String, modelVersion: Int, preBN: BatchNormLayerDesc,
        preActivation: ActivationLayerDesc, regularConv: ConvLayerDesc,
        gpoolConv: ConvLayerDesc, gpoolBN: BatchNormLayerDesc,
        gpoolActivation: ActivationLayerDesc, gpoolToBiasMul: MatMulLayerDesc,
        midBN: BatchNormLayerDesc, midActivation: ActivationLayerDesc,
        finalConv: ConvLayerDesc
    ) {
        self.name = name
        self.modelVersion = modelVersion
        self.preBN = preBN
        self.preActivation = preActivation
        self.regularConv = regularConv
        self.gpoolConv = gpoolConv
        self.gpoolBN = gpoolBN
        self.gpoolActivation = gpoolActivation
        self.gpoolToBiasMul = gpoolToBiasMul
        self.midBN = midBN
        self.midActivation = midActivation
        self.finalConv = finalConv
    }

    public var spatialConvDepth: Double {
        regularConv.spatialConvDepth + finalConv.spatialConvDepth
    }

    public var numParameters: Int64 {
        preBN.numParameters + regularConv.numParameters + gpoolConv.numParameters + gpoolBN.numParameters
            + gpoolToBiasMul.numParameters + midBN.numParameters + finalConv.numParameters
    }
}

public struct NestedBottleneckResidualBlockDesc: Sendable {
    public var name: String
    public var numBlocks: Int
    public var preBN: BatchNormLayerDesc
    public var preActivation: ActivationLayerDesc
    public var preConv: ConvLayerDesc
    public var blocks: [BlockDesc]
    public var postBN: BatchNormLayerDesc
    public var postActivation: ActivationLayerDesc
    public var postConv: ConvLayerDesc

    public var hasAnyTransformerBlocks: Bool {
        blocks.contains(where: \.hasAnyTransformerBlocks)
    }

    public var spatialConvDepth: Double {
        preConv.spatialConvDepth + blocks.reduce(0) { $0 + $1.spatialConvDepth } + postConv.spatialConvDepth
    }

    public var numParameters: Int64 {
        preBN.numParameters + preConv.numParameters + blocks.reduce(0) { $0 + $1.numParameters }
            + postBN.numParameters + postConv.numParameters
    }
}

public struct RMSNormLayerDesc: Sendable {
    public var name: String
    public var numChannels: Int
    public var epsilon: Float
    public var spatial: Bool
    public var cgroupSize: Int
    public var gamma: [Float]
    public var beta: [Float]

    public var numParameters: Int64 {
        Int64(gamma.count + beta.count)
    }
}

public struct TransformerRMSNormDesc: Sendable {
    public var name: String
    public var numChannels: Int
    public var epsilon: Float
    public var weight: [Float]

    public var numParameters: Int64 {
        Int64(weight.count)
    }
}

public struct TransformerAttentionDesc: Sendable {
    public var name: String
    public var numHeads: Int
    public var numKVHeads: Int
    public var qHeadDim: Int
    public var vHeadDim: Int
    public var useRope: Bool
    public var learnableRope: Bool
    public var preLN: TransformerRMSNormDesc
    public var qProj: MatMulLayerDesc
    public var kProj: MatMulLayerDesc
    public var vProj: MatMulLayerDesc
    public var outProj: MatMulLayerDesc
    public var ropeNumKVHeads: Int
    public var ropeNumPairs: Int
    public var ropeFreqs: [Float]
    public var ropeTheta: Float

    public var numParameters: Int64 {
        preLN.numParameters + qProj.numParameters + kProj.numParameters + vProj.numParameters
            + outProj.numParameters + Int64(ropeFreqs.count)
    }

    public func computeRopeCosSin(
        nnXLen: Int,
        nnYLen: Int,
        paddedNNXYLen: Int
    ) throws -> (cos: [Float], sin: [Float]) {
        guard useRope else {
            throw ModelParseError("TransformerAttentionDesc.computeRopeCosSin called but useRope is false")
        }
        let numPairs = qHeadDim / 2
        if learnableRope {
            guard ropeNumKVHeads == numKVHeads, ropeNumPairs == numPairs,
                  ropeFreqs.count == numKVHeads * numPairs * 2
            else {
                throw ModelParseError("\(name): inconsistent learnable RoPE dimensions")
            }
            var cosTable = [Float](repeating: 0, count: numKVHeads * numPairs * paddedNNXYLen)
            var sinTable = cosTable
            for head in 0 ..< numKVHeads {
                for pair in 0 ..< numPairs {
                    let freqX = ropeFreqs[(head * numPairs + pair) * 2]
                    let freqY = ropeFreqs[(head * numPairs + pair) * 2 + 1]
                    for y in 0 ..< nnYLen {
                        for x in 0 ..< nnXLen {
                            let xy = y * nnXLen + x
                            let angle = Float(x) * freqX + Float(y) * freqY
                            let index = (head * numPairs + pair) * paddedNNXYLen + xy
                            cosTable[index] = Foundation.cos(angle)
                            sinTable[index] = Foundation.sin(angle)
                        }
                    }
                }
            }
            return (cosTable, sinTable)
        }

        let numPairsPerDimension = numPairs / 2
        let dimensionHalf = qHeadDim / 2
        var cosTable = [Float](repeating: 0, count: numPairs * paddedNNXYLen)
        var sinTable = cosTable
        for pair in 0 ..< numPairs {
            for y in 0 ..< nnYLen {
                for x in 0 ..< nnXLen {
                    let adjustedPair = pair < numPairsPerDimension ? pair : pair - numPairsPerDimension
                    let frequency = 1 / Foundation.pow(ropeTheta, Float(2 * adjustedPair) / Float(dimensionHalf))
                    let angle = Float(pair < numPairsPerDimension ? y : x) * frequency
                    let index = pair * paddedNNXYLen + y * nnXLen + x
                    cosTable[index] = Foundation.cos(angle)
                    sinTable[index] = Foundation.sin(angle)
                }
            }
        }
        return (cosTable, sinTable)
    }
}

public struct TransformerFFNDesc: Sendable {
    public var name: String
    public var numChannels: Int
    public var ffnChannels: Int
    public var useSwiGLU: Bool
    public var preLN: TransformerRMSNormDesc
    public var linear1: MatMulLayerDesc
    public var linearGate: MatMulLayerDesc?
    public var linear2: MatMulLayerDesc

    public var numParameters: Int64 {
        preLN.numParameters + linear1.numParameters + (linearGate?.numParameters ?? 0) + linear2.numParameters
    }
}

public indirect enum BlockDesc: Sendable {
    case ordinary(ResidualBlockDesc)
    case globalPooling(GlobalPoolingResidualBlockDesc)
    case nestedBottleneck(NestedBottleneckResidualBlockDesc)
    case transformerAttention(TransformerAttentionDesc)
    case transformerFFN(TransformerFFNDesc)

    public var kind: BlockKind {
        switch self {
        case .ordinary: .regular
        case .globalPooling: .globalPooling
        case .nestedBottleneck: .nestedBottleneck
        case .transformerAttention: .transformerAttention
        case .transformerFFN: .transformerFFN
        }
    }

    public var hasAnyTransformerBlocks: Bool {
        switch self {
        case .transformerAttention, .transformerFFN: true
        case let .nestedBottleneck(block): block.hasAnyTransformerBlocks
        case .ordinary, .globalPooling: false
        }
    }

    public var spatialConvDepth: Double {
        switch self {
        case let .ordinary(block): block.spatialConvDepth
        case let .globalPooling(block): block.spatialConvDepth
        case let .nestedBottleneck(block): block.spatialConvDepth
        case .transformerAttention, .transformerFFN: 2
        }
    }

    public var numParameters: Int64 {
        switch self {
        case let .ordinary(block): block.numParameters
        case let .globalPooling(block): block.numParameters
        case let .nestedBottleneck(block): block.numParameters
        case let .transformerAttention(block): block.numParameters
        case let .transformerFFN(block): block.numParameters
        }
    }
}

public struct SGFMetadataEncoderDesc: Sendable {
    public var name: String
    public var metaEncoderVersion: Int
    public var numInputMetaChannels: Int
    public var mul1: MatMulLayerDesc
    public var bias1: MatBiasLayerDesc
    public var act1: ActivationLayerDesc
    public var mul2: MatMulLayerDesc
    public var bias2: MatBiasLayerDesc
    public var act2: ActivationLayerDesc
    public var mul3: MatMulLayerDesc

    public var numParameters: Int64 {
        mul1.numParameters + bias1.numParameters + mul2.numParameters + bias2.numParameters + mul3.numParameters
    }
}

public struct TrunkDesc: Sendable {
    public var name: String
    public var modelVersion: Int
    public var numBlocks: Int
    public var trunkNumChannels: Int
    public var midNumChannels: Int
    public var regularNumChannels: Int
    public var gpoolNumChannels: Int
    public var metaEncoderVersion: Int
    public var trunkNormKind: TrunkNormKind
    public var initialConv: ConvLayerDesc
    public var initialMatMul: MatMulLayerDesc
    public var sgfMetadataEncoder: SGFMetadataEncoderDesc?
    public var blocks: [BlockDesc]
    public var trunkTipBN: BatchNormLayerDesc?
    public var trunkTipRMSNorm: RMSNormLayerDesc?
    public var trunkTipActivation: ActivationLayerDesc

    public init(
        name: String, modelVersion: Int, numBlocks: Int, trunkNumChannels: Int,
        midNumChannels: Int, regularNumChannels: Int, gpoolNumChannels: Int,
        metaEncoderVersion: Int, trunkNormKind: TrunkNormKind,
        initialConv: ConvLayerDesc, initialMatMul: MatMulLayerDesc,
        sgfMetadataEncoder: SGFMetadataEncoderDesc?, blocks: [BlockDesc],
        trunkTipBN: BatchNormLayerDesc?, trunkTipRMSNorm: RMSNormLayerDesc?,
        trunkTipActivation: ActivationLayerDesc
    ) {
        self.name = name
        self.modelVersion = modelVersion
        self.numBlocks = numBlocks
        self.trunkNumChannels = trunkNumChannels
        self.midNumChannels = midNumChannels
        self.regularNumChannels = regularNumChannels
        self.gpoolNumChannels = gpoolNumChannels
        self.metaEncoderVersion = metaEncoderVersion
        self.trunkNormKind = trunkNormKind
        self.initialConv = initialConv
        self.initialMatMul = initialMatMul
        self.sgfMetadataEncoder = sgfMetadataEncoder
        self.blocks = blocks
        self.trunkTipBN = trunkTipBN
        self.trunkTipRMSNorm = trunkTipRMSNorm
        self.trunkTipActivation = trunkTipActivation
    }

    public var hasAnyTransformerBlocks: Bool {
        blocks.contains(where: \.hasAnyTransformerBlocks)
    }

    public var spatialConvDepth: Double {
        initialConv.spatialConvDepth + blocks.reduce(0) { $0 + $1.spatialConvDepth }
    }

    public var numParameters: Int64 {
        initialConv.numParameters + initialMatMul.numParameters + (sgfMetadataEncoder?.numParameters ?? 0)
            + blocks.reduce(0) { $0 + $1.numParameters } + (trunkTipBN?.numParameters ?? 0)
            + (trunkTipRMSNorm?.numParameters ?? 0)
    }
}

public struct PolicyHeadDesc: Sendable {
    public var name: String
    public var modelVersion: Int
    public var policyOutChannels: Int
    public var p1Conv: ConvLayerDesc
    public var g1Conv: ConvLayerDesc
    public var g1BN: BatchNormLayerDesc
    public var g1Activation: ActivationLayerDesc
    public var gpoolToBiasMul: MatMulLayerDesc
    public var p1BN: BatchNormLayerDesc
    public var p1Activation: ActivationLayerDesc
    public var p2Conv: ConvLayerDesc
    public var gpoolToPassMul: MatMulLayerDesc
    public var gpoolToPassBias: MatBiasLayerDesc?
    public var passActivation: ActivationLayerDesc?
    public var gpoolToPassMul2: MatMulLayerDesc?

    public init(
        name: String, modelVersion: Int, policyOutChannels: Int,
        p1Conv: ConvLayerDesc, g1Conv: ConvLayerDesc, g1BN: BatchNormLayerDesc,
        g1Activation: ActivationLayerDesc, gpoolToBiasMul: MatMulLayerDesc,
        p1BN: BatchNormLayerDesc, p1Activation: ActivationLayerDesc,
        p2Conv: ConvLayerDesc, gpoolToPassMul: MatMulLayerDesc,
        gpoolToPassBias: MatBiasLayerDesc?, passActivation: ActivationLayerDesc?,
        gpoolToPassMul2: MatMulLayerDesc?
    ) {
        self.name = name
        self.modelVersion = modelVersion
        self.policyOutChannels = policyOutChannels
        self.p1Conv = p1Conv
        self.g1Conv = g1Conv
        self.g1BN = g1BN
        self.g1Activation = g1Activation
        self.gpoolToBiasMul = gpoolToBiasMul
        self.p1BN = p1BN
        self.p1Activation = p1Activation
        self.p2Conv = p2Conv
        self.gpoolToPassMul = gpoolToPassMul
        self.gpoolToPassBias = gpoolToPassBias
        self.passActivation = passActivation
        self.gpoolToPassMul2 = gpoolToPassMul2
    }

    public var numParameters: Int64 {
        p1Conv.numParameters + g1Conv.numParameters + g1BN.numParameters + gpoolToBiasMul.numParameters
            + p1BN.numParameters + p2Conv.numParameters + gpoolToPassMul.numParameters
            + (gpoolToPassBias?.numParameters ?? 0) + (gpoolToPassMul2?.numParameters ?? 0)
    }
}

public struct ValueHeadDesc: Sendable {
    public var name: String
    public var modelVersion: Int
    public var v1Conv: ConvLayerDesc
    public var v1BN: BatchNormLayerDesc
    public var v1Activation: ActivationLayerDesc
    public var v2Mul: MatMulLayerDesc
    public var v2Bias: MatBiasLayerDesc
    public var v2Activation: ActivationLayerDesc
    public var v3Mul: MatMulLayerDesc
    public var v3Bias: MatBiasLayerDesc
    public var sv3Mul: MatMulLayerDesc
    public var sv3Bias: MatBiasLayerDesc
    public var vOwnershipConv: ConvLayerDesc

    public init(
        name: String, modelVersion: Int, v1Conv: ConvLayerDesc,
        v1BN: BatchNormLayerDesc, v1Activation: ActivationLayerDesc,
        v2Mul: MatMulLayerDesc, v2Bias: MatBiasLayerDesc,
        v2Activation: ActivationLayerDesc, v3Mul: MatMulLayerDesc,
        v3Bias: MatBiasLayerDesc, sv3Mul: MatMulLayerDesc,
        sv3Bias: MatBiasLayerDesc, vOwnershipConv: ConvLayerDesc
    ) {
        self.name = name
        self.modelVersion = modelVersion
        self.v1Conv = v1Conv
        self.v1BN = v1BN
        self.v1Activation = v1Activation
        self.v2Mul = v2Mul
        self.v2Bias = v2Bias
        self.v2Activation = v2Activation
        self.v3Mul = v3Mul
        self.v3Bias = v3Bias
        self.sv3Mul = sv3Mul
        self.sv3Bias = sv3Bias
        self.vOwnershipConv = vOwnershipConv
    }

    public var numParameters: Int64 {
        v1Conv.numParameters + v1BN.numParameters + v2Mul.numParameters + v2Bias.numParameters
            + v3Mul.numParameters + v3Bias.numParameters + sv3Mul.numParameters + sv3Bias.numParameters
            + vOwnershipConv.numParameters
    }
}

public struct ModelPostProcessParams: Sendable {
    public var tdScoreMultiplier: Double = 20
    public var scoreMeanMultiplier: Double = 20
    public var scoreStdevMultiplier: Double = 20
    public var leadMultiplier: Double = 20
    public var varianceTimeMultiplier: Double = 40
    public var shorttermValueErrorMultiplier: Double = 0.25
    public var shorttermScoreErrorMultiplier: Double = 30
    public var outputScaleMultiplier: Float = 1

    public init() {}
}

public struct ModelDesc: Sendable {
    public var name: String
    public var sha256: String
    public var modelVersion: Int
    public var numInputChannels: Int
    public var numInputGlobalChannels: Int
    public var numInputMetaChannels: Int
    public var numPolicyChannels: Int
    public var numValueChannels: Int
    public var numScoreValueChannels: Int
    public var numOwnershipChannels: Int
    public var metaEncoderVersion: Int
    public var postProcessParams: ModelPostProcessParams
    public var trunk: TrunkDesc
    public var policyHead: PolicyHeadDesc
    public var valueHead: ValueHeadDesc

    public init(
        name: String, sha256: String, modelVersion: Int, numInputChannels: Int,
        numInputGlobalChannels: Int, numInputMetaChannels: Int,
        numPolicyChannels: Int, numValueChannels: Int, numScoreValueChannels: Int,
        numOwnershipChannels: Int, metaEncoderVersion: Int,
        postProcessParams: ModelPostProcessParams, trunk: TrunkDesc,
        policyHead: PolicyHeadDesc, valueHead: ValueHeadDesc
    ) {
        self.name = name
        self.sha256 = sha256
        self.modelVersion = modelVersion
        self.numInputChannels = numInputChannels
        self.numInputGlobalChannels = numInputGlobalChannels
        self.numInputMetaChannels = numInputMetaChannels
        self.numPolicyChannels = numPolicyChannels
        self.numValueChannels = numValueChannels
        self.numScoreValueChannels = numScoreValueChannels
        self.numOwnershipChannels = numOwnershipChannels
        self.metaEncoderVersion = metaEncoderVersion
        self.postProcessParams = postProcessParams
        self.trunk = trunk
        self.policyHead = policyHead
        self.valueHead = valueHead
    }

    public var numParameters: Int64 {
        trunk.numParameters + policyHead.numParameters + valueHead.numParameters
    }

    public var hasAnyTransformerBlocks: Bool {
        trunk.hasAnyTransformerBlocks
    }

    public var shortInfoString: String {
        let isNested = trunk.blocks.contains { $0.kind == .nestedBottleneck }
        let kind: String = if isNested {
            hasAnyTransformerBlocks ? "nbt transformer" : "nbt convnet"
        } else {
            hasAnyTransformerBlocks ? "transformer" : "convnet"
        }
        return "\(kind), \(numParameters) params"
    }
}

public extension ConvLayerDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension BatchNormLayerDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension MatMulLayerDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension MatBiasLayerDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension ResidualBlockDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension GlobalPoolingResidualBlockDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension NestedBottleneckResidualBlockDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension RMSNormLayerDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension TransformerRMSNormDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension TransformerAttentionDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension TransformerFFNDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension SGFMetadataEncoderDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension TrunkDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension PolicyHeadDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension ValueHeadDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}

public extension ModelDesc {
    func getNumParameters() -> Int64 {
        numParameters
    }
}
