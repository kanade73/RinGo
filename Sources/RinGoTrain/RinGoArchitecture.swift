import Foundation
import RinGoModel

public enum RinGoArchitecture {
    /// The student architectures `ringo train -arch` can build from scratch. All variants keep
    /// the model-v8 g170 block grammar (ordinary residual + global-pooling residual only — the
    /// block set `TrainableNetwork.validate` accepts), differing only in trunk depth/width. The
    /// raw values (`b6c96` etc.) double as the `-arch` flag spelling and the string recorded in
    /// `config.json`.
    public enum Variant: String, CaseIterable, Sendable {
        case b6c96
        case b10c128
        case b15c192
        case b20c256
        case b24c320
        case b28c384

        /// Trunk residual-block count (`bN`).
        public var numBlocks: Int {
            switch self {
            case .b6c96: 6
            case .b10c128: 10
            case .b15c192: 15
            case .b20c256: 20
            case .b24c320: 24
            case .b28c384: 28
            }
        }

        /// Trunk channel width (`cC`).
        public var channels: Int {
            switch self {
            case .b6c96: 96
            case .b10c128: 128
            case .b15c192: 192
            case .b20c256: 256
            case .b24c320: 320
            case .b28c384: 384
            }
        }
    }

    /// The exact model-v8 g170-b6c96 topology, with placeholder tensors that are replaced by
    /// `TrainableNetwork` initialization. No seed model file is required at runtime. Pinned
    /// byte-for-byte by `ArchitectureTests` (1,027,911 parameters, real-model topology self-test);
    /// `build` reproduces it identically for the `.b6c96` variant.
    public static func b6c96() -> ModelDesc {
        make(.b6c96)
    }

    /// The 10-block / 128-channel student (audit "実験枠"). Same recipe, larger trunk.
    public static func b10c128() -> ModelDesc {
        make(.b10c128)
    }

    /// The 15-block / 192-channel student (audit "博打枠").
    public static func b15c192() -> ModelDesc {
        make(.b15c192)
    }

    /// The 20-block / 256-channel student.
    public static func b20c256() -> ModelDesc {
        make(.b20c256)
    }

    /// The 24-block / 320-channel student.
    public static func b24c320() -> ModelDesc {
        make(.b24c320)
    }

    /// The 28-block / 384-channel student. Largest supported trunk.
    public static func b28c384() -> ModelDesc {
        make(.b28c384)
    }

    /// Builds the `ModelDesc` for a variant.
    public static func make(_ variant: Variant) -> ModelDesc {
        build(name: "ringo-\(variant.rawValue)", numBlocks: variant.numBlocks, channels: variant.channels)
    }

    /// Constructs a g170-style v8 topology parameterized on trunk depth (`numBlocks`) and width
    /// (`channels`). Every secondary width follows the exact proportions the pinned b6c96 uses:
    ///
    ///   head   = round(channels / 3)   — gpool / policy p1,g1 / value v1 width (b6c96: 96 -> 32)
    ///   wide   = 2 * head              — regular-branch / value v2 hidden width (b6c96: 64)
    ///
    /// so for `channels == 96` (head 32, wide 64) this reproduces b6c96 byte-for-byte. Global
    /// pooling emits 3 statistics per channel, so every matmul fed by a pooled branch takes
    /// `3 * head` inputs (b6c96: 3 * 32 == 96, which is why the original literals read "96").
    static func build(name: String, numBlocks: Int, channels: Int) -> ModelDesc {
        let trunkChannels = channels
        let head = Int((Double(channels) / 3.0).rounded()) // gpool / p1 / g1 / v1 width
        let wide = 2 * head // regular-branch / value-v2 hidden width
        let pooled = 3 * head // fan-in of any matmul reading a global-pooled branch

        let gpoolPositions = Self.gpoolPositions(numBlocks: numBlocks)
        let blocks: [BlockDesc] = (1 ... numBlocks).map { index in
            let name = "rconv\(index)"
            if gpoolPositions.contains(index) {
                return .globalPooling(GlobalPoolingResidualBlockDesc(
                    name: name,
                    modelVersion: 8,
                    preBN: norm("\(name)/norm1", trunkChannels, scale: false),
                    preActivation: activation("\(name)/actv1"),
                    regularConv: conv("\(name)/w1a", 3, trunkChannels, wide),
                    gpoolConv: conv("\(name)/w1b", 3, trunkChannels, head),
                    gpoolBN: norm("\(name)/norm1b", head, scale: false),
                    gpoolActivation: activation("\(name)/actv1b"),
                    gpoolToBiasMul: matmul("\(name)/w1r", pooled, wide),
                    midBN: norm("\(name)/norm2", wide, scale: true),
                    midActivation: activation("\(name)/actv2"),
                    finalConv: conv("\(name)/w2", 3, wide, trunkChannels)
                ))
            }
            return .ordinary(ResidualBlockDesc(
                name: name,
                preBN: norm("\(name)/norm1", trunkChannels, scale: false),
                preActivation: activation("\(name)/actv1"),
                regularConv: conv("\(name)/w1", 3, trunkChannels, trunkChannels),
                midBN: norm("\(name)/norm2", trunkChannels, scale: true),
                midActivation: activation("\(name)/actv2"),
                finalConv: conv("\(name)/w2", 3, trunkChannels, trunkChannels)
            ))
        }

        let trunk = TrunkDesc(
            name: "trunk",
            modelVersion: 8,
            numBlocks: numBlocks,
            trunkNumChannels: trunkChannels,
            midNumChannels: trunkChannels,
            regularNumChannels: wide,
            gpoolNumChannels: head,
            metaEncoderVersion: 0,
            trunkNormKind: .standard,
            initialConv: conv("conv1", 5, 22, trunkChannels),
            initialMatMul: matmul("ginputw", 19, trunkChannels),
            sgfMetadataEncoder: nil,
            blocks: blocks,
            trunkTipBN: norm("trunk/norm", trunkChannels, scale: false),
            trunkTipRMSNorm: nil,
            trunkTipActivation: activation("trunk/actv")
        )
        let policy = PolicyHeadDesc(
            name: "policyhead",
            modelVersion: 8,
            policyOutChannels: 1,
            p1Conv: conv("p1/intermediate_conv/w", 1, trunkChannels, head),
            g1Conv: conv("g1/w", 1, trunkChannels, head),
            g1BN: norm("g1/norm", head, scale: false),
            g1Activation: activation("g1/actv"),
            gpoolToBiasMul: matmul("matmulg2w", pooled, head),
            p1BN: norm("p1/norm", head, scale: false),
            p1Activation: activation("p1/actv"),
            p2Conv: conv("p2/w", 1, head, 1),
            gpoolToPassMul: matmul("matmulpass", pooled, 1),
            gpoolToPassBias: nil,
            passActivation: nil,
            gpoolToPassMul2: nil
        )
        let value = ValueHeadDesc(
            name: "valuehead",
            modelVersion: 8,
            v1Conv: conv("v1/w", 1, trunkChannels, head),
            v1BN: norm("v1/norm", head, scale: false),
            v1Activation: activation("v1/actv"),
            v2Mul: matmul("v2/w", pooled, wide),
            v2Bias: bias("v2/b", wide),
            v2Activation: activation("v2/actv"),
            v3Mul: matmul("v3/w", wide, 3),
            v3Bias: bias("v3/b", 3),
            sv3Mul: matmul("sv3/w", wide, 4),
            sv3Bias: bias("sv3/b", 4),
            vOwnershipConv: conv("vownership/w", 1, head, 1)
        )
        return ModelDesc(
            name: name,
            sha256: "",
            modelVersion: 8,
            numInputChannels: 22,
            numInputGlobalChannels: 19,
            numInputMetaChannels: 0,
            numPolicyChannels: 1,
            numValueChannels: 3,
            numScoreValueChannels: 4,
            numOwnershipChannels: 1,
            metaEncoderVersion: 0,
            postProcessParams: ModelPostProcessParams(),
            trunk: trunk,
            policyHead: policy,
            valueHead: value
        )
    }

    /// The two global-pooling residual positions (1-based block indices). b6c96 places them at
    /// blocks 3 and 5 of 6; scaling those two fractions (3/6, 5/6) by `numBlocks` keeps the same
    /// relative placement as the trunk deepens. For `numBlocks == 6` this is exactly {3, 5}, so
    /// b6c96's `[false, false, true, false, true, false]` pattern is preserved unchanged.
    static func gpoolPositions(numBlocks: Int) -> Set<Int> {
        let first = Int((Double(numBlocks) * 3.0 / 6.0).rounded())
        let second = Int((Double(numBlocks) * 5.0 / 6.0).rounded())
        return [first, second]
    }

    private static func conv(_ name: String, _ kernel: Int, _ input: Int, _ output: Int) -> ConvLayerDesc {
        ConvLayerDesc(
            name: name,
            convYSize: kernel,
            convXSize: kernel,
            inChannels: input,
            outChannels: output,
            weights: [Float](repeating: 0, count: kernel * kernel * input * output)
        )
    }

    private static func matmul(_ name: String, _ input: Int, _ output: Int) -> MatMulLayerDesc {
        MatMulLayerDesc(name: name, inChannels: input, outChannels: output,
                        weights: [Float](repeating: 0, count: input * output))
    }

    private static func bias(_ name: String, _ channels: Int) -> MatBiasLayerDesc {
        MatBiasLayerDesc(name: name, numChannels: channels, weights: [Float](repeating: 0, count: channels))
    }

    private static func norm(_ name: String, _ channels: Int, scale: Bool) -> BatchNormLayerDesc {
        BatchNormLayerDesc(
            name: name,
            numChannels: channels,
            epsilon: 1e-20,
            hasScale: scale,
            hasBias: true,
            mean: [Float](repeating: 0, count: channels),
            variance: [Float](repeating: 1, count: channels),
            scale: [Float](repeating: 1, count: channels),
            bias: [Float](repeating: 0, count: channels),
            mergedScale: [Float](repeating: 1, count: channels),
            mergedBias: [Float](repeating: 0, count: channels)
        )
    }

    private static func activation(_ name: String) -> ActivationLayerDesc {
        ActivationLayerDesc(name: name, activation: .relu)
    }
}
