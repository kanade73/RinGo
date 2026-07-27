import Foundation
@testable import RinGoModel
import XCTest

/// SWA weight-averaging (`ModelAverager`, BREAKTHROUGH_IDEAS E4). Pure CPU: builds tiny synthetic
/// same-architecture descriptors and checks the arithmetic-mean contract, self-average identity,
/// shape/architecture refusals, and that an averaged descriptor survives the `ModelWriter` path.
final class ModelAveragerTests: XCTestCase {
    func testAverageWithItselfReproducesEveryTensorExactly() throws {
        let model = makeModel(seed: 0.5)
        let averaged = try ModelAverager.average([model, model])
        let source = ModelAverager.tensorList(model)
        let result = ModelAverager.tensorList(averaged)
        XCTAssertEqual(result.count, source.count)
        for index in source.indices {
            XCTAssertEqual(result[index].name, source[index].name)
            // Exact equality: (x + x) / 2 == x for every finite fp32 x with no overflow.
            XCTAssertEqual(result[index].values, source[index].values, "\(source[index].name)")
        }
    }

    func testAverageOfTwoModelsIsExactArithmeticMean() throws {
        let first = makeModel(seed: 1)
        let second = makeModel(seed: -3)
        let averaged = try ModelAverager.average([first, second])

        let firstTensors = ModelAverager.tensorList(first)
        let secondTensors = ModelAverager.tensorList(second)
        let resultTensors = ModelAverager.tensorList(averaged)
        XCTAssertEqual(resultTensors.count, firstTensors.count)
        for index in resultTensors.indices {
            XCTAssertEqual(resultTensors[index].name, firstTensors[index].name)
            let a = firstTensors[index].values
            let b = secondTensors[index].values
            let expected = zip(a, b).map { ($0 + $1) / Float(2) }
            XCTAssertEqual(resultTensors[index].values, expected, "\(firstTensors[index].name)")
        }
    }

    func testAveragedModelSurvivesModelWriterRoundTrip() throws {
        // Proves the averaged descriptor is a valid, writable, reloadable model without a GPU: the
        // exact contract the live `rawnn` check exercises, minus the forward pass. BN scales are
        // >= 1 so the writer's canonicalization is a no-op and the reload is tensor-exact.
        let averaged = try ModelAverager.average([makeModel(seed: 2), makeModel(seed: 4)])
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("katago-averager-\(UUID().uuidString).bin.gz")
        defer { try? FileManager.default.removeItem(at: url) }
        try ModelWriter.write(desc: averaged, to: url)
        let reloaded = try ModelDesc.loadFromFileMaybeGZipped(url.path)

        let expected = ModelAverager.tensorList(averaged)
        let actual = ModelAverager.tensorList(reloaded)
        XCTAssertEqual(actual.count, expected.count)
        for index in expected.indices {
            XCTAssertEqual(actual[index].name, expected[index].name)
            XCTAssertEqual(actual[index].values.count, expected[index].values.count, "\(expected[index].name)")
            for element in expected[index].values.indices {
                XCTAssertEqual(
                    actual[index].values[element],
                    expected[index].values[element],
                    accuracy: 1e-6,
                    "\(expected[index].name)[\(element)]"
                )
            }
        }
    }

    func testShapeMismatchIsRefusedAndNamesTheTensor() {
        var mismatched = makeModel(seed: 0)
        // Same architecture scalars, but one head bias has the wrong length -> per-tensor refusal.
        mismatched.valueHead.v3Bias.weights = [0, 0]
        XCTAssertThrowsError(try ModelAverager.average([makeModel(seed: 0), mismatched])) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("bias:value_v3_bias"), message)
            XCTAssertTrue(message.contains("shape mismatch"), message)
        }
    }

    func testArchitectureMismatchOnModelVersionIsRefused() {
        var other = makeModel(seed: 0)
        other.modelVersion = 10
        XCTAssertThrowsError(try ModelAverager.average([makeModel(seed: 0), other])) { error in
            XCTAssertTrue(String(describing: error).contains("modelVersion"), String(describing: error))
        }
    }

    func testMetadataEncoderIsRefusedAsUnsupported() {
        var meta = makeModel(seed: 0)
        meta.metaEncoderVersion = 1
        XCTAssertThrowsError(try ModelAverager.average([makeModel(seed: 0), meta])) { error in
            XCTAssertTrue(String(describing: error).contains("metadata encoder"), String(describing: error))
        }
    }

    func testEmptyInputIsRefused() {
        XCTAssertThrowsError(try ModelAverager.average([]))
    }

    // MARK: - Synthetic model builder (single 1-channel ordinary block; every tensor distinct)

    private func makeModel(seed: Float) -> ModelDesc {
        let trunkBlock = ResidualBlockDesc(
            name: "block",
            preBN: batchNorm("block_pre", seed: seed + 0.1),
            preActivation: activation("block_pre_act"),
            regularConv: conv("block_regular", seed: seed + 0.2),
            midBN: batchNorm("block_mid", seed: seed + 0.3),
            midActivation: activation("block_mid_act"),
            finalConv: conv("block_final", seed: seed + 0.4)
        )
        let trunk = TrunkDesc(
            name: "trunk",
            modelVersion: 8,
            numBlocks: 1,
            trunkNumChannels: 1,
            midNumChannels: 1,
            regularNumChannels: 1,
            gpoolNumChannels: 1,
            metaEncoderVersion: 0,
            trunkNormKind: .standard,
            initialConv: conv("initial_conv", seed: seed + 0.5),
            initialMatMul: matMul("initial_global", inChannels: 1, outChannels: 1, seed: seed + 0.6),
            sgfMetadataEncoder: nil,
            blocks: [.ordinary(trunkBlock)],
            trunkTipBN: batchNorm("trunk_tip", seed: seed + 0.7),
            trunkTipRMSNorm: nil,
            trunkTipActivation: activation("trunk_tip_act")
        )
        let policy = PolicyHeadDesc(
            name: "policy",
            modelVersion: 8,
            policyOutChannels: 1,
            p1Conv: conv("policy_p1", seed: seed + 1.1),
            g1Conv: conv("policy_g1", seed: seed + 1.2),
            g1BN: batchNorm("policy_g1_bn", seed: seed + 1.3),
            g1Activation: activation("policy_g1_act"),
            gpoolToBiasMul: matMul("policy_bias_mul", inChannels: 3, outChannels: 1, seed: seed + 1.4),
            p1BN: batchNorm("policy_p1_bn", seed: seed + 1.5),
            p1Activation: activation("policy_p1_act"),
            p2Conv: conv("policy_p2", seed: seed + 1.6),
            gpoolToPassMul: matMul("policy_pass", inChannels: 3, outChannels: 1, seed: seed + 1.7),
            gpoolToPassBias: nil,
            passActivation: nil,
            gpoolToPassMul2: nil
        )
        let value = ValueHeadDesc(
            name: "value",
            modelVersion: 8,
            v1Conv: conv("value_v1", seed: seed + 2.1),
            v1BN: batchNorm("value_v1_bn", seed: seed + 2.2),
            v1Activation: activation("value_v1_act"),
            v2Mul: matMul("value_v2", inChannels: 3, outChannels: 1, seed: seed + 2.3),
            v2Bias: bias("value_v2_bias", count: 1, seed: seed + 2.4),
            v2Activation: activation("value_v2_act"),
            v3Mul: matMul("value_v3", inChannels: 1, outChannels: 3, seed: seed + 2.5),
            v3Bias: bias("value_v3_bias", count: 3, seed: seed + 2.6),
            sv3Mul: matMul("value_score", inChannels: 1, outChannels: 4, seed: seed + 2.7),
            sv3Bias: bias("value_score_bias", count: 4, seed: seed + 2.8),
            vOwnershipConv: conv("value_ownership", seed: seed + 2.9)
        )
        return ModelDesc(
            name: "synthavg",
            sha256: "",
            modelVersion: 8,
            numInputChannels: 1,
            numInputGlobalChannels: 1,
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

    private func conv(_ name: String, seed: Float) -> ConvLayerDesc {
        ConvLayerDesc(
            name: name,
            convYSize: 1,
            convXSize: 1,
            inChannels: 1,
            outChannels: 1,
            dilationY: 1,
            dilationX: 1,
            weights: [seed]
        )
    }

    /// Canonical BatchNorm with a merged scale >= 1 (mean 0, variance 1) so the descriptor is
    /// already in the loader's inference-ready form and round-trips through `ModelWriter` exactly.
    private func batchNorm(_ name: String, seed: Float) -> BatchNormLayerDesc {
        let scale = 1 + abs(seed) * 0.25
        let biasValue = seed * 0.5
        return BatchNormLayerDesc(
            name: name,
            numChannels: 1,
            epsilon: 1e-20,
            hasScale: true,
            hasBias: true,
            mean: [0],
            variance: [1],
            scale: [scale],
            bias: [biasValue],
            mergedScale: [scale],
            mergedBias: [biasValue]
        )
    }

    private func activation(_ name: String) -> ActivationLayerDesc {
        ActivationLayerDesc(name: name, activation: .relu)
    }

    private func matMul(_ name: String, inChannels: Int, outChannels: Int, seed: Float) -> MatMulLayerDesc {
        let count = inChannels * outChannels
        let weights = (0 ..< count).map { seed + Float($0) * 0.01 }
        return MatMulLayerDesc(name: name, inChannels: inChannels, outChannels: outChannels, weights: weights)
    }

    private func bias(_ name: String, count: Int, seed: Float) -> MatBiasLayerDesc {
        let weights = (0 ..< count).map { seed + Float($0) * 0.01 }
        return MatBiasLayerDesc(name: name, numChannels: count, weights: weights)
    }
}
