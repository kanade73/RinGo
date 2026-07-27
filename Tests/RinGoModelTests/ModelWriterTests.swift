import Foundation
@testable import RinGoModel
import XCTest

final class ModelWriterTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"

    func testBinaryVersion8RoundTripPreservesEveryField() throws {
        let source = try loadFixture("g170-b6c96-s175395328-d26788732.bin.gz")
        let result = try roundTrip(source, binaryFloats: true)
        assertDescriptorEqual(source, result, accuracy: 1e-7)
    }

    func testBinaryVersion17RoundTripPreservesNestedTransformerAndRMSNorm() throws {
        let source = try loadFixture("b4c256h4nbttflrs-fson-silu-rsnh.bin.gz")
        let result = try roundTrip(source, binaryFloats: true)
        assertDescriptorEqual(source, result, accuracy: 1e-7)
    }

    func testTextVersion8RoundTripPreservesEveryField() throws {
        let source = try loadFixture("g170-b6c96-s175395328-d26788732.bin.gz")
        let result = try roundTrip(source, binaryFloats: false)
        assertDescriptorEqual(source, result, accuracy: 1e-7)
    }

    func testSyntheticNonCanonicalDescriptorRoundTripIsFunctionPreserving() throws {
        // This synthetic desc has a sub-1 trunk-tip BN scale (0.5), which the loader's
        // output-preserving transformToReduceActivations would never leave in place — a genuinely
        // inference-ready desc has every BN mergedScale magnitude >= 1. ModelWriter canonicalizes
        // such a desc before serializing, so the reload is function-preserving (equal to what
        // canonicalizing the input yields) rather than tensor-identical to the raw input. Assert
        // exactly that contract: write+reload == transformToReduceActivations(input).
        let source = syntheticDescriptor()
        XCTAssertEqual(source.trunk.trunkTipBN?.mergedScale, [0.5])
        var expected = source
        expected.transformToReduceActivations()
        let result = try roundTrip(source, binaryFloats: true)
        assertDescriptorEqual(expected, result, accuracy: 1e-6)
    }

    func testCanonicalSyntheticDescriptorRoundTripIsBitExact() throws {
        // A desc already in canonical form (every BN scale >= 1) must still round-trip tensor-exactly.
        var source = syntheticDescriptor()
        source.transformToReduceActivations()
        let result = try roundTrip(source, binaryFloats: true)
        assertDescriptorEqual(source, result, accuracy: 1e-7)
    }

    func testWriterRejectsMetadataEncoder() throws {
        var source = syntheticDescriptor()
        source.metaEncoderVersion = 1
        let url = temporaryURL(binaryFloats: true)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try ModelWriter.write(desc: source, to: url)) { error in
            XCTAssertEqual(
                String(describing: error),
                "Writing metaEncoderVersion > 0 is not supported"
            )
        }
    }

    private func loadFixture(_ name: String) throws -> ModelDesc {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"]
            ?? Self.defaultModelsDirectory
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw XCTSkip("KataGo model fixtures are missing; set KATAGO_MODELS_DIR (checked \(directory))")
        }
        let url = URL(fileURLWithPath: directory).appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("KataGo model fixture is missing: \(url.path)")
        }
        return try ModelDesc.loadFromFileMaybeGZipped(url.path)
    }

    private func roundTrip(_ source: ModelDesc, binaryFloats: Bool) throws -> ModelDesc {
        let url = temporaryURL(binaryFloats: binaryFloats)
        defer { try? FileManager.default.removeItem(at: url) }
        try ModelWriter.write(desc: source, to: url, binaryFloats: binaryFloats)
        return try ModelDesc.loadFromFileMaybeGZipped(url.path)
    }

    private func temporaryURL(binaryFloats: Bool) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("katago-writer-\(UUID().uuidString).\(binaryFloats ? "bin" : "txt").gz")
    }

    private func assertDescriptorEqual(
        _ lhs: ModelDesc,
        _ rhs: ModelDesc,
        accuracy: Float,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        assertRecursivelyEqual(lhs, rhs, path: "ModelDesc", accuracy: accuracy, file: file, line: line)
    }

    private func assertRecursivelyEqual(
        _ lhs: Any,
        _ rhs: Any,
        path: String,
        accuracy: Float,
        file: StaticString,
        line: UInt
    ) {
        if let lhs = lhs as? [Float], let rhs = rhs as? [Float] {
            XCTAssertEqual(lhs.count, rhs.count, "\(path).count", file: file, line: line)
            guard lhs.count == rhs.count else { return }
            for index in lhs.indices {
                XCTAssertEqual(lhs[index], rhs[index], accuracy: accuracy, "\(path)[\(index)]", file: file, line: line)
            }
            return
        }
        if let lhs = lhs as? Float, let rhs = rhs as? Float {
            XCTAssertEqual(lhs, rhs, accuracy: accuracy, path, file: file, line: line)
            return
        }
        if let lhs = lhs as? Double, let rhs = rhs as? Double {
            XCTAssertEqual(lhs, rhs, accuracy: Double(accuracy), path, file: file, line: line)
            return
        }
        if let lhs = lhs as? AnyHashable, let rhs = rhs as? AnyHashable {
            XCTAssertEqual(lhs, rhs, path, file: file, line: line)
            return
        }

        let lhsMirror = Mirror(reflecting: lhs)
        let rhsMirror = Mirror(reflecting: rhs)
        XCTAssertEqual(
            String(reflecting: type(of: lhs)),
            String(reflecting: type(of: rhs)),
            path,
            file: file,
            line: line
        )
        XCTAssertEqual(lhsMirror.displayStyle, rhsMirror.displayStyle, path, file: file, line: line)
        let lhsChildren = Array(lhsMirror.children)
        let rhsChildren = Array(rhsMirror.children)
        XCTAssertEqual(lhsChildren.count, rhsChildren.count, path, file: file, line: line)
        guard lhsChildren.count == rhsChildren.count else { return }
        if lhsChildren.isEmpty {
            XCTAssertEqual(String(reflecting: lhs), String(reflecting: rhs), path, file: file, line: line)
            return
        }
        for index in lhsChildren.indices {
            XCTAssertEqual(lhsChildren[index].label, rhsChildren[index].label, path, file: file, line: line)
            let label = lhsChildren[index].label ?? "[\(index)]"
            assertRecursivelyEqual(
                lhsChildren[index].value,
                rhsChildren[index].value,
                path: "\(path).\(label)",
                accuracy: accuracy,
                file: file,
                line: line
            )
        }
    }

    private func syntheticDescriptor() -> ModelDesc {
        let trunkBlock = ResidualBlockDesc(
            name: "block",
            preBN: batchNorm("block_pre", scale: 2),
            preActivation: activation("block_pre_act"),
            regularConv: conv("block_regular", weight: 0.25),
            midBN: batchNorm("block_mid", scale: 1.25),
            midActivation: activation("block_mid_act"),
            finalConv: conv("block_final", weight: -0.75)
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
            initialConv: conv("initial_conv", weight: 0.125),
            initialMatMul: matMul("initial_global", inChannels: 1, outChannels: 1, weights: [0.375]),
            sgfMetadataEncoder: nil,
            blocks: [.ordinary(trunkBlock)],
            trunkTipBN: batchNorm("trunk_tip", scale: 0.5),
            trunkTipRMSNorm: nil,
            trunkTipActivation: activation("trunk_tip_act")
        )
        let policy = PolicyHeadDesc(
            name: "policy",
            modelVersion: 8,
            policyOutChannels: 1,
            p1Conv: conv("policy_p1", weight: 0.2),
            g1Conv: conv("policy_g1", weight: 0.3),
            g1BN: batchNorm("policy_g1_bn", scale: 1.1),
            g1Activation: activation("policy_g1_act"),
            gpoolToBiasMul: matMul("policy_bias_mul", inChannels: 3, outChannels: 1, weights: [0.1, 0.2, 0.3]),
            p1BN: batchNorm("policy_p1_bn", scale: 1.2),
            p1Activation: activation("policy_p1_act"),
            p2Conv: conv("policy_p2", weight: 0.4),
            gpoolToPassMul: matMul("policy_pass", inChannels: 3, outChannels: 1, weights: [0.4, 0.5, 0.6]),
            gpoolToPassBias: nil,
            passActivation: nil,
            gpoolToPassMul2: nil
        )
        let value = ValueHeadDesc(
            name: "value",
            modelVersion: 8,
            v1Conv: conv("value_v1", weight: -0.2),
            v1BN: batchNorm("value_v1_bn", scale: 1.3),
            v1Activation: activation("value_v1_act"),
            v2Mul: matMul("value_v2", inChannels: 3, outChannels: 1, weights: [0.2, -0.3, 0.4]),
            v2Bias: bias("value_v2_bias", weights: [0.1]),
            v2Activation: activation("value_v2_act"),
            v3Mul: matMul("value_v3", inChannels: 1, outChannels: 3, weights: [0.1, 0.2, 0.3]),
            v3Bias: bias("value_v3_bias", weights: [0, 0.1, -0.1]),
            sv3Mul: matMul("value_score", inChannels: 1, outChannels: 4, weights: [0.1, 0.2, 0.3, 0.4]),
            sv3Bias: bias("value_score_bias", weights: [0, 0.1, 0.2, 0.3]),
            vOwnershipConv: conv("value_ownership", weight: 0.7)
        )
        return ModelDesc(
            name: "synthetic",
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

    private func conv(_ name: String, weight: Float) -> ConvLayerDesc {
        ConvLayerDesc(
            name: name,
            convYSize: 1,
            convXSize: 1,
            inChannels: 1,
            outChannels: 1,
            dilationY: 1,
            dilationX: 1,
            weights: [weight]
        )
    }

    private func batchNorm(_ name: String, scale: Float) -> BatchNormLayerDesc {
        BatchNormLayerDesc(
            name: name,
            numChannels: 1,
            epsilon: 1e-20,
            hasScale: true,
            hasBias: true,
            mean: [0],
            variance: [1],
            scale: [scale],
            bias: [0.125],
            mergedScale: [scale],
            mergedBias: [0.125]
        )
    }

    private func activation(_ name: String) -> ActivationLayerDesc {
        ActivationLayerDesc(name: name, activation: .relu)
    }

    private func matMul(
        _ name: String,
        inChannels: Int,
        outChannels: Int,
        weights: [Float]
    ) -> MatMulLayerDesc {
        MatMulLayerDesc(name: name, inChannels: inChannels, outChannels: outChannels, weights: weights)
    }

    private func bias(_ name: String, weights: [Float]) -> MatBiasLayerDesc {
        MatBiasLayerDesc(name: name, numChannels: weights.count, weights: weights)
    }
}
