import Foundation
import RinGoModel
@testable import RinGoTrain
import XCTest

final class ArchitectureTests: XCTestCase {
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"

    func testGeneratedB6C96TopologyMatchesRealModel() throws {
        guard let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"] else {
            throw XCTSkip("Set KATAGO_MODELS_DIR to run the real-model topology self-test")
        }
        let path = URL(fileURLWithPath: directory).appendingPathComponent(Self.modelFile).path
        guard FileManager.default.fileExists(atPath: path) else { throw XCTSkip("Model fixture missing: \(path)") }
        let expected = try ModelDesc.loadFromFileMaybeGZipped(path)
        assertTopologyEqual(RinGoArchitecture.b6c96(), expected)
    }

    func testGeneratedB6C96HasExpectedParameterCount() {
        XCTAssertEqual(RinGoArchitecture.b6c96().numParameters, 1_027_911)
    }

    /// WP-5 parameter-count pins for the two larger students. These are load-bearing: they lock the
    /// exact head-width scaling (head = round(C/3), wide = 2*head) and the two-gpool-block pattern
    /// so a future refactor of `build` that silently changes the topology fails here.
    func testGeneratedB10C128HasExpectedParameterCount() {
        XCTAssertEqual(RinGoArchitecture.b10c128().numParameters, 2_987_754)
    }

    func testGeneratedB15C192HasExpectedParameterCount() {
        XCTAssertEqual(RinGoArchitecture.b15c192().numParameters, 9_974_471)
    }

    func testGeneratedB20C256HasExpectedParameterCount() {
        XCTAssertEqual(RinGoArchitecture.b20c256().numParameters, 23_572_222)
    }

    func testGeneratedB24C320HasExpectedParameterCount() {
        XCTAssertEqual(RinGoArchitecture.b24c320().numParameters, 44_182_954)
    }

    func testGeneratedB28C384HasExpectedParameterCount() {
        XCTAssertEqual(RinGoArchitecture.b28c384().numParameters, 74_178_567)
    }

    /// `make(_:)` is the path `TrainCommand -arch` takes; it must agree with the named builders.
    func testVariantFactoryMatchesNamedBuilders() {
        XCTAssertEqual(RinGoArchitecture.make(.b6c96).numParameters, RinGoArchitecture.b6c96().numParameters)
        XCTAssertEqual(RinGoArchitecture.make(.b10c128).numParameters, RinGoArchitecture.b10c128().numParameters)
        XCTAssertEqual(RinGoArchitecture.make(.b15c192).numParameters, RinGoArchitecture.b15c192().numParameters)
        XCTAssertEqual(RinGoArchitecture.make(.b20c256).numParameters, RinGoArchitecture.b20c256().numParameters)
        XCTAssertEqual(RinGoArchitecture.make(.b24c320).numParameters, RinGoArchitecture.b24c320().numParameters)
        XCTAssertEqual(RinGoArchitecture.make(.b28c384).numParameters, RinGoArchitecture.b28c384().numParameters)
    }

    /// Every supported variant must stay inside `TrainableNetwork.validate`'s block set: model v8,
    /// standard trunk norm, no RMSNorm/transformer/nested-bottleneck/metadata-encoder, and exactly
    /// two global-pooling residual blocks with the rest ordinary (the audit's do-not-support list).
    func testAllVariantsUseOnlyOrdinaryAndGpoolBlocks() {
        for variant in RinGoArchitecture.Variant.allCases {
            let desc = RinGoArchitecture.make(variant)
            XCTAssertEqual(desc.modelVersion, 8, "\(variant)")
            XCTAssertEqual(desc.trunk.numBlocks, variant.numBlocks, "\(variant)")
            XCTAssertEqual(desc.trunk.blocks.count, variant.numBlocks, "\(variant)")
            XCTAssertEqual(desc.trunk.trunkNumChannels, variant.channels, "\(variant)")
            XCTAssertEqual(desc.trunk.trunkNormKind, .standard, "\(variant)")
            XCTAssertNil(desc.trunk.trunkTipRMSNorm, "\(variant)")
            XCTAssertFalse(desc.hasAnyTransformerBlocks, "\(variant)")
            let kinds = desc.trunk.blocks.map(\.kind)
            XCTAssertEqual(kinds.count(where: { $0 == .globalPooling }), 2, "\(variant): exactly two gpool blocks")
            XCTAssertEqual(
                kinds.count(where: { $0 == .regular }), variant.numBlocks - 2,
                "\(variant): all other blocks ordinary"
            )
            XCTAssertTrue(kinds.allSatisfy { $0 == .regular || $0 == .globalPooling }, "\(variant)")
        }
    }

    /// The b6c96 gpool blocks sit at 1-based indices 3 and 5; scaling those fractions by depth must
    /// keep two well-separated interior positions for the larger trunks.
    func testGpoolPositionsScaleWithDepth() {
        XCTAssertEqual(RinGoArchitecture.gpoolPositions(numBlocks: 6), [3, 5])
        XCTAssertEqual(RinGoArchitecture.gpoolPositions(numBlocks: 10), [5, 8])
        XCTAssertEqual(RinGoArchitecture.gpoolPositions(numBlocks: 15), [8, 13])
        XCTAssertEqual(RinGoArchitecture.gpoolPositions(numBlocks: 20), [10, 17])
        XCTAssertEqual(RinGoArchitecture.gpoolPositions(numBlocks: 24), [12, 20])
        XCTAssertEqual(RinGoArchitecture.gpoolPositions(numBlocks: 28), [14, 23])
    }

    private func assertTopologyEqual(
        _ lhs: ModelDesc, _ rhs: ModelDesc,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(lhs.modelVersion, rhs.modelVersion, file: file, line: line)
        XCTAssertEqual(lhs.numInputChannels, rhs.numInputChannels, file: file, line: line)
        XCTAssertEqual(lhs.numInputGlobalChannels, rhs.numInputGlobalChannels, file: file, line: line)
        XCTAssertEqual(lhs.numPolicyChannels, rhs.numPolicyChannels, file: file, line: line)
        XCTAssertEqual(lhs.numValueChannels, rhs.numValueChannels, file: file, line: line)
        XCTAssertEqual(lhs.numScoreValueChannels, rhs.numScoreValueChannels, file: file, line: line)
        XCTAssertEqual(lhs.numOwnershipChannels, rhs.numOwnershipChannels, file: file, line: line)
        XCTAssertEqual(lhs.trunk.name, rhs.trunk.name, file: file, line: line)
        XCTAssertEqual(lhs.trunk.modelVersion, rhs.trunk.modelVersion, file: file, line: line)
        XCTAssertEqual(lhs.trunk.metaEncoderVersion, rhs.trunk.metaEncoderVersion, file: file, line: line)
        XCTAssertEqual(lhs.trunk.trunkNormKind.rawValue, rhs.trunk.trunkNormKind.rawValue, file: file, line: line)
        XCTAssertEqual(lhs.trunk.numBlocks, rhs.trunk.numBlocks, file: file, line: line)
        XCTAssertEqual(lhs.trunk.trunkNumChannels, rhs.trunk.trunkNumChannels, file: file, line: line)
        XCTAssertEqual(lhs.trunk.midNumChannels, rhs.trunk.midNumChannels, file: file, line: line)
        XCTAssertEqual(lhs.trunk.regularNumChannels, rhs.trunk.regularNumChannels, file: file, line: line)
        XCTAssertEqual(lhs.trunk.gpoolNumChannels, rhs.trunk.gpoolNumChannels, file: file, line: line)
        assertConv(lhs.trunk.initialConv, rhs.trunk.initialConv, file: file, line: line)
        assertMatmul(lhs.trunk.initialMatMul, rhs.trunk.initialMatMul, file: file, line: line)
        XCTAssertEqual(lhs.trunk.blocks.map(\.kind), rhs.trunk.blocks.map(\.kind), file: file, line: line)
        for (left, right) in zip(lhs.trunk.blocks, rhs.trunk.blocks) {
            switch (left, right) {
            case let (.ordinary(left), .ordinary(right)):
                XCTAssertEqual(left.name, right.name, file: file, line: line)
                assertNorm(left.preBN, right.preBN, file: file, line: line)
                assertActivation(left.preActivation, right.preActivation, file: file, line: line)
                assertConv(left.regularConv, right.regularConv, file: file, line: line)
                assertNorm(left.midBN, right.midBN, file: file, line: line)
                assertActivation(left.midActivation, right.midActivation, file: file, line: line)
                assertConv(left.finalConv, right.finalConv, file: file, line: line)
            case let (.globalPooling(left), .globalPooling(right)):
                XCTAssertEqual(left.name, right.name, file: file, line: line)
                assertNorm(left.preBN, right.preBN, file: file, line: line)
                assertActivation(left.preActivation, right.preActivation, file: file, line: line)
                assertConv(left.regularConv, right.regularConv, file: file, line: line)
                assertConv(left.gpoolConv, right.gpoolConv, file: file, line: line)
                assertNorm(left.gpoolBN, right.gpoolBN, file: file, line: line)
                assertActivation(left.gpoolActivation, right.gpoolActivation, file: file, line: line)
                assertMatmul(left.gpoolToBiasMul, right.gpoolToBiasMul, file: file, line: line)
                assertNorm(left.midBN, right.midBN, file: file, line: line)
                assertActivation(left.midActivation, right.midActivation, file: file, line: line)
                assertConv(left.finalConv, right.finalConv, file: file, line: line)
            default: XCTFail("Mismatched block cases", file: file, line: line)
            }
        }
        assertNorm(lhs.trunk.trunkTipBN!, rhs.trunk.trunkTipBN!, file: file, line: line)
        assertActivation(lhs.trunk.trunkTipActivation, rhs.trunk.trunkTipActivation, file: file, line: line)
        XCTAssertEqual(lhs.policyHead.name, rhs.policyHead.name, file: file, line: line)
        XCTAssertEqual(lhs.policyHead.modelVersion, rhs.policyHead.modelVersion, file: file, line: line)
        XCTAssertEqual(lhs.policyHead.policyOutChannels, rhs.policyHead.policyOutChannels, file: file, line: line)
        assertConv(lhs.policyHead.p1Conv, rhs.policyHead.p1Conv, file: file, line: line)
        assertConv(lhs.policyHead.g1Conv, rhs.policyHead.g1Conv, file: file, line: line)
        assertNorm(lhs.policyHead.g1BN, rhs.policyHead.g1BN, file: file, line: line)
        assertMatmul(lhs.policyHead.gpoolToBiasMul, rhs.policyHead.gpoolToBiasMul, file: file, line: line)
        assertNorm(lhs.policyHead.p1BN, rhs.policyHead.p1BN, file: file, line: line)
        assertConv(lhs.policyHead.p2Conv, rhs.policyHead.p2Conv, file: file, line: line)
        assertMatmul(lhs.policyHead.gpoolToPassMul, rhs.policyHead.gpoolToPassMul, file: file, line: line)
        XCTAssertNil(lhs.policyHead.gpoolToPassBias, file: file, line: line)
        XCTAssertNil(rhs.policyHead.gpoolToPassBias, file: file, line: line)
        XCTAssertEqual(lhs.valueHead.name, rhs.valueHead.name, file: file, line: line)
        XCTAssertEqual(lhs.valueHead.modelVersion, rhs.valueHead.modelVersion, file: file, line: line)
        assertConv(lhs.valueHead.v1Conv, rhs.valueHead.v1Conv, file: file, line: line)
        assertNorm(lhs.valueHead.v1BN, rhs.valueHead.v1BN, file: file, line: line)
        assertMatmul(lhs.valueHead.v2Mul, rhs.valueHead.v2Mul, file: file, line: line)
        assertBias(lhs.valueHead.v2Bias, rhs.valueHead.v2Bias, file: file, line: line)
        assertMatmul(lhs.valueHead.v3Mul, rhs.valueHead.v3Mul, file: file, line: line)
        assertBias(lhs.valueHead.v3Bias, rhs.valueHead.v3Bias, file: file, line: line)
        assertMatmul(lhs.valueHead.sv3Mul, rhs.valueHead.sv3Mul, file: file, line: line)
        assertBias(lhs.valueHead.sv3Bias, rhs.valueHead.sv3Bias, file: file, line: line)
        assertConv(lhs.valueHead.vOwnershipConv, rhs.valueHead.vOwnershipConv, file: file, line: line)
    }

    private func assertConv(
        _ lhs: ConvLayerDesc, _ rhs: ConvLayerDesc,
        file: StaticString, line: UInt
    ) {
        XCTAssertEqual(lhs.name, rhs.name, file: file, line: line)
        XCTAssertEqual(
            [
                lhs.convYSize, lhs.convXSize, lhs.inChannels, lhs.outChannels,
                lhs.dilationY, lhs.dilationX,
            ],
            [
                rhs.convYSize, rhs.convXSize, rhs.inChannels, rhs.outChannels,
                rhs.dilationY, rhs.dilationX,
            ], file: file, line: line
        )
    }

    private func assertMatmul(
        _ lhs: MatMulLayerDesc, _ rhs: MatMulLayerDesc,
        file: StaticString, line: UInt
    ) {
        XCTAssertEqual(lhs.name, rhs.name, file: file, line: line)
        XCTAssertEqual([lhs.inChannels, lhs.outChannels], [rhs.inChannels, rhs.outChannels],
                       file: file, line: line)
    }

    private func assertBias(
        _ lhs: MatBiasLayerDesc, _ rhs: MatBiasLayerDesc,
        file: StaticString, line: UInt
    ) {
        XCTAssertEqual(lhs.name, rhs.name, file: file, line: line)
        XCTAssertEqual(lhs.numChannels, rhs.numChannels, file: file, line: line)
    }

    private func assertNorm(
        _ lhs: BatchNormLayerDesc, _ rhs: BatchNormLayerDesc,
        file: StaticString, line: UInt
    ) {
        XCTAssertEqual(lhs.name, rhs.name, file: file, line: line)
        XCTAssertEqual(lhs.numChannels, rhs.numChannels, file: file, line: line)
        XCTAssertEqual(lhs.epsilon, rhs.epsilon, file: file, line: line)
        XCTAssertEqual(lhs.hasScale, rhs.hasScale, file: file, line: line)
        XCTAssertEqual(lhs.hasBias, rhs.hasBias, file: file, line: line)
    }

    private func assertActivation(
        _ lhs: ActivationLayerDesc, _ rhs: ActivationLayerDesc,
        file: StaticString, line: UInt
    ) {
        XCTAssertEqual(lhs.name, rhs.name, file: file, line: line)
        XCTAssertEqual(lhs.activation.rawValue, rhs.activation.rawValue, file: file, line: line)
    }
}
