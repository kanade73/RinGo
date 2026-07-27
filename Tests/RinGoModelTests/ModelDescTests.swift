import Foundation
import RinGoModel
import XCTest

final class ModelDescTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"

    private func modelURL(_ name: String) throws -> URL {
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
        return url
    }

    private func load(_ name: String) throws -> ModelDesc {
        try ModelDesc.loadFromFileMaybeGZipped(modelURL(name).path)
    }

    func testVersion8BinaryB6C96() throws {
        let model = try load("g170-b6c96-s175395328-d26788732.bin.gz")
        XCTAssertEqual(model.modelVersion, 8)
        XCTAssertEqual(model.numInputChannels, 22)
        XCTAssertEqual(model.numInputGlobalChannels, 19)
        XCTAssertEqual(model.trunk.numBlocks, 6)
        XCTAssertEqual(model.trunk.trunkNumChannels, 96)
        XCTAssertEqual(model.trunk.blocks.map(\.kind), [
            .regular, .regular, .globalPooling, .regular, .globalPooling, .regular,
        ])
        XCTAssertEqual(model.trunk.gpoolNumChannels, 32)

        // Cross-check the channel fields in g170-b6c96-s175395328-d26788732-info.txt.
        XCTAssertEqual(model.trunk.midNumChannels, 96)
        XCTAssertEqual(model.trunk.regularNumChannels, 64)
        XCTAssertEqual(model.policyHead.p1Conv.outChannels, 32)
        XCTAssertEqual(model.policyHead.g1Conv.outChannels, 32)
        XCTAssertEqual(model.valueHead.v1Conv.outChannels, 32)
        XCTAssertEqual(model.valueHead.v2Mul.inChannels, 96)
        XCTAssertEqual(model.valueHead.v2Mul.outChannels, 64)
        XCTAssertEqual(model.valueHead.v3Mul.inChannels, 64)
        XCTAssertEqual(model.postProcessParams.tdScoreMultiplier, 20)
        XCTAssertEqual(model.postProcessParams.outputScaleMultiplier, 1)
        XCTAssertEqual(model.numParameters, 1_027_911)
    }

    func testVersion8TextMatchesBinary() throws {
        let binary = try load("g170-b6c96-s175395328-d26788732.bin.gz")
        let text = try load("g170-b6c96-s175395328-d26788732.txt.gz")
        XCTAssertEqual(text.modelVersion, binary.modelVersion)
        XCTAssertEqual(text.numParameters, binary.numParameters)
        XCTAssertEqual(text.trunk.blocks.map(\.kind), binary.trunk.blocks.map(\.kind))
        assertSamplesEqual(text.trunk.initialConv.weights, binary.trunk.initialConv.weights)
        assertSamplesEqual(text.trunk.initialMatMul.weights, binary.trunk.initialMatMul.weights)
        assertSamplesEqual(text.policyHead.p2Conv.weights, binary.policyHead.p2Conv.weights)
        assertSamplesEqual(text.valueHead.v2Mul.weights, binary.valueHead.v2Mul.weights)
    }

    func testVersion8BinaryB10C128() throws {
        let model = try load("g170e-b10c128-s1141046784-d204142634.bin.gz")
        XCTAssertEqual(model.modelVersion, 8)
        XCTAssertEqual(model.numInputChannels, 22)
        XCTAssertEqual(model.numInputGlobalChannels, 19)
        XCTAssertEqual(model.trunk.numBlocks, 10)
        XCTAssertEqual(model.trunk.trunkNumChannels, 128)
        XCTAssertFalse(model.hasAnyTransformerBlocks)
    }

    func testVersion17NestedBottleneckTransformerSiluRMSNorm() throws {
        let model = try load("b4c256h4nbttflrs-fson-silu-rsnh.bin.gz")
        XCTAssertEqual(model.modelVersion, 17)
        XCTAssertEqual(model.trunk.trunkNumChannels, 256)
        XCTAssertEqual(model.trunk.trunkNormKind, .rmsNorm)
        XCTAssertEqual(model.trunk.trunkTipActivation.activation, .silu)
        XCTAssertTrue(model.trunk.blocks.contains { $0.kind == .nestedBottleneck })
        XCTAssertTrue(model.hasAnyTransformerBlocks)
        let attention = try XCTUnwrap(allAttentionBlocks(in: model.trunk.blocks).first)
        XCTAssertTrue(attention.learnableRope)
        XCTAssertEqual(attention.ropeNumKVHeads, 4)
        XCTAssertTrue(try XCTUnwrap(allFFNBlocks(in: model.trunk.blocks).first).useSwiGLU)
        XCTAssertTrue(try XCTUnwrap(model.trunk.trunkTipRMSNorm).spatial)
        XCTAssertEqual(model.postProcessParams.shorttermScoreErrorMultiplier, 150)
        XCTAssertEqual(model.postProcessParams.outputScaleMultiplier, 1)
        XCTAssertEqual(model.numParameters, 2_067_785)
    }

    func testVersion17TransformerVariants() throws {
        let groupedQuery = try load("b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz")
        XCTAssertEqual(groupedQuery.modelVersion, 17)
        let gqaAttention = try XCTUnwrap(allAttentionBlocks(in: groupedQuery.trunk.blocks).first)
        XCTAssertEqual(gqaAttention.numHeads, 6)
        XCTAssertEqual(gqaAttention.numKVHeads, 3)
        XCTAssertEqual(gqaAttention.qHeadDim, 32)
        XCTAssertEqual(gqaAttention.vHeadDim, 16)
        XCTAssertTrue(gqaAttention.useRope)
        XCTAssertTrue(gqaAttention.learnableRope)
        // Conv weights from the first @BIN@ block, rearranged from the file's y,x,ic,oc order
        // into the struct's OIHW order (mirroring $KG/cpp/neuralnet/desc.cpp): entry i here is
        // file float ((y*convX + x)*inC + ic)*outC for oc=0, ic=i/9, (y,x) = row-major i%9.
        // Cross-checked against the raw little-endian file bytes independently of the parser.
        // This model's transformer trunk does not activation-rescale these layers after parsing.
        XCTAssertEqual(Array(groupedQuery.trunk.initialConv.weights.prefix(12)), [
            -0.110548824, -0.055038817, 0.093369566, 0.008044394,
            0.043376364, -0.03865581, -0.052445192, -0.035625283,
            0.0468643, -0.07068716, -0.02068068, -0.10733972,
        ])
        XCTAssertEqual(Array(groupedQuery.trunk.initialMatMul.weights.prefix(8)), [
            -0.07722447, 0.03106677, -0.010591503, -0.033845715,
            0.001235994, -0.022942625, -0.0045179557, -0.15196836,
        ])
        let ropeTables = try gqaAttention.computeRopeCosSin(nnXLen: 3, nnYLen: 2, paddedNNXYLen: 8)
        XCTAssertEqual(ropeTables.cos.count, 3 * 16 * 8)
        XCTAssertEqual(ropeTables.sin.count, ropeTables.cos.count)
        XCTAssertEqual(groupedQuery.numParameters, 851_881)

        let channelNorm = try load("b7c96h3tfrs-test5-cnorm.bin.gz")
        XCTAssertEqual(channelNorm.modelVersion, 17)
        XCTAssertEqual(channelNorm.trunk.trunkNormKind, .rmsNorm)
        let norm = try XCTUnwrap(channelNorm.trunk.trunkTipRMSNorm)
        XCTAssertFalse(norm.spatial)
        XCTAssertEqual(norm.cgroupSize, 0)
        XCTAssertFalse(allAttentionBlocks(in: channelNorm.trunk.blocks).isEmpty)
        XCTAssertEqual(channelNorm.numParameters, 818_953)
    }

    func testGarbageAndTruncatedDataThrowDescriptiveErrors() throws {
        XCTAssertThrowsError(try ModelDesc.parse(data: Data("garbage".utf8), binaryFloats: false)) { error in
            XCTAssertTrue(String(describing: error).contains("could not read token"))
        }

        let validURL = try modelURL("g170-b6c96-s175395328-d26788732.bin.gz")
        let compressed = try Data(contentsOf: validURL)
        let truncatedURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("katago-model-truncated-\(UUID().uuidString).bin.gz")
        try compressed.prefix(256).write(to: truncatedURL)
        defer { try? FileManager.default.removeItem(at: truncatedURL) }
        XCTAssertThrowsError(try ModelDesc.loadFromFileMaybeGZipped(truncatedURL.path)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("Error loading or parsing model file"), message)
        }
    }

    private func assertSamplesEqual(
        _ lhs: [Float],
        _ rhs: [Float],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.count, rhs.count, file: file, line: line)
        guard !lhs.isEmpty, lhs.count == rhs.count else { return }
        for index in Set([0, lhs.count / 7, lhs.count / 2, lhs.count - 1]) {
            XCTAssertEqual(lhs[index], rhs[index], accuracy: 1e-6, file: file, line: line)
        }
    }

    private func allAttentionBlocks(in blocks: [BlockDesc]) -> [TransformerAttentionDesc] {
        var result = [TransformerAttentionDesc]()
        for block in blocks {
            switch block {
            case let .transformerAttention(attention): result.append(attention)
            case let .nestedBottleneck(nested): result.append(contentsOf: allAttentionBlocks(in: nested.blocks))
            case .ordinary, .globalPooling, .transformerFFN: break
            }
        }
        return result
    }

    private func allFFNBlocks(in blocks: [BlockDesc]) -> [TransformerFFNDesc] {
        var result = [TransformerFFNDesc]()
        for block in blocks {
            switch block {
            case let .transformerFFN(ffn): result.append(ffn)
            case let .nestedBottleneck(nested): result.append(contentsOf: allFFNBlocks(in: nested.blocks))
            case .ordinary, .globalPooling, .transformerAttention: break
            }
        }
        return result
    }
}
