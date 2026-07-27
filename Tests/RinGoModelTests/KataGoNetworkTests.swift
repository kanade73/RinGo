import Foundation
import MLX
@testable import RinGoModel
import XCTest

final class KataGoNetworkTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"

    private struct Golden: Decodable {
        let modelVersion: Int
        let nnLen: Int
        let optimism: Float
        let spatialNHWC: [Float]
        let global: [Float]
        let policy: [Float]
        let valueLogits: [Float]
        let scoreValues: [Float]
        let ownership: [Float]
    }

    private struct ModelCase {
        let model: String
        let fixture: String
    }

    private let modelCases = [
        ModelCase(model: "g170-b6c96-s175395328-d26788732.bin.gz", fixture: "b6c96"),
        ModelCase(model: "g170e-b10c128-s1141046784-d204142634.bin.gz", fixture: "b10c128"),
        ModelCase(model: "b4c256h4nbttflrs-fson-silu-rsnh.bin.gz", fixture: "b4c256-transformer"),
        ModelCase(model: "b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz", fixture: "b7c96-gqa"),
        ModelCase(model: "b7c96h3tfrs-test5-cnorm.bin.gz", fixture: "b7c96-cnorm"),
    ]

    func testBatchNormActivationAndMaskSemantics() {
        let input = MLXArray([1 as Float, 2, 3, 4], [1, 2, 2, 1])
        let mask = MLXArray([1 as Float, 0, 1, 0], [1, 2, 2, 1])
        let output = kataGoBatchNorm(
            input,
            scale: MLXArray([2 as Float]),
            bias: MLXArray([-3 as Float]),
            activation: .relu,
            mask: mask
        )
        assertClose(output, [0, 0, 3, 0])
    }

    func testMaskedGPoolUsesEigenPaddedMaximum() {
        let input = MLXArray([0.25 as Float, 0, 0.75, 0, -0.2, 0], [1, 2, 3, 1])
        let mask = MLXArray([1 as Float, 0, 1, 0, 1, 0], [1, 2, 3, 1])
        let maskSum = mask.asType(.float32).sum(axes: [1, 2], keepDims: true)
        let output = kataGoGPool(input, mask: mask, maskSum: maskSum)
        let mean = Float(0.8 / 3.0)
        let scale = Float(Foundation.sqrt(3.0) * 0.1 - 1.4)
        assertClose(output, [mean, mean * scale, 0.75], accuracy: 1e-6)
    }

    func testPointwiseActivations() {
        let input = MLXArray([-2 as Float, -0.25, 0, 1.5])
        let values = [-2 as Float, -0.25, 0, 1.5]
        assertClose(kataGoActivation(input, .silu), values.map { $0 / (1 + Foundation.exp(-$0)) })
        assertClose(
            kataGoActivation(input, .mish),
            values.map { $0 * Foundation.tanh(Foundation.log1p(Foundation.exp($0))) }
        )
        assertClose(
            kataGoActivation(input, .mishScale8),
            values.map { $0 * Foundation.tanh(Foundation.log1p(Foundation.exp(8 * $0))) }
        )
    }

    func testSwiGLUAppliesSiluToLinearOneNotGate() {
        let value = MLXArray([-1 as Float, 2])
        let gate = MLXArray([3 as Float, -4])
        assertClose(
            kataGoSwiGLU(value, gate: gate),
            [-3 / (1 + Foundation.exp(1)), -8 / (1 + Foundation.exp(-2))]
        )
    }

    func testRoPERotatesAdjacentPairs() {
        let input = MLXArray([1 as Float, 2, 3, 4], [1, 1, 1, 4])
        let cosine = MLXArray([0 as Float, 0, 1, 1], [1, 1, 1, 4])
        let sine = MLXArray([1 as Float, 1, 0, 0], [1, 1, 1, 4])
        assertClose(kataGoApplyRoPE(input, cosine: cosine, sine: sine), [-2, 1, 3, 4])
    }

    func testOracleParityAllModels() throws {
        for modelCase in modelCases {
            let result = try evaluate(modelCase, precision: .float32)
            assertGolden(result.outputs, golden: result.golden, file: modelCase.fixture)
        }
    }

    /// Debug-only bisection tool (off by default): prints "DEBUG SWIFT ..." per-block trunk
    /// (and, for transformer blocks, attention/FFN internal) summary lines to stderr, in the
    /// same format as the reference Eigen backend's `.tools/dump_nn_debug` stderr output, so
    /// the two can be diffed line-by-line to find the first point of numerical divergence.
    /// Usage: `KATAGO_MLX_DEBUG_DUMP=<fixture> swift test --filter testDebugDumpTrunkSummaries`
    /// where `<fixture>` is one of the `modelCases` fixture names (e.g. `b6c96`).
    func testDebugDumpTrunkSummaries() throws {
        guard let fixture = ProcessInfo.processInfo.environment["KATAGO_MLX_DEBUG_DUMP"] else {
            throw XCTSkip(
                "Set KATAGO_MLX_DEBUG_DUMP=<fixture> (e.g. b6c96) to dump per-block trunk summaries "
                    + "to stderr for bisection against `.tools/dump_nn_debug`."
            )
        }
        guard let modelCase = modelCases.first(where: { $0.fixture == fixture }) else {
            XCTFail("Unknown KATAGO_MLX_DEBUG_DUMP fixture '\(fixture)'; known: \(modelCases.map(\.fixture))")
            return
        }
        let golden = try loadGolden(modelCase.fixture)
        let model = try loadModel(modelCase.model)
        let network = try KataGoNetwork(desc: model, nnXLen: golden.nnLen, nnYLen: golden.nnLen, precision: .float32)
        let spatial = MLXArray(golden.spatialNHWC, [1, golden.nnLen, golden.nnLen, model.numInputChannels])
        let global = MLXArray(golden.global, [1, model.numInputGlobalChannels])
        _ = network.forward(spatial: spatial, global: global, debugSink: KataGoNetwork.stderrDebugSink())
    }

    func testPolicyOptimismOneUsesSecondPolicyChannel() throws {
        let modelCase = ModelCase(
            model: "b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz",
            fixture: "b7c96-gqa-optimism1"
        )
        let result = try evaluate(modelCase, precision: .float32)
        assertGolden(result.outputs, golden: result.golden, file: modelCase.fixture)
    }

    func testFloat16Smoke() throws {
        let modelCase = modelCases[0]
        let fp32 = try evaluate(modelCase, precision: .float32)
        let fp16 = try evaluate(modelCase, precision: .float16)
        XCTAssertLessThan(maxPolicyDiff(fp16.outputs, fp32.outputs, optimism: 0), 0.5)
        XCTAssertLessThan(maxAbsDiff(array(fp16.outputs.valueLogits), array(fp32.outputs.valueLogits)), 0.05)
    }

    func testDeterministicForward() throws {
        let modelCase = modelCases[0]
        let golden = try loadGolden(modelCase.fixture)
        let model = try loadModel(modelCase.model)
        let network = try KataGoNetwork(desc: model, nnXLen: golden.nnLen, nnYLen: golden.nnLen, precision: .float32)
        let spatial = MLXArray(golden.spatialNHWC, [1, golden.nnLen, golden.nnLen, model.numInputChannels])
        let global = MLXArray(golden.global, [1, model.numInputGlobalChannels])
        let first = network.forward(spatial: spatial, global: global)
        let second = network.forward(spatial: spatial, global: global)
        eval(outputArrays(first) + outputArrays(second))
        for (lhs, rhs) in zip(outputArrays(first), outputArrays(second)) {
            XCTAssertEqual(array(lhs), array(rhs))
        }
    }

    private func evaluate(
        _ modelCase: ModelCase,
        precision: KataGoNetwork.Precision
    ) throws -> (outputs: ModelRawOutputs, golden: Golden) {
        let golden = try loadGolden(modelCase.fixture)
        let model = try loadModel(modelCase.model)
        XCTAssertEqual(model.modelVersion, golden.modelVersion)
        let network = try KataGoNetwork(desc: model, nnXLen: golden.nnLen, nnYLen: golden.nnLen, precision: precision)
        let spatial = MLXArray(golden.spatialNHWC, [1, golden.nnLen, golden.nnLen, model.numInputChannels])
        let global = MLXArray(golden.global, [1, model.numInputGlobalChannels])
        let outputs = network.forward(spatial: spatial, global: global)
        eval(outputArrays(outputs))
        return (outputs, golden)
    }

    private func assertGolden(
        _ outputs: ModelRawOutputs,
        golden: Golden,
        file: String,
        line: UInt = #line
    ) {
        let channels = outputs.policySpatialLogits.dim(-1)
        let spatial = array(outputs.policySpatialLogits)
        let pass = array(outputs.policyPassLogits)
        let policy = (0 ..< golden.nnLen * golden.nnLen).map { position in
            let normal = spatial[position * channels]
            let optimistic = channels >= 2 ? spatial[position * channels + 1] : normal
            return normal + (optimistic - normal) * golden.optimism
        } + [pass[0] + ((channels >= 2 ? pass[1] : pass[0]) - pass[0]) * golden.optimism]
        XCTAssertLessThanOrEqual(maxAbsDiff(policy, golden.policy), 2e-4, file, line: line)
        XCTAssertLessThanOrEqual(maxAbsDiff(array(outputs.valueLogits), golden.valueLogits), 2e-4, file, line: line)
        XCTAssertLessThanOrEqual(maxAbsDiff(array(outputs.scoreValues), golden.scoreValues), 1e-3, file, line: line)
        XCTAssertLessThanOrEqual(maxAbsDiff(array(outputs.ownership), golden.ownership), 2e-4, file, line: line)
    }

    private func maxPolicyDiff(
        _ lhs: ModelRawOutputs,
        _ rhs: ModelRawOutputs,
        optimism: Float
    ) -> Float {
        let channels = lhs.policySpatialLogits.dim(-1)
        func flattened(_ output: ModelRawOutputs) -> [Float] {
            let spatial = array(output.policySpatialLogits)
            let pass = array(output.policyPassLogits)
            let optimisticChannel = min(1, channels - 1)
            var result = [Float]()
            result.reserveCapacity(362)
            for position in 0 ..< 361 {
                let offset = position * channels
                let normal = spatial[offset]
                let optimistic = spatial[offset + optimisticChannel]
                result.append(normal + (optimistic - normal) * optimism)
            }
            result.append(pass[0] + (pass[optimisticChannel] - pass[0]) * optimism)
            return result
        }
        return maxAbsDiff(flattened(lhs), flattened(rhs))
    }

    private func loadModel(_ name: String) throws -> ModelDesc {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"]
            ?? Self.defaultModelsDirectory
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw XCTSkip("KataGo model fixtures are missing; set KATAGO_MODELS_DIR (checked \(directory))")
        }
        let path = URL(fileURLWithPath: directory).appendingPathComponent(name).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("KataGo model fixture is missing: \(path)")
        }
        return try ModelDesc.loadFromFileMaybeGZipped(path)
    }

    private func loadGolden(_ name: String) throws -> Golden {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json"))
        return try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }

    private func outputArrays(_ output: ModelRawOutputs) -> [MLXArray] {
        [
            output.policySpatialLogits,
            output.policyPassLogits,
            output.valueLogits,
            output.scoreValues,
            output.ownership,
        ]
    }

    private func array(_ value: MLXArray) -> [Float] {
        value.asArray(Float.self)
    }

    private func assertClose(
        _ actual: MLXArray,
        _ expected: [Float],
        accuracy: Float = 1e-5,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        eval(actual)
        XCTAssertEqual(array(actual).count, expected.count, file: file, line: line)
        XCTAssertLessThanOrEqual(maxAbsDiff(array(actual), expected), accuracy, file: file, line: line)
    }

    private func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        return zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
    }
}
