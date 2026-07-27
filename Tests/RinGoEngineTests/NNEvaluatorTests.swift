import Foundation
import MLX
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// Correctness tests for `NNEvaluator` (Sources/RinGoEngine/NNEvaluator.swift). These check
/// that batching/padding/bucketing/compiling never changes numerics relative to a direct,
/// non-batched `KataGoNetwork.forward` + `NNOutputPostprocessor.process` call (the same
/// primitives `ringo rawnn` uses), that fp16 stays within design.md's fp16 tolerances of
/// fp32, and that the actor behaves correctly under concurrent submission.
///
/// Uses the b6c96 test model (same fixture as `KataGoNetworkTests`/`NNOutputPostprocessorTests`)
/// at 19x19; positions are deterministic pseudo-random legal playouts (see `SplitMix64` /
/// `generatePositions`) rather than SGF fixtures, so an arbitrary number of distinct positions
/// (up to 32+, to cover every bucket) is cheap to generate.
final class NNEvaluatorTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"
    private static let nnLen = 19

    /// 40 distinct, legal, deterministically-generated 19x19 positions of increasing move
    /// number, shared read-only across tests. `Board`/`BoardHistory` are reference types but
    /// nothing in this file mutates a position once generated.
    private static let positionPool = generatePositions(count: 40, seed: 0xC0FF_EE12_3456_789A)

    // MARK: - Evaluator-vs-direct parity (padded, non-power-of-two, batched vs single)

    /// Design.md discipline point: batching/padding/bucketing must not change numerics. 5 is
    /// deliberately not a power of two, so this exercises zero-padding up to bucket 8.
    func testEvaluatorMatchesDirectForwardNonPowerOfTwoBatch() async throws {
        try await assertEvaluatorMatchesDirect(counts: [5], precision: .float32, tolerance: 1e-5)
    }

    /// Request counts 1, 2, 3, 8, 20, 32 exercise buckets {1, 2, 4, 8, 32} respectively (16 is
    /// never touched by this list, and is never compiled as a result) with `maxBatchSize: 32`.
    func testCompileBucketCoverage() async throws {
        try await assertEvaluatorMatchesDirect(counts: [1, 2, 3, 8, 20, 32], precision: .float32, tolerance: 1e-5)
    }

    private func assertEvaluatorMatchesDirect(
        counts: [Int],
        precision: KataGoNetwork.Precision,
        tolerance: Float,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let evaluator = try makeEvaluator(precision: precision, maxBatchSize: 32)
        let (network, desc) = try makeDirectNetwork(precision: precision)
        // whiteScoreMean/whiteLead are raw model output multiplied by ModelPostProcessParams'
        // scoreMeanMultiplier/leadMultiplier (20 for this fixture): a real, tiny (~1e-6, observed)
        // fp32 accumulation-order difference between the batch-32 *compiled* kernel and the
        // direct batch-1 *uncompiled* kernel — ordinary GPU floating-point non-associativity, not
        // a logic bug (policy/value/ownership, which aren't multiplier-amplified, stay within
        // `tolerance` even at count=32) — reads as up to ~1.2e-5 here. Scale this one field's
        // tolerance accordingly instead of loosening the others.
        let scoreMeanTolerance = tolerance * 5
        for count in counts {
            let positions = Array(Self.positionPool.prefix(count))
            let actual = try await evaluator.evaluate(requests(from: positions))
            XCTAssertEqual(actual.count, count, "count=\(count)", file: file, line: line)
            for (position, output) in zip(positions, actual) {
                let expected = try directForward(network: network, desc: desc, position: position)
                assertOutputsClose(
                    output,
                    expected,
                    policyAccuracy: tolerance,
                    valueAccuracy: tolerance,
                    scoreMeanAccuracy: scoreMeanTolerance,
                    ownershipAccuracy: tolerance,
                    "count=\(count)",
                    file: file,
                    line: line
                )
            }
        }
    }

    // MARK: - fp16 vs fp32 tolerances (design.md fp16 tolerances)

    func testFloat16WithinDesignTolerancesOfFloat32() async throws {
        let fp32 = try makeEvaluator(precision: .float32, maxBatchSize: 16)
        let fp16 = try makeEvaluator(precision: .float16, maxBatchSize: 16)
        // 11 is non-power-of-two too, for good measure.
        let positions = Array(Self.positionPool.prefix(11))
        let fp32Results = try await fp32.evaluate(requests(from: positions))
        let fp16Results = try await fp16.evaluate(requests(from: positions))
        XCTAssertEqual(fp32Results.count, fp16Results.count)
        for (expected, actual) in zip(fp32Results, fp16Results) {
            assertOutputsClose(
                actual,
                expected,
                policyAccuracy: 5e-3,
                valueAccuracy: 5e-3,
                scoreMeanAccuracy: 0.1,
                ownershipAccuracy: 2e-2
            )
        }
    }

    // MARK: - Concurrency smoke

    /// Submits 4 concurrent `evaluate` waves of mixed sizes and checks all complete and match a
    /// serially-computed reference (same waves, same requests, run one at a time), independent
    /// of concurrent submission/completion order.
    func testConcurrentWavesIndependentOfSubmissionOrder() async throws {
        let evaluator = try makeEvaluator(precision: .float32, maxBatchSize: 32)
        let waveSizes = [3, 5, 7, 2]
        var waves = [[NNRequest]]()
        var offset = 0
        for size in waveSizes {
            waves.append(requests(from: Array(Self.positionPool[offset ..< offset + size])))
            offset += size
        }

        var serial = [[NNOutput]]()
        for wave in waves {
            try await serial.append(evaluator.evaluate(wave))
        }

        let concurrent = try await withThrowingTaskGroup(of: (Int, [NNOutput]).self) { group -> [[NNOutput]] in
            for (index, wave) in waves.enumerated() {
                group.addTask { try await (index, evaluator.evaluate(wave)) }
            }
            var collected = [Int: [NNOutput]]()
            for try await (index, outputs) in group {
                collected[index] = outputs
            }
            return waves.indices.map { collected[$0]! }
        }

        for waveIndex in waves.indices {
            XCTAssertEqual(concurrent[waveIndex].count, serial[waveIndex].count, "wave=\(waveIndex)")
            for (actual, expected) in zip(concurrent[waveIndex], serial[waveIndex]) {
                assertOutputsClose(
                    actual,
                    expected,
                    policyAccuracy: 1e-4,
                    valueAccuracy: 1e-4,
                    scoreMeanAccuracy: 1e-3,
                    ownershipAccuracy: 1e-4,
                    "wave=\(waveIndex)"
                )
            }
        }
    }

    // MARK: - Determinism

    func testDeterministicRepeatedBatch() async throws {
        let evaluator = try makeEvaluator(precision: .float32, maxBatchSize: 32)
        let reqs = requests(from: Array(Self.positionPool.prefix(9)))
        let first = try await evaluator.evaluate(reqs)
        let second = try await evaluator.evaluate(reqs)
        XCTAssertEqual(first.count, second.count)
        for (a, b) in zip(first, second) {
            assertOutputsClose(a, b, policyAccuracy: 0, valueAccuracy: 0, scoreMeanAccuracy: 0, ownershipAccuracy: 0)
        }
    }

    // MARK: - API surface: single-position convenience and the prefilled-row seam

    func testSinglePositionConvenienceMatchesBatchPath() async throws {
        let evaluator = try makeEvaluator(precision: .float32, maxBatchSize: 8)
        let position = Self.positionPool[0]
        let single = try await evaluator.evaluate(
            NNRequest(board: position.board, history: position.history, nextPlayer: position.nextPlayer)
        )
        let batch = try await evaluator.evaluate([
            NNRequest(board: position.board, history: position.history, nextPlayer: position.nextPlayer),
        ])
        assertOutputsClose(
            single,
            batch[0],
            policyAccuracy: 0,
            valueAccuracy: 0,
            scoreMeanAccuracy: 0,
            ownershipAccuracy: 0
        )
    }

    /// Search worker threads are expected to precompute feature rows off the evaluator actor and
    /// submit them via `NNRequest.prefilled`; this must match the board-based path exactly.
    func testPrefilledFeatureRowMatchesBoardBasedPath() async throws {
        let evaluator = try makeEvaluator(precision: .float32, maxBatchSize: 8)
        let positions = Array(Self.positionPool.prefix(3))
        let boardBased = requests(from: positions)
        let prefilled = positions.map { position -> NNRequest in
            let row = NNInputs.fillRowV7(
                position.board,
                position.history,
                position.nextPlayer,
                nnXLen: Self.nnLen,
                nnYLen: Self.nnLen
            )
            return NNRequest(
                board: position.board,
                history: position.history,
                nextPlayer: position.nextPlayer,
                prefilled: .init(spatial: row.spatial, global: row.global)
            )
        }
        let a = try await evaluator.evaluate(boardBased)
        let b = try await evaluator.evaluate(prefilled)
        XCTAssertEqual(a.count, b.count)
        for (x, y) in zip(a, b) {
            assertOutputsClose(x, y, policyAccuracy: 0, valueAccuracy: 0, scoreMeanAccuracy: 0, ownershipAccuracy: 0)
        }
    }

    func testInvalidPrefilledRowSizeThrowsRatherThanCrashing() async throws {
        let evaluator = try makeEvaluator(precision: .float32, maxBatchSize: 4)
        let position = Self.positionPool[0]
        let bad = NNRequest(
            board: position.board,
            history: position.history,
            nextPlayer: position.nextPlayer,
            prefilled: .init(spatial: [1, 2, 3], global: [1, 2, 3])
        )
        do {
            _ = try await evaluator.evaluate([bad])
            XCTFail("Expected NNEvaluatorError for a malformed prefilled feature row")
        } catch is NNEvaluatorError {
            // expected
        }
    }

    // MARK: - Fixtures / helpers

    // MARK: - Per-eval symmetry transform correctness

    /// Geometric correctness of the `useSymmetry` scatter (input permute) + gather (output inverse
    /// permute). Forcing each of the 8 dihedral symmetries must leave the canonical evaluation of the
    /// SAME position essentially unchanged, because a convnet is *approximately* orientation-invariant:
    /// a correct transform re-orients the board, evaluates, and maps the spatial policy/ownership back,
    /// so the canonical top move and value stay close to the identity (S=0) evaluation. This is the
    /// check strength A/Bs cannot do (codex): a broken SCATTER feeds a scrambled board -> the value
    /// goes haywire; a broken GATHER maps the (scalar-unaffected value but) spatial policy to
    /// geometrically-wrong canonical moves -> the top move jumps. Both failure modes are caught here.
    func testForcedSymmetryStaysConsistentWithIdentity() async throws {
        let net = try modelPath()
        let position = Self.positionPool[20] // a mid-game, asymmetric position
        let request = NNRequest(board: position.board, history: position.history, nextPlayer: position.nextPlayer)

        func evaluate(forcedSymmetry: Int) async throws -> NNOutput {
            let evaluator = try NNEvaluator(
                modelPath: net,
                nnXLen: Self.nnLen,
                nnYLen: Self.nnLen,
                precision: .float32,
                maxBatchSize: 1,
                symmetry: .hash,
                forcedSymmetry: forcedSymmetry
            )
            return try await evaluator.evaluate(request)
        }
        func topLegalMove(_ output: NNOutput) -> Int {
            var best = -1
            var bestProb = -Float.greatestFiniteMagnitude
            for (index, prob) in output.policyProbs.enumerated() where prob >= 0 && prob > bestProb {
                bestProb = prob
                best = index
            }
            return best
        }

        let identity = try await evaluate(forcedSymmetry: 0)
        let identityTop = topLegalMove(identity)
        XCTAssertGreaterThanOrEqual(identityTop, 0, "identity eval produced no legal move")

        for symmetry in 1 ..< 8 {
            let output = try await evaluate(forcedSymmetry: symmetry)
            // Broken scatter (garbage board) would wreck the scalar value; approx-invariance keeps it close.
            XCTAssertEqual(
                output.whiteWinProb, identity.whiteWinProb, accuracy: 0.08,
                "S=\(symmetry): value diverged from identity — likely a broken input permute (scatter)"
            )
            // Broken gather scrambles the spatial policy in canonical coords -> the top move jumps.
            XCTAssertEqual(
                topLegalMove(output), identityTop,
                "S=\(symmetry): top canonical move differs from identity — likely a broken output inverse (gather)"
            )
        }
    }

    /// Carry-consistency + correctness (design v2 tests #1/#3): symmetry is drawn UPSTREAM and
    /// carried on `params.symmetry` precisely so `stage()` (input permute) and `postprocess()`
    /// (output inverse permute) apply the SAME orientation across the N/N-1 pipeline reorder. Driving
    /// a `.random`-mode evaluator with an explicit carried symmetry S must reproduce the known-correct
    /// `forcedSymmetry: S` path (`testForcedSymmetryStaysConsistentWithIdentity` already pins that as
    /// geometrically correct). If postprocess ignored the carried value while stage honored it — the
    /// exact bug upstream-carry guards against — the two would diverge (the top move would jump).
    func testCarriedSymmetryMatchesForcedSymmetry() async throws {
        let net = try modelPath()
        let position = Self.positionPool[20]
        let carriedEvaluator = try NNEvaluator(
            modelPath: net,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            maxBatchSize: 1,
            symmetry: .random
        )
        for symmetry in 0 ..< 8 {
            var params = MiscNNInputParams()
            params.symmetry = symmetry
            let carried = try await carriedEvaluator.evaluate(NNRequest(
                board: position.board,
                history: position.history,
                nextPlayer: position.nextPlayer,
                params: params
            ))
            let forcedEvaluator = try NNEvaluator(
                modelPath: net,
                nnXLen: Self.nnLen,
                nnYLen: Self.nnLen,
                precision: .float32,
                maxBatchSize: 1,
                symmetry: .off,
                forcedSymmetry: symmetry
            )
            let forced = try await forcedEvaluator.evaluate(NNRequest(
                board: position.board,
                history: position.history,
                nextPlayer: position.nextPlayer
            ))
            // Same S resolved through both paths => identical computation. Tolerances match the
            // cross-context concurrency test; a carry bug scrambles policy far above them.
            assertOutputsClose(
                carried,
                forced,
                policyAccuracy: 1e-4,
                valueAccuracy: 1e-4,
                scoreMeanAccuracy: 1e-3,
                ownershipAccuracy: 1e-4,
                "carried S=\(symmetry) must match forced S=\(symmetry)"
            )
        }
    }

    /// Round-trip correctness via the CARRIED path (design v2 test #3), the analog of
    /// `testForcedSymmetryStaysConsistentWithIdentity`: a convnet is approximately orientation-
    /// invariant, so re-orienting by a carried symmetry S, evaluating, and mapping the spatial
    /// policy/ownership back must leave the value near identity and the top canonical move unchanged.
    func testCarriedSymmetryStaysConsistentWithIdentity() async throws {
        let net = try modelPath()
        let position = Self.positionPool[20]
        let evaluator = try NNEvaluator(
            modelPath: net,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            maxBatchSize: 1,
            symmetry: .random
        )
        func evaluate(carried: Int) async throws -> NNOutput {
            var params = MiscNNInputParams()
            params.symmetry = carried
            return try await evaluator.evaluate(NNRequest(
                board: position.board,
                history: position.history,
                nextPlayer: position.nextPlayer,
                params: params
            ))
        }
        func topLegalMove(_ output: NNOutput) -> Int {
            var best = -1
            var bestProb = -Float.greatestFiniteMagnitude
            for (index, prob) in output.policyProbs.enumerated() where prob >= 0 && prob > bestProb {
                bestProb = prob
                best = index
            }
            return best
        }
        let identity = try await evaluate(carried: 0)
        let identityTop = topLegalMove(identity)
        XCTAssertGreaterThanOrEqual(identityTop, 0, "identity eval produced no legal move")
        for symmetry in 1 ..< 8 {
            let output = try await evaluate(carried: symmetry)
            XCTAssertEqual(
                output.whiteWinProb, identity.whiteWinProb, accuracy: 0.08,
                "carried S=\(symmetry): value diverged from identity (broken input permute?)"
            )
            XCTAssertEqual(
                topLegalMove(output), identityTop,
                "carried S=\(symmetry): top canonical move differs from identity (broken output inverse?)"
            )
        }
    }

    /// Precedence of the (internal) resolver `NNEvaluator.symmetry(for:)` (design v2 test #2):
    /// forcedSymmetryOverride > carried `params.symmetry` (0...7) > mode; `symmetryNotSpecified` and
    /// out-of-range carried values fall through to mode. Pinned directly on the resolver so the
    /// ordering itself is asserted, not merely its downstream numerical side effects.
    func testSymmetryPrecedence() async throws {
        let net = try modelPath()
        let position = Self.positionPool[3]
        func request(carried: Int) -> NNRequest {
            var params = MiscNNInputParams()
            params.symmetry = carried
            return NNRequest(
                board: position.board,
                history: position.history,
                nextPlayer: position.nextPlayer,
                params: params
            )
        }
        let unset = NNRequest(
            board: position.board,
            history: position.history,
            nextPlayer: position.nextPlayer
        )
        func makeEvaluator(_ mode: NNSymmetryMode, forced: Int? = nil) throws -> NNEvaluator {
            try NNEvaluator(
                modelPath: net,
                nnXLen: Self.nnLen,
                nnYLen: Self.nnLen,
                precision: .float32,
                maxBatchSize: 1,
                symmetry: mode,
                forcedSymmetry: forced
            )
        }
        let forced3 = try makeEvaluator(.random, forced: 3)
        let hash = try makeEvaluator(.hash)
        let off = try makeEvaluator(.off)
        let random = try makeEvaluator(.random)

        // Actor-isolated `symmetry(for:)` results are bound before asserting so the sync
        // `XCTAssertEqual` never wraps an `await` (which the formatter would hoist into a redundant,
        // sync-overload-selecting `await XCTAssertEqual(...)`).

        // forcedSymmetryOverride wins over a carried value, an unset request, and an out-of-range one.
        let forcedOverCarried = await forced3.symmetry(for: request(carried: 5))
        let forcedOverUnset = await forced3.symmetry(for: unset)
        let forcedOverBadRange = await forced3.symmetry(for: request(carried: MiscNNInputParams.symmetryAll))
        XCTAssertEqual(forcedOverCarried, 3)
        XCTAssertEqual(forcedOverUnset, 3)
        XCTAssertEqual(forcedOverBadRange, 3)

        // A carried value in 0...7 wins over the mode (even the hash mode).
        let carriedOverHash = await hash.symmetry(for: request(carried: 6))
        let carriedOverRandom = await random.symmetry(for: request(carried: 6))
        XCTAssertEqual(carriedOverHash, 6)
        XCTAssertEqual(carriedOverRandom, 6)

        // Unset (symmetryNotSpecified) falls through to the mode.
        let offUnset = await off.symmetry(for: unset)
        let randomUnset = await random.symmetry(for: unset)
        let hashUnset = await hash.symmetry(for: unset)
        let expectedHash = Int(NNInputs.getHash(
            position.board, position.history, position.nextPlayer, MiscNNInputParams()
        ).hash0 & 7)
        XCTAssertEqual(offUnset, 0)
        XCTAssertEqual(randomUnset, 0)
        XCTAssertEqual(hashUnset, expectedHash)

        // Out-of-range carried values are treated as unset and fall through to the mode.
        let offHigh = await off.symmetry(for: request(carried: 8))
        let offNegative = await off.symmetry(for: request(carried: -5))
        let offSymAll = await off.symmetry(for: request(carried: MiscNNInputParams.symmetryAll))
        XCTAssertEqual(offHigh, 0)
        XCTAssertEqual(offNegative, 0)
        XCTAssertEqual(offSymAll, 0)
    }

    private func modelPath() throws -> String {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"] ?? Self.defaultModelsDirectory
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw XCTSkip("KataGo model fixtures are missing; set KATAGO_MODELS_DIR (checked \(directory))")
        }
        let path = URL(fileURLWithPath: directory).appendingPathComponent(Self.modelFile).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("KataGo model fixture is missing: \(path)")
        }
        return path
    }

    private func makeEvaluator(precision: KataGoNetwork.Precision, maxBatchSize: Int) throws -> NNEvaluator {
        try NNEvaluator(
            modelPath: modelPath(),
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: precision,
            maxBatchSize: maxBatchSize
        )
    }

    private func makeDirectNetwork(
        precision: KataGoNetwork.Precision
    ) throws -> (network: KataGoNetwork, desc: ModelDesc) {
        let desc = try ModelDesc.loadFromFileMaybeGZipped(modelPath())
        let network = try KataGoNetwork(desc: desc, nnXLen: Self.nnLen, nnYLen: Self.nnLen, precision: precision)
        return (network, desc)
    }

    private func requests(from positions: [(board: Board, history: BoardHistory, nextPlayer: Player)]) -> [NNRequest] {
        positions.map { NNRequest(board: $0.board, history: $0.history, nextPlayer: $0.nextPlayer) }
    }

    /// Direct (non-batched, non-actor, uncompiled) forward + postprocess, mirroring
    /// `ringo rawnn`'s (`Sources/ringo/main.swift`) logic exactly: this is the ground
    /// truth `NNEvaluator`'s batched/padded/compiled path must reproduce.
    private func directForward(
        network: KataGoNetwork,
        desc: ModelDesc,
        position: (board: Board, history: BoardHistory, nextPlayer: Player),
        params: MiscNNInputParams = MiscNNInputParams()
    ) throws -> NNOutput {
        let row = NNInputs.fillRowV7(
            position.board,
            position.history,
            position.nextPlayer,
            params,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen
        )
        let spatial = MLXArray(row.spatial, [1, Self.nnLen, Self.nnLen, desc.numInputChannels])
        let global = MLXArray(row.global, [1, desc.numInputGlobalChannels])
        let outputs = network.forward(spatial: spatial, global: global)
        eval([
            outputs.policySpatialLogits,
            outputs.policyPassLogits,
            outputs.valueLogits,
            outputs.scoreValues,
            outputs.ownership,
        ])
        let policyChannels = outputs.policySpatialLogits.dim(-1)
        let spatialPolicy = outputs.policySpatialLogits.asArray(Float.self)
        let passPolicy = outputs.policyPassLogits.asArray(Float.self)
        let optimisticChannel = min(1, policyChannels - 1)
        let optimism = Float(params.policyOptimism)
        var policy = (0 ..< Self.nnLen * Self.nnLen).map { index -> Float in
            let offset = index * policyChannels
            let normal = spatialPolicy[offset]
            return normal + (spatialPolicy[offset + optimisticChannel] - normal) * optimism
        }
        policy.append(passPolicy[0] + (passPolicy[optimisticChannel] - passPolicy[0]) * optimism)
        let raw = NNRawResult(
            policyLogits: policy,
            valueLogits: outputs.valueLogits.asArray(Float.self),
            scoreValues: outputs.scoreValues.asArray(Float.self),
            ownership: outputs.ownership.asArray(Float.self)
        )
        return try NNOutputPostprocessor.process(
            raw: raw,
            board: position.board,
            history: position.history,
            nextPlayer: position.nextPlayer,
            params: params,
            postProcessParams: desc.postProcessParams,
            modelVersion: desc.modelVersion,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen
        )
    }

    private func assertOutputsClose(
        _ actual: NNOutput,
        _ expected: NNOutput,
        policyAccuracy: Float,
        valueAccuracy: Float,
        scoreMeanAccuracy: Float,
        ownershipAccuracy: Float,
        _ context: @autoclosure () -> String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let label = context()
        XCTAssertEqual(actual.policyProbs.count, expected.policyProbs.count, label, file: file, line: line)
        for (a, e) in zip(actual.policyProbs, expected.policyProbs) {
            if e < 0 || a < 0 {
                XCTAssertEqual(a, e, label, file: file, line: line)
            } else {
                XCTAssertEqual(a, e, accuracy: policyAccuracy, label, file: file, line: line)
            }
        }
        XCTAssertEqual(
            actual.whiteWinProb,
            expected.whiteWinProb,
            accuracy: valueAccuracy,
            label,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.whiteLossProb,
            expected.whiteLossProb,
            accuracy: valueAccuracy,
            label,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.whiteNoResultProb,
            expected.whiteNoResultProb,
            accuracy: valueAccuracy,
            label,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.whiteScoreMean,
            expected.whiteScoreMean,
            accuracy: scoreMeanAccuracy,
            label,
            file: file,
            line: line
        )
        XCTAssertEqual(actual.whiteLead, expected.whiteLead, accuracy: scoreMeanAccuracy, label, file: file, line: line)
        switch (actual.whiteOwnerMap, expected.whiteOwnerMap) {
        case let (actualMap?, expectedMap?):
            XCTAssertEqual(actualMap.count, expectedMap.count, label, file: file, line: line)
            for (a, e) in zip(actualMap, expectedMap) {
                XCTAssertEqual(a, e, accuracy: ownershipAccuracy, label, file: file, line: line)
            }
        case (nil, nil):
            break
        default:
            XCTFail("ownership presence mismatch \(label)", file: file, line: line)
        }
    }

    /// Deterministic, dependency-free RNG (RinGoCore's `DeterministicRand` is internal to that
    /// module) used only to generate varied test positions.
    private struct SplitMix64 {
        private var state: UInt64
        init(seed: UInt64) {
            state = seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// Plays a single, deterministic, always-legal random game on a 19x19 board and snapshots the
    /// position (via `Board.copy()`/`BoardHistory.copy()`) before each move, yielding `count`
    /// distinct positions of increasing move number. Never passes voluntarily (only when no
    /// legal non-pass move exists), so this comfortably reaches `count` up into the dozens.
    /// Prefers non-self-atari moves (cheap O(1) neighbor check): pure uniform-random legal play
    /// piles up an unrealistic density of 1-2-liberty chains as the game progresses, which is
    /// both unrepresentative of real positions and needlessly expensive for `fillRowV7`'s ladder
    /// search (see the identical note on `Benchmark.randomPlayoutPositions`).
    private static func generatePositions(
        count: Int,
        seed: UInt64
    ) -> [(board: Board, history: BoardHistory, nextPlayer: Player)] {
        var rng = SplitMix64(seed: seed)
        let board = Board(nnLen, nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        var nextPlayer = Player.black
        var positions = [(board: Board, history: BoardHistory, nextPlayer: Player)]()
        positions.reserveCapacity(count)
        var safety = 0
        while positions.count < count {
            safety += 1
            precondition(safety < count * 200, "random playout could not generate enough distinct positions")
            positions.append((board.copy(), history.copy(), nextPlayer))
            let candidates = board.playableLocations().filter { history.isLegal(board, $0, nextPlayer) }
            let safeCandidates = candidates.filter { board.getBoundNumLibertiesAfterPlay($0, nextPlayer).lower >= 2 }
            let pickFrom = safeCandidates.isEmpty ? candidates : safeCandidates
            var moved = false
            if !pickFrom.isEmpty {
                let index = Int(rng.next() % UInt64(pickFrom.count))
                moved = history.makeBoardMoveTolerant(board, pickFrom[index], nextPlayer)
            }
            if !moved {
                _ = history.makeBoardMoveTolerant(board, Board.passLoc, nextPlayer)
            }
            nextPlayer = nextPlayer.opponent
        }
        return positions
    }
}
