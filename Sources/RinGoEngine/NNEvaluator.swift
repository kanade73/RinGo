import Foundation
import MLX
import RinGoCore
import RinGoModel

/// How `NNEvaluator` picks the per-eval NN input symmetry (KataGo's `nnRandomize`) for a request
/// whose carried `MiscNNInputParams.symmetry` is unset. The design's precedence puts a carried
/// symmetry ABOVE this mode (see `NNEvaluator.symmetry(for:)`), so in practice this mode only
/// governs requests that arrive WITHOUT an explicit symmetry — which is every request in `.off`/
/// `.hash` mode, and only `preWarm` (never a real search leaf) in `.random` mode:
/// - `.off`: identity (symmetry 0). The default — behavior-identical to no symmetry, so oracle
///   parity and every existing gate are preserved.
/// - `.hash`: deterministic, selected from the request's full situational hash (`hash0 & 7`), so
///   search stays bit-reproducible while different positions get different orientations. This is
///   the previous `-symmetry on` behavior.
/// - `.random`: the true per-eval random symmetry is drawn UPSTREAM, once per node, by `Search`
///   (from its own seeded PRNG) and carried on `MiscNNInputParams.symmetry`; the evaluator merely
///   reads that carried value. A request that still reaches the evaluator unset in this mode (i.e.
///   `preWarm`, which must NOT consume the Search PRNG stream) is evaluated at identity, so warmup
///   never shifts the random sequence. The evaluator itself holds no PRNG and stays a pure function
///   of its input, so fp16 primary and fp32 fallback of the SAME request always agree.
public enum NNSymmetryMode: String, Sendable, CaseIterable {
    case off
    case hash
    case random
}

public struct NNEvaluatorError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

/// One position to evaluate. Either give `board`/`history`/`nextPlayer`/`params` alone and let
/// `NNEvaluator` compute the V7 feature planes internally, or precompute them (e.g. on a search
/// worker thread, off the evaluator's actor) and pass them as `prefilled` — `board`/`history`/
/// `nextPlayer`/`params` are still required in that case because `NNOutputPostprocessor` needs
/// them for legal-move masking and rules-dependent postprocessing, which are independent of
/// which channel computed the raw feature planes. This dual seam is what lets a future search's
/// worker threads submit leaves cheaply from any thread: heavy `fillRowV7` work can happen off
/// the evaluator actor, in parallel, before a batch is ever handed to it.
///
/// `Board` and `BoardHistory` are `@unchecked Sendable` reference types; treat a submitted
/// request as handed off to the evaluator until its `evaluate` call returns — don't mutate the
/// same `board`/`history` instance concurrently from another thread in the meantime.
public struct NNRequest: Sendable {
    /// Precomputed V7 feature planes for one position (NHWC spatial, matching
    /// `NNInputs.fillRowV7`'s default layout).
    public struct FeatureRow: Sendable {
        public var spatial: [Float]
        public var global: [Float]

        public init(spatial: [Float], global: [Float]) {
            self.spatial = spatial
            self.global = global
        }
    }

    public var board: Board
    public var history: BoardHistory
    public var nextPlayer: Player
    public var params: MiscNNInputParams
    public var prefilled: FeatureRow?

    public init(
        board: Board,
        history: BoardHistory,
        nextPlayer: Player,
        params: MiscNNInputParams = MiscNNInputParams(),
        prefilled: FeatureRow? = nil
    ) {
        self.board = board
        self.history = history
        self.nextPlayer = nextPlayer
        self.params = params
        self.prefilled = prefilled
    }
}

/// The plain-Swift surface used by search and GTP evaluator lifecycle management. Keeping this
/// protocol free of `MLXArray` lets command-flow tests substitute a CPU-only recorder.
public protocol NNEvaluating: Actor {
    func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput]
    func preWarm(_ requests: [NNRequest]) async throws
}

public extension NNEvaluating {
    func evaluate(_ request: NNRequest) async throws -> NNOutput {
        let outputs = try await evaluate([request])
        guard let output = outputs.first else {
            throw NNEvaluatorError("Internal error: evaluate([request]) returned no results")
        }
        return output
    }

    func preWarm(_ requests: [NNRequest]) async throws {
        _ = try await evaluate(requests)
    }
}

/// The MLX island (design.md "MLX discipline"): every `MLXArray` created by this port's
/// performance path lives and dies inside this actor. Requests arrive as plain Swift structs
/// (`NNRequest`) and results leave as plain Swift structs (`NNOutput`); no `MLXArray` crosses
/// the actor boundary in either direction, so callers (a future search included) never have a
/// chance to accidentally grow a graph across threads or eval it twice.
///
/// Architecture, and why, point by point against design.md's discipline list:
///
/// 1. **The MLX island** — `MLXArray` never appears in this type's public API (`NNRequest`,
///    `NNOutput`, `KataGoNetwork.Precision` are all plain Swift/RinGoModel-metadata types).
/// 2. **One eval per batch** — each physical batch is staged, built into exactly two input
///    tensors (spatial, global), run through one compiled forward graph, and materialized with
///    exactly one `eval()` call (batched across all 5 output tensors at once, not per-tensor).
/// 3. **compile() per bucket** — batch sizes are bucketed to powers of two up to
///    `maxBatchSize` (`bucketSizes`); each bucket gets its own `MLX.compile`d forward closure,
///    built once at `init` and reused for the actor's lifetime. Shapes are static per bucket
///    (padding guarantees this), so `compile` traces and fuses kernels exactly once per bucket,
///    on that bucket's first invocation — the benchmark measures and reports that warmup cost
///    separately from steady state.
/// 4. **Pipelining** — within one `evaluate` call that spans multiple physical batches (more
///    requests than `maxBatchSize`), batches are run with a classic "N/N-1" software pipeline:
///    batch N's `asyncEval` is issued, then (without waiting) the loop moves on to stage and
///    `asyncEval` batch N+1 immediately; only *then* does it block-extract batch N's results.
///    Because `asyncEval` just enqueues GPU work and returns, batch N's kernels run on the GPU
///    concurrently with batch N+1's CPU-side staging — CPU fill is never serialized behind a
///    GPU wait. The final "wait for GPU, then copy to Swift arrays" step for each batch is
///    additionally moved off the actor's own executor onto a plain `DispatchQueue` via a
///    checked continuation (`extractFloats`), so the actor itself is only ever busy with cheap
///    CPU work (staging, `asyncEval`, postprocessing) and is free to interleave *other* callers'
///    `evaluate` calls (their staging + `asyncEval`) while one caller's batch is GPU-bound. Two
///    reused staging buffers per bucket (double-buffered, see `Bucket`) back this: filling
///    buffer B for the next batch never touches buffer A's bytes while A's upload is still being
///    read (in practice `MLXArray(_:_:)` copies its input synchronously, so this is guaranteed
///    correct even without double buffering, but double buffering is what the design calls for
///    and is what keeps that guarantee from being an implementation-detail dependency).
/// 5. **Memory** — `gpuCacheLimitMB` sets `MLX.Memory.cacheLimit` once at startup if given;
///    staging buffers are allocated once per bucket and reused for the actor's lifetime; the
///    N/N-1 pipeline never holds more than two batches' worth of output tensors alive at once.
/// 6. **No hidden syncs** — nothing reads array *values* before the one designated sync point
///    (`extractFloats`, which calls `eval()` then `asArray(Float.self)`). Bucket selection reads
///    only `Int` counts computed in Swift; channel/row strides come from `ModelDesc` metadata,
///    never from inspecting a tensor's contents.
public actor NNEvaluator: NNEvaluating {
    /// Ascending physical batch sizes this evaluator will run on the GPU: powers of two, capped
    /// at (not necessarily including) `maxBatchSize` — e.g. `maxBatchSize: 24` yields
    /// `[1, 2, 4, 8, 16]`. A request chunk larger than the last bucket cannot occur: `evaluate`
    /// splits its input into chunks of at most `bucketSizes.last` before bucketing each chunk.
    private let bucketSizes: [Int]
    private let buckets: [Bucket]
    private let desc: ModelDesc
    private let nnXLen: Int
    private let nnYLen: Int
    private let numSpatialChannels: Int
    private let numGlobalChannels: Int
    /// Fallback symmetry mode for a request whose carried `params.symmetry` is unset (see
    /// `NNSymmetryMode`). Only `.hash` mode actually derives a non-identity symmetry here; `.off`
    /// and `.random`-when-unset (i.e. `preWarm`) both yield identity. The real per-node symmetry in
    /// `.random` mode arrives carried on the request from `Search`, so the evaluator keeps no PRNG
    /// of its own and stays a pure function of its input — fp16 primary and fp32 fallback of the
    /// same request therefore always apply the same orientation. Only engaged on square NN tensors
    /// (`nnXLen == nnYLen`).
    private let symmetryMode: NNSymmetryMode
    /// Test/verification hook: when set (0...7), forces that exact symmetry on EVERY request instead
    /// of the hash-derived one, so per-symmetry oracle parity (`nnForcedSymmetry=S`) and unit tests
    /// can compare a known orientation against the reference. `nil` = normal hash selection.
    private let forcedSymmetryOverride: Int?

    public init(
        modelPath: String,
        nnXLen: Int,
        nnYLen: Int,
        precision: KataGoNetwork.Precision = .float16,
        maxBatchSize: Int = 32,
        gpuCacheLimitMB: Int? = nil,
        symmetry mode: NNSymmetryMode = .off,
        forcedSymmetry: Int? = nil
    ) throws {
        let desc = try ModelDesc.loadFromFileMaybeGZipped(modelPath)
        try self.init(
            desc: desc,
            nnXLen: nnXLen,
            nnYLen: nnYLen,
            precision: precision,
            maxBatchSize: maxBatchSize,
            gpuCacheLimitMB: gpuCacheLimitMB,
            symmetry: mode,
            forcedSymmetry: forcedSymmetry
        )
    }

    public init(
        desc: ModelDesc,
        nnXLen: Int,
        nnYLen: Int,
        precision: KataGoNetwork.Precision = .float16,
        maxBatchSize: Int = 32,
        gpuCacheLimitMB: Int? = nil,
        symmetry mode: NNSymmetryMode = .off,
        forcedSymmetry: Int? = nil
    ) throws {
        guard maxBatchSize >= 1 else {
            throw NNEvaluatorError("maxBatchSize must be >= 1, got \(maxBatchSize)")
        }
        if let gpuCacheLimitMB {
            guard gpuCacheLimitMB >= 0 else {
                throw NNEvaluatorError("gpuCacheLimitMB must be >= 0, got \(gpuCacheLimitMB)")
            }
            MLX.Memory.cacheLimit = gpuCacheLimitMB * 1024 * 1024
        }

        self.desc = desc
        self.nnXLen = nnXLen
        self.nnYLen = nnYLen
        numSpatialChannels = desc.numInputChannels
        numGlobalChannels = desc.numInputGlobalChannels
        symmetryMode = mode
        forcedSymmetryOverride = forcedSymmetry

        var sizes = [Int]()
        var size = 1
        while size <= maxBatchSize {
            sizes.append(size)
            size <<= 1
        }
        bucketSizes = sizes

        let network = try KataGoNetwork(desc: desc, nnXLen: nnXLen, nnYLen: nnYLen, precision: precision)
        let spatialRowLength = desc.numInputChannels * nnXLen * nnYLen
        let globalRowLength = desc.numInputGlobalChannels
        buckets = sizes.map { bucketSize in
            Bucket(
                size: bucketSize,
                spatialRowLength: spatialRowLength,
                globalRowLength: globalRowLength,
                // Captures `network` (and, transitively, its MLXArray-valued weight parameters)
                // as constants baked into the compiled graph — the weights were already eval'd
                // once at `KataGoNetwork.init`, matching design.md point 3 ("weights are eval'd
                // once at load and captured as constants"). `MLX.compile`'s `f` parameter is not
                // itself required to be `@Sendable` (only the closure `compile` *returns* is),
                // so capturing the non-Sendable `KataGoNetwork` class here is legal; the returned
                // closure is only ever invoked from this actor.
                compiled: MLX.compile { arrays in
                    let raw = network.forward(spatial: arrays[0], global: arrays[1])
                    return [
                        raw.policySpatialLogits,
                        raw.policyPassLogits,
                        raw.valueLogits,
                        raw.scoreValues,
                        raw.ownership,
                    ]
                }
            )
        }
    }

    /// Evaluates a batch of positions, splitting into physical GPU batches of at most
    /// `bucketSizes.last`, each padded to its smallest covering power-of-two bucket. Results are
    /// returned in the same order as `requests`, independent of any internal chunking.
    public func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput] {
        guard !requests.isEmpty else { return [] }
        let physicalMax = bucketSizes.last!

        var results = [NNOutput]()
        results.reserveCapacity(requests.count)

        // Classic double-buffered ("N/N-1 lag") software pipeline: `asyncEval` a physical batch,
        // then immediately move on to staging + `asyncEval`-ing the *next* one before blocking to
        // extract the previous one's results. See the type doc comment, discipline point 4.
        var pending: (chunk: ArraySlice<NNRequest>, batch: PendingBatch)?

        var start = requests.startIndex
        while start < requests.endIndex {
            let end = min(start + physicalMax, requests.endIndex)
            let chunk = requests[start ..< end]
            let bucket = try bucket(forCount: chunk.count)
            let inputs = try stage(chunk, into: bucket)
            let outputs = bucket.compiled(inputs)
            asyncEval(outputs)

            if let pending {
                let flattened = await extractFloats(pending.batch)
                try results.append(contentsOf: postprocess(flattened, chunk: pending.chunk))
            }
            pending = (chunk, PendingBatch(outputs: outputs))
            start = end
        }
        if let pending {
            let flattened = await extractFloats(pending.batch)
            try results.append(contentsOf: postprocess(flattened, chunk: pending.chunk))
        }
        return results
    }

    /// Single-position convenience over `evaluate(_:[NNRequest]) async throws -> [NNOutput]`.
    public func evaluate(_ request: NNRequest) async throws -> NNOutput {
        let results = try await evaluate([request])
        guard let first = results.first else {
            throw NNEvaluatorError("Internal error: evaluate([request]) returned no results")
        }
        return first
    }

    public func preWarm(_ requests: [NNRequest]) async throws {
        _ = try await evaluate(requests)
    }

    private func bucket(forCount count: Int) throws -> Bucket {
        guard let match = buckets.first(where: { $0.size >= count }) else {
            throw NNEvaluatorError("Request chunk of \(count) exceeds the largest bucket (\(bucketSizes.last ?? 0))")
        }
        return match
    }

    /// Fills `bucket`'s next staging buffer from `chunk` (real rows via `fillRowV7` or
    /// `prefilled`, zero spatial + zero global for the padding tail — a zero spatial row has a
    /// zero mask channel, so the network treats padding rows as entirely masked-out, matching
    /// design.md's bucketing rule), then uploads it as the two `MLXArray` inputs the compiled
    /// forward closure expects. This is the only place staging buffers are touched; by the time
    /// this function returns, the `MLXArray`s already hold their own independent copy of the
    /// data (`MLXArray(_:_:)` copies synchronously), so the staging buffer is immediately free
    /// for its next use.
    private func stage(_ chunk: ArraySlice<NNRequest>, into bucket: Bucket) throws -> [MLXArray] {
        let spatialRowLength = numSpatialChannels * nnXLen * nnYLen
        let bucketSize = bucket.size
        return try bucket.withStaging { spatial, global in
            var row = 0
            for request in chunk {
                guard request.board.xSize <= nnXLen, request.board.ySize <= nnYLen else {
                    throw NNEvaluatorError(
                        "Board \(request.board.xSize)x\(request.board.ySize) exceeds evaluator nnXLen/nnYLen \(nnXLen)x\(nnYLen)"
                    )
                }
                let values: (spatial: [Float], global: [Float])
                if let prefilled = request.prefilled {
                    guard prefilled.spatial.count == spatialRowLength,
                          prefilled.global.count == numGlobalChannels
                    else {
                        throw NNEvaluatorError(
                            "Prefilled feature row size \(prefilled.spatial.count)/\(prefilled.global.count) does not "
                                + "match evaluator nnXLen/nnYLen/channel counts (expected \(spatialRowLength)/\(numGlobalChannels))"
                        )
                    }
                    values = (prefilled.spatial, prefilled.global)
                } else {
                    values = NNInputs.fillRowV7(
                        request.board,
                        request.history,
                        request.nextPlayer,
                        request.params,
                        nnXLen: nnXLen,
                        nnYLen: nnYLen
                    )
                }
                let inputSymmetry = symmetry(for: request)
                if inputSymmetry == 0 {
                    copy(values.spatial, into: &spatial, at: row * spatialRowLength)
                } else {
                    copyPermuted(values.spatial, into: &spatial, at: row * spatialRowLength, symmetry: inputSymmetry)
                }
                copy(values.global, into: &global, at: row * numGlobalChannels)
                row += 1
            }
            if row < bucketSize {
                zero(&spatial, from: row * spatialRowLength, count: (bucketSize - row) * spatialRowLength)
                zero(&global, from: row * numGlobalChannels, count: (bucketSize - row) * numGlobalChannels)
            }
            let spatialArray = MLXArray(spatial, [bucketSize, nnYLen, nnXLen, numSpatialChannels])
            let globalArray = MLXArray(global, [bucketSize, numGlobalChannels])
            return [spatialArray, globalArray]
        }
    }

    // MARK: - Per-eval input symmetry (KataGo nnRandomize)

    /// The 0...7 dihedral symmetry to apply to `request` (design v2 precedence): a test
    /// `forcedSymmetryOverride` wins; else a carried `params.symmetry` in 0...7 — the upstream
    /// `Search` per-node draw in `.random` mode — wins; else the evaluator's own `symmetryMode`
    /// decides (`.off`/`.random`-unset → identity, `.hash` → `hash0 & 7`). Returns 0 (identity) on a
    /// non-square NN tensor regardless of mode, since the dihedral transforms are only defined for
    /// `nnXLen == nnYLen`. Pure function of the request, so `stage()` (input permute) and
    /// `postprocess()` (output inverse permute) always agree even though the N/N-1 pipeline stages
    /// batch N+1 before postprocessing batch N. Not `private` so it is directly unit-testable
    /// (`@testable`), which is the only way to pin the precedence itself rather than its side effects.
    func symmetry(for request: NNRequest) -> Int {
        guard nnXLen == nnYLen else { return 0 }
        if let forcedSymmetryOverride { return forcedSymmetryOverride }
        let carried = request.params.symmetry
        if carried >= 0, carried < Symmetry.count { return carried }
        switch symmetryMode {
        case .off, .random:
            return 0
        case .hash:
            let hash = NNInputs.getHash(request.board, request.history, request.nextPlayer, request.params)
            return Int(hash.hash0 & 7)
        }
    }

    /// KataGo `copyInputsWithSymmetry`, written DIRECTLY into the staging buffer at `offset` (no
    /// intermediate array — the extra alloc+copy measured ~5% throughput at fixed compute): scatter
    /// each canonical spatial position `(x, y)` to `Symmetry.apply(x, y, S)`, carrying its whole
    /// contiguous channel vector (NHWC). Square only.
    private func copyPermuted(_ source: [Float], into destination: inout [Float], at offset: Int, symmetry sym: Int) {
        let size = nnXLen
        let channels = numSpatialChannels
        source.withUnsafeBufferPointer { src in
            for y in 0 ..< size {
                for x in 0 ..< size {
                    let dest = Symmetry.apply(x: x, y: y, size: size, sym)
                    let srcBase = (y * size + x) * channels
                    let dstBase = offset + (dest.y * size + dest.x) * channels
                    for channel in 0 ..< channels {
                        destination[dstBase + channel] = src[srcBase + channel]
                    }
                }
            }
        }
    }

    /// Inverse map for a spatial OUTPUT (policy grid / ownership), KataGo `copyOutputsWithSymmetry`:
    /// the canonical position `p` reads the transformed-frame position `Symmetry.apply(p, S)`.
    private func symmetrySourceIndex(canonicalPosition position: Int, symmetry sym: Int) -> Int {
        let size = nnXLen
        let src = Symmetry.apply(x: position % size, y: position / size, size: size, sym)
        return src.y * size + src.x
    }

    /// The one designated sync point (design.md point 6): blocks until `pending`'s outputs are
    /// materialized and copies them to plain Swift arrays. Runs on a plain `DispatchQueue`
    /// (`Self.extractionQueue`), not the actor's own executor or a Swift Concurrency cooperative
    /// thread, so `await`ing it here genuinely suspends this actor and lets another queued
    /// `evaluate` call make progress (stage + `asyncEval` its own batch) while this one waits on
    /// the GPU — see the type doc comment, discipline point 4.
    private func extractFloats(_ pending: PendingBatch) async -> [[Float]] {
        await withCheckedContinuation { continuation in
            Self.extractionQueue.async {
                eval(pending.outputs)
                continuation.resume(returning: pending.outputs.map { $0.asArray(Float.self) })
            }
        }
    }

    private func postprocess(
        _ flattened: [[Float]],
        chunk: ArraySlice<NNRequest>
    ) throws -> [NNOutput] {
        let spatialPolicy = flattened[0]
        let passPolicy = flattened[1]
        let valueFlat = flattened[2]
        let scoreFlat = flattened[3]
        let ownershipFlat = flattened[4]

        let policyChannels = desc.numPolicyChannels
        let valueChannels = desc.numValueChannels
        let scoreChannels = desc.numScoreValueChannels
        let spatialPositions = nnXLen * nnYLen
        let optimisticChannel = min(1, policyChannels - 1)

        var results = [NNOutput]()
        results.reserveCapacity(chunk.count)
        for (row, request) in chunk.enumerated() {
            let optimism = Float(request.params.policyOptimism)
            // Same deterministic symmetry the input was permuted by (recomputed, not carried — it is
            // a pure function of the request, so stage() and postprocess() always agree even though
            // the pipeline stages batch N+1 before postprocessing batch N). The spatial policy and
            // ownership come back in the transformed frame, so each canonical position reads its
            // transformed-frame source `Symmetry.apply(p, S)`; pass/value/score are symmetry-invariant.
            let outputSymmetry = symmetry(for: request)
            var policyLogits = [Float](repeating: 0, count: spatialPositions + 1)
            let spatialBase = row * spatialPositions * policyChannels
            for position in 0 ..< spatialPositions {
                let source = outputSymmetry == 0
                    ? position
                    : symmetrySourceIndex(canonicalPosition: position, symmetry: outputSymmetry)
                let offset = spatialBase + source * policyChannels
                let normal = spatialPolicy[offset]
                let optimistic = spatialPolicy[offset + optimisticChannel]
                policyLogits[position] = normal + optimism * (optimistic - normal)
            }
            let passBase = row * policyChannels
            let passNormal = passPolicy[passBase]
            let passOptimistic = passPolicy[passBase + optimisticChannel]
            policyLogits[spatialPositions] = passNormal + optimism * (passOptimistic - passNormal)

            let valueLogits = Array(valueFlat[(row * valueChannels) ..< ((row + 1) * valueChannels)])
            let scoreValues = Array(scoreFlat[(row * scoreChannels) ..< ((row + 1) * scoreChannels)])
            let ownershipBase = row * spatialPositions
            let ownership: [Float] = outputSymmetry == 0
                ? Array(ownershipFlat[ownershipBase ..< (ownershipBase + spatialPositions)])
                : (0 ..< spatialPositions).map { position in
                    ownershipFlat[ownershipBase + symmetrySourceIndex(
                        canonicalPosition: position,
                        symmetry: outputSymmetry
                    )]
                }

            let raw = NNRawResult(
                policyLogits: policyLogits,
                valueLogits: valueLogits,
                scoreValues: scoreValues,
                ownership: ownership
            )
            let output = try NNOutputPostprocessor.process(
                raw: raw,
                board: request.board,
                history: request.history,
                nextPlayer: request.nextPlayer,
                params: request.params,
                postProcessParams: desc.postProcessParams,
                modelVersion: desc.modelVersion,
                nnXLen: nnXLen,
                nnYLen: nnYLen
            )
            results.append(output)
        }
        return results
    }

    private static let extractionQueue = DispatchQueue(
        label: "ringo.NNEvaluator.extraction",
        attributes: .concurrent
    )

    /// One physical GPU batch size. Owns a compiled forward closure (traced once, on first use)
    /// and two reused, fixed-size staging buffer pairs (double-buffered; see the type doc
    /// comment on `NNEvaluator`, discipline point 4) so repeated evaluation at this bucket size
    /// never allocates.
    private final class Bucket {
        let size: Int
        let compiled: ([MLXArray]) -> [MLXArray]
        private var spatialA: [Float]
        private var globalA: [Float]
        private var spatialB: [Float]
        private var globalB: [Float]
        private var useA = true

        init(size: Int, spatialRowLength: Int, globalRowLength: Int, compiled: @escaping ([MLXArray]) -> [MLXArray]) {
            self.size = size
            self.compiled = compiled
            spatialA = [Float](repeating: 0, count: size * spatialRowLength)
            globalA = [Float](repeating: 0, count: size * globalRowLength)
            spatialB = spatialA
            globalB = globalA
        }

        func withStaging<T>(_ body: (inout [Float], inout [Float]) throws -> T) rethrows -> T {
            defer { useA.toggle() }
            return if useA {
                try body(&spatialA, &globalA)
            } else {
                try body(&spatialB, &globalB)
            }
        }
    }

    /// Wraps one physical batch's still-lazy output `MLXArray`s so `extractFloats` can hand them
    /// to a plain `DispatchQueue` closure. mlx-swift's `MLXArray` deliberately has no `Sendable`
    /// conformance (it is a lazy graph handle, not a value type) — this wrapper's
    /// `@unchecked Sendable` mirrors the same pattern mlx-swift's own `compile()` uses internally
    /// for `CompiledFunction` (`Transforms+Compile.swift`). It is safe here because each
    /// `PendingBatch` is hard-owned by exactly one `extractFloats` call and never touched from
    /// two threads at once, and because MLX's C core additionally serializes all `eval`/
    /// `asyncEval` calls behind a single recursive lock (`evalLock` in mlx-swift's
    /// `Transforms+Eval.swift`), so even a future coding mistake that touched a batch from two
    /// threads could not corrupt MLX's internal state.
    private final class PendingBatch: @unchecked Sendable {
        let outputs: [MLXArray]
        init(outputs: [MLXArray]) {
            self.outputs = outputs
        }
    }
}

private func copy(_ source: [Float], into destination: inout [Float], at offset: Int) {
    destination.withUnsafeMutableBufferPointer { dst in
        source.withUnsafeBufferPointer { src in
            guard let srcBase = src.baseAddress, let dstBase = dst.baseAddress, !src.isEmpty else { return }
            (dstBase + offset).update(from: srcBase, count: src.count)
        }
    }
}

private func zero(_ destination: inout [Float], from offset: Int, count: Int) {
    guard count > 0 else { return }
    destination.withUnsafeMutableBufferPointer { dst in
        guard let base = dst.baseAddress else { return }
        (base + offset).update(repeating: 0, count: count)
    }
}
