import Foundation
import MLX

/// One training sample in "value type" form: the element of `RinGoShard.samples` and the shape
/// tests construct directly. This is NOT how a shard stores its samples resident — a `RinGoShard`
/// keeps columnar (struct-of-arrays) backing stores and materializes a `RinGoSample` on demand
/// (see `RinGoShard.sample(at:)`). `spatial` is `[Float]` here for API compatibility, but the
/// resident bytes are `UInt8` (V7 9x9 planes are binary 0/1) — see `RinGoShard`'s doc comment.
public struct RinGoSample: Sendable {
    /// Dense probability distribution over all `nnLen*nnLen+1` moves (board points + pass).
    public let spatial: [Float]
    public let global: [Float]
    public let policyTarget: [Float]
    public let valueTarget: [Float]
    public let scoreTarget: Float
    /// False when `scoreTarget` is a synthetic placeholder (e.g. the resignation proxy) that must
    /// be excluded from the score loss and its normalization -- see `RinGoDataFormat` (v2).
    public let scoreValid: Bool
    public let ownershipTarget: [Int8]
    /// False when `ownershipTarget` is not a reliable final-position label (e.g. a resignation) and
    /// must be excluded from the ownership loss and its normalization.
    public let ownershipValid: Bool

    public init(
        spatial: [Float], global: [Float], policyTarget: [Float],
        valueTarget: [Float], scoreTarget: Float, scoreValid: Bool,
        ownershipTarget: [Int8], ownershipValid: Bool
    ) {
        self.spatial = spatial
        self.global = global
        self.policyTarget = policyTarget
        self.valueTarget = valueTarget
        self.scoreTarget = scoreTarget
        self.scoreValid = scoreValid
        self.ownershipTarget = ownershipTarget
        self.ownershipValid = ownershipValid
    }
}

/// A loaded RinGoData v2 shard (or the concatenation of several — see `RinGoDataset.load`).
///
/// ## Resident representation
/// The ON-DISK format is unchanged: fp32 spatial planes. In RAM this type stores each field as a
/// single contiguous per-shard buffer (struct-of-arrays), indexed by `sampleIndex * <field>Stride`
/// (all samples share the same stride for a given `nnLen`, so no per-sample offset table is
/// needed). This kills BOTH memory findings of the audit at once:
/// - spatial is packed as `[UInt8]` instead of `[Float]`. Every 9x9 V7 plane written by
///   `NNInputs.fillRowV7` is a binary 0/1 indicator (verified: every `set(pos, feature)` uses the
///   default value 1; the only non-binary features live in the *global* vector, never a plane).
///   Loading validates each value is exactly 0 or 1 and fails loudly with the offending
///   sample/plane on anything else — it never silently quantizes a continuous feature. u8 storage
///   removes spatial's ~4x fp32 bloat (~21GB → ~7GB on the full corpus).
/// - all fields are 8 large buffers per shard rather than ~5 small Swift arrays *per sample*
///   (~21M tiny heap allocations on the full corpus), removing the COW-header / dirty-page bloat.
///
/// Conversion back to `Float32` happens only at batch assembly (`batch(indices:)`), so the
/// `MLXArray`s handed to training are bit-identical to the old fp32 path (0 → 0.0, 1 → 1.0 exactly).
public struct RinGoShard: Sendable {
    public let nnLen: Int
    public private(set) var sampleCount: Int

    // Per-sample element counts (uniform for a given nnLen). Stored so `sample(at:)` /
    // `batch(indices:)` don't recompute them per access.
    let spatialStride: Int // 22 * nnLen * nnLen
    let globalStride: Int // 19
    let policyStride: Int // nnLen*nnLen + 1
    let valueStride: Int // 3
    let ownershipStride: Int // nnLen*nnLen

    // Contiguous columnar backing stores. `var` (not `let`) only so `RinGoDataset.load` can append
    // shard-by-shard in place; still a value type (COW) and `Sendable`.
    var spatialBytes: [UInt8]
    var globalValues: [Float]
    var policyValues: [Float]
    var valueTargets: [Float]
    var scoreTargets: [Float]
    var scoreValidFlags: [Bool]
    var ownershipValues: [Int8]
    var ownershipValidFlags: [Bool]

    /// The 6-`UInt32` RinGoData header (magic, version, nnLen, spatial channels, global channels,
    /// sample count) — 24 bytes. Exposed so `RinGoDataset.load` can size a capacity reservation
    /// from raw file sizes without reading them.
    static let headerByteCount = 24

    /// Designated initializer taking already-validated columnar buffers (used by `read`).
    init(
        nnLen: Int, sampleCount: Int,
        spatialStride: Int, globalStride: Int, policyStride: Int, valueStride: Int, ownershipStride: Int,
        spatialBytes: [UInt8], globalValues: [Float], policyValues: [Float], valueTargets: [Float],
        scoreTargets: [Float], scoreValidFlags: [Bool], ownershipValues: [Int8], ownershipValidFlags: [Bool]
    ) {
        self.nnLen = nnLen
        self.sampleCount = sampleCount
        self.spatialStride = spatialStride
        self.globalStride = globalStride
        self.policyStride = policyStride
        self.valueStride = valueStride
        self.ownershipStride = ownershipStride
        self.spatialBytes = spatialBytes
        self.globalValues = globalValues
        self.policyValues = policyValues
        self.valueTargets = valueTargets
        self.scoreTargets = scoreTargets
        self.scoreValidFlags = scoreValidFlags
        self.ownershipValues = ownershipValues
        self.ownershipValidFlags = ownershipValidFlags
    }

    /// Packs `[RinGoSample]` into columnar storage, validating spatial planes are binary 0/1
    /// (throwing with the offending sample/plane otherwise) exactly as `read` does. Provided for
    /// tests / synthetic shards; the production load path (`read`) builds columns straight from disk
    /// without ever materializing `[RinGoSample]`.
    public init(nnLen: Int, samples: [RinGoSample]) throws {
        guard nnLen > 0 else { throw RinGoShardError("RinGoData nnLen must be positive") }
        let area = nnLen * nnLen
        let spatialStride = 22 * area
        let policyStride = area + 1
        var spatialBytes = [UInt8]()
        spatialBytes.reserveCapacity(samples.count * spatialStride)
        var globalValues = [Float](); globalValues.reserveCapacity(samples.count * 19)
        var policyValues = [Float](); policyValues.reserveCapacity(samples.count * policyStride)
        var valueTargets = [Float](); valueTargets.reserveCapacity(samples.count * 3)
        var scoreTargets = [Float](); scoreTargets.reserveCapacity(samples.count)
        var scoreValidFlags = [Bool](); scoreValidFlags.reserveCapacity(samples.count)
        var ownershipValues = [Int8](); ownershipValues.reserveCapacity(samples.count * area)
        var ownershipValidFlags = [Bool](); ownershipValidFlags.reserveCapacity(samples.count)
        for (index, sample) in samples.enumerated() {
            guard sample.spatial.count == spatialStride else {
                throw RinGoShardError(
                    "Sample \(index) has \(sample.spatial.count) spatial values, expected \(spatialStride)"
                )
            }
            for (local, value) in sample.spatial.enumerated() {
                try spatialBytes.append(Self.binaryPlaneByte(value, sampleIndex: index, local: local, planeCount: 22))
            }
            guard sample.global.count == 19 else {
                throw RinGoShardError("Sample \(index) has \(sample.global.count) global values, expected 19")
            }
            globalValues.append(contentsOf: sample.global)
            guard sample.policyTarget.count == policyStride else {
                throw RinGoShardError(
                    "Sample \(index) has \(sample.policyTarget.count) policy values, expected \(policyStride)"
                )
            }
            policyValues.append(contentsOf: sample.policyTarget)
            guard sample.valueTarget.count == 3 else {
                throw RinGoShardError("Sample \(index) has \(sample.valueTarget.count) value values, expected 3")
            }
            valueTargets.append(contentsOf: sample.valueTarget)
            scoreTargets.append(sample.scoreTarget)
            scoreValidFlags.append(sample.scoreValid)
            guard sample.ownershipTarget.count == area else {
                throw RinGoShardError(
                    "Sample \(index) has \(sample.ownershipTarget.count) ownership values, expected \(area)"
                )
            }
            ownershipValues.append(contentsOf: sample.ownershipTarget)
            ownershipValidFlags.append(sample.ownershipValid)
        }
        self.init(
            nnLen: nnLen, sampleCount: samples.count,
            spatialStride: spatialStride, globalStride: 19, policyStride: policyStride,
            valueStride: 3, ownershipStride: area,
            spatialBytes: spatialBytes, globalValues: globalValues, policyValues: policyValues,
            valueTargets: valueTargets, scoreTargets: scoreTargets, scoreValidFlags: scoreValidFlags,
            ownershipValues: ownershipValues, ownershipValidFlags: ownershipValidFlags
        )
    }

    /// Lazy `RandomAccessCollection` of the shard's samples. Each subscript materializes exactly one
    /// `RinGoSample` from the columnar buffers (converting its spatial u8 → Float); the whole
    /// collection is never held resident. Supports `.count`, `.isEmpty`, subscripting, ranges and
    /// iteration — everything `EvaluateCommand` and the tests need from `shard.samples`.
    public var samples: RinGoSampleCollection {
        RinGoSampleCollection(shard: self)
    }

    /// Materializes a single `RinGoSample` from the columnar store (spatial u8 → Float, 0 → 0.0,
    /// 1 → 1.0 exactly).
    func sample(at index: Int) -> RinGoSample {
        let sBase = index * spatialStride
        let spatial = spatialBytes[sBase ..< sBase + spatialStride].map { Float($0) }
        let gBase = index * globalStride
        let pBase = index * policyStride
        let vBase = index * valueStride
        let oBase = index * ownershipStride
        return RinGoSample(
            spatial: spatial,
            global: Array(globalValues[gBase ..< gBase + globalStride]),
            policyTarget: Array(policyValues[pBase ..< pBase + policyStride]),
            valueTarget: Array(valueTargets[vBase ..< vBase + valueStride]),
            scoreTarget: scoreTargets[index],
            scoreValid: scoreValidFlags[index],
            ownershipTarget: Array(ownershipValues[oBase ..< oBase + ownershipStride]),
            ownershipValid: ownershipValidFlags[index]
        )
    }

    /// Assembles a `TrainingBatch` directly from the columnar store for the given sample indices,
    /// converting spatial u8 → Float32 inline. This is the training/validation hot path: it never
    /// materializes intermediate `RinGoSample` objects. The result is bit-identical to
    /// `Trainer.batch(samples: indices.map { sample(at: $0) }, nnLen: nnLen)` — see the identity
    /// test — because u8 0/1 maps to Float 0.0/1.0 exactly and every other field is copied verbatim.
    public func batch(indices: [Int]) -> TrainingBatch {
        let n = indices.count
        let area = nnLen * nnLen
        var spatial = [Float](); spatial.reserveCapacity(n * spatialStride)
        var global = [Float](); global.reserveCapacity(n * globalStride)
        var policy = [Float](); policy.reserveCapacity(n * policyStride)
        var value = [Float](); value.reserveCapacity(n * valueStride)
        var score = [Float](); score.reserveCapacity(n)
        var scoreValid = [Float](); scoreValid.reserveCapacity(n)
        var ownership = [Float](); ownership.reserveCapacity(n * ownershipStride)
        var ownershipValid = [Float](); ownershipValid.reserveCapacity(n)
        for index in indices {
            let sBase = index * spatialStride
            for k in 0 ..< spatialStride {
                spatial.append(Float(spatialBytes[sBase + k]))
            }
            let gBase = index * globalStride
            for k in 0 ..< globalStride {
                global.append(globalValues[gBase + k])
            }
            let pBase = index * policyStride
            for k in 0 ..< policyStride {
                policy.append(policyValues[pBase + k])
            }
            let vBase = index * valueStride
            for k in 0 ..< valueStride {
                value.append(valueTargets[vBase + k])
            }
            score.append(scoreTargets[index])
            scoreValid.append(scoreValidFlags[index] ? 1 : 0)
            let oBase = index * ownershipStride
            for k in 0 ..< ownershipStride {
                ownership.append(Float(ownershipValues[oBase + k]))
            }
            ownershipValid.append(ownershipValidFlags[index] ? 1 : 0)
        }
        return TrainingBatch(
            spatial: MLXArray(spatial, [n, nnLen, nnLen, 22]),
            global: MLXArray(global, [n, globalStride]),
            policyTarget: MLXArray(policy, [n, area + 1]),
            valueTarget: MLXArray(value, [n, valueStride]),
            scoreTarget: MLXArray(score),
            scoreValidTarget: MLXArray(scoreValid),
            ownershipTarget: MLXArray(ownership, [n, nnLen, nnLen, 1]),
            ownershipValidTarget: MLXArray(ownershipValid)
        )
    }

    /// Reserves room for `additional` more samples across every column so `RinGoDataset.load` can
    /// append hundreds of shards without repeatedly reallocating (and briefly ~doubling) the multi-GB
    /// spatial buffer.
    mutating func reserveCapacity(forAdditionalSamples additional: Int) {
        spatialBytes.reserveCapacity(spatialBytes.count + additional * spatialStride)
        globalValues.reserveCapacity(globalValues.count + additional * globalStride)
        policyValues.reserveCapacity(policyValues.count + additional * policyStride)
        valueTargets.reserveCapacity(valueTargets.count + additional * valueStride)
        scoreTargets.reserveCapacity(scoreTargets.count + additional)
        scoreValidFlags.reserveCapacity(scoreValidFlags.count + additional)
        ownershipValues.reserveCapacity(ownershipValues.count + additional * ownershipStride)
        ownershipValidFlags.reserveCapacity(ownershipValidFlags.count + additional)
    }

    /// Appends another shard's samples in place (both must share `nnLen`/strides — the caller checks).
    mutating func appendSamples(from other: RinGoShard) {
        spatialBytes.append(contentsOf: other.spatialBytes)
        globalValues.append(contentsOf: other.globalValues)
        policyValues.append(contentsOf: other.policyValues)
        valueTargets.append(contentsOf: other.valueTargets)
        scoreTargets.append(contentsOf: other.scoreTargets)
        scoreValidFlags.append(contentsOf: other.scoreValidFlags)
        ownershipValues.append(contentsOf: other.ownershipValues)
        ownershipValidFlags.append(contentsOf: other.ownershipValidFlags)
        sampleCount += other.sampleCount
    }

    /// The on-disk byte size of one sample in this shard's format (used only to size load-time
    /// capacity reservations from file sizes). floats (spatial + global + policy + value + score)
    /// at 4 bytes, plus scoreValid (1) + ownership Int8 (area) + ownershipValid (1).
    var onDiskSampleByteCount: Int {
        (spatialStride + globalStride + policyStride + valueStride + 1) * MemoryLayout<Float>.size
            + 1 + ownershipStride + 1
    }

    /// Maps a single spatial fp32 value to its packed `UInt8`, throwing (with the offending
    /// sample/position/plane) if it is not exactly 0 or 1. `local` is the flat index within the
    /// sample's spatial block; RinGoData spatial is NHWC (channel-innermost) so `plane = local %
    /// planeCount` and `position = local / planeCount`.
    static func binaryPlaneByte(_ value: Float, sampleIndex: Int, local: Int, planeCount: Int) throws -> UInt8 {
        if value == 0 { return 0 }
        if value == 1 { return 1 }
        let plane = local % planeCount
        let position = local / planeCount
        throw RinGoShardError(
            "RinGoData spatial plane must be binary 0/1 (9x9 V7 features are all binary indicators), "
                + "but sample \(sampleIndex) position \(position) plane \(plane) = \(value). Refusing to "
                + "quantize a continuous feature; regenerate the shard or investigate the producer."
        )
    }

    /// Reads a RinGoData v2 shard (see `Scripts/ringo-data-format.md` and
    /// `RinGoEngine.RinGoDataFormat`) straight into columnar storage. v1 shards (dropped one-hot
    /// policy index, no validity flags) are rejected outright -- there is no backward-compat path;
    /// regenerate with the current `ringo makedata` (~23 minutes for the full corpus).
    public static func read(data: Data) throws -> RinGoShard {
        var reader = LittleEndianReader(data: data)
        // Legacy magic retained for compatibility with existing shards.
        guard try reader.hasMagic("NNGD") else {
            throw RinGoShardError("Invalid RinGoData magic; expected NNGD")
        }
        let version = try reader.uint32(field: "version")
        guard version == 2 else {
            throw RinGoShardError(
                "Unsupported RinGoData version \(version); expected 2. v1 shards are no longer "
                    + "supported -- regenerate with the current `ringo makedata` "
                    + "(~23 minutes for the full corpus)."
            )
        }
        let nnLen = try Int(reader.uint32(field: "nnLen"))
        guard nnLen > 0 else { throw RinGoShardError("RinGoData nnLen must be positive") }
        guard try reader.uint32(field: "spatial channel count") == 22 else {
            throw RinGoShardError("RinGoData requires 22 spatial channels")
        }
        guard try reader.uint32(field: "global channel count") == 19 else {
            throw RinGoShardError("RinGoData requires 19 global channels")
        }
        let sampleCount = try Int(reader.uint32(field: "sample count"))
        let boardArea = try checkedMultiply(nnLen, nnLen)
        let spatialStride = try checkedMultiply(22, boardArea)
        let policyStride = try checkedAdd(boardArea, 1)

        var spatialBytes = [UInt8](); try spatialBytes.reserveCapacity(checkedMultiply(sampleCount, spatialStride))
        var globalValues = [Float](); try globalValues.reserveCapacity(checkedMultiply(sampleCount, 19))
        var policyValues = [Float](); try policyValues.reserveCapacity(checkedMultiply(sampleCount, policyStride))
        var valueTargets = [Float](); try valueTargets.reserveCapacity(checkedMultiply(sampleCount, 3))
        var scoreTargets = [Float](); scoreTargets.reserveCapacity(sampleCount)
        var scoreValidFlags = [Bool](); scoreValidFlags.reserveCapacity(sampleCount)
        var ownershipValues = [Int8](); try ownershipValues.reserveCapacity(checkedMultiply(sampleCount, boardArea))
        var ownershipValidFlags = [Bool](); ownershipValidFlags.reserveCapacity(sampleCount)

        for sampleIndex in 0 ..< sampleCount {
            try reader.appendBinarySpatial(
                count: spatialStride, planeCount: 22, sampleIndex: sampleIndex, into: &spatialBytes
            )
            try reader.appendFloat32Array(
                count: 19, field: "global features for sample \(sampleIndex)", into: &globalValues
            )
            try reader.appendFloat32Array(
                count: policyStride, field: "policy target for sample \(sampleIndex)", into: &policyValues
            )
            try reader.appendFloat32Array(
                count: 3, field: "value target for sample \(sampleIndex)", into: &valueTargets
            )
            try scoreTargets.append(reader.float32(field: "score target for sample \(sampleIndex)"))
            try scoreValidFlags.append(reader.bool(field: "score validity flag for sample \(sampleIndex)"))
            try reader.appendInt8Array(
                count: boardArea, field: "ownership target for sample \(sampleIndex)", into: &ownershipValues
            )
            try ownershipValidFlags.append(
                reader.bool(field: "ownership validity flag for sample \(sampleIndex)")
            )
        }
        guard reader.isAtEnd else { throw RinGoShardError("Unexpected trailing bytes in RinGoData shard") }
        return RinGoShard(
            nnLen: nnLen, sampleCount: sampleCount,
            spatialStride: spatialStride, globalStride: 19, policyStride: policyStride,
            valueStride: 3, ownershipStride: boardArea,
            spatialBytes: spatialBytes, globalValues: globalValues, policyValues: policyValues,
            valueTargets: valueTargets, scoreTargets: scoreTargets, scoreValidFlags: scoreValidFlags,
            ownershipValues: ownershipValues, ownershipValidFlags: ownershipValidFlags
        )
    }

    public static func read(url: URL) throws -> RinGoShard {
        try read(data: Data(contentsOf: url))
    }
}

/// Lazy view over a `RinGoShard`'s samples (see `RinGoShard.samples`). Holds the shard by value
/// (its columnar buffers are COW, so this is a cheap retain, not a copy) and materializes one
/// `RinGoSample` per subscript.
public struct RinGoSampleCollection: RandomAccessCollection, Sendable {
    public typealias Element = RinGoSample
    public typealias Index = Int

    let shard: RinGoShard

    public var startIndex: Int {
        0
    }

    public var endIndex: Int {
        shard.sampleCount
    }

    public func index(after i: Int) -> Int {
        i + 1
    }

    public func index(before i: Int) -> Int {
        i - 1
    }

    public subscript(position: Int) -> RinGoSample {
        shard.sample(at: position)
    }
}

public struct RinGoShardError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

public enum RinGoDataset {
    /// Loads every `.nngd`/`.bin` shard in `directory` and concatenates their samples into a single
    /// columnar `RinGoShard` (appended in place shard-by-shard so only one shard's raw `Data` is
    /// resident at a time, and the combined buffers are pre-reserved from file sizes to avoid
    /// reallocating the multi-GB spatial store 900+ times).
    ///
    /// - Parameter excluding: files to skip even though they live in `directory` and match the
    ///   shard extensions. This exists so the held-out validation shard (`val.nngd`, see
    ///   `TrainCommand.loadValidationShard`) never leaks into the training set when it lives in
    ///   the same directory as the training shards — the leakage bug this guards: without this, a
    ///   directory containing both training shards and `val.nngd` would train on validation data
    ///   too, silently invalidating the holdout. Paths are compared via `standardizedFileURL` so
    ///   callers don't need to worry about `..`/symlink formatting differences between how the
    ///   directory listing and the caller constructed the excluded URL.
    public static func load(directory: URL, excluding excludedURLs: Set<URL> = []) throws -> RinGoShard {
        let excludedPaths = Set(excludedURLs.map(\.standardizedFileURL.path))
        let urls = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ).filter { url in
            (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
                && ["nngd", "bin"].contains(url.pathExtension.lowercased())
                && !excludedPaths.contains(url.standardizedFileURL.path)
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard let firstURL = urls.first else {
            throw RinGoShardError("No .nngd or .bin shards found in \(directory.path)")
        }
        var combined = try RinGoShard.read(url: firstURL)
        let remaining = urls.dropFirst()
        if !remaining.isEmpty {
            // Reserve up front using the first shard's per-sample byte size. A size mismatch only
            // makes this a hint (append still grows as needed), never wrong.
            let sampleBytes = combined.onDiskSampleByteCount
            var estimatedRemaining = 0
            if sampleBytes > 0 {
                for url in remaining {
                    let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? nil
                    if let size, size > RinGoShard.headerByteCount {
                        estimatedRemaining += (size - RinGoShard.headerByteCount) / sampleBytes
                    }
                }
            }
            combined.reserveCapacity(forAdditionalSamples: estimatedRemaining)
            for url in remaining {
                let next = try RinGoShard.read(url: url)
                guard next.nnLen == combined.nnLen else {
                    throw RinGoShardError("All RinGoData shards must have the same nnLen")
                }
                combined.appendSamples(from: next)
            }
        }
        return combined
    }
}

private struct LittleEndianReader {
    let data: Data
    var offset = 0
    var isAtEnd: Bool {
        offset == data.count
    }

    var remainingCount: Int {
        data.count - offset
    }

    mutating func hasMagic(_ magic: StaticString) throws -> Bool {
        let count = magic.utf8CodeUnitCount
        try requireBytes(count, field: "magic")
        let start = offset
        offset += count
        return data.withUnsafeBytes { source in
            magic.withUTF8Buffer { expected in
                source[start ..< start + count].elementsEqual(expected)
            }
        }
    }

    mutating func uint32(field: String) throws -> UInt32 {
        try loadUnaligned(as: UInt32.self, field: field)
    }

    mutating func float32(field: String) throws -> Float {
        try Float(bitPattern: loadUnaligned(as: UInt32.self, field: field))
    }

    /// A single-byte boolean flag (0 = false, any other value = true), used for the v2 per-sample
    /// `scoreValid`/`ownershipValid` flags.
    mutating func bool(field: String) throws -> Bool {
        try requireBytes(1, field: field)
        let start = offset
        offset += 1
        return data.withUnsafeBytes { (source: UnsafeRawBufferPointer) in source[start] != 0 }
    }

    /// Reads `count` fp32 spatial values, validating each is exactly 0 or 1 and packing it as a
    /// `UInt8` appended to `destination` (which must be pre-reserved). Fails loudly on any
    /// non-binary value — see `RinGoShard.binaryPlaneByte`.
    mutating func appendBinarySpatial(
        count: Int, planeCount: Int, sampleIndex: Int, into destination: inout [UInt8]
    ) throws {
        let byteCount = try checkedMultiply(count, MemoryLayout<UInt32>.size)
        try requireBytes(byteCount, field: "spatial features for sample \(sampleIndex)")
        let start = offset
        offset += byteCount
        try data.withUnsafeBytes { source in
            for local in 0 ..< count {
                let bits = source.loadUnaligned(
                    fromByteOffset: start + local * MemoryLayout<UInt32>.size, as: UInt32.self
                )
                try destination.append(
                    RinGoShard.binaryPlaneByte(
                        Float(bitPattern: bits), sampleIndex: sampleIndex, local: local, planeCount: planeCount
                    )
                )
            }
        }
    }

    mutating func appendFloat32Array(count: Int, field: String, into destination: inout [Float]) throws {
        let byteCount = try checkedMultiply(count, MemoryLayout<UInt32>.size)
        try requireBytes(byteCount, field: field)
        let start = offset
        offset += byteCount
        // RinGoData is little-endian, as are supported ARM64/x86_64 hosts, so these unaligned
        // loads produce the on-disk float bit patterns directly (see `loadUnaligned`).
        data.withUnsafeBytes { source in
            for local in 0 ..< count {
                let bits = source.loadUnaligned(
                    fromByteOffset: start + local * MemoryLayout<UInt32>.size, as: UInt32.self
                )
                destination.append(Float(bitPattern: bits))
            }
        }
    }

    mutating func appendInt8Array(count: Int, field: String, into destination: inout [Int8]) throws {
        try requireBytes(count, field: field)
        let start = offset
        offset += count
        data.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
            for local in 0 ..< count {
                destination.append(Int8(bitPattern: source[start + local]))
            }
        }
    }

    private mutating func loadUnaligned<T>(as type: T.Type, field: String) throws -> T {
        let count = MemoryLayout<T>.size
        try requireBytes(count, field: field)
        let start = offset
        offset += count
        // RinGoData is little-endian. Supported ARM64/x86_64 hosts are also little-endian,
        // so these unaligned loads produce the on-disk integer and float bit patterns directly;
        // supporting a hypothetical big-endian host would require byte-swapping these loads.
        return data.withUnsafeBytes { source in
            source.loadUnaligned(fromByteOffset: start, as: type)
        }
    }

    private func requireBytes(_ count: Int, field: String) throws {
        guard count >= 0, offset <= data.count, count <= data.count - offset else {
            let remaining = max(0, data.count - min(offset, data.count))
            throw RinGoShardError(
                "Truncated RinGoData shard while reading \(field) at byte \(offset): "
                    + "expected \(count) bytes, only \(remaining) remain"
            )
        }
    }
}

private func checkedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else { throw RinGoShardError("RinGoData dimensions overflow") }
    return result.partialValue
}

private func checkedAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else { throw RinGoShardError("RinGoData dimensions overflow") }
    return result.partialValue
}
