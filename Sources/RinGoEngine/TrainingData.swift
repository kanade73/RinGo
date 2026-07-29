import Foundation
import RinGoCore
import RinGoModel

/// RinGoData v2
///
/// All multibyte values are little-endian. There is one file per shard. v2 is NOT backward
/// compatible with v1 (dropped the one-hot policy index and added per-sample validity flags for
/// distillation support); a v1 shard must be regenerated with
/// the current `ringo makedata` (~23 minutes for the full corpus) -- `TrainingShardReader`
/// rejects any other version with an explicit error rather than silently misreading the bytes.
///
/// Header, in order:
/// - 4 bytes: ASCII magic `NNGD`
/// - u32: version, always 2
/// - u32: nnLen
/// - u32: numSpatial, always 22
/// - u32: numGlobal, always 19
/// - u32: sampleCount
///
/// Each sample, in order:
/// - f32[22*nnLen*nnLen]: spatial features in NHWC order from fillRowV7, from the side-to-move perspective
/// - f32[19]: global features
/// - f32[nnLen*nnLen+1]: policyTarget, a dense probability distribution over all moves;
/// 0...nnLen*nnLen-1 is y*nnLen+x, and nnLen*nnLen is pass. Teacherless `makedata` writes a
/// one-hot vector (1.0 at the played move); a teacher-distilled shard may write a soft
/// distribution instead.
/// - f32[3]: valueTarget in win, loss, noResult order, from the side-to-move perspective. May be a
/// soft probability (e.g. a teacher's win-rate estimate) rather than a one-hot outcome.
/// - f32: scoreTarget, final score difference including komi, from the side-to-move perspective
/// - u8: scoreValid; 1 if scoreTarget reflects a genuine scored/estimated result, 0 if it is a
/// synthetic placeholder (e.g. the resignation proxy) that consumers must exclude from the score
/// loss and its normalization
/// - i8[nnLen*nnLen]: ownershipTarget; +1 is side-to-move owned, -1 is opponent owned, and 0 is
/// neutral/dame/seki-shared, using final-position area classification
/// - u8: ownershipValid; 1 if ownershipTarget reflects a genuine final-position label, 0 if it
/// must be excluded from the ownership loss and its normalization (e.g. a resignation, whose final
/// board position is not a reliable ownership label)
///
/// Jigo games are skipped because there is no draw value-target component. For resignations, the
/// score target is a deliberately synthetic, clamped proxy: +15.0 for the winner and -15.0 for
/// the loser, and BOTH `scoreValid` and `ownershipValid` are false.
public enum RinGoDataFormat {
    public static let magic = Data("NNGD".utf8)
    public static let version: UInt32 = 2
    public static let numSpatial: UInt32 = .init(NNInputs.numFeaturesSpatialV7)
    public static let numGlobal: UInt32 = .init(NNInputs.numFeaturesGlobalV7)
    public static let headerSize = 24

    public static func sampleSize(nnLen: Int) -> Int {
        let area = nnLen * nnLen
        let floatCount = Int(numSpatial) * area + Int(numGlobal) + (area + 1) + 3 + 1
        return floatCount * MemoryLayout<Float>.size
            + MemoryLayout<UInt8>.size // scoreValid
            + area * MemoryLayout<Int8>.size // ownershipTarget
            + MemoryLayout<UInt8>.size // ownershipValid
    }
}

public struct TrainingSample: Equatable, Sendable {
    public var spatial: [Float]
    public var global: [Float]
    /// Dense probability distribution over all `nnLen*nnLen+1` moves (board points + pass).
    public var policyTarget: [Float]
    public var valueTarget: [Float]
    public var scoreTarget: Float
    /// False for samples whose `scoreTarget` is a synthetic placeholder (see the format doc above).
    public var scoreValid: Bool
    public var ownershipTarget: [Int8]
    /// False for samples whose `ownershipTarget` is not a reliable final-position label.
    public var ownershipValid: Bool

    public init(
        spatial: [Float],
        global: [Float],
        policyTarget: [Float],
        valueTarget: [Float],
        scoreTarget: Float,
        scoreValid: Bool,
        ownershipTarget: [Int8],
        ownershipValid: Bool
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

public struct TrainingShardHeader: Equatable, Sendable {
    public let version: UInt32
    public let nnLen: Int
    public let numSpatial: Int
    public let numGlobal: Int
    public let sampleCount: Int
}

public enum TrainingDataError: Error, CustomStringConvertible, Equatable {
    case invalidArgument(String)
    case invalidSample(String)
    case invalidShard(String)

    public var description: String {
        switch self {
        case let .invalidArgument(message), let .invalidSample(message), let .invalidShard(message): message
        }
    }
}

public enum TrainingSymmetry: Int, CaseIterable, Sendable {
    case identity = 0
    case rotate90 = 1
    case rotate180 = 2
    case rotate270 = 3
    case reflect = 4
    case reflectRotate90 = 5
    case reflectRotate180 = 6
    case reflectRotate270 = 7

    public func transform(x: Int, y: Int, length: Int) -> (x: Int, y: Int) {
        let last = length - 1
        return switch self {
        case .identity: (x, y)
        case .rotate90: (last - y, x)
        case .rotate180: (last - x, last - y)
        case .rotate270: (y, last - x)
        case .reflect: (last - x, y)
        case .reflectRotate90: (last - y, last - x)
        case .reflectRotate180: (x, last - y)
        case .reflectRotate270: (y, x)
        }
    }

    /// Permutes a dense `nnLen*nnLen+1` policy distribution consistently with the board transform
    /// this case applies to spatial features / ownership. The trailing pass entry (index `area`) is
    /// always invariant -- pass is not a board point, so no symmetry moves it.
    public func transform(policy: [Float], nnLen: Int) -> [Float] {
        // Identity is the permutation that moves nothing: return the input unchanged instead of
        // copying it entry by entry. This is the hot path for teacherless/distillation
        // generation, which runs with `symmetries == [.identity]`.
        if self == .identity { return policy }
        let area = nnLen * nnLen
        precondition(policy.count == area + 1, "policy distribution must have nnLen*nnLen+1 entries")
        var result = [Float](repeating: 0, count: area + 1)
        for source in 0 ..< area {
            let point = transform(x: source % nnLen, y: source / nnLen, length: nnLen)
            result[point.y * nnLen + point.x] = policy[source]
        }
        result[area] = policy[area]
        return result
    }

    public func transform(sample: TrainingSample, nnLen: Int) -> TrainingSample {
        // Identity moves no board point: skip the element-by-element spatial/ownership copy.
        // `TrainingSample`'s arrays are copy-on-write, so returning the input is byte-identical
        // to the permuted-copy path and never aliases mutable state.
        if self == .identity { return sample }
        let area = nnLen * nnLen
        var spatial = [Float](repeating: 0, count: sample.spatial.count)
        var ownership = [Int8](repeating: 0, count: area)
        for source in 0 ..< area {
            let point = transform(x: source % nnLen, y: source / nnLen, length: nnLen)
            let destination = point.y * nnLen + point.x
            let sourceOffset = source * Int(RinGoDataFormat.numSpatial)
            let destinationOffset = destination * Int(RinGoDataFormat.numSpatial)
            for feature in 0 ..< Int(RinGoDataFormat.numSpatial) {
                spatial[destinationOffset + feature] = sample.spatial[sourceOffset + feature]
            }
            ownership[destination] = sample.ownershipTarget[source]
        }
        return TrainingSample(
            spatial: spatial,
            global: sample.global,
            policyTarget: transform(policy: sample.policyTarget, nnLen: nnLen),
            valueTarget: sample.valueTarget,
            scoreTarget: sample.scoreTarget,
            scoreValid: sample.scoreValid,
            ownershipTarget: ownership,
            ownershipValid: sample.ownershipValid
        )
    }
}

public enum TrainingGameResult: Equatable, Sendable {
    case scored(winner: Player, whiteMinusBlack: Float)
    case resignation(winner: Player)
    case jigo

    public static let resignationScoreProxy: Float = 15

    public static func parse(_ raw: String) -> TrainingGameResult? {
        let result = raw.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        if ["0", "DRAW", "JIGO", "B+0", "W+0"].contains(result) { return .jigo }
        guard result.count > 2, result[result.index(result.startIndex, offsetBy: 1)] == "+" else { return nil }
        let winner: Player
        switch result.first {
        case "B": winner = .black
        case "W": winner = .white
        default: return nil
        }
        let detail = String(result.dropFirst(2))
        if detail == "R" || detail == "RESIGN" || detail == "RESIGNATION" {
            return .resignation(winner: winner)
        }
        guard let margin = Float(detail), margin.isFinite, margin >= 0 else { return nil }
        if margin == 0 { return .jigo }
        return .scored(winner: winner, whiteMinusBlack: winner == .white ? margin : -margin)
    }

    public var winner: Player? {
        switch self {
        case let .scored(winner, _), let .resignation(winner): winner
        case .jigo: nil
        }
    }

    public var whiteMinusBlackScore: Float? {
        switch self {
        case let .scored(_, score): score
        case let .resignation(winner): winner == .white ? Self.resignationScoreProxy : -Self.resignationScoreProxy
        case .jigo: nil
        }
    }
}

public enum TrainingShardWriter {
    @discardableResult
    public static func write(samples: [TrainingSample], nnLen: Int, to url: URL) throws -> TrainingShardHeader {
        guard nnLen > 0, nnLen <= Board.maxLen else {
            throw TrainingDataError.invalidArgument("nnLen must be in 1...\(Board.maxLen)")
        }
        guard samples.count <= Int(UInt32.max) else {
            throw TrainingDataError.invalidArgument("A shard cannot contain more than UInt32.max samples")
        }
        let area = nnLen * nnLen
        let expectedSpatial = Int(RinGoDataFormat.numSpatial) * area
        // Validate every sample's field sizes up front, before any bytes are produced, so a
        // malformed sample throws exactly the same `invalidSample` errors the old per-scalar writer
        // did (the bulk fill below assumes these invariants hold).
        for sample in samples {
            guard sample.spatial.count == expectedSpatial else {
                throw TrainingDataError.invalidSample("Spatial feature count does not match nnLen")
            }
            guard sample.global.count == Int(RinGoDataFormat.numGlobal) else {
                throw TrainingDataError.invalidSample("Global feature count is not 19")
            }
            guard sample.policyTarget.count == area + 1 else {
                throw TrainingDataError.invalidSample("Policy target count is not nnLen*nnLen+1")
            }
            guard sample.valueTarget.count == 3 else {
                throw TrainingDataError.invalidSample("Value target count is not 3")
            }
            guard sample.ownershipTarget.count == area else {
                throw TrainingDataError.invalidSample("Ownership target count does not match nnLen")
            }
        }

        // Bulk serialization: fill one preallocated contiguous buffer with per-field
        // memcpy/stores instead of ~2,000 `Data.append` calls per sample (~8 billion for the full
        // corpus -- the write-side twin of the already-fixed 55x read-side bug). RinGoData is
        // little-endian and every supported host (arm64/x86_64) is little-endian, so a `[Float]`/
        // `[Int8]` already holds exactly the on-disk byte order and copies straight through; the u32
        // header fields and the per-sample score float are stored through their `littleEndian` bit
        // pattern to stay correct even on a hypothetical big-endian host (mirrors the read side in
        // `RinGoShard`). The output is byte-identical to the old writer (pinned by the golden test
        // in `TrainingDataFormatTests`).
        let totalSize = RinGoDataFormat.headerSize + samples.count * RinGoDataFormat.sampleSize(nnLen: nnLen)
        var data = Data(count: totalSize)
        data.withUnsafeMutableBytes { rawBuffer in
            var writer = LittleEndianByteWriter(buffer: rawBuffer)
            writer.putBytes(RinGoDataFormat.magic)
            writer.putUInt32(RinGoDataFormat.version)
            writer.putUInt32(UInt32(nnLen))
            writer.putUInt32(RinGoDataFormat.numSpatial)
            writer.putUInt32(RinGoDataFormat.numGlobal)
            writer.putUInt32(UInt32(samples.count))
            for sample in samples {
                writer.putFloats(sample.spatial)
                writer.putFloats(sample.global)
                writer.putFloats(sample.policyTarget)
                writer.putFloats(sample.valueTarget)
                writer.putFloat(sample.scoreTarget)
                writer.putByte(sample.scoreValid ? 1 : 0)
                writer.putInt8s(sample.ownershipTarget)
                writer.putByte(sample.ownershipValid ? 1 : 0)
            }
        }
        try data.write(to: url, options: .atomic)
        return TrainingShardHeader(
            version: RinGoDataFormat.version,
            nnLen: nnLen,
            numSpatial: Int(RinGoDataFormat.numSpatial),
            numGlobal: Int(RinGoDataFormat.numGlobal),
            sampleCount: samples.count
        )
    }
}

public struct TrainingShardReader: Sendable {
    public let header: TrainingShardHeader
    private let data: Data

    public init(url: URL) throws {
        try self.init(data: Data(contentsOf: url))
    }

    public init(data: Data) throws {
        guard data.count >= RinGoDataFormat.headerSize else {
            throw TrainingDataError.invalidShard("Shard is shorter than its header")
        }
        guard data.prefix(4) == RinGoDataFormat.magic else {
            throw TrainingDataError.invalidShard("Shard magic is not NNGD")
        }
        let version = data.readUInt32(at: 4)
        let nnLen = Int(data.readUInt32(at: 8))
        let numSpatial = Int(data.readUInt32(at: 12))
        let numGlobal = Int(data.readUInt32(at: 16))
        let sampleCount = Int(data.readUInt32(at: 20))
        guard version == RinGoDataFormat.version else {
            throw TrainingDataError.invalidShard(
                "Unsupported RinGoData version \(version); expected \(RinGoDataFormat.version). "
                    + "v1 shards are no longer supported -- regenerate with the current `ringo "
                    + "makedata` (~23 minutes for the full corpus)."
            )
        }
        guard nnLen > 0, nnLen <= Board.maxLen else {
            throw TrainingDataError.invalidShard("Invalid nnLen \(nnLen)")
        }
        guard numSpatial == Int(RinGoDataFormat.numSpatial), numGlobal == Int(RinGoDataFormat.numGlobal) else {
            throw TrainingDataError.invalidShard("Feature counts are not 22 spatial and 19 global")
        }
        let expectedSize = RinGoDataFormat.headerSize + sampleCount * RinGoDataFormat.sampleSize(nnLen: nnLen)
        guard data.count == expectedSize else {
            throw TrainingDataError.invalidShard("Shard size is \(data.count) bytes; expected \(expectedSize)")
        }
        header = TrainingShardHeader(
            version: version,
            nnLen: nnLen,
            numSpatial: numSpatial,
            numGlobal: numGlobal,
            sampleCount: sampleCount
        )
        self.data = data
    }

    public func sample(at index: Int) throws -> TrainingSample {
        guard index >= 0, index < header.sampleCount else {
            throw TrainingDataError.invalidArgument("Sample index is outside 0..<\(header.sampleCount)")
        }
        let area = header.nnLen * header.nnLen
        var offset = RinGoDataFormat.headerSize + index * RinGoDataFormat.sampleSize(nnLen: header.nnLen)
        let spatial = readFloats(count: header.numSpatial * area, offset: &offset)
        let global = readFloats(count: header.numGlobal, offset: &offset)
        let policy = readFloats(count: area + 1, offset: &offset)
        let value = readFloats(count: 3, offset: &offset)
        let score = data.readFloat(at: offset)
        offset += MemoryLayout<Float>.size
        let scoreValid = data[offset] != 0
        offset += MemoryLayout<UInt8>.size
        let ownership = (0 ..< area).map { Int8(bitPattern: data[offset + $0]) }
        offset += area
        let ownershipValid = data[offset] != 0
        return TrainingSample(
            spatial: spatial,
            global: global,
            policyTarget: policy,
            valueTarget: value,
            scoreTarget: score,
            scoreValid: scoreValid,
            ownershipTarget: ownership,
            ownershipValid: ownershipValid
        )
    }

    private func readFloats(count: Int, offset: inout Int) -> [Float] {
        let result = (0 ..< count).map { data.readFloat(at: offset + $0 * MemoryLayout<Float>.size) }
        offset += count * MemoryLayout<Float>.size
        return result
    }
}

public struct TrainingDataOptions: Sendable {
    public var sgfDirectory: URL
    public var outputDirectory: URL
    public var nnLen: Int
    public var shardSize: Int
    public var rules: Rules
    public var minimumMoves: Int
    public var symmetries: Int
    /// Fraction of SGF files (in [0, 1)) to reserve as a held-out validation shard.
    /// When > 0, `generate` deterministically streams the chosen fraction (file order is the same
    /// every run via a stateful shuffle seeded from a fixed value) into a single `val.nngd`,
    /// while the rest of the files flow into the existing `shard-*.nngd` shards. When 0 (the
    /// default), the val shard is skipped, preserving the historical one-set behavior.
    public var valRatio: Float

    public init(
        sgfDirectory: URL,
        outputDirectory: URL,
        nnLen: Int = 9,
        shardSize: Int = 4096,
        rules: Rules = Rules(
            koRule: .simple,
            scoringRule: .area,
            taxRule: .none,
            multiStoneSuicideLegal: false,
            whiteHandicapBonusRule: .n,
            friendlyPassOk: true,
            komi: 7.5
        ),
        minimumMoves: Int = 8,
        symmetries: Int = 0,
        valRatio: Float = 0
    ) {
        self.sgfDirectory = sgfDirectory
        self.outputDirectory = outputDirectory
        self.nnLen = nnLen
        self.shardSize = shardSize
        self.rules = rules
        self.minimumMoves = minimumMoves
        self.symmetries = symmetries
        self.valRatio = valRatio
    }
}

public struct TrainingDataSummary: Equatable, Sendable {
    public var filesFound = 0
    public var gamesUsed = 0
    public var parseSkips = 0
    public var boardSizeSkips = 0
    public var minimumMoveSkips = 0
    public var jigoSkips = 0
    public var resultSkips = 0
    public var illegalMoveSkips = 0
    public var totalSamples = 0
    public var totalShards = 0
    public var elapsedSeconds = 0.0
    public var validationFiles = 0
    public var validationSamples = 0
    /// Teacher (distillation) mode only: number of raw positions labeled by the teacher network
    /// (pre-symmetry -- one teacher eval per raw position, see `generate(_:teacher:teacherBatchSize:)`).
    public var teacherPositionsLabeled = 0
    /// Teacher (distillation) mode only: wall-clock seconds spent inside teacher `evaluate` calls.
    /// `teacherPositionsLabeled / teacherSeconds` is the teacher labeling throughput.
    public var teacherSeconds = 0.0

    public var gamesSkipped: Int {
        parseSkips + boardSizeSkips + minimumMoveSkips + jigoSkips + resultSkips + illegalMoveSkips
    }
}

public enum TrainingDataPipeline {
    public static func generate(_ options: TrainingDataOptions) throws -> TrainingDataSummary {
        try validate(options)
        let start = Date()
        let fileManager = FileManager.default
        let (files, trainFiles, valFiles) = try prepareRun(options, fileManager: fileManager)

        var summary = TrainingDataSummary()
        summary.filesFound = files.count
        try processFiles(
            trainFiles, options: options, fileManager: fileManager,
            shardPrefix: "shard-", outputDirectory: options.outputDirectory,
            shardCap: options.shardSize, summary: &summary
        )
        if !valFiles.isEmpty {
            summary.validationFiles = valFiles.count
            let shardsBefore = summary.totalShards
            try processFiles(
                valFiles, options: options, fileManager: fileManager,
                shardPrefix: "val", outputDirectory: options.outputDirectory,
                shardCap: .max, summary: &summary
            )
            // The single val shard is rolled back out of `totalShards` so existing tooling reading
            // that field still sees exactly the train shard count (use `validationFiles` /
            // `validationSamples` to know whether a val.nngd was produced).
            summary.totalShards -= max(0, summary.totalShards - shardsBefore)
        }
        summary.elapsedSeconds = Date().timeIntervalSince(start)
        return summary
    }

    /// Distillation variant of `generate`: identical file selection,
    /// validation split, and shard layout as the teacherless path, but every extracted position's
    /// targets are REPLACED by the `teacher` network's outputs (soft policy, value distribution,
    /// scoremean, and quantized ownership -- see `TeacherLabeler`). Positions are labeled BEFORE
    /// symmetry expansion (one teacher eval per raw position; the v2 symmetry transforms then
    /// permute the dense targets correctly), and evaluated in batches of `teacherBatchSize`.
    ///
    /// Teacher labels are trustworthy for ANY position regardless of how the game ended, so both
    /// `scoreValid` and `ownershipValid` are set true for every distilled sample -- this is exactly
    /// what makes distillation sidestep the resignation-ownership problem.
    ///
    /// `serial` selects between two byte-identical implementations: the default overlapped pipeline
    /// (CPU parse/feature-fill of the next batch and shard serialization of the
    /// previous batch run concurrently with the teacher's GPU forward of the current batch) and the
    /// legacy strict CPU<->GPU alternation, kept as an `-serial` escape hatch for A/B measurement and
    /// as the parity oracle. Both paths emit the same sample order, targets, shard boundaries, and
    /// val-split membership.
    ///
    /// `teacherVisits` selects the label SOURCE (Phase P). `teacherVisits <= 1` is the raw path
    /// above (one teacher forward per position, targets from the net output). `teacherVisits >= 2`
    /// switches to SEARCH labeling: each position is labeled by an N-visit PUCT search driven by the
    /// same teacher, whose normalized root visit distribution becomes the policy target and whose
    /// backed-up root winrate becomes the value target (score/ownership stay from the raw root net --
    /// see `TeacherLabeler.searchTargets`). Search mode is inherently ~N sequential-ish evals per
    /// position, so it runs the serial path (the per-position search dominates cost; pipelining the
    /// CPU parse/serialize around it buys little, and the `serial` flag is ignored in this mode);
    /// batching happens WITHIN each search via the evaluator's leaf batches. `precision` and
    /// `fp32FallbackFactory` configure the per-position `Search` (matching how production search is
    /// built) and are unused by the raw path.
    public static func generate(
        _ options: TrainingDataOptions,
        teacher: NNEvaluating,
        teacherBatchSize: Int,
        serial: Bool = false,
        teacherVisits: Int = 1,
        teacherTarget: TeacherLabeler.PolicyTargetKind = .visits,
        searchSettings: SearchSettings = SearchSettings(),
        precision: KataGoNetwork.Precision = .float16,
        fp32FallbackFactory: (@Sendable () throws -> any NNEvaluating)? = nil
    ) async throws -> TrainingDataSummary {
        try validate(options)
        guard teacherBatchSize > 0 else {
            throw TrainingDataError.invalidArgument("teacherBatchSize must be positive")
        }
        guard teacherVisits >= 1 else {
            throw TrainingDataError.invalidArgument("teacherVisits must be >= 1")
        }
        let start = Date()
        let fileManager = FileManager.default
        let (files, trainFiles, valFiles) = try prepareRun(options, fileManager: fileManager)

        var summary = TrainingDataSummary()
        summary.filesFound = files.count

        func runPass(_ passFiles: [URL], shardPrefix: String, shardCap: Int) async throws {
            if teacherVisits >= 2 {
                try await processFilesWithTeacherSearch(
                    passFiles, options: options, teacher: teacher, teacherVisits: teacherVisits,
                    teacherTarget: teacherTarget, searchSettings: searchSettings, precision: precision,
                    fp32FallbackFactory: fp32FallbackFactory, shardPrefix: shardPrefix,
                    outputDirectory: options.outputDirectory, shardCap: shardCap, summary: &summary
                )
            } else {
                try await processFilesWithTeacher(
                    passFiles, options: options, teacher: teacher, teacherBatchSize: teacherBatchSize,
                    serial: serial, shardPrefix: shardPrefix, outputDirectory: options.outputDirectory,
                    shardCap: shardCap, summary: &summary
                )
            }
        }

        try await runPass(trainFiles, shardPrefix: "shard-", shardCap: options.shardSize)
        if !valFiles.isEmpty {
            summary.validationFiles = valFiles.count
            let shardsBefore = summary.totalShards
            try await runPass(valFiles, shardPrefix: "val", shardCap: .max)
            summary.totalShards -= max(0, summary.totalShards - shardsBefore)
        }
        summary.elapsedSeconds = Date().timeIntervalSince(start)
        return summary
    }

    private static func validate(_ options: TrainingDataOptions) throws {
        guard options.nnLen > 1, options.nnLen <= Board.maxLen else {
            throw TrainingDataError.invalidArgument("nnLen must be in 2...\(Board.maxLen)")
        }
        guard options.shardSize > 0 else { throw TrainingDataError.invalidArgument("shardSize must be positive") }
        guard options.minimumMoves >= 0 else {
            throw TrainingDataError.invalidArgument("minimumMoves cannot be negative")
        }
        guard options.symmetries == 0 || options.symmetries == 8 else {
            throw TrainingDataError.invalidArgument("symmetries must be 0 or 8")
        }
        guard options.valRatio >= 0, options.valRatio < 1, options.valRatio.isFinite else {
            throw TrainingDataError.invalidArgument("valRatio must be a finite float in [0, 1)")
        }
    }

    /// Enumerates the SGF corpus, prepares the output directory (creating it, clearing old train
    /// shards, and dropping any stale `val.nngd`), and splits the file list into train/val subsets.
    /// Shared by both the teacherless and teacher-distillation entry points so their run setup is
    /// byte-for-byte identical.
    private static func prepareRun(
        _ options: TrainingDataOptions,
        fileManager: FileManager
    ) throws -> (files: [URL], train: [URL], val: [URL]) {
        let files = try sgfFiles(in: options.sgfDirectory, fileManager: fileManager)
        try fileManager.createDirectory(at: options.outputDirectory, withIntermediateDirectories: true)
        try removeOldShards(in: options.outputDirectory, fileManager: fileManager)
        if options.valRatio > 0 {
            // A re-run with val-ratio set overwrites the prior val.nngd deterministically (the
            // shuffle is seeded), so drop a stale copy here to keep the output directory honest.
            let valURL = options.outputDirectory.appendingPathComponent("val.nngd")
            try? fileManager.removeItem(at: valURL)
        }

        // Validation split: only when valRatio > 0 is the file list shuffled. The shuffle is
        // seeded so a re-run with the same valRatio at the same files picks the SAME val set (a
        // drifting val set would silently make validation-loss curves uncomparable across resumes).
        // When valRatio == 0 we keep the historical behavior: process files in the lexicographic
        // order they were collected, no shuffle, no val.nngd.
        if options.valRatio > 0, files.count > 1 {
            let valCount = max(1, Int((Float(files.count) * options.valRatio).rounded(.down)))
            let (train, val) = dedupShuffleAndSplit(files: files, valCount: valCount)
            return (files, train, val)
        }
        return (files, files, [])
    }

    /// Process one file subset. Train pass writes shards `shard-%05d.nngd` (split at `shardCap`);
    /// validation pass (signalled by `shardCap == .max`) accumulates all samples and writes them to
    /// a single `val.nngd`. The summary accumulates skipping and sample/shard counts into the caller's
    /// summary; the caller (validation only) rolls back the single val shard from `totalShards`.
    @discardableResult
    private static func processFiles(
        _ files: [URL],
        options: TrainingDataOptions,
        fileManager _: FileManager,
        shardPrefix: String,
        outputDirectory: URL,
        shardCap: Int,
        summary: inout TrainingDataSummary
    ) throws -> TrainingDataSummary {
        let isValPass = shardCap == .max
        var shard = [TrainingSample]()
        shard.reserveCapacity(min(shardCap, options.shardSize))
        var shardIndex = summary.totalShards

        func flush() throws {
            let name = isValPass
                ? "val.nngd"
                : String(format: "%@%05d.nngd", shardPrefix, shardIndex)
            let url = outputDirectory.appendingPathComponent(name)
            try TrainingShardWriter.write(samples: shard, nnLen: options.nnLen, to: url)
            summary.totalShards += 1
            if !isValPass { shardIndex += 1 }
            shard.removeAll(keepingCapacity: true)
        }

        func append(_ sample: TrainingSample) throws {
            shard.append(sample)
            if !isValPass, shard.count == shardCap { try flush() }
        }

        for file in files {
            guard let prepared = try preparedGame(file: file, options: options, summary: &summary) else {
                continue
            }
            let generated = samples(
                game: prepared.game,
                positions: prepared.positions,
                result: prepared.result,
                finalArea: prepared.finalArea,
                nnLen: options.nnLen,
                allSymmetries: options.symmetries == 8
            )
            summary.gamesUsed += 1
            summary.totalSamples += generated.count
            if isValPass { summary.validationSamples += generated.count }
            for sample in generated {
                try append(sample)
            }
        }
        if !shard.isEmpty { try flush() }
        return summary
    }

    /// Dispatches teacher-distillation shard generation to the overlapped pipeline (default) or the
    /// legacy serial path (`serial == true`). Both are byte-identical; see `generate(_:teacher:
    /// teacherBatchSize:serial:)`.
    @discardableResult
    private static func processFilesWithTeacher(
        _ files: [URL],
        options: TrainingDataOptions,
        teacher: NNEvaluating,
        teacherBatchSize: Int,
        serial: Bool,
        shardPrefix: String,
        outputDirectory: URL,
        shardCap: Int,
        summary: inout TrainingDataSummary
    ) async throws -> TrainingDataSummary {
        if serial {
            return try await processFilesWithTeacherSerial(
                files, options: options, teacher: teacher, teacherBatchSize: teacherBatchSize,
                shardPrefix: shardPrefix, outputDirectory: outputDirectory,
                shardCap: shardCap, summary: &summary
            )
        }
        return try await processFilesWithTeacherPipelined(
            files, options: options, teacher: teacher, teacherBatchSize: teacherBatchSize,
            shardPrefix: shardPrefix, outputDirectory: outputDirectory,
            shardCap: shardCap, summary: &summary
        )
    }

    /// One buffered position awaiting a teacher label, shared by the serial and pipelined paths: the
    /// identity-sample features (reused verbatim as the distilled sample's spatial/global AND as the
    /// prefilled teacher request) plus the side-to-move needed to reproject the teacher's
    /// white-perspective outputs. `NNRequest`/`[Float]`/`Player` are all `Sendable`, and each
    /// request's `board`/`history` are freshly copied per position in `prepare` and never mutated
    /// afterwards, so a work item is safe to hand across the pipeline's stage tasks.
    private struct TeacherWorkItem {
        let spatial: [Float]
        let global: [Float]
        let nextPlayer: Player
        let request: NNRequest
    }

    /// A full teacher batch of buffered positions handed from the producer stage to the GPU stage.
    private struct PendingTeacherBatch {
        let items: [TeacherWorkItem]
    }

    /// A teacher batch plus its GPU outputs, handed from the GPU stage to the consumer stage. Kept
    /// paired (and in FIFO batch order) so the consumer emits samples in exactly serial order.
    private struct LabeledTeacherBatch {
        let items: [TeacherWorkItem]
        let outputs: [NNOutput]
    }

    /// The partial result each pipeline stage contributes back after the task group drains. The
    /// producer owns the file-level counters, the evaluator owns teacher timing, and the consumer
    /// owns the shard count; merging them after the group completes avoids any shared mutable state
    /// between the concurrently running stages.
    private enum TeacherStageOutcome {
        case producer(counts: TrainingDataSummary)
        case evaluator(teacherSeconds: Double, positionsLabeled: Int)
        case consumer(shardsWritten: Int)
    }

    /// Overlapped teacher pipeline. Three stages run concurrently on distinct batches
    /// inside a throwing task group:
    ///
    /// - **Producer** (CPU): SGF parse -> legal replay -> `fillRowV7` feature fill, grouping raw
    ///   positions into `teacherBatchSize` batches in the same file/position order as the serial
    ///   path.
    /// - **Evaluator** (GPU): one `teacher.evaluate` per batch, in FIFO order (the teacher's
    ///   `maxBatchSize` is the CLI teacher batch, so each call is one GPU batch).
    /// - **Consumer** (CPU): `TeacherLabeler` target conversion + symmetry expansion + bulk shard
    ///   serialization, appending in FIFO order with the identical `shardCap` boundaries.
    ///
    /// Ordering is preserved because each hand-off is a single-producer/single-consumer FIFO
    /// `AsyncStream`, so the output is byte-identical to the serial path (proven by
    /// `TeacherPipelineParityTests`). Memory is bounded by credit-based backpressure: the consumer
    /// hands one credit back per fully consumed batch and the producer must hold a credit to publish,
    /// so at most `maxInFlightBatches` batches are ever alive across the whole pipeline.
    @discardableResult
    private static func processFilesWithTeacherPipelined(
        _ files: [URL],
        options: TrainingDataOptions,
        teacher: NNEvaluating,
        teacherBatchSize: Int,
        shardPrefix: String,
        outputDirectory: URL,
        shardCap: Int,
        summary: inout TrainingDataSummary
    ) async throws -> TrainingDataSummary {
        let isValPass = shardCap == .max
        let nnLen = options.nnLen
        let symmetries: [TrainingSymmetry] = options.symmetries == 8 ? TrainingSymmetry.allCases : [.identity]
        let startingShardIndex = summary.totalShards
        // Small fixed pipeline depth (2-3 in flight). This is the credit count, so
        // the producer + evaluator + consumer can each hold one batch with no starvation, and total
        // resident batches never exceed this.
        let maxInFlightBatches = 3

        let (batchStream, batchContinuation) = AsyncStream.makeStream(of: PendingTeacherBatch.self)
        let (labeledStream, labeledContinuation) = AsyncStream.makeStream(of: LabeledTeacherBatch.self)
        let (creditStream, creditContinuation) = AsyncStream.makeStream(of: Void.self)
        for _ in 0 ..< maxInFlightBatches {
            creditContinuation.yield(())
        }

        var producerCounts = TrainingDataSummary()
        var teacherSeconds = 0.0
        var positionsLabeled = 0
        var shardsWritten = 0

        try await withThrowingTaskGroup(of: TeacherStageOutcome.self) { group in
            // Producer: CPU parse + feature fill, grouped into teacher batches in serial order.
            group.addTask {
                defer { batchContinuation.finish() }
                var counts = TrainingDataSummary()
                var creditIterator = creditStream.makeAsyncIterator()
                var batch = [TeacherWorkItem]()
                batch.reserveCapacity(teacherBatchSize)

                func publish() async {
                    guard !batch.isEmpty else { return }
                    _ = await creditIterator.next() // backpressure: wait for a free in-flight slot
                    batchContinuation.yield(PendingTeacherBatch(items: batch))
                    batch.removeAll(keepingCapacity: true)
                }

                for file in files {
                    try Task.checkCancellation()
                    guard let prepared = try preparedGame(file: file, options: options, summary: &counts) else {
                        continue
                    }
                    counts.gamesUsed += 1
                    let generatedCount = prepared.positions.count * symmetries.count
                    counts.totalSamples += generatedCount
                    if isValPass { counts.validationSamples += generatedCount }
                    for position in prepared.positions {
                        let base = identitySample(
                            game: prepared.game, position: position,
                            result: prepared.result, finalArea: prepared.finalArea, nnLen: nnLen
                        )
                        let request = NNRequest(
                            board: position.board,
                            history: position.history,
                            nextPlayer: position.nextPlayer,
                            params: MiscNNInputParams(),
                            prefilled: NNRequest.FeatureRow(spatial: base.spatial, global: base.global)
                        )
                        batch.append(TeacherWorkItem(
                            spatial: base.spatial, global: base.global,
                            nextPlayer: position.nextPlayer, request: request
                        ))
                        if batch.count >= teacherBatchSize { await publish() }
                    }
                }
                await publish() // final partial batch
                return .producer(counts: counts)
            }

            // Evaluator: one GPU forward per batch, FIFO, timing only the teacher call.
            group.addTask {
                defer { labeledContinuation.finish() }
                var seconds = 0.0
                var labeled = 0
                for await pending in batchStream {
                    try Task.checkCancellation()
                    let clock = Date()
                    let outputs = try await teacher.evaluate(pending.items.map(\.request))
                    seconds += Date().timeIntervalSince(clock)
                    labeled += pending.items.count
                    labeledContinuation.yield(LabeledTeacherBatch(items: pending.items, outputs: outputs))
                }
                return .evaluator(teacherSeconds: seconds, positionsLabeled: labeled)
            }

            // Consumer: target conversion + symmetry expansion + shard serialization, FIFO. Owns the
            // shard buffer and index, and hands a credit back after each fully consumed batch.
            group.addTask {
                var written = 0
                var shard = [TrainingSample]()
                shard.reserveCapacity(min(shardCap, options.shardSize))
                var shardIndex = startingShardIndex

                func flush() throws {
                    let name = isValPass
                        ? "val.nngd"
                        : String(format: "%@%05d.nngd", shardPrefix, shardIndex)
                    let url = outputDirectory.appendingPathComponent(name)
                    try TrainingShardWriter.write(samples: shard, nnLen: nnLen, to: url)
                    written += 1
                    if !isValPass { shardIndex += 1 }
                    shard.removeAll(keepingCapacity: true)
                }

                func append(_ sample: TrainingSample) throws {
                    shard.append(sample)
                    if !isValPass, shard.count == shardCap { try flush() }
                }

                for await labeledBatch in labeledStream {
                    try Task.checkCancellation()
                    for (item, output) in zip(labeledBatch.items, labeledBatch.outputs) {
                        let targets = TeacherLabeler.targets(from: output, nextPlayer: item.nextPlayer, nnLen: nnLen)
                        let relabeled = TrainingSample(
                            spatial: item.spatial,
                            global: item.global,
                            policyTarget: targets.policy,
                            valueTarget: targets.value,
                            scoreTarget: targets.score,
                            scoreValid: true,
                            ownershipTarget: targets.ownership,
                            ownershipValid: true
                        )
                        for symmetry in symmetries {
                            try append(symmetry.transform(sample: relabeled, nnLen: nnLen))
                        }
                    }
                    creditContinuation.yield(()) // release the in-flight slot this batch held
                }
                if !shard.isEmpty { try flush() }
                creditContinuation.finish()
                return .consumer(shardsWritten: written)
            }

            for try await outcome in group {
                switch outcome {
                case let .producer(counts): producerCounts = counts
                case let .evaluator(seconds, labeled):
                    teacherSeconds = seconds
                    positionsLabeled = labeled
                case let .consumer(written): shardsWritten = written
                }
            }
        }

        // Merge each stage's partial contribution into the caller's running summary. Skip counters
        // and file-level counts come from the producer; teacher timing from the evaluator; shard
        // count from the consumer.
        summary.gamesUsed += producerCounts.gamesUsed
        summary.totalSamples += producerCounts.totalSamples
        summary.validationSamples += producerCounts.validationSamples
        summary.parseSkips += producerCounts.parseSkips
        summary.boardSizeSkips += producerCounts.boardSizeSkips
        summary.minimumMoveSkips += producerCounts.minimumMoveSkips
        summary.jigoSkips += producerCounts.jigoSkips
        summary.resultSkips += producerCounts.resultSkips
        summary.illegalMoveSkips += producerCounts.illegalMoveSkips
        summary.teacherSeconds += teacherSeconds
        summary.teacherPositionsLabeled += positionsLabeled
        summary.totalShards += shardsWritten
        return summary
    }

    /// Legacy strict-alternation teacher path (the pre-overlap path): buffer positions, block on the
    /// teacher `evaluate`, then serialize -- CPU and GPU never overlap. Retained as the `-serial`
    /// escape hatch and as the byte-identity oracle for the pipelined path.
    @discardableResult
    private static func processFilesWithTeacherSerial(
        _ files: [URL],
        options: TrainingDataOptions,
        teacher: NNEvaluating,
        teacherBatchSize: Int,
        shardPrefix: String,
        outputDirectory: URL,
        shardCap: Int,
        summary: inout TrainingDataSummary
    ) async throws -> TrainingDataSummary {
        let isValPass = shardCap == .max
        let nnLen = options.nnLen
        let symmetries: [TrainingSymmetry] = options.symmetries == 8 ? TrainingSymmetry.allCases : [.identity]
        var shard = [TrainingSample]()
        shard.reserveCapacity(min(shardCap, options.shardSize))
        var shardIndex = summary.totalShards

        func flush() throws {
            let name = isValPass
                ? "val.nngd"
                : String(format: "%@%05d.nngd", shardPrefix, shardIndex)
            let url = outputDirectory.appendingPathComponent(name)
            try TrainingShardWriter.write(samples: shard, nnLen: options.nnLen, to: url)
            summary.totalShards += 1
            if !isValPass { shardIndex += 1 }
            shard.removeAll(keepingCapacity: true)
        }

        func append(_ sample: TrainingSample) throws {
            shard.append(sample)
            if !isValPass, shard.count == shardCap { try flush() }
        }

        // One buffered position awaiting a teacher label: its identity features (reused as the
        // prefilled teacher request) plus the side-to-move needed to reproject the teacher's
        // white-perspective outputs.
        var pending = [(features: TrainingSample, nextPlayer: Player, request: NNRequest)]()
        pending.reserveCapacity(teacherBatchSize)

        func drain() async throws {
            guard !pending.isEmpty else { return }
            let clock = Date()
            let outputs = try await teacher.evaluate(pending.map(\.request))
            summary.teacherSeconds += Date().timeIntervalSince(clock)
            summary.teacherPositionsLabeled += pending.count
            for (item, output) in zip(pending, outputs) {
                let targets = TeacherLabeler.targets(from: output, nextPlayer: item.nextPlayer, nnLen: nnLen)
                let relabeled = TrainingSample(
                    spatial: item.features.spatial,
                    global: item.features.global,
                    policyTarget: targets.policy,
                    valueTarget: targets.value,
                    scoreTarget: targets.score,
                    scoreValid: true,
                    ownershipTarget: targets.ownership,
                    ownershipValid: true
                )
                for symmetry in symmetries {
                    try append(symmetry.transform(sample: relabeled, nnLen: nnLen))
                }
            }
            pending.removeAll(keepingCapacity: true)
        }

        for file in files {
            guard let prepared = try preparedGame(file: file, options: options, summary: &summary) else {
                continue
            }
            summary.gamesUsed += 1
            let generatedCount = prepared.positions.count * symmetries.count
            summary.totalSamples += generatedCount
            if isValPass { summary.validationSamples += generatedCount }
            for position in prepared.positions {
                let base = identitySample(
                    game: prepared.game, position: position,
                    result: prepared.result, finalArea: prepared.finalArea, nnLen: nnLen
                )
                let request = NNRequest(
                    board: position.board,
                    history: position.history,
                    nextPlayer: position.nextPlayer,
                    params: MiscNNInputParams(),
                    prefilled: NNRequest.FeatureRow(spatial: base.spatial, global: base.global)
                )
                pending.append((features: base, nextPlayer: position.nextPlayer, request: request))
                if pending.count >= teacherBatchSize { try await drain() }
            }
        }
        try await drain()
        if !shard.isEmpty { try flush() }
        return summary
    }

    /// Teacher-SEARCH labeling path (Phase P, `-teacher-visits N` with N >= 2). Identical file
    /// selection / val split / shard layout / sample order to the raw teacher paths -- it reuses the
    /// same `preparedGame` extraction, so search mode only replaces the per-position TARGETS. For
    /// each raw position it runs a fresh `teacherVisits`-visit PUCT search from that position using
    /// the shared teacher evaluator, then converts the search's root visit distribution + backed-up
    /// winrate (plus the raw root net's score/ownership) into v2 targets via
    /// `TeacherLabeler.searchTargets`.
    ///
    /// Positions are processed strictly serially: each search is ~N sequential-ish evals and
    /// dominates the CPU parse/serialize cost, so the WP-4b overlap the raw pipeline chases buys
    /// little here (stated in `generate`'s doc). Throughput comes instead from batching WITHIN each
    /// search -- the `Search` collects up to `min(leafBatchSize, numSearchWorkers)` leaves per
    /// `evaluate`, so one GPU forward covers several playouts. `summary.teacherSeconds` accumulates
    /// the wall-clock inside `runSearch` (the search's teacher evals) and `teacherPositionsLabeled`
    /// counts raw positions, so `teacherPositionsLabeled / teacherSeconds` is the search labeling
    /// throughput (positions/s), mirroring the raw path's meaning of those fields.
    @discardableResult
    private static func processFilesWithTeacherSearch(
        _ files: [URL],
        options: TrainingDataOptions,
        teacher: NNEvaluating,
        teacherVisits: Int,
        teacherTarget: TeacherLabeler.PolicyTargetKind,
        searchSettings: SearchSettings,
        precision: KataGoNetwork.Precision,
        fp32FallbackFactory: (@Sendable () throws -> any NNEvaluating)?,
        shardPrefix: String,
        outputDirectory: URL,
        shardCap: Int,
        summary: inout TrainingDataSummary
    ) async throws -> TrainingDataSummary {
        let isValPass = shardCap == .max
        let nnLen = options.nnLen
        let symmetries: [TrainingSymmetry] = options.symmetries == 8 ? TrainingSymmetry.allCases : [.identity]
        var shard = [TrainingSample]()
        shard.reserveCapacity(min(shardCap, options.shardSize))
        var shardIndex = summary.totalShards

        func flush() throws {
            let name = isValPass
                ? "val.nngd"
                : String(format: "%@%05d.nngd", shardPrefix, shardIndex)
            let url = outputDirectory.appendingPathComponent(name)
            try TrainingShardWriter.write(samples: shard, nnLen: nnLen, to: url)
            summary.totalShards += 1
            if !isValPass { shardIndex += 1 }
            shard.removeAll(keepingCapacity: true)
        }

        func append(_ sample: TrainingSample) throws {
            shard.append(sample)
            if !isValPass, shard.count == shardCap { try flush() }
        }

        for file in files {
            guard let prepared = try preparedGame(file: file, options: options, summary: &summary) else {
                continue
            }
            summary.gamesUsed += 1
            let generatedCount = prepared.positions.count * symmetries.count
            summary.totalSamples += generatedCount
            if isValPass { summary.validationSamples += generatedCount }
            for position in prepared.positions {
                // The identity sample supplies the stored spatial/global features (byte-identical to
                // every other makedata path); only the targets are search-derived. The `Search`
                // recomputes its own root features internally, so the stored features and the search
                // input agree by construction (both are fillRowV7 at this position).
                let base = identitySample(
                    game: prepared.game, position: position,
                    result: prepared.result, finalArea: prepared.finalArea, nnLen: nnLen
                )
                let search = Search(
                    evaluator: teacher,
                    nnXLen: nnLen,
                    nnYLen: nnLen,
                    precision: precision,
                    settings: searchSettings,
                    board: position.board,
                    history: position.history,
                    nextPlayer: position.nextPlayer,
                    fp32FallbackFactory: fp32FallbackFactory
                )
                let clock = Date()
                let result = try await search.runSearch(maxVisits: Int64(teacherVisits))
                summary.teacherSeconds += Date().timeIntervalSince(clock)
                summary.teacherPositionsLabeled += 1
                guard let rootOutput = result.rootNNOutput else {
                    throw TrainingDataError.invalidSample("Teacher search produced no root NN output")
                }
                let targets = TeacherLabeler.searchTargets(
                    visitsByMove: result.visitsByMove,
                    whiteWinProb: result.whiteWinProb,
                    rootOutput: rootOutput,
                    boardXSize: position.board.xSize,
                    nextPlayer: position.nextPlayer,
                    nnLen: nnLen,
                    target: teacherTarget,
                    childWinValues: result.rootChildWinValues
                )
                let relabeled = TrainingSample(
                    spatial: base.spatial,
                    global: base.global,
                    policyTarget: targets.policy,
                    valueTarget: targets.value,
                    scoreTarget: targets.score,
                    scoreValid: true,
                    ownershipTarget: targets.ownership,
                    ownershipValid: true
                )
                for symmetry in symmetries {
                    try append(symmetry.transform(sample: relabeled, nnLen: nnLen))
                }
            }
        }
        if !shard.isEmpty { try flush() }
        return summary
    }

    /// The subset of a game's data needed to emit training samples: the parsed game, its per-move
    /// positions (each carrying the board/history/side-to-move to feed a teacher), the reconciled
    /// game result, and the final-position area classification (ownership label source).
    private struct PreparedGame {
        let game: SGFGame
        let positions: [Position]
        let result: TrainingGameResult
        let finalArea: [Color]
    }

    /// Loads one SGF and runs the full acceptance pipeline (parse -> board size -> minimum moves ->
    /// legal replay -> result reconciliation), incrementing the appropriate skip counter and
    /// returning `nil` when a game is rejected. Shared verbatim by the teacherless (`processFiles`)
    /// and teacher-distillation (`processFilesWithTeacher`) paths so BOTH extract exactly the same
    /// set of positions -- teacher mode only replaces the per-position TARGETS, never which
    /// positions are kept.
    private static func preparedGame(
        file: URL,
        options: TrainingDataOptions,
        summary: inout TrainingDataSummary
    ) throws -> PreparedGame? {
        let game: SGFGame
        do {
            game = try SGFReader.load(file, defaultRules: options.rules)
        } catch {
            summary.parseSkips += 1
            return nil
        }
        guard game.boardXSize == options.nnLen, game.boardYSize == options.nnLen else {
            summary.boardSizeSkips += 1
            return nil
        }
        guard game.moves.count >= options.minimumMoves else {
            summary.minimumMoveSkips += 1
            return nil
        }
        let prepared: (board: Board, history: BoardHistory, positions: [Position], endedByPasses: Bool)
        do {
            guard let value = try prepare(game: game) else {
                summary.illegalMoveSkips += 1
                return nil
            }
            prepared = value
        } catch {
            summary.illegalMoveSkips += 1
            return nil
        }
        let result: TrainingGameResult
        let scoredByPasses = prepared.endedByPasses && prepared.history.isScored
        if let rawResult = game.result, let parsed = TrainingGameResult.parse(rawResult) {
            if parsed == .jigo {
                summary.jigoSkips += 1
                return nil
            }
            result = parsed
        } else if scoredByPasses, let winner = prepared.history.winner {
            result = .scored(winner: winner, whiteMinusBlack: prepared.history.finalWhiteMinusBlackScore)
        } else if scoredByPasses, prepared.history.winner == nil {
            summary.jigoSkips += 1
            return nil
        } else {
            summary.resultSkips += 1
            return nil
        }
        let finalArea = prepared.history.getAreaNow(prepared.board)
        return PreparedGame(game: game, positions: prepared.positions, result: result, finalArea: finalArea)
    }

    /// Deterministic Fisher–Yates shuffle with a fixed seed so re-runs at the same valRatio
    /// produce exactly the same val set (validation loss curves stay comparable across resumes).
    /// Splits into `(trainFiles, valFiles)` where the first `valCount` entries after the shuffle
    /// are the val files (the rest are train).
    private static func dedupShuffleAndSplit(files: [URL], valCount: Int) -> (train: [URL], val: [URL]) {
        var shuffled = files
        var seed: UInt64 = 0x5EED_BEEF_9BAD_C0DE
        for index in stride(from: shuffled.count - 1, through: 1, by: -1) {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let upper = UInt64(truncatingIfNeeded: index + 1)
            let j = Int((seed >> 33) % upper)
            shuffled.swapAt(index, j)
        }
        let cut = min(valCount, shuffled.count)
        let val = Array(shuffled.prefix(cut))
        let train = Array(shuffled.dropFirst(cut))
        return (train, val)
    }

    private typealias Position = (board: Board, history: BoardHistory, nextPlayer: Player, move: Move)

    private static func prepare(
        game: SGFGame
    ) throws -> (board: Board, history: BoardHistory, positions: [Position], endedByPasses: Bool)? {
        var rules = game.rules
        rules.koRule = .positional
        let initial = try game.position(at: 0, rules: rules)
        let board = initial.board
        let history = initial.history
        var positions = [Position]()
        positions.reserveCapacity(game.moves.count)
        for move in game.moves {
            positions.append((board.copy(), history.copy(), move.pla, move))
            guard history.makeBoardMoveTolerant(board, move.loc, move.pla, preventEncore: false) else { return nil }
        }
        let passes = game.moves.count >= 2 && game.moves.suffix(2).allSatisfy { $0.loc == Board.passLoc }
        return (board, history, positions, passes)
    }

    private static func samples(
        game: SGFGame,
        positions: [Position],
        result: TrainingGameResult,
        finalArea: [Color],
        nnLen: Int,
        allSymmetries: Bool
    ) -> [TrainingSample] {
        let symmetries: [TrainingSymmetry] = allSymmetries ? TrainingSymmetry.allCases : [.identity]
        // A resignation's final board
        // position is not a reliable score or ownership label (the loser typically resigns with
        // stones still on the board that would never survive to a real scoring), so only genuinely
        // scored games mark their score/ownership targets valid. Teacherless makedata therefore
        // writes a one-hot policy but still respects this per-game validity split.
        return positions.flatMap { position -> [TrainingSample] in
            let base = identitySample(
                game: game, position: position, result: result, finalArea: finalArea, nnLen: nnLen
            )
            return symmetries.map { $0.transform(sample: base, nnLen: nnLen) }
        }
    }

    /// Builds the single identity (unrotated) teacherless sample for one position: a one-hot policy
    /// at the played move, the game-outcome value/score, and final-area ownership. Score and
    /// ownership validity follow the same per-game rule as before:
    /// only genuinely scored games mark them valid; a resignation leaves both false. Teacher mode
    /// reuses only this sample's precomputed features (spatial/global) and replaces every target.
    private static func identitySample(
        game: SGFGame,
        position: Position,
        result: TrainingGameResult,
        finalArea: [Color],
        nnLen: Int
    ) -> TrainingSample {
        let area = nnLen * nnLen
        let isValid = if case .scored = result { true } else { false }
        let features = NNInputs.fillRowV7(
            position.board,
            position.history,
            position.nextPlayer,
            MiscNNInputParams(),
            nnXLen: nnLen,
            nnYLen: nnLen,
            useNHWC: true
        )
        let movePosition = NNPos.locToPos(position.move.loc, game.boardXSize, nnLen, nnLen)
        var policy = [Float](repeating: 0, count: area + 1)
        policy[movePosition] = 1
        let winner = result.winner!
        let value: [Float] = winner == position.nextPlayer ? [1, 0, 0] : [0, 1, 0]
        let scoreWhiteMinusBlack = result.whiteMinusBlackScore!
        let score = position.nextPlayer == .white ? scoreWhiteMinusBlack : -scoreWhiteMinusBlack
        let ownership = ownershipTarget(
            area: finalArea,
            boardXSize: game.boardXSize,
            player: position.nextPlayer,
            nnLen: nnLen
        )
        return TrainingSample(
            spatial: features.spatial,
            global: features.global,
            policyTarget: policy,
            valueTarget: value,
            scoreTarget: score,
            scoreValid: isValid,
            ownershipTarget: ownership,
            ownershipValid: isValid
        )
    }

    private static func ownershipTarget(
        area: [Color],
        boardXSize: Int,
        player: Player,
        nnLen: Int
    ) -> [Int8] {
        var result = [Int8](repeating: 0, count: nnLen * nnLen)
        for y in 0 ..< nnLen {
            for x in 0 ..< nnLen {
                let owner = area[Location.getLoc(x, y, boardXSize)]
                result[y * nnLen + x] = owner == player.color ? 1 : owner == player.opponent.color ? -1 : 0
            }
        }
        return result
    }

    private static func sgfFiles(in directory: URL, fileManager: FileManager) throws -> [URL] {
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { throw TrainingDataError.invalidArgument("Cannot enumerate \(directory.path)") }
        var files = [URL]()
        for case let file as URL in enumerator where file.pathExtension.lowercased() == "sgf" {
            if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                files.append(file)
            }
        }
        return files.sorted { $0.path < $1.path }
    }

    private static func removeOldShards(in directory: URL, fileManager: FileManager) throws {
        let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        for file in files where file.lastPathComponent.hasPrefix("shard-") && file.pathExtension == "nngd" {
            try fileManager.removeItem(at: file)
        }
    }
}

/// Writes RinGoData v2 fields into a preallocated little-endian byte buffer with a running cursor.
/// Bulk per-field copies replace the old per-scalar `Data.append` path. Assumes a
/// little-endian host (as does the read side in `RinGoShard`/`TrainingShardReader`): a `[Float]`/
/// `[Int8]`'s in-memory bytes are already the on-disk bytes, so array fields copy straight through.
/// Copies are byte-wise (`copyMemory`), so the unaligned per-sample float offsets that arise from
/// the odd `sampleSize` are safe.
private struct LittleEndianByteWriter {
    let buffer: UnsafeMutableRawBufferPointer
    var offset = 0

    mutating func putBytes(_ bytes: Data) {
        guard !bytes.isEmpty else { return }
        bytes.withUnsafeBytes { source in
            (buffer.baseAddress! + offset).copyMemory(from: source.baseAddress!, byteCount: source.count)
        }
        offset += bytes.count
    }

    mutating func putUInt32(_ value: UInt32) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { source in
            (buffer.baseAddress! + offset).copyMemory(from: source.baseAddress!, byteCount: source.count)
        }
        offset += MemoryLayout<UInt32>.size
    }

    mutating func putFloat(_ value: Float) {
        putUInt32(value.bitPattern)
    }

    mutating func putByte(_ value: UInt8) {
        buffer.storeBytes(of: value, toByteOffset: offset, as: UInt8.self)
        offset += MemoryLayout<UInt8>.size
    }

    mutating func putFloats(_ values: [Float]) {
        guard !values.isEmpty else { return }
        values.withUnsafeBytes { source in
            (buffer.baseAddress! + offset).copyMemory(from: source.baseAddress!, byteCount: source.count)
        }
        offset += values.count * MemoryLayout<Float>.size
    }

    mutating func putInt8s(_ values: [Int8]) {
        guard !values.isEmpty else { return }
        values.withUnsafeBytes { source in
            (buffer.baseAddress! + offset).copyMemory(from: source.baseAddress!, byteCount: source.count)
        }
        offset += values.count
    }
}

private extension Data {
    func readUInt32(at offset: Int) -> UInt32 {
        UInt32(self[offset]) | UInt32(self[offset + 1]) << 8 | UInt32(self[offset + 2]) << 16
            | UInt32(self[offset + 3]) << 24
    }

    func readFloat(at offset: Int) -> Float {
        Float(bitPattern: readUInt32(at: offset))
    }
}
