import Foundation
import RinGoCore
import RinGoEngine
import RinGoModel

enum MakeDataCommand {
    static let usage = """
    Usage: ringo makedata -sgf-dir <dir> [-out <dir>] [-nnlen 9] \
      [-shard-size 4096] [-rules chinese] [-min-moves 8] [-symmetries 0|8] [-val-ratio 0.0] \
      [-teacher-model <net.bin.gz>] [-teacher-precision fp16|fp32] [-teacher-batch-size 256] \
      [-teacher-visits N] [-teacher-target visits|completed-q] [-gpu-cache-limit-mb <N>] [-serial]
           ringo makedata -verify <shardfile> [-sample <zero-based-index>]
    """

    static func main(_ arguments: [String]) async throws {
        if arguments.first == "-verify" {
            try verify(arguments)
        } else {
            try await generate(arguments)
        }
    }

    private static func generate(_ arguments: [String]) async throws {
        var sgfDirectory: String?
        var outputDirectory = "ringo-data"
        var nnLen = 9
        var shardSize = 4096
        var ruleName = "chinese"
        var minimumMoves = 8
        var symmetries = 0
        var valRatio: Float = 0
        var teacherModel: String?
        var teacherPrecision: KataGoNetwork.Precision = .float16
        var teacherBatchSize = 256
        var teacherVisits = 1
        var teacherTarget: TeacherLabeler.PolicyTargetKind = .visits
        var gpuCacheLimitMB: Int?
        var serial = false
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            // Valueless flags are handled before the value guard below.
            if flag == "-serial" {
                serial = true
                index += 1
                continue
            }
            guard index + 1 < arguments.count else { throw CLIError(description: "Missing value for \(flag)") }
            let value = arguments[index + 1]
            switch flag {
            case "-sgf-dir": sgfDirectory = value
            case "-out": outputDirectory = value
            case "-nnlen": nnLen = try integer(value, named: flag)
            case "-shard-size": shardSize = try integer(value, named: flag)
            case "-rules": ruleName = value
            case "-min-moves": minimumMoves = try integer(value, named: flag)
            case "-symmetries": symmetries = try integer(value, named: flag)
            case "-val-ratio":
                guard let parsed = Float(value), parsed >= 0, parsed < 1, parsed.isFinite else {
                    throw CLIError(description: "Invalid -val-ratio (must be a float in [0, 1))")
                }
                valRatio = parsed
            case "-teacher-model": teacherModel = value
            case "-teacher-precision":
                switch value {
                case "fp16": teacherPrecision = .float16
                case "fp32": teacherPrecision = .float32
                default: throw CLIError(description: "Invalid -teacher-precision (must be fp16 or fp32)")
                }
            case "-teacher-batch-size":
                let parsed = try integer(value, named: flag)
                guard parsed > 0 else { throw CLIError(description: "Invalid -teacher-batch-size (must be positive)") }
                teacherBatchSize = parsed
            case "-teacher-visits":
                let parsed = try integer(value, named: flag)
                guard parsed >= 1 else { throw CLIError(description: "Invalid -teacher-visits (must be >= 1)") }
                teacherVisits = parsed
            case "-teacher-target":
                guard let parsed = TeacherLabeler.PolicyTargetKind(rawValue: value) else {
                    throw CLIError(description: "Invalid -teacher-target (must be visits or completed-q)")
                }
                teacherTarget = parsed
            case "-gpu-cache-limit-mb":
                let parsed = try integer(value, named: flag)
                guard parsed >= 0 else { throw CLIError(description: "Invalid -gpu-cache-limit-mb (must be >= 0)") }
                gpuCacheLimitMB = parsed
            default: throw CLIError(description: "Unknown option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard let sgfDirectory else { throw CLIError(description: usage) }
        let rules: Rules
        do {
            rules = try Rules.parseRulesWithoutKomi(ruleName, komi: 7.5)
        } catch {
            throw CLIError(description: "Invalid rules '\(ruleName)': \(error)")
        }
        let outputURL = URL(fileURLWithPath: outputDirectory)
        let options = TrainingDataOptions(
            sgfDirectory: URL(fileURLWithPath: sgfDirectory),
            outputDirectory: outputURL,
            nnLen: nnLen,
            shardSize: shardSize,
            rules: rules,
            minimumMoves: minimumMoves,
            symmetries: symmetries,
            valRatio: valRatio
        )

        let summary: TrainingDataSummary
        if let teacherModel {
            // Distillation: load the teacher once, then label every extracted position. In raw mode
            // (teacher-visits <= 1) the evaluator's max batch is the requested teacher batch (its
            // physical GPU batches bucket up to this), so one teacher `evaluate` per drain is one GPU
            // batch. In search mode (teacher-visits >= 2) each position is labeled by an N-visit PUCT
            // search whose leaf batches bucket up to the same max.
            let desc = try ModelDesc.loadFromFileMaybeGZipped(teacherModel)
            let teacher = try NNEvaluator(
                desc: desc,
                nnXLen: nnLen,
                nnYLen: nnLen,
                precision: teacherPrecision,
                maxBatchSize: teacherBatchSize,
                // Bound the MLX GPU cache for the (potentially hour-long) labeling run instead of
                // letting it grow unbounded (audit P2, mirrors the train/NNEvaluator knob). Opt-in:
                // nil keeps MLX's default, matching how TrainCommand exposes it.
                gpuCacheLimitMB: gpuCacheLimitMB
            )
            // fp16 resilience for the (hour-scale) search run: a rare non-finite fp16 policy on a
            // synthetic position would otherwise abort the whole labeling run. Give the per-position
            // Search an fp32 fallback factory built from the same desc, exactly as production search
            // is constructed. Unused when the teacher is already fp32 (search skips the fallback).
            let searchNNLen = nnLen
            let fp32FallbackFactory: (@Sendable () throws -> any NNEvaluating)? = if teacherPrecision == .float32 {
                nil
            } else {
                {
                    try NNEvaluator(
                        desc: desc, nnXLen: searchNNLen, nnYLen: searchNNLen,
                        precision: .float32, maxBatchSize: 1
                    )
                }
            }
            if serial, teacherVisits >= 2 {
                print("makedata: note -serial is implied by teacher-search mode (-teacher-visits >= 2)")
            }
            summary = try await TrainingDataPipeline.generate(
                options, teacher: teacher, teacherBatchSize: teacherBatchSize, serial: serial,
                teacherVisits: teacherVisits, teacherTarget: teacherTarget, precision: teacherPrecision,
                fp32FallbackFactory: fp32FallbackFactory
            )
            try writeMakedataConfig(
                to: outputURL, teacherModel: teacherModel,
                precision: teacherPrecision, batchSize: teacherBatchSize,
                teacherVisits: teacherVisits, teacherTarget: teacherTarget, summary: summary
            )
        } else {
            summary = try TrainingDataPipeline.generate(options)
        }

        let rate = summary.elapsedSeconds > 0 ? Double(summary.gamesUsed) / summary.elapsedSeconds : 0
        print(
            "makedata: games used=\(summary.gamesUsed) skipped=\(summary.gamesSkipped) "
                + "(parse=\(summary.parseSkips), size=\(summary.boardSizeSkips), "
                + "minMoves=\(summary.minimumMoveSkips), jigo=\(summary.jigoSkips), "
                + "result=\(summary.resultSkips), illegalMove=\(summary.illegalMoveSkips)); "
                + "samples=\(summary.totalSamples) shards=\(summary.totalShards) "
                + String(format: "elapsed=%.3fs games/sec=%.2f", summary.elapsedSeconds, rate)
        )
        // O-4 (P2, CODEBASE_REVIEW_0711.md): a high skip rate silently produces a smaller-than-
        // expected training set from an ostensibly-large SGF corpus (bad rules match, wrong
        // -min-moves, a corpus that is mostly jigos/aborted games, etc.) with nothing in the
        // normal one-line summary above calling it out. Loud enough (stderr, WARNING) to be
        // noticed in a scrollback without failing the run -- a high skip rate is a smell, not
        // necessarily an error (e.g. deliberately filtering a mixed-quality corpus).
        if let warning = Self.highSkipRateWarning(gamesUsed: summary.gamesUsed, gamesSkipped: summary.gamesSkipped) {
            FileHandle.standardError.write(Data((warning + "\n").utf8))
        }
        if teacherModel != nil {
            let throughput = summary.teacherSeconds > 0
                ? Double(summary.teacherPositionsLabeled) / summary.teacherSeconds
                : 0
            // End-to-end samples/sec (wall-clock, including all CPU parse/feature/serialize work)
            // alongside the teacher-only figure, so the overlap win (audit P1-1) is visible: with
            // strict alternation the two rates diverge; with the pipeline the e2e rate approaches
            // the teacher rate.
            let e2eThroughput = summary.elapsedSeconds > 0
                ? Double(summary.totalSamples) / summary.elapsedSeconds
                : 0
            // In search mode `throughput` is positions searched per second (the reportable positions/s
            // figure); in raw mode it is the teacher-forward rate. The trailing tag disambiguates.
            let modeTag = teacherVisits >= 2 ? " (search visits=\(teacherVisits))" : (serial ? " (serial)" : "")
            print(
                "makedata: teacher labeled positions=\(summary.teacherPositionsLabeled) "
                    + String(
                        format: "teacherSeconds=%.3fs teacher-positions/sec=%.1f e2e-samples/sec=%.1f%@",
                        summary.teacherSeconds, throughput, e2eThroughput, modeTag
                    )
            )
        }
        if summary.validationSamples > 0 {
            print(
                "makedata: validation split files=\(summary.validationFiles) "
                    + "samples=\(summary.validationSamples) -> val.nngd"
            )
        }
    }

    /// Returns a WARNING line when skipped games exceed 30% of all games seen (parsed + skipped),
    /// or `nil` when the skip rate is unremarkable. `nil` also when `gamesUsed + gamesSkipped == 0`
    /// (an empty/no-games run is a separate, already-visible condition -- `totalSamples=0` in the
    /// summary line above -- not a skip-RATE problem). Pure function (no I/O) so it is directly
    /// unit-testable without an SGF fixture; see `MakeDataCommandTests`.
    static func highSkipRateWarning(gamesUsed: Int, gamesSkipped: Int) -> String? {
        let total = gamesUsed + gamesSkipped
        guard total > 0 else { return nil }
        let ratio = Double(gamesSkipped) / Double(total)
        guard ratio > 0.3 else { return nil }
        return String(
            format: "makedata: WARNING skipped %.1f%% of games (%d/%d) -- check -sgf-dir input "
                + "quality (parse/board-size/min-moves/jigo/result/illegal-move filters)",
            ratio * 100, gamesSkipped, total
        )
    }

    /// Writes `makedata-config.json` into the shard directory so a distilled dataset is
    /// self-describing (originality-disclosure requirement, TRAINING_FIX_PLAN.md D2): it records
    /// which official KataGo net was used as the teacher, at what precision and batch size, and the
    /// resulting counts.
    private static func writeMakedataConfig(
        to directory: URL,
        teacherModel: String,
        precision: KataGoNetwork.Precision,
        batchSize: Int,
        teacherVisits: Int,
        teacherTarget: TeacherLabeler.PolicyTargetKind,
        summary: TrainingDataSummary
    ) throws {
        // teacher_visits records the label source (1 = raw net output; >= 2 = N-visit teacher
        // search, the sharper Phase-P targets) so a distilled dataset stays self-describing.
        // teacher_target records the search POLICY construction (WP-13): "visits" (normalized visit
        // counts) or "completed-q" (Gumbel completed-Q improved policy). Only meaningful when
        // teacher_visits >= 2 (raw mode has no search policy); recorded unconditionally for a
        // uniform, self-describing provenance record.
        let dictionary: [String: Any] = [
            "distilled": true,
            "teacher_model": URL(fileURLWithPath: teacherModel).lastPathComponent,
            "teacher_model_path": teacherModel,
            "teacher_precision": precision == .float16 ? "fp16" : "fp32",
            "teacher_batch_size": batchSize,
            "teacher_visits": teacherVisits,
            "teacher_target": teacherTarget.rawValue,
            "teacher_label_source": teacherVisits >= 2 ? "search" : "raw",
            "teacher_positions_labeled": summary.teacherPositionsLabeled,
            "total_samples": summary.totalSamples,
            "validation_samples": summary.validationSamples,
            "created_at": ISO8601DateFormatter().string(from: Date()),
        ]
        let data = try JSONSerialization.data(
            withJSONObject: dictionary, options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: directory.appendingPathComponent("makedata-config.json"), options: .atomic)
    }

    private static func verify(_ arguments: [String]) throws {
        guard arguments.count == 2 || arguments.count == 4, arguments.count >= 2 else {
            throw CLIError(description: usage)
        }
        let sampleIndex: Int
        if arguments.count == 4 {
            guard arguments[2] == "-sample" else { throw CLIError(description: usage) }
            sampleIndex = try integer(arguments[3], named: "-sample")
        } else {
            sampleIndex = 0
        }
        let reader = try TrainingShardReader(url: URL(fileURLWithPath: arguments[1]))
        let header = reader.header
        print(
            "magic=NNGD version=\(header.version) nnLen=\(header.nnLen) "
                + "numSpatial=\(header.numSpatial) numGlobal=\(header.numGlobal) "
                + "sampleCount=\(header.sampleCount)"
        )
        guard header.sampleCount > 0 else {
            print("No samples")
            return
        }
        let sample = try reader.sample(at: sampleIndex)
        let area = header.nnLen * header.nnLen
        var nonzero = [Int](repeating: 0, count: header.numSpatial)
        for position in 0 ..< area {
            for feature in 0 ..< header.numSpatial where sample.spatial[position * header.numSpatial + feature] != 0 {
                nonzero[feature] += 1
            }
        }
        // policyTarget is a dense nnLen*nnLen+1 distribution (v2): report its argmax move (the
        // played move for a teacherless one-hot shard) and how concentrated the mass is, rather
        // than dumping all 82 floats.
        let topProbability = sample.policyTarget.max() ?? 0
        let topIndex = sample.policyTarget.firstIndex(of: topProbability) ?? 0
        let topMove = topIndex == area ? "pass" : "\(topIndex)"
        let own = sample.ownershipTarget.reduce(into: (selfOwned: 0, opponentOwned: 0, neutral: 0)) { counts, value in
            if value > 0 { counts.selfOwned += 1 } else if value < 0 { counts.opponentOwned += 1 } else {
                counts.neutral += 1
            }
        }
        print(
            "sample=\(sampleIndex) policyTopMove=\(topMove) policyTopProbability=\(topProbability) "
                + "valueTarget=\(sample.valueTarget) scoreTarget=\(sample.scoreTarget) scoreValid=\(sample.scoreValid)"
        )
        print("global=\(sample.global)")
        print("spatialNonzeroByPlane=\(nonzero)")
        print(
            "ownership self=\(own.selfOwned) opponent=\(own.opponentOwned) neutral=\(own.neutral) "
                + "ownershipValid=\(sample.ownershipValid)"
        )
    }

    private static func integer(_ value: String, named option: String) throws -> Int {
        guard let result = Int(value) else { throw CLIError(description: "Invalid integer for \(option)") }
        return result
    }
}
