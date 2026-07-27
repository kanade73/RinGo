import CryptoKit
import Foundation
import MLX
import RinGoModel
import RinGoTrain

enum TrainCommand {
    static let usage = "ringo train -data <dir> -out <rundir> [-arch b6c96|b10c128|b15c192|b20c256|b24c320|b28c384] "
        + "[-batch 256] [-steps N] [-lr 3e-4] "
        + "[-value-weight 3] [-ownership-weight 0.5] [-score-weight 0.02] [-max-grad-norm 5] "
        + "[-gpu-cache-limit-mb <N>] [-checkpoint-interval N] [-snapshot-interval N] "
        + "[-val-data <dir>] [-val-interval N] [-init-model <net.bin.gz>] [-fresh]"

    static func main(_ arguments: [String]) throws {
        var dataPath: String?
        var outputPath: String?
        var archName = "b6c96"
        var batchSize = 256
        var steps = 100_000
        var learningRate: Float = 1e-3
        var valueWeight: Float = defaultValueWeight
        var ownershipWeight: Float = defaultOwnershipWeight
        var scoreWeight: Float = defaultScoreWeight
        var maxGradNorm: Float = defaultMaxGradNorm
        var gpuCacheLimitMB: Int?
        var checkpointInterval = 1000
        var snapshotInterval = 5000
        var validationDataPath: String?
        var validationInterval: Int?
        var initModelPath: String?
        var fresh = false
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            if flag == "-fresh" {
                fresh = true
                index += 1
                continue
            }
            guard index + 1 < arguments.count else { throw CLIError(description: "Missing value for \(flag)") }
            let value = arguments[index + 1]
            switch flag {
            case "-data": dataPath = value
            case "-out": outputPath = value
            case "-arch": archName = value
            case "-batch":
                guard let parsed = Int(value), parsed > 0 else { throw CLIError(description: "Invalid -batch") }
                batchSize = parsed
            case "-steps":
                guard let parsed = Int(value), parsed > 0 else { throw CLIError(description: "Invalid -steps") }
                steps = parsed
            case "-lr":
                guard let parsed = Float(value), parsed > 0, parsed.isFinite else {
                    throw CLIError(description: "Invalid -lr")
                }
                learningRate = parsed
            case "-value-weight":
                guard let parsed = Float(value), parsed >= 0, parsed.isFinite else {
                    throw CLIError(description: "Invalid -value-weight")
                }
                valueWeight = parsed
            case "-ownership-weight":
                guard let parsed = Float(value), parsed >= 0, parsed.isFinite else {
                    throw CLIError(description: "Invalid -ownership-weight")
                }
                ownershipWeight = parsed
            case "-score-weight":
                guard let parsed = Float(value), parsed >= 0, parsed.isFinite else {
                    throw CLIError(description: "Invalid -score-weight")
                }
                scoreWeight = parsed
            case "-max-grad-norm":
                guard let parsed = Float(value), parsed > 0, parsed.isFinite else {
                    throw CLIError(description: "Invalid -max-grad-norm")
                }
                maxGradNorm = parsed
            case "-gpu-cache-limit-mb":
                guard let parsed = Int(value), parsed >= 0 else {
                    throw CLIError(description: "Invalid -gpu-cache-limit-mb '\(value)'")
                }
                gpuCacheLimitMB = parsed
            case "-checkpoint-interval":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -checkpoint-interval (must be a positive int)")
                }
                checkpointInterval = parsed
            case "-snapshot-interval":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -snapshot-interval (must be a positive int)")
                }
                snapshotInterval = parsed
            case "-val-data":
                validationDataPath = value
            case "-val-interval":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -val-interval (must be a positive int)")
                }
                validationInterval = parsed
            case "-init-model": initModelPath = value
            default: throw CLIError(description: "Unknown train option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard let dataPath, let outputPath else { throw CLIError(description: usage) }
        guard let arch = RinGoArchitecture.Variant(rawValue: archName) else {
            let supported = RinGoArchitecture.Variant.allCases.map(\.rawValue).joined(separator: ", ")
            throw CLIError(description: "Invalid -arch '\(archName)'. Supported: \(supported)")
        }
        if validationInterval != nil, validationDataPath == nil {
            throw CLIError(description: "-val-interval requires -val-data")
        }
        // WP-8: validate -init-model before the (expensive) shard load so config errors fail fast.
        // Two independent guards: (1) the file must exist and (2) it must not be combined with a
        // resume of an existing checkpoint (that pairing is ambiguous — see `initModelResumeConflict`).
        if let initModelPath, !FileManager.default.fileExists(atPath: initModelPath) {
            throw CLIError(description: "-init-model file not found: \(initModelPath)")
        }
        let earlyCheckpointExists = FileManager.default.fileExists(
            atPath: URL(fileURLWithPath: outputPath)
                .appendingPathComponent("checkpoint.safetensors").path
        )
        if let message = initModelResumeConflict(
            initModelPath: initModelPath,
            resume: shouldResume(checkpointExists: earlyCheckpointExists, fresh: fresh)
        ) {
            throw CLIError(description: message)
        }
        // P0-1: if -val-data points inside (or at) the same directory as -data, its val.nngd must
        // never also be read as a training shard — otherwise the "held-out" validation set is
        // actually part of train, and validation loss stops meaning anything. This exclusion is
        // unconditional, NOT gated on -val-data being passed at all: see `trainingExclusionURLs`.
        let dataDirectoryURL = URL(fileURLWithPath: dataPath)
        let excludedShardURLs = trainingExclusionURLs(
            dataDirectory: dataDirectoryURL,
            validationDataPath: validationDataPath
        )
        if let warning = unmonitoredValidationShardWarning(
            dataDirectory: dataDirectoryURL, validationDataPath: validationDataPath
        ) {
            FileHandle.standardError.write(Data((warning + "\n").utf8))
        }
        let dataset = try RinGoDataset.load(directory: dataDirectoryURL, excluding: excludedShardURLs)
        guard dataset.nnLen == 9 else {
            throw CLIError(description: "RinGo v1 training requires 9x9 shards; found nnLen \(dataset.nnLen)")
        }
        // P2 (audit): a validation shard is only ever consumed when -val-interval schedules periodic
        // validation — `Trainer.train` ignores `validation` when `validationInterval == nil`. Loading
        // ~200k samples just to print a count would waste resident memory for nothing, so when
        // -val-data is given WITHOUT -val-interval we skip the load and warn. This does NOT affect the
        // val.nngd training exclusion (that keys off the PATH via `trainingExclusionURLs`, not the
        // loaded shard) nor the standalone `evaluate` subcommand (it loads its own data).
        let validation: RinGoShard?
        if let validationDataPath {
            if validationInterval != nil {
                validation = try loadValidationShard(from: URL(fileURLWithPath: validationDataPath))
            } else {
                validation = nil
                FileHandle.standardError.write(Data((
                    "Note: -val-data given without -val-interval; validation loss will NOT be monitored, "
                        + "so the validation shard is not loaded (saving its resident memory). Pass "
                        + "-val-interval N to enable periodic validation.\n"
                ).utf8))
            }
        } else {
            validation = nil
        }
        if let validation, validation.nnLen != dataset.nnLen {
            throw CLIError(description: "Validation shard nnLen \(validation.nnLen) != training \(dataset.nnLen)")
        }
        let output = URL(fileURLWithPath: outputPath)
        let checkpoint = output.appendingPathComponent("checkpoint.safetensors")
        let checkpointExists = FileManager.default.fileExists(atPath: checkpoint.path)
        let resume = shouldResume(checkpointExists: checkpointExists, fresh: fresh)
        let warmupSteps = min(1000, max(1, steps / 20))

        // config.json: snapshot this run's parameters at startup so an unattended run directory is
        // self-describing (P0-3.1) and metrics-trim progress is auditable (P0-3.3). P0-2 (audit):
        // an earlier version of this merged the new snapshot into whatever config.json already
        // existed on disk, keeping any pre-existing key — which let stale values (e.g. a
        // validation_shard_count computed by an older, buggy build) survive silently forever
        // across reruns/resumes of the same output directory. `writeConfig` now always writes the
        // CURRENT invocation's values with no merge; the resume branch below layers only the two
        // genuinely run-state keys (resume_step, metrics_csv_trimmed_at_step) on top afterward.
        let configURL = output.appendingPathComponent("config.json")
        // Resume safety (WP-5): config.json records this run's architecture, and `writeConfig`
        // below overwrites it with the CURRENT `-arch`. So BEFORE that overwrite, if we're resuming
        // an existing run, compare the requested `-arch` against the architecture the previous run
        // recorded and refuse a mismatch loudly rather than misloading the checkpoint into a
        // differently shaped network. (For a pre-WP-5 checkpoint with no recorded arch, this check
        // is a no-op and `TrainableNetwork.replaceParameters`' key/shape check is the backstop.)
        if resume, let previousArch = previousConfigArch(at: configURL), previousArch != arch.rawValue {
            throw CLIError(description: "Refusing to resume: the checkpoint at \(checkpoint.path) was "
                + "trained with -arch \(previousArch), but this run requested -arch \(arch.rawValue). "
                + "Architectures cannot be mixed on the same run. Re-run with -arch \(previousArch) to "
                + "continue it, or start over with -fresh (ideally pointing -out at a new directory).")
        }
        let lossWeights = LossWeights(
            policy: 1, value: valueWeight, score: scoreWeight, ownership: ownershipWeight
        )
        // WP-8: record init-model provenance (path + sha256 prefix of the exact file loaded) so a
        // warm-started run directory is self-describing about where its starting weights came from.
        let initModelSha256 = initModelPath.flatMap { fileSha256Prefix($0) }
        try writeConfig(
            at: configURL, arch: arch.rawValue,
            dataPath: dataPath, validationDataPath: validationDataPath,
            samples: dataset.sampleCount,
            validationSamples: validation?.sampleCount,
            batchSize: batchSize, steps: steps, learningRate: learningRate,
            warmupSteps: warmupSteps, weightDecay: TrainCommand.defaultWeightDecay,
            maximumGradientNorm: maxGradNorm,
            lossWeights: lossWeights,
            checkpointInterval: checkpointInterval, snapshotInterval: snapshotInterval,
            metricsInterval: 1, gpuCacheLimitMB: gpuCacheLimitMB,
            resumeStep: nil,
            metricsTrimmedAtStep: nil,
            initModelPath: initModelPath,
            initModelSha256: initModelSha256
        )

        let architecture = RinGoArchitecture.make(arch)
        let network = try TrainableNetwork(desc: architecture, nnXLen: 9, nnYLen: 9)
        // WP-8: warm-start the (otherwise random-init) network from an exported model. The
        // `initModelResumeConflict` guard above already rejected `-init-model` + a resumed
        // checkpoint, so this branch always starts from step 0 with a fresh (zeroed) optimizer.
        if let initModelPath {
            let donor: ModelDesc
            do {
                donor = try ModelDesc.loadFromFileMaybeGZipped(initModelPath)
            } catch {
                throw CLIError(description: "-init-model: could not parse \(initModelPath): \(error)")
            }
            do {
                try network.loadDonorParameters(from: donor)
            } catch {
                throw CLIError(description: "-init-model: failed to load \(initModelPath) into a "
                    + "\(arch.rawValue) network: \(error) Its architecture must match -arch "
                    + "\(arch.rawValue) (same block structure, widths, and layer names).")
            }
        }
        let trainer = try Trainer(network: network, configuration: TrainerConfiguration(
            batchSize: batchSize,
            totalSteps: steps,
            learningRate: learningRate,
            warmupSteps: warmupSteps,
            weightDecay: TrainCommand.defaultWeightDecay,
            maximumGradientNorm: maxGradNorm,
            lossWeights: lossWeights,
            gpuCacheLimitMB: gpuCacheLimitMB,
            validationInterval: validationInterval
        ))
        if resume {
            do {
                try trainer.loadCheckpoint(from: checkpoint)
            } catch {
                throw CLIError(description: "Failed to load checkpoint at \(checkpoint.path): \(error). "
                    + "Use -fresh to start over without loading it.")
            }
            // `Trainer.train()` only appends to metrics.csv; it never truncates. If a previous run
            // was interrupted between checkpoints (checkpointInterval steps apart, default 1000),
            // the log can already contain rows PAST the checkpoint's restored step (progress that
            // was logged but whose weights were never checkpointed and are therefore gone). Left
            // alone, resuming would re-log those same step numbers a second time, so the file would
            // contain duplicate/overlapping steps right at the resume point — harmless to training
            // itself (the checkpoint, not the log, is the source of truth for weights/optimizer
            // state), but confusing for anyone plotting the loss curve. Trim any such rows before
            // training continues, so the log stays a clean, strictly-increasing record.
            try trimMetrics(at: output.appendingPathComponent("metrics.csv"), keepingStepsUpTo: trainer.step)
            try updateConfig(at: configURL, resumeStep: trainer.step, metricsTrimmedAtStep: trainer.step)
        }
        // Print the run's configuration once, up front, so a long unattended run is self-describing.
        // The GPU cache limit is read back from MLX after the Trainer applied it, and labelled
        // "(user-set)" vs "(MLX default)" exactly as `ringo benchmark` does.
        printBanner(
            arch: arch, parameterCount: architecture.numParameters,
            dataPath: dataPath, outputPath: outputPath, samples: dataset.sampleCount,
            validationDataPath: validationDataPath, validationSamples: validation?.sampleCount,
            batchSize: batchSize, steps: steps, learningRate: learningRate, warmupSteps: warmupSteps,
            valueWeight: valueWeight, ownershipWeight: ownershipWeight, scoreWeight: scoreWeight,
            maxGradNorm: maxGradNorm, gpuCacheLimitMB: gpuCacheLimitMB, resumeStep: resume ? trainer.step : nil
        )
        try trainer.train(dataset: dataset, outputDirectory: output, validation: validation)
        try trainer.saveCheckpoint(to: checkpoint)
        try ModelWriter.write(desc: network.exportDesc(), to: output.appendingPathComponent("model-final.bin.gz"))
        try installBestValModel(in: output)
    }

    static func shouldResume(checkpointExists: Bool, fresh: Bool) -> Bool {
        checkpointExists && !fresh
    }

    /// The single held-out validation shard's filename inside a `-val-data` directory. Exposed
    /// separately from `loadValidationShard` so callers can also use it as the `excluding:` set
    /// passed to `RinGoDataset.load` (P0-1: without this, the training loader would read
    /// `val.nngd` too when it lives in the same directory as the training shards).
    static func validationShardURL(from directory: URL) -> URL {
        directory.appendingPathComponent("val.nngd")
    }

    /// The shard URLs `-data`'s directory listing must never feed into the training set (P0-1).
    ///
    /// Always includes `<dataDirectory>/val.nngd`, regardless of whether `-val-data` was passed at
    /// all: `makedata -val-ratio` always writes `val.nngd` alongside the training shards, so a
    /// shard directory can carry a holdout that this particular `train` invocation never asked to
    /// validate against. Without this unconditional exclusion, running plain `train -data <dir>`
    /// on such a directory would silently fold `val.nngd` into training — and a later, separate
    /// `ringo evaluate` run against that same `val.nngd` would then be evaluating on data the
    /// model already trained on, with no warning that the holdout had been compromised.
    ///
    /// Also includes the shard resolved from an explicit `-val-data <dir>` (its `val.nngd`), which
    /// may live in a different directory than `-data` and so isn't already covered by the above.
    static func trainingExclusionURLs(dataDirectory: URL, validationDataPath: String?) -> Set<URL> {
        let implied = validationShardURL(from: dataDirectory)
        let resolved = validationDataPath.map { validationShardURL(from: URL(fileURLWithPath: $0)) }
        return Set([implied, resolved].compactMap(\.self))
    }

    /// A human-readable warning when `dataDirectory` contains a `val.nngd` that this run is
    /// excluding from training but NOT validating against (no `-val-data` given) — `nil` when
    /// there's nothing to warn about. Surfaced so a user reusing a `makedata -val-ratio` shard
    /// directory without `-val-data` finds out their holdout exists and isn't being monitored,
    /// rather than silently training with an untouched-but-also-unused val.nngd sitting there.
    static func unmonitoredValidationShardWarning(dataDirectory: URL, validationDataPath: String?) -> String? {
        guard validationDataPath == nil else { return nil }
        let url = validationShardURL(from: dataDirectory)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return "Note: \(url.path) exists but -val-data was not given; it is excluded from "
            + "training (not used as train data) but validation loss is NOT being monitored "
            + "this run. Pass -val-data \(dataDirectory.path) to validate against it, or "
            + "move/remove the file if it isn't meant to be a holdout."
    }

    /// Loads the single held-out validation shard `val.nngd` from a shard directory.
    /// `RinGoDataset.load(directory:)` would load *every* .nngd in the directory, which includes
    /// the training shards and defeats the purpose of a validation split.
    static func loadValidationShard(from directory: URL) throws -> RinGoShard {
        let url = validationShardURL(from: directory)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw CLIError(description: "Validation shard not found at \(url.path). "
                + "Run makedata with -val-ratio or place a val.nngd file there.")
        }
        return try RinGoShard.read(url: url)
    }

    /// Rewrites `metrics.csv` (header + rows with `step <= keepStep`), dropping any tail rows a
    /// prior, interrupted run logged past the checkpoint it actually managed to save. A missing
    /// file, or a file with only a header, is left as-is (nothing to trim). A row whose step
    /// column doesn't parse as an integer is dropped rather than kept, since it can't be compared.
    static func trimMetrics(at url: URL, keepingStepsUpTo keepStep: Int) throws {
        guard let contents = try? String(contentsOf: url, encoding: .utf8) else { return }
        var lines = contents.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        guard let header = lines.first else { return }
        lines.removeFirst()
        let kept = lines.filter { line in
            guard let stepField = line.split(separator: ",", maxSplits: 1).first,
                  let step = Int(stepField)
            else { return false }
            return step <= keepStep
        }
        guard kept.count != lines.count else { return }
        let rewritten = ([header] + kept).joined(separator: "\n") + "\n"
        try rewritten.write(to: url, atomically: true, encoding: .utf8)
    }

    private static func printBanner(
        arch: RinGoArchitecture.Variant, parameterCount: Int64,
        dataPath: String, outputPath: String, samples: Int,
        validationDataPath: String?, validationSamples: Int?,
        batchSize: Int, steps: Int, learningRate: Float, warmupSteps: Int,
        valueWeight: Float, ownershipWeight: Float, scoreWeight: Float,
        maxGradNorm: Float, gpuCacheLimitMB: Int?, resumeStep: Int?
    ) {
        let cacheBytes = MLX.Memory.cacheLimit
        let cacheLabel = gpuCacheLimitMB != nil
            ? "\(formatBytes(cacheBytes)) (user-set)"
            : "\(formatBytes(cacheBytes)) (MLX default)"
        let runState = if let resumeStep {
            "resuming from step \(resumeStep) (target: \(steps))"
        } else {
            "fresh start (target: \(steps))"
        }
        // P0-1: this line is train-only precisely because `dataset` (the `samples` count here) was
        // loaded with the validation shard excluded — see the `excluding:` RinGoDataset.load call
        // above. The separate `val:` line (if present) makes both counts explicit so a run's log is
        // self-auditing without having to dig into config.json.
        let validationLine = if let validationDataPath, let validationSamples {
            "\n  val:  \(validationDataPath)  (\(validationSamples) samples)"
        } else {
            ""
        }
        let banner = """
        RinGo training (\(arch.rawValue), 9x9, \(parameterCount) params)
          run:  \(runState)
          data: \(dataPath)  (\(samples) train samples)\(validationLine)
          out:  \(outputPath)
          batch: \(batchSize)   steps: \(steps)   lr: \(learningRate)   warmup: \(warmupSteps)
          loss weights: policy=1 value=\(valueWeight) ownership=\(ownershipWeight) score=\(scoreWeight)
          max grad norm: \(maxGradNorm)   GPU cache limit: \(cacheLabel)
        """
        print(banner)
    }

    private static func formatBytes(_ bytes: Int) -> String {
        String(format: "%.0f MB", Double(bytes) / (1024 * 1024))
    }

    /// Defaults that `TrainCommand` may expose as CLI flags or feed into the trainer.
    /// Pinned here as named constants so `config.json` records the same numbers the trainer will
    /// actually run with, rather than baking literals into two places.
    static let defaultWeightDecay: Float = 0.01
    static let defaultMaxGradNorm: Float = 5
    static let defaultValueWeight: Float = 3
    static let defaultOwnershipWeight: Float = 0.5
    static let defaultScoreWeight: Float = 0.02

    /// Writes `config.json` (P0-3.1) into the run directory, unconditionally overwriting whatever
    /// was there before with the values this invocation actually computed.
    ///
    /// P0-2 (audit finding): an earlier version merged onto any existing file, keeping every
    /// pre-existing top-level key and only adding keys missing from it. That let derived values —
    /// `shard_count`, `validation_shard_count` in particular — go stale forever once written once
    /// under an older or buggier build: a resumed or rerun training job would keep reporting a
    /// wrong validation sample count no matter how many times the loader was fixed afterward,
    /// because the merge never let a corrected value overwrite the file's existing one. There is no
    /// upside to preserving old values here: every field this function writes is either directly
    /// determined by this invocation's CLI arguments/loaded data (hyperparameters, sample counts,
    /// paths) or is cheap to recompute (`git_commit`, `git_dirty`, `start_time`). Config.json is
    /// therefore always a live, trustworthy snapshot of "what is this run doing right now" — never
    /// a mix of this run and some earlier one. The two genuinely run-state fields that are NOT known
    /// yet at this call site on a resume (`resume_step`, `metrics_csv_trimmed_at_step`, both only
    /// known after `Trainer.loadCheckpoint` runs) are layered on afterward by `updateConfig`, which
    /// deliberately does still read-modify-write — but only ever touches those two keys.
    static func writeConfig(
        at url: URL,
        arch: String,
        dataPath: String,
        validationDataPath: String?,
        samples: Int,
        validationSamples: Int?,
        batchSize: Int,
        steps: Int,
        learningRate: Float,
        warmupSteps: Int,
        weightDecay: Float,
        maximumGradientNorm: Float,
        lossWeights: LossWeights,
        checkpointInterval: Int,
        snapshotInterval: Int,
        metricsInterval: Int,
        gpuCacheLimitMB: Int?,
        resumeStep: Int?,
        metricsTrimmedAtStep: Int?,
        initModelPath: String? = nil,
        initModelSha256: String? = nil
    ) throws {
        var dictionary: [String: Any] = [
            "git_commit": gitCommitShort(),
            "git_dirty": gitDirty(),
            "arch": arch,
            "data_directory": dataPath,
            "shard_count": samples,
            "nn_len": 9,
            "batch_size": batchSize,
            "total_steps": steps,
            "learning_rate": learningRate,
            "warmup_steps": warmupSteps,
            "weight_decay": weightDecay,
            "max_grad_norm": maximumGradientNorm,
            "loss_weights": [
                "policy": lossWeights.policy,
                "value": lossWeights.value,
                "score": lossWeights.score,
                "ownership": lossWeights.ownership,
            ],
            "checkpoint_interval": checkpointInterval,
            "snapshot_interval": snapshotInterval,
            "metrics_interval": metricsInterval,
            "gpu_cache_limit_mb": gpuCacheLimitMB as Any,
            "start_time": ISO8601DateFormatter().string(from: Date()),
        ]
        if let validationDataPath {
            dictionary["validation_data_directory"] = validationDataPath
            dictionary["validation_shard_count"] = (validationSamples ?? 0) as Any
        }
        if let resumeStep { dictionary["resume_step"] = resumeStep }
        if let metricsTrimmedAtStep { dictionary["metrics_csv_trimmed_at_step"] = metricsTrimmedAtStep }
        if let initModelPath {
            dictionary["init_model"] = initModelPath
            if let initModelSha256 { dictionary["init_model_sha256"] = initModelSha256 }
        }

        try writeJSON(dictionary, at: url)
    }

    /// WP-8: `-init-model` and resuming an existing checkpoint are mutually exclusive — both would
    /// determine the starting weights, and silently preferring one over the other would be a
    /// footgun (e.g. quietly ignoring the requested init model and continuing an old run). Returns
    /// a user-facing error message when the pairing is present, or `nil` when it is safe to proceed.
    ///
    /// `-fresh` (which makes `resume == false`) disambiguates toward `-init-model`, so the conflict
    /// only fires when a checkpoint would actually be resumed. Pure — unit-tested directly.
    static func initModelResumeConflict(initModelPath: String?, resume: Bool) -> String? {
        guard initModelPath != nil, resume else { return nil }
        return "Refusing to run: -init-model was given but -out already contains a "
            + "checkpoint.safetensors that would be resumed. These are ambiguous (both set the "
            + "starting weights). Either resume the existing run (drop -init-model), or warm-start "
            + "from the model into a FRESH run — point -out at a new directory, or pass -fresh to "
            + "ignore the existing checkpoint."
    }

    /// The lowercase-hex SHA-256 prefix of the file at `path` (default 12 chars), or `nil` when the
    /// file can't be read. Used purely for `config.json` init-model provenance, so a short prefix
    /// is plenty to identify which exported net a warm-started run started from.
    static func fileSha256Prefix(_ path: String, length: Int = 12) -> String? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }
        let hex = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(length))
    }

    /// The architecture string recorded in a previous run's `config.json` (`nil` if the file is
    /// absent or predates WP-5's `arch` key). Read BEFORE `writeConfig` overwrites the file, so a
    /// resume can compare it against the requested `-arch` and refuse a mismatch (WP-5, task 4).
    static func previousConfigArch(at url: URL) -> String? {
        (try? readJSON(at: url))?["arch"] as? String
    }

    /// Updates run-state keys only (`resume_step`, `metrics_csv_trimmed_at_step`); never overwrites
    /// the snapshot keys written at startup (P0-3.3: trim progress must be auditable across resumes).
    static func updateConfig(at url: URL, resumeStep: Int?, metricsTrimmedAtStep: Int?) throws {
        var existing = (try? readJSON(at: url)) ?? [:]
        if let resumeStep { existing["resume_step"] = resumeStep }
        if let metricsTrimmedAtStep { existing["metrics_csv_trimmed_at_step"] = metricsTrimmedAtStep }
        try writeJSON(existing, at: url)
    }

    /// Selects the final `model-best-val.bin.gz` (P0-1.2): if `validation_metrics.csv` exists and
    /// contains a row whose step matches a saved `model-step-*.bin.gz`, symlinks to that snapshot;
    /// otherwise copies `model-final.bin.gz` to `model-best-val.bin.gz` as the safe fallback.
    static func installBestValModel(in outputDirectory: URL) throws {
        let bestValURL = outputDirectory.appendingPathComponent("model-best-val.bin.gz")
        let finalURL = outputDirectory.appendingPathComponent("model-final.bin.gz")
        guard let bestStep = try readBestValidationStep(in: outputDirectory),
              let snapshotURL = findSnapshot(for: bestStep, in: outputDirectory),
              FileManager.default.fileExists(atPath: snapshotURL.path)
        else {
            try? FileManager.default.removeItem(at: bestValURL)
            try FileManager.default.copyItem(at: finalURL, to: bestValURL)
            return
        }
        linkOrCopy(from: snapshotURL, to: bestValURL)
    }

    /// Reads `validation_metrics.csv` and returns the step with the smallest `total_loss`.
    static func readBestValidationStep(in directory: URL) throws -> Int? {
        let url = directory.appendingPathComponent("validation_metrics.csv")
        guard let contents = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let rows = contents.split(separator: "\n", omittingEmptySubsequences: true).dropFirst()
        var bestStep: Int?
        var bestLoss: Float = .infinity
        for row in rows {
            let columns = row.split(separator: ",")
            // Schema: step,policy_loss,value_loss,score_loss,ownership_loss,total_loss,sample_count
            guard columns.count >= 6,
                  let step = Int(columns[0]),
                  let totalLoss = Float(columns[5])
            else { continue }
            if totalLoss < bestLoss {
                bestLoss = totalLoss
                bestStep = step
            }
        }
        return bestStep
    }

    /// Returns the URL of `model-step-XXXXXXXX.bin.gz` matching the given step, if it exists.
    static func findSnapshot(for step: Int, in directory: URL) -> URL? {
        let name = String(format: "model-step-%08d.bin.gz", step)
        let url = directory.appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Symlinks if possible (preserves the snapshot it points at when later pruned), falls back to a
    /// real copy when symlinks aren't available. The fallback is rare but keeps the file present for
    /// a host that has a read-only target or some other POSIX-odd file system.
    static func linkOrCopy(from source: URL, to destination: URL) {
        try? FileManager.default.removeItem(at: destination)
        do {
            try FileManager.default.createSymbolicLink(at: destination, withDestinationURL: source)
        } catch {
            try? FileManager.default.copyItem(at: source, to: destination)
        }
    }

    private static func gitCommitShort() -> String {
        guard let output = try? shellCapture("/usr/bin/git", arguments: ["rev-parse", "--short", "HEAD"]) else {
            return "unknown"
        }
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "unknown" : trimmed
    }

    private static func gitDirty() -> Bool {
        guard let output = try? shellCapture("/usr/bin/git", arguments: ["status", "--porcelain"]) else {
            return false
        }
        return !output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func shellCapture(_ command: String, arguments: [String]) throws -> String {
        let pipe = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command)
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    private static func readJSON(at url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        let object = try JSONSerialization.jsonObject(with: data, options: [])
        return (object as? [String: Any]) ?? [:]
    }

    private static func writeJSON(_ dictionary: [String: Any], at url: URL) throws {
        let data = try JSONSerialization.data(
            withJSONObject: dictionary, options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: url, options: .atomic)
    }
}
