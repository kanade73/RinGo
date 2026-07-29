import Foundation
@testable import ringo
import RinGoTrain
import XCTest

final class TrainCommandTests: XCTestCase {
    func testResumesOnlyWhenCheckpointExistsAndFreshWasNotRequested() {
        XCTAssertTrue(TrainCommand.shouldResume(checkpointExists: true, fresh: false))
        XCTAssertFalse(TrainCommand.shouldResume(checkpointExists: true, fresh: true))
        XCTAssertFalse(TrainCommand.shouldResume(checkpointExists: false, fresh: false))
        XCTAssertFalse(TrainCommand.shouldResume(checkpointExists: false, fresh: true))
    }

    /// Regression for a real gap found verifying resume-from-checkpoint on the host: an
    /// interrupted run can log metrics rows past the step its last checkpoint actually captured
    /// (checkpoints only land every `checkpointInterval` steps, but every step gets logged).
    /// Resuming without trimming those orphaned tail rows first produces duplicate/overlapping
    /// step numbers in metrics.csv once the resumed run re-logs the same steps. This must not
    /// happen: `trimMetrics` is called (see `main`) right after a checkpoint loads, before
    /// training continues.
    func testTrimMetricsDropsRowsPastTheRestoredStepAndKeepsTheHeader() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("trim-metrics-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("metrics.csv")

        let header = "step,learning_rate,policy_loss,value_loss,score_loss,ownership_loss,total_loss"
        let rows = (1 ... 1130).map { "\($0),0.0003,1,1,1,1,4" }
        try ([header] + rows).joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)

        try TrainCommand.trimMetrics(at: url, keepingStepsUpTo: 1000)

        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        XCTAssertEqual(lines.first, header)
        let steps = lines.dropFirst().map { Int($0.split(separator: ",", maxSplits: 1).first!)! }
        XCTAssertEqual(steps, Array(1 ... 1000))
    }

    func testTrimMetricsIsANoOpWhenNothingExceedsTheRestoredStep() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("trim-metrics-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("metrics.csv")
        let content = "step,learning_rate,policy_loss,value_loss,score_loss,ownership_loss,total_loss\n1,1,1,1,1,1,1\n"
        try content.write(to: url, atomically: true, encoding: .utf8)

        try TrainCommand.trimMetrics(at: url, keepingStepsUpTo: 1000)

        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), content)
    }

    func testTrimMetricsToleratesAMissingFile() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).csv")
        XCTAssertNoThrow(try TrainCommand.trimMetrics(at: missing, keepingStepsUpTo: 1000))
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing.path))
    }

    // MARK: - Snapshot/checkpoint intervals

    /// The PRE_TOURNAMENT spec strictly asks for snapshotInterval to default to 5000 (vs the old
    /// 1000) so fresh runs don't snapshot twice as often as they checkpoint. Pinning this asserts
    /// the default didn't drift back silently.
    func testTrainerConfigurationSnapshotIntervalDefaultsTo5000WhenOmitted() {
        let configuration = TrainerConfiguration(totalSteps: 100)
        XCTAssertEqual(configuration.checkpointInterval, 1000)
        XCTAssertEqual(configuration.snapshotInterval, 5000)
        XCTAssertNil(configuration.validationInterval)
    }

    /// RinGo's small-scale Step A needs stronger value/ownership signal and looser gradient clipping
    /// than KataGo's reference settings for large-scale data. These defaults are tuned from the
    /// `TrainingDiagnosticsTests` runs.
    func testTrainCommandDefaultsFavorValueOwnershipLearning() {
        XCTAssertEqual(TrainCommand.defaultValueWeight, 3)
        XCTAssertEqual(TrainCommand.defaultOwnershipWeight, 0.5)
        XCTAssertEqual(TrainCommand.defaultScoreWeight, 0.02)
        XCTAssertEqual(TrainCommand.defaultMaxGradNorm, 5)
    }

    // MARK: - model-best-val selection

    func testInstallBestValCopiesModelFinalWhenNoValidationMetricsExist() throws {
        let directory = try makeTempDirectory()
        let final = directory.appendingPathComponent("model-final.bin.gz")
        try Data("FINAL".utf8).write(to: final)

        try TrainCommand.installBestValModel(in: directory)

        let bestVal = directory.appendingPathComponent("model-best-val.bin.gz")
        XCTAssertTrue(FileManager.default.fileExists(atPath: bestVal.path))
        XCTAssertEqual(try Data(contentsOf: bestVal), Data("FINAL".utf8))
    }

    func testInstallBestValSymlinksToSnapshotMatchingBestValidationStep() throws {
        let directory = try makeTempDirectory()
        let final = directory.appendingPathComponent("model-final.bin.gz")
        try Data("FINAL".utf8).write(to: final)
        let snap = directory.appendingPathComponent("model-step-00005000.bin.gz")
        try Data("SNAP-5000".utf8).write(to: snap)
        let otherSnap = directory.appendingPathComponent("model-step-00010000.bin.gz")
        try Data("SNAP-10000".utf8).write(to: otherSnap)
        let metrics = directory.appendingPathComponent("validation_metrics.csv")
        // Schema: step,policy_loss,value_loss,score_loss,ownership_loss,total_loss,sample_count
        let csv = """
        step,policy_loss,value_loss,score_loss,ownership_loss,total_loss,sample_count
        5000,1,1,1,1,0.7,1000
        10000,1,1,1,1,1.2,1000
        """
        try Data(csv.utf8).write(to: metrics)

        try TrainCommand.installBestValModel(in: directory)

        let bestVal = directory.appendingPathComponent("model-best-val.bin.gz")
        // Resolve the symlink and read what it points at: should be the 5000 snapshot's bytes.
        let resolved = bestVal.resolvingSymlinksInPath()
        XCTAssertEqual(try Data(contentsOf: resolved), Data("SNAP-5000".utf8))
    }

    func testInstallBestValCopiesFinalWhenBestValidationStepHasNoSnapshot() throws {
        let directory = try makeTempDirectory()
        let final = directory.appendingPathComponent("model-final.bin.gz")
        try Data("FINAL".utf8).write(to: final)
        let metrics = directory.appendingPathComponent("validation_metrics.csv")
        let csv = """
        step,policy_loss,value_loss,score_loss,ownership_loss,total_loss,sample_count
        9999,1,1,1,1,0.7,1000
        """
        try Data(csv.utf8).write(to: metrics)

        try TrainCommand.installBestValModel(in: directory)

        let bestVal = directory.appendingPathComponent("model-best-val.bin.gz")
        XCTAssertEqual(try Data(contentsOf: bestVal), Data("FINAL".utf8))
    }

    func testReadBestValidationStepPicksTheSmallestTotalLoss() throws {
        let directory = try makeTempDirectory()
        let metrics = directory.appendingPathComponent("validation_metrics.csv")
        let csv = """
        step,policy_loss,value_loss,score_loss,ownership_loss,total_loss,sample_count
        100,1,1,1,1,2.0,100
        200,1,1,1,1,0.5,100
        300,1,1,1,1,1.5,100
        """
        try Data(csv.utf8).write(to: metrics)

        XCTAssertEqual(try TrainCommand.readBestValidationStep(in: directory), 200)
    }

    func testReadBestValidationStepReturnsNilWhenMetricsFileMissing() throws {
        let directory = try makeTempDirectory()
        XCTAssertNil(try TrainCommand.readBestValidationStep(in: directory))
    }

    func testFindSnapshotResolvesOnlyForExistingFilesWithExactStepName() throws {
        let directory = try makeTempDirectory()
        let snap = directory.appendingPathComponent("model-step-00005000.bin.gz")
        try Data().write(to: snap)
        XCTAssertEqual(
            TrainCommand.findSnapshot(for: 5000, in: directory),
            snap
        )
        XCTAssertNil(TrainCommand.findSnapshot(for: 6000, in: directory))
    }

    // MARK: - Hardening: val.nngd exclusion must not depend on -val-data being passed

    /// The orchestrator-flagged residual leak: a shard directory produced by `makedata -val-ratio`
    /// always has `val.nngd` sitting next to the training shards, whether or not this particular
    /// `train` invocation passes `-val-data`. `trainingExclusionURLs` must exclude it either way.
    func testTrainingExclusionURLsAlwaysExcludesValNngdInTheDataDirectoryEvenWithoutValData() {
        let directory = URL(fileURLWithPath: "/tmp/shards")

        let withoutValData = TrainCommand.trainingExclusionURLs(dataDirectory: directory, validationDataPath: nil)
        XCTAssertEqual(withoutValData, [directory.appendingPathComponent("val.nngd")])

        let withValData = TrainCommand.trainingExclusionURLs(
            dataDirectory: directory, validationDataPath: directory.path
        )
        XCTAssertEqual(withValData, [directory.appendingPathComponent("val.nngd")])
    }

    /// `-val-data` pointing at a DIFFERENT directory than `-data` contributes its own val.nngd to
    /// the exclusion set too, on top of (not instead of) the implied one in the `-data` directory.
    func testTrainingExclusionURLsIncludesBothImpliedAndExplicitValidationShards() {
        let dataDirectory = URL(fileURLWithPath: "/tmp/shards")
        let otherValDirectory = URL(fileURLWithPath: "/tmp/other-val")

        let excluded = TrainCommand.trainingExclusionURLs(
            dataDirectory: dataDirectory, validationDataPath: otherValDirectory.path
        )

        XCTAssertEqual(excluded, [
            dataDirectory.appendingPathComponent("val.nngd"),
            otherValDirectory.appendingPathComponent("val.nngd"),
        ])
    }

    func testUnmonitoredValidationShardWarningFiresOnlyWhenValNngdExistsAndValDataWasNotPassed() throws {
        let directory = try makeTempDirectory()
        try Data("shard".utf8).write(to: directory.appendingPathComponent("val.nngd"))

        // val.nngd exists, -val-data not given: warn.
        let warning = TrainCommand.unmonitoredValidationShardWarning(
            dataDirectory: directory, validationDataPath: nil
        )
        XCTAssertNotNil(warning)
        XCTAssertTrue(warning?.contains("val.nngd") ?? false)
        XCTAssertTrue(warning?.contains("-val-data") ?? false)

        // val.nngd exists, but -val-data WAS given: this run is monitoring it, no warning needed.
        XCTAssertNil(TrainCommand.unmonitoredValidationShardWarning(
            dataDirectory: directory, validationDataPath: directory.path
        ))
    }

    func testUnmonitoredValidationShardWarningIsNilWhenNoValNngdExists() throws {
        let directory = try makeTempDirectory()
        XCTAssertNil(TrainCommand.unmonitoredValidationShardWarning(
            dataDirectory: directory, validationDataPath: nil
        ))
    }

    // MARK: - config.json

    func testWriteConfigCreatesFileWithExpectedRunSnapshotKeys() throws {
        let directory = try makeTempDirectory()
        let configURL = directory.appendingPathComponent("config.json")

        try TrainCommand.writeConfig(
            at: configURL, arch: "b6c96", dataPath: "shards/", validationDataPath: "shards/val",
            samples: 4_000_000, validationSamples: 200_000,
            batchSize: 256, steps: 50000, learningRate: 3e-4, warmupSteps: 1000,
            weightDecay: 0.01, maximumGradientNorm: 1,
            lossWeights: LossWeights(policy: 1, value: 3, score: 0.02, ownership: 0.5),
            checkpointInterval: 1000, snapshotInterval: 5000, metricsInterval: 1,
            gpuCacheLimitMB: 4096, resumeStep: nil, metricsTrimmedAtStep: nil
        )

        let parsed = try TrainCommandTests.readJSON(at: configURL)
        // Hyperparameter keys (the immutable snapshot set). Float -> Double round-trip through JSON
        // introduces the usual fp slack (~7-digit Float precision), so the float-valued keys are
        // checked with tolerance; everything else is exact.
        XCTAssertEqual(parsed["arch"] as? String, "b6c96")
        XCTAssertEqual(parsed["batch_size"] as? Int, 256)
        XCTAssertEqual(parsed["total_steps"] as? Int, 50000)
        let parsedLearningRate = parsed["learning_rate"] as? Double
        XCTAssertEqual(parsedLearningRate ?? .nan, 3e-4, accuracy: 1e-6)
        XCTAssertEqual(parsed["warmup_steps"] as? Int, 1000)
        let parsedWeightDecay = parsed["weight_decay"] as? Double
        XCTAssertEqual(parsedWeightDecay ?? .nan, 0.01, accuracy: 1e-6)
        let parsedMaxGradNorm = parsed["max_grad_norm"] as? Double
        XCTAssertEqual(parsedMaxGradNorm ?? .nan, 1, accuracy: 1e-6)
        let parsedLossWeights = parsed["loss_weights"] as? [String: Double]
        XCTAssertNotNil(parsedLossWeights)
        XCTAssertEqual(parsedLossWeights?["policy"] ?? .nan, 1, accuracy: 1e-6)
        XCTAssertEqual(parsedLossWeights?["value"] ?? .nan, 3, accuracy: 1e-6)
        XCTAssertEqual(parsedLossWeights?["score"] ?? .nan, 0.02, accuracy: 1e-6)
        XCTAssertEqual(parsedLossWeights?["ownership"] ?? .nan, 0.5, accuracy: 1e-6)
        XCTAssertEqual(parsed["checkpoint_interval"] as? Int, 1000)
        XCTAssertEqual(parsed["snapshot_interval"] as? Int, 5000)
        XCTAssertEqual(parsed["nn_len"] as? Int, 9)
        XCTAssertEqual(parsed["data_directory"] as? String, "shards/")
        XCTAssertEqual(parsed["shard_count"] as? Int, 4_000_000)
        XCTAssertEqual(parsed["validation_data_directory"] as? String, "shards/val")
        XCTAssertEqual(parsed["validation_shard_count"] as? Int, 200_000)
        XCTAssertNotNil(parsed["git_commit"])
        XCTAssertNotNil(parsed["start_time"])
        XCTAssertNil(parsed["resume_step"])
        XCTAssertNil(parsed["metrics_csv_trimmed_at_step"])
    }

    /// An earlier version of `writeConfig` merged onto any pre-existing config.json,
    /// keeping every key already on disk and only filling in ones missing from it. That let stale
    /// values — e.g. a `validation_shard_count` written by an older, buggy build — survive forever
    /// across reruns of the same output directory, no matter how many times the underlying bug got
    /// fixed. `writeConfig` must now fully overwrite: every field reflects THIS call's arguments,
    /// full stop, even fields ("immutable snapshot" fields in the old design) that a prior write
    /// had already set to something else.
    func testWriteConfigFullyOverwritesStaleValuesOnRerun() throws {
        let directory = try makeTempDirectory()
        let configURL = directory.appendingPathComponent("config.json")
        let stale: [String: Any] = [
            "git_commit": "STALE-C0FFEE", "start_time": "T0", "batch_size": 16, "total_steps": 1000,
            "validation_shard_count": 4_156_433, "snapshot_interval": 1,
        ]
        try TrainCommandTests.writeJSON(stale, at: configURL)

        try TrainCommand.writeConfig(
            at: configURL, arch: "b6c96", dataPath: "shards/", validationDataPath: "shards/val",
            samples: 3_949_477,
            validationSamples: 206_956, batchSize: 999, steps: 9999, learningRate: 1, warmupSteps: 1,
            weightDecay: 0.01, maximumGradientNorm: 1,
            lossWeights: LossWeights(policy: 1, value: 3, score: 0.02, ownership: 0.5),
            checkpointInterval: 1000, snapshotInterval: 5000, metricsInterval: 1,
            gpuCacheLimitMB: nil, resumeStep: nil, metricsTrimmedAtStep: nil
        )

        let parsed = try TrainCommandTests.readJSON(at: configURL)
        // Every stale value from the prior run is fully replaced by this invocation's own values —
        // none of the old file's content survives, not even the values the old design treated as
        // an immutable snapshot.
        XCTAssertNotEqual(parsed["git_commit"] as? String, "STALE-C0FFEE")
        XCTAssertNotEqual(parsed["start_time"] as? String, "T0")
        XCTAssertEqual(parsed["batch_size"] as? Int, 999)
        XCTAssertEqual(parsed["total_steps"] as? Int, 9999)
        XCTAssertEqual(parsed["shard_count"] as? Int, 3_949_477)
        XCTAssertEqual(parsed["validation_shard_count"] as? Int, 206_956)
        XCTAssertEqual(parsed["snapshot_interval"] as? Int, 5000)
    }

    /// Direct pin for the audit's headline symptom: `validation_shard_count` must equal the actual
    /// val shard sample count passed in, not the combined train+val total a leaky loader would
    /// have produced (`shard_count` and `validation_shard_count` are asserted as DIFFERENT numbers
    /// here specifically to catch a regression that accidentally re-sums them).
    func testWriteConfigValidationShardCountMatchesActualValidationSampleCount() throws {
        let directory = try makeTempDirectory()
        let configURL = directory.appendingPathComponent("config.json")

        try TrainCommand.writeConfig(
            at: configURL, arch: "b6c96", dataPath: "shards/", validationDataPath: "shards/val",
            samples: 3_949_477,
            validationSamples: 206_956, batchSize: 256, steps: 50000, learningRate: 3e-4, warmupSteps: 1000,
            weightDecay: 0.01, maximumGradientNorm: 1,
            lossWeights: LossWeights(policy: 1, value: 3, score: 0.02, ownership: 0.5),
            checkpointInterval: 1000, snapshotInterval: 5000, metricsInterval: 1,
            gpuCacheLimitMB: nil, resumeStep: nil, metricsTrimmedAtStep: nil
        )

        let parsed = try TrainCommandTests.readJSON(at: configURL)
        XCTAssertEqual(parsed["shard_count"] as? Int, 3_949_477)
        XCTAssertEqual(parsed["validation_shard_count"] as? Int, 206_956)
        XCTAssertNotEqual(parsed["shard_count"] as? Int, parsed["validation_shard_count"] as? Int)
    }

    func testUpdateConfigUpdatesRunStateKeysOnlyWithoutTouchingImmutableSnapshot() throws {
        let directory = try makeTempDirectory()
        let configURL = directory.appendingPathComponent("config.json")
        let initial: [String: Any] = ["git_commit": "ORIG", "start_time": "T0"]
        try TrainCommandTests.writeJSON(initial, at: configURL)

        try TrainCommand.updateConfig(at: configURL, resumeStep: 4000, metricsTrimmedAtStep: 4000)

        let parsed = try TrainCommandTests.readJSON(at: configURL)
        XCTAssertEqual(parsed["git_commit"] as? String, "ORIG")
        XCTAssertEqual(parsed["start_time"] as? String, "T0")
        XCTAssertEqual(parsed["resume_step"] as? Int, 4000)
        XCTAssertEqual(parsed["metrics_csv_trimmed_at_step"] as? Int, 4000)
    }

    // MARK: - WP-5: -arch record + resume mismatch guard

    /// The architecture `writeConfig` records must be readable back by `previousConfigArch` (this is
    /// exactly what the resume path compares against the requested `-arch` before overwriting the
    /// file). A config with no `arch` key (a pre-WP-5 run) reads back `nil`, disabling the early
    /// guard so the checkpoint's own shape check becomes the backstop instead.
    func testPreviousConfigArchRoundTripsAndToleratesMissingKey() throws {
        let directory = try makeTempDirectory()
        let configURL = directory.appendingPathComponent("config.json")

        try TrainCommand.writeConfig(
            at: configURL, arch: "b10c128", dataPath: "shards/", validationDataPath: nil,
            samples: 100, validationSamples: nil,
            batchSize: 256, steps: 50000, learningRate: 1e-3, warmupSteps: 1000,
            weightDecay: 0.01, maximumGradientNorm: 5,
            lossWeights: LossWeights(policy: 1, value: 3, score: 0.02, ownership: 0.5),
            checkpointInterval: 1000, snapshotInterval: 5000, metricsInterval: 1,
            gpuCacheLimitMB: nil, resumeStep: nil, metricsTrimmedAtStep: nil
        )
        XCTAssertEqual(TrainCommand.previousConfigArch(at: configURL), "b10c128")

        let noArch = directory.appendingPathComponent("legacy.json")
        try TrainCommandTests.writeJSON(["git_commit": "OLD"], at: noArch)
        XCTAssertNil(TrainCommand.previousConfigArch(at: noArch))

        let missing = directory.appendingPathComponent("does-not-exist.json")
        XCTAssertNil(TrainCommand.previousConfigArch(at: missing))
    }

    // MARK: - WP-8: -init-model interaction rules + provenance

    /// The `-init-model` + existing-checkpoint ambiguity guard: it must fire ONLY when a checkpoint
    /// would actually be resumed AND an init model was requested. `-fresh` (resume == false)
    /// disambiguates toward the init model, so no conflict there; and neither flag alone conflicts.
    func testInitModelResumeConflictFiresOnlyWhenResumingAnExistingCheckpointWithInitModel() {
        // Both present and the checkpoint would resume -> conflict.
        XCTAssertNotNil(TrainCommand.initModelResumeConflict(initModelPath: "net.bin.gz", resume: true))
        // -init-model into a fresh run (no resume) -> allowed.
        XCTAssertNil(TrainCommand.initModelResumeConflict(initModelPath: "net.bin.gz", resume: false))
        // Plain resume with no -init-model -> allowed.
        XCTAssertNil(TrainCommand.initModelResumeConflict(initModelPath: nil, resume: true))
        // Neither -> allowed.
        XCTAssertNil(TrainCommand.initModelResumeConflict(initModelPath: nil, resume: false))
    }

    /// The conflict message must actually name the moving parts so a user can act on it: the flag,
    /// the checkpoint, and the two escape hatches (-fresh / a fresh -out directory).
    func testInitModelResumeConflictMessageIsActionable() {
        let message = TrainCommand.initModelResumeConflict(initModelPath: "net.bin.gz", resume: true)
        XCTAssertNotNil(message)
        XCTAssertTrue(message?.contains("-init-model") ?? false)
        XCTAssertTrue(message?.contains("checkpoint.safetensors") ?? false)
        XCTAssertTrue(message?.contains("-fresh") ?? false)
    }

    /// A repeatable file gets a stable, nonempty lowercase-hex prefix; a missing file yields nil.
    func testFileSha256PrefixIsStableForContentAndNilForMissingFile() throws {
        let directory = try makeTempDirectory()
        let file = directory.appendingPathComponent("model.bin.gz")
        try Data("ringo-donor-weights".utf8).write(to: file)

        let first = TrainCommand.fileSha256Prefix(file.path)
        let second = TrainCommand.fileSha256Prefix(file.path)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first?.count, 12)
        XCTAssertTrue(first?.allSatisfy(\.isHexDigit) ?? false)
        XCTAssertNil(TrainCommand.fileSha256Prefix(directory.appendingPathComponent("nope").path))
    }

    /// `config.json` must record init-model provenance (path + sha256 prefix) when a run warm-starts
    /// from a model, and must OMIT both keys entirely on an ordinary (random-init) run.
    func testWriteConfigRecordsInitModelProvenanceOnlyWhenGiven() throws {
        let directory = try makeTempDirectory()
        let withInit = directory.appendingPathComponent("with-init.json")
        try TrainCommand.writeConfig(
            at: withInit, arch: "b6c96", dataPath: "shards/", validationDataPath: nil,
            samples: 100, validationSamples: nil,
            batchSize: 256, steps: 50000, learningRate: 1e-3, warmupSteps: 1000,
            weightDecay: 0.01, maximumGradientNorm: 5,
            lossWeights: LossWeights(policy: 1, value: 3, score: 0.02, ownership: 0.5),
            checkpointInterval: 1000, snapshotInterval: 5000, metricsInterval: 1,
            gpuCacheLimitMB: nil, resumeStep: nil, metricsTrimmedAtStep: nil,
            initModelPath: "champions/model-best-val.bin.gz", initModelSha256: "abc123def456"
        )
        let parsedWith = try TrainCommandTests.readJSON(at: withInit)
        XCTAssertEqual(parsedWith["init_model"] as? String, "champions/model-best-val.bin.gz")
        XCTAssertEqual(parsedWith["init_model_sha256"] as? String, "abc123def456")

        let withoutInit = directory.appendingPathComponent("without-init.json")
        try TrainCommand.writeConfig(
            at: withoutInit, arch: "b6c96", dataPath: "shards/", validationDataPath: nil,
            samples: 100, validationSamples: nil,
            batchSize: 256, steps: 50000, learningRate: 1e-3, warmupSteps: 1000,
            weightDecay: 0.01, maximumGradientNorm: 5,
            lossWeights: LossWeights(policy: 1, value: 3, score: 0.02, ownership: 0.5),
            checkpointInterval: 1000, snapshotInterval: 5000, metricsInterval: 1,
            gpuCacheLimitMB: nil, resumeStep: nil, metricsTrimmedAtStep: nil
        )
        let parsedWithout = try TrainCommandTests.readJSON(at: withoutInit)
        XCTAssertNil(parsedWithout["init_model"])
        XCTAssertNil(parsedWithout["init_model_sha256"])
    }

    // MARK: - helpers

    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("train-cmd-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
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
