import Foundation
import MLX
import RinGoModel
import RinGoTrain

/// Parsed `ringo evaluate` flags. Split out from `main` so flag defaults/validation are
/// testable without touching MLX (see `EvaluateCommandTests`).
struct EvaluateOptions {
    var model = ""
    var data = ""
    var batchSize = 256
    var outputPath = "val_metrics.csv"
    // Defaults deliberately mirror `TrainCommand`'s `-value-weight`/`-ownership-weight`/
    // `-score-weight` defaults, NOT `LossWeights()`'s bare struct defaults (1.5/0.06). Audit
    // finding I-1 (CODEBASE_REVIEW_0711.md): `evaluate` used to build `LossComputer()` with no
    // weights argument at all, so its `total_loss` used 1.5/0.06 while every real training run
    // uses 3/0.5 (TrainCommand's `-value-weight`/`-ownership-weight` defaults) — the two
    // `total_loss` curves were silently NOT comparable despite this file's docstring claiming
    // they could be overlaid. Referencing `TrainCommand`'s named constants (not copying the
    // literals) keeps the two subcommands from drifting apart again.
    var valueWeight: Float = TrainCommand.defaultValueWeight
    var ownershipWeight: Float = TrainCommand.defaultOwnershipWeight
    var scoreWeight: Float = TrainCommand.defaultScoreWeight
}

/// `ringo evaluate`: pinned-load validation-loss subcommand (P0-4.2).
///
/// Loads a trained `.bin.gz` exactly like `rawnn` does (via `KataGoNetwork.forward` on
/// `ModelRawOutputs`), runs it in batches over a validation shard, and writes a single row of
/// per-component + total loss to a CSV using the SAME loss formulas as training (`LossComputer`
/// is shared code, not a reimplementation). Pure forward pass — no grad, no optimizer update —
/// so this matches what `Trainer.evaluate(validation:)` produces during a training run and the
/// two loss curves can be overlaid -- PROVIDED the loss weights match too (see `EvaluateOptions`
/// above): pass `-value-weight`/`-ownership-weight`/`-score-weight` matching whatever `train` run
/// produced the model, or rely on the defaults which already match `train`'s defaults.
enum EvaluateCommand {
    static let usage = "ringo evaluate -model <bin.gz> -data <dir|shard.nngd> "
        + "[-batch 256] [-out val_metrics.csv] [-value-weight 3] [-ownership-weight 0.5] "
        + "[-score-weight 0.02]"

    static func parse(_ arguments: [String]) throws -> EvaluateOptions {
        var options = EvaluateOptions()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else { throw CLIError(description: "Missing value for \(flag)") }
            let value = arguments[index + 1]
            switch flag {
            case "-model": options.model = value
            case "-data": options.data = value
            case "-batch":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -batch (must be a positive int)")
                }
                options.batchSize = parsed
            case "-out": options.outputPath = value
            case "-value-weight":
                guard let parsed = Float(value), parsed >= 0, parsed.isFinite else {
                    throw CLIError(description: "Invalid -value-weight")
                }
                options.valueWeight = parsed
            case "-ownership-weight":
                guard let parsed = Float(value), parsed >= 0, parsed.isFinite else {
                    throw CLIError(description: "Invalid -ownership-weight")
                }
                options.ownershipWeight = parsed
            case "-score-weight":
                guard let parsed = Float(value), parsed >= 0, parsed.isFinite else {
                    throw CLIError(description: "Invalid -score-weight")
                }
                options.scoreWeight = parsed
            default: throw CLIError(description: "Unknown evaluate option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard !options.model.isEmpty, !options.data.isEmpty else { throw CLIError(description: usage) }
        return options
    }

    static func main(_ arguments: [String]) throws {
        let options = try parse(arguments)
        let url = URL(fileURLWithPath: options.data)
        let isDirectory = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        let dataset = try isDirectory
            ? RinGoDataset.load(directory: url)
            : RinGoShard.read(url: url)
        guard !dataset.samples.isEmpty else { throw CLIError(description: "Validation shard is empty") }

        let desc = try ModelDesc.loadFromFileMaybeGZipped(options.model)
        // fp32 match with training-time loss (training is fp32-only per `design.md` Island 2 #5);
        // the loss formula must be apples-to-apples with the values logged in `metrics.csv`.
        let network = try KataGoNetwork(desc: desc, nnXLen: dataset.nnLen, nnYLen: dataset.nnLen, precision: .float32)
        let lossComputer = LossComputer(weights: LossWeights(
            policy: 1, value: options.valueWeight, score: options.scoreWeight, ownership: options.ownershipWeight
        ))

        var policySum: Float = 0
        var valueSum: Float = 0
        var scoreSum: Float = 0
        var ownershipSum: Float = 0
        var totalSum: Float = 0
        var totalSamples = 0
        var stepStart = 0
        while stepStart < dataset.samples.count {
            let end = min(stepStart + options.batchSize, dataset.samples.count)
            let slice = Array(dataset.samples[stepStart ..< end])
            let batch = Trainer.batch(samples: slice, nnLen: dataset.nnLen)
            let outputs = network.forward(spatial: batch.spatial, global: batch.global)
            let losses = lossComputer(outputs: outputs, batch: batch)
            eval(losses.policy, losses.value, losses.score, losses.ownership, losses.total)
            let count = Float(slice.count)
            policySum += losses.policy.item(Float.self) * count
            valueSum += losses.value.item(Float.self) * count
            scoreSum += losses.score.item(Float.self) * count
            ownershipSum += losses.ownership.item(Float.self) * count
            totalSum += losses.total.item(Float.self) * count
            totalSamples += slice.count
            stepStart = end
        }
        let n = Float(max(1, totalSamples))
        let result = ValidationMetrics(
            step: 0,
            policyLoss: policySum / n,
            valueLoss: valueSum / n,
            scoreLoss: scoreSum / n,
            ownershipLoss: ownershipSum / n,
            totalLoss: totalSum / n,
            sampleCount: totalSamples
        )
        try Self.writeCSV(result: result, to: URL(fileURLWithPath: options.outputPath))
        print(String(format: "evaluate: samples=%d total=%.5f pol=%.5f val=%.5f score=%.5f own=%.5f",
                     totalSamples, result.totalLoss, result.policyLoss, result.valueLoss,
                     result.scoreLoss, result.ownershipLoss))
    }

    static func writeCSV(result: ValidationMetrics, to url: URL) throws {
        let header = "policy_loss,value_loss,score_loss,ownership_loss,total_loss,sample_count\n"
        let line = "\(result.policyLoss),\(result.valueLoss),\(result.scoreLoss),"
            + "\(result.ownershipLoss),\(result.totalLoss),\(result.sampleCount)\n"
        try Data((header + line).utf8).write(to: url, options: .atomic)
    }
}
