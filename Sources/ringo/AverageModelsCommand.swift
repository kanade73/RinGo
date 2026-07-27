import Foundation
import RinGoModel

/// Parsed `ringo averagemodels` flags. Split from `main` so parsing/validation are testable
/// without touching MLX or the filesystem (see `AverageModelsCommandTests`).
struct AverageModelsOptions {
    var models = [String]()
    var output = ""
}

/// `ringo averagemodels`: SWA (stochastic weight averaging, BREAKTHROUGH_IDEAS E4).
///
/// Loads two-or-more same-architecture RinGo `.bin.gz` snapshots, averages every weight tensor
/// (fp32 accumulation, `ModelAverager`), and writes the mean as a standard `.bin.gz` via
/// `ModelWriter` — the same serialization/canonicalization path `train` uses to export, so the
/// output loads identically under `gtp`/`rawnn`. It only produces the averaged net; strength is
/// left to a separate A/B match.
enum AverageModelsCommand {
    static let usage = "ringo averagemodels -models <a.bin.gz,b.bin.gz[,...]> -out <avg.bin.gz>"

    static func parse(_ arguments: [String]) throws -> AverageModelsOptions {
        var options = AverageModelsOptions()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else { throw CLIError(description: "Missing value for \(flag)") }
            let value = arguments[index + 1]
            switch flag {
            case "-models":
                options.models = value
                    .split(separator: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
            case "-out": options.output = value
            default: throw CLIError(description: "Unknown averagemodels option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard options.models.count >= 2 else {
            throw CLIError(description: "averagemodels needs at least two comma-separated -models inputs\n\(usage)")
        }
        guard !options.output.isEmpty else { throw CLIError(description: usage) }
        return options
    }

    static func main(_ arguments: [String]) throws {
        let options = try parse(arguments)
        var descriptors = [ModelDesc]()
        descriptors.reserveCapacity(options.models.count)
        for path in options.models {
            try descriptors.append(ModelDesc.loadFromFileMaybeGZipped(path))
        }
        let averaged = try ModelAverager.average(descriptors)
        try ModelWriter.write(desc: averaged, to: URL(fileURLWithPath: options.output))
        print(
            "averagemodels: wrote \(options.output) — mean of \(descriptors.count) models (\(averaged.shortInfoString))"
        )
    }
}
