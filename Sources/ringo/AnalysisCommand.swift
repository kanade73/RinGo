import Foundation
import RinGoEngine

/// `ringo analysisexport` -- turn a local SGF corpus into KataGo analysis-engine queries.
enum AnalysisExportCommand {
    static let usage = """
    Usage: ringo analysisexport -sgf-dir <dir> [-out queries.jsonl] [-manifest manifest.json] \
      [-nnlen 9] [-rules chinese] [-komi <override>] [-max-games N] [-max-visits 1000] \
      [-include-ownership true|false] [-include-policy true|false] [-min-moves 8]
    """

    static func main(_ arguments: [String]) throws {
        var sgfDirectory: String?
        var queriesOut = "queries.jsonl"
        var manifestOut = "manifest.json"
        var nnLen = 9
        var rulesName = "chinese"
        var komiOverride: Float?
        var maxGames = 0
        var maxVisits = 1000
        var includeOwnership = true
        var includePolicy = true
        var minimumMoves = 8

        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else { throw CLIError(description: "Missing value for \(flag)") }
            let value = arguments[index + 1]
            switch flag {
            case "-sgf-dir": sgfDirectory = value
            case "-out": queriesOut = value
            case "-manifest": manifestOut = value
            case "-nnlen": nnLen = try integer(value, named: flag)
            case "-rules": rulesName = value
            case "-komi":
                guard let parsed = Float(value), parsed.isFinite else {
                    throw CLIError(description: "Invalid -komi (must be a finite number)")
                }
                komiOverride = parsed
            case "-max-games": maxGames = try integer(value, named: flag)
            case "-max-visits": maxVisits = try integer(value, named: flag)
            case "-include-ownership": includeOwnership = try boolean(value, named: flag)
            case "-include-policy": includePolicy = try boolean(value, named: flag)
            case "-min-moves": minimumMoves = try integer(value, named: flag)
            default: throw CLIError(description: "Unknown option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard let sgfDirectory else { throw CLIError(description: usage) }

        let options = AnalysisExportOptions(
            sgfDirectory: URL(fileURLWithPath: sgfDirectory),
            queriesOutput: URL(fileURLWithPath: queriesOut),
            manifestOutput: URL(fileURLWithPath: manifestOut),
            nnLen: nnLen,
            rulesName: rulesName,
            komiOverride: komiOverride,
            maxGames: maxGames,
            maxVisits: maxVisits,
            includeOwnership: includeOwnership,
            includePolicy: includePolicy,
            minimumMoves: minimumMoves
        )
        let summary = try AnalysisExporter.run(options)
        print(
            "analysisexport: games exported=\(summary.gamesExported) positions=\(summary.totalPositions) "
                + "skipped=\(summary.gamesSkipped) (parse=\(summary.parseSkips), size=\(summary.boardSizeSkips), "
                + "minMoves=\(summary.minimumMoveSkips), illegalMove=\(summary.illegalMoveSkips)) "
                + "-> \(queriesOut) + \(manifestOut) (maxVisits=\(maxVisits))"
        )
    }
}

/// `ringo analysisingest` -- convert analysis-engine responses into RinGoData v2 shards.
enum AnalysisIngestCommand {
    static let usage = """
    Usage: ringo analysisingest -responses responses.jsonl -manifest manifest.json \
      -sgf-dir <dir> [-out ringo-analysis] [-nnlen 9] [-shard-size 4096] \
      -winrates-as black|white|sidetomove [-target visits|completed-q]
      (-winrates-as MUST match the analysis engine config's reportAnalysisWinratesAs; the shipped
       analysis_example.cfg default is BLACK.)
    """

    static func main(_ arguments: [String]) throws {
        var responsesFile: String?
        var manifestFile: String?
        var sgfDirectory: String?
        var outputDirectory = "ringo-analysis"
        var nnLen = 9
        var shardSize = 4096
        var perspective: AnalysisWinratePerspective?
        var policyKind: TeacherLabeler.PolicyTargetKind = .visits

        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else { throw CLIError(description: "Missing value for \(flag)") }
            let value = arguments[index + 1]
            switch flag {
            case "-responses": responsesFile = value
            case "-manifest": manifestFile = value
            case "-sgf-dir": sgfDirectory = value
            case "-out": outputDirectory = value
            case "-nnlen": nnLen = try integer(value, named: flag)
            case "-shard-size": shardSize = try integer(value, named: flag)
            case "-winrates-as":
                guard let parsed = AnalysisWinratePerspective(rawValue: value.lowercased()) else {
                    throw CLIError(description: "Invalid -winrates-as (must be black, white, or sidetomove)")
                }
                perspective = parsed
            case "-target":
                guard let parsed = TeacherLabeler.PolicyTargetKind(rawValue: value) else {
                    throw CLIError(description: "Invalid -target (must be visits or completed-q)")
                }
                policyKind = parsed
            default: throw CLIError(description: "Unknown option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard let responsesFile, let manifestFile, let sgfDirectory else {
            throw CLIError(description: usage)
        }
        guard let perspective else {
            throw CLIError(description: "-winrates-as is required (must match the engine's "
                + "reportAnalysisWinratesAs; shipped analysis_example.cfg default is BLACK)\n\(usage)")
        }

        let options = AnalysisIngestOptions(
            responsesFile: URL(fileURLWithPath: responsesFile),
            manifestFile: URL(fileURLWithPath: manifestFile),
            sgfDirectory: URL(fileURLWithPath: sgfDirectory),
            outputDirectory: URL(fileURLWithPath: outputDirectory),
            nnLen: nnLen,
            shardSize: shardSize,
            perspective: perspective,
            policyKind: policyKind
        )
        let summary = try AnalysisIngestor.run(options)
        let rate = summary.elapsedSeconds > 0 ? Double(summary.positionsLabeled) / summary.elapsedSeconds : 0
        print(
            "analysisingest: labeled=\(summary.positionsLabeled) shards=\(summary.totalShards) "
                + "responsesSeen=\(summary.responsesSeen) skipped=\(summary.responsesSkipped) "
                + "(interim=\(summary.interimSkips), error=\(summary.errorSkips), "
                + "unmatchedId=\(summary.unmatchedIdSkips), turnRange=\(summary.turnRangeSkips), "
                + "duplicate=\(summary.duplicateSkips), emptyPos=\(summary.emptyPositionSkips), "
                + "playerMismatch=\(summary.currentPlayerMismatchSkips)) "
                + String(format: "elapsed=%.3fs positions/sec=%.1f", summary.elapsedSeconds, rate)
                + " winrates-as=\(perspective.rawValue) target=\(policyKind.rawValue)"
        )
        if summary.currentPlayerMismatchSkips > 0 {
            FileHandle.standardError.write(Data((
                "analysisingest: WARNING \(summary.currentPlayerMismatchSkips) responses had a "
                    + "side-to-move that disagreed with the SGF replay -- check that -manifest matches "
                    + "-responses and that the SGFs under -sgf-dir are the exported ones.\n"
            ).utf8))
        }
    }
}

private func integer(_ value: String, named option: String) throws -> Int {
    guard let result = Int(value) else { throw CLIError(description: "Invalid integer for \(option)") }
    return result
}

private func boolean(_ value: String, named option: String) throws -> Bool {
    switch value.lowercased() {
    case "true", "1", "yes": return true
    case "false", "0", "no": return false
    default: throw CLIError(description: "Invalid boolean for \(option) (use true/false)")
    }
}
