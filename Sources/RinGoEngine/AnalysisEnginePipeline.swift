import Foundation
import RinGoCore

public enum AnalysisDataError: Error, CustomStringConvertible, Equatable {
    case invalidArgument(String)
    case invalidManifest(String)

    public var description: String {
        switch self {
        case let .invalidArgument(message), let .invalidManifest(message): message
        }
    }
}

// MARK: - Query + manifest schema

/// One analysis-engine query (see `docs/Analysis_Engine.md`). One query per game; `analyzeTurns`
/// batches every position of that game into the single query -- the doc-endorsed efficient form
/// ("Each turn generates a separate response").
public struct AnalysisQuery: Codable, Sendable {
    public let id: String
    public let moves: [[String]]
    public let initialStones: [[String]]?
    public let rules: String
    public let komi: Double
    public let boardXSize: Int
    public let boardYSize: Int
    public let analyzeTurns: [Int]
    public let maxVisits: Int
    public let includeOwnership: Bool
    public let includePolicy: Bool
}

/// Manifest row binding a query `id` back to its source. `moveCount` is authoritative: the query's
/// `analyzeTurns` is exactly `0..<moveCount`, so a response `turnNumber` is valid iff it is in
/// `0..<moveCount`, and ingest replays `turnNumber` SGF moves to rebuild that position.
public struct AnalysisManifestEntry: Codable, Sendable {
    public let id: String
    /// Path to the SGF relative to the export `-sgf-dir` (ingest rejoins it with its own `-sgf-dir`).
    public let sgf: String
    public let komi: Double
    public let rules: String
    public let boardXSize: Int
    public let boardYSize: Int
    public let moveCount: Int
}

/// Sidecar `manifest.json`: the global query settings plus one entry per game.
public struct AnalysisManifest: Codable, Sendable {
    public let version: Int
    public let rules: String
    public let maxVisits: Int
    public let includeOwnership: Bool
    public let includePolicy: Bool
    public let queries: [AnalysisManifestEntry]

    public static let currentVersion = 1
}

// MARK: - Export

public struct AnalysisExportOptions: Sendable {
    public var sgfDirectory: URL
    public var queriesOutput: URL
    public var manifestOutput: URL
    public var nnLen: Int
    public var rulesName: String
    public var komiOverride: Float?
    public var maxGames: Int
    public var maxVisits: Int
    public var includeOwnership: Bool
    public var includePolicy: Bool
    public var minimumMoves: Int

    public init(
        sgfDirectory: URL,
        queriesOutput: URL,
        manifestOutput: URL,
        nnLen: Int = 9,
        rulesName: String = "chinese",
        komiOverride: Float? = nil,
        maxGames: Int = 0,
        maxVisits: Int = 1000,
        includeOwnership: Bool = true,
        includePolicy: Bool = true,
        minimumMoves: Int = 8
    ) {
        self.sgfDirectory = sgfDirectory
        self.queriesOutput = queriesOutput
        self.manifestOutput = manifestOutput
        self.nnLen = nnLen
        self.rulesName = rulesName
        self.komiOverride = komiOverride
        self.maxGames = maxGames
        self.maxVisits = maxVisits
        self.includeOwnership = includeOwnership
        self.includePolicy = includePolicy
        self.minimumMoves = minimumMoves
    }
}

public struct AnalysisExportSummary: Equatable, Sendable {
    public var filesFound = 0
    public var gamesExported = 0
    public var parseSkips = 0
    public var boardSizeSkips = 0
    public var minimumMoveSkips = 0
    public var illegalMoveSkips = 0
    public var totalPositions = 0

    public var gamesSkipped: Int {
        parseSkips + boardSizeSkips + minimumMoveSkips + illegalMoveSkips
    }
}

public enum AnalysisExporter {
    /// Walks the SGF corpus (same recursive `.sgf` walk as `makedata`), applies the STRUCTURAL
    /// filters -- parse, board size == nnLen, `>= minimumMoves`, legal replay -- and emits one query
    /// per accepted game plus the manifest. The result-based skips `makedata` applies (jigo /
    /// unparseable RE) are intentionally OMITTED: analysis labeling does not use the game outcome, so
    /// jigos and resignations are perfectly good positions to relabel (this is the same reasoning that
    /// lets teacher distillation mark every sample valid).
    public static func run(_ options: AnalysisExportOptions) throws -> AnalysisExportSummary {
        guard options.nnLen > 1, options.nnLen <= Board.maxLen else {
            throw AnalysisDataError.invalidArgument("nnLen must be in 2...\(Board.maxLen)")
        }
        guard options.minimumMoves >= 0 else {
            throw AnalysisDataError.invalidArgument("minimumMoves cannot be negative")
        }
        guard options.maxVisits > 0 else {
            throw AnalysisDataError.invalidArgument("maxVisits must be positive")
        }
        let fileManager = FileManager.default
        let files = try sgfFiles(in: options.sgfDirectory, fileManager: fileManager)

        var summary = AnalysisExportSummary()
        summary.filesFound = files.count
        var entries = [AnalysisManifestEntry]()
        var queryLines = Data()

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        for file in files {
            if options.maxGames > 0, summary.gamesExported >= options.maxGames { break }
            let defaultRules = try resolveRules(options.rulesName, komi: options.komiOverride ?? 7.5)
            let game: SGFGame
            do {
                game = try SGFReader.load(file, defaultRules: defaultRules)
            } catch {
                summary.parseSkips += 1
                continue
            }
            guard game.boardXSize == options.nnLen, game.boardYSize == options.nnLen else {
                summary.boardSizeSkips += 1
                continue
            }
            guard game.moves.count >= options.minimumMoves else {
                summary.minimumMoveSkips += 1
                continue
            }
            let komi = options.komiOverride ?? game.komi
            // Legal-replay validity, matching makedata's koRule=positional tolerant replay.
            do {
                _ = try game.position(at: game.moves.count, rules: resolveRules(options.rulesName, komi: komi))
            } catch {
                summary.illegalMoveSkips += 1
                continue
            }

            let id = String(format: "q%06d", summary.gamesExported)
            let moves = game.moves.map { [
                playerString($0.pla),
                Location.toString($0.loc, game.boardXSize, game.boardYSize),
            ] }
            let initialStones = game.placements.isEmpty
                ? nil
                : game.placements.map { [
                    playerString($0.pla),
                    Location.toString($0.loc, game.boardXSize, game.boardYSize),
                ] }
            let query = AnalysisQuery(
                id: id,
                moves: moves,
                initialStones: initialStones,
                rules: options.rulesName,
                komi: Double(komi),
                boardXSize: game.boardXSize,
                boardYSize: game.boardYSize,
                analyzeTurns: Array(0 ..< game.moves.count),
                maxVisits: options.maxVisits,
                includeOwnership: options.includeOwnership,
                includePolicy: options.includePolicy
            )
            try queryLines.append(encoder.encode(query))
            queryLines.append(0x0A) // newline
            entries.append(AnalysisManifestEntry(
                id: id,
                sgf: relativePath(of: file, under: options.sgfDirectory),
                komi: Double(komi),
                rules: options.rulesName,
                boardXSize: game.boardXSize,
                boardYSize: game.boardYSize,
                moveCount: game.moves.count
            ))
            summary.gamesExported += 1
            summary.totalPositions += game.moves.count
        }

        try queryLines.write(to: options.queriesOutput, options: .atomic)
        let manifest = AnalysisManifest(
            version: AnalysisManifest.currentVersion,
            rules: options.rulesName,
            maxVisits: options.maxVisits,
            includeOwnership: options.includeOwnership,
            includePolicy: options.includePolicy,
            queries: entries
        )
        let manifestEncoder = JSONEncoder()
        manifestEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try manifestEncoder.encode(manifest).write(to: options.manifestOutput, options: .atomic)
        return summary
    }
}

// MARK: - Ingest

public struct AnalysisIngestOptions: Sendable {
    public var responsesFile: URL
    public var manifestFile: URL
    public var sgfDirectory: URL
    public var outputDirectory: URL
    public var nnLen: Int
    public var shardSize: Int
    public var perspective: AnalysisWinratePerspective
    public var policyKind: TeacherLabeler.PolicyTargetKind

    public init(
        responsesFile: URL,
        manifestFile: URL,
        sgfDirectory: URL,
        outputDirectory: URL,
        nnLen: Int = 9,
        shardSize: Int = 4096,
        perspective: AnalysisWinratePerspective,
        policyKind: TeacherLabeler.PolicyTargetKind = .visits
    ) {
        self.responsesFile = responsesFile
        self.manifestFile = manifestFile
        self.sgfDirectory = sgfDirectory
        self.outputDirectory = outputDirectory
        self.nnLen = nnLen
        self.shardSize = shardSize
        self.perspective = perspective
        self.policyKind = policyKind
    }
}

public struct AnalysisIngestSummary: Equatable, Sendable {
    public var responsesSeen = 0
    public var interimSkips = 0
    public var errorSkips = 0
    public var unmatchedIdSkips = 0
    public var turnRangeSkips = 0
    public var duplicateSkips = 0
    public var emptyPositionSkips = 0
    public var currentPlayerMismatchSkips = 0
    public var positionsLabeled = 0
    public var totalShards = 0
    public var elapsedSeconds = 0.0

    public var responsesSkipped: Int {
        interimSkips + errorSkips + unmatchedIdSkips + turnRangeSkips + duplicateSkips
            + emptyPositionSkips + currentPlayerMismatchSkips
    }
}

public enum AnalysisIngestor {
    /// Reads the analysis engine's JSON-lines responses, matches each to the manifest + original SGF
    /// by `id`+`turnNumber` (order-independent), rebuilds V7 features locally, converts to v2 targets
    /// via `AnalysisLabeler`, and writes standard v2 shards + a provenance JSON. Every failure mode
    /// (interim report, error line, unknown id, out-of-range turn, duplicate, empty position, side-to
    /// -move mismatch) is skipped and counted rather than fatal.
    public static func run(_ options: AnalysisIngestOptions) throws -> AnalysisIngestSummary {
        guard options.nnLen > 1, options.nnLen <= Board.maxLen else {
            throw AnalysisDataError.invalidArgument("nnLen must be in 2...\(Board.maxLen)")
        }
        guard options.shardSize > 0 else {
            throw AnalysisDataError.invalidArgument("shardSize must be positive")
        }
        let start = Date()
        let fileManager = FileManager.default

        let manifestData = try Data(contentsOf: options.manifestFile)
        let manifest: AnalysisManifest
        do {
            manifest = try JSONDecoder().decode(AnalysisManifest.self, from: manifestData)
        } catch {
            throw AnalysisDataError.invalidManifest("Could not decode manifest: \(error)")
        }
        var entriesByID = [String: AnalysisManifestEntry]()
        for entry in manifest.queries {
            entriesByID[entry.id] = entry
        }

        try fileManager.createDirectory(at: options.outputDirectory, withIntermediateDirectories: true)
        try removeOldShards(in: options.outputDirectory, fileManager: fileManager)

        var summary = AnalysisIngestSummary()
        var shard = [TrainingSample]()
        shard.reserveCapacity(options.shardSize)
        var shardIndex = 0
        var gameCache = [String: SGFGame]()
        var labeled = Set<String>() // "id#turn" keys already emitted (dedup)

        func flush() throws {
            let url = options.outputDirectory.appendingPathComponent(String(format: "shard-%05d.nngd", shardIndex))
            try TrainingShardWriter.write(samples: shard, nnLen: options.nnLen, to: url)
            summary.totalShards += 1
            shardIndex += 1
            shard.removeAll(keepingCapacity: true)
        }

        let responsesText = try String(contentsOf: options.responsesFile, encoding: .utf8)
        let decoder = JSONDecoder()
        for rawLine in responsesText.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            summary.responsesSeen += 1
            guard let lineData = line.data(using: .utf8),
                  let response = try? decoder.decode(AnalysisResponse.self, from: lineData)
            else {
                summary.errorSkips += 1
                continue
            }
            if response.error != nil || response.warning != nil, response.rootInfo == nil {
                summary.errorSkips += 1
                continue
            }
            if response.isDuringSearch == true {
                summary.interimSkips += 1
                continue
            }
            guard let id = response.id, let turn = response.turnNumber,
                  response.rootInfo != nil, response.moveInfos != nil
            else {
                summary.errorSkips += 1
                continue
            }
            guard let entry = entriesByID[id] else {
                summary.unmatchedIdSkips += 1
                continue
            }
            guard turn >= 0, turn < entry.moveCount else {
                summary.turnRangeSkips += 1
                continue
            }
            let key = "\(id)#\(turn)"
            if labeled.contains(key) {
                summary.duplicateSkips += 1
                continue
            }

            let game: SGFGame
            if let cached = gameCache[id] {
                game = cached
            } else {
                let sgfURL = options.sgfDirectory.appendingPathComponent(entry.sgf)
                let rules = try resolveRules(entry.rules, komi: Float(entry.komi))
                game = try SGFReader.load(sgfURL, defaultRules: rules)
                gameCache[id] = game
            }
            guard turn < game.moves.count else {
                summary.turnRangeSkips += 1
                continue
            }
            let nextPlayer = game.moves[turn].pla

            // Turn-alignment cross-check: the engine's own side-to-move must match our SGF replay.
            let reportedPlayer = response.rootInfo?.currentPlayer.flatMap(playerFromString)
            if let reportedPlayer, reportedPlayer != nextPlayer {
                summary.currentPlayerMismatchSkips += 1
                continue
            }

            let rules = try resolveRules(entry.rules, komi: Float(entry.komi))
            let position = try game.position(at: turn, rules: rules)
            guard let targets = AnalysisLabeler.targets(
                response: response,
                nextPlayer: nextPlayer,
                boardXSize: game.boardXSize,
                boardYSize: game.boardYSize,
                nnLen: options.nnLen,
                perspective: options.perspective,
                policyKind: options.policyKind
            ) else {
                summary.emptyPositionSkips += 1
                continue
            }

            let features = NNInputs.fillRowV7(
                position.board, position.history, nextPlayer,
                MiscNNInputParams(), nnXLen: options.nnLen, nnYLen: options.nnLen, useNHWC: true
            )
            let sample = TrainingSample(
                spatial: features.spatial,
                global: features.global,
                policyTarget: targets.policy,
                valueTarget: targets.value,
                scoreTarget: targets.score,
                // Analysis scoreLead is always a genuine estimate; ownership is valid only when a
                // correctly-sized ownership array was actually returned (matching what the labeler
                // could apply).
                scoreValid: true,
                ownershipTarget: targets.ownership,
                ownershipValid: response.ownership?.count == game.boardYSize * game.boardXSize
            )
            shard.append(sample)
            labeled.insert(key)
            summary.positionsLabeled += 1
            if shard.count == options.shardSize { try flush() }
        }
        if !shard.isEmpty { try flush() }

        summary.elapsedSeconds = Date().timeIntervalSince(start)
        try writeProvenance(options: options, manifest: manifest, summary: summary)
        return summary
    }

    private static func writeProvenance(
        options: AnalysisIngestOptions,
        manifest: AnalysisManifest,
        summary: AnalysisIngestSummary
    ) throws {
        let dictionary: [String: Any] = [
            "distilled": true,
            "label_source": "katago-analysis-engine",
            "analysis_max_visits": manifest.maxVisits,
            "winrates_as": options.perspective.rawValue,
            "policy_target": options.policyKind.rawValue,
            "responses_file": options.responsesFile.lastPathComponent,
            "manifest_file": options.manifestFile.lastPathComponent,
            "responses_seen": summary.responsesSeen,
            "responses_skipped": summary.responsesSkipped,
            "positions_labeled": summary.positionsLabeled,
            "total_shards": summary.totalShards,
            "created_at": ISO8601DateFormatter().string(from: Date()),
        ]
        let data = try JSONSerialization.data(withJSONObject: dictionary, options: [.prettyPrinted, .sortedKeys])
        try data.write(
            to: options.outputDirectory.appendingPathComponent("analysis-ingest-config.json"),
            options: .atomic
        )
    }
}

// MARK: - Shared helpers

private func resolveRules(_ name: String, komi: Float) throws -> Rules {
    var rules: Rules
    do {
        rules = try Rules.parseRulesWithoutKomi(name, komi: komi)
    } catch {
        throw AnalysisDataError.invalidArgument("Invalid rules '\(name)': \(error)")
    }
    // Match makedata's tolerant replay: positional superko keeps real SGFs replayable and makes the
    // feature-building position byte-identical to the makedata path.
    rules.koRule = .positional
    rules.komi = komi
    return rules
}

private func playerString(_ player: Player) -> String {
    player == .black ? "B" : "W"
}

private func playerFromString(_ value: String) -> Player? {
    switch value.uppercased() {
    case "B", "BLACK": .black
    case "W", "WHITE": .white
    default: nil
    }
}

private func relativePath(of file: URL, under directory: URL) -> String {
    let filePath = file.standardizedFileURL.path
    let dirPath = directory.standardizedFileURL.path
    if filePath.hasPrefix(dirPath + "/") {
        return String(filePath.dropFirst(dirPath.count + 1))
    }
    return file.lastPathComponent
}

private func sgfFiles(in directory: URL, fileManager: FileManager) throws -> [URL] {
    guard let enumerator = fileManager.enumerator(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    ) else { throw AnalysisDataError.invalidArgument("Cannot enumerate \(directory.path)") }
    var files = [URL]()
    for case let file as URL in enumerator where file.pathExtension.lowercased() == "sgf" {
        // Resolve symlinks before the regular-file test: a symlink's OWN isRegularFile is false, so
        // a directory of symlinks (`rankgames -link-dir`, the WP-19/D2 active-learning selection)
        // would otherwise be silently skipped -- yet that directory is meant to feed straight into
        // `analysisexport -sgf-dir`. Keep the original (link) URL in the list so relative paths stay
        // anchored to `directory`; SGF loading follows the link transparently.
        let resolved = file.resolvingSymlinksInPath()
        if try resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            files.append(file)
        }
    }
    return files.sorted { $0.path < $1.path }
}

private func removeOldShards(in directory: URL, fileManager: FileManager) throws {
    let files = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
    for file in files where file.lastPathComponent.hasPrefix("shard-") && file.pathExtension == "nngd" {
        try fileManager.removeItem(at: file)
    }
}
