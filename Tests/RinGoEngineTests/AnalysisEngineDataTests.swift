import Foundation
import RinGoCore
@testable import RinGoEngine
import XCTest

/// WP-14 unit tests: KataGo analysis-engine ingest (`AnalysisLabeler`, `AnalysisExporter`,
/// `AnalysisIngestor`). All CPU-only; the response fixtures are HAND-WRITTEN JSON following the
/// `docs/Analysis_Engine.md` schema (`id`, `turnNumber`, `isDuringSearch`, `moveInfos[].{move,
/// visits,winrate,prior}`, `rootInfo.{winrate,scoreLead,rawWinrate,currentPlayer}`, `ownership`,
/// `policy`). The perspective mappings are pinned against the doc statement "All values will be from
/// the perspective of `reportAnalysisWinratesAs`" (shipped default BLACK).
final class AnalysisEngineDataTests: XCTestCase {
    private func decode(_ json: String) throws -> AnalysisResponse {
        try JSONDecoder().decode(AnalysisResponse.self, from: Data(json.utf8))
    }

    /// A 3x3 response. A3 -> dense pos 0, B3 -> pos 1, pass -> pos 9 (area). Ownership is 9 floats
    /// row-major; policy is 10 entries (9 board + pass) with `-1` on illegal (positions 2...8).
    private let fixture3x3 = """
    {
      "id": "q0",
      "turnNumber": 0,
      "rootInfo": { "winrate": 0.70, "scoreLead": 4.0, "rawWinrate": 0.65, "currentPlayer": "B", "visits": 100 },
      "moveInfos": [
        { "move": "A3", "visits": 60, "winrate": 0.72, "prior": 0.5 },
        { "move": "B3", "visits": 30, "winrate": 0.68, "prior": 0.3 },
        { "move": "pass", "visits": 10, "winrate": 0.50, "prior": 0.2 }
      ],
      "ownership": [0.9, -0.9, 0.1, -0.4, 0.6, -0.6, 0.49, -0.49, 0.0],
      "policy": [0.6, 0.3, -1, -1, -1, -1, -1, -1, -1, 0.1]
    }
    """

    // MARK: - Policy distribution

    func testVisitsPolicyIsDenseDistributionWithPass() throws {
        let response = try decode(fixture3x3)
        let targets = try XCTUnwrap(AnalysisLabeler.targets(
            response: response, nextPlayer: .black, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .black
        ))
        XCTAssertEqual(targets.policy.count, 10)
        XCTAssertEqual(targets.policy.reduce(0, +), 1.0, accuracy: 1e-6)
        XCTAssertEqual(targets.policy[0], 0.6, accuracy: 1e-6) // A3: 60/100
        XCTAssertEqual(targets.policy[1], 0.3, accuracy: 1e-6) // B3: 30/100
        XCTAssertEqual(targets.policy[9], 0.1, accuracy: 1e-6) // pass: 10/100
        XCTAssertTrue(targets.policy.allSatisfy { $0 >= 0 })
        // Every other (unsearched) point carries no mass.
        for pos in 2 ... 8 {
            XCTAssertEqual(targets.policy[pos], 0)
        }
    }

    // MARK: - Perspective (both colors, all three reportAnalysisWinratesAs settings)

    /// With `reportAnalysisWinratesAs = BLACK`, a BLACK-to-move position needs no flip; a WHITE-to
    /// -move position flips value (win<->loss), negates score, and negates ownership.
    func testPerspectiveBlackConfigFlipsOnlyForWhiteToMove() throws {
        let response = try decode(fixture3x3)
        let black = try XCTUnwrap(AnalysisLabeler.targets(
            response: response, nextPlayer: .black, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .black
        ))
        XCTAssertEqual(black.value, [0.7, 0.3, 0])
        XCTAssertEqual(black.score, 4.0, accuracy: 1e-6)
        XCTAssertEqual(black.ownership, [1, -1, 0, 0, 1, -1, 0, 0, 0])

        let white = try XCTUnwrap(AnalysisLabeler.targets(
            response: response, nextPlayer: .white, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .black
        ))
        XCTAssertEqual(white.value[0], 1 - 0.7, accuracy: 1e-6)
        XCTAssertEqual(white.value[1], 0.7, accuracy: 1e-6)
        XCTAssertEqual(white.score, -4.0, accuracy: 1e-6)
        XCTAssertEqual(white.ownership, [-1, 1, 0, 0, -1, 1, 0, 0, 0])
        // Policy is perspective-independent.
        XCTAssertEqual(black.policy, white.policy)
    }

    /// With `reportAnalysisWinratesAs = WHITE`, the mirror image: WHITE-to-move needs no flip,
    /// BLACK-to-move flips.
    func testPerspectiveWhiteConfigFlipsOnlyForBlackToMove() throws {
        let response = try decode(fixture3x3)
        let white = try XCTUnwrap(AnalysisLabeler.targets(
            response: response, nextPlayer: .white, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .white
        ))
        XCTAssertEqual(white.value, [0.7, 0.3, 0])
        XCTAssertEqual(white.score, 4.0, accuracy: 1e-6)
        XCTAssertEqual(white.ownership, [1, -1, 0, 0, 1, -1, 0, 0, 0])

        let black = try XCTUnwrap(AnalysisLabeler.targets(
            response: response, nextPlayer: .black, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .white
        ))
        XCTAssertEqual(black.value[0], 0.3, accuracy: 1e-6)
        XCTAssertEqual(black.score, -4.0, accuracy: 1e-6)
        XCTAssertEqual(black.ownership, [-1, 1, 0, 0, -1, 1, 0, 0, 0])
    }

    /// With `reportAnalysisWinratesAs = SIDETOMOVE`, the reported perspective IS the side to move, so
    /// nothing ever flips -- both colors keep the reported winrate/score/ownership verbatim.
    func testPerspectiveSideToMoveNeverFlips() throws {
        let response = try decode(fixture3x3)
        for player in [Player.black, .white] {
            let targets = try XCTUnwrap(AnalysisLabeler.targets(
                response: response, nextPlayer: player, boardXSize: 3, boardYSize: 3, nnLen: 3,
                perspective: .sidetomove
            ))
            XCTAssertEqual(targets.value, [0.7, 0.3, 0], "sidetomove must not flip for \(player)")
            XCTAssertEqual(targets.score, 4.0, accuracy: 1e-6)
            XCTAssertEqual(targets.ownership, [1, -1, 0, 0, 1, -1, 0, 0, 0])
        }
    }

    // MARK: - Ownership quantization

    /// Continuous [-1,1] ownership quantizes to the v2 `{-1,0,+1}` Int8 alphabet by nearest rounding,
    /// exactly as `TeacherLabeler` does (so `LossComputer`'s `(v+1)/2` BCE decode stays valid).
    func testOwnershipQuantizesToTernaryAlphabet() throws {
        let response = try decode(fixture3x3)
        let targets = try XCTUnwrap(AnalysisLabeler.targets(
            response: response, nextPlayer: .black, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .black
        ))
        XCTAssertTrue(targets.ownership.allSatisfy { $0 == -1 || $0 == 0 || $0 == 1 })
        // 0.49 rounds to 0; 0.6 rounds to 1; -0.4 rounds to 0 -- pinning the nearest-rounding rule.
        XCTAssertEqual(targets.ownership[6], 0) // 0.49
        XCTAssertEqual(targets.ownership[4], 1) // 0.6
        XCTAssertEqual(targets.ownership[3], 0) // -0.4
    }

    /// When the response omits `ownership` (query without `includeOwnership`), the target is all-zero
    /// and the ingest marks `ownershipValid=false` (checked in the pipeline test below).
    func testMissingOwnershipYieldsZeroTarget() throws {
        let json = """
        { "id": "q0", "turnNumber": 0,
          "rootInfo": { "winrate": 0.6, "scoreLead": 1.0, "currentPlayer": "B" },
          "moveInfos": [ { "move": "A3", "visits": 10, "winrate": 0.6 } ] }
        """
        let targets = try XCTUnwrap(try AnalysisLabeler.targets(
            response: decode(json), nextPlayer: .black, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .black
        ))
        XCTAssertTrue(targets.ownership.allSatisfy { $0 == 0 })
    }

    // MARK: - Empty / completed-Q

    /// A response with no `moveInfos` cannot form a policy target and must return nil (skip+count).
    func testEmptyMoveInfosReturnsNil() throws {
        let json = """
        { "id": "q0", "turnNumber": 0,
          "rootInfo": { "winrate": 0.5, "scoreLead": 0.0, "currentPlayer": "B" }, "moveInfos": [] }
        """
        XCTAssertNil(try AnalysisLabeler.targets(
            response: decode(json), nextPlayer: .black, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .black
        ))
    }

    /// The completed-Q policy target (reusing WP-13's `TeacherLabeler.completedQPolicy`) is a valid
    /// dense distribution: legal-masked (illegal `-1` positions stay 0), pass included, non-negative,
    /// finite, summing to 1. Value/score/ownership are unchanged from the visits path.
    func testCompletedQPolicyIsValidDistribution() throws {
        let response = try decode(fixture3x3)
        let targets = try XCTUnwrap(AnalysisLabeler.targets(
            response: response, nextPlayer: .black, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .black, policyKind: .completedQ
        ))
        XCTAssertEqual(targets.policy.reduce(0, +), 1.0, accuracy: 1e-5)
        XCTAssertTrue(targets.policy.allSatisfy { $0 >= 0 && $0.isFinite })
        for pos in 2 ... 8 {
            XCTAssertEqual(targets.policy[pos], 0, "illegal move must stay masked")
        }
        XCTAssertGreaterThan(targets.policy[9], 0, "legal pass must carry mass")
        // Value/score/ownership match the visits path (only the policy construction changes).
        let visits = try XCTUnwrap(AnalysisLabeler.targets(
            response: response, nextPlayer: .black, boardXSize: 3, boardYSize: 3, nnLen: 3,
            perspective: .black, policyKind: .visits
        ))
        XCTAssertEqual(targets.value, visits.value)
        XCTAssertEqual(targets.score, visits.score)
        XCTAssertEqual(targets.ownership, visits.ownership)
    }

    // MARK: - Ingestor: matching, ordering, tolerance

    /// Builds one 9x9 response JSON line. Board moves are GTP strings; ownership (81) and policy (82)
    /// are optional. `interim` marks an `isDuringSearch` report.
    private func responseLine(
        id: String, turn: Int, winrate: Double, scoreLead: Double, currentPlayer: String,
        moves: [(String, Int, Double)], includeOwnership: Bool = true, includePolicy: Bool = true,
        interim: Bool = false
    ) throws -> String {
        var object: [String: Any] = [
            "id": id, "turnNumber": turn,
            "rootInfo": [
                "winrate": winrate, "scoreLead": scoreLead, "rawWinrate": winrate - 0.02,
                "currentPlayer": currentPlayer, "visits": 500,
            ],
            "moveInfos": moves.map { ["move": $0.0, "visits": $0.1, "winrate": $0.2, "prior": 1.0 / 82.0] },
        ]
        if interim { object["isDuringSearch"] = true }
        if includeOwnership { object["ownership"] = (0 ..< 81).map { Double(($0 % 5) - 2) / 2.0 } }
        if includePolicy {
            var policy = [Double](repeating: -1, count: 82)
            policy[0] = 0.5; policy[1] = 0.3; policy[81] = 0.2
            object["policy"] = policy
        }
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        return try XCTUnwrap(String(bytes: data, encoding: .utf8))
    }

    private func writeManifest(to url: URL, entries: [AnalysisManifestEntry]) throws {
        let manifest = AnalysisManifest(
            version: 1, rules: "chinese", maxVisits: 500,
            includeOwnership: true, includePolicy: true, queries: entries
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(manifest).write(to: url)
    }

    /// End-to-end ingest: out-of-order responses match by id+turn; a missing turn is tolerated; and
    /// interim / error / duplicate / unmatched-id lines are each skipped and counted.
    func testIngestMatchesOutOfOrderAndToleratesGapsAndJunk() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("in")
        let out = root.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 5-move game: B E5, W F5, B D6, W C4, B G3. Turns 0..4. nextPlayer(turn t) = moves[t].pla.
        try Data("(;SZ[9]KM[7]RU[Chinese]RE[B+R];B[ee];W[ff];B[df];W[cd];B[gg])".utf8)
            .write(to: input.appendingPathComponent("a.sgf"))
        try writeManifest(to: root.appendingPathComponent("manifest.json"), entries: [
            AnalysisManifestEntry(id: "q0", sgf: "a.sgf", komi: 7.0, rules: "chinese",
                                  boardXSize: 9, boardYSize: 9, moveCount: 5),
        ])

        let mv: [(String, Int, Double)] = [("E5", 300, 0.6), ("F5", 150, 0.55), ("pass", 50, 0.5)]
        let lines = try [
            responseLine(id: "q0", turn: 0, winrate: 0.70, scoreLead: 3.0, currentPlayer: "B",
                         moves: mv, interim: true), // interim -> skip
            responseLine(id: "q0", turn: 4, winrate: 0.80, scoreLead: 5.0, currentPlayer: "B", moves: mv),
            "{\"error\":\"bad query\",\"id\":\"q0\"}", // error -> skip
            responseLine(id: "q0", turn: 0, winrate: 0.70, scoreLead: 3.0, currentPlayer: "B", moves: mv),
            responseLine(id: "q0", turn: 2, winrate: 0.65, scoreLead: 2.0, currentPlayer: "B", moves: mv),
            responseLine(id: "q0", turn: 4, winrate: 0.80, scoreLead: 5.0, currentPlayer: "B", moves: mv), // dup
            responseLine(id: "zzz", turn: 0, winrate: 0.5, scoreLead: 0.0, currentPlayer: "B", moves: mv), // unknown id
        ]
        let responsesURL = root.appendingPathComponent("responses.jsonl")
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: responsesURL)

        let summary = try AnalysisIngestor.run(AnalysisIngestOptions(
            responsesFile: responsesURL, manifestFile: root.appendingPathComponent("manifest.json"),
            sgfDirectory: input, outputDirectory: out, perspective: .black
        ))
        XCTAssertEqual(summary.responsesSeen, 7)
        XCTAssertEqual(summary.positionsLabeled, 3) // turns 0, 2, 4
        XCTAssertEqual(summary.interimSkips, 1)
        XCTAssertEqual(summary.errorSkips, 1)
        XCTAssertEqual(summary.duplicateSkips, 1)
        XCTAssertEqual(summary.unmatchedIdSkips, 1)
        XCTAssertEqual(summary.turnRangeSkips, 0)
        XCTAssertEqual(summary.totalShards, 1)

        let reader = try TrainingShardReader(url: out.appendingPathComponent("shard-00000.nngd"))
        XCTAssertEqual(reader.header.sampleCount, 3)
        // Every emitted sample is a valid v2 sample: dense policy sums to 1, labels valid.
        var winByTurn = Set<Float>()
        for index in 0 ..< 3 {
            let sample = try reader.sample(at: index)
            XCTAssertEqual(sample.policyTarget.reduce(0, +), 1.0, accuracy: 1e-4)
            XCTAssertTrue(sample.scoreValid)
            XCTAssertTrue(sample.ownershipValid)
            winByTurn.insert(sample.valueTarget[0])
        }
        // Turn 0 (black to move, no flip) -> 0.80? No: winrates by turn are 0.70/0.65/0.80 for
        // turns 0/2/4. Turns 0 and 2 are black-to-move (no flip); turn 4 is black-to-move too
        // (moves[4] is B). So side-to-move win == reported winrate for all three.
        XCTAssertTrue(winByTurn.contains(0.70))
        XCTAssertTrue(winByTurn.contains(0.65))
        XCTAssertTrue(winByTurn.contains(0.80))
    }

    /// `ownershipValid` reflects whether the response actually carried an ownership array.
    func testIngestOwnershipValidFollowsResponse() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("in")
        let out = root.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("(;SZ[9]KM[7]RU[Chinese];B[ee];W[ff];B[df])".utf8)
            .write(to: input.appendingPathComponent("a.sgf"))
        try writeManifest(to: root.appendingPathComponent("manifest.json"), entries: [
            AnalysisManifestEntry(id: "q0", sgf: "a.sgf", komi: 7.0, rules: "chinese",
                                  boardXSize: 9, boardYSize: 9, moveCount: 3),
        ])
        let mv: [(String, Int, Double)] = [("E5", 300, 0.6), ("pass", 50, 0.5)]
        let lines = try [
            responseLine(id: "q0", turn: 0, winrate: 0.6, scoreLead: 1.0, currentPlayer: "B",
                         moves: mv, includeOwnership: true),
            responseLine(id: "q0", turn: 1, winrate: 0.4, scoreLead: -1.0, currentPlayer: "W",
                         moves: mv, includeOwnership: false),
        ]
        let responsesURL = root.appendingPathComponent("responses.jsonl")
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: responsesURL)
        let summary = try AnalysisIngestor.run(AnalysisIngestOptions(
            responsesFile: responsesURL, manifestFile: root.appendingPathComponent("manifest.json"),
            sgfDirectory: input, outputDirectory: out, perspective: .sidetomove
        ))
        XCTAssertEqual(summary.positionsLabeled, 2)
        let reader = try TrainingShardReader(url: out.appendingPathComponent("shard-00000.nngd"))
        // Sample order == successful-response order: turn 0 (ownership present), turn 1 (absent).
        XCTAssertTrue(try reader.sample(at: 0).ownershipValid)
        XCTAssertFalse(try reader.sample(at: 1).ownershipValid)
    }

    /// A response whose `rootInfo.currentPlayer` disagrees with our own SGF replay (a turn
    /// misalignment) is skipped and counted, never silently mislabeled.
    func testIngestSkipsCurrentPlayerMismatch() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("in")
        let out = root.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("(;SZ[9]KM[7]RU[Chinese];B[ee];W[ff];B[df])".utf8)
            .write(to: input.appendingPathComponent("a.sgf"))
        try writeManifest(to: root.appendingPathComponent("manifest.json"), entries: [
            AnalysisManifestEntry(id: "q0", sgf: "a.sgf", komi: 7.0, rules: "chinese",
                                  boardXSize: 9, boardYSize: 9, moveCount: 3),
        ])
        // Turn 0 is black-to-move, but claim "W" -> mismatch.
        let mv = [("E5", 300, 0.6)]
        let line = try responseLine(id: "q0", turn: 0, winrate: 0.6, scoreLead: 1.0,
                                    currentPlayer: "W", moves: mv)
        let responsesURL = root.appendingPathComponent("responses.jsonl")
        try Data((line + "\n").utf8).write(to: responsesURL)
        let summary = try AnalysisIngestor.run(AnalysisIngestOptions(
            responsesFile: responsesURL, manifestFile: root.appendingPathComponent("manifest.json"),
            sgfDirectory: input, outputDirectory: out, perspective: .black
        ))
        XCTAssertEqual(summary.positionsLabeled, 0)
        XCTAssertEqual(summary.currentPlayerMismatchSkips, 1)
    }

    // MARK: - Roundtrip on real kifu SGFs

    /// Export queries from 3 real `kifu/` SGFs, craft mock responses for a few of their turns, ingest,
    /// and verify the resulting shard. Skipped if the (read-only) `kifu/` corpus is not present.
    func testRoundtripOnRealKifuSgfs() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let kifu = packageRoot.appendingPathComponent("kifu")
        let names = ["23005.sgf", "43929.sgf", "45380.sgf"]
        try XCTSkipUnless(
            names.allSatisfy { FileManager.default.fileExists(atPath: kifu.appendingPathComponent($0).path) },
            "kifu/ corpus not present"
        )

        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("in")
        let out = root.appendingPathComponent("out")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for name in names {
            try FileManager.default.copyItem(
                at: kifu.appendingPathComponent(name), to: input.appendingPathComponent(name)
            )
        }

        let queriesURL = root.appendingPathComponent("queries.jsonl")
        let manifestURL = root.appendingPathComponent("manifest.json")
        let exportSummary = try AnalysisExporter.run(AnalysisExportOptions(
            sgfDirectory: input, queriesOutput: queriesURL, manifestOutput: manifestURL,
            maxGames: 3, maxVisits: 500
        ))
        XCTAssertEqual(exportSummary.gamesExported, 3)
        XCTAssertGreaterThan(exportSummary.totalPositions, 0)

        // The queries file is valid JSON-lines, one decodable query per game with an analyzeTurns
        // list covering exactly its moves.
        let manifest = try JSONDecoder().decode(AnalysisManifest.self, from: Data(contentsOf: manifestURL))
        XCTAssertEqual(manifest.queries.count, 3)
        let queryLines = try String(contentsOf: queriesURL, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
        XCTAssertEqual(queryLines.count, 3)
        let firstQuery = try JSONDecoder().decode(
            AnalysisQuery.self, from: Data(queryLines[0].utf8)
        )
        XCTAssertEqual(firstQuery.analyzeTurns, Array(0 ..< manifest.queries[0].moveCount))
        XCTAssertEqual(firstQuery.boardXSize, 9)
        XCTAssertFalse(firstQuery.moves.isEmpty)

        // Craft mock responses for turns 0..3 of the first exported game and ingest them.
        let entry = manifest.queries[0]
        let mv: [(String, Int, Double)] = [("E5", 200, 0.6), ("C3", 200, 0.55), ("pass", 100, 0.5)]
        var lines = [String]()
        for turn in 0 ..< min(4, entry.moveCount) {
            let cp = try XCTUnwrap(sideToMovePlayerString(sgf: input.appendingPathComponent(entry.sgf), turn: turn))
            try lines.append(responseLine(
                id: entry.id, turn: turn, winrate: 0.6, scoreLead: 2.0, currentPlayer: cp, moves: mv
            ))
        }
        let responsesURL = root.appendingPathComponent("responses.jsonl")
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: responsesURL)

        let ingestSummary = try AnalysisIngestor.run(AnalysisIngestOptions(
            responsesFile: responsesURL, manifestFile: manifestURL,
            sgfDirectory: input, outputDirectory: out, perspective: .sidetomove
        ))
        XCTAssertEqual(ingestSummary.positionsLabeled, lines.count)
        XCTAssertEqual(ingestSummary.currentPlayerMismatchSkips, 0)

        let reader = try TrainingShardReader(url: out.appendingPathComponent("shard-00000.nngd"))
        XCTAssertEqual(reader.header.sampleCount, lines.count)
        XCTAssertEqual(reader.header.nnLen, 9)
        for index in 0 ..< reader.header.sampleCount {
            let sample = try reader.sample(at: index)
            XCTAssertEqual(sample.spatial.count, 22 * 81)
            XCTAssertEqual(sample.policyTarget.reduce(0, +), 1.0, accuracy: 1e-4)
            XCTAssertEqual(sample.valueTarget[0], 0.6, accuracy: 1e-6) // sidetomove: no flip
            XCTAssertEqual(sample.valueTarget[2], 0)
        }
    }

    /// A directory of SYMLINKS to SGFs (as produced by `rankgames -link-dir`, the WP-19/D2
    /// active-learning selection) must feed straight into `analysisexport -sgf-dir`. The corpus
    /// walk resolves symlinks before its regular-file test, so this locks that a link is not
    /// silently skipped (a symlink's own isRegularFile is false).
    func testExportFollowsSymlinkedSgfs() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let realDir = root.appendingPathComponent("real")
        let linkDir = root.appendingPathComponent("links")
        try FileManager.default.createDirectory(at: realDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: linkDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let realSGF = realDir.appendingPathComponent("a.sgf")
        try Data("(;SZ[9]KM[7]RU[Chinese]RE[B+R];B[ee];W[ff];B[df];W[cd];B[gg])".utf8).write(to: realSGF)
        try FileManager.default.createSymbolicLink(
            at: linkDir.appendingPathComponent("rank0000_a.sgf"), withDestinationURL: realSGF
        )

        let summary = try AnalysisExporter.run(AnalysisExportOptions(
            sgfDirectory: linkDir,
            queriesOutput: root.appendingPathComponent("queries.jsonl"),
            manifestOutput: root.appendingPathComponent("manifest.json"),
            maxVisits: 500,
            minimumMoves: 4 // the fixture is a 5-move game; keep the filter from masking the symlink test
        ))
        XCTAssertEqual(summary.gamesExported, 1)
        XCTAssertEqual(summary.totalPositions, 5)
    }

    /// Helper: the side to move at `turn` for an SGF, as "B"/"W", used to craft correctly-aligned
    /// mock responses (mirrors the ingest rule nextPlayer == moves[turn].pla).
    private func sideToMovePlayerString(sgf: URL, turn: Int) -> String? {
        guard let game = try? SGFReader.load(sgf), turn < game.moves.count else { return nil }
        return game.moves[turn].pla == .black ? "B" : "W"
    }
}
