import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// WP-ANALYZE: `kata-genmove_analyze` — the GTP command the CGOS/uec-go tournament viewer's client
/// (`.tools/cgos-client/bin/gtpengine.py`) auto-detects and switches to, so RinGo's per-move
/// evaluation (winrate/score/pv/ownership) shows on the live game page.
///
/// Gates:
///  (a) FORMAT — the response parses under a Swift mirror of the client's `AnalyzeResultParser`
///      (fields present, floats in range, pv/move legal, `play` line last).
///  (b) EQUIVALENCE — from an identical seeded position, `kata-genmove_analyze` selects the same
///      move AND reports the same root visit distribution as `genmove`, with graph-search off AND on.
///  (c) OWNERSHIP — emitted iff `ownership true` requested, exactly boardArea (81) floats in [-1,1].
///  (d) list_commands / known_command advertise it.
///
/// Search-dependent tests need real MLX inference; they skip if the KataGo model fixture is absent.
final class GTPAnalyzeTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"
    private static let nnLen = 9

    // MARK: - (d) command advertisement (no model needed)

    func testListCommandsAndKnownCommandAdvertiseKataGenmoveAnalyze() async {
        let engine = makeStubEngine()

        let list = await engine.handle("list_commands")
        let listText = try? XCTUnwrap(list.response)
        XCTAssertTrue(listText?.contains("kata-genmove_analyze") == true,
                      "list_commands must include kata-genmove_analyze: \(listText ?? "nil")")

        let known = await engine.handle("known_command kata-genmove_analyze")
        XCTAssertEqual(known.response, "= true\n\n")
    }

    func testKataGenmoveAnalyzeRequiresColorArgument() async {
        let engine = makeStubEngine()
        let missing = await engine.handle("kata-genmove_analyze")
        XCTAssertEqual(missing.response, "? kata-genmove_analyze requires a color\n\n")
    }

    // MARK: - (a) format

    func testResponseParsesUnderAnalyzeResultParserMirror() async throws {
        let engine = try await makeEngine(graphSearch: false)
        _ = await engine.handle("boardsize 9")
        _ = await engine.handle("komi 7")

        let raw = await engine.handle("kata-genmove_analyze b ownership true")
        let response = try XCTUnwrap(raw.response)
        let parsed = try Self.parseGenmoveAnalyzeResponse(response)

        // A legal final move on the play line.
        XCTAssertTrue(parsed.move == "pass" || parsed.move == "resign"
            || Location.ofString(parsed.move, Self.nnLen, Self.nnLen) != nil,
            "play line move not legal: \(parsed.move)")

        let analysis = try XCTUnwrap(parsed.analysis, "expected an analysis line before the play line")
        let info = try Self.parseAnalysisLine(analysis)

        XCTAssertFalse(info.moves.isEmpty, "expected at least one info block")
        for move in info.moves {
            XCTAssertTrue(move.moveString == "pass"
                || Location.ofString(move.moveString, Self.nnLen, Self.nnLen) != nil,
                "info move not a legal vertex: \(move.moveString)")
            XCTAssertGreaterThanOrEqual(move.visits, 0)
            XCTAssertTrue((0 ... 1).contains(move.winrate), "winrate out of [0,1]: \(move.winrate)")
            XCTAssertTrue((0 ... 1).contains(move.prior), "prior out of [0,1]: \(move.prior)")
            XCTAssertTrue(move.scoreLead.isFinite, "scoreLead not finite: \(move.scoreLead)")
            XCTAssertGreaterThanOrEqual(move.order, 0)
            XCTAssertFalse(move.pv.isEmpty, "pv must have at least one move")
            for pvMove in move.pv {
                XCTAssertTrue(pvMove == "pass"
                    || Location.ofString(pvMove, Self.nnLen, Self.nnLen) != nil,
                    "pv move not legal: \(pvMove)")
            }
        }

        // Selected move must appear among the reported info moves (unless a resign/pass shortcut).
        if parsed.move != "resign", parsed.move != "pass" {
            XCTAssertTrue(info.moves.contains { $0.moveString.lowercased() == parsed.move.lowercased() },
                          "selected move \(parsed.move) absent from info blocks")
        }
    }

    // MARK: - (b) equivalence with genmove — graph off and on

    func testAnalyzeMatchesGenmoveMoveAndVisitDistributionGraphOff() async throws {
        try await assertAnalyzeEquivalentToGenmove(graphSearch: false)
    }

    func testAnalyzeMatchesGenmoveMoveAndVisitDistributionGraphOn() async throws {
        try await assertAnalyzeEquivalentToGenmove(graphSearch: true)
    }

    private func assertAnalyzeEquivalentToGenmove(graphSearch: Bool) async throws {
        let genmoveEngine = try await makeEngine(graphSearch: graphSearch)
        let analyzeEngine = try await makeEngine(graphSearch: graphSearch)
        for engine in [genmoveEngine, analyzeEngine] {
            _ = await engine.handle("boardsize 9")
            _ = await engine.handle("komi 7")
            _ = await engine.handle("play b e5")
            _ = await engine.handle("play w c3")
        }

        let genmoveMove = try await Self.responseMove(genmoveEngine.handle("genmove b"))
        let analyzeRaw = await analyzeEngine.handle("kata-genmove_analyze b")
        let analyzeResponse = try XCTUnwrap(analyzeRaw.response)
        let analyzeParsed = try Self.parseGenmoveAnalyzeResponse(analyzeResponse)

        XCTAssertEqual(analyzeParsed.move.lowercased(), genmoveMove.lowercased(),
                       "graph-search=\(graphSearch): analyze selected a different move than genmove")

        let genmoveVisitsRaw = await genmoveEngine.lastGenmoveVisitsByMove()
        let analyzeVisitsRaw = await analyzeEngine.lastGenmoveVisitsByMove()
        let genmoveVisits = try XCTUnwrap(genmoveVisitsRaw)
        let analyzeVisits = try XCTUnwrap(analyzeVisitsRaw)
        XCTAssertEqual(genmoveVisits.count, analyzeVisits.count,
                       "graph-search=\(graphSearch): differing number of root children")
        for (lhs, rhs) in zip(genmoveVisits, analyzeVisits) {
            XCTAssertEqual(lhs.loc, rhs.loc, "graph-search=\(graphSearch): visit-distribution move mismatch")
            XCTAssertEqual(lhs.visits, rhs.visits, "graph-search=\(graphSearch): visit-count mismatch at \(lhs.loc)")
        }
    }

    // MARK: - (c) ownership gating

    func testOwnershipEmittedOnlyWhenRequestedWithBoardAreaFloats() async throws {
        let withEngine = try await makeEngine(graphSearch: false)
        let withoutEngine = try await makeEngine(graphSearch: false)
        for engine in [withEngine, withoutEngine] {
            _ = await engine.handle("boardsize 9")
            _ = await engine.handle("komi 7")
        }

        let withRaw = await withEngine.handle("kata-genmove_analyze b ownership true")
        let withResponse = try XCTUnwrap(withRaw.response)
        let withParsed = try Self.parseGenmoveAnalyzeResponse(withResponse)
        let withInfo = try Self.parseAnalysisLine(XCTUnwrap(withParsed.analysis))
        let ownership = try XCTUnwrap(withInfo.ownership, "ownership true must emit an ownership block")
        XCTAssertEqual(ownership.count, Self.nnLen * Self.nnLen, "ownership must be boardArea (81) floats")
        for value in ownership {
            XCTAssertTrue((-1.0 ... 1.0).contains(value), "ownership value out of [-1,1]: \(value)")
        }

        let withoutRaw = await withoutEngine.handle("kata-genmove_analyze b ownership false")
        let withoutResponse = try XCTUnwrap(withoutRaw.response)
        let withoutParsed = try Self.parseGenmoveAnalyzeResponse(withoutResponse)
        let withoutInfo = try Self.parseAnalysisLine(XCTUnwrap(withoutParsed.analysis))
        XCTAssertNil(withoutInfo.ownership, "ownership false must NOT emit an ownership block")
    }

    // MARK: - Response / analysis-line parsing (mirror of the CGOS client)

    private struct GenmoveAnalyzeResponse {
        var analysis: String?
        var move: String
    }

    /// Mirrors `EngineConnector._sendRawGTPCommand` + `genmoveKata`: strip the leading `=`/`?` marker,
    /// gather non-blank lines, the LAST non-`play` line is the analysis, the `play` line gives the move.
    private static func parseGenmoveAnalyzeResponse(_ response: String) throws -> GenmoveAnalyzeResponse {
        var analysis: String?
        var move: String?
        for rawLine in response.split(separator: "\n", omittingEmptySubsequences: false) {
            var line = String(rawLine)
            if line.first == "=" || line.first == "?" { line.removeFirst() }
            line = line.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("play ") {
                move = String(line.dropFirst("play ".count)).trimmingCharacters(in: .whitespaces)
                break
            }
            analysis = line
        }
        let unwrappedMove = try XCTUnwrap(move, "response had no play line: \(response.debugDescription)")
        return GenmoveAnalyzeResponse(analysis: analysis, move: unwrappedMove)
    }

    private struct InfoMove {
        var moveString = ""
        var visits = 0
        var winrate = Double.nan
        var scoreLead = Double.nan
        var prior = Double.nan
        var order = -1
        var pv: [String] = []
    }

    private struct AnalysisLine {
        var moves: [InfoMove] = []
        var ownership: [Double]?
    }

    /// Strict mirror of `AnalyzeResultParser`: token stream of `info`/`ownership` top-level blocks.
    private static func parseAnalysisLine(_ line: String) throws -> AnalysisLine {
        let tokens = line.split(separator: " ").map(String.init)
        var result = AnalysisLine()
        var index = 0
        func isTopLevel(_ token: String) -> Bool {
            token == "info" || token == "ownership"
        }

        while index < tokens.count {
            let token = tokens[index]
            index += 1
            if token == "info" {
                var move = InfoMove()
                while index < tokens.count, !isTopLevel(tokens[index]) {
                    let key = tokens[index]
                    index += 1
                    if key == "pv" {
                        while index < tokens.count, !isTopLevel(tokens[index]) {
                            move.pv.append(tokens[index])
                            index += 1
                        }
                        break
                    }
                    guard index < tokens.count else { break }
                    let value = tokens[index]
                    index += 1
                    switch key {
                    case "move": move.moveString = value
                    case "visits": move.visits = try XCTUnwrap(Int(value), "visits not an int: \(value)")
                    case "winrate": move.winrate = try XCTUnwrap(Double(value), "winrate not a float: \(value)")
                    case "scoreLead": move.scoreLead = try XCTUnwrap(Double(value), "scoreLead not a float: \(value)")
                    case "prior": move.prior = try XCTUnwrap(Double(value), "prior not a float: \(value)")
                    case "order": move.order = try XCTUnwrap(Int(value), "order not an int: \(value)")
                    default: break // tolerate unknown keys, like the real parser
                    }
                }
                result.moves.append(move)
            } else if token == "ownership" {
                var values: [Double] = []
                while index < tokens.count, !isTopLevel(tokens[index]) {
                    try values.append(XCTUnwrap(Double(tokens[index]), "ownership not a float: \(tokens[index])"))
                    index += 1
                }
                result.ownership = values
            }
        }
        return result
    }

    private static func responseMove(_ result: (response: String?, shouldQuit: Bool)) throws -> String {
        let response = try XCTUnwrap(result.response)
        return response.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Fixtures

    private func makeEngine(graphSearch: Bool) async throws -> GTPEngine {
        let modelPath = try modelPath()
        let desc = try ModelDesc.loadFromFileMaybeGZipped(modelPath)
        let evaluator = try NNEvaluator(
            desc: desc,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            maxBatchSize: 8
        )
        var settings = SearchSettings()
        settings.maxVisits = 32
        settings.numSearchWorkers = 4
        settings.leafBatchSize = 8
        settings.useGraphSearch = graphSearch
        return GTPEngine(
            evaluator: evaluator,
            modelDesc: desc,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            settings: settings,
            rules: .getTrompTaylorish(),
            nnLength: .fixed(Self.nnLen),
            maxBatchSize: 8
        )
    }

    private func makeStubEngine() -> GTPEngine {
        var settings = SearchSettings()
        settings.maxVisits = 1
        return GTPEngine(
            evaluator: StubAnalyzeEvaluator(),
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            settings: settings,
            rules: .getTrompTaylorish(),
            nnLength: .fixed(Self.nnLen),
            warmupBucketSizes: [1, 2],
            evaluatorFactory: { _, _ in StubAnalyzeEvaluator() }
        )
    }

    private func modelPath() throws -> String {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"] ?? Self.defaultModelsDirectory
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw XCTSkip("KataGo model fixtures are missing; set KATAGO_MODELS_DIR (checked \(directory))")
        }
        let path = URL(fileURLWithPath: directory).appendingPathComponent(Self.modelFile).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("KataGo model fixture is missing: \(path)")
        }
        return path
    }
}

private actor StubAnalyzeEvaluator: NNEvaluating {
    func evaluate(_: [NNRequest]) async throws -> [NNOutput] {
        throw GTPError("Stub evaluator cannot evaluate positions")
    }

    func preWarm(_: [NNRequest]) {}
}
