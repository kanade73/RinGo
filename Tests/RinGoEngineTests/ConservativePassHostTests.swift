import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// HOST-ONLY / real b6c96 model. These tests require MLX/Metal and are intentionally skipped in
/// CPU CI. Set `RUN_HOST_ONLY_MLX_TESTS=1` on the tournament host to run the five-game gate.
final class ConservativePassHostTests: XCTestCase {
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"

    func testHostOnlyFiveRealModelGamesEndCleanlyByDoublePass() async throws {
        guard ProcessInfo.processInfo.environment["RUN_HOST_ONLY_MLX_TESTS"] == "1" else {
            throw XCTSkip("HOST-ONLY: set RUN_HOST_ONLY_MLX_TESTS=1 on a Metal-capable host")
        }
        let description = try ModelDesc.loadFromFileMaybeGZipped(modelPath())
        let evaluator = try NNEvaluator(
            desc: description,
            nnXLen: 9,
            nnYLen: 9,
            precision: .float16,
            maxBatchSize: 4
        )
        var settings = SearchSettings()
        settings.maxVisits = 24
        settings.leafBatchSize = 4
        settings.numSearchWorkers = 4
        settings.resignEnabled = false

        var openings = [[Loc]]()
        for gameIndex in 0 ..< 5 {
            let opening = try await runGame(
                index: gameIndex,
                evaluator: evaluator,
                description: description,
                settings: settings
            )
            openings.append(opening)
        }
        XCTAssertEqual(Set(openings).count, 5, "seeded openings must make all five games distinct")
    }

    private func runGame(
        index: Int,
        evaluator: NNEvaluator,
        description: ModelDesc,
        settings: SearchSettings
    ) async throws -> [Loc] {
        let rules = Rules(
            koRule: .positional,
            scoringRule: .area,
            taxRule: .none,
            multiStoneSuicideLegal: true,
            komi: 7
        )
        let black = GTPEngine(
            evaluator: evaluator,
            modelDesc: description,
            nnXLen: 9,
            nnYLen: 9,
            precision: .float16,
            settings: settings,
            rules: rules,
            nnLength: .fixed(9),
            maxBatchSize: 4,
            initialTimeSeconds: 120
        )
        let white = GTPEngine(
            evaluator: evaluator,
            modelDesc: description,
            nnXLen: 9,
            nnYLen: 9,
            precision: .float16,
            settings: settings,
            rules: rules,
            nnLength: .fixed(9),
            maxBatchSize: 4,
            initialTimeSeconds: 120
        )
        for engine in [black, white] {
            let boardSizeResult = await engine.handle("boardsize 9")
            XCTAssertEqual(boardSizeResult.response, "= \n\n")
            let komiResult = await engine.handle("komi 7")
            XCTAssertEqual(komiResult.response, "= \n\n")
            let timeResult = await engine.handle("time_settings 120 0 0")
            XCTAssertEqual(timeResult.response, "= \n\n")
        }

        let board = Board(9, 9)
        let history = BoardHistory(board, pla: .black, rules: rules)
        let opening = try await playSeededOpening(
            gameIndex: index,
            engines: [black, white],
            board: board,
            history: history
        )
        var consecutivePasses = 0
        var endedByDoublePass = false
        for moveIndex in opening.count ..< 120 {
            let player = history.presumedNextMovePla
            let current = player == .black ? black : white
            let opponent = player == .black ? white : black
            let color = player == .black ? "B" : "W"
            let genmoveResult = await current.handle("genmove \(color)")
            let moveText = try responseText(XCTUnwrap(genmoveResult.response))
            XCTAssertNotEqual(moveText, "resign", "game \(index), move \(moveIndex): resigned")
            guard moveText != "resign", let loc = Location.ofString(moveText, 9, 9) else {
                XCTFail("game \(index), move \(moveIndex): invalid move \(moveText)")
                throw GTPError("game \(index), move \(moveIndex): invalid move \(moveText)")
            }
            XCTAssertTrue(history.isLegal(board, loc, player), "game \(index), move \(moveIndex): illegal move")
            history.makeBoardMoveAssumeLegal(board, loc, player)
            let relayResult = await opponent.handle("play \(color) \(moveText)")
            XCTAssertEqual(relayResult.response, "= \n\n")
            consecutivePasses = loc == Board.passLoc ? consecutivePasses + 1 : 0
            if consecutivePasses == 2 {
                endedByDoublePass = true
                break
            }
        }

        XCTAssertTrue(endedByDoublePass, "game \(index): did not double-pass within 120 moves")
        XCTAssertLessThanOrEqual(history.moveHistory.count, 120)
        XCTAssertTrue(history.isGameFinished)
        XCTAssertTrue(history.isScored)
        let expectedScore = formattedScore(history.finalWhiteMinusBlackScore)
        for engine in [black, white] {
            let scoreResult = await engine.handle("final_score")
            let actual = try responseText(XCTUnwrap(scoreResult.response))
            XCTAssertEqual(actual, expectedScore, "game \(index): GTP/reference score mismatch")
            let timeManager = await engine.currentTimeManager()
            XCTAssertGreaterThan(timeManager.timeLeftSeconds, 0)
        }

        let independentArea = board.calculateIndependentLifeArea(
            keepTerritories: false,
            keepStones: false,
            isMultiStoneSuicideLegal: rules.multiStoneSuicideLegal
        ).area
        for loc in board.playableLocations() {
            guard let player = board.colors[loc].player else { continue }
            XCTAssertNotEqual(
                independentArea[loc],
                player.opponent.color,
                "game \(index): removable \(player) stone remains at \(Location.toString(loc, 9, 9))"
            )
        }
        return opening
    }

    private func playSeededOpening(
        gameIndex: Int,
        engines: [GTPEngine],
        board: Board,
        history: BoardHistory
    ) async throws -> [Loc] {
        var random = SplitMix64(seed: 0x4347_4F53_4F50_454E &+ UInt64(gameIndex))
        var opening = [Loc]()
        for _ in 0 ..< 4 {
            let player = history.presumedNextMovePla
            let legalMoves = board.playableLocations().filter { history.isLegal(board, $0, player) }
            let loc = legalMoves[random.nextIndex(upperBound: legalMoves.count)]
            opening.append(loc)
            history.makeBoardMoveAssumeLegal(board, loc, player)

            let color = player == .black ? "B" : "W"
            let vertex = Location.toString(loc, 9, 9)
            for engine in engines {
                let result = await engine.handle("play \(color) \(vertex)")
                XCTAssertEqual(result.response, "= \n\n")
            }
        }
        return opening
    }

    private func modelPath() throws -> String {
        let directory = ProcessInfo.processInfo.environment["KATAGO_MODELS_DIR"]
            ?? Self.defaultModelsDirectory
        let path = URL(fileURLWithPath: directory).appendingPathComponent(Self.modelFile).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("HOST-ONLY model fixture missing: \(path)")
        }
        return path
    }

    private func responseText(_ response: String) -> String {
        String(response.dropFirst(2)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func formattedScore(_ whiteMinusBlack: Float) -> String {
        if whiteMinusBlack == 0 { return "0" }
        let prefix = whiteMinusBlack > 0 ? "W+" : "B+"
        let magnitude = abs(whiteMinusBlack)
        let number = magnitude == magnitude.rounded(.towardZero)
            ? String(format: "%.0f", magnitude) : String(format: "%.1f", magnitude)
        return prefix + number
    }
}

private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func nextIndex(upperBound: Int) -> Int {
        precondition(upperBound > 0)
        let bound = UInt64(upperBound)
        let limit = UInt64.max - UInt64.max % bound
        var value = next()
        while value >= limit {
            value = next()
        }
        return Int(value % bound)
    }

    private mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}
