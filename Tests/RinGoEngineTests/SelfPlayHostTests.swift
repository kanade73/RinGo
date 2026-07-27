import Foundation
@testable import ringo
import RinGoCore
import RinGoEngine
import XCTest

/// HOST-ONLY: constructs a real NNEvaluator and executes MLX/Metal graphs. CPU CI must skip it.
final class SelfPlayHostTests: XCTestCase {
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"

    func testHostOnlyTwoRealGamesTerminateRoundTripAndAreSeedReproducible() async throws {
        guard ProcessInfo.processInfo.environment["RUN_HOST_ONLY_MLX_TESTS"] == "1" else {
            throw XCTSkip("HOST-ONLY: set RUN_HOST_ONLY_MLX_TESTS=1 on a Metal-capable host")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ringo-selfplay-host-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let first = root.appendingPathComponent("first")
        let repeatRun = root.appendingPathComponent("repeat")
        let different = root.appendingPathComponent("different")
        var options = try SelfPlayOptions(
            model: modelPath(),
            games: 2,
            outputDirectory: first.path,
            visits: 4,
            precision: .float16,
            temperature: 0.9,
            temperatureMoves: 8,
            dirichletAlpha: 0.15,
            noiseWeight: 0.25,
            seed: 1,
            komi: 7,
            size: 9,
            maxBatchSize: 2
        )
        let firstFiles = try await SelfPlayCommand.run(options, date: "2026-07-06")
        XCTAssertEqual(firstFiles.count, 2)
        for file in firstFiles {
            try assertFinishedScorableRoundTrip(file)
        }

        options.outputDirectory = repeatRun.path
        let repeatedFiles = try await SelfPlayCommand.run(options, date: "2026-07-06")
        let firstData = try firstFiles.map { try Data(contentsOf: $0) }
        let repeatedData = try repeatedFiles.map { try Data(contentsOf: $0) }
        XCTAssertEqual(firstData, repeatedData)

        options.seed = 2
        options.outputDirectory = different.path
        let differentFiles = try await SelfPlayCommand.run(options, date: "2026-07-06")
        let differentData = try differentFiles.map { try Data(contentsOf: $0) }
        XCTAssertNotEqual(firstData, differentData)
    }

    private func assertFinishedScorableRoundTrip(_ file: URL) throws {
        let source = try String(contentsOf: file, encoding: .utf8)
        let game = try SGFReader.parse(source)
        XCTAssertEqual(game.boardXSize, 9)
        XCTAssertEqual(game.boardYSize, 9)
        XCTAssertEqual(game.komi, 7)
        XCTAssertFalse(game.moves.isEmpty)
        let doublePass = game.moves.count >= 2
            && game.moves.suffix(2).allSatisfy { $0.loc == Board.passLoc }
        XCTAssertTrue(doublePass || game.moves.count == 400)

        let position = try game.position(at: game.moves.count)
        if !position.history.isScored {
            position.history.endAndScoreGameNow(position.board)
        }
        XCTAssertTrue(position.history.isGameFinished)
        XCTAssertTrue(position.history.isScored)
        XCTAssertEqual(game.result, formattedResult(position.history.finalWhiteMinusBlackScore))
    }

    private func formattedResult(_ score: Float) -> String {
        if score == 0 { return "0" }
        let prefix = score > 0 ? "W+" : "B+"
        let magnitude = abs(score)
        let number = magnitude == magnitude.rounded(.towardZero)
            ? String(format: "%.0f", magnitude) : String(format: "%.1f", magnitude)
        return prefix + number
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
}
