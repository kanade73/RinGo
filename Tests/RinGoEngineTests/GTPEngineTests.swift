import Foundation
@testable import ringo
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// Drives `GTPEngine` (Sources/RinGoEngine/GTPEngine.swift) through a canned command script
/// directly via `handle(_:)` -- no process/stdio involved, per the task spec ("factor the loop so
/// tests can drive it without spawning a process"). Uses the b6c96 test model at fp32, `nnLen =
/// 9`, and a small visit count purely for test speed.
final class GTPEngineTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"
    private static let nnLen = 9

    func testNNLengthArgumentParsing() throws {
        let defaults = try GTPCommand.parse(["-model", "model.bin.gz"])
        XCTAssertEqual(defaults.nnLength, .auto)

        let automatic = try GTPCommand.parse(["-model", "model.bin.gz", "-nnlen", "auto"])
        XCTAssertEqual(automatic.nnLength, .auto)

        let fixed = try GTPCommand.parse(["-model", "model.bin.gz", "-nnlen", "9"])
        XCTAssertEqual(fixed.nnLength, .fixed(9))

        XCTAssertThrowsError(try GTPCommand.parse(["-model", "model.bin.gz", "-nnlen", "invalid"])) {
            XCTAssertEqual(
                String(describing: $0),
                "Invalid -nnlen 'invalid'; expected auto or an integer from 1 to \(Board.maxLen)"
            )
        }
    }

    func testFixedNNLengthRejectsLargerBoard() async {
        let recorder = EvaluatorFactoryRecorder()
        let engine = makeStubEngine(nnLength: .fixed(9), initialNNLen: 9, recorder: recorder)

        let result = await engine.handle("boardsize 19")

        XCTAssertEqual(result.response, "? boardsize 19 exceeds fixed nnlen 9\n\n")
        XCTAssertFalse(result.shouldQuit)
        XCTAssertEqual(recorder.calls, [])
    }

    func testAutoNNLengthRebuildsOnceForChangedSize() async throws {
        let recorder = EvaluatorFactoryRecorder()
        let engine = makeStubEngine(nnLength: .auto, initialNNLen: 19, recorder: recorder)
        try await engine.preWarm(bucketSizes: [1, 2])

        let firstChange = await engine.handle("boardsize 9")
        XCTAssertEqual(firstChange.response, "= \n\n")
        XCTAssertEqual(recorder.calls, [.init(x: 9, y: 9)])
        let rebuilt = try XCTUnwrap(recorder.evaluators.first)
        let warmedBatchSizes = await rebuilt.warmedBatchSizes()
        XCTAssertEqual(warmedBatchSizes, [1, 2])

        let repeatedSize = await engine.handle("boardsize 9")
        XCTAssertEqual(repeatedSize.response, "= \n\n")
        XCTAssertEqual(recorder.calls, [.init(x: 9, y: 9)])
    }

    func testScriptedSessionProducesWellFormedResponses() async throws {
        let engine = try await makeEngine()

        // (command line, optional exact expected response text, whether a response is expected)
        var responses: [(line: String, response: String?, shouldQuit: Bool)] = []
        let script = [
            "1 protocol_version",
            "2 name",
            "3 version",
            "4 list_commands",
            "5 boardsize 9",
            "6 clear_board",
            "7 komi 7.5",
            "8 play B E5",
            "9 play W C3",
            "10 genmove B",
            "11 showboard",
            "12 undo",
            "13 final_score",
            "14 nonexistent_command",
            "15 quit",
        ]
        for line in script {
            let (response, shouldQuit) = await engine.handle(line)
            responses.append((line, response, shouldQuit))
            if shouldQuit { break }
        }

        for (line, response, _) in responses {
            guard let response else {
                XCTFail("line '\(line)' produced no response at all")
                continue
            }
            XCTAssertTrue(
                response.hasPrefix("=") || response.hasPrefix("?"),
                "line '\(line)': malformed response \(response.debugDescription)"
            )
            XCTAssertTrue(
                response.hasSuffix("\n\n"),
                "line '\(line)': response missing blank-line terminator: \(response.debugDescription)"
            )
            // Every request here carries a leading numeric id; it must be echoed right after the
            // marker with no intervening space.
            let id = line.split(separator: " ", maxSplits: 1)[0]
            XCTAssertTrue(
                response.hasPrefix("=\(id) ") || response.hasPrefix("?\(id) "),
                "line '\(line)': id not echoed correctly in \(response.debugDescription)"
            )
        }

        XCTAssertEqual(responses[0].response, "=1 2\n\n")
        XCTAssertEqual(responses[1].response, "=2 ringo\n\n")
        XCTAssertTrue(responses[3].response?.contains("genmove") == true)
        XCTAssertTrue(responses[3].response?.contains("quit") == true)

        // genmove (index 9) must answer with a syntactically valid move: a vertex on the 9x9
        // board, "pass", or "resign".
        let genmoveText = try XCTUnwrap(responses[9].response?
            .dropFirst("=10 ".count)
            .trimmingCharacters(in: .whitespacesAndNewlines))
        XCTAssertTrue(
            genmoveText == "pass" || genmoveText == "resign"
                || Location.ofString(genmoveText, Self.nnLen, Self.nnLen) != nil,
            "genmove returned an unparseable move: \(genmoveText)"
        )

        // final_score (index 12) must be "0", "B+<n>", or "W+<n>".
        let scoreText = try XCTUnwrap(responses[12].response?
            .dropFirst("=13 ".count)
            .trimmingCharacters(in: .whitespacesAndNewlines))
        XCTAssertTrue(
            scoreText == "0" || scoreText.hasPrefix("B+") || scoreText.hasPrefix("W+"),
            "final_score returned an unparseable score: \(scoreText)"
        )

        // Unknown command (index 13) must be a "?" error mentioning "unknown command".
        XCTAssertTrue(responses[13].response?.hasPrefix("?14 ") == true)
        XCTAssertTrue(responses[13].response?.lowercased().contains("unknown command") == true)

        // quit (index 14) must signal shouldQuit.
        XCTAssertTrue(responses.last?.shouldQuit == true)
    }

    /// HOST-ONLY / MLX-dependent: exercises real compile buckets and GPU inference.
    func testHostOnlyAutoNNLengthScriptedSessionSwitches9To19() async throws {
        guard ProcessInfo.processInfo.environment["RUN_HOST_ONLY_MLX_TESTS"] == "1" else {
            throw XCTSkip("HOST-ONLY: set RUN_HOST_ONLY_MLX_TESTS=1 on a Metal-capable host")
        }
        let engine = try await makeEngine(nnLength: .auto, initialNNLen: 19)
        try await engine.preWarm(bucketSizes: [1, 2, 4, 8])

        let board9 = await engine.handle("boardsize 9")
        XCTAssertEqual(board9.response, "= \n\n")
        let move9 = try await responseText(engine.handle("genmove B"))
        XCTAssertTrue(
            move9 == "pass" || move9 == "resign" || Location.ofString(move9, 9, 9) != nil,
            "9x9 genmove returned an invalid vertex: \(move9)"
        )

        let board19 = await engine.handle("boardsize 19")
        XCTAssertEqual(board19.response, "= \n\n")
        let move19 = try await responseText(engine.handle("genmove B"))
        XCTAssertTrue(
            move19 == "pass" || move19 == "resign" || Location.ofString(move19, 19, 19) != nil,
            "19x19 genmove returned an invalid vertex: \(move19)"
        )
    }

    /// HOST-ONLY / real-model tournament clock smoke test. Two independent 9x9 engines each get
    /// the CLI `-time 60` override and must finish a game without either local clock reaching zero.
    func testHostOnlyRealModelNineByNineGameCompletesWithinSixtySecondsPerSide() async throws {
        guard ProcessInfo.processInfo.environment["RUN_HOST_ONLY_MLX_TESTS"] == "1" else {
            throw XCTSkip("HOST-ONLY: set RUN_HOST_ONLY_MLX_TESTS=1 on a Metal-capable host")
        }
        let options = try GTPCommand.parse(["-model", "fixture.bin.gz", "-time", "60"])
        let time = try XCTUnwrap(options.timeSeconds)
        let black = try await makeEngine(initialTimeSeconds: time)
        let white = try await makeEngine(initialTimeSeconds: time)
        for engine in [black, white] {
            let boardResponse = await engine.handle("boardsize 9")
            XCTAssertEqual(boardResponse.response, "= \n\n")
            let komiResponse = await engine.handle("komi 7.0")
            XCTAssertEqual(komiResponse.response, "= \n\n")
        }

        var nextPlayer = Player.black
        var consecutivePasses = 0
        var completed = false
        for _ in 0 ..< 400 {
            let current = nextPlayer == .black ? black : white
            let opponent = nextPlayer == .black ? white : black
            let color = nextPlayer == .black ? "B" : "W"
            let move = try await responseText(current.handle("genmove \(color)"))
            if move == "resign" {
                completed = true
                break
            }
            let relay = await opponent.handle("play \(color) \(move)")
            XCTAssertEqual(relay.response, "= \n\n")
            consecutivePasses = move == "pass" ? consecutivePasses + 1 : 0
            if consecutivePasses == 2 {
                completed = true
                break
            }
            nextPlayer = nextPlayer.opponent
        }

        XCTAssertTrue(completed, "real-model game did not finish within the tournament 400-move limit")
        let blackClock = await black.currentTimeManager()
        let whiteClock = await white.currentTimeManager()
        XCTAssertGreaterThan(blackClock.timeLeftSeconds, 0)
        XCTAssertGreaterThan(whiteClock.timeLeftSeconds, 0)
    }

    func testUnknownCommandWithoutIdStillWellFormed() async {
        let engine = makeStubEngine(
            nnLength: .fixed(19),
            initialNNLen: 19,
            recorder: EvaluatorFactoryRecorder()
        )
        let (response, shouldQuit) = await engine.handle("frobnicate")
        XCTAssertFalse(shouldQuit)
        guard let response else {
            XCTFail("expected a response")
            return
        }
        XCTAssertEqual(response, "? unknown command\n\n")
    }

    func testKnownCommandReportsTrueForImplementedCommandsAndFalseOtherwise() async {
        let engine = makeStubEngine(
            nnLength: .fixed(19),
            initialNNLen: 19,
            recorder: EvaluatorFactoryRecorder()
        )

        let knownExamples = [
            "protocol_version", "name", "version", "list_commands", "known_command",
            "boardsize", "clear_board", "komi", "play", "genmove", "kata-genmove_analyze",
            "time_settings",
            "time_left", "undo", "showboard", "final_score", "quit",
        ]
        for command in knownExamples {
            let (response, shouldQuit) = await engine.handle("known_command \(command)")
            XCTAssertFalse(shouldQuit)
            XCTAssertEqual(response, "= true\n\n", "expected known_command \(command) to report true")
        }

        let unknownExamples = ["frobnicate", "cgos-genmove_analyze", "lz-genmove_analyze", ""]
        for command in unknownExamples where !command.isEmpty {
            let (response, shouldQuit) = await engine.handle("known_command \(command)")
            XCTAssertFalse(shouldQuit)
            XCTAssertEqual(response, "= false\n\n", "expected known_command \(command) to report false")
        }

        // Missing argument is a protocol error, matching the other commands' arg-count checks.
        let missingArg = await engine.handle("known_command")
        XCTAssertEqual(missingArg.response, "? known_command requires a command name argument\n\n")
        XCTAssertFalse(missingArg.shouldQuit)
    }

    func testBlankLineProducesNoResponse() async {
        let engine = makeStubEngine(
            nnLength: .fixed(19),
            initialNNLen: 19,
            recorder: EvaluatorFactoryRecorder()
        )
        let (response, shouldQuit) = await engine.handle("   ")
        XCTAssertNil(response)
        XCTAssertFalse(shouldQuit)
    }

    // MARK: - Fixtures

    private func makeEngine(
        nnLength: GTPNNLength = .fixed(GTPEngineTests.nnLen),
        initialNNLen: Int = GTPEngineTests.nnLen,
        initialTimeSeconds: Double? = nil
    ) async throws -> GTPEngine {
        let modelPath = try modelPath()
        let desc = try ModelDesc.loadFromFileMaybeGZipped(modelPath)
        let evaluator = try NNEvaluator(
            desc: desc,
            nnXLen: initialNNLen,
            nnYLen: initialNNLen,
            precision: .float32,
            maxBatchSize: 8
        )
        var settings = SearchSettings()
        settings.maxVisits = 16
        settings.numSearchWorkers = 4
        settings.leafBatchSize = 8
        return GTPEngine(
            evaluator: evaluator,
            modelDesc: desc,
            nnXLen: initialNNLen,
            nnYLen: initialNNLen,
            precision: .float32,
            settings: settings,
            rules: .getTrompTaylorish(),
            nnLength: nnLength,
            maxBatchSize: 8,
            initialTimeSeconds: initialTimeSeconds
        )
    }

    private func makeStubEngine(
        nnLength: GTPNNLength,
        initialNNLen: Int,
        recorder: EvaluatorFactoryRecorder
    ) -> GTPEngine {
        var settings = SearchSettings()
        settings.maxVisits = 1
        return GTPEngine(
            evaluator: StubEvaluator(),
            nnXLen: initialNNLen,
            nnYLen: initialNNLen,
            precision: .float32,
            settings: settings,
            rules: .getTrompTaylorish(),
            nnLength: nnLength,
            warmupBucketSizes: [1, 2],
            evaluatorFactory: recorder.make
        )
    }

    private func responseText(_ result: (response: String?, shouldQuit: Bool)) throws -> String {
        let response = try XCTUnwrap(result.response)
        XCTAssertFalse(result.shouldQuit)
        return response.dropFirst(2).trimmingCharacters(in: .whitespacesAndNewlines)
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

private actor StubEvaluator: NNEvaluating {
    private var warmups = [Int]()

    func evaluate(_: [NNRequest]) async throws -> [NNOutput] {
        throw GTPError("Stub evaluator cannot evaluate positions")
    }

    func preWarm(_ requests: [NNRequest]) {
        warmups.append(requests.count)
    }

    func warmedBatchSizes() -> [Int] {
        warmups
    }
}

private final class EvaluatorFactoryRecorder: @unchecked Sendable {
    struct Call: Equatable {
        var x: Int
        var y: Int
    }

    private let lock = NSLock()
    private var recordedCalls = [Call]()
    private var recordedEvaluators = [StubEvaluator]()

    var calls: [Call] {
        lock.withLock { recordedCalls }
    }

    var evaluators: [StubEvaluator] {
        lock.withLock { recordedEvaluators }
    }

    func make(nnXLen: Int, nnYLen: Int) -> any NNEvaluating {
        let evaluator = StubEvaluator()
        lock.withLock {
            recordedCalls.append(Call(x: nnXLen, y: nnYLen))
            recordedEvaluators.append(evaluator)
        }
        return evaluator
    }
}
