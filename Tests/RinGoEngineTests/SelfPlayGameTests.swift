import RinGoCore
@testable import RinGoEngine
import XCTest

final class SelfPlayGameTests: XCTestCase {
    func testStubEvaluatorGameEndsByDoublePassAndIsReproducible() async throws {
        func play(seed: UInt64) async throws -> BoardHistory {
            var settings = SearchSettings()
            settings.leafBatchSize = 1
            settings.numSearchWorkers = 1
            settings.resignEnabled = false
            settings.randomSeed = seed
            return try await SelfPlayGameGenerator.play(
                evaluator: TestPolicyEvaluator(nnXLen: 9, nnYLen: 9, preferredLoc: Board.passLoc),
                nnXLen: 9,
                nnYLen: 9,
                precision: .float32,
                configuration: SelfPlayGameConfiguration(
                    boardSize: 9,
                    visits: 1,
                    temperature: 0.9,
                    temperatureMoves: 8,
                    rules: .getTrompTaylorish(),
                    searchSettings: settings
                )
            )
        }

        let first = try await play(seed: 17)
        let second = try await play(seed: 17)
        XCTAssertEqual(first.moveHistory, [
            Move(loc: Board.passLoc, pla: .black),
            Move(loc: Board.passLoc, pla: .white),
        ])
        XCTAssertEqual(second.moveHistory, first.moveHistory)
        XCTAssertTrue(first.isGameFinished)
        XCTAssertTrue(first.isScored)
    }

    func testMoveCapAdjudicatesCurrentPosition() async throws {
        var settings = SearchSettings()
        settings.leafBatchSize = 1
        settings.numSearchWorkers = 1
        let history = try await SelfPlayGameGenerator.play(
            evaluator: TestPolicyEvaluator(
                nnXLen: 9,
                nnYLen: 9,
                preferredLoc: Location.getLoc(0, 0, 9)
            ),
            nnXLen: 9,
            nnYLen: 9,
            precision: .float32,
            configuration: SelfPlayGameConfiguration(
                boardSize: 9,
                visits: 1,
                temperature: 0,
                temperatureMoves: 0,
                maximumMoves: 1,
                rules: .getTrompTaylorish(),
                searchSettings: settings
            )
        )
        XCTAssertEqual(history.moveHistory.count, 1)
        XCTAssertTrue(history.isGameFinished)
        XCTAssertTrue(history.isScored)
    }

    // MARK: - WP-9 training-data capture

    func testVisitDistributionNormalizesAndMapsPass() {
        let distribution = SelfPlayGameGenerator.visitDistribution(
            [(Location.getLoc(0, 0, 9), 3), (Board.passLoc, 1)],
            boardXSize: 9,
            nnLen: 9
        )
        XCTAssertEqual(distribution.count, 82)
        XCTAssertEqual(distribution.reduce(0, +), 1, accuracy: 1e-6)
        XCTAssertEqual(distribution[0], 0.75, accuracy: 1e-6) // (0,0) -> pos 0
        XCTAssertEqual(distribution[81], 0.25, accuracy: 1e-6) // pass -> area index
        // Legal-mask consistency: only the two searched (legal) moves carry mass; all else is 0.
        for index in 1 ..< 81 {
            XCTAssertEqual(distribution[index], 0)
        }
        // Degenerate no-visit case: an all-zero vector (cannot arise for visits >= 1 in practice).
        XCTAssertEqual(SelfPlayGameGenerator.visitDistribution([], boardXSize: 9, nnLen: 9).reduce(0, +), 0)
    }

    func testTrainingSamplePerspectiveForDecisiveGame() {
        // Black plays one stone on an otherwise empty 3x3; area scoring (komi 0) gives black the
        // whole board, so black wins by 9. Two records for the SAME final position differ only in
        // side-to-move, exercising the value/score/ownership perspective flip.
        let rules = Rules(scoringRule: .area, komi: 0)
        let board = Board(3, 3)
        let history = BoardHistory(board, pla: .black, rules: rules)
        history.makeBoardMoveAssumeLegal(board, Location.getLoc(1, 1, 3), .black)
        history.endAndScoreGameNow(board)
        XCTAssertEqual(history.winner, .black)
        XCTAssertEqual(history.finalWhiteMinusBlackScore, -9)

        let result = SelfPlayResult(
            history: history,
            records: [dummyRecord(nextPlayer: .black, nnLen: 3), dummyRecord(nextPlayer: .white, nnLen: 3)],
            finalArea: history.getAreaNow(board),
            boardXSize: 3
        )
        let samples = SelfPlayGameGenerator.trainingSamples(from: result, nnLen: 3)
        XCTAssertEqual(samples.count, 2)
        let black = samples[0]
        let white = samples[1]

        XCTAssertEqual(black.valueTarget, [1, 0, 0]) // black (mover) won
        XCTAssertEqual(white.valueTarget, [0, 1, 0]) // white (mover) lost
        XCTAssertEqual(black.scoreTarget, 9) // +9 from black's perspective
        XCTAssertEqual(white.scoreTarget, -9)
        XCTAssertTrue(black.scoreValid && black.ownershipValid)
        XCTAssertTrue(white.scoreValid && white.ownershipValid)
        // Black owns every point: +1 for the black mover, exactly negated for the white mover.
        XCTAssertEqual(black.ownershipTarget, [Int8](repeating: 1, count: 9))
        XCTAssertEqual(white.ownershipTarget, black.ownershipTarget.map { -$0 })
    }

    func testJigoProducesSoftValueTarget() {
        // Empty board, area scoring, komi 0 -> exact draw: winner nil but the game IS scored, so a
        // soft [0.5, 0.5, 0] value target is emitted (legal since WP-2a) rather than skipping it.
        let rules = Rules(scoringRule: .area, komi: 0)
        let board = Board(2, 2)
        let history = BoardHistory(board, pla: .black, rules: rules)
        history.endAndScoreGameNow(board)
        XCTAssertNil(history.winner)
        XCTAssertEqual(history.finalWhiteMinusBlackScore, 0)

        let result = SelfPlayResult(
            history: history,
            records: [dummyRecord(nextPlayer: .black, nnLen: 2)],
            finalArea: history.getAreaNow(board),
            boardXSize: 2
        )
        let samples = SelfPlayGameGenerator.trainingSamples(from: result, nnLen: 2)
        XCTAssertEqual(samples.count, 1)
        XCTAssertEqual(samples[0].valueTarget, [0.5, 0.5, 0])
        XCTAssertEqual(samples[0].scoreTarget, 0)
        XCTAssertTrue(samples[0].scoreValid)
        XCTAssertTrue(samples[0].ownershipValid)
    }

    func testRecordedGameRoundTripsThroughShard() async throws {
        var settings = SearchSettings()
        settings.leafBatchSize = 1
        settings.numSearchWorkers = 1
        settings.resignEnabled = false
        settings.randomSeed = 7
        let result = try await SelfPlayGameGenerator.playRecording(
            evaluator: TestPolicyEvaluator(nnXLen: 9, nnYLen: 9, preferredLoc: Board.passLoc),
            nnXLen: 9,
            nnYLen: 9,
            precision: .float32,
            configuration: SelfPlayGameConfiguration(
                boardSize: 9,
                visits: 1,
                temperature: 0,
                temperatureMoves: 0,
                rules: .getTrompTaylorish(),
                searchSettings: settings
            ),
            recordTrainingData: true
        )
        // Black passes, white passes -> two recorded positions.
        XCTAssertEqual(result.records.count, 2)
        for record in result.records {
            XCTAssertEqual(record.policyTarget.count, 82)
            XCTAssertEqual(record.policyTarget.reduce(0, +), 1, accuracy: 1e-5)
            XCTAssertEqual(record.spatial.count, 22 * 81)
            XCTAssertEqual(record.global.count, 19)
        }

        let samples = SelfPlayGameGenerator.trainingSamples(from: result, nnLen: 9)
        XCTAssertEqual(samples.count, 2)
        // Empty-board all-pass with komi 7.5 -> white wins; perspective per mover.
        XCTAssertEqual(result.history.winner, .white)
        XCTAssertEqual(samples[0].valueTarget, [0, 1, 0]) // move 0: black to move, black lost
        XCTAssertEqual(samples[1].valueTarget, [1, 0, 0]) // move 1: white to move, white won
        XCTAssertEqual(samples[0].scoreTarget, -7.5)
        XCTAssertEqual(samples[1].scoreTarget, 7.5)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("wp9-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let shardURL = directory.appendingPathComponent("selfplay-7-00000.nngd")
        try TrainingShardWriter.write(samples: samples, nnLen: 9, to: shardURL)

        let reader = try TrainingShardReader(url: shardURL)
        XCTAssertEqual(reader.header.version, 2)
        XCTAssertEqual(reader.header.sampleCount, 2)
        for index in 0 ..< samples.count {
            XCTAssertEqual(try reader.sample(at: index), samples[index])
        }
    }

    private func dummyRecord(nextPlayer: Player, nnLen: Int) -> SelfPlayPositionRecord {
        let area = nnLen * nnLen
        var policy = [Float](repeating: 0, count: area + 1)
        policy[area] = 1 // arbitrary valid distribution (pass); not asserted by the perspective tests
        return SelfPlayPositionRecord(
            spatial: [Float](repeating: 0, count: 22 * area),
            global: [Float](repeating: 0, count: 19),
            policyTarget: policy,
            nextPlayer: nextPlayer
        )
    }
}
