import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// Correctness tests for `Search` (Sources/RinGoEngine/Search.swift), using the b6c96 test model
/// at fp32 and small visit counts (see README.en.md "Verification model" for the fixture path and
/// `KATAGO_MODELS_DIR` seam). Runs at `nnLen = 9` throughout (not the CLI's 19) purely so these
/// stay fast — correctness of the search algorithm doesn't depend on board size.
final class SearchTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"
    private static let nnLen = 9

    // MARK: - Determinism

    /// Fixed position, two independent 128-visit searches, single-worker mode (per the task
    /// spec: multi-worker search is only deterministic modulo floating-point summation order,
    /// since concurrent leaf batches can complete in either order against the non-associative
    /// running-mean backup). Asserts both the chosen move and the exact visit distribution match.
    func testDeterministicRepeatedSearchSingleWorker() async throws {
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
        settings.numSearchWorkers = 1
        settings.leafBatchSize = 8
        let (board, history, nextPlayer) = fixedMidgamePosition()

        func run() async throws -> SearchResult {
            let search = Search(
                evaluator: evaluator,
                modelDesc: desc,
                nnXLen: Self.nnLen,
                nnYLen: Self.nnLen,
                precision: .float32,
                settings: settings,
                board: board,
                history: history,
                nextPlayer: nextPlayer
            )
            return try await search.runSearch(maxVisits: 128)
        }

        let first = try await run()
        let second = try await run()
        XCTAssertEqual(first.bestMove, second.bestMove)
        XCTAssertEqual(first.rootVisits, second.rootVisits)
        XCTAssertEqual(first.visitsByMove.map(\.loc), second.visitsByMove.map(\.loc))
        XCTAssertEqual(first.visitsByMove.map(\.visits), second.visitsByMove.map(\.visits))
    }

    // MARK: - Legality / termination

    /// 30 alternating genmove-equivalents at 48 visits on 9x9: every chosen move must be legal at
    /// the moment it's played, and two consecutive passes must end the game (`BoardHistory`'s
    /// already-verified-parity double-pass rule, exercised end-to-end through `Search`).
    func testAlternatingSearchMovesAllLegalAndDoublePassEndsGame() async throws {
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
        settings.numSearchWorkers = 4
        settings.leafBatchSize = 8

        let board = Board(Self.nnLen, Self.nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        let search = Search(
            evaluator: evaluator,
            modelDesc: desc,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            settings: settings,
            board: board,
            history: history,
            nextPlayer: .black
        )

        var consecutivePasses = 0
        var movesPlayed = 0
        for moveIndex in 0 ..< 30 {
            let mover = history.presumedNextMovePla
            let result = try await search.runSearch(maxVisits: 48)
            let loc = result.bestMove
            XCTAssertTrue(
                history.isLegal(board, loc, mover),
                "move \(moveIndex) (\(mover)) chose an illegal loc \(loc)"
            )
            history.makeBoardMoveAssumeLegal(board, loc, mover)
            try await search.makeMove(loc)
            movesPlayed += 1
            consecutivePasses = loc == Board.passLoc ? consecutivePasses + 1 : 0
            if consecutivePasses >= 2 { break }
        }
        XCTAssertGreaterThan(movesPlayed, 0)

        if consecutivePasses >= 2 {
            XCTAssertTrue(history.isGameFinished, "two consecutive passes should have ended the game")
        } else {
            // The engine didn't choose to pass twice in a row within the move budget above (a
            // perfectly fine, common outcome at 9x9/48 visits) -- independently verify the
            // termination *mechanism* still works from wherever play ended up.
            let firstPasser = history.presumedNextMovePla
            history.makeBoardMoveAssumeLegal(board, Board.passLoc, firstPasser)
            let secondPasser = history.presumedNextMovePla
            history.makeBoardMoveAssumeLegal(board, Board.passLoc, secondPasser)
            XCTAssertTrue(history.isGameFinished, "two consecutive passes should end the game")
        }
    }

    // MARK: - Tactics: atari-capture puzzles

    /// Puzzle 1: a lone white stone in atari with black to move; the only sane move is the
    /// capture.
    func testSolvesSingleStoneAtariCaptureBlackToMove() async throws {
        let center = Loc9.at(4, 4)
        let board = Board(Self.nnLen, Self.nnLen)
        XCTAssertTrue(board.setStonesFailIfNoLibs([
            Move(loc: center, pla: .white),
            Move(loc: Loc9.at(3, 4), pla: .black),
            Move(loc: Loc9.at(5, 4), pla: .black),
            Move(loc: Loc9.at(4, 3), pla: .black),
        ]))
        let capture = Loc9.at(4, 5)
        try await assertSolvesCapture(board: board, mover: .black, expectedMove: capture)
    }

    /// Puzzle 2: the same shape mirrored (black stone in atari, white to move), to exercise the
    /// perspective flip (utility sign convention) independently of puzzle 1.
    func testSolvesSingleStoneAtariCaptureWhiteToMove() async throws {
        let center = Loc9.at(4, 4)
        let board = Board(Self.nnLen, Self.nnLen)
        XCTAssertTrue(board.setStonesFailIfNoLibs([
            Move(loc: center, pla: .black),
            Move(loc: Loc9.at(3, 4), pla: .white),
            Move(loc: Loc9.at(5, 4), pla: .white),
            Move(loc: Loc9.at(4, 3), pla: .white),
        ]))
        let capture = Loc9.at(4, 5)
        try await assertSolvesCapture(board: board, mover: .white, expectedMove: capture)
    }

    /// Puzzle 3: a *two-stone* central group in atari, a structurally distinct shape from the
    /// single-stone puzzles 1/2 (multi-stone capture, five surrounding stones). Deliberately kept
    /// central rather than on the first line: with a strong value net, capturing an already-dead
    /// small group when the game is otherwise decided is *not* the net's move — it prefers the
    /// biggest open point (a first-line "capture" here draws a policy prior near 1e-4, so no amount
    /// of search picks it, which is correct go). A capture the net actually wants is one that is
    /// also the biggest point on the board; here the recapture at the center is exactly that
    /// (policy prior ~0.39, the clear top move), so the search resolves it robustly.
    func testSolvesTwoStoneAtariCaptureCenter() async throws {
        let board = Board(Self.nnLen, Self.nnLen)
        XCTAssertTrue(board.setStonesFailIfNoLibs([
            Move(loc: Loc9.at(4, 4), pla: .white),
            Move(loc: Loc9.at(4, 5), pla: .white),
            Move(loc: Loc9.at(3, 4), pla: .black),
            Move(loc: Loc9.at(5, 4), pla: .black),
            Move(loc: Loc9.at(3, 5), pla: .black),
            Move(loc: Loc9.at(5, 5), pla: .black),
            Move(loc: Loc9.at(4, 6), pla: .black),
        ]))
        let capture = Loc9.at(4, 3)
        try await assertSolvesCapture(board: board, mover: .black, expectedMove: capture)
    }

    private func assertSolvesCapture(
        board: Board,
        mover: Player,
        expectedMove: Loc,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
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
        settings.numSearchWorkers = 4
        settings.leafBatchSize = 8
        let history = BoardHistory(board, pla: mover, rules: .getTrompTaylorish(), encorePhase: 0)
        let search = Search(
            evaluator: evaluator,
            modelDesc: desc,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            settings: settings,
            board: board,
            history: history,
            nextPlayer: mover
        )
        let result = try await search.runSearch(maxVisits: 256)
        XCTAssertEqual(
            result.bestMove,
            expectedMove,
            "expected the capturing move \(Location.toString(expectedMove, board.xSize, board.ySize)), "
                + "got \(Location.toString(result.bestMove, board.xSize, board.ySize))",
            file: file,
            line: line
        )
    }

    // MARK: - Random-symmetry mode determinism (KataGo nnRandomize, design v2 test #4)

    /// `.random` symmetry mode must stay deterministic under a fixed seed. The per-node symmetry
    /// draws come from a Search-owned SplitMix64 stream seeded by `settings.symmetrySeed` (separate
    /// from the root-noise/sampling stream), and are carried on `MiscNNInputParams.symmetry` — so two
    /// fresh single-worker searches with the same seed draw the same sequence and produce
    /// bit-identical move selection AND visit distribution. This is the reproducibility guarantee the
    /// upstream-carry design buys (`preWarm` is not on this path and cannot shift the sequence).
    func testRandomSymmetryModeIsDeterministicUnderFixedSeed() async throws {
        let modelPath = try modelPath()
        let desc = try ModelDesc.loadFromFileMaybeGZipped(modelPath)
        let evaluator = try NNEvaluator(
            desc: desc,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            maxBatchSize: 8,
            symmetry: .random
        )
        var settings = SearchSettings()
        settings.numSearchWorkers = 1
        settings.leafBatchSize = 8
        settings.symmetryMode = .random
        settings.symmetrySeed = 0x1234_5678_9ABC_DEF0
        let (board, history, nextPlayer) = fixedMidgamePosition()

        func run() async throws -> SearchResult {
            let search = Search(
                evaluator: evaluator,
                modelDesc: desc,
                nnXLen: Self.nnLen,
                nnYLen: Self.nnLen,
                precision: .float32,
                settings: settings,
                board: board,
                history: history,
                nextPlayer: nextPlayer
            )
            return try await search.runSearch(maxVisits: 128)
        }

        let first = try await run()
        let second = try await run()
        XCTAssertEqual(first.bestMove, second.bestMove)
        XCTAssertEqual(first.rootVisits, second.rootVisits)
        XCTAssertEqual(first.visitsByMove.map(\.loc), second.visitsByMove.map(\.loc))
        XCTAssertEqual(first.visitsByMove.map(\.visits), second.visitsByMove.map(\.visits))
    }

    // MARK: - Fixtures / helpers

    private enum Loc9 {
        static func at(_ x: Int, _ y: Int) -> Loc {
            Location.getLoc(x, y, SearchTests.nnLen)
        }
    }

    /// A short, deterministic, legal 9x9 opening (corner + side approach moves) used only as "a
    /// fixed position" for the determinism test -- the specific moves don't matter.
    private func fixedMidgamePosition() -> (board: Board, history: BoardHistory, nextPlayer: Player) {
        let board = Board(Self.nnLen, Self.nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        let moves: [(Int, Int, Player)] = [
            (2, 2, .black), (6, 6, .white), (6, 2, .black), (2, 6, .white),
            (4, 4, .black), (4, 2, .white), (3, 3, .black),
        ]
        for (x, y, pla) in moves {
            let loc = Location.getLoc(x, y, Self.nnLen)
            XCTAssertTrue(history.makeBoardMoveTolerant(board, loc, pla))
        }
        return (board, history, history.presumedNextMovePla)
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
