import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// End-to-end tests for `SearchSettings.rootNumSymmetries` (KataGo `rootNumSymmetriesToSample`) —
/// the root-only NN symmetry averaging wired through `Search`. Uses the b6c96 fixture at 9x9 fp32
/// (same seam as `SearchTests`; set `KATAGO_MODELS_DIR`). The core trick: `k = 8` draws ALL eight
/// dihedral symmetries (a partial Fisher-Yates over `0..<8` selecting 8 is a permutation of the
/// whole set), and the mean of a SET is order-independent — so the root's averaged NN output must
/// equal the average of the eight forced-symmetry evaluations regardless of the draw order, giving
/// a deterministic ground truth to check the real search path against.
final class RootSymmetryTests: XCTestCase {
    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"
    private static let nnLen = 9

    /// Fresh root, `k = 8`: the root's averaged NN output equals the average of the eight
    /// forced-symmetry evaluations (order-independent ground truth), AND differs measurably from the
    /// plain identity (symmetry-0) evaluation on an asymmetric position — proving the averaging path
    /// actually ran rather than silently returning a single eval.
    func testFreshRootAverageEqualsAllEightSymmetryMean() async throws {
        let (desc, evaluator) = try await makeEvaluator()
        let (board, history, nextPlayer) = fixedMidgamePosition()

        var settings = SearchSettings()
        settings.numSearchWorkers = 1
        settings.leafBatchSize = 8
        settings.rootNumSymmetries = 8
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
        let result = try await search.runSearch(maxVisits: 1)
        let rootOutput = try XCTUnwrap(result.rootNNOutput)

        let groundTruth = try await averageAllEightSymmetries(
            desc: desc, evaluator: evaluator, board: board, history: history, nextPlayer: nextPlayer
        )
        assertNNOutputsClose(rootOutput, groundTruth, tolerance: 5e-4)

        // Averaging genuinely changed the prior vs a single (identity) eval on this asymmetric board.
        let identity = try await singleSymmetryEval(
            desc: desc, evaluator: evaluator, board: board, history: history, nextPlayer: nextPlayer, symmetry: 0
        )
        XCTAssertGreaterThan(
            legalPolicyL1(rootOutput, identity), 1e-3,
            "k=8 averaged root policy is indistinguishable from a single-symmetry eval — averaging did not run"
        )
    }

    /// `k = 1` (the default): the root's NN output is byte-identical to a plain single-symmetry
    /// (identity) evaluation — the off path must not perturb today's behavior.
    func testK1RootMatchesSingleEval() async throws {
        let (desc, evaluator) = try await makeEvaluator()
        let (board, history, nextPlayer) = fixedMidgamePosition()

        var settings = SearchSettings()
        settings.numSearchWorkers = 1
        settings.leafBatchSize = 8
        // rootNumSymmetries defaults to 1.
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
        let result = try await search.runSearch(maxVisits: 1)
        let rootOutput = try XCTUnwrap(result.rootNNOutput)

        let identity = try await singleSymmetryEval(
            desc: desc, evaluator: evaluator, board: board, history: history, nextPlayer: nextPlayer, symmetry: 0
        )
        assertNNOutputsClose(rootOutput, identity, tolerance: 1e-6)
    }

    /// Reused-root re-average path: after `makeMove` re-roots onto a played child (first evaluated as
    /// a single-symmetry leaf), a fresh `runSearch` with `k = 8` must REPLACE that root's stored NN
    /// output with the eight-symmetry average of the reused position — not leave the stale single
    /// eval. Checks the re-averaged root equals the all-eight ground truth for the reused position
    /// and differs from its single identity eval.
    func testReusedRootIsReaveraged() async throws {
        let (desc, evaluator) = try await makeEvaluator()
        let (board, history, nextPlayer) = fixedMidgamePosition()

        var settings = SearchSettings()
        settings.numSearchWorkers = 1
        settings.leafBatchSize = 8
        settings.rootNumSymmetries = 8
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
        // First search expands the root and several children.
        _ = try await search.runSearch(maxVisits: 48)
        let topChild = await search.topRootChild()
        let top = try XCTUnwrap(topChild, "root expanded no children")
        XCTAssertGreaterThan(top.visits, 0, "reused child must carry visits so the reused-root branch runs")

        // Re-root onto that child (both in our reference position and in the search).
        let reusedBoard = board.copy()
        let reusedHistory = history.copy()
        reusedHistory.makeBoardMoveAssumeLegal(reusedBoard, top.loc, nextPlayer)
        let reusedNextPlayer = nextPlayer.opponent
        try await search.makeMove(top.loc)
        let reusedVisits = await search.currentRootVisits()
        XCTAssertGreaterThan(reusedVisits, 0, "tree reuse did not retain the child subtree")

        // Second search re-averages the reused root.
        let result = try await search.runSearch(maxVisits: 1)
        let rootOutput = try XCTUnwrap(result.rootNNOutput)

        let groundTruth = try await averageAllEightSymmetries(
            desc: desc, evaluator: evaluator,
            board: reusedBoard, history: reusedHistory, nextPlayer: reusedNextPlayer
        )
        assertNNOutputsClose(rootOutput, groundTruth, tolerance: 5e-4)

        let identity = try await singleSymmetryEval(
            desc: desc, evaluator: evaluator,
            board: reusedBoard, history: reusedHistory, nextPlayer: reusedNextPlayer, symmetry: 0
        )
        XCTAssertGreaterThan(
            legalPolicyL1(rootOutput, identity), 1e-3,
            "reused root was not re-averaged — its policy still matches the single-symmetry leaf eval"
        )
    }

    // MARK: - Helpers

    /// Averages the position's eight forced-symmetry evaluations, batched exactly as the search does
    /// (one 8-request `evaluate`), so the per-symmetry numerics match the search's own path.
    private func averageAllEightSymmetries(
        desc _: ModelDesc,
        evaluator: NNEvaluator,
        board: Board,
        history: BoardHistory,
        nextPlayer: Player
    ) async throws -> NNOutput {
        let requests = (0 ..< 8).map { symmetry -> NNRequest in
            var params = MiscNNInputParams()
            params.symmetry = symmetry
            return NNRequest(board: board, history: history, nextPlayer: nextPlayer, params: params)
        }
        let outputs = try await evaluator.evaluate(requests)
        return NNOutput.averaged(outputs)
    }

    private func singleSymmetryEval(
        desc _: ModelDesc,
        evaluator: NNEvaluator,
        board: Board,
        history: BoardHistory,
        nextPlayer: Player,
        symmetry: Int
    ) async throws -> NNOutput {
        var params = MiscNNInputParams()
        params.symmetry = symmetry
        return try await evaluator.evaluate(NNRequest(
            board: board, history: history, nextPlayer: nextPlayer, params: params
        ))
    }

    /// L1 distance between two policies over positions legal in both (illegal `< 0` entries skipped).
    private func legalPolicyL1(_ a: NNOutput, _ b: NNOutput) -> Float {
        var sum: Float = 0
        for (pa, pb) in zip(a.policyProbs, b.policyProbs) where pa >= 0 && pb >= 0 {
            sum += abs(pa - pb)
        }
        return sum
    }

    private func assertNNOutputsClose(
        _ actual: NNOutput,
        _ expected: NNOutput,
        tolerance: Float,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.whiteWinProb, expected.whiteWinProb, accuracy: tolerance, "win", file: file, line: line)
        XCTAssertEqual(
            actual.whiteLossProb,
            expected.whiteLossProb,
            accuracy: tolerance,
            "loss",
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.whiteScoreMean, expected.whiteScoreMean, accuracy: tolerance * 40, "scoreMean", file: file,
            line: line
        )
        XCTAssertEqual(actual.whiteLead, expected.whiteLead, accuracy: tolerance * 40, "lead", file: file, line: line)
        XCTAssertEqual(actual.policyProbs.count, expected.policyProbs.count, "policy size", file: file, line: line)
        for (a, e) in zip(actual.policyProbs, expected.policyProbs) {
            if a < 0 || e < 0 {
                XCTAssertEqual(a, e, "policy legality mask", file: file, line: line)
            } else {
                XCTAssertEqual(a, e, accuracy: tolerance, "policy prob", file: file, line: line)
            }
        }
        switch (actual.whiteOwnerMap, expected.whiteOwnerMap) {
        case let (a?, e?):
            for (av, ev) in zip(a, e) {
                XCTAssertEqual(av, ev, accuracy: tolerance, "ownership", file: file, line: line)
            }
        case (nil, nil):
            break
        default:
            XCTFail("ownership presence mismatch", file: file, line: line)
        }
    }

    private func makeEvaluator() async throws -> (ModelDesc, NNEvaluator) {
        let path = try modelPath()
        let desc = try ModelDesc.loadFromFileMaybeGZipped(path)
        let evaluator = try NNEvaluator(
            desc: desc,
            nnXLen: Self.nnLen,
            nnYLen: Self.nnLen,
            precision: .float32,
            maxBatchSize: 8
        )
        return (desc, evaluator)
    }

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
