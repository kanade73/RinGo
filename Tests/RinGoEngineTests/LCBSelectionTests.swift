import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// WP-11 LCB (Lower Confidence Bound) final-move selection. The first group tests the pure,
/// model-free numerics (`SearchLCB` + `SearchNode.accumulateLeaf`) on synthetic child stats, so it
/// runs fast without the NN fixtures. The final test is a host-gated end-to-end check that a real
/// `runSearch` picks a legal LCB move and that the off-path stays visit-argmax.
final class LCBSelectionTests: XCTestCase {
    // KataGo GTP defaults (searchparams.cpp) and this port's utility factors (SearchSettings.swift:
    // winLoss 1.0 + staticScore 0.1 + dynamicScore 0.3).
    private static let utilityRangeRadius = 1.4
    private static let lcbStdevs = 5.0
    private static let minVisitProp = 0.15
    private static let nnLen = 9

    private static func loc(_ x: Int, _ y: Int) -> Loc {
        Location.getLoc(x, y, nnLen)
    }

    // MARK: - Variance tracking (backup accumulation)

    /// `SearchNode.accumulateLeaf` (the single source of truth for backup's mean/second-moment
    /// update, shared with `applyBackup`) must produce the exact running mean and mean-of-squares,
    /// so LCB's value variance `utilitySqAvg - utilityAvg^2` is correct.
    func testAccumulateLeafTracksMeanAndVariance() {
        let node = makeNode()
        let values = [0.5, -0.5, 0.5, -0.5, 0.3]
        for value in values {
            node.accumulateLeaf(utility: value, utilitySq: value * value, scoreMean: 0, scoreMeanSq: 0, winLoss: 0)
        }
        let mean = values.reduce(0, +) / Double(values.count)
        let meanSq = values.map { $0 * $0 }.reduce(0, +) / Double(values.count)
        XCTAssertEqual(node.visits, Int64(values.count))
        XCTAssertEqual(node.weightSum, Double(values.count), accuracy: 1e-12)
        XCTAssertEqual(node.utilityAvg, mean, accuracy: 1e-12)
        XCTAssertEqual(node.utilitySqAvg, meanSq, accuracy: 1e-12)
        // Variance = E[x^2] - E[x]^2, non-negative and matching the population variance.
        let variance = node.utilitySqAvg - node.utilityAvg * node.utilityAvg
        let populationVariance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(values.count)
        XCTAssertEqual(variance, populationVariance, accuracy: 1e-12)
        XCTAssertGreaterThan(variance, 0)
    }

    /// Antisymmetric values around zero: mean 0, so E[x^2] equals the variance exactly.
    func testAccumulateLeafZeroMeanVariance() {
        let node = makeNode()
        for value in [1.0, 0.0, -1.0] {
            node.accumulateLeaf(utility: value, utilitySq: value * value, scoreMean: 0, scoreMeanSq: 0, winLoss: 0)
        }
        XCTAssertEqual(node.utilityAvg, 0.0, accuracy: 1e-12)
        XCTAssertEqual(node.utilitySqAvg, 2.0 / 3.0, accuracy: 1e-12)
    }

    // MARK: - LCB radius / perspective

    /// A child with no visits gets the maximum radius and minimum LCB, so it can never win.
    func testZeroVisitChildGetsMaxRadius() {
        let (lcb, radius) = SearchLCB.selfUtilityLCBAndRadius(
            visits: 0, utilityAvg: 0.9, utilitySqAvg: 0.81,
            moverIsWhite: true, utilityRangeRadius: Self.utilityRangeRadius, lcbStdevs: Self.lcbStdevs
        )
        XCTAssertEqual(radius, 2.0 * Self.utilityRangeRadius * Self.lcbStdevs, accuracy: 1e-12)
        XCTAssertEqual(lcb, -radius, accuracy: 1e-12)
    }

    /// More visits at the same value/variance shrinks the radius and raises the LCB.
    func testMoreVisitsRaisesLCB() {
        func lcb(visits: Int64) -> Double {
            SearchLCB.selfUtilityLCBAndRadius(
                visits: visits, utilityAvg: 0.2, utilitySqAvg: 0.2 * 0.2 + 0.05,
                moverIsWhite: true, utilityRangeRadius: Self.utilityRangeRadius, lcbStdevs: Self.lcbStdevs
            ).lcb
        }
        XCTAssertLessThan(lcb(visits: 10), lcb(visits: 100))
        XCTAssertLessThan(lcb(visits: 100), lcb(visits: 1000))
    }

    /// The self-utility is signed from the mover's perspective: for a positive (white-favoring)
    /// utility, black-to-move sees it as bad, so black's LCB is below white's.
    func testPerspectiveFlipsSign() {
        func lcb(white: Bool) -> Double {
            SearchLCB.selfUtilityLCBAndRadius(
                visits: 50, utilityAvg: 0.3, utilitySqAvg: 0.3 * 0.3 + 0.01,
                moverIsWhite: white, utilityRangeRadius: Self.utilityRangeRadius, lcbStdevs: Self.lcbStdevs
            ).lcb
        }
        XCTAssertGreaterThan(lcb(white: true), lcb(white: false))
    }

    // MARK: - Selection: override / refusal / equivalence

    /// A lower-visit child with a clearly better (higher, low-variance) value wins over the
    /// most-visited child once it clears the `minVisitPropForLCB` visit floor.
    func testLCBOverridesMaxVisitsWhenLowerVisitChildIsClearlyBetter() {
        let mostVisited = Self.loc(3, 3) // 100 visits, mediocre + noisy value
        let underdog = Self.loc(5, 5) // 40 visits (>= 0.15 * 100), high + confident value
        let children = [
            SearchLCB.ChildStat(loc: mostVisited, visits: 100, utilityAvg: 0.10, utilitySqAvg: 0.30, policyProb: 0.30),
            SearchLCB.ChildStat(
                loc: underdog,
                visits: 40,
                utilityAvg: 0.30,
                utilitySqAvg: 0.30 * 0.30 + 1e-4,
                policyProb: 0.20
            ),
        ]
        let selected = SearchLCB.selectedMove(
            children: children, moverIsWhite: true,
            utilityRangeRadius: Self.utilityRangeRadius, lcbStdevs: Self.lcbStdevs,
            minVisitPropForLCB: Self.minVisitProp
        )
        XCTAssertEqual(selected, underdog, "the confident higher-value lower-visit child should win LCB")
        // Sanity: plain max-visits selection would have picked the other child.
        XCTAssertEqual(children.max { $0.visits < $1.visits }?.loc, mostVisited)
    }

    /// The same better-value underdog is REFUSED when its visits fall below
    /// `minVisitPropForLCB * maxVisits`, so selection falls back to the most-visited child.
    func testLCBRefusesOverrideBelowMinVisitProp() {
        let mostVisited = Self.loc(3, 3) // 100 visits
        let underdog = Self.loc(5, 5) // 10 visits < 0.15 * 100 = 15 -> ineligible
        let children = [
            SearchLCB.ChildStat(loc: mostVisited, visits: 100, utilityAvg: 0.10, utilitySqAvg: 0.30, policyProb: 0.30),
            SearchLCB.ChildStat(
                loc: underdog,
                visits: 10,
                utilityAvg: 0.30,
                utilitySqAvg: 0.30 * 0.30 + 1e-4,
                policyProb: 0.20
            ),
        ]
        let selected = SearchLCB.selectedMove(
            children: children, moverIsWhite: true,
            utilityRangeRadius: Self.utilityRangeRadius, lcbStdevs: Self.lcbStdevs,
            minVisitPropForLCB: Self.minVisitProp
        )
        XCTAssertEqual(selected, mostVisited, "an ineligible (too-few-visits) child must not override")
    }

    /// When every child shares the same value and variance, the highest-visit child has the
    /// tightest bound and therefore the best LCB, so LCB reduces to plain max-visits selection.
    func testEqualValuesReduceToMaxVisits() {
        let mostVisited = Self.loc(3, 3)
        let children = [
            SearchLCB.ChildStat(
                loc: Self.loc(5, 5),
                visits: 30,
                utilityAvg: 0.2,
                utilitySqAvg: 0.2 * 0.2 + 0.05,
                policyProb: 0.2
            ),
            SearchLCB.ChildStat(
                loc: mostVisited,
                visits: 100,
                utilityAvg: 0.2,
                utilitySqAvg: 0.2 * 0.2 + 0.05,
                policyProb: 0.3
            ),
            SearchLCB.ChildStat(
                loc: Self.loc(2, 2),
                visits: 60,
                utilityAvg: 0.2,
                utilitySqAvg: 0.2 * 0.2 + 0.05,
                policyProb: 0.25
            ),
        ]
        let selected = SearchLCB.selectedMove(
            children: children, moverIsWhite: true,
            utilityRangeRadius: Self.utilityRangeRadius, lcbStdevs: Self.lcbStdevs,
            minVisitPropForLCB: Self.minVisitProp
        )
        XCTAssertEqual(selected, mostVisited)
    }

    /// Black to move: LCB maximizes the bound in black's favor, i.e. the child with the most
    /// negative (best-for-black) white-perspective utility, mirroring the perspective handling.
    func testBlackToMovePicksBestForBlack() {
        let bestForBlack = Self.loc(5, 5) // very negative white utility = great for black
        let children = [
            SearchLCB.ChildStat(
                loc: Self.loc(3, 3),
                visits: 100,
                utilityAvg: 0.10,
                utilitySqAvg: 0.10 * 0.10 + 1e-4,
                policyProb: 0.30
            ),
            SearchLCB.ChildStat(
                loc: bestForBlack,
                visits: 80,
                utilityAvg: -0.40,
                utilitySqAvg: 0.40 * 0.40 + 1e-4,
                policyProb: 0.25
            ),
        ]
        let selected = SearchLCB.selectedMove(
            children: children, moverIsWhite: false,
            utilityRangeRadius: Self.utilityRangeRadius, lcbStdevs: Self.lcbStdevs,
            minVisitPropForLCB: Self.minVisitProp
        )
        XCTAssertEqual(selected, bestForBlack)
    }

    /// No children -> nil (caller falls back to visit selection).
    func testEmptyChildrenReturnsNil() {
        XCTAssertNil(SearchLCB.selectedMove(
            children: [], moverIsWhite: true,
            utilityRangeRadius: Self.utilityRangeRadius, lcbStdevs: Self.lcbStdevs,
            minVisitPropForLCB: Self.minVisitProp
        ))
    }

    // MARK: - End-to-end (host-gated on NN fixtures)

    /// A real 128-visit search: with LCB off the chosen move is the visit-argmax (bit-identical to
    /// pre-WP-11 behavior); with LCB on the chosen move is legal and among the searched children.
    func testRunSearchLCBOnOffEndToEnd() async throws {
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

        let board = Board(Self.nnLen, Self.nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        func freshSearch() -> Search {
            Search(
                evaluator: evaluator, modelDesc: desc, nnXLen: Self.nnLen, nnYLen: Self.nnLen,
                precision: .float32, settings: settings, board: board, history: history, nextPlayer: .black
            )
        }

        let off = try await freshSearch().runSearch(maxVisits: 128, useLcbForSelection: false)
        let on = try await freshSearch().runSearch(maxVisits: 128, useLcbForSelection: true)

        // Off-mode selection is exactly the most-visited child (visitsByMove is visit-descending).
        XCTAssertEqual(off.bestMove, off.visitsByMove.first?.loc)
        // On-mode move must be legal and one the search actually expanded (or a legal pass).
        XCTAssertTrue(history.isLegal(board, on.bestMove, .black))
        let searchedLocs = Set(on.visitsByMove.map(\.loc))
        XCTAssertTrue(on.bestMove == Board.passLoc || searchedLocs.contains(on.bestMove))
        // The recorded visit distribution is unaffected by LCB (same tree, same visits).
        XCTAssertEqual(on.visitsByMove.map(\.visits), off.visitsByMove.map(\.visits))
    }

    // MARK: - Fixtures

    private func makeNode() -> SearchNode {
        let board = Board(Self.nnLen, Self.nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        return SearchNode(board: board, history: history, nextPla: .black)
    }

    private static let defaultModelsDirectory =
        "../katago-origin/KataGo/cpp/tests/models"
    private static let modelFile = "g170-b6c96-s175395328-d26788732.bin.gz"

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
