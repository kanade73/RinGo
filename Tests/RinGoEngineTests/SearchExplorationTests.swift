import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

final class SearchExplorationTests: XCTestCase {
    func testExplorationDefaultsAreOff() {
        let settings = SearchSettings()
        XCTAssertEqual(settings.rootDirichletAlpha, 0)
        XCTAssertEqual(settings.rootNoiseWeight, 0)
        XCTAssertEqual(settings.selectionTemperature, 0)
        XCTAssertEqual(settings.randomSeed, 0)
    }

    func testTemperatureZeroAlwaysUsesStableArgmax() {
        for seed in 0 ..< 100 {
            var random = SearchRandom(seed: UInt64(seed))
            XCTAssertEqual(
                SearchSampling.sampleVisitIndex(visits: [4, 12, 12, 3], temperature: 0, random: &random),
                1
            )
        }
    }

    func testHighTemperatureSpreadsSamplesAcrossChildren() throws {
        var random = SearchRandom(seed: 123)
        var counts = [Int](repeating: 0, count: 3)
        for _ in 0 ..< 600 {
            let index = try XCTUnwrap(SearchSampling.sampleVisitIndex(
                visits: [100, 30, 10],
                temperature: 10,
                random: &random
            ))
            counts[index] += 1
        }
        XCTAssertTrue(counts.allSatisfy { $0 > 120 }, "high temperature should visibly spread: \(counts)")
    }

    func testTemperatureSamplingIsSeedReproducibleAndSeedSensitive() {
        func sequence(seed: UInt64) -> [Int] {
            var random = SearchRandom(seed: seed)
            return (0 ..< 32).map { _ in
                SearchSampling.sampleVisitIndex(
                    visits: [50, 25, 10, 5],
                    temperature: 0.9,
                    random: &random
                )!
            }
        }
        XCTAssertEqual(sequence(seed: 99), sequence(seed: 99))
        XCTAssertNotEqual(sequence(seed: 99), sequence(seed: 100))
    }

    func testDirichletMixingIsNormalizedAndObeysWeightFormula() {
        let priors = [2.0, 3.0, 5.0]
        var random = SearchRandom(seed: 7)
        let mixed = SearchSampling.mixDirichlet(
            priors: priors,
            alpha: 0.15,
            weight: 0.25,
            random: &random
        )
        XCTAssertEqual(mixed.reduce(0, +), 1, accuracy: 1e-12)

        let normalized = [0.2, 0.3, 0.5]
        let recoveredNoise = zip(mixed, normalized).map { value, prior in
            (value - 0.75 * prior) / 0.25
        }
        XCTAssertEqual(recoveredNoise.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertTrue(recoveredNoise.allSatisfy { $0 >= 0 })
    }

    func testDirichletMixingIsSeedReproducibleAndSeedSensitive() {
        func mix(seed: UInt64) -> [Double] {
            var random = SearchRandom(seed: seed)
            return SearchSampling.mixDirichlet(
                priors: [0.1, 0.2, 0.3, 0.4],
                alpha: 0.15,
                weight: 0.25,
                random: &random
            )
        }
        XCTAssertEqual(mix(seed: 1), mix(seed: 1))
        XCTAssertNotEqual(mix(seed: 1), mix(seed: 2))
    }

    func testExplicitOffSettingsMatchDefaultsBitForBitWithStubEvaluator() async throws {
        let board = Board(9, 9)
        let rules = Rules.getTrompTaylorish()
        let history = BoardHistory(board, pla: .black, rules: rules)
        let defaults = SearchSettings()
        var explicitOff = defaults
        explicitOff.rootDirichletAlpha = 0
        explicitOff.rootNoiseWeight = 0
        explicitOff.selectionTemperature = 0
        explicitOff.randomSeed = UInt64.max

        func run(_ settings: SearchSettings) async throws -> SearchResult {
            let search = Search(
                evaluator: UniformStubEvaluator(nnLen: 9),
                nnXLen: 9,
                nnYLen: 9,
                precision: .float32,
                settings: settings,
                board: board,
                history: history,
                nextPlayer: .black,
                fp32FallbackFactory: nil
            )
            return try await search.runSearch(maxVisits: 32)
        }

        let baseline = try await run(defaults)
        let off = try await run(explicitOff)
        XCTAssertEqual(off.bestMove, baseline.bestMove)
        XCTAssertEqual(off.rootVisits, baseline.rootVisits)
        XCTAssertEqual(off.visitsAchieved, baseline.visitsAchieved)
        XCTAssertEqual(off.whiteWinProb, baseline.whiteWinProb)
        XCTAssertEqual(off.whiteScoreMean, baseline.whiteScoreMean)
        XCTAssertEqual(off.visitsByMove.map(\.loc), baseline.visitsByMove.map(\.loc))
        XCTAssertEqual(off.visitsByMove.map(\.visits), baseline.visitsByMove.map(\.visits))
    }

    /// Regression for a crash found running real self-play (not reproducible via any CPU-only
    /// test that predates root noise, since it needs noise ON + tree reuse across >=2 moves):
    /// `prepareRootNoiseIfNeeded` used to unconditionally overwrite `root.untriedMoves` with a
    /// fresh full legal-move scan. After `makeMove` reuses a former child as the new root — and
    /// that child was itself expanded (gained children of its own) while being explored during
    /// the previous move's search — that overwrite reintroduced already-expanded locations into
    /// the untried queue, so a later visit called `addChild` a second time for the same `loc`.
    /// `root.children` (and therefore `visitsByMove`, and anything else that builds a `[Loc: _]`
    /// dictionary from it, e.g. conservativePass's cleanup scan) then had a duplicate `loc`,
    /// which crashed the process the first time such a dictionary got built. This test doesn't
    /// need to reach that specific dictionary-building call site — the duplicate in `children`
    /// itself is the bug, so asserting `visitsByMove` has no repeated `loc` after noisy tree
    /// reuse is a direct, sufficient regression check.
    func testRootNoiseWithTreeReuseNeverDuplicatesAChild() async throws {
        let board = Board(9, 9)
        let rules = Rules.getTrompTaylorish()
        let history = BoardHistory(board, pla: .black, rules: rules)
        var settings = SearchSettings()
        settings.rootDirichletAlpha = 0.15
        settings.rootNoiseWeight = 0.25
        settings.randomSeed = 7

        let search = Search(
            evaluator: UniformStubEvaluator(nnLen: 9),
            nnXLen: 9,
            nnYLen: 9,
            precision: .float32,
            settings: settings,
            board: board,
            history: history,
            nextPlayer: .black,
            fp32FallbackFactory: nil
        )

        // Enough visits that the chosen move's own subtree gets expanded (has grandchildren)
        // before it becomes the new root via tree reuse.
        let first = try await search.runSearch(maxVisits: 200)
        try await search.makeMove(first.bestMove)
        // `rootPolicyProbs` was reset to nil by `makeMove`, so this second call re-enters
        // `prepareRootNoiseIfNeeded` on the reused (already-partially-expanded) root — exactly
        // the previously-crashing path.
        let second = try await search.runSearch(maxVisits: 100)

        let locs = second.visitsByMove.map(\.loc)
        XCTAssertEqual(locs.count, Set(locs).count, "root.children must never contain a duplicate loc")
    }
}

private actor UniformStubEvaluator: NNEvaluating {
    let nnLen: Int

    init(nnLen: Int) {
        self.nnLen = nnLen
    }

    func evaluate(_ requests: [NNRequest]) -> [NNOutput] {
        requests.map(output)
    }

    private func output(_ request: NNRequest) -> NNOutput {
        let policySize = nnLen * nnLen + 1
        var policy = [Float](repeating: -1, count: policySize)
        var legalPositions = [Int]()
        for position in 0 ..< policySize {
            let loc = NNPos.posToLoc(
                position,
                request.board.xSize,
                request.board.ySize,
                nnLen,
                nnLen
            )
            if loc != Board.nullLoc, request.history.isLegal(request.board, loc, request.nextPlayer) {
                legalPositions.append(position)
            }
        }
        for position in legalPositions {
            policy[position] = 1 / Float(legalPositions.count)
        }
        return NNOutput(
            nnHash: Hash128(),
            whiteWinProb: 0.5,
            whiteLossProb: 0.5,
            whiteNoResultProb: 0,
            whiteScoreMean: 0,
            whiteScoreMeanSq: 1,
            whiteLead: 0,
            varTimeLeft: 0,
            shorttermWinlossError: 0,
            shorttermScoreError: 0,
            policyProbs: policy,
            policyOptimismUsed: 0,
            nnXLen: nnLen,
            nnYLen: nnLen,
            whiteOwnerMap: nil
        )
    }
}
