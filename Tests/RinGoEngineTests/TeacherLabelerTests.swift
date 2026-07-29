import Foundation
import RinGoCore
@testable import RinGoEngine
import XCTest

/// Unit tests for the teacher-distillation target conversion (`TeacherLabeler`) and the
/// teacher-mode data pipeline (`TrainingDataPipeline.generate(_:teacher:teacherBatchSize:)`).
/// These use hand-built `NNOutput`s and a deterministic stub teacher, so no Metal/MLX device is
/// needed -- the producer/consumer perspective mapping and the pipeline wiring are exercised on
/// pure Swift values.
final class TeacherLabelerTests: XCTestCase {
    private func output(
        policyProbs: [Float],
        whiteWin: Float,
        whiteLoss: Float,
        whiteNoResult: Float,
        whiteScoreMean: Float,
        whiteOwnerMap: [Float]?
    ) -> NNOutput {
        NNOutput(
            nnHash: Hash128(0, 0),
            whiteWinProb: whiteWin,
            whiteLossProb: whiteLoss,
            whiteNoResultProb: whiteNoResult,
            whiteScoreMean: whiteScoreMean,
            whiteScoreMeanSq: 0,
            whiteLead: 0,
            varTimeLeft: -1,
            shorttermWinlossError: -1,
            shorttermScoreError: -1,
            policyProbs: policyProbs,
            policyOptimismUsed: 0,
            nnXLen: 3,
            nnYLen: 3,
            whiteOwnerMap: whiteOwnerMap
        )
    }

    /// The dense policy must copy legal probabilities through and set illegal (the `-1` sentinel)
    /// entries to EXACTLY 0, leaving a distribution that still sums to ~1 over the legal moves.
    func testPolicyMasksIllegalToZeroAndPreservesLegalSum() {
        // 3x3: area 9, +1 pass = 10 entries. Illegal at 1 and 4; legal sum = 1.0.
        var probs = [Float](repeating: -1, count: 10)
        probs[0] = 0.5
        probs[2] = 0.2
        probs[3] = 0.1
        probs[9] = 0.2 // pass
        let targets = TeacherLabeler.targets(
            from: output(
                policyProbs: probs, whiteWin: 0.5, whiteLoss: 0.4, whiteNoResult: 0.1,
                whiteScoreMean: 0, whiteOwnerMap: nil
            ),
            nextPlayer: .black, nnLen: 3
        )
        XCTAssertEqual(targets.policy.count, 10)
        XCTAssertEqual(targets.policy[1], 0)
        XCTAssertEqual(targets.policy[4], 0)
        XCTAssertEqual(targets.policy[5], 0)
        XCTAssertEqual(targets.policy[0], 0.5)
        XCTAssertEqual(targets.policy[9], 0.2)
        XCTAssertEqual(targets.policy.reduce(0, +), 1.0, accuracy: 1e-6)
        XCTAssertTrue(targets.policy.allSatisfy { $0 >= 0 })
    }

    /// The value target is `[win, loss, noResult]` from the SIDE-TO-MOVE perspective. This test is
    /// written so it fails loudly if the win/loss channels were ever swapped in either direction:
    /// white-to-move keeps white's probabilities; black-to-move swaps win<->loss (white's win is
    /// black's loss) and leaves noResult untouched.
    func testValueCategoryMappingRespectsSideToMovePerspective() {
        let out = output(
            policyProbs: [Float](repeating: 0.1, count: 10),
            whiteWin: 0.7, whiteLoss: 0.2, whiteNoResult: 0.1,
            whiteScoreMean: 0, whiteOwnerMap: nil
        )
        let whiteToMove = TeacherLabeler.targets(from: out, nextPlayer: .white, nnLen: 3)
        XCTAssertEqual(whiteToMove.value, [0.7, 0.2, 0.1])
        let blackToMove = TeacherLabeler.targets(from: out, nextPlayer: .black, nnLen: 3)
        XCTAssertEqual(blackToMove.value, [0.2, 0.7, 0.1])
        // noResult is perspective-independent; win/loss are genuinely swapped, not merely relabeled.
        XCTAssertEqual(whiteToMove.value[2], blackToMove.value[2])
        XCTAssertNotEqual(whiteToMove.value[0], blackToMove.value[0])
    }

    /// Score is the teacher's white-perspective scoremean reprojected to the side to move (negated
    /// for black). The training loss divides `scoreTarget` by 20, so the stored value stays in
    /// points, matching the teacherless path.
    func testScoreMappingNegatesForBlackToMove() {
        let out = output(
            policyProbs: [Float](repeating: 0.1, count: 10),
            whiteWin: 0.5, whiteLoss: 0.4, whiteNoResult: 0.1,
            whiteScoreMean: 5.0, whiteOwnerMap: nil
        )
        XCTAssertEqual(TeacherLabeler.targets(from: out, nextPlayer: .white, nnLen: 3).score, 5.0)
        XCTAssertEqual(TeacherLabeler.targets(from: out, nextPlayer: .black, nnLen: 3).score, -5.0)
    }

    /// Ownership quantizes the teacher's continuous white-perspective tanh output into the `{-1,0,1}`
    /// Int8 alphabet v2 uses. The roundtrip constraint is the one the training loss imposes:
    /// `Trainer.batch` reads the Int8 straight into a Float and `LossComputer` maps it via
    /// `(v + 1) / 2` into a [0,1] BCE target, so the stored value MUST be in `{-1,0,1}` and the
    /// decoded probability must stay within the nearest-rounding error (0.25 in probability space)
    /// of the teacher's continuous value.
    func testOwnershipQuantizesToTernaryAndRoundtripsWithinError() {
        // white-perspective ownership for a 3x3 board (9 points).
        let owner: [Float] = [0.95, -0.95, 0.10, -0.40, 0.60, -0.60, 0.49, -0.49, 0.0]
        let out = output(
            policyProbs: [Float](repeating: 0.1, count: 10),
            whiteWin: 0.5, whiteLoss: 0.4, whiteNoResult: 0.1,
            whiteScoreMean: 0, whiteOwnerMap: owner
        )
        // white to move: side-to-move ownership == white-perspective ownership directly.
        let white = TeacherLabeler.targets(from: out, nextPlayer: .white, nnLen: 3).ownership
        XCTAssertEqual(white, [1, -1, 0, 0, 1, -1, 0, 0, 0])
        XCTAssertTrue(white.allSatisfy { $0 == -1 || $0 == 0 || $0 == 1 })
        for (index, quantized) in white.enumerated() {
            // Decode exactly as LossComputer does, and compare to the teacher's true probability.
            let decodedProbability = (Float(quantized) + 1) / 2
            let trueProbability = (owner[index] + 1) / 2
            XCTAssertLessThanOrEqual(abs(decodedProbability - trueProbability), 0.25)
        }
        // black to move: every point flips sign, so ownership is the negation of the white case.
        let black = TeacherLabeler.targets(from: out, nextPlayer: .black, nnLen: 3).ownership
        XCTAssertEqual(black, white.map { Int8(-$0) })
    }

    // MARK: - Pipeline (stub teacher)

    /// Returns a fixed, deterministic `NNOutput` for every request, so the distillation pipeline can
    /// be exercised end-to-end without a real network / Metal device.
    private actor StubTeacher: NNEvaluating {
        let template: NNOutput
        init(template: NNOutput) {
            self.template = template
        }

        func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput] {
            requests.map { _ in template }
        }
    }

    func testTeacherModePipelineReplacesTargetsAndMarksAllValid() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let outputDir = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // A resignation-ending game: teacherless this would leave score/ownership INVALID, but the
        // teacher can label any position, so every distilled sample must come out valid.
        try Data("(;SZ[9]KM[7.5]RE[W+R];B[ee];W[ff];B[dd];W[cc];B[gg])".utf8)
            .write(to: input.appendingPathComponent("a.sgf"))

        // Uniform soft policy over all 82 moves (top prob 1/82 << 1, i.e. NOT one-hot), plus a
        // distinctive value/score/ownership so we can trace the perspective mapping per position.
        let uniform = [Float](repeating: 1.0 / 82.0, count: 82)
        let template = output3x3ReplacementIsIrrelevant(
            policyProbs: uniform, whiteWin: 0.7, whiteLoss: 0.2, whiteNoResult: 0.1,
            whiteScoreMean: 5.0, whiteOwnerMap: [Float](repeating: 0.9, count: 81)
        )
        let teacher = StubTeacher(template: template)

        let options = TrainingDataOptions(
            sgfDirectory: input, outputDirectory: outputDir, nnLen: 9, shardSize: 64,
            rules: chineseRules(), minimumMoves: 2, symmetries: 0
        )
        // teacherBatchSize 2 forces multiple drains (5 positions -> 2, 2, 1) so ordering across
        // buffer flushes is covered.
        let summary = try await TrainingDataPipeline.generate(options, teacher: teacher, teacherBatchSize: 2)
        XCTAssertEqual(summary.gamesUsed, 1)
        XCTAssertEqual(summary.totalSamples, 5)
        XCTAssertEqual(summary.teacherPositionsLabeled, 5)

        let reader = try TrainingShardReader(url: outputDir.appendingPathComponent("shard-00000.nngd"))
        XCTAssertEqual(reader.header.sampleCount, 5)
        for index in 0 ..< 5 {
            let sample = try reader.sample(at: index)
            // Soft policy: sums to ~1, and NOT one-hot (top prob well below 1).
            XCTAssertEqual(sample.policyTarget.reduce(0, +), 1.0, accuracy: 1e-4)
            XCTAssertLessThan(sample.policyTarget.max() ?? 1, 0.5)
            // Teacher labels are trustworthy regardless of game ending -> always valid.
            XCTAssertTrue(sample.scoreValid)
            XCTAssertTrue(sample.ownershipValid)
            // noResult channel is perspective-independent.
            XCTAssertEqual(sample.valueTarget[2], 0.1, accuracy: 1e-6)
            XCTAssertEqual(abs(sample.scoreTarget), 5.0, accuracy: 1e-4)
            XCTAssertTrue(sample.ownershipTarget.allSatisfy { $0 == 1 || $0 == -1 })
        }
        // Position 0 is black to move (B plays first): win/loss swap, score negates, ownership flips.
        let first = try reader.sample(at: 0)
        XCTAssertEqual(first.valueTarget, [0.2, 0.7, 0.1])
        XCTAssertEqual(first.scoreTarget, -5.0, accuracy: 1e-4)
        XCTAssertTrue(first.ownershipTarget.allSatisfy { $0 == -1 })
        // Position 1 is white to move: white's own probabilities, positive score, +1 ownership.
        let second = try reader.sample(at: 1)
        XCTAssertEqual(second.valueTarget, [0.7, 0.2, 0.1])
        XCTAssertEqual(second.scoreTarget, 5.0, accuracy: 1e-4)
        XCTAssertTrue(second.ownershipTarget.allSatisfy { $0 == 1 })
        // (makedata-config.json is written by MakeDataCommand, not the pipeline, so it is verified
        // in the real-model smoke run, not here.)
    }

    /// The `nnLen` of the template does not matter for the pipeline test (the stub ignores the
    /// request and returns a fixed 9x9-sized output); this helper just keeps the call site readable.
    private func output3x3ReplacementIsIrrelevant(
        policyProbs: [Float], whiteWin: Float, whiteLoss: Float, whiteNoResult: Float,
        whiteScoreMean: Float, whiteOwnerMap: [Float]?
    ) -> NNOutput {
        NNOutput(
            nnHash: Hash128(0, 0), whiteWinProb: whiteWin, whiteLossProb: whiteLoss,
            whiteNoResultProb: whiteNoResult, whiteScoreMean: whiteScoreMean, whiteScoreMeanSq: 0,
            whiteLead: 0, varTimeLeft: -1, shorttermWinlossError: -1, shorttermScoreError: -1,
            policyProbs: policyProbs, policyOptimismUsed: 0, nnXLen: 9, nnYLen: 9,
            whiteOwnerMap: whiteOwnerMap
        )
    }

    // MARK: - Serial vs pipelined byte-identity (WP-4b)

    /// A stateless, deterministic teacher whose output is a PURE function of the request's prefilled
    /// features, so the overlapped pipeline and the serial path receive identical per-position labels
    /// regardless of how batches are scheduled. Any divergence in sample order, target assignment, or
    /// shard boundary between the two paths therefore shows up as a byte difference.
    private actor PositionSensitiveStub: NNEvaluating {
        func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput] {
            requests.map { request in
                let spatialSum = request.prefilled?.spatial.prefix(64).reduce(0, +) ?? 0
                // A bounded, position-dependent signal that drives every target field.
                let signal = spatialSum.truncatingRemainder(dividingBy: 11)
                let win = min(0.95, max(0.05, 0.5 + signal * 0.03))
                let loss = max(0.01, 1 - win - 0.02)
                var policy = [Float](repeating: 1.0 / 82.0, count: 82)
                policy[Int(abs(signal).rounded()) % 82] += 0.01
                let ownerValue: Float = signal >= 0 ? 0.9 : -0.9
                return NNOutput(
                    nnHash: Hash128(0, 0),
                    whiteWinProb: win, whiteLossProb: loss, whiteNoResultProb: 0.02,
                    whiteScoreMean: signal, whiteScoreMeanSq: 0, whiteLead: 0,
                    varTimeLeft: -1, shorttermWinlossError: -1, shorttermScoreError: -1,
                    policyProbs: policy, policyOptimismUsed: 0, nnXLen: 9, nnYLen: 9,
                    whiteOwnerMap: [Float](repeating: ownerValue, count: 81)
                )
            }
        }
    }

    func testSerialAndPipelinedTeacherPathsProduceByteIdenticalShards() async throws {
        // Cover several batches, several shards, and a val split, at both symmetry settings.
        try await assertSerialAndPipelinedMatch(symmetries: 0, valRatio: 0)
        try await assertSerialAndPipelinedMatch(symmetries: 0, valRatio: 0.34)
        try await assertSerialAndPipelinedMatch(symmetries: 8, valRatio: 0.34)
    }

    private func assertSerialAndPipelinedMatch(symmetries: Int, valRatio: Float) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let serialOut = root.appendingPathComponent("serial")
        let pipelinedOut = root.appendingPathComponent("pipelined")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // A mix of scored and resignation games with enough moves to span multiple teacher batches.
        let games = [
            "(;SZ[9]KM[7.5]RE[B+2.5];B[cc];W[gg];B[cg];W[gc];B[ee];W[dd])",
            "(;SZ[9]KM[7.5]RE[W+R];B[ee];W[ff];B[dd];W[cc];B[gg];W[bb])",
            "(;SZ[9]KM[7.5]RE[B+0.5];B[dd];W[ff];B[fd];W[df];B[ee])",
            "(;SZ[9]KM[7.5]RE[W+3.5];B[cc];W[gg];B[gc];W[cg];B[ee];W[ce];B[ec])",
            "(;SZ[9]KM[7.5]RE[B+R];B[ee];W[cc];B[gg];W[cg];B[gc])",
            "(;SZ[9]KM[7.5]RE[W+1.5];B[dd];W[ff];B[fd];W[df];B[ee];W[ce])",
        ]
        for (offset, game) in games.enumerated() {
            try Data(game.utf8).write(to: input.appendingPathComponent("g\(offset).sgf"))
        }

        func options(_ output: URL) -> TrainingDataOptions {
            TrainingDataOptions(
                sgfDirectory: input, outputDirectory: output, nnLen: 9, shardSize: 4,
                rules: chineseRules(), minimumMoves: 2, symmetries: symmetries, valRatio: valRatio
            )
        }

        let teacher = PositionSensitiveStub()
        // teacherBatchSize 3 with 6 short games forces many drains and partial final batches.
        let serialSummary = try await TrainingDataPipeline.generate(
            options(serialOut), teacher: teacher, teacherBatchSize: 3, serial: true
        )
        let pipelinedSummary = try await TrainingDataPipeline.generate(
            options(pipelinedOut), teacher: teacher, teacherBatchSize: 3, serial: false
        )

        // Summaries agree on everything that is not wall-clock.
        XCTAssertEqual(serialSummary.gamesUsed, pipelinedSummary.gamesUsed)
        XCTAssertEqual(serialSummary.totalSamples, pipelinedSummary.totalSamples)
        XCTAssertEqual(serialSummary.totalShards, pipelinedSummary.totalShards)
        XCTAssertEqual(serialSummary.validationSamples, pipelinedSummary.validationSamples)
        XCTAssertEqual(serialSummary.teacherPositionsLabeled, pipelinedSummary.teacherPositionsLabeled)
        XCTAssertGreaterThan(serialSummary.totalShards, 1, "expected multiple shards to exercise boundaries")

        // Every produced file (train shards + val.nngd) is byte-for-byte identical.
        let serialFiles = try shardFileNames(in: serialOut)
        let pipelinedFiles = try shardFileNames(in: pipelinedOut)
        XCTAssertEqual(serialFiles, pipelinedFiles, "shard file sets differ (symmetries=\(symmetries))")
        XCTAssertFalse(serialFiles.isEmpty)
        for name in serialFiles {
            let a = try Data(contentsOf: serialOut.appendingPathComponent(name))
            let b = try Data(contentsOf: pipelinedOut.appendingPathComponent(name))
            XCTAssertEqual(a, b, "bytes differ for \(name) (symmetries=\(symmetries), valRatio=\(valRatio))")
        }
    }

    // MARK: - Teacher SEARCH targets (Phase P, -teacher-visits N)

    /// The search policy target is the normalized root visit distribution: legal-masked (only legal
    /// moves are expanded), pass included when searched, and summing to 1. Board points map through
    /// `NNPos.locToPos` and pass maps to the trailing `area` entry.
    func testSearchTargetsNormalizeVisitDistributionAndIncludePass() {
        let boardXSize = 3
        let pointA = Location.getLoc(0, 0, boardXSize) // dense pos 0
        let pointB = Location.getLoc(2, 1, boardXSize) // dense pos 5
        let visits: [(loc: Loc, visits: Int64)] = [(pointA, 3), (pointB, 1), (Board.passLoc, 2)]
        let rootOut = output(
            policyProbs: [Float](repeating: 0.1, count: 10),
            whiteWin: 0.8, whiteLoss: 0.15, whiteNoResult: 0.05,
            whiteScoreMean: 5.0, whiteOwnerMap: [Float](repeating: 0.9, count: 9)
        )
        let targets = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.8, rootOutput: rootOut,
            boardXSize: boardXSize, nextPlayer: .white, nnLen: 3
        )
        XCTAssertEqual(targets.policy.count, 10)
        XCTAssertEqual(targets.policy.reduce(0, +), 1.0, accuracy: 1e-6)
        XCTAssertEqual(targets.policy[0], 3.0 / 6.0, accuracy: 1e-6)
        XCTAssertEqual(targets.policy[5], 1.0 / 6.0, accuracy: 1e-6)
        XCTAssertEqual(targets.policy[9], 2.0 / 6.0, accuracy: 1e-6) // pass (index == area)
        XCTAssertTrue(targets.policy.allSatisfy { $0 >= 0 })
        // Value is the search winrate (white to move keeps white's winrate); noResult folds into it.
        XCTAssertEqual(targets.value[0], 0.8, accuracy: 1e-6)
        XCTAssertEqual(targets.value[1], 0.2, accuracy: 1e-6)
        XCTAssertEqual(targets.value[2], 0)
        // Score/ownership pass through from the raw root net exactly as the raw-distillation path.
        XCTAssertEqual(targets.score, 5.0, accuracy: 1e-4)
        XCTAssertTrue(targets.ownership.allSatisfy { $0 == 1 }) // white-perspective 0.9 -> +1
    }

    /// The value target uses the SIDE-TO-MOVE perspective (like `targets(from:)`): white-to-move
    /// keeps white's search winrate, black-to-move takes `1 - winrate`; score negates for black.
    /// The visit-derived policy is player-agnostic, so it is identical across perspectives.
    func testSearchTargetsValuePerspectiveFlipsForBlackToMove() {
        let visits: [(loc: Loc, visits: Int64)] = [(Location.getLoc(0, 0, 3), 4)]
        let rootOut = output(
            policyProbs: [Float](repeating: 0.1, count: 10),
            whiteWin: 0.8, whiteLoss: 0.2, whiteNoResult: 0,
            whiteScoreMean: 4.0, whiteOwnerMap: nil
        )
        let white = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.8, rootOutput: rootOut,
            boardXSize: 3, nextPlayer: .white, nnLen: 3
        )
        let black = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.8, rootOutput: rootOut,
            boardXSize: 3, nextPlayer: .black, nnLen: 3
        )
        XCTAssertEqual(white.value[0], 0.8, accuracy: 1e-6)
        XCTAssertEqual(black.value[0], 0.2, accuracy: 1e-6)
        XCTAssertEqual(white.score, 4.0, accuracy: 1e-4)
        XCTAssertEqual(black.score, -4.0, accuracy: 1e-4)
        XCTAssertEqual(white.policy, black.policy)
    }

    // MARK: - Gumbel completed-Q policy target (WP-13, -teacher-target completed-q)

    /// The completed-Q policy is a valid dense distribution: legal-masked (illegal `-1`-sentinel
    /// positions stay exactly 0), pass included when legal, non-negative, and summing to 1. Value,
    /// score, and ownership are unchanged from the visits path.
    func testCompletedQTargetsAreValidDistribution() {
        let boardXSize = 3
        let locA = Location.getLoc(0, 0, boardXSize) // dense pos 0
        // Illegal at pos 4 and pos 7 (`-1` sentinel); every other position (incl. pass at 9) legal.
        var probs = [Float](repeating: 0.1, count: 10)
        probs[4] = -1
        probs[7] = -1
        let rootOut = output(
            policyProbs: probs, whiteWin: 0.6, whiteLoss: 0.3, whiteNoResult: 0.1,
            whiteScoreMean: 2.0, whiteOwnerMap: [Float](repeating: 0.9, count: 9)
        )
        let visits: [(loc: Loc, visits: Int64)] = [(locA, 5), (Board.passLoc, 3)]
        let childWinValues: [Loc: Double] = [locA: 0.5, Board.passLoc: -0.2]
        let targets = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.6, rootOutput: rootOut,
            boardXSize: boardXSize, nextPlayer: .white, nnLen: 3,
            target: .completedQ, childWinValues: childWinValues
        )
        XCTAssertEqual(targets.policy.count, 10)
        XCTAssertEqual(targets.policy.reduce(0, +), 1.0, accuracy: 1e-6)
        XCTAssertTrue(targets.policy.allSatisfy { $0 >= 0 })
        XCTAssertEqual(targets.policy[4], 0, "illegal position must stay masked to 0")
        XCTAssertEqual(targets.policy[7], 0, "illegal position must stay masked to 0")
        XCTAssertGreaterThan(targets.policy[9], 0, "legal pass must receive some mass")
        // Value/score/ownership are the shared (visits-path) values, not touched by the policy mode.
        XCTAssertEqual(targets.value[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(targets.score, 2.0, accuracy: 1e-4)
        XCTAssertTrue(targets.ownership.allSatisfy { $0 == 1 })
    }

    /// Discriminating test: a LOW-visit child with HIGH Q must get MORE mass under completed-q than
    /// under raw visit counts, and must overtake a HIGH-visit child with mediocre Q -- the exact
    /// improvement Gumbel completed-Q is for. Under visits the high-visit child dominates; under
    /// completed-q the high-Q child does. (Hand-checked against the mctx formula: with equal 0.1
    /// priors, visits A=10/B=2, q_A=0.0/q_B=0.9, sigmaScale=(50+10)*0.1=6, the high-Q child B wins.)
    func testCompletedQBoostsLowVisitHighQChildOverVisits() {
        let boardXSize = 3
        let locA = Location.getLoc(0, 0, boardXSize) // pos 0: many visits, neutral Q
        let locB = Location.getLoc(1, 0, boardXSize) // pos 1: few visits, high Q
        let rootOut = output(
            policyProbs: [Float](repeating: 0.1, count: 10), // uniform prior over all 10 legal moves
            whiteWin: 0.55, whiteLoss: 0.45, whiteNoResult: 0,
            whiteScoreMean: 0, whiteOwnerMap: nil
        )
        let visits: [(loc: Loc, visits: Int64)] = [(locA, 10), (locB, 2)]
        let childWinValues: [Loc: Double] = [locA: 0.0, locB: 0.9]

        let visitsTarget = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.55, rootOutput: rootOut,
            boardXSize: boardXSize, nextPlayer: .white, nnLen: 3,
            target: .visits, childWinValues: childWinValues
        )
        let completedQTarget = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.55, rootOutput: rootOut,
            boardXSize: boardXSize, nextPlayer: .white, nnLen: 3,
            target: .completedQ, childWinValues: childWinValues
        )
        // Both are valid distributions.
        XCTAssertEqual(completedQTarget.policy.reduce(0, +), 1.0, accuracy: 1e-6)
        // Visits mode: the high-visit child A dominates its low-visit sibling B.
        XCTAssertGreaterThan(visitsTarget.policy[0], visitsTarget.policy[1])
        // Completed-q mode: the high-Q child B overtakes the high-visit child A...
        XCTAssertGreaterThan(completedQTarget.policy[1], completedQTarget.policy[0])
        // ...and B carries strictly MORE mass than it did under raw visit counts (the improvement).
        XCTAssertGreaterThan(completedQTarget.policy[1], visitsTarget.policy[1])
        // Symmetrically, the mediocre-Q high-visit child A loses mass relative to the visits target.
        XCTAssertLessThan(completedQTarget.policy[0], visitsTarget.policy[0])
    }

    /// Default-mode byte-identity: omitting `target` must equal `target: .visits` exactly (the
    /// pre-WP-13 behavior), and the completed-q mode must differ ONLY in the policy -- value, score,
    /// and ownership are byte-identical across modes.
    func testDefaultTargetIsVisitsAndCompletedQOnlyChangesPolicy() {
        let boardXSize = 3
        let locA = Location.getLoc(0, 0, boardXSize)
        let locB = Location.getLoc(2, 1, boardXSize) // dense pos 5
        let visits: [(loc: Loc, visits: Int64)] = [(locA, 3), (locB, 1), (Board.passLoc, 2)]
        let childWinValues: [Loc: Double] = [locA: -0.1, locB: 0.8, Board.passLoc: 0.0]
        let rootOut = output(
            policyProbs: [Float](repeating: 0.1, count: 10),
            whiteWin: 0.8, whiteLoss: 0.15, whiteNoResult: 0.05,
            whiteScoreMean: 5.0, whiteOwnerMap: [Float](repeating: 0.9, count: 9)
        )
        let defaulted = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.8, rootOutput: rootOut,
            boardXSize: boardXSize, nextPlayer: .white, nnLen: 3
        )
        let explicitVisits = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.8, rootOutput: rootOut,
            boardXSize: boardXSize, nextPlayer: .white, nnLen: 3,
            target: .visits, childWinValues: childWinValues
        )
        let completedQ = TeacherLabeler.searchTargets(
            visitsByMove: visits, whiteWinProb: 0.8, rootOutput: rootOut,
            boardXSize: boardXSize, nextPlayer: .white, nnLen: 3,
            target: .completedQ, childWinValues: childWinValues
        )
        // Default == explicit .visits, byte-for-byte (same Targets value).
        XCTAssertEqual(defaulted, explicitVisits)
        // Completed-q changes the policy but leaves value/score/ownership identical.
        XCTAssertNotEqual(completedQ.policy, defaulted.policy)
        XCTAssertEqual(completedQ.value, defaulted.value)
        XCTAssertEqual(completedQ.score, defaulted.score)
        XCTAssertEqual(completedQ.ownership, defaulted.ownership)
    }

    /// End-to-end search-mode pipeline with a deterministic stub teacher: every distilled sample's
    /// policy target is a valid distribution (sums to 1, non-negative), value has noResult folded to
    /// 0 with win+loss summing to 1, and every teacher label is marked valid. Perspective is checked
    /// against a white-favored teacher: the black-to-move root is losing, the white-to-move root wins.
    func testTeacherSearchModePipelineProducesValidDistributionTargets() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let outputDir = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("(;SZ[9]KM[7.5]RE[W+R];B[ee];W[ff];B[dd];W[cc];B[gg])".utf8)
            .write(to: input.appendingPathComponent("a.sgf"))

        // Deterministic teacher: white favored (0.7 vs 0.3). fp32 precision so the per-position
        // search never needs the (nil) fp32 fallback factory. teacherVisits 6 -> coarse distribution.
        let teacher = TestPolicyEvaluator(
            nnXLen: 9, nnYLen: 9, preferredLoc: Location.getLoc(4, 4, 9),
            whiteWinProb: 0.7, whiteLossProb: 0.3
        )
        let options = TrainingDataOptions(
            sgfDirectory: input, outputDirectory: outputDir, nnLen: 9, shardSize: 64,
            rules: chineseRules(), minimumMoves: 2, symmetries: 0
        )
        let summary = try await TrainingDataPipeline.generate(
            options, teacher: teacher, teacherBatchSize: 8, teacherVisits: 6, precision: .float32
        )
        XCTAssertEqual(summary.gamesUsed, 1)
        XCTAssertEqual(summary.totalSamples, 5)
        XCTAssertEqual(summary.teacherPositionsLabeled, 5)

        let reader = try TrainingShardReader(url: outputDir.appendingPathComponent("shard-00000.nngd"))
        XCTAssertEqual(reader.header.sampleCount, 5)
        for index in 0 ..< 5 {
            let sample = try reader.sample(at: index)
            XCTAssertEqual(sample.policyTarget.reduce(0, +), 1.0, accuracy: 1e-4)
            XCTAssertTrue(sample.policyTarget.allSatisfy { $0 >= 0 })
            XCTAssertEqual(sample.valueTarget[2], 0)
            XCTAssertEqual(sample.valueTarget[0] + sample.valueTarget[1], 1.0, accuracy: 1e-6)
            XCTAssertTrue(sample.scoreValid)
            XCTAssertTrue(sample.ownershipValid)
            XCTAssertTrue(sample.ownershipTarget.allSatisfy { $0 == -1 || $0 == 0 || $0 == 1 })
        }
        // Position 0: black to move against a white-favored teacher -> black is losing (win < 0.5).
        XCTAssertLessThan(try reader.sample(at: 0).valueTarget[0], 0.5)
        // Position 1: white to move -> white is winning (win > 0.5).
        XCTAssertGreaterThan(try reader.sample(at: 1).valueTarget[0], 0.5)
    }

    /// End-to-end wiring for `-teacher-target completed-q`: the Gumbel completed-Q policy must thread
    /// from `generate(teacherTarget:)` through a REAL per-position search tree (its
    /// `rootChildWinValues`) and land as a valid, legal-masked, sum-to-1 distribution with all labels
    /// valid. Covers the numerically-degenerate edge case a constant-value stub exercises (all
    /// completed-Q equal -> [0,1] rescale denominator floors to epsilon -> sigma == 0 -> the policy
    /// falls back to the renormalized prior), proving that path stays finite and normalized.
    func testTeacherSearchCompletedQPipelineProducesValidDistributionTargets() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let outputDir = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("(;SZ[9]KM[7.5]RE[W+R];B[ee];W[ff];B[dd];W[cc];B[gg])".utf8)
            .write(to: input.appendingPathComponent("a.sgf"))

        let teacher = TestPolicyEvaluator(
            nnXLen: 9, nnYLen: 9, preferredLoc: Location.getLoc(4, 4, 9),
            whiteWinProb: 0.7, whiteLossProb: 0.3
        )
        let options = TrainingDataOptions(
            sgfDirectory: input, outputDirectory: outputDir, nnLen: 9, shardSize: 64,
            rules: chineseRules(), minimumMoves: 2, symmetries: 0
        )
        let summary = try await TrainingDataPipeline.generate(
            options, teacher: teacher, teacherBatchSize: 8, teacherVisits: 6,
            teacherTarget: .completedQ, precision: .float32
        )
        XCTAssertEqual(summary.gamesUsed, 1)
        XCTAssertEqual(summary.totalSamples, 5)

        let reader = try TrainingShardReader(url: outputDir.appendingPathComponent("shard-00000.nngd"))
        XCTAssertEqual(reader.header.sampleCount, 5)
        for index in 0 ..< 5 {
            let sample = try reader.sample(at: index)
            XCTAssertEqual(sample.policyTarget.reduce(0, +), 1.0, accuracy: 1e-4)
            XCTAssertTrue(sample.policyTarget.allSatisfy { $0 >= 0 && $0.isFinite })
            XCTAssertTrue(sample.scoreValid)
            XCTAssertTrue(sample.ownershipValid)
        }
    }

    /// Regression guard: `-teacher-visits 1` (and omitting it) must route to the unchanged raw
    /// teacher path, byte-for-byte. Any future refactor that let the flag leak into the raw path
    /// would break this.
    func testTeacherVisitsOneIsByteIdenticalToRawMode() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let defaultOut = root.appendingPathComponent("default")
        let explicitOut = root.appendingPathComponent("explicit")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("(;SZ[9]KM[7.5]RE[W+R];B[ee];W[ff];B[dd];W[cc];B[gg])".utf8)
            .write(to: input.appendingPathComponent("a.sgf"))
        let template = output3x3ReplacementIsIrrelevant(
            policyProbs: [Float](repeating: 1.0 / 82.0, count: 82), whiteWin: 0.6, whiteLoss: 0.3,
            whiteNoResult: 0.1, whiteScoreMean: 3.0, whiteOwnerMap: [Float](repeating: 0.5, count: 81)
        )
        let teacher = StubTeacher(template: template)
        func options(_ out: URL) -> TrainingDataOptions {
            TrainingDataOptions(
                sgfDirectory: input, outputDirectory: out, nnLen: 9, shardSize: 64,
                rules: chineseRules(), minimumMoves: 2, symmetries: 0
            )
        }
        _ = try await TrainingDataPipeline.generate(options(defaultOut), teacher: teacher, teacherBatchSize: 4)
        _ = try await TrainingDataPipeline.generate(
            options(explicitOut), teacher: teacher, teacherBatchSize: 4, teacherVisits: 1
        )
        let names = try shardFileNames(in: defaultOut)
        XCTAssertEqual(names, try shardFileNames(in: explicitOut))
        XCTAssertFalse(names.isEmpty)
        for name in names {
            let a = try Data(contentsOf: defaultOut.appendingPathComponent(name))
            let b = try Data(contentsOf: explicitOut.appendingPathComponent(name))
            XCTAssertEqual(a, b, "teacher-visits 1 diverged from the raw path for \(name)")
        }
    }

    private func shardFileNames(in directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .map(\.lastPathComponent)
            .filter { $0.hasSuffix(".nngd") }
            .sorted()
    }

    private func chineseRules() -> Rules {
        var rules = Rules(
            koRule: .simple, scoringRule: .area, taxRule: .none,
            multiStoneSuicideLegal: false, whiteHandicapBonusRule: .n,
            friendlyPassOk: true, komi: 7.5
        )
        rules.koRule = .positional
        return rules
    }
}
