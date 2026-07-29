import Foundation
@testable import ringo
import RinGoCore
@testable import RinGoEngine
import XCTest

final class TrainingDataFormatTests: XCTestCase {
    func testTwoSyntheticGamesHaveExactHeaderAndFirstSampleBytes() throws {
        let first = syntheticSample(firstMove: Move(loc: Location.getLoc(0, 0, 2), pla: .black), score: 1.5)
        let second = syntheticSample(firstMove: Move(loc: Location.getLoc(1, 1, 2), pla: .white), score: -2.5)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let shard = directory.appendingPathComponent("golden.nngd")

        try TrainingShardWriter.write(samples: [first, second], nnLen: 2, to: shard)
        let bytes = try Data(contentsOf: shard)
        XCTAssertEqual(
            Array(bytes.prefix(RinGoDataFormat.headerSize)),
            [78, 78, 71, 68, 2, 0, 0, 0, 2, 0, 0, 0, 22, 0, 0, 0, 19, 0, 0, 0, 2, 0, 0, 0]
        )
        let sampleSize = RinGoDataFormat.sampleSize(nnLen: 2)
        let firstBytes = bytes.subdata(
            in: RinGoDataFormat.headerSize ..< RinGoDataFormat.headerSize + sampleSize
        )
        XCTAssertEqual(firstBytes, independentlyEncoded(first))
        // The bulk writer must serialize the SECOND sample byte-for-byte too, at the
        // unaligned offset that the odd v2 sampleSize produces (24 + sampleSize is not 4-aligned).
        let secondBytes = bytes.subdata(
            in: RinGoDataFormat.headerSize + sampleSize ..< RinGoDataFormat.headerSize + 2 * sampleSize
        )
        XCTAssertEqual(secondBytes, independentlyEncoded(second))
        XCTAssertEqual(bytes.count, 24 + 2 * sampleSize)

        let decoded = try TrainingShardReader(data: bytes)
        XCTAssertEqual(decoded.header.sampleCount, 2)
        XCTAssertEqual(try decoded.sample(at: 0), first)
        XCTAssertEqual(try decoded.sample(at: 1), second)
    }

    private func syntheticSample(firstMove: Move, score: Float) -> TrainingSample {
        let board = Board(2, 2)
        var rules = Rules(komi: 0.5)
        rules.koRule = .positional
        let history = BoardHistory(board, pla: firstMove.pla, rules: rules)
        let features = NNInputs.fillRowV7(
            board,
            history,
            firstMove.pla,
            MiscNNInputParams(),
            nnXLen: 2,
            nnYLen: 2,
            useNHWC: true
        )
        var policy = [Float](repeating: 0, count: 5)
        policy[NNPos.locToPos(firstMove.loc, 2, 2, 2)] = 1
        return TrainingSample(
            spatial: features.spatial,
            global: features.global,
            policyTarget: policy,
            valueTarget: score > 0 ? [1, 0, 0] : [0, 1, 0],
            scoreTarget: score,
            scoreValid: true,
            ownershipTarget: [1, 0, -1, 0],
            ownershipValid: true
        )
    }

    private func independentlyEncoded(_ sample: TrainingSample) -> Data {
        var bytes = [UInt8]()
        func append(_ value: UInt8) {
            bytes.append(value)
        }
        func append(_ value: Float) {
            let bits = value.bitPattern
            bytes.append(UInt8(truncatingIfNeeded: bits))
            bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
            bytes.append(UInt8(truncatingIfNeeded: bits >> 16))
            bytes.append(UInt8(truncatingIfNeeded: bits >> 24))
        }
        sample.spatial.forEach(append)
        sample.global.forEach(append)
        sample.policyTarget.forEach(append)
        sample.valueTarget.forEach(append)
        append(sample.scoreTarget)
        append(sample.scoreValid ? UInt8(1) : UInt8(0))
        bytes.append(contentsOf: sample.ownershipTarget.map { UInt8(bitPattern: $0) })
        append(sample.ownershipValid ? UInt8(1) : UInt8(0))
        return Data(bytes)
    }
}

final class TrainingSymmetryTests: XCTestCase {
    func testIdentityDoesNotChangeSample() {
        let sample = asymmetricSample()
        XCTAssertEqual(TrainingSymmetry.identity.transform(sample: sample, nnLen: 3), sample)
    }

    /// The dense policy distribution must permute point-for-point EXACTLY like the spatial/
    /// ownership planes for every one of the 8 symmetries -- this is what keeps a distilled soft
    /// policy target aligned with the board after augmentation (WP-2).
    func testPolicyAndOwnershipUseSameTransformForAllEightSymmetries() {
        let sample = asymmetricSample()
        for symmetry in TrainingSymmetry.allCases {
            let transformed = symmetry.transform(sample: sample, nnLen: 3)
            for source in 0 ..< 9 {
                let point = symmetry.transform(x: source % 3, y: source / 3, length: 3)
                let destination = point.y * 3 + point.x
                XCTAssertEqual(transformed.ownershipTarget[destination], sample.ownershipTarget[source])
                XCTAssertEqual(transformed.spatial[destination * 22], sample.spatial[source * 22])
                XCTAssertEqual(transformed.policyTarget[destination], sample.policyTarget[source])
            }
            // The pass entry (index 9) never moves under any symmetry.
            XCTAssertEqual(transformed.policyTarget[9], sample.policyTarget[9])
            XCTAssertEqual(transformed.policyTarget.count, sample.policyTarget.count)
            XCTAssertEqual(transformed.policyTarget, symmetry.transform(policy: sample.policyTarget, nnLen: 3))
            XCTAssertEqual(transformed.global, sample.global)
            XCTAssertEqual(transformed.scoreValid, sample.scoreValid)
            XCTAssertEqual(transformed.ownershipValid, sample.ownershipValid)
        }
    }

    /// A one-hot pass policy (all mass on index nnLen*nnLen) must stay exactly one-hot-at-pass
    /// under every symmetry, since pass is not a board point that any rotation/reflection can move.
    func testPassIsInvariantAndRotationsCompose() {
        var passOneHot = [Float](repeating: 0, count: 10)
        passOneHot[9] = 1
        for symmetry in TrainingSymmetry.allCases {
            XCTAssertEqual(symmetry.transform(policy: passOneHot, nnLen: 3), passOneHot)
        }
        var point = (x: 0, y: 1)
        for _ in 0 ..< 4 {
            point = TrainingSymmetry.rotate90.transform(x: point.x, y: point.y, length: 3)
        }
        XCTAssertEqual(point.x, 0)
        XCTAssertEqual(point.y, 1)

        for first in TrainingSymmetry.allCases {
            for second in TrainingSymmetry.allCases {
                XCTAssertTrue(TrainingSymmetry.allCases.contains { candidate in
                    (0 ..< 9).allSatisfy { index in
                        let intermediate = second.transform(x: index % 3, y: index / 3, length: 3)
                        let composed = first.transform(x: intermediate.x, y: intermediate.y, length: 3)
                        let expected = candidate.transform(x: index % 3, y: index / 3, length: 3)
                        return expected.x == composed.x && expected.y == composed.y
                    }
                })
            }
        }
    }

    private func asymmetricSample() -> TrainingSample {
        var spatial = [Float](repeating: 0, count: 9 * 22)
        for index in 0 ..< 9 {
            spatial[index * 22] = Float(index + 1)
        }
        // A distinct, non-one-hot value per board point (plus a distinguishable pass entry) so a
        // permutation bug can't hide behind a degenerate all-zeros-but-one-entry distribution.
        var policy = (0 ..< 9).map { Float($0 + 1) }
        policy.append(42)
        return TrainingSample(
            spatial: spatial,
            global: Array(0 ..< 19).map(Float.init),
            policyTarget: policy,
            valueTarget: [1, 0, 0],
            scoreTarget: 4.5,
            scoreValid: true,
            ownershipTarget: [-1, 0, 1, 1, -1, 0, 0, 1, -1],
            ownershipValid: true
        )
    }
}

final class TrainingGameResultTests: XCTestCase {
    func testParsesBlackAndWhiteScoreResults() {
        XCTAssertEqual(TrainingGameResult.parse("B+3.5"), .scored(winner: .black, whiteMinusBlack: -3.5))
        XCTAssertEqual(TrainingGameResult.parse(" W+0.5 "), .scored(winner: .white, whiteMinusBlack: 0.5))
    }

    func testParsesBlackAndWhiteResignations() {
        XCTAssertEqual(TrainingGameResult.parse("B+R"), .resignation(winner: .black))
        XCTAssertEqual(TrainingGameResult.parse("w+resign"), .resignation(winner: .white))
    }

    func testJigoIsExplicitlySkippedAndInvalidResultIsUnparseable() {
        for value in ["0", "Draw", "Jigo", "B+0", "W+0", "B+0.0"] {
            XCTAssertEqual(TrainingGameResult.parse(value), .jigo)
        }
        XCTAssertNil(TrainingGameResult.parse("Void"))
        XCTAssertNil(TrainingGameResult.parse("B+unknown"))
    }

    func testResignationUsesSignedFifteenPointProxy() throws {
        XCTAssertEqual(TrainingGameResult.resignationScoreProxy, 15)
        XCTAssertEqual(TrainingGameResult.resignation(winner: .white).whiteMinusBlackScore, 15)
        XCTAssertEqual(TrainingGameResult.resignation(winner: .black).whiteMinusBlackScore, -15)

        let game = try SGFReader.parse("(;SZ[9]RE[B+R];B[aa];W[bb])")
        XCTAssertEqual(game.result, "B+R")
    }
}

final class TrainingDataEndToEndTests: XCTestCase {
    func testThreeSGFsProduceDeterministicShardsAndPolicy() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let nested = input.appendingPathComponent("nested")
        let output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try write("(;SZ[9]KM[7.5]RE[B+2.5];B[aa];W[bb])", to: input.appendingPathComponent("a.sgf"))
        try write("(;SZ[9]KM[7.5]RE[W+R];B[cc];W[dd])", to: nested.appendingPathComponent("b.sgf"))
        try write("(;SZ[9]KM[7.5];B[ee];W[ff];B[];W[])", to: nested.appendingPathComponent("c.SGF"))

        let summary = try TrainingDataPipeline.generate(TrainingDataOptions(
            sgfDirectory: input,
            outputDirectory: output,
            nnLen: 9,
            shardSize: 5,
            rules: chineseRules(),
            minimumMoves: 2,
            symmetries: 0
        ))
        XCTAssertEqual(summary.filesFound, 3)
        XCTAssertEqual(summary.gamesUsed, 3)
        XCTAssertEqual(summary.gamesSkipped, 0)
        XCTAssertEqual(summary.totalSamples, 8)
        XCTAssertEqual(summary.totalShards, 2)
        XCTAssertGreaterThan(summary.elapsedSeconds, 0)

        let first = try TrainingShardReader(url: output.appendingPathComponent("shard-00000.nngd"))
        let second = try TrainingShardReader(url: output.appendingPathComponent("shard-00001.nngd"))
        XCTAssertEqual(first.header.sampleCount, 5)
        XCTAssertEqual(second.header.sampleCount, 3)
        // a.sgf: RE[B+2.5], a genuinely scored game -> a one-hot policy and valid score/ownership.
        XCTAssertEqual(try oneHotIndex(first.sample(at: 0).policyTarget), 0)
        XCTAssertTrue(try first.sample(at: 0).scoreValid)
        XCTAssertTrue(try first.sample(at: 0).ownershipValid)
        XCTAssertEqual(try oneHotIndex(first.sample(at: 1).policyTarget), 10)
        // b.sgf: RE[W+R], a resignation -> the synthetic score proxy AND both validity flags false
        // (a resigned game's final position isn't a
        // trustworthy score or ownership label).
        XCTAssertEqual(try first.sample(at: 2).scoreTarget, -15)
        XCTAssertFalse(try first.sample(at: 2).scoreValid)
        XCTAssertFalse(try first.sample(at: 2).ownershipValid)
        // c.SGF: no RE tag, ends by two passes and is scored by area rules -> also valid.
        XCTAssertEqual(try oneHotIndex(second.sample(at: 1).policyTarget), 81)
        XCTAssertEqual(try oneHotIndex(second.sample(at: 2).policyTarget), 81)
        XCTAssertTrue(try second.sample(at: 1).scoreValid)
        XCTAssertTrue(try second.sample(at: 1).ownershipValid)

        let symmetricOutput = root.appendingPathComponent("symmetric-output")
        let symmetricSummary = try TrainingDataPipeline.generate(TrainingDataOptions(
            sgfDirectory: input,
            outputDirectory: symmetricOutput,
            nnLen: 9,
            shardSize: 64,
            rules: chineseRules(),
            minimumMoves: 2,
            symmetries: 8
        ))
        XCTAssertEqual(symmetricSummary.totalSamples, 64)
        let symmetric = try TrainingShardReader(url: symmetricOutput.appendingPathComponent("shard-00000.nngd"))
        XCTAssertEqual(try oneHotIndex(symmetric.sample(at: 0).policyTarget), 0)
        XCTAssertEqual(try oneHotIndex(symmetric.sample(at: 1).policyTarget), 8)
    }

    /// The exact index holding all the probability mass of a one-hot dense policy distribution, or
    /// `nil` if it isn't one-hot (more than one nonzero entry, or the single nonzero entry isn't 1).
    private func oneHotIndex(_ policy: [Float]) -> Int? {
        let nonzero = policy.indices.filter { policy[$0] != 0 }
        guard nonzero.count == 1, policy[nonzero[0]] == 1 else { return nil }
        return nonzero[0]
    }

    func testSkipReasonsAreCountedIndependently() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try write("not sgf", to: input.appendingPathComponent("parse.sgf"))
        try write("(;SZ[13]RE[B+1];B[aa];W[bb])", to: input.appendingPathComponent("size.sgf"))
        try write("(;SZ[9]RE[B+1];B[aa])", to: input.appendingPathComponent("short.sgf"))
        try write("(;SZ[9]RE[Jigo];B[aa];W[bb])", to: input.appendingPathComponent("jigo.sgf"))
        try write("(;SZ[9];B[aa];W[bb])", to: input.appendingPathComponent("result.sgf"))

        let summary = try TrainingDataPipeline.generate(TrainingDataOptions(
            sgfDirectory: input,
            outputDirectory: output,
            nnLen: 9,
            shardSize: 10,
            rules: chineseRules(),
            minimumMoves: 2,
            symmetries: 0
        ))
        XCTAssertEqual(summary.parseSkips, 1)
        XCTAssertEqual(summary.boardSizeSkips, 1)
        XCTAssertEqual(summary.minimumMoveSkips, 1)
        XCTAssertEqual(summary.jigoSkips, 1)
        XCTAssertEqual(summary.resultSkips, 1)
        XCTAssertEqual(summary.gamesSkipped, 5)
        XCTAssertEqual(summary.totalShards, 0)
    }

    private func write(_ string: String, to url: URL) throws {
        try Data(string.utf8).write(to: url)
    }

    private func chineseRules() -> Rules {
        var rules = Rules(
            koRule: .simple,
            scoringRule: .area,
            taxRule: .none,
            multiStoneSuicideLegal: false,
            whiteHandicapBonusRule: .n,
            friendlyPassOk: true,
            komi: 7.5
        )
        rules.koRule = .positional
        return rules
    }
}

final class TrainingDataValRatioTests: XCTestCase {
    /// With three SGFs and a 0.5 valRatio, one file is held out into a single `val.nngd` and the
    /// rest flow through the existing `shard-*.nngd` path. `totalShards` must still report the TRAIN
    /// shard count (not include val.nngd); the val count lives in
    /// `validationSamples`/`validationFiles`.
    func testValRatioSplitsFilesIntoTrainShardsAndSingleValNngd() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try write("(;SZ[9]KM[7.5]RE[B+2.5];B[aa];W[bb])", to: input.appendingPathComponent("a.sgf"))
        try write("(;SZ[9]KM[7.5]RE[W+R];B[cc];W[dd])", to: input.appendingPathComponent("b.sgf"))
        try write("(;SZ[9]KM[7.5]RE[B+1.5];B[ee];W[ff])", to: input.appendingPathComponent("c.sgf"))

        let summary = try TrainingDataPipeline.generate(TrainingDataOptions(
            sgfDirectory: input,
            outputDirectory: output,
            nnLen: 9,
            shardSize: 64,
            rules: chineseRules(),
            minimumMoves: 2,
            symmetries: 0,
            valRatio: 0.5
        ))
        XCTAssertEqual(summary.filesFound, 3)
        XCTAssertEqual(summary.gamesSkipped, 0)
        XCTAssertEqual(summary.validationFiles, 1) // floor(3 * 0.5) = 1
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("val.nngd").path))
        let trainShards = try FileManager.default.contentsOfDirectory(at: output, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "nngd" && $0.lastPathComponent.hasPrefix("shard-") }
        XCTAssertEqual(trainShards.count, summary.totalShards)
        XCTAssertEqual(
            summary.totalSamples,
            summary.validationSamples + trainShards.reduce(0) { $0 + shardSampleCount(at: $1) }
        )
        XCTAssertGreaterThan(summary.validationSamples, 0)
        XCTAssertGreaterThan(summary.totalShards, 0)
    }

    /// Re-running the same val-ratio at the same files must produce the SAME val.nngd bytes
    /// (the shuffle is fixed-seed). A non-deterministic val split would silently make validation-
    /// loss curves uncomparable across resumes.
    func testValRatioIsDeterministicAcrossRuns() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let first = root.appendingPathComponent("first")
        let second = root.appendingPathComponent("second")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for i in 0 ..< 6 {
            try write("(;SZ[9]KM[7.5]RE[B+\(i).5];B[aa];W[bb])",
                      to: input.appendingPathComponent("g\(i).sgf"))
        }

        _ = try TrainingDataPipeline.generate(TrainingDataOptions(
            sgfDirectory: input, outputDirectory: first, nnLen: 9, shardSize: 64,
            rules: chineseRules(), minimumMoves: 2, symmetries: 0, valRatio: 0.34
        ))
        _ = try TrainingDataPipeline.generate(TrainingDataOptions(
            sgfDirectory: input, outputDirectory: second, nnLen: 9, shardSize: 64,
            rules: chineseRules(), minimumMoves: 2, symmetries: 0, valRatio: 0.34
        ))

        let firstVal = try Data(contentsOf: first.appendingPathComponent("val.nngd"))
        let secondVal = try Data(contentsOf: second.appendingPathComponent("val.nngd"))
        XCTAssertEqual(firstVal, secondVal)
    }

    /// valRatio == 0 must preserve the historical behavior: no `val.nngd` produced, val fields in
    /// the summary read zero, and a single train shard with all samples.
    func testValRatioZeroSkipsValShardEntirely() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try write("(;SZ[9]KM[7.5]RE[B+2.5];B[aa];W[bb])", to: input.appendingPathComponent("a.sgf"))

        let summary = try TrainingDataPipeline.generate(TrainingDataOptions(
            sgfDirectory: input, outputDirectory: output, nnLen: 9, shardSize: 64,
            rules: chineseRules(), minimumMoves: 2, symmetries: 0, valRatio: 0
        ))
        XCTAssertEqual(summary.validationFiles, 0)
        XCTAssertEqual(summary.validationSamples, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("val.nngd").path))
    }

    /// Regression: TrainCommand used to pass the shard directory to `RinGoDataset.load`, which loads
    /// every .nngd file including the training shards. The held-out validation must be `val.nngd` only.
    func testLoadValidationShardReadsOnlyValNngd() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let input = root.appendingPathComponent("input")
        let output = root.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: input, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for i in 0 ..< 4 {
            try write("(;SZ[9]KM[7.5]RE[B+\(i).5];B[aa];W[bb];B[cc];W[dd])",
                      to: input.appendingPathComponent("g\(i).sgf"))
        }

        let summary = try TrainingDataPipeline.generate(TrainingDataOptions(
            sgfDirectory: input, outputDirectory: output, nnLen: 9, shardSize: 64,
            rules: chineseRules(), minimumMoves: 2, symmetries: 0, valRatio: 0.5
        ))
        XCTAssertGreaterThan(summary.validationSamples, 0)

        let validation = try TrainCommand.loadValidationShard(from: output)
        XCTAssertEqual(validation.samples.count, summary.validationSamples)
    }

    private func shardSampleCount(at url: URL) -> Int {
        let bytes = (try? Data(contentsOf: url)) ?? Data()
        guard bytes.count >= RinGoDataFormat.headerSize else { return 0 }
        // sampleCount is the 5th little-endian u32 (offset 20).
        let offset = 20
        let raw = UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8
            | UInt32(bytes[offset + 2]) << 16 | UInt32(bytes[offset + 3]) << 24
        return Int(raw)
    }

    private func write(_ string: String, to url: URL) throws {
        try Data(string.utf8).write(to: url)
    }

    private func chineseRules() -> Rules {
        var rules = Rules(
            koRule: .simple,
            scoringRule: .area,
            taxRule: .none,
            multiStoneSuicideLegal: false,
            whiteHandicapBonusRule: .n,
            friendlyPassOk: true,
            komi: 7.5
        )
        rules.koRule = .positional
        return rules
    }
}
