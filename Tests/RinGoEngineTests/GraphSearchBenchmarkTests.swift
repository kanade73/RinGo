import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// Graph-search benchmark: quantifies (a) the per-playout Board/History COPY share
/// graph mode adds, (b) end-to-end search NPS with graph search ON vs OFF at 9x9 / V1600 / fp16 on the
/// real net, per position (opening vs midgame) and aggregate, and (c) the transposition dedup graph mode
/// buys — merges and NN-evals-saved — which is what pays for the DAG bookkeeping. The design gate is
/// ≤5% NPS overhead ON-vs-OFF. Env-gated so it never runs in `make check`; re-run with:
///
///   KATAGO_GRAPH_BENCH=1 swift test --filter GraphSearchBenchmarkTests
///
/// Uses `.tools/best-validated.bin.gz` (the real b20 net) by default; override with KATAGO_BENCH_NET.
/// Skips cleanly when the net or Metal host is unavailable.
final class GraphSearchBenchmarkTests: XCTestCase {
    private static let defaultNet = ".tools/best-validated.bin.gz"
    private static let nnLen = 9
    private static let visits: Int64 = 1600

    /// Forwarding evaluator that counts the total number of leaf NN evaluations requested, so the
    /// benchmark can report how many evals graph mode's transposition merges save vs pure tree.
    private actor CountingEvaluator: NNEvaluating {
        let inner: any NNEvaluating
        private(set) var evalCount = 0
        init(_ inner: any NNEvaluating) {
            self.inner = inner
        }

        func evaluate(_ requests: [NNRequest]) async throws -> [NNOutput] {
            evalCount += requests.count
            return try await inner.evaluate(requests)
        }

        func preWarm(_ requests: [NNRequest]) async throws {
            try await inner.preWarm(requests)
        }

        func snapshotCount() -> Int {
            evalCount
        }
    }

    private struct PositionRun {
        var nps: Double
        var evals: Int
        var merges: Int
        var visits: Int64
        var seconds: Double
    }

    func testGraphSearchNPSAndCopyShare() async throws {
        guard ProcessInfo.processInfo.environment["KATAGO_GRAPH_BENCH"] == "1" else {
            throw XCTSkip("set KATAGO_GRAPH_BENCH=1 to run the graph-search NPS / copy-share benchmark")
        }
        let netPath = ProcessInfo.processInfo.environment["KATAGO_BENCH_NET"] ?? Self.defaultNet
        guard FileManager.default.fileExists(atPath: netPath) else {
            throw XCTSkip("benchmark net missing: \(netPath) (set KATAGO_BENCH_NET)")
        }
        let desc = try ModelDesc.loadFromFileMaybeGZipped(netPath)
        let evaluator = try NNEvaluator(
            desc: desc, nnXLen: Self.nnLen, nnYLen: Self.nnLen, precision: .float16, maxBatchSize: 16
        )
        let positions = try benchmarkPositions()
        let labels = ["opening(empty)", "midgame(5-move)", "midgame(10-move)"]

        // Warm up both modes (fp16 compile + first-eval cost is not part of steady-state NPS).
        for graph in [false, true] {
            _ = try await search(evaluator, desc, positions[0], graph: graph).runSearch(maxVisits: 200)
        }

        // (b) Per-position NPS ON vs OFF + (c) merges / evals-saved. A fresh counting evaluator per run
        // isolates that run's eval count; graph merges dedup transpositions so ON evaluates fewer leaves.
        var runs: [Bool: [PositionRun]] = [false: [], true: []]
        for graph in [false, true] {
            for position in positions {
                let counter = CountingEvaluator(evaluator)
                let engine = search(counter, desc, position, graph: graph)
                let start = ProcessInfo.processInfo.systemUptime
                let result = try await engine.runSearch(maxVisits: Self.visits)
                let seconds = ProcessInfo.processInfo.systemUptime - start
                let evals = await counter.snapshotCount()
                let merges = await engine.graphDiagnosticsSnapshot().merges
                runs[graph, default: []].append(PositionRun(
                    nps: Double(result.visitsAchieved) / seconds, evals: evals,
                    merges: merges, visits: result.visitsAchieved, seconds: seconds
                ))
            }
        }
        let off = try XCTUnwrap(runs[false])
        let on = try XCTUnwrap(runs[true])

        // Aggregate overhead across positions (total visits / total wall), the design's headline number.
        let npsOffAgg = Double(off.reduce(Int64(0)) { $0 + $1.visits }) / off.reduce(0.0) { $0 + $1.seconds }
        let npsOnAgg = Double(on.reduce(Int64(0)) { $0 + $1.visits }) / on.reduce(0.0) { $0 + $1.seconds }
        let overheadAgg = (npsOffAgg - npsOnAgg) / npsOffAgg

        // (a) Copy micro-benchmark: cost of the per-playout Board+History copy pair graph mode adds,
        // measured at the densest (midgame) position, as a share of one OFF playout's wall time.
        let mid = positions[positions.count - 1]
        let copyIterations = 200_000
        let copyStart = ProcessInfo.processInfo.systemUptime
        var sink = 0
        for _ in 0 ..< copyIterations {
            let b = mid.board.copy()
            let h = mid.history.copy()
            sink &+= b.xSize &+ h.moveHistory.count
        }
        let copySeconds = ProcessInfo.processInfo.systemUptime - copyStart
        XCTAssertGreaterThanOrEqual(sink, 0) // keep the copies from being optimized away
        let copyNsPerPlayout = copySeconds / Double(copyIterations) * 1e9
        let offNsPerPlayout = 1e9 / npsOffAgg
        let copyShare = copyNsPerPlayout / offNsPerPlayout

        var lines: [String] = [
            "",
            "=== WP-GS3.5 graph-search benchmark (9x9, V\(Self.visits), fp16, \(desc.name)) ===",
            "positions: \(positions.count)   net: \(netPath)",
            "(a) per-playout Board+History copy: \(fmt(copyNsPerPlayout, "%.0f")) ns"
                + " (\(fmt(copyShare * 100, "%.2f"))% of one OFF playout's \(fmt(offNsPerPlayout, "%.0f")) ns)",
            "(b) per-position NPS ON vs OFF + (c) transposition dedup:",
        ]
        for index in positions.indices {
            let o = off[index]
            let n = on[index]
            let overhead = (o.nps - n.nps) / o.nps
            let evalsSaved = o.evals - n.evals
            lines.append(
                "    \(labels[index].padding(toLength: 16, withPad: " ", startingAt: 0))"
                    + " OFF \(fmt(o.nps, "%4.0f")) v/s   ON \(fmt(n.nps, "%4.0f")) v/s"
                    + "   overhead \(fmt(overhead * 100, "%+5.1f"))%"
                    + "   merges=\(n.merges)  evals OFF=\(o.evals) ON=\(n.evals) saved=\(evalsSaved)"
                    + " (\(fmt(Double(evalsSaved) / Double(max(o.evals, 1)) * 100, "%.1f"))%)"
            )
        }
        let verdict = overheadAgg <= 0.05 ? "PASS (<=5%)" : "FAIL (>5%)"
        lines.append("    AGGREGATE        OFF \(fmt(npsOffAgg, "%4.0f")) v/s   ON \(fmt(npsOnAgg, "%4.0f")) v/s"
            + "   overhead \(fmt(overheadAgg * 100, "%+5.1f"))% -> gate <=5%: \(verdict)")
        lines.append("=====================================================================")
        lines.append("")
        FileHandle.standardError.write(Data(lines.joined(separator: "\n").utf8))

        // Soft guard against a gross regression (the precise ≤5% verdict is printed above); graph mode
        // must not be catastrophically slower than pure tree at this scale.
        XCTAssertLessThan(overheadAgg, 0.15, "graph-search NPS overhead is far above the ≤5% design gate")
    }

    // MARK: - Fixtures

    private typealias BenchPosition = (board: Board, history: BoardHistory, nextPlayer: Player)

    private func fmt(_ value: Double, _ spec: String) -> String {
        String(format: spec, value)
    }

    private func search(
        _ evaluator: any NNEvaluating,
        _ desc: ModelDesc,
        _ position: BenchPosition,
        graph: Bool
    ) -> Search {
        var settings = SearchSettings()
        settings.numSearchWorkers = 8
        settings.leafBatchSize = 16
        settings.useGraphSearch = graph
        return Search(
            evaluator: evaluator, modelDesc: desc, nnXLen: Self.nnLen, nnYLen: Self.nnLen,
            precision: .float16, settings: settings,
            board: position.board, history: position.history, nextPlayer: position.nextPlayer
        )
    }

    private func benchmarkPositions() throws -> [BenchPosition] {
        try [
            emptyPosition(),
            played([(2, 2, .black), (6, 6, .white), (6, 2, .black), (2, 6, .white), (4, 4, .black)]),
            played([
                (2, 2, .black), (6, 6, .white), (2, 6, .black), (6, 2, .white), (4, 4, .black),
                (3, 6, .white), (6, 4, .black), (2, 4, .white), (4, 6, .black), (4, 2, .white),
            ]),
        ]
    }

    private func emptyPosition() -> BenchPosition {
        let board = Board(Self.nnLen, Self.nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        return (board, history, .black)
    }

    private func played(_ moves: [(Int, Int, Player)]) throws -> BenchPosition {
        let board = Board(Self.nnLen, Self.nnLen)
        let history = BoardHistory(board, pla: .black, rules: .getTrompTaylorish(), encorePhase: 0)
        for (x, y, pla) in moves {
            guard history.makeBoardMoveTolerant(board, Location.getLoc(x, y, Self.nnLen), pla) else {
                throw BenchmarkError("illegal setup move (\(x),\(y))")
            }
        }
        return (board, history, history.presumedNextMovePla)
    }
}

private struct BenchmarkError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) {
        self.description = description
    }
}
