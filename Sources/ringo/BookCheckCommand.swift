import Foundation
import MLX
import RinGoCore
import RinGoEngine
import RinGoModel

/// Parsed `ringo bookcheck` flags.
struct BookCheckOptions {
    var bookPath = ""
    var modelPath = ""
    var precision: KataGoNetwork.Precision = .float16
    var maxDepth = 20
    var maxPositions = 2000
    var branch = 3
    var nnLen = 9
    var rulesName = "chinese"
    var komi: Float = 7.0
    var outPath = "bookcheck.tsv"
}

/// `ringo bookcheck`: does the net already "know" what the opening book knows?
///
/// Walks the opening book breadth-first from the empty board. At every visited book position, the
/// candidate continuations are the book's own recommended move PLUS the net's own top-`-branch`
/// raw-policy moves (simulating "the opponent plays something else"); the walk recurses into
/// whichever of those children is ALSO a book position (the book, not the net, decides what counts
/// as "in scope" -- this is what gives the walk breadth beyond the book's single main line). At
/// every visited position it runs one RAW (no-search) NN policy forward -- the same
/// `NNOutputPostprocessor` path `rawnn`/`rankgames` use -- and compares the net's top-1 move
/// against the book's recommendation.
///
/// High agreement means the book is redundant (already internalized in the net's weights); low
/// agreement quantifies exactly how much opening knowledge the net is missing, which is the
/// empirical case for (or against) spending deep-label budget on opening positions.
///
/// The lookup key (`BoardHistory.getSituationAndSimpleKoHash`) depends only on the board position,
/// simple-ko location, and side to move -- NOT on scoring/ko rules -- so a book built under
/// Tromp-Taylor still hits under the Chinese rules this traversal (and real games) use; see
/// `OpeningBook.swift`'s doc comment and `GTPEngine.genmove`'s book probe.
enum BookCheckCommand {
    static let usage = """
    Usage: ringo bookcheck -book <file.nngb> -model <net.bin.gz> [-precision fp16|fp32] \
      [-max-depth 20] [-max-positions 2000] [-branch 3] [-nnlen 9] [-rules chinese] [-komi 7.0] \
      [-out bookcheck.tsv]
    """

    // MARK: - Argument parsing

    static func parse(_ arguments: [String]) throws -> BookCheckOptions {
        var options = BookCheckOptions()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else { throw CLIError(description: "Missing value for \(flag)") }
            let value = arguments[index + 1]
            switch flag {
            case "-book": options.bookPath = value
            case "-model": options.modelPath = value
            case "-precision":
                switch value {
                case "fp32": options.precision = .float32
                case "fp16": options.precision = .float16
                default: throw CLIError(description: "Invalid -precision (must be fp32 or fp16)")
                }
            case "-max-depth":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -max-depth (must be > 0)")
                }
                options.maxDepth = parsed
            case "-max-positions":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -max-positions (must be > 0)")
                }
                options.maxPositions = parsed
            case "-branch":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -branch (must be > 0)")
                }
                options.branch = parsed
            case "-nnlen":
                guard let parsed = Int(value), parsed > 1, parsed <= Board.maxLen else {
                    throw CLIError(description: "Invalid -nnlen (must be in 2...\(Board.maxLen))")
                }
                options.nnLen = parsed
            case "-rules": options.rulesName = value
            case "-komi":
                guard let komi = Float(value), komi.isFinite else {
                    throw CLIError(description: "Invalid -komi '\(value)'")
                }
                options.komi = komi
            case "-out": options.outPath = value
            default: throw CLIError(description: "Unknown bookcheck option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard !options.bookPath.isEmpty, !options.modelPath.isEmpty else {
            throw CLIError(description: usage)
        }
        return options
    }

    // MARK: - Types

    /// One BFS-visited book position: board/history/mover plus the book's own entry, looked up once
    /// at enqueue time so the traversal never re-probes the book for a node it already confirmed.
    struct Node {
        let board: Board
        let history: BoardHistory
        let nextPlayer: Player
        let ply: Int
        let entry: OpeningBook.Entry
    }

    /// One TSV row / comparison result at a single visited book position.
    struct Row {
        let ply: Int
        let bookMove: String
        let bookVisits: Float
        let bookWLBlack: Float
        let netTopMove: String
        let netTopProb: Float
        let netBookProb: Float
        let bookMoveRank: Int
        let agree: Bool
    }

    // MARK: - Entry point

    static func main(_ arguments: [String]) throws {
        let options = try parse(arguments)
        let start = Date()

        status("Loading opening book \(options.bookPath)...")
        let book = try OpeningBook.load(path: options.bookPath)
        status("Loaded \(book.count) entries (\(book.boardXSize)x\(book.boardYSize), minVisits=\(book.minVisits))")

        status("Loading model \(options.modelPath)...")
        let desc = try ModelDesc.loadFromFileMaybeGZipped(options.modelPath)
        let rules = try Rules.parseRulesWithoutKomi(options.rulesName, komi: options.komi)
        let network = try KataGoNetwork(
            desc: desc, nnXLen: options.nnLen, nnYLen: options.nnLen, precision: options.precision
        )

        let (rows, illegalBookMoves) = try walk(
            book: book,
            rules: rules,
            network: network,
            desc: desc,
            options: options
        )

        try writeTSV(rows, to: URL(fileURLWithPath: options.outPath))

        let elapsed = Date().timeIntervalSince(start)
        printSummary(
            rows: rows,
            options: options,
            book: book,
            desc: desc,
            illegalBookMoves: illegalBookMoves,
            elapsed: elapsed
        )
    }

    // MARK: - Traversal

    /// Breadth-first walk of the book, one NN batch per depth level (`evaluateLevel` batches every
    /// node at the same ply into a single forward pass). Returns every visited row plus a count of
    /// book moves that, unexpectedly, were not in the net's legal-move mask at their own position.
    private static func walk(
        book: OpeningBook, rules: Rules, network: KataGoNetwork, desc: ModelDesc, options: BookCheckOptions
    ) throws -> (rows: [Row], illegalBookMoves: Int) {
        let rootBoard = Board(book.boardXSize, book.boardYSize)
        let rootHistory = BoardHistory(rootBoard, pla: .black, rules: rules, encorePhase: 0)
        guard let rootEntry = book.lookup(board: rootBoard, nextPlayer: .black) else {
            throw CLIError(description:
                "bookcheck: the empty board has no book entry -- hash construction bug "
                    + "(book has \(book.count) entries; the opening position must be one of them)")
        }

        var visited = Set<Hash128>()
        visited.insert(BoardHistory.getSituationAndSimpleKoHash(rootBoard, nextPlayer: .black))
        var frontier = [Node(board: rootBoard, history: rootHistory, nextPlayer: .black, ply: 1, entry: rootEntry)]

        var rows = [Row]()
        var illegalBookMoves = 0
        var processed = 0

        while !frontier.isEmpty, processed < options.maxPositions {
            let take = min(frontier.count, options.maxPositions - processed)
            let level = Array(frontier.prefix(take))
            frontier.removeFirst(take)

            let outputs = try evaluateLevel(level, network: network, desc: desc, nnLen: options.nnLen)

            var nextLevel = [Node]()
            for (node, output) in zip(level, outputs) {
                processed += 1
                let ranked = rankedLegalPolicy(output)
                let (row, bookMoveLegal) = makeRow(node: node, output: output, ranked: ranked, nnLen: options.nnLen)
                rows.append(row)
                if !bookMoveLegal { illegalBookMoves += 1 }

                guard node.ply < options.maxDepth else { continue }
                let candidates = candidateMoves(
                    entry: node.entry, ranked: ranked, board: node.board, branch: options.branch, nnLen: options.nnLen
                )
                for loc in candidates {
                    guard node.history.isLegal(node.board, loc, node.nextPlayer) else { continue }
                    let childBoard = node.board.copy()
                    let childHistory = node.history.copy()
                    childHistory.makeBoardMoveAssumeLegal(childBoard, loc, node.nextPlayer)
                    let childPlayer = childHistory.presumedNextMovePla
                    let hash = BoardHistory.getSituationAndSimpleKoHash(childBoard, nextPlayer: childPlayer)
                    guard !visited.contains(hash) else { continue }
                    guard let childEntry = book.lookup(board: childBoard, nextPlayer: childPlayer) else { continue }
                    visited.insert(hash)
                    nextLevel.append(Node(
                        board: childBoard, history: childHistory, nextPlayer: childPlayer,
                        ply: node.ply + 1, entry: childEntry
                    ))
                }
            }
            frontier.append(contentsOf: nextLevel)
        }
        return (rows, illegalBookMoves)
    }

    /// Batched raw (no-search) NN forward for every node at one BFS level, mirroring the manual
    /// `KataGoNetwork.forward` + `NNOutputPostprocessor.process` path `rawnn`/`rankgames` use --
    /// one `eval()` for the whole level, then one postprocess per row.
    private static func evaluateLevel(
        _ level: [Node], network: KataGoNetwork, desc: ModelDesc, nnLen: Int
    ) throws -> [NNOutput] {
        guard !level.isEmpty else { return [] }
        var spatialFlat = [Float]()
        spatialFlat.reserveCapacity(level.count * nnLen * nnLen * desc.numInputChannels)
        var globalFlat = [Float]()
        globalFlat.reserveCapacity(level.count * desc.numInputGlobalChannels)
        for node in level {
            let features = NNInputs.fillRowV7(node.board, node.history, node.nextPlayer, nnXLen: nnLen, nnYLen: nnLen)
            spatialFlat.append(contentsOf: features.spatial)
            globalFlat.append(contentsOf: features.global)
        }
        let spatialArray = MLXArray(spatialFlat, [level.count, nnLen, nnLen, desc.numInputChannels])
        let globalArray = MLXArray(globalFlat, [level.count, desc.numInputGlobalChannels])
        let outputs = network.forward(spatial: spatialArray, global: globalArray)
        eval([
            outputs.policySpatialLogits, outputs.policyPassLogits,
            outputs.valueLogits, outputs.scoreValues, outputs.ownership,
        ])

        let policyChannels = outputs.policySpatialLogits.dim(-1)
        let valueChannels = outputs.valueLogits.dim(-1)
        let scoreChannels = outputs.scoreValues.dim(-1)
        let spatialPositions = nnLen * nnLen
        let spatialPolicy = outputs.policySpatialLogits.asArray(Float.self)
        let passPolicy = outputs.policyPassLogits.asArray(Float.self)
        let valueFlat = outputs.valueLogits.asArray(Float.self)
        let scoreFlat = outputs.scoreValues.asArray(Float.self)
        let ownershipFlat = outputs.ownership.asArray(Float.self)

        var results = [NNOutput]()
        results.reserveCapacity(level.count)
        for (row, node) in level.enumerated() {
            var policyLogits = [Float](repeating: 0, count: spatialPositions + 1)
            let spatialBase = row * spatialPositions * policyChannels
            for position in 0 ..< spatialPositions {
                policyLogits[position] = spatialPolicy[spatialBase + position * policyChannels]
            }
            policyLogits[spatialPositions] = passPolicy[row * policyChannels]
            let raw = NNRawResult(
                policyLogits: policyLogits,
                valueLogits: Array(valueFlat[(row * valueChannels) ..< ((row + 1) * valueChannels)]),
                scoreValues: Array(scoreFlat[(row * scoreChannels) ..< ((row + 1) * scoreChannels)]),
                ownership: Array(ownershipFlat[(row * spatialPositions) ..< ((row + 1) * spatialPositions)])
            )
            let output = try NNOutputPostprocessor.process(
                raw: raw, board: node.board, history: node.history, nextPlayer: node.nextPlayer,
                postProcessParams: desc.postProcessParams, modelVersion: desc.modelVersion,
                nnXLen: nnLen, nnYLen: nnLen
            )
            results.append(output)
        }
        return results
    }

    // MARK: - Per-position comparison

    /// Legal policy entries sorted by probability, descending (ties broken by policy index for
    /// determinism). `NNOutput.policyProbs` uses the C++ sentinel -1 for illegal/padded positions.
    static func rankedLegalPolicy(_ output: NNOutput) -> [(pos: Int, prob: Float)] {
        var ranked = [(pos: Int, prob: Float)]()
        ranked.reserveCapacity(output.policyProbs.count)
        for (pos, prob) in output.policyProbs.enumerated() where prob >= 0 {
            ranked.append((pos, prob))
        }
        ranked.sort { $0.prob != $1.prob ? $0.prob > $1.prob : $0.pos < $1.pos }
        return ranked
    }

    /// Builds one TSV row comparing the book's recommendation against the net's raw policy at
    /// `node`. Returns `(row, bookMoveLegal)`; `bookMoveLegal` is false only if the book's recorded
    /// move somehow isn't in the net's legal-move mask at this position (should never happen -- the
    /// caller tracks it as a diagnostic count).
    static func makeRow(
        node: Node, output: NNOutput, ranked: [(pos: Int, prob: Float)], nnLen: Int
    ) -> (row: Row, bookMoveLegal: Bool) {
        let board = node.board
        let bookPos = NNPos.locToPos(node.entry.loc, board.xSize, nnLen, nnLen)
        let bookProb = output.policyProbs[bookPos]
        let bookMoveLegal = bookProb >= 0
        let rank = (ranked.firstIndex { $0.pos == bookPos }).map { $0 + 1 } ?? (ranked.count + 1)
        let top = ranked.first
        let topLoc = top.map { NNPos.posToLoc($0.pos, board.xSize, board.ySize, nnLen, nnLen) } ?? Board.nullLoc
        let row = Row(
            ply: node.ply,
            bookMove: Location.toString(node.entry.loc, board.xSize, board.ySize),
            bookVisits: node.entry.visits,
            bookWLBlack: node.entry.wl,
            netTopMove: Location.toString(topLoc, board.xSize, board.ySize),
            netTopProb: top?.prob ?? 0,
            netBookProb: max(0, bookProb),
            bookMoveRank: rank,
            agree: top?.pos == bookPos
        )
        return (row, bookMoveLegal)
    }

    /// Candidate continuations from `node`: the book's own recommendation plus the net's top-
    /// `branch` legal policy moves, deduplicated. The caller (`walk`) recurses into whichever of
    /// these lead to a position ALSO present in the book -- this is what gives the traversal breadth
    /// beyond the book's single main line (WP-30 spec point 2).
    static func candidateMoves(
        entry: OpeningBook.Entry, ranked: [(pos: Int, prob: Float)], board: Board, branch: Int, nnLen: Int
    ) -> [Loc] {
        var seen = Set<Loc>()
        var locs = [Loc]()
        func add(_ loc: Loc) {
            if seen.insert(loc).inserted { locs.append(loc) }
        }
        add(entry.loc)
        for (pos, _) in ranked.prefix(branch) {
            add(NNPos.posToLoc(pos, board.xSize, board.ySize, nnLen, nnLen))
        }
        return locs
    }

    // MARK: - Output

    static func writeTSV(_ rows: [Row], to url: URL) throws {
        var text = "ply\tbook_move\tbook_visits\tbook_wl_black\tnet_top1_move\tnet_top1_prob\tnet_book_prob\tbook_move_rank\tagree\n"
        for row in rows {
            text += String(
                format: "%d\t%@\t%.1f\t%.4f\t%@\t%.4f\t%.4f\t%d\t%d\n",
                row.ply, row.bookMove, row.bookVisits, row.bookWLBlack,
                row.netTopMove, row.netTopProb, row.netBookProb, row.bookMoveRank, row.agree ? 1 : 0
            )
        }
        try Data(text.utf8).write(to: url, options: .atomic)
    }

    /// Prints the stdout summary: overall + by-ply agreement, mean rank, mean book-move probability,
    /// and the top-5 miss rate -- the deliverable numbers for the "does the net already know the
    /// book" question.
    private static func printSummary(
        rows: [Row], options: BookCheckOptions, book: OpeningBook, desc: ModelDesc, illegalBookMoves: Int,
        elapsed: Double
    ) {
        guard !rows.isEmpty else {
            print("bookcheck: no book positions visited -- check -book/-model and hash construction")
            return
        }
        let precisionName = options.precision == .float16 ? "fp16" : "fp32"
        print(
            "bookcheck: \(options.bookPath) (\(book.count) entries) vs \(options.modelPath) (\(desc.shortInfoString))"
        )
        print("  precision=\(precisionName) nnlen=\(options.nnLen) rules=\(options.rulesName) komi=\(options.komi)")
        print(
            "  visited=\(rows.count) (max-positions=\(options.maxPositions), "
                + "max-depth=\(options.maxDepth), branch=\(options.branch))"
        )
        if illegalBookMoves > 0 {
            print("  WARNING: \(illegalBookMoves) book move(s) were not in the net's legal-move mask (unexpected)")
        }

        if let root = rows.first(where: { $0.ply == 1 }) {
            print("")
            print("Ply 1 (empty board) sanity check:")
            print(
                "  book recommends: \(root.bookMove)  "
                    + "(visits=\(Int(root.bookVisits)), wl_black=\(String(format: "%.3f", root.bookWLBlack)))"
            )
            print("  net top-1:       \(root.netTopMove)  (p=\(String(format: "%.3f", root.netTopProb)))")
            print("  agree: \(root.agree ? "yes" : "no")")
        }

        let agreeCount = rows.count(where: \.agree)
        let overallRate = Double(agreeCount) / Double(rows.count) * 100
        let meanRank = Double(rows.reduce(0) { $0 + $1.bookMoveRank }) / Double(rows.count)
        let meanBookProb = Double(rows.reduce(0) { $0 + $1.netBookProb }) / Double(rows.count)
        let outsideTop5 = rows.count(where: { $0.bookMoveRank > 5 })
        let outsideTop5Rate = Double(outsideTop5) / Double(rows.count) * 100

        print("")
        print("Overall top-1 agreement: \(agreeCount)/\(rows.count) = \(String(format: "%.1f", overallRate))%")
        print("Mean rank of book move in net policy: \(String(format: "%.2f", meanRank))")
        print("Mean net probability assigned to book move: \(String(format: "%.4f", meanBookProb))")
        print(
            "Fraction of positions where book move is outside net's top-5: "
                + "\(outsideTop5)/\(rows.count) = \(String(format: "%.1f", outsideTop5Rate))%"
        )

        print("")
        print("Agreement by ply:")
        print(" ply      n   agree%")
        let maxPly = rows.map(\.ply).max() ?? 0
        for ply in 1 ... maxPly {
            let atPly = rows.filter { $0.ply == ply }
            guard !atPly.isEmpty else { continue }
            let agreeAtPly = atPly.count(where: \.agree)
            let rate = Double(agreeAtPly) / Double(atPly.count) * 100
            print(String(format: "%4d  %5d  %6.1f%%", ply, atPly.count, rate))
        }

        print("")
        print(String(format: "elapsed=%.1fs -> %@", elapsed, options.outPath))
    }

    private static func status(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}
