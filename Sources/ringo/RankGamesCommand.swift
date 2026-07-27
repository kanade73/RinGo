import Foundation
import MLX
import RinGoCore
import RinGoEngine
import RinGoModel

/// Parsed `ringo rankgames` flags. Split out from `main` so flag defaults/validation are
/// testable without touching MLX (see `RankGamesCommandTests`).
struct RankGamesOptions {
    var sgfDirectory = ""
    var studentModel = ""
    var teacherModel = ""
    // Student defaults to fp32 (our own net, the same precision `evaluate` pins for loss parity);
    // teacher defaults to fp16 (faster, and matches how the existing 32v teacher labels were made).
    var studentPrecision: KataGoNetwork.Precision = .float32
    var teacherPrecision: KataGoNetwork.Precision = .float16
    var batchSize = 256
    var nnLen = 9
    var rulesName = "chinese"
    var minimumMoves = 8
    // 0 == rank/emit every accepted game; > 0 == still writes the full ranking.tsv but only
    // symlinks the top N into -link-dir.
    var top = 0
    var linkDirectory: String?
    var outputPath = "ranking.tsv"
    /// Optional per-position dump (one row per replayed position: sgf_path, turn_number, kl,
    /// student_top_move, teacher_top_move). nil == the default (ranking.tsv only), unchanged.
    var positionsOutputPath: String?
}

/// `ringo rankgames`: active-learning selection for deep labeling (BREAKTHROUGH_IDEAS.md D2).
///
/// Ranks a local SGF corpus by student-vs-teacher raw-policy disagreement so that the highest-value
/// games get expensive 1000-visit analysis-engine labels first. Replays every SGF with the SAME
/// tolerant filters as `analysisexport` (parse -> board size == nnLen -> `>= min-moves` -> legal
/// positional-superko replay; game-result skips omitted, since ranking never uses the outcome),
/// builds V7 features per position, runs one RAW batched forward of each net (no search), and scores
/// each game by how surprised the student is by the teacher's policy.
enum RankGamesCommand {
    static let usage = """
    Usage: ringo rankgames -sgf-dir <dir> -student <model.bin.gz> -teacher <model.bin.gz> \
      [-student-precision fp32|fp16] [-teacher-precision fp32|fp16] [-batch 256] [-nnlen 9] \
      [-rules chinese] [-min-moves 8] [-top N] [-link-dir <dir>] [-out ranking.tsv] \
      [-positions-out positions.tsv]
    """

    /// Number of per-position KLs averaged into the per-game score (see `scoreGame`).
    private static let topPositionsPerGame = 5

    static func parse(_ arguments: [String]) throws -> RankGamesOptions {
        var options = RankGamesOptions()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard index + 1 < arguments.count else { throw CLIError(description: "Missing value for \(flag)") }
            let value = arguments[index + 1]
            switch flag {
            case "-sgf-dir": options.sgfDirectory = value
            case "-student": options.studentModel = value
            case "-teacher": options.teacherModel = value
            case "-student-precision": options.studentPrecision = try precision(value, named: flag)
            case "-teacher-precision": options.teacherPrecision = try precision(value, named: flag)
            case "-batch":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -batch (must be a positive int)")
                }
                options.batchSize = parsed
            case "-nnlen":
                guard let parsed = Int(value), parsed > 1, parsed <= Board.maxLen else {
                    throw CLIError(description: "Invalid -nnlen (must be in 2...\(Board.maxLen))")
                }
                options.nnLen = parsed
            case "-rules": options.rulesName = value
            case "-min-moves":
                guard let parsed = Int(value), parsed >= 0 else {
                    throw CLIError(description: "Invalid -min-moves (must be >= 0)")
                }
                options.minimumMoves = parsed
            case "-top":
                guard let parsed = Int(value), parsed >= 0 else {
                    throw CLIError(description: "Invalid -top (must be >= 0)")
                }
                options.top = parsed
            case "-link-dir": options.linkDirectory = value
            case "-out": options.outputPath = value
            case "-positions-out": options.positionsOutputPath = value
            default: throw CLIError(description: "Unknown rankgames option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard !options.sgfDirectory.isEmpty, !options.studentModel.isEmpty, !options.teacherModel.isEmpty else {
            throw CLIError(description: usage)
        }
        return options
    }

    private static func precision(_ value: String, named flag: String) throws -> KataGoNetwork.Precision {
        switch value {
        case "fp32": return .float32
        case "fp16": return .float16
        default: throw CLIError(description: "Invalid \(flag) (must be fp32 or fp16)")
        }
    }

    /// One ranked game: absolute SGF path, per-game disagreement score, move count, and the turn
    /// indices of the most-surprising positions (highest per-position KL first).
    struct RankedGame {
        let path: String
        let score: Float
        let moveCount: Int
        let topPositions: [Int]
    }

    static func main(_ arguments: [String]) throws {
        let options = try parse(arguments)
        let start = Date()

        let studentDesc = try ModelDesc.loadFromFileMaybeGZipped(options.studentModel)
        let teacherDesc = try ModelDesc.loadFromFileMaybeGZipped(options.teacherModel)
        // Both nets must consume the same V7 feature layout (they do for every b6..b40 KataGo net):
        // we build the features once and feed the identical MLXArray to both, so a channel-count
        // mismatch would be a silent shape error later. Fail loudly and early instead.
        guard studentDesc.numInputChannels == teacherDesc.numInputChannels,
              studentDesc.numInputGlobalChannels == teacherDesc.numInputGlobalChannels
        else {
            throw CLIError(description: "student/teacher input-channel mismatch "
                + "(student \(studentDesc.numInputChannels)/\(studentDesc.numInputGlobalChannels) vs "
                + "teacher \(teacherDesc.numInputChannels)/\(teacherDesc.numInputGlobalChannels))")
        }
        let spatialChannels = studentDesc.numInputChannels
        let globalChannels = studentDesc.numInputGlobalChannels

        // Bound the MLX GPU cache for this (potentially many-minute, 300k+ position) unattended run
        // rather than letting it grow unbounded -- conventions.md "Set MLX.Memory.cacheLimit
        // explicitly ... for any long-running, unattended MLX process". 9x9 nets are tiny; 512 MB is
        // ample headroom for both nets' working set at batch 256.
        MLX.Memory.cacheLimit = 512 * 1024 * 1024

        let student = try KataGoNetwork(
            desc: studentDesc, nnXLen: options.nnLen, nnYLen: options.nnLen, precision: options.studentPrecision
        )
        let teacher = try KataGoNetwork(
            desc: teacherDesc, nnXLen: options.nnLen, nnYLen: options.nnLen, precision: options.teacherPrecision
        )

        let rules = try resolveRankRules(options.rulesName)
        let files = try sgfFiles(in: URL(fileURLWithPath: options.sgfDirectory))

        var ranked = [RankedGame]()
        var totalPositions = 0
        var parseSkips = 0, sizeSkips = 0, minMoveSkips = 0, illegalSkips = 0
        // Only accumulated when -positions-out is set; nil keeps the default (ranking.tsv only) path.
        var positionRows: [PositionRow]? = options.positionsOutputPath != nil ? [] : nil
        for (fileIndex, file) in files.enumerated() {
            let game: SGFGame
            do {
                game = try SGFReader.load(file, defaultRules: rules)
            } catch {
                parseSkips += 1
                continue
            }
            guard game.boardXSize == options.nnLen, game.boardYSize == options.nnLen else {
                sizeSkips += 1
                continue
            }
            guard game.moves.count >= options.minimumMoves else {
                minMoveSkips += 1
                continue
            }
            guard let positions = replay(game: game, rules: rules) else {
                illegalSkips += 1
                continue
            }
            let scores = scorePositions(
                positions: positions, student: student, teacher: teacher,
                spatialChannels: spatialChannels, globalChannels: globalChannels,
                nnLen: options.nnLen, batchSize: options.batchSize
            )
            let kls = scores.map(\.kl)
            totalPositions += kls.count
            let (score, topPositions) = scoreGame(kls)
            ranked.append(RankedGame(
                path: file.standardizedFileURL.path,
                score: score,
                moveCount: game.moves.count,
                topPositions: topPositions
            ))
            if positionRows != nil {
                // turn_number == the position's index in the replay array, which equals
                // analysisexport's turnNumber: both designate the position BEFORE the t-th SGF move
                // (see AnalysisExporter `analyzeTurns: 0..<moveCount` / AnalysisIngestor
                // `game.moves[turn].pla`). Path matches ranking.tsv so the two outputs join cleanly.
                let sgfPath = file.standardizedFileURL.path
                for (turnNumber, positionScore) in scores.enumerated() {
                    positionRows?.append(PositionRow(
                        sgfPath: sgfPath,
                        turnNumber: turnNumber,
                        kl: positionScore.kl,
                        studentTopMove: gtpMoveString(pos: positionScore.studentTopPos, nnLen: options.nnLen),
                        teacherTopMove: gtpMoveString(pos: positionScore.teacherTopPos, nnLen: options.nnLen)
                    ))
                }
            }
            if (fileIndex + 1) % 500 == 0 {
                FileHandle.standardError.write(Data(
                    "rankgames: processed \(fileIndex + 1)/\(files.count) files, ranked \(ranked.count)\n".utf8
                ))
            }
        }

        // Deterministic order: score descending, path ascending on ties -> identical ranking.tsv
        // across re-runs (the forwards are already deterministic at fixed precision).
        ranked.sort { $0.score != $1.score ? $0.score > $1.score : $0.path < $1.path }

        try writeRanking(ranked, to: URL(fileURLWithPath: options.outputPath))
        var linkedCount = 0
        if let linkDirectory = options.linkDirectory {
            linkedCount = try makeSymlinks(
                ranked: ranked, top: options.top, into: URL(fileURLWithPath: linkDirectory)
            )
        }
        if let positionsOutputPath = options.positionsOutputPath, let positionRows {
            try writePositions(positionRows, to: URL(fileURLWithPath: positionsOutputPath))
        }

        let elapsed = Date().timeIntervalSince(start)
        let rate = elapsed > 0 ? Double(totalPositions) / elapsed : 0
        print(
            "rankgames: ranked=\(ranked.count) positions=\(totalPositions) "
                + "skipped=(parse=\(parseSkips), size=\(sizeSkips), minMoves=\(minMoveSkips), "
                + "illegal=\(illegalSkips)) -> \(options.outputPath)"
                + (options.linkDirectory != nil ? " (linked \(linkedCount) -> \(options.linkDirectory!))" : "")
                + (options.positionsOutputPath != nil
                    ? " (positions=\(positionRows?.count ?? 0) -> \(options.positionsOutputPath!))" : "")
                + String(format: " elapsed=%.3fs positions/sec=%.1f", elapsed, rate)
        )
    }

    // MARK: - Replay

    /// One replayed position: the board/history/side-to-move BEFORE turn `t` (t == the SGF move
    /// index), exactly the positions `analysisexport` would export as `analyzeTurns 0..<moveCount`.
    private struct ReplayPosition {
        let board: Board
        let history: BoardHistory
        let nextPlayer: Player
    }

    /// Incremental tolerant replay mirroring `TrainingDataPipeline.prepare`: positional superko, one
    /// `board.copy()`/`history.copy()` snapshot per move. Returns `nil` (whole game skipped) if any
    /// move is illegal, matching `analysisexport`'s legal-replay filter.
    private static func replay(game: SGFGame, rules: Rules) -> [ReplayPosition]? {
        let initial: (board: Board, history: BoardHistory, nextPlayer: Player)
        do {
            initial = try game.position(at: 0, rules: rules)
        } catch {
            return nil
        }
        let board = initial.board
        let history = initial.history
        var positions = [ReplayPosition]()
        positions.reserveCapacity(game.moves.count)
        for move in game.moves {
            positions.append(ReplayPosition(board: board.copy(), history: history.copy(), nextPlayer: move.pla))
            guard history.makeBoardMoveTolerant(board, move.loc, move.pla, preventEncore: false) else { return nil }
        }
        return positions
    }

    // MARK: - Scoring

    /// One scored position: the KL(teacher || student) plus each net's argmax legal policy index
    /// (into the `area+1` vector, where `area` is the pass slot). Kept as raw indices so the GTP
    /// vertex string is formed only when `-positions-out` is actually written.
    struct PositionScore {
        let kl: Float
        let studentTopPos: Int
        let teacherTopPos: Int
    }

    /// Runs both nets over one game's positions (chunked by `batchSize`) and returns, per position,
    /// the KL(teacher || student) over legal moves and each net's argmax legal move. Streams per game
    /// and eval()s each chunk so no more than one game's + one batch's features/logits are ever live
    /// (conventions.md MLX discipline). The argmax is a cheap CPU pass over the same logit planes the
    /// KL already reads, so it is computed unconditionally (default `ranking.tsv` output unchanged).
    private static func scorePositions(
        positions: [ReplayPosition],
        student: KataGoNetwork,
        teacher: KataGoNetwork,
        spatialChannels: Int,
        globalChannels: Int,
        nnLen: Int,
        batchSize: Int
    ) -> [PositionScore] {
        let area = nnLen * nnLen
        var scores = [PositionScore]()
        scores.reserveCapacity(positions.count)
        var chunkStart = 0
        while chunkStart < positions.count {
            let chunkEnd = min(chunkStart + batchSize, positions.count)
            let chunk = positions[chunkStart ..< chunkEnd]
            let count = chunk.count

            var spatialFlat = [Float]()
            spatialFlat.reserveCapacity(count * area * spatialChannels)
            var globalFlat = [Float]()
            globalFlat.reserveCapacity(count * globalChannels)
            var legalPerPosition = [[Int]]()
            legalPerPosition.reserveCapacity(count)
            for position in chunk {
                let features = NNInputs.fillRowV7(
                    position.board, position.history, position.nextPlayer,
                    nnXLen: nnLen, nnYLen: nnLen, useNHWC: true
                )
                spatialFlat.append(contentsOf: features.spatial)
                globalFlat.append(contentsOf: features.global)
                legalPerPosition.append(legalMoveIndices(position: position, nnLen: nnLen))
            }

            let spatialArray = MLXArray(spatialFlat, [count, nnLen, nnLen, spatialChannels])
            let globalArray = MLXArray(globalFlat, [count, globalChannels])
            let studentLogits = policyChannelZero(
                student.forward(spatial: spatialArray, global: globalArray),
                area: area
            )
            let teacherLogits = policyChannelZero(
                teacher.forward(spatial: spatialArray, global: globalArray),
                area: area
            )
            // One eval() per chunk drains both nets' graphs together; asArray then pulls the [count,
            // area+1] logit planes to the CPU for the exact KL below.
            eval(studentLogits, teacherLogits)
            let studentFlat = studentLogits.asArray(Float.self)
            let teacherFlat = teacherLogits.asArray(Float.self)

            let stride = area + 1
            for local in 0 ..< count {
                let offset = local * stride
                let legal = legalPerPosition[local]
                let kl = klTeacherStudent(
                    student: studentFlat, teacher: teacherFlat,
                    offset: offset, legal: legal
                )
                // fp16 can (rarely, on synthetic nets) emit a non-finite logit -> non-finite KL.
                // Treat that as a recoverable 0 rather than poisoning the game score / crashing
                // (conventions.md: non-finite policy in postprocessing is recoverable, never fatal).
                scores.append(PositionScore(
                    kl: kl.isFinite ? kl : 0,
                    studentTopPos: argmaxLegal(studentFlat, offset: offset, legal: legal),
                    teacherTopPos: argmaxLegal(teacherFlat, offset: offset, legal: legal)
                ))
            }
            chunkStart = chunkEnd
        }
        return scores
    }

    /// Argmax over `legal` policy indices of `logits[offset + index]` (softmax is monotonic, so the
    /// argmax logit is the argmax probability). Non-finite logits fail every `>` comparison and are
    /// skipped; the fallback is the first legal index (always the same masked distribution the KL sees).
    static func argmaxLegal(_ logits: [Float], offset: Int, legal: [Int]) -> Int {
        var bestPos = legal.first ?? (logits.count - offset - 1)
        var bestLogit = -Float.greatestFiniteMagnitude
        for index in legal {
            let value = logits[offset + index]
            if value > bestLogit {
                bestLogit = value
                bestPos = index
            }
        }
        return bestPos
    }

    /// Formats a policy index (into the `area+1` vector) as the engine's canonical GTP vertex string
    /// -- the same `Location.toString` form GTPEngine prints for moves ("A1".."T9", "pass").
    static func gtpMoveString(pos: Int, nnLen: Int) -> String {
        if pos == nnLen * nnLen { return "pass" }
        let x = pos % nnLen
        let y = pos / nnLen
        return Location.toString(Location.getLoc(x, y, nnLen), nnLen, nnLen)
    }

    /// Extracts base-policy logits (spatial channel 0 + pass channel 0) as a `[count, area+1]`
    /// tensor. Channel 0 is the plain policy (optimism blends channel 0 with channel 1 at optimism>0;
    /// we compare at optimism 0), and slicing channel 0 is robust to nets with 1 vs 2 policy channels.
    private static func policyChannelZero(_ outputs: ModelRawOutputs, area: Int) -> MLXArray {
        let count = outputs.policySpatialLogits.dim(0)
        let spatial = outputs.policySpatialLogits[.ellipsis, 0 ..< 1].reshaped([count, area])
        let pass = outputs.policyPassLogits[.ellipsis, 0 ..< 1].reshaped([count, 1])
        return concatenated([spatial, pass], axis: 1)
    }

    /// Legal-move indices into the `area+1` policy vector for one position: on-board points where
    /// `board.isLegal` holds for the side-to-move (simple-ko aware; superko history is approximated,
    /// which is immaterial for a disagreement score), plus the always-legal pass at index `area`.
    private static func legalMoveIndices(position: ReplayPosition, nnLen: Int) -> [Int] {
        let board = position.board
        let mssl = position.history.rules.multiStoneSuicideLegal
        var legal = [Int]()
        legal.reserveCapacity(nnLen * nnLen + 1)
        for y in 0 ..< board.ySize {
            for x in 0 ..< board.xSize {
                let loc = Location.getLoc(x, y, board.xSize)
                if board.isLegal(loc, position.nextPlayer, mssl) {
                    legal.append(NNPos.xyToPos(x, y, nnLen))
                }
            }
        }
        legal.append(nnLen * nnLen) // pass
        return legal
    }

    /// KL(teacher || student) over the legal moves, computed in log-space (no epsilon needed).
    ///
    /// Direction rationale (WP-19 spec): we want the games where the STUDENT most disagrees with the
    /// TEACHER, i.e. where the student assigns low probability to moves the teacher favors. With the
    /// teacher as the reference distribution, KL(teacher||student) = sum_i p_t(i)*log(p_t(i)/p_s(i))
    /// is dominated exactly by moves with large teacher mass p_t(i) but small student mass p_s(i) --
    /// the "student is surprised by the teacher's move" signal we want to prioritize for deep
    /// labeling. (KL(student||teacher) would instead penalize the student being over-confident on
    /// moves the teacher dislikes, the wrong signal here.) It is >= 0 and 0 iff the two legal-move
    /// distributions coincide.
    static func klTeacherStudent(
        student: [Float], teacher: [Float], offset: Int, legal: [Int]
    ) -> Float {
        guard !legal.isEmpty else { return 0 }
        // Stable log-softmax over legal moves for each net (subtract the max, then log-sum-exp).
        var studentMax = -Float.greatestFiniteMagnitude
        var teacherMax = -Float.greatestFiniteMagnitude
        for index in legal {
            studentMax = max(studentMax, student[offset + index])
            teacherMax = max(teacherMax, teacher[offset + index])
        }
        var studentSumExp: Float = 0
        var teacherSumExp: Float = 0
        for index in legal {
            studentSumExp += Foundation.exp(student[offset + index] - studentMax)
            teacherSumExp += Foundation.exp(teacher[offset + index] - teacherMax)
        }
        let studentLogZ = studentMax + Foundation.log(studentSumExp)
        let teacherLogZ = teacherMax + Foundation.log(teacherSumExp)
        var kl: Float = 0
        for index in legal {
            let teacherLogProb = teacher[offset + index] - teacherLogZ
            let studentLogProb = student[offset + index] - studentLogZ
            let teacherProb = Foundation.exp(teacherLogProb)
            kl += teacherProb * (teacherLogProb - studentLogProb)
        }
        return kl
    }

    /// Per-game score = mean of the top-`topPositionsPerGame` per-position KLs (or all positions if
    /// fewer). Rationale (WP-19 spec): a game earns its expensive deep label from a FEW positions of
    /// concentrated disagreement, not from a high average -- averaging over every move would bury a
    /// game whose single critical position the student badly misreads under its many quiet, agreed
    /// positions. Returns the score and the turn indices of those top positions (highest KL first).
    static func scoreGame(_ kls: [Float]) -> (score: Float, topPositions: [Int]) {
        guard !kls.isEmpty else { return (0, []) }
        let ordered = kls.enumerated().sorted { lhs, rhs in
            lhs.element != rhs.element ? lhs.element > rhs.element : lhs.offset < rhs.offset
        }
        let topCount = min(topPositionsPerGame, ordered.count)
        let top = ordered.prefix(topCount)
        let score = top.reduce(Float(0)) { $0 + $1.element } / Float(topCount)
        return (score, top.map(\.offset))
    }

    // MARK: - Output

    static func writeRanking(_ ranked: [RankedGame], to url: URL) throws {
        var text = "path\tscore\tmove_count\ttop_positions\n"
        for game in ranked {
            let positions = game.topPositions.map(String.init).joined(separator: ",")
            text += String(format: "%@\t%.6f\t%d\t%@\n", game.path, game.score, game.moveCount, positions)
        }
        try Data(text.utf8).write(to: url, options: .atomic)
    }

    /// One row of the `-positions-out` TSV: the SGF path and the replay `turn_number` (which equals
    /// analysisexport's `turnNumber` -- both name the position before the t-th SGF move), the
    /// per-position KL, and each net's argmax legal move as a GTP vertex string.
    struct PositionRow {
        let sgfPath: String
        let turnNumber: Int
        let kl: Float
        let studentTopMove: String
        let teacherTopMove: String
    }

    /// Writes the one-row-per-position TSV. Header/columns are stable so `sgf_path`+`turn_number`
    /// join directly against an analysisexport manifest/responses labeling of the same corpus.
    static func writePositions(_ rows: [PositionRow], to url: URL) throws {
        var text = "sgf_path\tturn_number\tkl\tstudent_top_move\tteacher_top_move\n"
        for row in rows {
            text += String(
                format: "%@\t%d\t%.6f\t%@\t%@\n",
                row.sgfPath, row.turnNumber, row.kl, row.studentTopMove, row.teacherTopMove
            )
        }
        try Data(text.utf8).write(to: url, options: .atomic)
    }

    /// Populates `directory` with symlinks to the top-N ranked games (all ranked games if `top <= 0`)
    /// so the selection feeds straight into `analysisexport -sgf-dir <link-dir>`. Names are
    /// `rankNNNN_<original>.sgf` -- rank-ordered (so the directory listing preserves priority) and
    /// collision-free even when two source games share a basename. Returns the number of links made.
    private static func makeSymlinks(ranked: [RankedGame], top: Int, into directory: URL) throws -> Int {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let limit = top > 0 ? min(top, ranked.count) : ranked.count
        var linked = 0
        for rank in 0 ..< limit {
            let source = URL(fileURLWithPath: ranked[rank].path)
            let base = source.deletingPathExtension().lastPathComponent
            let name = String(format: "rank%04d_%@.sgf", rank, base)
            let destination = directory.appendingPathComponent(name)
            try? fileManager.removeItem(at: destination) // idempotent re-runs
            try fileManager.createSymbolicLink(at: destination, withDestinationURL: source)
            linked += 1
        }
        return linked
    }

    // MARK: - Corpus walk / rules

    /// Tolerant replay rules matching makedata/analysisexport: positional superko keeps real SGFs
    /// replayable and makes the feature-building position byte-identical to those pipelines.
    private static func resolveRankRules(_ name: String) throws -> Rules {
        var rules: Rules
        do {
            rules = try Rules.parseRulesWithoutKomi(name, komi: 7.5)
        } catch {
            throw CLIError(description: "Invalid rules '\(name)': \(error)")
        }
        rules.koRule = .positional
        return rules
    }

    /// Recursive `.sgf` walk (same shape as makedata/analysisexport), sorted for a stable file order.
    private static func sgfFiles(in directory: URL) throws -> [URL] {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { throw CLIError(description: "Cannot enumerate \(directory.path)") }
        var files = [URL]()
        for case let file as URL in enumerator where file.pathExtension.lowercased() == "sgf" {
            // Resolve symlinks before the regular-file test: a symlink's OWN isRegularFile is false,
            // so a directory of symlinks -- exactly what `rankgames -link-dir` / analysisexport pass
            // around (see AnalysisEnginePipeline.sgfFiles) -- would otherwise be silently skipped
            // (ranked=0). Keep the original (link) URL so paths stay anchored to `directory` and join
            // against analysisexport output; SGF loading follows the link transparently.
            let resolved = file.resolvingSymlinksInPath()
            if (try? resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                files.append(file)
            }
        }
        return files.sorted { $0.path < $1.path }
    }
}
