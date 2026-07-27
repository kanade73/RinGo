import Foundation
import RinGoCore
import RinGoEngine

/// `ringo bookconvert`: offline conversion of a katagobooks.org 9x9 opening-book archive
/// (5.5M `html/<shard>/<hash>.html` pages, each embedding a JS data blob) into the flat,
/// hash-keyed `OpeningBook` table the engine probes at runtime.
///
/// ## How it walks the book (symmetry handling — the correctness hazard)
/// The book canonicalizes every position under the dihedral group, and each `links` edge carries a
/// `linkSyms` transform (`child.symmetryOfNode` in KataGo, the map *childNodespace -> this page's
/// display frame*). We never trust the book's hash; instead we keep our OWN concrete board and an
/// accumulated symmetry `A` mapping *our concrete frame -> the current page's display frame*
/// (`A == 0` at the empty root). Then:
///   - to follow a link at display index `posD`, the concrete move is `apply(posD, invert(A))`;
///   - after playing it, the child page's accumulated symmetry is `A' = compose(A, invert(linkSyms[posD]))`.
/// After every replayed move we re-derive the child page's `board` array under `A'` and assert it
/// matches our concrete board cell-for-cell — the loud test that catches any orientation/composition
/// bug (`--` a mismatch aborts the whole conversion).
///
/// The walk is breadth-first (shallow openings first, which is what a book is for), deduplicated by
/// our own situational hash, so it naturally enumerates every book-known continuation in every
/// orientation that can actually arise in a real game — each keyed exactly as the engine will key it.
enum BookConvertCommand {
    struct Options {
        var bookDir = ""
        var outPath = ""
        var minVisits: Double = 1e4
        var maxPositions = Int.max
        var rules = "tromp-taylor"
        var komi: Float = 7.0
        var quiet = false
    }

    // MARK: - Entry point

    static func main(_ arguments: [String]) throws {
        let options = try parse(arguments)
        let rules = try Rules.parseRulesWithoutKomi(options.rules, komi: options.komi)

        let converter = Converter(
            bookDir: options.bookDir,
            minVisits: options.minVisits,
            maxPositions: options.maxPositions,
            rules: rules,
            quiet: options.quiet
        )
        let start = Date()
        try converter.run()
        let elapsed = Date().timeIntervalSince(start)

        let book = OpeningBook(
            boardXSize: 9,
            boardYSize: 9,
            minVisits: Float(options.minVisits),
            table: converter.entries
        )
        try book.write(to: options.outPath)

        let rate = elapsed > 0 ? Double(converter.pagesRead) / elapsed : 0
        log("""
        bookconvert done:
          pages read            : \(converter.pagesRead)
          positions visited     : \(converter.visited.count)
          entries stored         : \(converter.entries.count)  (min-visits cutoff = \(options.minVisits))
          validation checks OK  : \(converter.validationChecks)
          skipped (cutoff)      : \(converter.skippedByCutoff)
          skipped (no best move): \(converter.skippedNoBestMove)
          illegal links skipped : \(converter.illegalLinks)
          unreadable pages      : \(converter.unreadablePages)
          elapsed               : \(String(format: "%.1f", elapsed)) s  (\(String(format: "%.0f", rate)) pages/s)
          output                : \(options.outPath)  (\(book.count) entries, \(OpeningBook.headerSize + book
            .count * OpeningBook.recordSize) bytes)
        """)
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    // MARK: - Converter

    final class Converter {
        let bookDir: String
        let minVisits: Double
        let maxPositions: Int
        let rules: Rules
        let quiet: Bool

        var visited = Set<Hash128>()
        var entries = [Hash128: OpeningBook.Entry]()
        var validationChecks = 0
        var pagesRead = 0
        var skippedByCutoff = 0
        var skippedNoBestMove = 0
        var illegalLinks = 0
        var unreadablePages = 0

        init(bookDir: String, minVisits: Double, maxPositions: Int, rules: Rules, quiet: Bool) {
            self.bookDir = bookDir
            self.minVisits = minVisits
            self.maxPositions = maxPositions
            self.rules = rules
            self.quiet = quiet
        }

        /// A queued position, stored as the concrete move path from the root (replayed on demand so
        /// the BFS frontier stays tiny — no board/history copies are held in the queue).
        struct Frontier {
            let path: [Loc]
            let accumSym: Int
            let pagePath: String
        }

        func run() throws {
            let rootPath = (bookDir as NSString).appendingPathComponent("root/root.html")
            guard FileManager.default.fileExists(atPath: rootPath) else {
                throw CLIError(description: "bookconvert: root page not found at \(rootPath)")
            }
            var queue = [Frontier(path: [], accumSym: 0, pagePath: rootPath)]
            var head = 0
            var lastLog = Date()

            while head < queue.count {
                if visited.count >= maxPositions { break }
                let node = queue[head]
                head += 1

                let (board, history, nextPla) = try replay(node.path)
                // A page that cannot be read/parsed is skipped (never aborts the run) — but a BOARD
                // validation mismatch below is a symmetry bug and MUST abort. These are distinct
                // failure classes on purpose.
                let page: BookPage
                do {
                    page = try parsePage(path: node.pagePath)
                } catch {
                    unreadablePages += 1
                    FileHandle.standardError.write(Data(
                        "Warning: skipping unreadable book page \(node.pagePath): \(error)\n".utf8
                    ))
                    continue
                }
                pagesRead += 1

                // The page's mover must match our replayed position (parity/legality sanity).
                let pageNextPla: Player = page.nextPla == 1 ? .black : .white
                guard pageNextPla == nextPla else {
                    throw CLIError(description:
                        "bookconvert: nextPla mismatch at \(node.pagePath): page=\(page.nextPla) replay=\(nextPla.rawValue)")
                }

                try validateBoard(board: board, page: page, accumSym: node.accumSym, path: node.pagePath)
                validationChecks += 1

                let hash = BoardHistory.getSituationAndSimpleKoHash(board, nextPlayer: nextPla)
                if visited.contains(hash) { continue }
                visited.insert(hash)

                storeBestMove(page: page, board: board, nextPla: nextPla, accumSym: node.accumSym, hash: hash)

                enqueueChildren(page: page, node: node, board: board, history: history, nextPla: nextPla, into: &queue)

                if !quiet, Date().timeIntervalSince(lastLog) > 5 {
                    lastLog = Date()
                    log("  ... visited \(visited.count), stored \(entries.count), queue \(queue.count - head)")
                }
            }
        }

        /// Replay a concrete move path from the empty root, returning the resulting position.
        private func replay(_ path: [Loc]) throws -> (Board, BoardHistory, Player) {
            let board = Board(9, 9)
            let history = BoardHistory(board, pla: .black, rules: rules, encorePhase: 0)
            var pla: Player = .black
            for loc in path {
                guard history.isLegal(board, loc, pla) else {
                    throw CLIError(description: "bookconvert: replay hit an illegal move (internal walk bug)")
                }
                history.makeBoardMoveAssumeLegal(board, loc, pla)
                pla = pla.opponent
            }
            return (board, history, pla)
        }

        /// Store `moves[0]` (mapped into our concrete frame) as the recommendation for this position,
        /// subject to the min-visits cutoff. Positions whose top book move is the `'other'` aggregate
        /// bucket carry no single move and are skipped.
        private func storeBestMove(page: BookPage, board: Board, nextPla: Player, accumSym: Int, hash: Hash128) {
            guard let best = page.bestMove else {
                skippedNoBestMove += 1
                return
            }
            guard best.visits >= minVisits else {
                skippedByCutoff += 1
                return
            }
            let loc: Loc
            if best.isPass {
                loc = Board.passLoc
            } else {
                let (cx, cy) = Symmetry.apply(x: best.x, y: best.y, size: 9, Symmetry.invert(accumSym))
                loc = Location.getLoc(cx, cy, 9)
            }
            guard best.isPass || board.isLegal(loc, nextPla) else {
                // Should never happen given the book only lists legal moves; skip rather than store junk.
                illegalLinks += 1
                return
            }
            entries[hash] = OpeningBook.Entry(loc: loc, wl: Float(best.wl), visits: Float(best.visits))
        }

        private func enqueueChildren(
            page: BookPage,
            node: Frontier,
            board: Board,
            history: BoardHistory,
            nextPla: Player,
            into queue: inout [Frontier]
        ) {
            for posD in page.links.keys.sorted() {
                guard let childRel = page.links[posD], let childSym = page.linkSyms[posD] else { continue }
                let loc: Loc
                if posD == 81 {
                    loc = Board.passLoc
                } else {
                    let (cx, cy) = Symmetry.apply(x: posD % 9, y: posD / 9, size: 9, Symmetry.invert(node.accumSym))
                    loc = Location.getLoc(cx, cy, 9)
                }
                guard history.isLegal(board, loc, nextPla) else {
                    illegalLinks += 1
                    continue
                }
                let childSymAccum = Symmetry.compose(node.accumSym, Symmetry.invert(childSym))
                let childPath = resolveRelative(base: node.pagePath, relative: childRel)
                queue.append(Frontier(path: node.path + [loc], accumSym: childSymAccum, pagePath: childPath))
            }
        }

        /// Assert `getSymBoard(ourBoard, accumSym) == page.board`, cell-for-cell. A failure means the
        /// symmetry composition diverged from the book — abort loudly (the whole point of the check).
        private func validateBoard(board: Board, page: BookPage, accumSym: Int, path: String) throws {
            for y in 0 ..< 9 {
                for x in 0 ..< 9 {
                    let (dx, dy) = Symmetry.apply(x: x, y: y, size: 9, accumSym)
                    let expected = page.board[dy * 9 + dx]
                    let ours = colorInt(board.colors[Location.getLoc(x, y, 9)])
                    if expected != ours {
                        throw CLIError(description: """
                        bookconvert: board validation FAILED at \(path)
                          cell our(\(x),\(y)) -> display(\(dx),\(dy)) accumSym=\(accumSym): page=\(expected) ours=\(
                              ours
                          )
                          (symmetry composition bug — conversion aborted)
                        """)
                    }
                }
            }
        }

        private func colorInt(_ color: Color) -> Int {
            switch color {
            case .black: 1
            case .white: 2
            default: 0
            }
        }
    }

    // MARK: - Argument parsing

    static func parse(_ arguments: [String]) throws -> Options {
        var options = Options()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            if flag == "-quiet" {
                options.quiet = true
                index += 1
                continue
            }
            guard index + 1 < arguments.count else {
                throw CLIError(description: "Missing value for \(flag)")
            }
            let value = arguments[index + 1]
            switch flag {
            case "-book-dir": options.bookDir = value
            case "-out": options.outPath = value
            case "-min-visits":
                guard let parsed = Double(value), parsed >= 0 else {
                    throw CLIError(description: "Invalid -min-visits '\(value)'")
                }
                options.minVisits = parsed
            case "-max-positions":
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError(description: "Invalid -max-positions '\(value)'")
                }
                options.maxPositions = parsed
            case "-rules": options.rules = value
            case "-komi":
                guard let komi = Float(value), komi.isFinite else {
                    throw CLIError(description: "Invalid -komi '\(value)'")
                }
                options.komi = komi
            default: throw CLIError(description: "Unknown option \(flag)\n\(usage)")
            }
            index += 2
        }
        guard !options.bookDir.isEmpty, !options.outPath.isEmpty else {
            throw CLIError(description: usage)
        }
        return options
    }

    static let usage = """
    Usage: ringo bookconvert -book-dir <html-dir> -out <file.nngb> \
      [-min-visits N=10000] [-max-positions N] [-rules <str>=tromp-taylor] [-komi f=7.0] [-quiet]
    """
}

// MARK: - Path resolution

func resolveRelative(base: String, relative: String) -> String {
    let baseDir = (base as NSString).deletingLastPathComponent
    let joined = (baseDir as NSString).appendingPathComponent(relative)
    return URL(fileURLWithPath: joined).standardizedFileURL.path
}
