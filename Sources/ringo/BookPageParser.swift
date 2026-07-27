import Foundation

/// The fields we need from one katagobooks.org book page's inline `<script>` data blob. Everything
/// here is in the page's own *display frame* (the node's canonical orientation): `board` is
/// row-major `y*9+x`, `links`/`linkSyms` are keyed by the same index (81 = pass), and `bestMove` is
/// `moves[0]` (the book's own recommended move) in display coordinates.
struct BookPage {
    let nextPla: Int
    let board: [Int]
    let links: [Int: String]
    let linkSyms: [Int: Int]
    let bestMove: BestMove?
}

/// `moves[0]` reduced to what the converter stores: a single representative coordinate (or pass),
/// plus the book's value and visit scale. `moves[0]` is never the `'other'` aggregate bucket unless
/// the node is a pure leaf with no in-book children, in which case `bestMove` is `nil`.
struct BestMove {
    let x: Int
    let y: Int
    let isPass: Bool
    let wl: Double
    let visits: Double
}

enum BookPageError: Error, CustomStringConvertible {
    case missingField(String, path: String)
    case badBoard(count: Int, path: String)

    var description: String {
        switch self {
        case let .missingField(field, path): "book page \(path): missing/unparseable field '\(field)'"
        case let .badBoard(count, path): "book page \(path): board had \(count) cells, expected 81"
        }
    }
}

func parsePage(path: String) throws -> BookPage {
    let content = try String(contentsOfFile: path, encoding: .utf8)
    return try parsePageContent(content, path: path)
}

/// Parse a page's data blob from raw HTML text. Split out from disk I/O so it is unit-testable.
func parsePageContent(_ content: String, path: String) throws -> BookPage {
    guard let nextPla = scanInt(in: content, after: "const nextPla = ") else {
        throw BookPageError.missingField("nextPla", path: path)
    }
    guard let boardBody = sliceBetween(content, start: "const board = [", end: "]") else {
        throw BookPageError.missingField("board", path: path)
    }
    let board = boardBody.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    guard board.count == 81 else { throw BookPageError.badBoard(count: board.count, path: path) }

    let links = parseLinks(content)
    let linkSyms = parseLinkSyms(content)
    let bestMove = parseFirstMove(content)

    return BookPage(nextPla: nextPla, board: board, links: links, linkSyms: linkSyms, bestMove: bestMove)
}

// MARK: - Field parsers

private func parseLinks(_ content: String) -> [Int: String] {
    guard let body = sliceBetween(content, start: "const links = {", end: "}") else { return [:] }
    var result = [Int: String]()
    for item in body.split(separator: ",") {
        guard let colon = item.firstIndex(of: ":") else { continue }
        guard let key = Int(item[..<colon].trimmingCharacters(in: .whitespaces)) else { continue }
        var value = item[item.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        value = value.trimmingCharacters(in: CharacterSet(charactersIn: "'"))
        if !value.isEmpty { result[key] = value }
    }
    return result
}

private func parseLinkSyms(_ content: String) -> [Int: Int] {
    guard let body = sliceBetween(content, start: "const linkSyms = {", end: "}") else { return [:] }
    var result = [Int: Int]()
    for item in body.split(separator: ",") {
        let parts = item.split(separator: ":")
        guard parts.count == 2,
              let key = Int(parts[0].trimmingCharacters(in: .whitespaces)),
              let value = Int(parts[1].trimmingCharacters(in: .whitespaces)) else { continue }
        result[key] = value
    }
    return result
}

/// Extract `moves[0]` (the recommended move). Returns `nil` when the top entry is `'pass'`-less
/// `'other'` aggregate (a pure leaf) or has no coordinate.
private func parseFirstMove(_ content: String) -> BestMove? {
    guard let arrayStart = content.range(of: "const moves = [") else { return nil }
    guard let objectBody = firstBalancedBraces(content, from: arrayStart.upperBound) else { return nil }

    let wl = scanDouble(in: objectBody, after: "'wl':") ?? 0
    let visits = scanDouble(in: objectBody, after: "'v':") ?? 0

    if objectBody.contains("'xy':") {
        // First coordinate pair after `'xy':[[`.
        guard let pairBody = sliceBetween(objectBody, start: "'xy':[[", end: "]") else { return nil }
        let nums = pairBody.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard nums.count >= 2 else { return nil }
        return BestMove(x: nums[0], y: nums[1], isPass: false, wl: wl, visits: visits)
    }
    if objectBody.contains("'move':'pass'") {
        return BestMove(x: 0, y: 0, isPass: true, wl: wl, visits: visits)
    }
    return nil // 'other' aggregate — no single recommended move.
}

// MARK: - Scanning primitives

/// The substring strictly between the first occurrence of `start` and the next occurrence of `end`.
func sliceBetween(_ content: String, start: String, end: String) -> String? {
    guard let startRange = content.range(of: start) else { return nil }
    guard let endRange = content.range(of: end, range: startRange.upperBound ..< content.endIndex) else {
        return nil
    }
    return String(content[startRange.upperBound ..< endRange.lowerBound])
}

/// The `{ ... }` substring (contents only, braces excluded) starting at the first `{` at/after
/// `from`, matched by brace balancing.
private func firstBalancedBraces(_ content: String, from: String.Index) -> String? {
    guard let open = content[from...].firstIndex(of: "{") else { return nil }
    var depth = 0
    var index = open
    while index < content.endIndex {
        let char = content[index]
        if char == "{" { depth += 1 } else if char == "}" {
            depth -= 1
            if depth == 0 {
                return String(content[content.index(after: open) ..< index])
            }
        }
        index = content.index(after: index)
    }
    return nil
}

/// Parse an integer that immediately follows `marker`, up to the next `;` / `,` / whitespace.
func scanInt(in content: String, after marker: String) -> Int? {
    guard let range = content.range(of: marker) else { return nil }
    var digits = ""
    var index = range.upperBound
    while index < content.endIndex {
        let char = content[index]
        if char.isNumber || (digits.isEmpty && char == "-") { digits.append(char) } else { break }
        index = content.index(after: index)
    }
    return Int(digits)
}

/// Parse a floating-point number that immediately follows `marker`.
func scanDouble(in content: String, after marker: String) -> Double? {
    guard let range = content.range(of: marker) else { return nil }
    var token = ""
    var index = range.upperBound
    while index < content.endIndex {
        let char = content[index]
        if char.isNumber || char == "." || char == "-" || char == "+" || char == "e" || char == "E" {
            token.append(char)
        } else { break }
        index = content.index(after: index)
    }
    return Double(token)
}
