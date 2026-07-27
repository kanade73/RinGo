import Foundation
import RinGoCore

/// A flat, hash-keyed opening-book lookup table produced offline by `ringo bookconvert` from a
/// katagobooks.org 9x9 book archive, and probed by the engine at the very start of `genmove`.
///
/// ## Binary format (`.nngb`, little-endian)
/// A 32-byte header followed by `entryCount` fixed-width 28-byte records:
/// ```
/// Header (32 bytes)
///   0   char[8]  magic     = "NGOBOOK1"
///   8   u32      version   = 1
///   12  u32      entryCount
///   16  u8       boardXSize
///   17  u8       boardYSize
///   18  u16      (reserved, 0)
///   20  f32      minVisits (the -min-visits cutoff used when building; provenance only)
///   24  u64      (reserved, 0)
/// Record (28 bytes) x entryCount
///   0   u64      hash0   \  BoardHistory.getSituationAndSimpleKoHash(board, nextPlayer)
///   8   u64      hash1   /  (our own Zobrist situational hash — self-consistent with runtime)
///   16  i32      loc     -- our padded engine Loc of the book's recommended move (passLoc = 1)
///   20  f32      wl      -- book win/loss value at that move (Black's perspective, ~[-1, 1])
///   24  f32      visits  -- book visit scale for the recommended move (multi-counts transpositions)
/// ```
/// The key is exactly the hash the engine computes at runtime, so a probe is a single dictionary
/// lookup. Records are written sorted by hash for reproducible files, but load does not rely on it.
public struct OpeningBook: Sendable {
    /// One stored recommendation: the move plus the book metadata that justified it.
    public struct Entry: Sendable {
        public let loc: Loc
        public let wl: Float
        public let visits: Float

        public init(loc: Loc, wl: Float, visits: Float) {
            self.loc = loc
            self.wl = wl
            self.visits = visits
        }
    }

    public static let magic = "NGOBOOK1"
    public static let recordSize = 28
    public static let headerSize = 32

    public let boardXSize: Int
    public let boardYSize: Int
    public let minVisits: Float
    private let table: [Hash128: Entry]

    public var count: Int {
        table.count
    }

    public init(boardXSize: Int, boardYSize: Int, minVisits: Float, table: [Hash128: Entry]) {
        self.boardXSize = boardXSize
        self.boardYSize = boardYSize
        self.minVisits = minVisits
        self.table = table
    }

    /// Probe the book for the position `(board, nextPlayer)`. Returns the stored entry if the
    /// position's situational hash is in the table, otherwise `nil`. Legality is the caller's
    /// responsibility (the engine re-checks before playing).
    public func lookup(board: Board, nextPlayer: Player) -> Entry? {
        table[BoardHistory.getSituationAndSimpleKoHash(board, nextPlayer: nextPlayer)]
    }

    /// Direct hash probe (used by the converter's dedup / tests).
    public func lookup(hash: Hash128) -> Entry? {
        table[hash]
    }

    // MARK: - Serialization

    public func serialized() -> Data {
        var data = Data(capacity: Self.headerSize + table.count * Self.recordSize)
        data.append(contentsOf: Array(Self.magic.utf8))
        data.appendLE(UInt32(1))
        data.appendLE(UInt32(table.count))
        data.append(UInt8(boardXSize))
        data.append(UInt8(boardYSize))
        data.appendLE(UInt16(0))
        data.appendLE(minVisits.bitPattern)
        data.appendLE(UInt64(0))
        // Sort by hash for a deterministic, reproducible file.
        for (hash, entry) in table.sorted(by: { $0.key < $1.key }) {
            data.appendLE(hash.hash0)
            data.appendLE(hash.hash1)
            data.appendLE(UInt32(bitPattern: Int32(entry.loc)))
            data.appendLE(entry.wl.bitPattern)
            data.appendLE(entry.visits.bitPattern)
        }
        return data
    }

    public func write(to path: String) throws {
        try serialized().write(to: URL(fileURLWithPath: path))
    }

    public static func load(path: String) throws -> OpeningBook {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try parse(data)
    }

    public static func parse(_ data: Data) throws -> OpeningBook {
        guard data.count >= headerSize else {
            throw OpeningBookError.truncated("header (\(data.count) bytes)")
        }
        let bytes = [UInt8](data)
        guard Array(bytes[0 ..< 8]) == Array(magic.utf8) else {
            throw OpeningBookError.badMagic
        }
        let version = readLE(UInt32.self, bytes, 8)
        guard version == 1 else { throw OpeningBookError.badVersion(Int(version)) }
        let entryCount = Int(readLE(UInt32.self, bytes, 12))
        let boardXSize = Int(bytes[16])
        let boardYSize = Int(bytes[17])
        let minVisits = Float(bitPattern: readLE(UInt32.self, bytes, 20))
        let expected = headerSize + entryCount * recordSize
        guard bytes.count >= expected else {
            throw OpeningBookError.truncated("expected \(expected) bytes, got \(bytes.count)")
        }
        var table = [Hash128: Entry](minimumCapacity: entryCount)
        var offset = headerSize
        for _ in 0 ..< entryCount {
            let hash0 = readLE(UInt64.self, bytes, offset)
            let hash1 = readLE(UInt64.self, bytes, offset + 8)
            let loc = Loc(Int32(bitPattern: readLE(UInt32.self, bytes, offset + 16)))
            let wl = Float(bitPattern: readLE(UInt32.self, bytes, offset + 20))
            let visits = Float(bitPattern: readLE(UInt32.self, bytes, offset + 24))
            table[Hash128(hash0, hash1)] = Entry(loc: loc, wl: wl, visits: visits)
            offset += recordSize
        }
        return OpeningBook(
            boardXSize: boardXSize, boardYSize: boardYSize, minVisits: minVisits, table: table
        )
    }
}

public enum OpeningBookError: Error, CustomStringConvertible {
    case badMagic
    case badVersion(Int)
    case truncated(String)

    public var description: String {
        switch self {
        case .badMagic: "opening book: bad magic (not an NGOBOOK1 file)"
        case let .badVersion(v): "opening book: unsupported version \(v)"
        case let .truncated(detail): "opening book: file truncated — \(detail)"
        }
    }
}

// MARK: - Little-endian byte helpers

private extension Data {
    mutating func appendLE(_ value: UInt16) {
        append(UInt8(value & 0xFF))
        append(UInt8((value >> 8) & 0xFF))
    }

    mutating func appendLE(_ value: UInt32) {
        for shift in stride(from: 0, through: 24, by: 8) {
            append(UInt8((value >> UInt32(shift)) & 0xFF))
        }
    }

    mutating func appendLE(_ value: UInt64) {
        for shift in stride(from: 0, through: 56, by: 8) {
            append(UInt8((value >> UInt64(shift)) & 0xFF))
        }
    }
}

private func readLE(_: UInt16.Type, _ bytes: [UInt8], _ offset: Int) -> UInt16 {
    UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
}

private func readLE(_: UInt32.Type, _ bytes: [UInt8], _ offset: Int) -> UInt32 {
    var value: UInt32 = 0
    for index in 0 ..< 4 {
        value |= UInt32(bytes[offset + index]) << UInt32(index * 8)
    }
    return value
}

private func readLE(_: UInt64.Type, _ bytes: [UInt8], _ offset: Int) -> UInt64 {
    var value: UInt64 = 0
    for index in 0 ..< 8 {
        value |= UInt64(bytes[offset + index]) << UInt64(index * 8)
    }
    return value
}
