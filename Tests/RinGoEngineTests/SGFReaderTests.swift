import Foundation
import RinGoCore
@testable import RinGoEngine
import XCTest

final class SGFReaderTests: XCTestCase {
    func testCoreFixturesMatchGoldenMetadata() throws {
        let fixtureDirectory = repoRoot.appendingPathComponent("Tests/RinGoCoreTests/Fixtures")
        let goldenDirectory = repoRoot.appendingPathComponent("Tests/RinGoCoreTests/Goldens")
        let fixtures = try FileManager.default.contentsOfDirectory(
            at: fixtureDirectory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "sgf" }
        XCTAssertFalse(fixtures.isEmpty)
        for fixture in fixtures {
            let goldenURL = goldenDirectory
                .appendingPathComponent(fixture.deletingPathExtension().lastPathComponent + ".json")
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(contentsOf: goldenURL)) as? [String: Any]
            )
            let expectedRuleString = try XCTUnwrap(json["rules"] as? String)
            let defaultRules = try rules(expectedRuleString, komi: 7.5)
            let game = try SGFReader.load(fixture, defaultRules: defaultRules)
            let expectedX = try XCTUnwrap(json["boardXSize"] as? Int)
            let expectedY = try XCTUnwrap(json["boardYSize"] as? Int)
            XCTAssertEqual(game.boardXSize, expectedX, fixture.lastPathComponent)
            XCTAssertEqual(game.boardYSize, expectedY, fixture.lastPathComponent)
            XCTAssertTrue(game.rules.equalsIgnoringKomi(defaultRules), fixture.lastPathComponent)
            XCTAssertEqual(game.komi, Float((json["rules"] as? String)?.contains("0.5") == true ? 0.5 :
                    fixture.lastPathComponent == "japanese-midgame.sgf" ? 6.5 : 7.5))
            try assertMoves(game.moves, json["moves"] as? [[String]] ?? [], xSize: expectedX, ySize: expectedY)
            try assertMoves(
                game.placements,
                json["initialStones"] as? [[String]] ?? [],
                xSize: expectedX,
                ySize: expectedY
            )
            let moveNumber = try XCTUnwrap(json["moveNum"] as? Int)
            let position = try game.position(at: moveNumber)
            XCTAssertEqual(position.nextPlayer.rawValue, try Int8(XCTUnwrap(json["nextPla"] as? Int)))
        }
    }

    func testRectangularBoard() throws {
        let game = try SGFReader.parse("(;GM[1]FF[4]SZ[13:9]KM[5.5]RU[Chinese];B[ma];W[ai])")
        XCTAssertEqual(game.boardXSize, 13)
        XCTAssertEqual(game.boardYSize, 9)
        XCTAssertEqual(game.moves.count, 2)
        let position = try game.position(at: 2)
        XCTAssertEqual(position.board.colors[Location.getLoc(12, 0, 13)], .black)
        XCTAssertEqual(position.board.colors[Location.getLoc(0, 8, 13)], .white)
    }

    func testHandicapSetupAndPL() throws {
        let game = try SGFReader.parse("(;SZ[9]KM[0.5]HA[2]AB[cc][gg]AW[ee]PL[B])")
        XCTAssertEqual(game.placements.count, 3)
        let position = try game.position(at: 0)
        XCTAssertEqual(position.nextPlayer, .black)
        XCTAssertEqual(position.history.initialTurnNumber, 3)
        XCTAssertEqual(position.board.colors[Location.getLoc(2, 2, 9)], .black)
        XCTAssertEqual(position.board.colors[Location.getLoc(4, 4, 9)], .white)
    }

    func testPassForms() throws {
        let game = try SGFReader.parse("(;SZ[19];B[];W[tt])")
        XCTAssertEqual(game.moves.map(\.loc), [Board.passLoc, Board.passLoc])
    }

    func testErrorsAreExplicit() {
        XCTAssertThrowsError(try SGFReader.parse("(;SZ[19](;B[aa])(;B[bb]))")) {
            XCTAssertTrue(String(describing: $0).contains("variations"))
        }
        XCTAssertThrowsError(try SGFReader.parse("(;SZ[9]AE[aa])")) {
            XCTAssertTrue(String(describing: $0).contains("AE"))
        }
        XCTAssertThrowsError(try SGFReader.parse("(;SZ[9]AB[aa:cc])")) {
            XCTAssertTrue(String(describing: $0).contains("ranges"))
        }
        XCTAssertThrowsError(try SGFReader.parse("(;SZ[9];B[jj])")) {
            XCTAssertTrue(String(describing: $0).contains("out-of-bounds"))
        }
        XCTAssertThrowsError(try SGFReader.parse("(;SZ[1])"))
    }

    private func rules(_ string: String, komi: Float) throws -> Rules {
        if string.first == "{" { return try Rules.parseRules(string) }
        return try Rules.parseRulesWithoutKomi(string, komi: komi)
    }

    private func assertMoves(_ actual: [Move], _ expected: [[String]], xSize: Int, ySize _: Int) throws {
        XCTAssertEqual(actual.count, expected.count)
        for (move, item) in zip(actual, expected) {
            let player: Player = item[0] == "B" ? .black : .white
            let loc: Loc
            if item[1].isEmpty {
                loc = Board.passLoc
            } else {
                let bytes = Array(item[1].utf8)
                loc = Location.getLoc(Int(bytes[0] - 97), Int(bytes[1] - 97), xSize)
            }
            XCTAssertEqual(move, Move(loc: loc, pla: player))
        }
    }

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
