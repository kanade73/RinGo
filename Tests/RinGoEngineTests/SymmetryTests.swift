import RinGoCore
import XCTest

/// Pure-function tests for the dihedral symmetry algebra (`RinGoCore.Symmetry`) the opening-book
/// converter relies on. These pin the group structure (identity / inverse / associativity) and the
/// consistency between `compose` and `apply`, which is exactly the property the `A' = compose(A,
/// invert(linkSym))` walk update depends on.
final class SymmetryTests: XCTestCase {
    private let size = 9
    private var points: [(Int, Int)] {
        (0 ..< size).flatMap { y in (0 ..< size).map { x in (x, y) } }
    }

    func testIdentityIsSymmetryZero() {
        for (x, y) in points {
            let (rx, ry) = Symmetry.apply(x: x, y: y, size: size, 0)
            XCTAssertEqual(rx, x)
            XCTAssertEqual(ry, y)
        }
    }

    func testComposeWithIdentity() {
        for s in 0 ..< Symmetry.count {
            XCTAssertEqual(Symmetry.compose(0, s), s)
            XCTAssertEqual(Symmetry.compose(s, 0), s)
        }
    }

    func testInverseIsGroupInverse() {
        for s in 0 ..< Symmetry.count {
            XCTAssertEqual(Symmetry.compose(s, Symmetry.invert(s)), 0, "compose(\(s), invert)")
            XCTAssertEqual(Symmetry.compose(Symmetry.invert(s), s), 0, "compose(invert, \(s))")
        }
        // Only 5 and 6 are the non-involution pair; everything else is self-inverse.
        XCTAssertEqual(Symmetry.invert(5), 6)
        XCTAssertEqual(Symmetry.invert(6), 5)
        for s in [0, 1, 2, 3, 4, 7] {
            XCTAssertEqual(Symmetry.invert(s), s)
        }
    }

    func testComposeMatchesAppliedComposition() {
        // apply(apply(p, f), g) == apply(p, compose(f, g)) for every point and every pair.
        for f in 0 ..< Symmetry.count {
            for g in 0 ..< Symmetry.count {
                let composed = Symmetry.compose(f, g)
                for (x, y) in points {
                    let (ix, iy) = Symmetry.apply(x: x, y: y, size: size, f)
                    let step = Symmetry.apply(x: ix, y: iy, size: size, g)
                    let direct = Symmetry.apply(x: x, y: y, size: size, composed)
                    XCTAssertEqual(step.x, direct.x, "f=\(f) g=\(g) at (\(x),\(y)) x")
                    XCTAssertEqual(step.y, direct.y, "f=\(f) g=\(g) at (\(x),\(y)) y")
                }
            }
        }
    }

    func testApplyThenInverseIsIdentity() {
        for s in 0 ..< Symmetry.count {
            for (x, y) in points {
                let (sx, sy) = Symmetry.apply(x: x, y: y, size: size, s)
                let back = Symmetry.apply(x: sx, y: sy, size: size, Symmetry.invert(s))
                XCTAssertEqual(back.x, x)
                XCTAssertEqual(back.y, y)
            }
        }
    }

    func testAssociativity() {
        for a in 0 ..< Symmetry.count {
            for b in 0 ..< Symmetry.count {
                for c in 0 ..< Symmetry.count {
                    XCTAssertEqual(
                        Symmetry.compose(Symmetry.compose(a, b), c),
                        Symmetry.compose(a, Symmetry.compose(b, c)),
                        "assoc \(a),\(b),\(c)"
                    )
                }
            }
        }
    }

    func testAllEightAreDistinctPermutations() {
        // A stone at an asymmetric point must land on 8 distinct images under the 8 symmetries.
        var images = Set<Int>()
        for s in 0 ..< Symmetry.count {
            let (x, y) = Symmetry.apply(x: 1, y: 0, size: size, s)
            images.insert(y * size + x)
        }
        XCTAssertEqual(images.count, 8)
    }

    func testTengenIsFixedByEverySymmetry() {
        for s in 0 ..< Symmetry.count {
            let (x, y) = Symmetry.apply(x: 4, y: 4, size: size, s)
            XCTAssertEqual(x, 4)
            XCTAssertEqual(y, 4)
        }
    }

    /// Concrete anchor tied to the real book: the four board corners are one orbit, and following a
    /// corner link with the book's recorded `linkSyms` must land them all on the SAME canonical
    /// square. In `root.html`, corners (0,0)/(8,0)/(0,8)/(8,8) carry linkSyms {1,3,0,2}; composing
    /// `invert(linkSym)` onto each corner (accumSym starts at 0) must map them to one shared cell.
    func testCornerLinkSymsCollapseToOneCanonicalSquare() {
        let corners: [(x: Int, y: Int, linkSym: Int)] = [
            (0, 0, 1), (8, 0, 3), (0, 8, 0), (8, 8, 2),
        ]
        var canonical = Set<Int>()
        for corner in corners {
            let accum = Symmetry.compose(0, Symmetry.invert(corner.linkSym))
            let (cx, cy) = Symmetry.apply(x: corner.x, y: corner.y, size: size, accum)
            canonical.insert(cy * size + cx)
        }
        XCTAssertEqual(canonical.count, 1, "all four corners must canonicalize to one square, got \(canonical)")
    }
}
