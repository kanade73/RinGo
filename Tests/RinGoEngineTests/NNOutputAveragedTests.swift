import Foundation
import RinGoCore
@testable import RinGoEngine
import RinGoModel
import XCTest

/// Unit tests for `NNOutput.averaged` (Sources/RinGoEngine/NNOutput.swift), the element-wise
/// averaging that backs KataGo's `rootNumSymmetriesToSample` root symmetry averaging. Pure and
/// deterministic (no MLX): pins the averaging math itself — the mean of every scalar/value/score/
/// error field and ownership, the policy mean that preserves the illegal (`< 0`) mask, and the
/// legality-sign-mismatch fallback to the first output — independent of any real net.
final class NNOutputAveragedTests: XCTestCase {
    private func makeOutput(
        win: Float,
        loss: Float,
        noResult: Float = 0,
        scoreMean: Float,
        scoreMeanSq: Float = 0,
        lead: Float,
        varTimeLeft: Float = 0,
        shorttermWinlossError: Float = 0,
        shorttermScoreError: Float = 0,
        policy: [Float],
        optimism: Float = 0,
        ownership: [Float]? = nil
    ) -> NNOutput {
        NNOutput(
            nnHash: Hash128(0xDEAD_BEEF_0000_1111, 0x2222_3333_4444_5555),
            whiteWinProb: win,
            whiteLossProb: loss,
            whiteNoResultProb: noResult,
            whiteScoreMean: scoreMean,
            whiteScoreMeanSq: scoreMeanSq,
            whiteLead: lead,
            varTimeLeft: varTimeLeft,
            shorttermWinlossError: shorttermWinlossError,
            shorttermScoreError: shorttermScoreError,
            policyProbs: policy,
            policyOptimismUsed: optimism,
            nnXLen: 3,
            nnYLen: 3,
            whiteOwnerMap: ownership
        )
    }

    /// Test #2 (all-identical → single eval): averaging identical outputs returns that output. The
    /// integration counterpart is `RootSymmetryTests` on the D4-symmetric empty board.
    func testAveragingIdenticalOutputsReturnsInput() {
        let o = makeOutput(
            win: 0.6, loss: 0.4, scoreMean: 3.5, lead: 2.25,
            policy: [0.5, 0.25, -1, 0.25, -1], optimism: 1.0, ownership: [0.1, -0.2, 0.3]
        )
        let averaged = NNOutput.averaged([o, o, o])
        XCTAssertEqual(averaged.whiteWinProb, o.whiteWinProb, accuracy: 1e-6)
        XCTAssertEqual(averaged.whiteLossProb, o.whiteLossProb, accuracy: 1e-6)
        XCTAssertEqual(averaged.whiteScoreMean, o.whiteScoreMean, accuracy: 1e-6)
        XCTAssertEqual(averaged.whiteLead, o.whiteLead, accuracy: 1e-6)
        for (a, e) in zip(averaged.policyProbs, o.policyProbs) {
            XCTAssertEqual(a, e, accuracy: 1e-6)
        }
        XCTAssertEqual(averaged.policyOptimismUsed, 1.0, accuracy: 1e-6)
        for (a, e) in zip(averaged.whiteOwnerMap ?? [], o.whiteOwnerMap ?? []) {
            XCTAssertEqual(a, e, accuracy: 1e-6)
        }
    }

    /// A single-element set is returned unchanged (the `k == 1`-equivalent shape of the averaging
    /// helper itself — matches a non-averaged eval byte-for-byte).
    func testAveragingSingleElementReturnsInput() {
        let o = makeOutput(win: 0.7, loss: 0.3, scoreMean: 1.0, lead: 1.0, policy: [1.0, -1])
        let averaged = NNOutput.averaged([o])
        XCTAssertEqual(averaged.whiteWinProb, 0.7)
        XCTAssertEqual(averaged.policyProbs, [1.0, -1])
    }

    /// Test #3 (elementwise mean + legality mask): every scalar field is the arithmetic mean; legal
    /// policy positions average while the `-1` illegal sentinel is preserved (all-`-1` → `-1`); the
    /// ownership map is the mean.
    func testAveragingIsElementwiseMeanAndPreservesIllegalMask() {
        // Position 2 is illegal in both; positions 0/1/3/4 legal.
        let a = makeOutput(
            win: 0.8, loss: 0.2, noResult: 0.0, scoreMean: 4.0, scoreMeanSq: 20.0, lead: 3.0,
            varTimeLeft: 10, shorttermWinlossError: 0.1, shorttermScoreError: 2.0,
            policy: [0.4, 0.2, -1, 0.2, 0.2], optimism: 0.5, ownership: [0.2, 0.4, -0.6]
        )
        let b = makeOutput(
            win: 0.4, loss: 0.6, noResult: 0.0, scoreMean: 0.0, scoreMeanSq: 40.0, lead: -1.0,
            varTimeLeft: 20, shorttermWinlossError: 0.3, shorttermScoreError: 4.0,
            policy: [0.2, 0.4, -1, 0.1, 0.3], optimism: 0.5, ownership: [0.4, 0.0, 0.6]
        )
        let m = NNOutput.averaged([a, b])
        XCTAssertEqual(m.whiteWinProb, 0.6, accuracy: 1e-6)
        XCTAssertEqual(m.whiteLossProb, 0.4, accuracy: 1e-6)
        XCTAssertEqual(m.whiteNoResultProb, 0.0, accuracy: 1e-6)
        XCTAssertEqual(m.whiteScoreMean, 2.0, accuracy: 1e-6)
        XCTAssertEqual(m.whiteScoreMeanSq, 30.0, accuracy: 1e-6)
        XCTAssertEqual(m.whiteLead, 1.0, accuracy: 1e-6)
        XCTAssertEqual(m.varTimeLeft, 15.0, accuracy: 1e-6)
        XCTAssertEqual(m.shorttermWinlossError, 0.2, accuracy: 1e-6)
        XCTAssertEqual(m.shorttermScoreError, 3.0, accuracy: 1e-6)
        XCTAssertEqual(m.policyProbs[0], 0.3, accuracy: 1e-6)
        XCTAssertEqual(m.policyProbs[1], 0.3, accuracy: 1e-6)
        XCTAssertEqual(m.policyProbs[2], -1.0, accuracy: 1e-6, "illegal (-1) mask must be preserved")
        XCTAssertEqual(m.policyProbs[3], 0.15, accuracy: 1e-6)
        XCTAssertEqual(m.policyProbs[4], 0.25, accuracy: 1e-6)
        let owner = m.whiteOwnerMap ?? []
        XCTAssertEqual(owner.count, 3)
        XCTAssertEqual(owner[0], 0.3, accuracy: 1e-6)
        XCTAssertEqual(owner[1], 0.2, accuracy: 1e-6)
        XCTAssertEqual(owner[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(m.policyOptimismUsed, 0.5, accuracy: 1e-6, "matching optimisms keep the common value")
    }

    /// The legality-sign-mismatch safety valve (KataGo's hash-collision guard): if any output
    /// disagrees with the first on which positions are legal, the averaged policy is the FIRST
    /// output's policy verbatim (all other fields still average normally).
    func testAveragingFallsBackToFirstPolicyOnLegalityMismatch() {
        // Position 2: legal (0.1) in `a` but illegal (-1) in `b` — a legality-sign mismatch.
        let a = makeOutput(win: 0.5, loss: 0.5, scoreMean: 1.0, lead: 1.0, policy: [0.4, 0.2, 0.1, 0.3])
        let b = makeOutput(win: 0.3, loss: 0.7, scoreMean: 3.0, lead: 2.0, policy: [0.5, 0.3, -1, 0.2])
        let m = NNOutput.averaged([a, b])
        XCTAssertEqual(m.policyProbs, a.policyProbs, "policy must fall back to the first output verbatim")
        // Non-policy fields still average.
        XCTAssertEqual(m.whiteWinProb, 0.4, accuracy: 1e-6)
        XCTAssertEqual(m.whiteScoreMean, 2.0, accuracy: 1e-6)
    }

    /// Mismatched optimisms average (rather than take the first); ownership averages only over the
    /// outputs that carry a map.
    func testAveragingOptimismMeanAndPartialOwnership() {
        let a = makeOutput(win: 0.5, loss: 0.5, scoreMean: 0, lead: 0, policy: [1.0], optimism: 0.2, ownership: [0.4])
        let b = makeOutput(win: 0.5, loss: 0.5, scoreMean: 0, lead: 0, policy: [1.0], optimism: 0.8, ownership: nil)
        let m = NNOutput.averaged([a, b])
        XCTAssertEqual(m.policyOptimismUsed, 0.5, accuracy: 1e-6, "mismatched optimisms average")
        // Only `a` has ownership, so the mean is over the 1 present map (not divided by 2).
        let owner = m.whiteOwnerMap ?? []
        XCTAssertEqual(owner.first, 0.4)
    }
}
