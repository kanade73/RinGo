import Foundation
@testable import ringo
import XCTest

/// WP-27 (Codex R1 finding #5): unit tests for the GTP per-command wall-clock watchdog. A hung
/// MLX/Metal evaluation blocks `array.wait()` forever holding mlx-swift's global `evalLock`, which
/// no `Task.cancel()` can interrupt, so the search actor (and the whole GTP loop) wedges with no
/// in-process recovery. `GTPCommand.withWatchdog` is the loud-failure guard around each command; it
/// force-exits the process on breach so the supervisor restarts. These tests drive the timeout race
/// with a synthetic slow/fast body (no MLX evaluator, no real process exit — `onTimeout` just sets a
/// flag) to prove it fires iff the body outlives the deadline, exactly once, and honors the disable.
final class GTPCommandWatchdogTests: XCTestCase {
    /// Thread-safe one-shot flag standing in for the production `_exit(70)` side effect.
    private final class FiredBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        func fire() {
            lock.withLock { value = true }
        }

        var fired: Bool {
            lock.withLock { value }
        }
    }

    func testWatchdogFiresWhenBodyOutlivesDeadline() async {
        let fired = FiredBox()
        // 50ms deadline, ~400ms body: the watchdog must fire while the body is still "running".
        let result = await GTPCommand.withWatchdog(
            seconds: 0.05,
            onTimeout: { fired.fire() },
            body: {
                try? await Task.sleep(nanoseconds: 400_000_000)
                return 7
            }
        )
        XCTAssertEqual(result, 7, "the body's result is still returned after the watchdog fires")
        XCTAssertTrue(fired.fired, "a body that outlives the deadline must trip the watchdog")
    }

    func testWatchdogDoesNotFireWhenBodyBeatsDeadline() async {
        let fired = FiredBox()
        // Fast body under a generous 300ms deadline: the watchdog must be cancelled, never firing.
        let result = await GTPCommand.withWatchdog(
            seconds: 0.3,
            onTimeout: { fired.fire() },
            body: { 42 }
        )
        XCTAssertEqual(result, 42)
        // Wait well past the deadline to prove the cancelled watchdog stays silent.
        try? await Task.sleep(nanoseconds: 500_000_000)
        XCTAssertFalse(fired.fired, "a body that beat the deadline must never trip the watchdog")
    }

    func testWatchdogDisabledWhenSecondsZero() async {
        let fired = FiredBox()
        // seconds == 0 disables the watchdog entirely: even a slow body must not trip it.
        let result = await GTPCommand.withWatchdog(
            seconds: 0,
            onTimeout: { fired.fire() },
            body: {
                try? await Task.sleep(nanoseconds: 100_000_000)
                return true
            }
        )
        XCTAssertTrue(result)
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertFalse(fired.fired, "seconds == 0 must fully disable the watchdog")
    }

    func testLatchClaimsExactlyOnce() {
        let latch = WatchdogLatch()
        XCTAssertTrue(latch.claim(), "the first claim wins")
        XCTAssertFalse(latch.claim(), "every subsequent claim loses")
        XCTAssertFalse(latch.claim())
    }

    func testWatchdogSecondsFlagParsing() throws {
        // Default is the 180s tournament-safe value.
        XCTAssertEqual(try GTPCommand.parse(["-model", "m.bin.gz"]).genmoveWatchdogSeconds, 180)
        XCTAssertEqual(
            try GTPCommand.parse(["-model", "m.bin.gz", "-genmove-watchdog-seconds", "45"]).genmoveWatchdogSeconds,
            45
        )
        // 0 is accepted (disables the watchdog).
        XCTAssertEqual(
            try GTPCommand.parse(["-model", "m.bin.gz", "-genmove-watchdog-seconds", "0"]).genmoveWatchdogSeconds,
            0
        )
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "m.bin.gz", "-genmove-watchdog-seconds", "-1"]))
        XCTAssertThrowsError(try GTPCommand.parse(["-model", "m.bin.gz", "-genmove-watchdog-seconds", "nope"]))
    }
}
