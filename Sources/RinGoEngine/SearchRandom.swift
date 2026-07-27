import Foundation

/// Small deterministic generator used for reproducible self-play. SplitMix64 is also used by
/// existing benchmark and host-test helpers in this repository; keeping the algorithm here makes
/// search noise and final-move sampling share one explicitly seeded stream.
struct SearchRandom {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    mutating func unitDouble() -> Double {
        // The half-unit offset keeps the result strictly inside (0, 1), which avoids log(0) in
        // the gamma and Box-Muller transforms while retaining 53 deterministic random bits.
        (Double(next() >> 11) + 0.5) * 0x1.0p-53
    }
}

enum SearchSampling {
    /// Returns normalized priors mixed with a symmetric Dirichlet draw. Illegal moves must be
    /// excluded by the caller. Both the returned vector and the generated noise sum to one.
    static func mixDirichlet(
        priors: [Double],
        alpha: Double,
        weight: Double,
        random: inout SearchRandom
    ) -> [Double] {
        precondition(!priors.isEmpty)
        precondition(alpha > 0 && alpha.isFinite)
        precondition((0 ... 1).contains(weight) && weight.isFinite)

        let positive = priors.map { max(0, $0) }
        let priorSum = positive.reduce(0, +)
        let normalized = priorSum > 0
            ? positive.map { $0 / priorSum }
            : [Double](repeating: 1 / Double(priors.count), count: priors.count)
        let gamma = priors.map { _ in gammaSample(shape: alpha, random: &random) }
        let gammaSum = gamma.reduce(0, +)
        let noise = gammaSum > 0 && gammaSum.isFinite
            ? gamma.map { $0 / gammaSum }
            : [Double](repeating: 1 / Double(priors.count), count: priors.count)
        return zip(normalized, noise).map { prior, draw in
            (1 - weight) * prior + weight * draw
        }
    }

    /// Samples an index proportional to `visitCount^(1 / temperature)`. Temperature zero is the
    /// stable first-maximum argmax used by search before self-play support was added.
    static func sampleVisitIndex(
        visits: [Int64],
        temperature: Double,
        random: inout SearchRandom
    ) -> Int? {
        guard !visits.isEmpty else { return nil }
        if temperature <= 0 {
            var best = visits.startIndex
            for index in visits.indices.dropFirst() where visits[index] > visits[best] {
                best = index
            }
            return best
        }
        precondition(temperature.isFinite)

        let positiveIndices = visits.indices.filter { visits[$0] > 0 }
        guard !positiveIndices.isEmpty else { return visits.startIndex }
        let inverseTemperature = 1 / temperature
        let maxLogWeight = positiveIndices.map { log(Double(visits[$0])) * inverseTemperature }.max()!
        let weights = visits.indices.map { index in
            visits[index] > 0
                ? exp(log(Double(visits[index])) * inverseTemperature - maxLogWeight)
                : 0
        }
        let total = weights.reduce(0, +)
        var target = random.unitDouble() * total
        for index in weights.indices where weights[index] > 0 {
            target -= weights[index]
            if target <= 0 { return index }
        }
        return positiveIndices.last
    }

    private static func gammaSample(shape: Double, random: inout SearchRandom) -> Double {
        if shape < 1 {
            return gammaSample(shape: shape + 1, random: &random)
                * pow(random.unitDouble(), 1 / shape)
        }

        // Marsaglia and Tsang's method for Gamma(shape, 1), shape >= 1.
        let d = shape - 1 / 3
        let c = 1 / sqrt(9 * d)
        while true {
            let normal = sqrt(-2 * log(random.unitDouble()))
                * cos(2 * Double.pi * random.unitDouble())
            let base = 1 + c * normal
            if base <= 0 { continue }
            let cube = base * base * base
            let uniform = random.unitDouble()
            let acceptedQuickly = uniform < 1 - 0.0331 * normal * normal * normal * normal
            let acceptedExactly = log(uniform) < 0.5 * normal * normal + d * (1 - cube + log(cube))
            if acceptedQuickly || acceptedExactly {
                return d * cube
            }
        }
    }
}
