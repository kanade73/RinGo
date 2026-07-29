import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// AdamW work-alike whose learning rate and bias-correction step counter are `MLXArray` entries in
/// `innerState()`, not Swift `Float`s.
///
/// ## Why this exists
/// `Trainer`'s compiled training step traces the whole forward+backward+optimizer-update graph
/// once via `MLX.compile(inputs:outputs:)`. On every later call with the same input shapes, MLX
/// replays the CACHED Metal graph without re-invoking the Swift closure that built it
/// (`Transforms+Compile.swift`'s `CompiledFunction.innerCall`: `inner(tracers:)`, the only place
/// that reads Swift-captured scalars, runs solely on a genuine cache miss). `MLXOptimizers.AdamW`
/// keeps `learningRate` as a plain Swift `Float`, so it gets baked into that one trace as an
/// arithmetic CONSTANT -- every later call replays that SAME lr value forever, freezing
/// warmup/cosine schedules at whatever they were during the one-time retrace. All prior training
/// runs were effectively constant-lr.
///
/// `MLXArray`s that are part of an optimizer's `innerState()` don't have this problem: on EVERY
/// call (hit or miss), `innerCall` rebinds `stateInputs` to whatever the current Swift-side arrays
/// are (`inputs.flatMap { $0.innerState() }`, evaluated fresh at the top of every call) and feeds
/// them into the compiled kernel as real graph inputs, then writes the kernel's outputs back into
/// those same array objects via `MLXArray._updateInternal`. That is exactly how the per-parameter
/// Adam moments already flow correctly through the compiled step; this type puts `learningRate`
/// and the bias-correction `step` counter through the same channel (matching Python MLX, where
/// the learning rate lives in `optimizer.state`).
///
/// ## State layout
/// `innerState()` returns, in order: the per-parameter `(m, v)` moments in the canonical
/// sorted-key order `NestedDictionary` already uses (stable across calls because it is derived
/// from the model's own parameter names), THEN the global `learningRate` scalar, THEN the global
/// `step` scalar. The two globals are always appended last so the per-parameter section's length
/// (fixed by the model architecture) never shifts their position.
public final class RinGoAdamW: Optimizer {
    /// Per-parameter Adam moment state (first and second moment estimates).
    private struct MomentState: Updatable {
        var m: MLXArray
        var v: MLXArray

        init(zeros parameter: MLXArray) {
            m = MLXArray.zeros(like: parameter)
            v = MLXArray.zeros(like: parameter)
        }

        init(m: MLXArray, v: MLXArray) {
            self.m = m
            self.v = v
        }

        func innerState() -> [MLXArray] {
            [m, v]
        }
    }

    private var stateStorage = NestedDictionary<String, MomentState>()
    /// Learning rate held as an `MLXArray` -- see the type doc comment. The setter REPLACES this
    /// property with a new array (rather than mutating in place); since `innerState()` reads
    /// whatever this property currently holds at call time, that is sufficient for the fresh value
    /// to reach the compiled step every time `Trainer` sets it before invoking that step.
    private var learningRateArray: MLXArray
    /// Global bias-correction step counter, incremented inside the (lazy, compiled) update graph
    /// itself -- mirrors `MLXOptimizers.AdamState`'s per-parameter `step`, just shared once across
    /// all parameters instead of duplicated per tensor.
    private var stepArray: MLXArray

    public var betas: (Float, Float)
    public var eps: Float
    public var weightDecay: Float

    public var learningRate: Float {
        get { learningRateArray.item(Float.self) }
        set { learningRateArray = MLXArray(newValue) }
    }

    public init(
        learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8,
        weightDecay: Float = 0.01
    ) {
        learningRateArray = MLXArray(learningRate)
        stepArray = MLXArray(0)
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
    }

    public func innerState() -> [MLXArray] {
        stateStorage.flattenedValues().flatMap { $0.innerState() } + [learningRateArray, stepArray]
    }

    /// Materializes zeroed `(m, v)` state for every key in `parameters` that doesn't already have
    /// state, without touching `learningRate`/`step` or any state that already exists. Used by
    /// `Trainer.loadCheckpoint` to give the state tree its final shape before restoring saved
    /// values positionally through `innerState()` -- replaces the "zero-lr dummy `update()` call"
    /// trick the old `MLXOptimizers.AdamW` needed (that type exposes no state setter at all).
    func primeState(for parameters: ModuleParameters) {
        for (key, parameter) in parameters.flattened() where stateStorage[unwrapping: key] == nil {
            stateStorage[unwrapping: key] = MomentState(zeros: parameter)
        }
    }

    public func update(model: Module, gradients: ModuleParameters) {
        // `[unwrapping:]` is a single-level lookup into the dictionary's OWN top-level keys, not a
        // dotted-path traversal -- `model.parameters()` is still the true hierarchical tree (e.g.
        // top key "values" holding a nested dictionary), so it must be flattened to the same
        // dotted-key form `gradients.flattened()` produces before keys line up.
        let modelParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        stepArray += 1
        let flatGradients = gradients.flattened()
        var newParameters: [(String, MLXArray)] = []
        newParameters.reserveCapacity(flatGradients.count)
        var newState = NestedDictionary<String, MomentState>()
        for (key, gradient) in flatGradients {
            guard let parameter = modelParameters[key] else {
                fatalError("RinGoAdamW: gradient for unknown parameter \(key)")
            }
            let state = stateStorage[unwrapping: key] ?? MomentState(zeros: parameter)
            let (newParameter, updatedState) = applySingle(
                gradient: gradient, parameter: parameter, state: state, key: key
            )
            newParameters.append((key, newParameter))
            newState[unwrapping: key] = updatedState
        }
        stateStorage = newState
        model.update(parameters: ModuleParameters.unflattened(newParameters))
    }

    /// Parameter key substrings excluded from decoupled weight decay: normalization affine params
    /// (`TrainableNetwork`'s `bn-scale:`/`bn-bias:` key namespace) and bias vectors (`bias:` keys --
    /// this substring also matches inside `bn-bias:`, which should be excluded anyway, so a single
    /// check covers both). Standard AdamW hygiene: decay only the weight kernels (`conv:`,
    /// `matmul:`), not biases or norm scale/shift.
    private static func isExcludedFromWeightDecay(_ key: String) -> Bool {
        key.contains("bias:") || key.contains("bn-scale:")
    }

    /// Mirrors `MLXOptimizers.Adam.applySingle`'s `biasCorrection: true` branch and
    /// `AdamW.applySingle`'s decoupled-decay wrapper exactly, substituting the MLXArray-backed
    /// `learningRateArray`/`stepArray` for the Swift-Float `learningRate` / per-parameter
    /// `state.step` the library versions use.
    private func applySingle(
        gradient: MLXArray, parameter: MLXArray, state: MomentState, key: String
    ) -> (MLXArray, MomentState) {
        let (b1, b2) = betas
        let m = b1 * state.m + (1 - b1) * gradient
        let v = b2 * state.v + (1 - b2) * square(gradient)

        let c1 = learningRateArray / (1 - pow(b1, stepArray))
        let c2 = rsqrt(1 - pow(b2, stepArray))
        let update = (c1 * m) / (sqrt(v) * c2 + eps)

        var decayedParameter = parameter
        if weightDecay != 0, !Self.isExcludedFromWeightDecay(key) {
            decayedParameter = parameter * (1 - learningRateArray * weightDecay)
        }

        return (decayedParameter - update, MomentState(m: m, v: v))
    }
}
