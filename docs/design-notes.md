# Design notes

Background for the parts of this port whose *shape* is not obvious from the code: how MLX's lazy
evaluation is kept under control, which numerical conventions parity depends on, and what the
optional DAG search does. Source comments point here rather than restating any of it.

This file is part of the public repository; see `README.md` for build, usage, and verification.

## MLX discipline (lazy evaluation)

`MLXArray` values are nodes in a lazily-built graph. Nothing computes until something forces
materialization — `eval()`, `asyncEval()`, `.item()`, `.asArray()`, printing, or any control flow
that branches on a value. Left unmanaged this produces two failure modes that are easy to ship and
hard to see: graphs that keep growing because nothing ever collapses them, and per-layer syncs that
serialize the GPU and destroy throughput while every test still passes.

The port therefore confines MLX to two "islands", both built from the same forward code path
(`KataGoNetwork.forwardImpl`). A change to that function affects both, so both have to be
re-verified whenever it is touched: oracle parity for inference, the training smoke tests for
training.

### Island 1 — inference (`NNEvaluator`, used by GTP, search, and `benchmark`)

1. **Nothing escapes the island.** Every `MLXArray` lives inside `NNEvaluator`, a dedicated actor.
   Inputs arrive as Swift `[Float]` buffers, outputs leave as plain Swift structs. No `MLXArray`
   crosses the boundary — that is what prevents accidental graph growth and cross-thread retain
   cycles.
2. **One `eval()` per batch.** Trunk and both heads are built as a single graph and materialized
   once per physical batch. Never per layer; no `.item()` or `.asData()` mid-graph.
3. **`compile()` per batch bucket.** Batch sizes are bucketed to powers of two up to
   `nnMaxBatchSize`, padding the tail with zero rows and zero masks so shapes stay static. The
   forward is `MLX.compile`d once per bucket and reused for the evaluator's lifetime. Weights are
   `eval()`'d once at load and captured as constants.
4. **Pipelining.** `asyncEval` on batch N overlaps with CPU-side feature extraction for batch N+1;
   a double-buffered staging pair per bucket avoids allocation churn.
5. **Memory is opt-in, not defaulted.** `gpuCacheLimitMB` is an init parameter (and a CLI flag)
   that defaults to unset. MLX's own default is unbounded cache growth, which is fine for a short
   benchmark and not fine for an hours-long GTP or self-play run sharing unified memory with the
   rest of the system.
6. **No hidden syncs.** Anything that reads array contents inside the forward path — value-dependent
   control flow, `.item()`, printing — is banned and checked by review.

### Island 2 — training (`Trainer` / `TrainableNetwork`, used by `ringo train`)

The same discipline, adapted to a differentiable step:

1. **Shared forward code, injected parameters.** `TrainableNetwork` calls the same
   `KataGoNetwork.forwardImpl` that inference uses, through an `@_spi(Training)` entry point that
   substitutes a differentiable parameter lookup for the baked-in evaluated constants. That is what
   makes "the inference graph doubles as the training graph" true in code rather than only in
   description. Scope is plain conv + global-pooling fixup nets; transformer, nested-bottleneck, and
   RMSNorm descriptors are rejected with an explicit error.
2. **One `eval()` per step.** The closure passed to `valueAndGrad` returns every loss component, so
   the breakdown comes back with the gradient. There is no second forward pass for logging;
   `metricsInterval` gates only whether components are printed, never whether they are computed.
   Running forward twice per step to recover a metrics breakdown is an easy-to-miss regression worth
   tens of percent of wall clock.
3. **The whole step is compiled.** Forward, `valueAndGrad`, gradient clipping, and
   `optimizer.update` are wrapped in one `MLX.compile(inputs: state, outputs: state)` with
   `state = [network.store, optimizer]` — MLX's own canonical training-loop pattern, built once
   outside the loop. Batch shape must stay constant call to call for the trace to be reused. Note
   that mlx-swift 0.31.4 re-reads `optimizer.learningRate` on every compiled call rather than baking
   it into the trace, so a learning-rate schedule is honoured under `compile()`.
4. **Batch loading is not pipelined against compute.** No overlap between CPU-side assembly of step
   N+1 and GPU compute for step N. Acceptable at the scale this trains at; worth revisiting if batch
   size or per-sample CPU cost grows enough to matter next to a forward+backward pass.
5. **fp32 only.** There is no fp16 training path — it would need loss scaling, which is not
   implemented.
6. **The dataset is resident, not streamed.** Shards load into RAM up front. `TrainingPerformanceProbes`
   in the test suite measures what that costs for the columnar representation actually used versus a
   materialized `[RinGoSample]` array; the gap is the reason the columnar form exists.

## Numerical conventions

These are the conventions parity actually depends on. Changing any of them changes outputs.

- **TF32 must stay off.** mlx-swift 0.31.x enables TF32 by default, which silently degrades even
  fp32 GEMM and SDPA to reduced precision (order 1e-2 relative error) on capable GPUs.
  `KataGoNetwork.init` latches `MLX_ENABLE_TF32=0`. Removing it breaks oracle parity.
- **BatchNorm is folded at load time** into a per-channel scale and bias, matching the reference
  implementation's merged form. Running-statistics BN is never evaluated at inference.
- **Layout is NHWC**, matching the reference Eigen backend's NSC ordering rather than the NCHW that
  its CUDA backend uses. Shapes are documented at every MLX boundary; shape mismatches are the most
  common source of MLX bugs.
- **Heads cast to fp32 before postprocessing**, whatever the trunk precision.
- **fp16 is the default for real networks and is not safe for every checkpoint.** Some small
  synthetic test networks overflow fp16 to non-finite policy logits. Postprocessing treats a
  non-finite policy sum as a recoverable error — fall back or reject, never crash — and the parity
  fixtures run at fp32.
- **fp32 `compile()` warmup can be far slower than fp16's** for the same shapes, so long-lived
  processes pre-warm every bucket at startup rather than paying it mid-game.

## Graph search (DAG with transpositions)

Off by default (`-graph-search off`); when off, the tree path is unchanged.

Enabled, the search becomes a DAG: positions are identified by a superko-safe chained hash
(`GraphHash`, ported from the reference implementation's `graphhash`), and nodes reached by
different move orders are shared through a single node table. Because this port's search is a
single-threaded actor rather than a thread pool, none of the reference implementation's atomics,
mutex pools, or table sharding are needed — one dictionary keyed by hash suffices. What does carry
over is the accounting:

- **Edge visits are tracked separately from node visits.** A shared node's visit count is not the
  sum of the ways it was reached, so each parent-child edge carries its own count and children
  catch up as they are revisited.
- **Backup recomputes rather than accumulates.** A node's statistics are rebuilt from scratch from
  its children's weighted values plus its own evaluation at weight 1, which is what keeps sharing
  consistent when several parents update the same node.
- **Node identity is approximate; legality is exact.** A playout carries the real board and history
  down the descent, so legality is always judged against the actual path, never against the hash.
  If a selected move is illegal on the current path, the evaluation is regenerated for that path.
  On returning to the root, legality is re-applied strictly to every child.
- **Cycles terminate the playout.** If a descent reaches a node already on the current path, the
  edge visit is counted and the playout stops. `childVisits >= edgeVisits` is deliberately *not* an
  invariant under this rule.
- **Re-rooting garbage-collects by mark and sweep.** Advancing a move marks everything reachable
  from the new root with the current generation and deletes the rest.

This port has no cross-search evaluation cache, so unlike the reference implementation it gains not
only shared statistics from transpositions but also the avoided duplicate network evaluations.

Deliberately not ported, because the features they modify do not exist here: subtree value bias,
pattern bonus, noise pruning, the value-weight-exponent and uncertainty machinery, and the
cross-search evaluation cache.

## Verification and non-goals

Ground truth is the reference C++ implementation built with its Eigen CPU backend, treated as
read-only. Fixtures are committed at three levels — input features (plane-exact), network outputs
(within 1e-4 at fp32), and end-to-end `rawnn` output — and regenerated by `make goldens` and
`Scripts/check-rawnn-parity.sh`. A green `make check` proves compilation and unit logic; it does not
prove numerical agreement with the reference implementation, which is what the oracle scripts are
for.

Search intentionally implements a subset of the reference implementation's behaviour. The skipped
items are listed with reasons in `Sources/RinGoEngine/SearchSettings.swift`.
