# RinGo

[![CI](https://github.com/kanade73/RinGo/actions/workflows/ci.yml/badge.svg)](https://github.com/kanade73/RinGo/actions/workflows/ci.yml)

KataGo's Go engine ported to Swift + [MLX](https://github.com/ml-explore/mlx-swift) for Apple
Silicon: model loading (v8–v17 incl. nested-bottleneck and transformer nets), a numerically
faithful MLX forward pass, V7 input features, a batched/compiled evaluator, PUCT MCTS, and a
GTP front-end.

Numerical fidelity is verified against the reference C++ implementation (Eigen backend) at
fp32: policy/value/score/ownership match within ≤1e-4 on all five reference test models, and
the current strongest public net (`kata1-zhizi-b40c768nbt`, 232M params) matches the oracle's
printed outputs exactly.

## What this is / what this is not

RinGo is an independent Swift + MLX implementation, not a fork; no KataGo C++ source is
included. No network weights are distributed—bring your own KataGo-format `.bin.gz` network
from https://katagotraining.org/.

## Requirements

- macOS 14 or later on Apple Silicon.
- Swift 6.2 or later, provided by a Swift toolchain or the Xcode Command Line Tools.
- A Metal device. The test suite exercises real MLX kernels.
- For `make check`: `brew install swiftformat swiftlint`. Building and running the engine itself
  needs neither.
- Only for the oracle path below (`make oracle`, `make goldens`, `Scripts/check-rawnn-parity.sh`):
  `cmake`, `brew install eigen libzip`, `python3`, and a KataGo checkout.

## Quick start

```sh
make setup                      # resolve SwiftPM deps (mlx-swift pinned exact 0.31.4)
Scripts/build-metallib.sh       # required once for SwiftPM-CLI runs (MLX base kernels)
make check                      # lint + build + full test suite (~2 min)

# Play via GTP (works with gogui / Sabaki / lizgoban as a GTP engine):
swift run -c release ringo gtp -model <net.bin.gz> -visits 400

# Raw NN eval of an SGF position (diffable against reference `evalsgf -raw-nn`):
swift run -c release ringo rawnn -model <net.bin.gz> -sgf game.sgf -move-num 30

# Throughput:
swift run -c release ringo benchmark -model <net.bin.gz> -batch-sizes 1,8,32
```

Real `kata1-*.bin.gz` nets from https://katagotraining.org/ (v8+) work at the default fp16
precision. The small nets under the reference clone's `cpp/tests/models/` are for parity testing
and need `-precision fp32`, since several produce non-finite policy output in fp16.

## Training and evaluation (9×9)

**Scope of this revision.** Inference, search, and GTP play run at any board size (RinGo played
a 19×19 tournament with a KataGo network — see below). **The training pipeline is 9×9 only**:
`ringo train` rejects shards whose `nnLen != 9`. Everything in this section is 9×9.

The supported route to a network of your own is **supervised distillation from a KataGo teacher,
starting from random initialisation** — no KataGo weights are converted or inherited. A teacher
network is used only to label positions; the student's parameters are initialised randomly and
trained by this repository's own MLX training loop (`Sources/RinGoTrain`).

### Pipeline

```sh
# 0. Build once.
make setup && Scripts/build-metallib.sh
swift build -c release --product ringo

# 1. Positions. Any SGF source works; self-play against an existing net is self-contained.
.build/release/ringo selfplay -model <teacher.bin.gz> -games 6 \
    -out data/sgf -visits 60 -size 9 -komi 7 -precision fp32

# 2. Label. Writes RinGoData v2 shards (`.nngd`) with policy / value / score / ownership
#    targets, plus a held-out split.
.build/release/ringo makedata -sgf-dir data/sgf -out data/shards -nnlen 9 \
    -teacher-model <teacher.bin.gz> -teacher-precision fp32 \
    -teacher-visits 1 -teacher-target completed-q -val-ratio 0.2

# 3. Train from random initialisation. The output directory must already exist.
mkdir -p runs/r1
.build/release/ringo train -data data/shards -out runs/r1 -arch b6c96 \
    -batch 32 -steps 40 -val-data data/shards -val-interval 20 -snapshot-interval 40

# 4. Evaluate on the held-out shard.
.build/release/ringo evaluate -model runs/r1/model-best-val.bin.gz -data data/shards/val.nngd

# 5. Play with the result.
.build/release/ringo gtp -model runs/r1/model-best-val.bin.gz -visits 100 -nnlen 9 -precision fp32
```

Architectures accepted by `-arch`: `b6c96`, `b10c128`, `b15c192`, `b20c256`, `b24c320`, `b28c384`.

Policy targets: `-teacher-target visits` is the normalised root visit distribution;
`completed-q` is the Gumbel completed-Q improved policy (Danihelka et al., ICLR 2022), built from
the root prior and per-child Q values.

### What a run produces

`runs/r1/` contains `config.json` (full hyperparameters), `training.log`, `metrics.csv`,
`validation_metrics.csv`, `checkpoint.safetensors` (resumable optimiser state),
`model-final.bin.gz`, and `model-best-val.bin.gz`. Exported models are in KataGo's `.bin.gz`
format, so the same file loads in RinGo and in reference KataGo.

`data/shards/makedata-config.json` records the labelling provenance (teacher model, visits,
target kind, rules, symmetries). The on-disk shard format itself (`.nngd`, "RinGoData v2") is
specified in [`Scripts/ringo-data-format.md`](Scripts/ringo-data-format.md).

### Holdout integrity

`makedata -val-ratio` writes the held-out split as `val.nngd` **alongside** the training shards.
`ringo train` excludes that path from its training set unconditionally — whether or not
`-val-data` was passed — so pointing `-data` at the directory cannot silently fold the holdout
into training, and a later `ringo evaluate` on the same file is a genuine holdout measurement.

### Worked example (verified 2026-07-27, Apple M5)

6 self-play games → 375 positions → 329 train / 46 held out. Training `b6c96` (1,027,911
parameters) from random initialisation for 40 steps, then evaluating the selected checkpoint:

```
evaluate: samples=46 total=7.15756 pol=4.35212 val=0.83451 score=0.03208 own=0.60254
```

which matches step 40 of `validation_metrics.csv` (`total_loss=7.157561`) — the standalone
evaluator and the in-training validation agree. The resulting network answers `genmove`.

This is a smoke-scale run, included to document the exact interface; it is far below the data and
step counts needed for a competitive network.

## Performance (Apple M5, 32 GB, release, fp16)

| model | batch | evals/s | reference (Eigen CPU, 8 threads) |
|---|---|---|---|
| b6c96 (1.0M) | 32 | ~3,000 | ~1,365 |
| kata1-zhizi-b40c768nbt (232M) | 32 | ~39 | ~4.8 |

19×19 genmove at 400 visits with b6c96: ~0.17 s/move.

The reference column is KataGo's Eigen **CPU** backend, so this is not a like-for-like
accelerator comparison — it is the same build these numbers are verified against, which is why it
is the baseline quoted here. For a fair GPU-to-GPU figure on a Mac, compare against KataGo's
OpenCL backend instead; that measurement is not included.

## Tournament results

- CGF Open 2026 (Japan), 2026-07-25, 9x9: 7th of 16 (6W-5L-1D), running RinGo's
  own trained network.
- 2026-07-26, 19x19: 4th of 12 (5W-2L), running the official KataGo network
  `kata1-b18c384nbt`. RinGo has no 19x19 network of its own; this was with the organiser's
  prior permission.
- Top-placed student entry on both days.

## Layout

- `Sources/RinGoCore` — board/rules/history + V7 features (pure Swift, no MLX; ported
  function-by-function from the C++ and verified against reference test vectors).
- `Sources/RinGoModel` — `.bin.gz`/`.txt.gz` model parser + `KataGoNetwork` MLX forward.
- `Sources/RinGoEngine` — postprocessing, SGF reader, batched `NNEvaluator` actor
  (compile-per-bucket and pipelined to keep MLX execution disciplined), MCTS, and GTP.
- `Sources/RinGoTrain` — training losses, optimizer, data loading, and trainer support.
- `Sources/ringo` — the `ringo` command-line executable and its subcommands.
- `Scripts/oracle/` — tools that extract ground truth from a reference KataGo build
  (`dump_v7`, `dump_nn`, `dump_nn_debug` per-block bisection).

The architecture separates pure-Swift Go logic, MLX model execution, search, training, and
the CLI; verification uses reference fixtures with explicit numerical tolerances and non-goals.
[`docs/design-notes.md`](docs/design-notes.md) covers the parts whose shape is not obvious from
the code: how MLX's lazy evaluation is kept under control, the numerical conventions parity
depends on, and what the optional DAG search does.

`Scripts/run-matches.sh` plays a match against another GTP program to measure strength; it drives
`gogui-twogtp`, which it expects at `../gogui/bin/gogui-twogtp` and which is not bundled here.

## Verification model

Ground truth is the reference C++ KataGo (read-only clone, Eigen CPU backend):
committed golden fixtures cover features (plane-exact), network outputs (≤1e-4 fp32), and
end-to-end `rawnn` output (18/18 position×model pairs). `make oracle` rebuilds the reference
binary; `make goldens` / `Scripts/gen-*-goldens.sh` regenerate fixtures; and
`Scripts/check-rawnn-parity.sh` re-runs those 18 end-to-end comparisons against it, reporting how
many pairs it actually checked.

Model-backed tests locate reference nets via `KATAGO_MODELS_DIR`, defaulting to
`../katago-origin/KataGo/cpp/tests/models` (a KataGo checkout beside this repo). Without those
nets, the tests use `XCTSkip` rather than fail: `make check` runs 438 tests in either setup,
reporting 40 skips when the nets are present and 92 skips in a fresh clone when they are absent.

## License

MIT (see [LICENSE](LICENSE)). Third-party attributions are listed in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
