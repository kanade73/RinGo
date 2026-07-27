#!/usr/bin/env bash
# Strength evaluation: play our engine against an opponent GTP program and report results.
# Uses gogui-twogtp (bundled at ../gogui/bin/gogui-twogtp relative to the repo's parent).
#
# Usage:
#   Scripts/run-matches.sh -model <our-net.bin.gz> [options]
#
# Options (all optional except -model):
#   -model <file>       our net (.bin.gz) — REQUIRED
#   -opp  <gtp-cmd>     full opponent GTP command line, quoted. Default: reference KataGo Eigen
#                       at the same net (self-play sanity). Examples:
#                         -opp "gnugo --mode gtp --level 10"
#                         -opp "/path/katago gtp -model other.bin.gz -config cfg"
#   -games <N>          number of games, colors alternate (default 20)
#   -size  <N>          board size (default 9)
#   -komi  <f>          komi (default 7.0 for 9x9)
#   -visits <N>         our engine visits per move (default 200)
#   -oppvisits <N>      only used for the default reference opponent (default 200)
#   -precision fp16|fp32  our engine precision (default fp32; fp16 is faster but less robust)
#   -nnlen <auto|N>     our evaluator size (default auto; use 9 for faster 9x9 runs)
#   -max-batch-size <N> our engine max NN batch size (default 16; lower uses less memory)
#   -resign-enabled true|false  our engine's resign behavior (default true). An uncalibrated/newly
#                       trained net's value head can be badly miscalibrated, so a low-winrate
#                       resign can end games (and skew win rate) on noise rather than real play —
#                       pass false for evaluation runs of such nets (see TRAINING_PROCESS_AUDIT.md
#                       P0-6). The opponent and referee commands are unaffected by this flag.
#   -time <seconds>     our engine per-game time budget (GTP-side `-time`, self-budgeting across
#                       the game). Default unset — no -time flag is emitted (current behavior:
#                       visits-only budgeting). Only applied to the OURS command; -opp is a
#                       free-form command string, so for equal-TIME matches also embed the
#                       opponent's own time flag inside -opp (e.g. reference katago's
#                       maxTimePerMove-equivalent override), pass -time here for OURS, and set
#                       -visits very high so time (not visit count) is the binding constraint on
#                       our side.
#   -ponder on|off       our engine's ponder mode (GTP-side `-ponder`). Default unset — no -ponder
#                       flag is emitted (current behavior: no pondering). Only applied to the OURS
#                       command, same reasoning as -time above — embed the opponent's own pondering
#                       flag inside -opp if an equal-stack match needs it there too.
#   -symmetry off|hash|random  our engine's per-eval NN input symmetry (GTP-side `-symmetry`,
#                       KataGo `nnRandomize`). off (default, unset) = current behavior; hash =
#                       deterministic hash-selected; random = true per-node random symmetry.
#                       Only applied to the OURS command.
#   -symmetry-seed <n>  seed for `-symmetry random` (GTP-side `-symmetry-seed`). Unset keeps the
#                       engine's fixed default; set it to vary/reproduce a random-symmetry run.
#   -graph-search on|off  our engine's DAG/transposition search (GTP-side `-graph-search`, KataGo
#                       `useGraphSearch`). off (default, unset) = current pure-tree behavior. Only
#                       applied to the OURS command (same reasoning as -symmetry above); for an
#                       equal-stack A/B embed the opponent's own flag inside -opp if needed.
#   -referee <gtp-cmd>  optional referee GTP command for official final_score/RES_R
#   -maxmoves <N>       move cap per game (default 400, matches CGF rule)
#   -openings <dir>     directory of opening SGF files, cycled across games (gogui-twogtp
#                       `-openings`; with -alternate it uses each opening for one game per color
#                       before advancing). Without this, two deterministic engines play the SAME
#                       game every time — default is unset (no openings) for backward compat; pass
#                       Scripts/openings-9x9 for real evaluation runs.
#   -out <dir>          SGF + results output dir (default .tools/matches/<timestamp-less run>)
#
# Prints the gogui-twogtp result table (wins/losses, by color) and the SGF/dat location.
# NOTE: real matches are slow (each game = many NN evals). Start with -games 2 to smoke it.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
GT="${GOGUI_TWOGTP:-$REPO/../gogui/bin/gogui-twogtp}"
REF_KATAGO="${KATAGO_REF_BIN:-$REPO/.tools/katago-eigen-build/katago}"

# Homebrew's OpenJDK is keg-only on macOS, so /usr/bin/java may still fail even when Java is
# installed. Match TamaGo-mlx's gogui wrapper setup so gogui-twogtp can find Java by default.
if [ -z "${GOGUI_JAVA_HOME:-}" ] && [ -d /opt/homebrew/opt/openjdk ]; then
  export GOGUI_JAVA_HOME=/opt/homebrew/opt/openjdk
fi

model=""; opp=""; referee=""; games=20; size=9; komi=7.0; visits=200; oppvisits=200
nnlen=auto; max_batch_size=16
precision=fp32; maxmoves=400; out="$REPO/.tools/matches/run"; resign_enabled=true; openings=""
time_budget=""; ponder=""; num_search_workers=""; time_expected_moves=""; cpuct_exploration=""; symmetry=""; symmetry_seed=""; root_symmetries=""; graph_search=""
while [ $# -gt 0 ]; do
  case "$1" in
    -model) model="$2"; shift 2;;
    -opp) opp="$2"; shift 2;;
    -referee) referee="$2"; shift 2;;
    -games) games="$2"; shift 2;;
    -size) size="$2"; shift 2;;
    -komi) komi="$2"; shift 2;;
    -visits) visits="$2"; shift 2;;
    -oppvisits) oppvisits="$2"; shift 2;;
    -precision) precision="$2"; shift 2;;
    -nnlen) nnlen="$2"; shift 2;;
    -max-batch-size) max_batch_size="$2"; shift 2;;
    -num-search-workers) num_search_workers="$2"; shift 2;;
    -time-expected-moves) time_expected_moves="$2"; shift 2;;
    -cpuct-exploration) cpuct_exploration="$2"; shift 2;;
    -symmetry) symmetry="$2"; shift 2;;
    -symmetry-seed) symmetry_seed="$2"; shift 2;;
    -root-symmetries) root_symmetries="$2"; shift 2;;
    -graph-search) graph_search="$2"; shift 2;;
    -resign-enabled) resign_enabled="$2"; shift 2;;
    -time) time_budget="$2"; shift 2;;
    -ponder) ponder="$2"; shift 2;;
    -maxmoves) maxmoves="$2"; shift 2;;
    -openings) openings="$2"; shift 2;;
    -out) out="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[ -n "$model" ] || { echo "error: -model is required" >&2; exit 2; }
case "$resign_enabled" in
  true|false) ;;
  *) echo "error: -resign-enabled must be true or false (got '$resign_enabled')" >&2; exit 2;;
esac
case "$ponder" in
  ''|on|off) ;;
  *) echo "error: -ponder must be on or off (got '$ponder')" >&2; exit 2;;
esac
if [ -n "$openings" ]; then
  [ -d "$openings" ] || { echo "error: -openings dir not found: $openings" >&2; exit 2; }
  [ -n "$(find "$openings" -maxdepth 1 -name '*.sgf' -print -quit 2>/dev/null)" ] \
    || { echo "error: -openings dir has no .sgf files: $openings" >&2; exit 2; }
fi
[ -x "$GT" ] || { echo "error: gogui-twogtp not found/executable at $GT (set GOGUI_TWOGTP)" >&2; exit 2; }
[ -f "$REPO/default.metallib" ] || { echo "error: default.metallib missing — run Scripts/build-metallib.sh" >&2; exit 2; }

# Build our engine once (release) so the GTP command is a fast binary, not `swift run`.
swift build -c release --product ringo >/dev/null
OURBIN="$REPO/.build/release/ringo"
# gogui-twogtp launches each program with its own CWD = the launcher's CWD, so run from repo root
# where default.metallib lives.
ours="$OURBIN gtp -model $model -precision $precision -visits $visits -rules chinese -nnlen $nnlen -max-batch-size $max_batch_size -resign-enabled $resign_enabled"
# Not using an array for the optional -time/-ponder flags: macOS ships bash 3.2, where
# "${arr[@]}" on an EMPTY array trips "unbound variable" under `set -u` (same reason as
# OPENINGS_FLAG below).
[ -z "$time_budget" ] || ours="$ours -time $time_budget"
[ -z "$ponder" ] || ours="$ours -ponder $ponder"
[ -z "$num_search_workers" ] || ours="$ours -num-search-workers $num_search_workers"
[ -z "$time_expected_moves" ] || ours="$ours -time-expected-moves $time_expected_moves"
[ -z "$cpuct_exploration" ] || ours="$ours -cpuct-exploration $cpuct_exploration"
[ -z "$symmetry" ] || ours="$ours -symmetry $symmetry"
[ -z "$symmetry_seed" ] || ours="$ours -symmetry-seed $symmetry_seed"
[ -z "$root_symmetries" ] || ours="$ours -root-symmetries $root_symmetries"
[ -z "$graph_search" ] || ours="$ours -graph-search $graph_search"

if [ -z "$opp" ]; then
  [ -x "$REF_KATAGO" ] || { echo "error: default opponent needs the reference build; run 'make oracle'" >&2; exit 2; }
  # The reference KataGo gtp needs a full config; use its shipped example and override the few
  # keys we care about (a bare minimal config makes it abort on missing required keys).
  REF_CFG="${KATAGO_REF:-$REPO/../katago-origin/KataGo}/cpp/configs/gtp_example.cfg"
  [ -f "$REF_CFG" ] || { echo "error: reference gtp_example.cfg not found at $REF_CFG" >&2; exit 2; }
  opp="$REF_KATAGO gtp -model $model -config $REF_CFG -override-config maxVisits=$oppvisits,numSearchThreads=1,nnRandomize=false,chosenMoveTemperature=0.0,logDir=,logToStderr=false"
  echo "No -opp given; using reference KataGo (Eigen) at the SAME net as a sanity opponent."
fi

mkdir -p "$out"
echo "Ours : $ours"
echo "Opp  : $opp"
[ -z "$referee" ] || echo "Ref  : $referee"
echo "Games: $games  Size: $size  Komi: $komi  MaxMoves: $maxmoves  nnLen: $nnlen  MaxBatch: $max_batch_size  ResignEnabled: $resign_enabled"
[ -z "$openings" ] || echo "Openings: $openings"
echo "Out  : $out"

# Not using an array for the optional -openings flag: macOS ships bash 3.2, where "${arr[@]}" on an
# EMPTY array trips "unbound variable" under `set -u` (same reason as run-step-a.sh's VAL_FLAG).
# Unquoted expansion below word-splits the flag string; it is empty (a no-op) unless -openings was
# given, and $openings itself was already validated to exist above so it is safe unquoted here too.
OPENINGS_FLAG=""
[ -z "$openings" ] || OPENINGS_FLAG="-openings $openings"

if [ -n "$referee" ]; then
  "$GT" -black "$ours" -white "$opp" -referee "$referee" -alternate -auto -games "$games" \
    -size "$size" -komi "$komi" -maxmoves "$maxmoves" $OPENINGS_FLAG \
    -sgffile "$out/game" -verbose 2>"$out/gtp.log" || true
else
  "$GT" -black "$ours" -white "$opp" -alternate -auto -games "$games" \
    -size "$size" -komi "$komi" -maxmoves "$maxmoves" $OPENINGS_FLAG \
    -sgffile "$out/game" -verbose 2>"$out/gtp.log" || true
fi

# gogui-twogtp writes <prefix>.dat with a results table and a summary at the end.
if [ -f "$out/game.dat" ]; then
  echo "=== results (tail of $out/game.dat) ==="
  tail -n 20 "$out/game.dat"
  echo "=== summary (uses RES_R when a referee is configured) ==="
  awk -F'\t' '
    !/^#/ && NF >= 12 {
      res = ($4 != "" && $4 != "?" ? $4 : $2)
      if ($4 == "" || $4 == "?") missingRef = 1
      if ($6 != "-") dup++
      winner = substr(res, 1, 1)
      if (winner != "B" && winner != "W") {
        unknown++
        next
      }
      # gogui-twogtp -alternate writes RES_* normalized back to the original command slots:
      # B = our -black command, W = the opponent -white command.
      # Do not apply ALT again here; that double-corrects swapped-color games.
      if (winner == "B") ours++
      else opp++
    }
    END {
      total = ours + opp + unknown
      printf "ours %d - opp %d - draw/unknown %d (total %d, duplicates %d)\n", ours+0, opp+0, unknown+0, total+0, dup+0
      if (missingRef) {
        print "WARNING: RES_R is missing for at least one game; pass -referee for reliable scoring."
      }
    }' "$out/game.dat"
else
  echo "No .dat produced — see $out/gtp.log for errors."
fi
