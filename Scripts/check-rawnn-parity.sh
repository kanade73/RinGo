#!/usr/bin/env bash
set -euo pipefail
# A non-matching glob must not silently degrade into "checked nothing, exited 0" — this script
# is the end-to-end parity gate, so an empty fixture set is a failure, not a pass.
shopt -s nullglob

cd "$(dirname "$0")/.."

FIXTURE_DIR=Tests/RinGoCoreTests/Fixtures
GOLDEN_DIR=Tests/RinGoCoreTests/Goldens
ORACLE=${KATAGO_ORACLE:-.tools/katago-eigen-build/katago}
MODEL_DIR=${KATAGO_MODELS_DIR:-../katago-origin/KataGo/cpp/tests/models}
models=(
  g170-b6c96-s175395328-d26788732.bin.gz
  b4c256h4nbttflrs-fson-silu-rsnh.bin.gz
)

if [[ ! -x "$ORACLE" ]]; then
  echo "Oracle executable not found: $ORACLE" >&2
  exit 2
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/ringo-rawnn.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
cat >"$tmp/oracle.cfg" <<'EOF'
maxVisits = 1
numSearchThreads = 1
nnRandomize = false
logToStdout = false
EOF

status=0
checked=0
for sgf in "$FIXTURE_DIR"/*.sgf; do
  name=$(basename "$sgf" .sgf)
  if [[ "$name" == *encore-phase-2* ]]; then
    echo "SKIP $name (encore phase 2)"
    continue
  fi
  golden="$GOLDEN_DIR/$name.json"
  if [[ ! -f "$golden" ]]; then
    echo "SKIP $name (no golden metadata)"
    continue
  fi
  IFS=$'\t' read -r move_num rules nnlen < <(python3 - "$golden" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
print(g["moveNum"], json.dumps(g["rules"]), max(g["boardXSize"], g["boardYSize"]), sep="\t")
PY
  )
  rules=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]))' "$rules")

  for model_name in "${models[@]}"; do
    model="$MODEL_DIR/$model_name"
    label="$name x ${model_name%.bin.gz}"
    if [[ ! -f "$model" ]]; then
      echo "FAIL $label (model missing: $model)"
      status=1
      continue
    fi
    oracle_out="$tmp/oracle-$name-${model_name%%.*}.txt"
    swift_out="$tmp/swift-$name-${model_name%%.*}.txt"
    if ! "$ORACLE" evalsgf \
      -config "$tmp/oracle.cfg" -model "$model" -raw-nn -m "$move_num" \
      -override-rules "$rules" "$sgf" >"$oracle_out" 2>"$tmp/oracle.err"; then
      echo "FAIL $label (oracle failed)"
      cat "$tmp/oracle.err" >&2
      status=1
      continue
    fi
    if ! swift run ringo rawnn \
      -model "$model" -sgf "$sgf" -move-num "$move_num" -rules "$rules" \
      -optimism 0 -precision fp32 -nnlen "$nnlen" >"$swift_out"; then
      echo "FAIL $label (Swift rawnn failed)"
      status=1
      continue
    fi

    if python3 - "$oracle_out" "$swift_out" "$nnlen" <<'PY'
import re, sys

def parse(path, size):
    lines = open(path).read().splitlines()
    values = {}
    wanted = {
        "Win", "Loss", "NoResult", "ScoreMean", "ScoreMeanSq", "Lead",
        "VarTimeLeft", "STWinlossError", "STScoreError", "OptimismUsed",
    }
    policy_at = None
    for i, line in enumerate(lines):
        match = re.match(r"^(\w+)\s+([-+0-9.eE]+)(c?)$", line.strip())
        if match and match.group(1) in wanted:
            value = float(match.group(2))
            if match.group(3) == "c":
                value /= 100.0
            values[match.group(1)] = value
        if line == "Policy":
            policy_at = i
    if policy_at is None:
        raise ValueError(f"Policy block missing in {path}")
    pass_value = int(lines[policy_at + 1].split()[1])
    policy = []
    for line in lines[policy_at + 2:policy_at + 2 + size]:
        policy.extend(None if token == "-" else int(token) for token in line.split())
    owner = []
    for line in lines[policy_at + 2 + size:policy_at + 2 + 2 * size]:
        tokens = line.split()
        if len(tokens) != size:
            break
        owner.extend(int(token) for token in tokens)
    return values, policy + [pass_value], owner

oracle, swift = parse(sys.argv[1], int(sys.argv[3])), parse(sys.argv[2], int(sys.argv[3]))
errors = []
for key in ("Win", "Loss", "NoResult"):
    if abs(oracle[0][key] - swift[0][key]) > 0.005:
        errors.append(f"{key}: {oracle[0][key]} vs {swift[0][key]}")
for key in ("ScoreMean", "Lead"):
    if abs(oracle[0][key] - swift[0][key]) > 0.02:
        errors.append(f"{key}: {oracle[0][key]} vs {swift[0][key]}")
if len(oracle[1]) != len(swift[1]):
    errors.append("policy size differs")
else:
    for i, (a, b) in enumerate(zip(oracle[1], swift[1])):
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 3):
            errors.append(f"policy[{i}]: {a} vs {b}")
            break
if len(oracle[2]) != len(swift[2]):
    errors.append(f"ownership size differs: {len(oracle[2])} vs {len(swift[2])}")
else:
    for i, (a, b) in enumerate(zip(oracle[2], swift[2])):
        if abs(a - b) > 3:
            errors.append(f"ownership[{i}]: {a} vs {b}")
            break
if errors:
    print("; ".join(errors), file=sys.stderr)
    sys.exit(1)
PY
    then
      echo "PASS $label"
      checked=$((checked + 1))
    else
      echo "FAIL $label"
      status=1
    fi
  done
done

if ((checked == 0)); then
  echo "No position x model pair was checked — $FIXTURE_DIR/*.sgf matched nothing," >&2
  echo "or every fixture lacked a golden in $GOLDEN_DIR. Refusing to report success." >&2
  exit 1
fi

echo "checked $checked position x model pairs"
exit "$status"
