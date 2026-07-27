#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
models_dir=${KATAGO_MODELS_DIR:-../katago-origin/KataGo/cpp/tests/models}
dump_nn="$repo_dir/.tools/dump_nn"
sgf="$repo_dir/Tests/KataGoModelTests/Fixtures/opening.sgf"
fixtures="$repo_dir/Tests/KataGoModelTests/Fixtures"

generate() {
    model=$1
    output=$2
    optimism=${3:-0}
    "$dump_nn" "$models_dir/$model" "$sgf" 4 Chinese 19 "$optimism" > "$fixtures/$output"
}

generate g170-b6c96-s175395328-d26788732.bin.gz b6c96.json
generate g170e-b10c128-s1141046784-d204142634.bin.gz b10c128.json
generate b4c256h4nbttflrs-fson-silu-rsnh.bin.gz b4c256-transformer.json
generate b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz b7c96-gqa.json
generate b7c96h3tfrs-test5-cnorm.bin.gz b7c96-cnorm.json
generate b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz b7c96-gqa-optimism1.json 1
