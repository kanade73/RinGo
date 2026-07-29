#!/usr/bin/env bash
# Build the oracle dumpers (dump_v7 features, dump_nn raw net outputs) against the
# reference KataGo Eigen build's object files.
# Prereq: `make oracle` (or any cmake Eigen build of the reference tree).
# Usage: Scripts/oracle/build-dumper.sh [BUILD_DIR]
set -euo pipefail
cd "$(dirname "$0")/../.."
KG=${KATAGO_REF:-../katago-origin/KataGo}
B=${1:-.tools/katago-eigen-build}
LIBZIP=${LIBZIP_LIB:-$(brew --prefix libzip)/lib/libzip.dylib}
[ -f "$LIBZIP" ] || { echo "libzip not found at $LIBZIP — 'brew install libzip', or set LIBZIP_LIB" >&2; exit 1; }
OBJS=$(find "$B/CMakeFiles/katago.dir" -name "*.o" ! -name "main.cpp.o" ! -path "*/command/*" ! -path "*/distributed/*" ! -path "*/tests/*")
for tool in dump_v7 dump_nn; do
  c++ -std=c++17 -O2 -fsigned-char -I"$KG/cpp" -I"$KG/cpp/external" -DUSE_EIGEN_BACKEND \
    -c "Scripts/oracle/$tool.cpp" -o "$B/$tool.o"
  # shellcheck disable=SC2086
  c++ -O2 -arch arm64 "$B/$tool.o" $OBJS -o ".tools/$tool" \
    "$(xcrun --show-sdk-path)/usr/lib/libz.tbd" "$LIBZIP"
  echo "built .tools/$tool"
done

# dump_nn_debug: same as dump_nn but with the reference backend recompiled to print
# per-block/per-layer activation summaries (stdout, before the final JSON line).
EIGEN_INC=$(brew --prefix eigen)/include/eigen3
c++ -std=c++17 -O3 -fsigned-char -I"$KG/cpp" -I"$KG/cpp/external" -I"$EIGEN_INC" \
  -DUSE_EIGEN_BACKEND -DDEBUG_INTERMEDIATE_VALUES -DNDEBUG \
  -c "$KG/cpp/neuralnet/eigenbackend.cpp" -o "$B/eigenbackend_debug.o"
OBJS_NO_EIGEN=$(printf '%s\n' $OBJS | grep -v 'neuralnet/eigenbackend.cpp.o')
# shellcheck disable=SC2086
c++ -O2 -arch arm64 "$B/dump_nn.o" "$B/eigenbackend_debug.o" $OBJS_NO_EIGEN -o ".tools/dump_nn_debug" \
  "$(xcrun --show-sdk-path)/usr/lib/libz.tbd" "$LIBZIP"
echo "built .tools/dump_nn_debug"
