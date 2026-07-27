#!/usr/bin/env bash
# Build MLX's base Metal kernel library (default.metallib) for SwiftPM-CLI runs.
#
# mlx-swift compiles most kernels at runtime (JIT), but needs a small base library
# that Xcode would normally compile from the .metal files in the Cmlx target.
# Plain `swift build`/`swift test` skips .metal sources entirely, so MLX aborts at
# startup with "Failed to load the default metallib". MLX's last-resort lookup is
# the relative path "default.metallib" (i.e. the process CWD), so we build the
# library once and drop it in the repo root, where `make check` / `swift test` run.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal
OUT_DIR=.tools/metallib-build
[ -d "$SRC" ] || { echo "mlx-swift checkout not found — run 'make setup' first" >&2; exit 1; }
mkdir -p "$OUT_DIR"
airs=()
while IFS= read -r -d '' f; do
  base=$(basename "$f" .metal)
  air="$OUT_DIR/$base.air"
  xcrun -sdk macosx metal -fno-fast-math -I "$SRC" -c "$f" -o "$air"
  airs+=("$air")
done < <(find "$SRC" -name "*.metal" -print0)
xcrun -sdk macosx metallib "${airs[@]}" -o default.metallib
echo "built default.metallib ($(du -h default.metallib | cut -f1)) from ${#airs[@]} kernels"
