#!/bin/bash
# build_qmm_mscale.sh — compila la sonda M-scaling di mlx_quantized_matmul (M=1..32).
# Linka le statiche MLX + frameworks; libmlx e' C++ -> serve -lc++.
# Uso: bash LLMEngine/tools/build_qmm_mscale.sh [build_dir]   (default LLMEngine/mlx/build-gmbasic)
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
B="${1:-$ROOT/LLMEngine/mlx/build-gmbasic}"
MLXC="$ROOT/libs/mlx-deps/mlx-c-src"; MLXS="$ROOT/libs/mlx-deps/mlx-src"
[ -e "$B/_deps/mlx_c-build/libmlxc.a" ] || { echo "libmlxc.a non in $B/_deps" >&2; exit 2; }
INC="-I$MLXC -I$MLXS -I$B/_deps/metal_cpp-src -I$ROOT/LLMEngine/mlx"
STATIC=("$B/_deps/mlx_c-build/libmlxc.a" "$B/_deps/mlx-build/libmlx.a" \
        "$B/_deps/cjson-build/libcjson.a" "$B/_deps/mlx-build/mlx/io/libgguflib.a" "$B/jaccl/libjaccl.a")
OUT="$ROOT/LLMEngine/tools/llm_qmm_mscale_probe"
clang -std=c17 -O2 -Wno-deprecated-declarations $INC "$ROOT/LLMEngine/tools/llm_qmm_mscale_probe.c" \
  "${STATIC[@]}" -lc++ -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
  -o "$OUT" && echo "OK -> $OUT"
