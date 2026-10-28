#!/bin/zsh
# dsv41_metpack_test.sh — verifica i kernel di packing Metal (fp8 grp32, fp4 gs16/gs32)
# contro il packing HOST (llm_dsv41_pack_fp8 / llm_dsv41_pack_activation). Atteso: ALL PASS.
# Uso: tools/dsv41_metpack_test.sh
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
BIN=/tmp/llm_dsv41_metpack_test
CXX=/usr/bin/c++

/usr/bin/cc -std=gnu11 -O2 -Wno-unused-function -c \
  -I"$SRC" -I"$ROOT/libs/mlx-deps/mlx-c-src" -I"$ROOT/libs/mlx-deps/mlx-src" \
  -I"$SRC/build/_deps/metal_cpp-src/include" \
  "$SRC/tests/llm_dsv41_metpack_test.c" -o /tmp/dsv41_metpack_test.o

"$CXX" -O2 /tmp/dsv41_metpack_test.o \
  "$SRC"/build/_deps/mlx_c-build/libmlxc.a "$SRC"/build/_deps/mlx-build/libmlx.a \
  "$SRC"/build/_deps/cjson-build/libcjson.a \
  "$SRC"/build/_deps/mlx-build/mlx/io/libgguflib.a "$SRC"/build/jaccl/libjaccl.a \
  -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
  -o "$BIN"

"$BIN"
