#!/bin/zsh
# dsv41_engram_test.sh — verifica load-free dell'integrazione Engram nel forward MLX.
# Compila/linka a libmlxc.a/libmlx.a ed esegue il confronto col riferimento tiny. Atteso: ALL PASS.
# Uso: tools/dsv41_engram_test.sh
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
BIN=/tmp/llm_dsv41_engram_test

/usr/bin/cc -std=gnu11 -O2 -Wno-unused-function -c \
  -I"$SRC" -I"$ROOT/libs/mlx-deps/mlx-c-src" -I"$ROOT/libs/mlx-deps/mlx-src" \
  -I"$SRC/build/_deps/metal_cpp-src/include" \
  "$SRC/tests/llm_dsv41_mlx_engram_test.c" -o /tmp/dsv41_engram_test.o

/usr/bin/c++ -O2 /tmp/dsv41_engram_test.o \
  "$SRC"/build/_deps/mlx_c-build/libmlxc.a "$SRC"/build/_deps/mlx-build/libmlx.a \
  "$SRC"/build/_deps/cjson-build/libcjson.a \
  "$SRC"/build/_deps/mlx-build/mlx/io/libgguflib.a "$SRC"/build/jaccl/libjaccl.a \
  -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
  -o "$BIN"

"$BIN"
