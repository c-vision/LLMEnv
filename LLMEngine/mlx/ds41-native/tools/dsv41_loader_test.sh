#!/bin/zsh
# dsv41_loader_test.sh — verifica del loader safetensors DeepSeek V4.1.
# Rigenera il fixture tiny (nessun download), compila `llm_dsv41_loader_test.c`, lo linka a
# libmlxc.a/libmlx.a/libcjson.a e lo esegue. Se esiste /tmp/dsv41_index.json (index reale HF),
# esegue anche la copertura dei nomi. Atteso: ALL PASS.
# Uso: tools/dsv41_loader_test.sh
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
BIN=/tmp/llm_dsv41_loader_test

"$HOME/ai/env-omlx/bin/python" "$ROOT/tools/gen_dsv41_st_ref.py" >/dev/null

/usr/bin/cc -std=gnu11 -O2 -Wno-unused-function -c \
  -I"$SRC" -I"$ROOT/libs/mlx-deps/mlx-c-src" -I"$ROOT/libs/mlx-deps/mlx-src" \
  -I"$SRC/build/_deps/metal_cpp-src/include" -I"$SRC/build/_deps/cjson-src" \
  "$SRC/tests/llm_dsv41_loader_test.c" -o /tmp/dsv41_loader_test.o

/usr/bin/c++ -O2 /tmp/dsv41_loader_test.o \
  "$SRC"/build/_deps/mlx_c-build/libmlxc.a "$SRC"/build/_deps/mlx-build/libmlx.a \
  "$SRC"/build/_deps/cjson-build/libcjson.a \
  "$SRC"/build/_deps/mlx-build/mlx/io/libgguflib.a "$SRC"/build/jaccl/libjaccl.a \
  -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
  -o "$BIN"

if [[ -f /tmp/dsv41_index.json ]]; then
  "$BIN" /tmp/dsv41_index.json /tmp/dsv41_st/tiny.safetensors
else
  "$BIN"
fi
