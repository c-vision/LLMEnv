#!/bin/zsh
# dsv41_engine_test.sh — verifica integrazione engine (ARCH_TABLE) DeepSeek V4.1.
# Richiede il build del target gmb_ai_mlx (build/libgmb_ai_mlx.dylib) e il checkpoint sintetico
# /tmp/dsv41_st/sharded (tools/gen_dsv41_st_ref.py). Atteso: ALL PASS.
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
BIN=/tmp/llm_dsv41_engine_test
# fixture (idempotente)
[[ -f /tmp/dsv41_st/sharded/config.json ]] || "$HOME/ai/env-omlx/bin/python" "$ROOT/tools/gen_dsv41_st_ref.py" >/dev/null
/usr/bin/cc -std=gnu11 -O2 -I"$SRC" "$SRC/tests/llm_dsv41_engine_test.c" \
  -L"$SRC/build" -lgmb_ai_mlx \
  "$SRC"/build/_deps/mlx_c-build/libmlxc.a "$SRC"/build/_deps/mlx-build/libmlx.a \
  "$SRC"/build/_deps/cjson-build/libcjson.a \
  "$SRC"/build/_deps/mlx-build/mlx/io/libgguflib.a "$SRC"/build/jaccl/libjaccl.a \
  -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
  -Wl,-rpath,"$SRC/build" -o "$BIN"
"$BIN"
