#!/bin/zsh
# dsv41_engram_table_test.sh — verifica loader righe tabella Engram. ALL PASS.
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
[[ -f /tmp/dsv41_engram_table.safetensors ]] || "$HOME/ai/env-omlx/bin/python" "$ROOT/tools/gen_dsv41_engram_table_ref.py" >/dev/null
/usr/bin/cc -O2 -I"$SRC" -I"$SRC/build/_deps/cjson-src" -o /tmp/llm_dsv41_engram_table_test \
  "$SRC/tests/llm_dsv41_engram_table_test.c" "$SRC/build/_deps/cjson-build/libcjson.a" -lm
/tmp/llm_dsv41_engram_table_test
