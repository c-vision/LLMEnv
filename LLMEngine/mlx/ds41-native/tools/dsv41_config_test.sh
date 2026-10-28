#!/bin/zsh
# dsv41_config_test.sh — verifica del parsing config.json DeepSeek V4.1.
# Confronta con i config reali (inferenza + root) se presenti in /tmp. Atteso: ALL PASS.
# Uso: tools/dsv41_config_test.sh
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
BIN=/tmp/llm_dsv41_config_test

/usr/bin/cc -std=gnu11 -O2 -Wall \
  -I"$SRC" -I"$ROOT/libs/mlx-deps/mlx-c-src" -I"$ROOT/libs/mlx-deps/mlx-src" \
  -I"$SRC/build/_deps/cjson-src" \
  "$SRC/tests/llm_dsv41_config_test.c" "$SRC/build/_deps/cjson-build/libcjson.a" -lm -o "$BIN"

"$BIN"
