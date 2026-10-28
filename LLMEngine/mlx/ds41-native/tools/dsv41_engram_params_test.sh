#!/bin/zsh
# dsv41_engram_params_test.sh — verifica primes/offsets Engram vs num_embeddings reali. ALL PASS.
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
/usr/bin/cc -O2 -I"$SRC" -I"$ROOT/libs/mlx-deps/mlx-c-src" -I"$ROOT/libs/mlx-deps/mlx-src" \
  -o /tmp/llm_dsv41_engram_params_test "$SRC/tests/llm_dsv41_engram_params_test.c" -lm
/tmp/llm_dsv41_engram_params_test
