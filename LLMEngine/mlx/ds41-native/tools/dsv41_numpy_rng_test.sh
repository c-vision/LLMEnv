#!/bin/zsh
# dsv41_numpy_rng_test.sh — verifica del port C di np.random.default_rng. ALL PASS.
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
/usr/bin/cc -O2 -I"$SRC" -o /tmp/llm_dsv41_numpy_rng_test "$SRC/tests/llm_dsv41_numpy_rng_test.c" -lm
/tmp/llm_dsv41_numpy_rng_test
