#!/bin/zsh
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
/usr/bin/cc -O2 -I"$SRC" -o /tmp/llm_dsv41_dspark_test "$SRC/tests/llm_dsv41_dspark_test.c" -lm
/tmp/llm_dsv41_dspark_test
