#!/bin/zsh
# dsv41_engram_real_test.sh — verifica Engram reale (token_map file + primes + PCG64 + hash).
# Richiede il token_map generato da tools/gen_dsv41_engram_real_ref.py (nel dir modello).
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
MODEL=/Users/gaetano/ai/Models/DeepSeek-V4.1-Flash
[[ -f "$MODEL/engram_token_map.bin" ]] || "$HOME/ai/env-omlx/bin/python" "$ROOT/tools/gen_dsv41_engram_real_ref.py" >/dev/null
/usr/bin/cc -O2 -I"$SRC" -o /tmp/llm_dsv41_engram_real_test "$SRC/tests/llm_dsv41_engram_real_test.c" -lm
/tmp/llm_dsv41_engram_real_test "$MODEL/engram_token_map.bin"
