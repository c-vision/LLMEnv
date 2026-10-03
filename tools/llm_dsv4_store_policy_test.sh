#!/usr/bin/env bash
# llm_dsv4_store_policy_test.sh — D19-follow-up A: ORACOLO offline della policy dello store dsv4.
# Verifica: eviction LRU (default) vs hotness min-freq (opt-in), feed osservata in claim,
# l'esperto caldo sopravvive. Nessun modello, nessun file, deterministico.
# Uso: tools/llm_dsv4_store_policy_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/LLMEngine/mlx/tests/llm_dsv4_store_policy_test.c"
BIN="${TMPDIR:-/tmp}/llm_dsv4_store_policy_test"
cc -std=gnu99 -O1 -Wall -Wextra \
   -I"$ROOT/LLMEngine/mlx" -I"$ROOT/LLMCodec/include" -I"$ROOT/LLMEngine/Options" \
   "$SRC" -o "$BIN"
"$BIN"
