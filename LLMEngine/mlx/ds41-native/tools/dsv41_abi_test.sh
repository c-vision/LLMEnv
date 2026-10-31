#!/bin/zsh
# dsv41_abi_test.sh — contratto ABI di ds41-native, model-free (nessun checkpoint richiesto).
# Richiede il build di llm_ai_mlx (cmake --build build --target llm_ai_mlx). Atteso: ALL PASS.
set -eu
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
LIB="$ROOT/build/LLMEngine/mlx"
BIN="${TMPDIR:-/tmp}/llm_dsv41_abi_test"
/usr/bin/cc -std=gnu11 -O2 -I"$ROOT/LLMEngine/include" \
  "$ROOT/LLMEngine/mlx/ds41-native/tests/llm_dsv41_abi_test.c" \
  -L"$LIB" -lllm_ai_mlx -Wl,-rpath,"$LIB" -o "$BIN"
"$BIN"
