#!/usr/bin/env bash
# llm_plan_gate.sh — PLAN/QUALITY GATE (§4.5 I5/I6, §6 crit.3): misura il decode (tok/s) EAGER vs
# le OPZIONI del registro (`--opt`) e DICHIARA ON/OFF — "vince la misura vs l'eager O resta OFF
# dichiarandolo". Metodo unit-first (§0sexies/§0septies): si misura l'unita' (decode), non si
# indovina. Warm-up incluso (la 1a run e' fredda per cache-prefix).
#
# Uso: MODEL=<dir MLX> [NP=2048] [NG=64] [THR=0.02] bash LLMEngine/tools/llm_plan_gate.sh [k=v ...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DYL="${LLM_MLX_DYLIB:-$ROOT/build/dist/osx-arm64/libllm_ai_mlx.dylib}"
MDL="${MODEL:-}"; NP="${NP:-2048}"; NG="${NG:-64}"; THR="${THR:-0.02}"
[ -n "$MDL" ] || { echo "uso: MODEL=<dir MLX> $0 [k=v ...]"; exit 2; }
[ -f "$DYL" ] || { echo "dylib assente: $DYL"; exit 2; }
BIN=/tmp/decode_profile; [ -x "$BIN" ] || cc -O2 -o "$BIN" "$ROOT/LLMEngine/tools/decode_profile.c"

tokps() { "$BIN" "$DYL" "$MDL" "$NP" "$NG" "$@" --quiet 2>/dev/null | awk '/^ *[0-9]/{v=$4} END{print v}'; }

tokps >/dev/null                       # warm-up
BASE=$(tokps)
echo "# eager base: ${BASE} tok/s (nprompt=$NP gen=$NG)"
for opt in "$@"; do
  tokps --opt "$opt" >/dev/null       # warm-up opzione
  V=$(tokps --opt "$opt")
  awk -v b="$BASE" -v v="$V" -v o="$opt" -v t="$THR" 'BEGIN{
    d = (b>0)? (v-b)/b : 0;
    if (d > t) printf "  %-24s %8.1f tok/s  -> ON  (vince: +%.1f%%)\n", o, v, d*100;
    else       printf "  %-24s %8.1f tok/s  -> OFF (non vince: %+.1f%%)\n", o, v, d*100;
  }'
done
