#!/usr/bin/env bash
# llm_autotune.sh — AUTOTUNER (I5) + GMB OPTIMIZATION RECORD (§4.5): misura il decode per una GRIGLIA
# di configurazioni (eager + opzioni del registro), sceglie la MIGLIORE (tok/s) e scrive il RECORD
# (JSON). Esecuzione reale: decide per misura, non per intuizione.
# Uso: MODEL=<dir MLX> [NP=2048] [NG=64] [OUT=/tmp/opt_record.json] bash tools/llm_autotune.sh [cfg ...]
#   cfg: "nome:k=v,k=v" (vuoto = eager). Es: bash tools/llm_autotune.sh "eager:" "spec:spec_ngram=4" "kvq:kvquant=8:512"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DYL="${LLM_MLX_DYLIB:-$ROOT/build/dist/osx-arm64/libllm_ai_mlx.dylib}"
MDL="${MODEL:-}"; NP="${NP:-2048}"; NG="${NG:-64}"; OUT="${OUT:-/tmp/opt_record.json}"
[ -n "$MDL" ] || { echo "uso: MODEL=<dir MLX> $0 [nome:k=v,.. ...]"; exit 2; }
BIN=/tmp/decode_profile; [ -x "$BIN" ] || cc -O2 -o "$BIN" "$ROOT/tools/decode_profile.c"
tok() { "$BIN" "$DYL" "$MDL" "$NP" "$NG" "$@" --quiet 2>/dev/null | awk '/^ *[0-9]/{v=$4} END{print v}'; }
tok >/dev/null
BEST=""; BESTV=0; REC=""
for cfg in "$@"; do
  name="${cfg%%:*}"; opts="${cfg#*:}"; args=""
  [ -n "$opts" ] && IFS=',' read -ra KV <<< "$opts" && for kv in "${KV[@]}"; do args+="--opt $kv "; done
  tok $args >/dev/null; v=$(tok $args)
  REC+="    {\"name\":\"$name\",\"opts\":\"$opts\",\"tok_s\":$v},"
  awk -v a="$v" -v b="$BESTV" 'BEGIN{exit !(a>b)}' && { BEST="$name"; BESTV="$v"; }
done
REC="${REC%,}"
printf '{\n  "model": "%s",\n  "nprompt": %s, "ngen": %s,\n  "winner": "%s",\n  "tok_s": %s,\n  "trials": [\n%s\n  ]\n}\n' "$MDL" "$NP" "$NG" "$BEST" "$BESTV" "$REC" > "$OUT"
echo "# AUTOTUNE: winner=$BEST ($BESTV tok/s) -> $OUT"
