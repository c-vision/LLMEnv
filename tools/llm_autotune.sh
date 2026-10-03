#!/usr/bin/env bash
# llm_autotune.sh — AUTOTUNER (I5) + GMB OPTIMIZATION RECORD (§4.5): misura il decode per una GRIGLIA
# di configurazioni (eager + opzioni del registro), sceglie la MIGLIORE e scrive il RECORD (JSON).
# Esecuzione reale: decide per misura, non per intuizione.
#
# Metrica (oracoli.md): **ms/token** (stabile), NON tok/s. Con speculazione ON i tok/s non sono
# confrontabili tra run diverse; la misura e' ripetuta (REPS) e si riporta min/mediana/max.
#
# Uso: MODEL=<dir MLX> [NP=2048] [NG=64] [REPS=2] [OUT=/tmp/opt_record.json] bash tools/llm_autotune.sh [cfg ...]
#   cfg: "nome:k=v,k=v" (vuoto = eager). Es: bash tools/llm_autotune.sh "eager:" "spec:spec_ngram=4" "kvq:kvquant=8:512"
set -euo pipefail
export LC_ALL=C
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DYL="${LLM_MLX_DYLIB:-$ROOT/build/dist/osx-arm64/libllm_ai_mlx.dylib}"
MDL="${MODEL:-}"; NP="${NP:-2048}"; NG="${NG:-64}"; REPS="${REPS:-2}"; OUT="${OUT:-/tmp/opt_record.json}"
[ -n "$MDL" ] || { echo "uso: MODEL=<dir MLX> $0 [nome:k=v,.. ...]"; exit 2; }
BIN=/tmp/decode_profile
if [ ! -x "$BIN" ] || [ "$ROOT/tools/decode_profile.c" -nt "$BIN" ]; then
  cc -O2 -o "$BIN" "$ROOT/tools/decode_profile.c"
fi
# ms/token = campo 4 (nprompt prefill_ms decode_ms ms/token tok/s act peak cache). Minimo su REPS.
mstok() { "$BIN" "$DYL" "$MDL" "$NP" "$NG" "$@" --quiet 2>/dev/null \
          | awk '/^ *[0-9]+ /{print $4}' | sort -n | awk 'NR==1{min=$1} {a[NR]=$1} END{print (min!=""?min:"nan")}'; }
minv() { awk 'BEGIN{m=1e18}{if($1+0>0 && $1+0<m)m=$1+0}END{print (m<1e18?m:"nan")}'; }

BEST=""; BESTV=1e18; REC=""
for cfg in "$@"; do
  name="${cfg%%:*}"; opts="${cfg#*:}"; args=""
  [ -n "$opts" ] && IFS=',' read -ra KV <<< "$opts" && for kv in "${KV[@]}"; do args+="--opt $kv "; done
  mstok $args >/dev/null                      # warmup (una run scartata)
  vals=""; for r in $(seq 1 "$REPS"); do vals+="$(mstok $args)"$'\n'; done
  med=$(printf '%s' "$vals" | minv)           # minimo (best-of) su REPS: robusto a una run anomala
  REC+="    {\"name\":\"$name\",\"opts\":\"$opts\",\"ms_per_token_min\":$med},"
  awk -v a="$med" -v b="$BESTV" 'BEGIN{exit !(a<b)}' && { BEST="$name"; BESTV="$med"; }
  if printf '%s' "$opts" | grep -qiE 'spec|dflash|mtp|ngram' && ! printf '%s' "$opts" | grep -q 'spec_ngram=0'; then
    echo "  # nota: '$name' attiva speculazione -> ms/token e acceptance, non tok/s"
  fi
done
REC="${REC%,}"
printf '{\n  "model": "%s",\n  "nprompt": %s, "ngen": %s, "reps": %s,\n  "metric": "ms_per_token_min",\n  "winner": "%s",\n  "ms_per_token": %s,\n  "trials": [\n%s\n  ]\n}\n' \
  "$MDL" "$NP" "$NG" "$REPS" "$BEST" "$BESTV" "$REC" > "$OUT"
echo "# AUTOTUNE: winner=$BEST ($BESTV ms/token) -> $OUT"
