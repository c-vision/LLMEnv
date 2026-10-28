#!/usr/bin/env bash
# ab_order_control.sh — per ogni modello esegue ab_engine_rapid.sh nei DUE ordini
# (gmb-first / rapid-first) e stampa i due rapporti. Cancella il bias di warm-up: se i
# due rapporti concordano, il delta e' reale; se divergono, e' order-bias.
# Uso: tools/ab_order_control.sh [--ns 512] [--m 32] [--reps 2] [--port 8044] [--batch 1|2]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS=512; M=32; REPS=2; PORT=8044; BATCH=1
while [ $# -gt 0 ]; do case "$1" in
  --ns) NS="$2"; shift 2;; --m) M="$2"; shift 2;; --reps) REPS="$2"; shift 2;;
  --port) PORT="$2"; shift 2;; --batch) BATCH="$2"; shift 2;; *) echo "?$1"; exit 2;; esac; done
B1=(
"Qwen3-Coder-30B-A3B(controllo+)|/Users/gaetano/ai/Models/Qwen3-Coder-30B-A3B-Instruct-8bit"
"Ternary-Bonsai-4B|/Users/gaetano/ai/Models/Ternary-Bonsai-4B-mlx-2bit"
"gemma-4-31B|/Users/gaetano/ai/Models/gemma-4-31b-it-4bit"
)
B2=(
"Mixtral-8x22B|/Users/gaetano/ai/Models/Mixtral-8x22B-4bit"
"gpt-oss-120b|/Users/gaetano/ai/Models/gpt-oss-120b-mxfp4-bf16"
)
case "$BATCH" in 1) LIST=("${B1[@]}");; 2) LIST=("${B2[@]}");; *) LIST=("${B1[@]}" "${B2[@]}");; esac
TS="$(date +%Y%m%d_%H%M%S)"; LOG="/tmp/ab_order_$TS"; mkdir -p "$LOG"
say(){ printf '%s\n' "$*" | tee -a "$LOG/order.txt"; }
for entry in "${LIST[@]}"; do
  name="${entry%%|*}"; dir="${entry##*|}"
  [ -f "$dir/config.json" ] || { say "$name ASSENTE"; continue; }
  for ord in gmb-first rapid-first; do
    "$ROOT/tools/ab_engine_rapid.sh" --model "$dir" --ns "$NS" --m "$M" --reps "$REPS" \
      --port "$PORT" --order "$ord" >"$LOG/$name.$ord.log" 2>&1
    r=$(grep -oE 'GMB/rapid=[0-9.]+x' "$LOG/$name.$ord.log" | tail -1)
    g=$(grep -oE 'gmb=[0-9.]+' "$LOG/$name.$ord.log" | tail -1)
    p=$(grep -oE 'rapid=[0-9.]+' "$LOG/$name.$ord.log" | tail -1)
    say "$(printf '%-30s [%-11s] %s %s %s' "$name" "$ord" "$g" "$p" "$r")"
  done
done
say "log: $LOG"
