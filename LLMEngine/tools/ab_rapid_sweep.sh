#!/usr/bin/env bash
# ab_rapid_sweep.sh — A/B APPAAIATO GMB vs rapid per tutti i modelli (aggiorna la tabella §13.3).
# Gira ab_engine_rapid.sh in sequenza (stessa finestra per modello) e raccoglie GMB/rapid.
# Uso: tools/ab_rapid_sweep.sh [--ns 512] [--m 32] [--reps 2] [--batch 1|2|all] [--port 8042]
# Log: /tmp/ab_sweep_<ts>/
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NS=512; M=32; REPS=2; BATCH=all; PORT=8042
while [ $# -gt 0 ]; do case "$1" in
  --ns) NS="$2"; shift 2;; --m) M="$2"; shift 2;; --reps) REPS="$2"; shift 2;;
  --batch) BATCH="$2"; shift 2;; --port) PORT="$2"; shift 2;; *) echo "?$1"; exit 2;; esac; done
TS="$(date +%Y%m%d_%H%M%S)"; LOG="/tmp/ab_sweep_$TS"; mkdir -p "$LOG"
say(){ printf '%s\n' "$*" | tee -a "$LOG/sweep.txt"; }

B1=(
"Llama-3.2-1B|/Users/gaetano/ai/Models/Llama-3.2-1B-Instruct-4bit"
"Ternary-Bonsai-4B|/Users/gaetano/ai/Models/Ternary-Bonsai-4B-mlx-2bit"
"gpt-oss-20b|/Users/gaetano/ai/Models/gpt-oss-20b-MXFP4-Q8"
"Qwen3.6-27B-4bit|/Users/gaetano/ai/Models/Qwen3.6-27B-4bit"
"gemma-4-31B|/Users/gaetano/ai/Models/gemma-4-31b-it-4bit"
"Qwen3.6-35B-A3B|/Users/gaetano/ai/Models/Qwen3.6-35B-A3B-OptiQ-4bit-textonly"
"Qwen3.5-Opus-27B|/Users/gaetano/ai/Models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit"
"DeepSeek-R1-32B|/Users/gaetano/ai/Models/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit"
"Qwen3-Coder-30B-A3B|/Users/gaetano/ai/Models/Qwen3-Coder-30B-A3B-Instruct-8bit"
"Qwen3.8-27B-8bit|/Users/gaetano/ai/Models/Qwen3.8-27B-8bit"
)
B2=(
"Mixtral-8x22B|/Users/gaetano/ai/Models/Mixtral-8x22B-4bit"
"GLM-4.5-Air|/Users/gaetano/ai/Models/GLM-4.5-Air-4bit"
"Qwen3.5-122B-A10B|/Users/gaetano/ai/Models/Qwen3.5-122B-A10B-mxfp4"
"Qwen3-Coder-Next|/Users/gaetano/ai/Models/Qwen3-Coder-Next-mxfp4-mlx"
)
case "$BATCH" in
  1) LIST=("${B1[@]}");; 2) LIST=("${B2[@]}");; all) LIST=("${B1[@]}" "${B2[@]}");;
  *) echo "batch ?"; exit 2;;
esac
for entry in "${LIST[@]}"; do
  name="${entry%%|*}"; dir="${entry##*|}"
  if [ ! -f "$dir/config.json" ]; then say "$name ASSENTE"; continue; fi
  say "===== $name  $(date +%H:%M:%S)  load=$(sysctl -n vm.loadavg | awk '{gsub(/[{}]/,"");print $1}') ====="
  perl -e 'alarm 3000; exec @ARGV' "$ROOT/tools/ab_engine_rapid.sh" --model "$dir" \
    --ns "$NS" --m "$M" --reps "$REPS" --port "$PORT" 2>&1 | tee -a "$LOG/$name.ab" \
    | grep -E "^  N=|dati incompleti" | tee -a "$LOG/sweep.txt"
done
say "log: $LOG"
