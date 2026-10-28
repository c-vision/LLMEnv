#!/bin/zsh
# =============================================================================
# ab_all_ctx_mlxlm.sh — GMBasic vs mlx_lm su TUTTI i modelli, a contesti crescenti
# (prefill+decode), entrambi gli ordini. TSV incrementale + log.
#
# Uso: tools/ab_all_ctx_mlxlm.sh [batch1|batch2|all] [contexts] [gen] [reps]
#   default: batch1  512,2048,4096,8192  32  2
# Risultato: $OUT/ctx.tsv  (model order N gmb_pref mlxlm_pref gmb_dec mlxlm_dec)
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BATCH="${1:-batch1}"
CTX="${2:-512,2048,4096,8192}"
GEN="${3:-32}"
REPS="${4:-2}"
MDIR="/Users/gaetano/ai/Models"
TS="$(date +%Y%m%d_%H%M%S)"; OUT="$ROOT/tools/ab-results/$TS-ctx"; mkdir -p "$OUT"
TSV="$OUT/ctx.tsv"; : > "$TSV"
PY="/Users/gaetano/ai/env312/bin/python"
l1() { uptime | sed 's/.*load averages: //' | awk '{print $1}' | tr ',' '.'; }

BATCH1=(Llama-3.2-1B-Instruct-4bit Ternary-Bonsai-4B-mlx-2bit DeepSeek-V2-Lite-Chat-4bit-mlx \
  Qwen3.6-27B-4bit Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit gemma-4-31b-it-4bit \
  Qwen3.6-35B-A3B-OptiQ-4bit-textonly Kimi-Linear-48B-A3B-Instruct-4bit Qwen3.8-27B-8bit \
  Qwen3-Coder-30B-A3B-Instruct-8bit DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit gpt-oss-20b-MXFP4-Q8)
BATCH2=(Qwen3-Coder-Next-mxfp4-mlx GLM-4.5-Air-4bit gpt-oss-120b-mxfp4-bf16 \
  Qwen3.5-122B-A10B-mxfp4 Mixtral-8x22B-4bit)

case "$BATCH" in
  batch1) MODELS=(${BATCH1[@]});;
  batch2) MODELS=(${BATCH2[@]});;
  all)    MODELS=(${BATCH1[@]} ${BATCH2[@]});;
  *) echo "batch?"; exit 2;;
esac

echo "[ctx] batch=$BATCH contexts=$CTX gen=$GEN reps=$REPS modelli=${#MODELS[@]} log=$OUT (load1=$(l1))"
for MNAME in ${MODELS[@]}; do
  MD="$MDIR/$MNAME"; [ -d "$MD" ] || { echo "SKIP $MNAME (assente)"; continue; }
  echo "[ctx] === $MNAME (load1=$(l1)) ==="
  "$PY" "$ROOT/tools/ab_ctx_mlxlm.py" --model "$MD" --contexts "$CTX" --gen "$GEN" \
        --reps "$REPS" --out "$TSV" 2>&1 | tee -a "$OUT/${MNAME}.log"
done
echo "[ctx] fatto. TSV: $TSV"
