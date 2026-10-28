#!/bin/zsh
# ab_models_under30g.sh — A/B GMB vs mlx_lm sui modelli locali < 30GB, ENTRAMBI gli ordini,
# PREFILL + DECODE, con il kernel oQ A8 (primitivo fuso) attivo sul lato GMB.
# Uso: tools/ab_models_under30g.sh [gruppo] [N] [M]      (gruppo: small|mid|all, default all)
# Output: tools/ab-results/<ts>-under30g/risultati.tsv  (model order N pf_g pf_ml dfx_g dfx_ml pfx dfx)
# Nota: la lista è quella dei modelli locali < 30GB; ognuno viene saltato se assente/non caricabile.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GROUP="${1:-all}"; N="${2:-4096}"; M="${3:-64}"
MDIR="/Users/gaetano/ai/Models"
TS="$(date +%Y%m%d_%H%M%S)"; OUT="$ROOT/tools/ab-results/$TS-under30g"; mkdir -p "$OUT"
TSV="$OUT/risultati.tsv"; : > "$TSV"

SMALL=(Llama-3.2-1B-Instruct-4bit Ternary-Bonsai-4B-mlx-2bit Qwen3-8B-4bit \
       DeepSeek-V2-Lite-Chat-4bit-mlx gpt-oss-20b-MXFP4-Q8)
MID=(Qwen3.6-27B-4bit gemma-4-31b-it-4bit Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit \
     Qwen3.6-35B-A3B-OptiQ-4bit-textonly Kimi-Linear-48B-A3B-Instruct-4bit Qwen3.8-27B-8bit)
case "$GROUP" in
  small) MODELS=(${SMALL[@]});;
  mid)   MODELS=(${MID[@]});;
  all)   MODELS=(${SMALL[@]} ${MID[@]});;
  *) echo "gruppo? small|mid|all"; exit 2;;
esac
l1() { uptime | sed 's/.*load averages: //' | awk '{print $1}' | tr ',' '.'; }
echo "[under30g] gruppo=$GROUP N=$N M=$M modelli=${#MODELS[@]} out=$OUT (load1=$(l1))"

for MNAME in ${MODELS[@]}; do
  MD="$MDIR/$MNAME"; [ -d "$MD" ] || { echo "SKIP $MNAME (assente)"; continue; }
  for ORD in gmb-first mlxlm-first; do
    LOGLINE="$OUT/${MNAME}.${ORD}.log"
    tools/ab_engine_mlxlm.sh --model "$MD" --ns "$N" --m "$M" --reps 2 --order "$ORD" \
      > "$LOGLINE" 2>&1
    SUM=$(grep "PREFILL gmb" "$LOGLINE" | tail -1)
    if [ -n "$SUM" ]; then
      echo "$SUM" | sed -E "s/^ *N=([0-9]+) +PREFILL gmb=([0-9.,]+) mlx_lm=([0-9.,]+) -> ([0-9.,]+)x \| DECODE gmb=([0-9.,]+) mlx_lm=([0-9.,]+) -> ([0-9.,]+)x/$MNAME\t$ORD\t\1\t\2\t\3\t\4\t\5\t\6\t\7/" | tr ',' '.' >> "$TSV"
      echo "  [$MNAME $ORD] $SUM  (load1=$(l1))"
    else
      echo "[$MNAME $ORD] ERRORE/nessun dato (vedi $LOGLINE)"
      echo -e "$MNAME\t$ORD\t-\t-\t-\t-\t-\t-\t-" >> "$TSV"
    fi
  done
done
echo "[under30g] fatto. TSV: $TSV"
