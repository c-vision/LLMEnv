#!/bin/zsh
# ⚠️ SUPERATO (2026-09-23) — NON USARE PER NUMERI NUOVI. Usa `tools/bench4_stable.sh`.
# Motivo: misura il prefill a N=512 (finestra ~34 ms -> misura il BOOST GPU, non il throughput:
# bimodale) e ricarica il modello a ogni misura (lento). Ha prodotto "regressioni" fantasma
# (V2-Lite 0.37x, 122B 0.49x) che la misura affidabile ha smentito (parita'). Resta solo come
# riferimento storico. Vedi MEASUREMENT-PROTOCOL errore #11 / #11-bis.
# bench4_paired.sh — misura APPAIATA engine vs mlx_lm per modello (stesso momento, cosi'
# il Δ non e' gonfiato da contesa fra run lontane). Nasce 2026-09-17 dal caso
# Qwen3-Coder-Next: la tabella §9.1 misurava engine e mlx in momenti diversi e su margini
# piccoli produceva Δ fantasma (−8,6% reale +40..+143%).
#
# Metodo (identico a bench4_compare.sh, SENZA Rapid e senza filtro di dimensione):
#  - prefill N=512: ours `prefill 2048 512`; mlx_lm `generate` prompt ~512 (riga "Prompt:").
#  - decode ctx 64:  ours `decode 48`;          mlx_lm `generate` prompt ~64 (riga "Generation:").
# Output: TSV  name<TAB>pf_ours<TAB>dc_ours<TAB>pf_mlx<TAB>dc_mlx
# Uso: tools/bench4_paired.sh [out.tsv]   (default /tmp/bench4_paired.tsv)
set -u
ROOT=/Users/gaetano/Documents/Projects/GMBasic
B=$ROOT/test_mlx_bench/bin/Release/net10.0/test_mlx_bench.dll
BIN=/Users/gaetano/ai/env312/bin
PY=$BIN/python3
MLX=$BIN/mlx_lm
OUT=${1:-/tmp/bench4_paired.tsv}
MDIR=/Users/gaetano/ai/Models
export DYLD_LIBRARY_PATH=$ROOT/../LLMEnv/LLMEngine/mlx/build-gmbasic
: > "$OUT"
tmo() { local t=$1; shift; perl -e 'alarm shift; exec @ARGV' "$t" "$@"; }

P64="In the beginning the universe was created. This has made a lot of people very angry and been widely regarded as a bad move. Many were increasingly of the opinion that they had made a big mistake altogether, and some said that the idea was a bad one from the start."
P512=""; for i in $(seq 1 9); do P512="$P512$P64 "; done

# name:dir  (solo i modelli con controparte mlx_lm caricabile)
MODELS=(
  "Ternary-Bonsai-4B-2bit:$MDIR/Ternary-Bonsai-4B-mlx-2bit"
  "Llama-3.2-1B-4bit:$MDIR/Llama-3.2-1B-Instruct-4bit"
  "gpt-oss-20b-MXFP4-Q8:$MDIR/gpt-oss-20b-MXFP4-Q8"
  "DeepSeek-V2-Lite-4bit-8G:$MDIR/DeepSeek-V2-Lite-Chat-4bit-mlx"
  "Qwen3-8B-4bit:$MDIR/Qwen3-8B-4bit"
  "Qwen3-Coder-30B-A3B-8bit:$MDIR/Qwen3-Coder-30B-A3B-Instruct-8bit"
  "Qwen3.5-27B-6bit:$MDIR/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit"
  "Qwen3.6-27B-4bit:$MDIR/Qwen3.6-27B-4bit"
  "Qwen3.6-35B-A3B-4bit:$MDIR/Qwen3.6-35B-A3B-OptiQ-4bit-textonly"
  "Qwen3.8-27B-8bit:$MDIR/Qwen3.8-27B-8bit"
  "DeepSeek-R1-Distill-32B-8bit:$MDIR/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit"
  "Qwen3.5-122B-A10B-4bit-61G:$MDIR/Qwen3.5-122B-A10B-mxfp4"
  "GLM-4.5-Air-4bit-56G:$MDIR/GLM-4.5-Air-4bit"
  "gpt-oss-120b-61G:$MDIR/gpt-oss-120b-mxfp4-bf16"
)

echo "== paired start $(date) load=$(sysctl -n vm.loadavg) =="
printf "%s\t%s\t%s\t%s\t%s\n" name pf_ours dc_ours pf_mlx dc_mlx | tee -a "$OUT"
for entry in "${MODELS[@]}"; do
  name=${entry%%:*}; M=${entry#*:}
  [ -d "$M" ] || { echo "SKIP $name (dir assente)"; continue; }
  pfo="-"; dco="-"; pfm="-"; dcm="-"
  r=$(tmo 900 dotnet "$B" "$M" prefill 2048 512 2>&1 | grep -oE "tok/s=[0-9,]+" | tail -1 | tr ',' '.')
  [ -n "$r" ] && pfo=${r#tok/s=}
  r=$(tmo 900 dotnet "$B" "$M" decode 48 2>&1 | grep -oE "tok/s=[0-9,]+" | tail -1 | tr ',' '.')
  [ -n "$r" ] && dco=${r#tok/s=}
  farm="/tmp/ref4p_$name"; rm -rf "$farm"; mkdir -p "$farm" 2>/dev/null
  for f in "$M"/*; do ln -s "$f" "$farm/" 2>/dev/null; done
  if [ -f "$farm/tokenizer_config.json" ]; then
    rm -f "$farm/tokenizer_config.json"
    "$PY" - "$M/tokenizer_config.json" "$farm/tokenizer_config.json" <<'PYEOF' 2>/dev/null
import json,sys
t=json.load(open(sys.argv[1]))
if t.get("tokenizer_class")=="TokenizersBackend": t["tokenizer_class"]="PreTrainedTokenizerFast"
json.dump(t,open(sys.argv[2],"w"))
PYEOF
  fi
  o=$(tmo 1800 "$MLX" generate --model "$farm" --prompt "$P512" --max-tokens 48 --temp 0 2>&1)
  r=$(echo "$o" | grep -E "^Prompt:" | grep -oE "[0-9.]+ tokens-per-sec" | awk '{print $1}')
  [ -n "$r" ] && pfm=$r
  o=$(tmo 1800 "$MLX" generate --model "$farm" --prompt "$P64" --max-tokens 48 --temp 0 2>&1)
  r=$(echo "$o" | grep -E "^Generation:" | grep -oE "[0-9.]+ tokens-per-sec" | awk '{print $1}')
  [ -n "$r" ] && dcm=$r
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$pfo" "$dco" "$pfm" "$dcm" | tee -a "$OUT"
  rm -rf "$farm"
done
echo "== paired end $(date) load=$(sysctl -n vm.loadavg) =="
