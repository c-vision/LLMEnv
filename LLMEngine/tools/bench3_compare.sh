#!/bin/zsh
# bench3_compare.sh — sweep comparativo su modelli locali: GMBasic vs mlx_lm vs Rapid-MLX.
# Uso: tools/bench3_compare.sh [max_gb]   (default 30)
# Output: /tmp/bench3.tsv  (modello<TAB>ours<TAB>mlx_lm<TAB>rapid) + /tmp/bench3.log
# Regola §J: numeri validi solo in finestra pulita (load<4/idle>85%); qui il carico e' registrato.
set -u
MAXGB=${1:-30}
ROOT=/Users/gaetano/Documents/Projects/GMBasic
B=$ROOT/test_mlx_bench/bin/Release/net10.0/test_mlx_bench.dll
BIN=/Users/gaetano/ai/env312/bin
PY=$BIN/python3
OUT=/tmp/bench3.tsv
LOG=/tmp/bench3.log
: > "$OUT"

# timeout portabile (macOS non ha timeout(1))
tmo() { local t=$1; shift; perl -e 'alarm shift; exec @ARGV' "$t" "$@"; }

MODELS=()
for d in /Users/gaetano/ai/Models/*/; do
  n=$(basename "$d")
  [ "$n" = "gguf" ] && continue
  [ "$n" = "media" ] && continue
  gb=$(du -sm "$d" 2>/dev/null | awk '{print int($1/1024)}')
  [ "${gb:-999}" -le "$MAXGB" ] && MODELS+=("$d")
done

echo "== bench3 start $(date) carico=$(sysctl -n vm.loadavg) modelli=${#MODELS[@]} (<=${MAXGB}GB)" | tee -a "$LOG"
printf "%-45s %10s %10s %10s\n" "modello" "ours" "mlx_lm" "rapid" | tee -a "$LOG"

for M in "${MODELS[@]}"; do
  name=$(basename "$M")
  ours="-"; mlx="-"; rapid="-"

  # --- ours (greedy decode 48) ---
  r=$(tmo 900 dotnet "$B" "$M" decode 48 2>&1 | grep -oE "tok/s=[0-9,]+" | tail -1 | tr ',' '.')
  [ -n "$r" ] && ours=${r#tok/s=}

  # --- mlx_lm (symlink farm + tokenizer_class patch in copia) ---
  farm="/tmp/ref_$name"; rm -rf "$farm"; mkdir -p "$farm" 2>/dev/null
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
  r=$(tmo 1200 "$BIN/mlx_lm" generate --model "$farm" --prompt "The capital of France is" --max-tokens 48 --temp 0 2>&1 | grep -E "^Generation:" | grep -oE "[0-9.]+ tokens-per-sec" | tail -1 | awk '{print $1}')
  [ -n "$r" ] && mlx=$r

  # --- Rapid-MLX (tier speed, server auto-boot) ---
  r=$(tmo 1200 "$BIN/rapid-mlx" bench "$M" --tier speed --num-prompts 1 --max-tokens 48 2>&1 | grep -oE "tps=[0-9.]+" | tail -1)
  [ -n "$r" ] && rapid=${r#tps=}
  pkill -f "rapid-mlx" 2>/dev/null
  sleep 2

  printf "%-45s %10s %10s %10s\n" "$name" "$ours" "$mlx" "$rapid" | tee -a "$LOG"
  printf "%s\t%s\t%s\t%s\n" "$name" "$ours" "$mlx" "$rapid" >> "$OUT"
done
echo "== bench3 end $(date) carico=$(sysctl -n vm.loadavg)" | tee -a "$LOG"
