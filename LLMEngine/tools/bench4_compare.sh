#!/bin/zsh
# bench4_compare.sh — tabella COMPLETA prefill+decode: GMBasic vs mlx_lm vs Rapid-MLX.
# Uso: tools/bench4_compare.sh [max_gb]   (default 30)
# Output: /tmp/bench4.tsv  nome<TAB>prefill_ours<TAB>decode_ours<TAB>prefill_mlx<TAB>decode_mlx<TAB>prefill_rapid<TAB>decode_rapid
#
# METODO (contesti allineati — vedi Benchmark §8):
#  - prefill: N=512 token su tutti. ours `prefill 2048 512`; mlx_lm prompt da ~512 token (riga "Prompt:");
#    rapid = 512 / (TTFT del server con max_tokens=1).
#  - decode: contesto 64 su tutti. ours `decode 48`; mlx_lm prompt ~64 token (riga "Generation:");
#    rapid bench tier speed --long-prompt-tokens 64 (campo tps=).
set -u
MAXGB=${1:-30}
MINGB=${2:-0}
FILTER=${3:-}
ROOT=/Users/gaetano/Documents/Projects/GMBasic
B=$ROOT/test_mlx_bench/bin/Release/net10.0/test_mlx_bench.dll
BIN=/Users/gaetano/ai/env312/bin
# RAPID: reference UFFICIALE pinnata (2026-09-17). PRIMA usava env312 che ha un EDITABLE del FORK
# `c-vision/Rapid-MLX-NextGen` (metadata STALE 0.10.3, codice ~0.13.1) -> le colonne rapid non erano
# l'upstream. Ora: venv dedicato con `rapid-mlx==0.14.2` (upstream `raullenchai/Rapid-MLX`, tag v0.14.2);
# sorgente clonato in /Users/gaetano/ai/Rapid-MLX-official (il fork in Documents/Projects NON si tocca).
BIN_RAPID=/Users/gaetano/ai/env-rapid-official/bin
RAPID=$BIN_RAPID/rapid-mlx
# RAPID_SPEC: none|mtp (default none) -> separa "engine vs engine" dal divario prodotto (MTP).
RAPID_SPEC=${RAPID_SPEC:-none}
PY=$BIN/python3
OUT=${BENCH4_OUT:-/tmp/bench4.tsv}
LOG=/tmp/bench4.log
: > "$OUT"
tmo() { local t=$1; shift; perl -e 'alarm shift; exec @ARGV' "$t" "$@"; }

P64="In the beginning the universe was created. This has made a lot of people very angry and been widely regarded as a bad move. Many were increasingly of the opinion that they had made a big mistake altogether, and some said that the idea was a bad one from the start."
P512=""
for i in $(seq 1 9); do P512="$P512$P64 "; done

MODELS=()
for d in /Users/gaetano/ai/Models/*/; do
  n=$(basename "$d"); [ "$n" = "gguf" ] && continue; [ "$n" = "media" ] && continue
  [ -n "$FILTER" ] && case "$n" in *"$FILTER"*) ;; *) continue ;; esac
  gb=$(du -sm "$d" 2>/dev/null | awk '{print int($1/1024)}')
  [ "${gb:-999}" -le "$MAXGB" ] && [ "${gb:-999}" -gt "$MINGB" ] && MODELS+=("$d")
done

echo "== bench4 start $(date) carico=$(sysctl -n vm.loadavg) modelli=${#MODELS[@]} range=(${MINGB},${MAXGB}] rapid=$("$RAPID" --version 2>/dev/null | head -1) spec=$RAPID_SPEC" | tee -a "$LOG"
printf "%-42s %9s %9s %9s %9s %9s %9s\n" modello pf_ours dc_ours pf_mlx dc_mlx pf_rapid dc_rapid | tee -a "$LOG"

for M in "${MODELS[@]}"; do
  name=$(basename "$M")
  pfo="-"; dco="-"; pfm="-"; dcm="-"; pfr="-"; dcr="-"

  # ---- OURS ----
  r=$(tmo 900 dotnet "$B" "$M" prefill 2048 512 2>&1 | grep -oE "tok/s=[0-9,]+" | tail -1 | tr ',' '.')
  [ -n "$r" ] && pfo=${r#tok/s=}
  r=$(tmo 900 dotnet "$B" "$M" decode 48 2>&1 | grep -oE "tok/s=[0-9,]+" | tail -1 | tr ',' '.')
  [ -n "$r" ] && dco=${r#tok/s=}

  # ---- mlx_lm (symlink farm + patch tokenizer in copia) ----
  farm="/tmp/ref4_$name"; rm -rf "$farm"; mkdir -p "$farm" 2>/dev/null
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
  # prefill 512 (riga Prompt) + decode a contesto 512 (riga Generation)
  o=$(tmo 1800 "$BIN/mlx_lm" generate --model "$farm" --prompt "$P512" --max-tokens 48 --temp 0 2>&1)
  r=$(echo "$o" | grep -E "^Prompt:" | grep -oE "[0-9.]+ tokens-per-sec" | awk '{print $1}')
  [ -n "$r" ] && pfm=$r
  # decode a contesto 64
  o=$(tmo 1200 "$BIN/mlx_lm" generate --model "$farm" --prompt "$P64" --max-tokens 48 --temp 0 2>&1)
  r=$(echo "$o" | grep -E "^Generation:" | grep -oE "[0-9.]+ tokens-per-sec" | awk '{print $1}')
  [ -n "$r" ] && dcm=$r

  # ---- Rapid-MLX: decode dal bench, prefill dal server (TTFT con 512 token, max_tokens=1) ----
  r=$(tmo 1200 "$RAPID" bench "$M" --tier speed --num-prompts 1 --max-tokens 48 --long-prompt-tokens 64 --spec-decode "$RAPID_SPEC" 2>&1 | grep -oE "tps=[0-9.]+" | tail -1)
  [ -n "$r" ] && dcr=${r#tps=}
  pkill -f "rapid-mlx" 2>/dev/null; sleep 2
  # --pflash off: PFlash (compressione del prompt) e' AUTO di default in 0.14.x -> senza questo la
  # riga prefill rapid non elabora tutti i token (colonna prefill non valida; vedi Benchmark §9).
  ( "$RAPID" serve "$M" --port 8543 --pflash off >/tmp/rapid_srv.log 2>&1 & ) ; sleep 1
  for w in $(seq 1 40); do curl -s -m 2 localhost:8543/v1/models >/dev/null 2>&1 && break; sleep 3; done
  if curl -s -m 2 localhost:8543/v1/models >/dev/null 2>&1; then
    ttft=$("$PY" - "$P512" <<'PYEOF' 2>/dev/null
import json,sys,time,urllib.request
p=sys.argv[1]
def req(prompt,mt):
    b=json.dumps({"model":"m","prompt":prompt,"max_tokens":mt,"temperature":0}).encode()
    r=urllib.request.Request("http://localhost:8543/v1/completions",data=b,headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(r,timeout=300) as f: f.read()
    return time.time()-t0
try:
    t1=req("hi",1)          # overhead di base (1 token)
    t2=req(prompt,1)        # prefill 512 + 1 token
    print("%.4f" % (t2-t1)) # secondi di prefill (sottratto l'overhead)
except Exception:
    print("")
PYEOF
)
    if [ -n "$ttft" ]; then pfr=$("$PY" -c "t=$ttft; print('%.1f'%(512/t) if t>0 else '-')" 2>/dev/null); fi
  fi
  pkill -f "rapid-mlx" 2>/dev/null

  printf "%-42s %9s %9s %9s %9s %9s %9s\n" "$name" "$pfo" "$dco" "$pfm" "$dcm" "$pfr" "$dcr" | tee -a "$LOG"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$pfo" "$dco" "$pfm" "$dcm" "$pfr" "$dcr" >> "$OUT"
  sleep 2
done
echo "== bench4 end $(date) carico=$(sysctl -n vm.loadavg)" | tee -a "$LOG"
