#!/bin/bash
# rss_ab.sh — ORACOLO RSS: A/B PAIRED interleaved di UN'opzione del registro, misurando il
# picco RSS del processo (deterministico-ish: NON è tok/s, è memoria).
#
# Motivazione: i dati dell'esecuzione del modello BALLANO anche con macchina leggera. Per
# questo si fanno N rep INTERLACCIATE (A,B,A,B,...) e si riporta min/mediana/max: la varianza
# è l'oracolo della fiducia, non un singolo numero.
#
# Uso:
#   rss_ab.sh <model_id> <prompt_file> <context> <opt_name> <valA> <valB> [reps] [-- extra opts...]
# Esempio (R1, Fix B bf16 vs fp32 su gpt-oss-20b):
#   rss_ab.sh 11 /tmp/p16k.txt 32768 LLM_GPTOSS_BF16 1 0 3 -- prefix_cache=off spec_ngram=0
#
# Ogni run: `fastllm --opt <opt_name>=<v> [--opt extra...] run <model>` con stdin dal prompt.
# L'oracolo è `maximum resident set size` di /usr/bin/time -l (NON il PEAK del watchdog:
# qui il child è il binario del modello, quindi /usr/bin/time è il wrapper giusto).
set -u
BIN="${FASTLLM:-$HOME/Documents/Projects/FastLLM/build-release/fastllm}"
MODEL="${1:?model_id}"; PROMPT="${2:?prompt_file}"; CTX="${3:?context}"
OPT="${4:?opt_name}"; VA="${5:?valA}"; VB="${6:?valB}"; REPS="${7:-3}"; shift 7
EXTRA=()
if [ "${1:-}" = "--" ]; then shift; for x in "$@"; do EXTRA+=(--opt "$x"); done; fi
[ -f "$PROMPT" ] || { echo "prompt non trovato: $PROMPT" >&2; exit 2; }
mkdir -p /tmp/oracoli_rss
run1() { # val tag
  local v="$1"
  local tag="$2"
  local log="/tmp/oracoli_rss/${OPT}_${v}_${tag}.log"
  /usr/bin/time -l "$BIN" --opt "$OPT=$v" --opt "context=$CTX" "${EXTRA[@]}" run "$MODEL" \
      < "$PROMPT" >/dev/null 2>"$log"
  local rss; rss=$(grep 'maximum resident set size' "$log" | awk '{print $1}')
  local ctxt; ctxt=$(grep -oE 'contesto: [0-9]+ token' "$log" | head -1)
  awk -v opt="$OPT" -v v="$v" -v tag="$tag" -v rss="$rss" -v ctxt="$ctxt" \
      'BEGIN{printf "  opt=%s=%s rep=%s rss=%.2fGB %s\n",opt,v,tag,rss/1073741824,ctxt}'
  echo "$rss" >> "/tmp/oracoli_rss/${OPT}_${v}.rss"
}
: > "/tmp/oracoli_rss/${OPT}_${VA}.rss"; : > "/tmp/oracoli_rss/${OPT}_${VB}.rss"
for r in $(seq 1 "$REPS"); do run1 "$VA" "$r"; run1 "$VB" "$r"; done
echo "== riepilogo picco RSS (GB) =="
for v in "$VA" "$VB"; do
  awk -v opt="$OPT" -v v="$v" 'BEGIN{min=1e18;max=0;n=0;s=0}
    {n++; s+=$1; if($1<min)min=$1; if($1>max)max=$1}
    END{printf "  %s=%s: n=%d min=%.2f med=%.2f max=%.2f (GB)\n",opt,v,n,min/1073741824,(s/n)/1073741824,max/1073741824}' \
    "/tmp/oracoli_rss/${OPT}_${v}.rss"
done
