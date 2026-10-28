#!/bin/bash
# repeat_ids.sh — ripete la STESSA richiesta N volte (processi freschi) con GEN_IDS=1 e confronta
# gli ID greedy. Rileva LEAK DI STATO / non-determinismo tra richieste.
#
# Motivazione (oracoli.md §5.18): "un server fresco nasconde i bug di stato: manda la stessa
# richiesta 5 volte". Un leak di stato ricorrente/KV tra richieste produce output IDENTICO alla
# prima richiesta e diverso alla quinta -> qui lo si vede con un diff degli ID.
#
# Uso: repeat_ids.sh <cap_GB> <model_id> <prompt_file|-> [reps] [-- extra opts...]
# Esempio: repeat_ids.sh 40 /path/modello /tmp/p.txt 5 -- prefix_cache=off spec_ngram=0
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CAP="${1:?cap_GB}"; MODEL="${2:?model_id}"; PROMPT="${3:?prompt|-}"; REPS="${4:-5}"; shift 4
EXTRA=(); if [ "${1:-}" = "--" ]; then shift; for x in "$@"; do EXTRA+=(--opt "$x"); done; fi
BIN="${FASTLLM:-$HOME/Documents/Projects/FastLLM/build-release/fastllm}"
OUT=/tmp/oracoli_repeat; mkdir -p "$OUT"

for r in $(seq 1 "$REPS"); do
  log="$OUT/rep${r}.log"
  env GEN_IDS=1 "$HERE/wd.sh" "$CAP" "$PROMPT" "$BIN" "${EXTRA[@]}" run "$MODEL" >"$log" 2>&1
  grep -oE '\[genid\] [0-9]+' "$log" | awk '{print $2}' > "$OUT/ids${r}.txt"
  n=$(wc -l < "$OUT/ids${r}.txt" | tr -d ' ')
  ctxt=$(grep -oE 'contesto: [0-9]+ token' "$log" | head -1)
  echo "rep $r: $n id  ${ctxt:-}"
done

echo "== confronto (vs rep 1) =="
base="$OUT/ids1.txt"
[ -s "$base" ] || { echo "ERRORE: nessun [genid] nel rep 1 (GEN_IDS attivo? log in $OUT)" >&2; exit 2; }
ok=1
for r in $(seq 2 "$REPS"); do
  if diff -q "$base" "$OUT/ids${r}.txt" >/dev/null; then
    echo "rep $r: IDENTICO"
  else
    echo "rep $r: DIVERSO"; ok=0
    diff "$base" "$OUT/ids${r}.txt" | head -4 | sed 's/^/   /'
  fi
done
if [ "$ok" = 1 ]; then
  echo "=> DETERMINISTICO su $REPS reps (nessun leak di stato evidente)"
else
  echo "=> NON deterministico: leak di stato o reduction non riproducibile (log in $OUT)"
  exit 1
fi
