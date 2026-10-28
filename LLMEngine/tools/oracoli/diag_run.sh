#!/bin/bash
# diag_run.sh — esegue un modello con gli ORACOLI DETERMINISTICI attivi e salva il log.
# Non measure "velocità": produce i numeri spendibili (nodi/token, MB_in/token, id greedy).
#
# Uso:  diag_run.sh <cap_GB> <model_id> <prompt_file|-> [--opt k=v ...]
#
# Oracoli attivati (env di diagnosi, ammessi: NON sono parametri di comportamento):
#   LLM_DISPATCH=1  -> profilo G0: nodi/token, MB/token, MB_in/token, floor_us/tok, spec_tok/pass
#   GEN_IDS=1       -> dump degli ID token emessi (parità ID-per-ID tra due path)
#   DF2_STATS=1     -> accettazione cumulativa DFlash2 (se il path è speculativo)
#
# Il comando gira SEMPRE sotto watchdog (wd.sh): protegge la macchina da run fuori controllo.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CAP="${1:?cap_GB}"; MODEL="${2:?model_id}"; PROMPT="${3:?prompt|-}"; shift 3
BIN="${FASTLLM:-$HOME/Documents/Projects/FastLLM/build-release/fastllm}"
LOG="/tmp/oracoli_diag_$(date +%H%M%S).log"
echo "log -> $LOG" >&2
env LLM_DISPATCH=1 GEN_IDS=1 DF2_STATS=1 \
    "$HERE/wd.sh" "$CAP" "$PROMPT" "$BIN" "$@" run "$MODEL" 2>"$LOG"
echo "== sintesi ==" >&2
grep -E 'nodi_totali|contesto:|\[genid\]|DF2_STATS' "$LOG" | head -40 >&2
echo "(log completo: $LOG)" >&2
