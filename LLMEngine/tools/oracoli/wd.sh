#!/bin/bash
# wd.sh — WATCHDOG per esecuzioni modello. Uccide il comando se l'RSS del processo
# supera il tetto (GB). PROTEGGE la macchina: una run modello sbagliata può saturare la
# RAM e far riavviare il Mac (è già successo: una run a ~80GB).
#
# Uso:  wd.sh <cap_GB> <prompt_file|-> <comando...>
#   cap_GB      : tetto di RSS in GB (arrotondere per eccesso; modello 27GB -> 45)
#   prompt_file : file dato in stdin al comando (prompt lungo), oppure '-' per nessuno
#   comando     : il comando completo (es. ./build-release/fastllm --opt ... run 4 "...")
#
# Stampa su stderr alla fine: "WD_EXIT=<rc> PEAK=<GB>". Il PEAK è il picco RSS osservato.
#
# NOTA: misura il RSS del processo FIGLIO diretto. Se il child è un wrapper che fa fork/exec
# (es. /usr/bin/time), il RSS monitorato NON è quello del modello: in quel caso usa l'oracolo
# del wrapper (es. `maximum resident set size` di /usr/bin/time -l) e NON questo PEAK.
set -u
CAP_GB="${1:?uso: wd.sh <cap_GB> <prompt|-| comando...>}"; shift
PROMPT="${1:?serve prompt_file o -}"; shift
CAP_KB=$(( CAP_GB * 1024 * 1024 ))
if [ "$PROMPT" = "-" ]; then "$@" & else "$@" < "$PROMPT" & fi
PID=$!; PEAK=0
while kill -0 "$PID" 2>/dev/null; do
  RSS=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
  if [ -n "$RSS" ]; then
    [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
    if [ "$RSS" -gt "$CAP_KB" ]; then
      echo "WATCHDOG KILL RSS=$((RSS/1048576))GB cap=${CAP_GB}GB" >&2
      kill -9 "$PID" 2>/dev/null; break
    fi
  fi
  sleep 2
done
wait "$PID" 2>/dev/null; RC=$?
echo "WD_EXIT=$RC PEAK=$((PEAK/1048576))GB" >&2
exit "$RC"
