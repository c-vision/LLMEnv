#!/bin/bash
# spec_stats.sh — estrae da un log di diagnosi i numeri SPENDIBILI della speculazione:
# PASS-TIME (ms/passo) e ACCEPTANCE. MAI i tok/s tra run diverse.
#
# Motivazione (oracoli.md §0/§2f/§5.17): con speculazione ON il throughput e'
# `pass_time / token_accettati_per_passata`, e qualsiasi variazione numerica rimescola quali bozze
# vengono accettate: i tok/s NON sono confrontabili. Confronta il pass-time a batch fisso e riporta
# sempre l'acceptance accanto.
#
# Uso: spec_stats.sh <log>
#   Il log si produce con: diag_run.sh <cap_GB> <model> <prompt> [--opt ...]
set -u
LOG="${1:?log}"
[ -f "$LOG" ] || { echo "log non trovato: $LOG" >&2; exit 2; }

echo "== acceptance (DFlash2) =="
grep -E '\[df2stats\] total accepted' "$LOG" | tail -1
grep -E '\[df2stats\] r=' "$LOG" | tail -3
echo "== per-round (DFlash2) =="
grep -E '\[df2vdbg\]' "$LOG" | tail -3

echo "== pass-time / build-eval =="
grep -E 'tempo ms/passo' "$LOG" | tail -2
grep -E 'min/passo|max/passo' "$LOG" | tail -2

echo "== altri contatori spec =="
grep -iE 'spec_drafts|spec_accepted|drafts_offered|drafts_accepted|spec_tok|acceptance|reject' "$LOG" | tail -8

if ! grep -qE '\[df2stats\]|\[df2vdbg\]|ms/passo' "$LOG"; then
  echo
  echo "WARN: nel log non c'e' nessun numero di speculazione (acceptance/pass-time)."
  echo "      Se speculazione e' ON, ri-esegui con gli oracoli: diag_run.sh ... (LLM_DISPATCH/GEN_IDS/DF2_STATS)."
  echo "      Se confronti soltanto tok/s, il verdetto NON e' valido (oracoli.md §0/§2f)."
fi
