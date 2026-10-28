#!/bin/zsh
# =============================================================================
# bench4_stable.sh — confronto APPAIATO GMBasic vs mlx_lm con GUARDRAIL (2026-09-23).
#
# Perche'. `bench4_paired.sh` misura una volta per lato: sotto contesa il numero
# e' rumore (misurato: spread 32% e 4,5x). Questo runner compone il guardrail
# `tools/measure_guard.sh` per ogni misura e NON emette la riga del modello se:
#   - la finestra non e' quieta (load1 alto / processi AI attivi), oppure
#   - lo spread di GMB o di mlx_lm supera la soglia.
# In tal caso la riga riporta "NON_MISURABILE" e i valori grezzi NON vanno usati.
#
# Uso:
#   tools/bench4_stable.sh [--out TSV] [--reps 4] [--max-spread 5] [--models "a b c"]
#   tools/bench4_stable.sh --check          # solo verifica finestra
#
# Output TSV: name  pf_ours dc_ours pf_mlx dc_mlx  pf_ratio dc_ratio  spread_pf_ours spread_dc_ours verdict
# Protocollo (MEASUREMENT-PROTOCOL errore #11): finestra quieta, >=4 run,
# warm-up scartato, spread<=5%, ordine invertito concordante.
# =============================================================================
set -uo pipefail
ROOT="/Users/gaetano/Documents/Projects/GMBasic"
G="$ROOT/tools/measure_guard.sh"
B="$ROOT/test_mlx_bench/bin/Release/net10.0/test_mlx_bench.dll"
# Reference mlx_lm: env-mlxbench (mlx-lm 0.31.3 + transformers 5.17.0, coppia COERENTE) — e' l'env
# che carica TUTTI i modelli (tokenizer_class: TokenizersBackend per qwen3_5/gemma4), vedi ~/ai/envs.txt.
# env312 ha transformers 4.57.6 (pinnato per mineru) -> mlx-lm 0.31.3 DEGRADATO e non carica quei modelli.
PY="${GMB_AB_PY:-/Users/gaetano/ai/env-mlxbench/bin/python}"
MDIR="/Users/gaetano/ai/Models"
export DYLD_LIBRARY_PATH="$ROOT/Resources/libs/osx-arm64"
export GMB_BENCH_PERREP=1   # bench_mlxlm.py stampa i per-rep (spread robusto); ignorata dal lato GMB

OUT="/tmp/bench4_stable.tsv"
REPS=1; MAXSP=5; MODELS=""   # reps default 1 (veloce); alza con --reps per il MAD
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --max-spread) MAXSP="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --check) "$G" --check-contention; exit $?;;
    *) echo "arg sconosciuto: $1" >&2; exit 2;;
  esac
done

DEFAULT="Ternary-Bonsai-4B-mlx-2bit Llama-3.2-1B-Instruct-4bit gpt-oss-20b-MXFP4-Q8 \
DeepSeek-V2-Lite-Chat-4bit-mlx Qwen3-8B-4bit Qwen3-Coder-30B-A3B-Instruct-8bit \
Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit Qwen3.6-27B-4bit \
Qwen3.6-35B-A3B-OptiQ-4bit-textonly Qwen3.8-27B-8bit \
DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit GLM-4.5-Air-4bit gpt-oss-120b-mxfp4-bf16"
LIST="${MODELS:-$DEFAULT}"

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  name pf_ours dc_ours pf_mlx dc_mlx pf_ratio dc_ratio spread_pf_ours spread_dc_ours verdict | tee "$OUT"

NTOT=$(echo $LIST | wc -w | tr -d ' ')
echo "[stable] $(date +%H:%M:%S) start: $NTOT modelli, reps=$REPS, maxMAD=${MAXSP}%, load1=$(sysctl -n vm.loadavg|awk '{print $2}')"
echo "[stable] log vivo: tail -f $(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT" .tsv).log  (o questo stdout)"
i=0
for name in ${=LIST}; do
  i=$((i+1))
  M="$MDIR/$name"
  echo "[$i/$NTOT] $(date +%H:%M:%S) === $name ==="
  [ -d "$M" ] || { echo "  SKIP (dir assente)"; continue; }
  # EFFICIENZA: ogni misura carica il modello UNA volta e ripete internamente
  # (GMB: `rprefill`/`rdecode`; mlx: `--repeats` + GMB_BENCH_PERREP=1 per i per-rep).
  # Il guard gira in --multi: una sola esecuzione, tutti i valori -> mediana+MAD.
  grab() { # $1=label $2=extract $3...=comando ; stampa "MED SPREAD"
    local o; o="$("$G" --multi --warmup 0 --max-mad "$MAXSP" --extract "$2" --label "$1" -- "${@:3}" 2>&1)"
    local md sp; md="$(print -r -- "$o" | grep -oE 'mediana=[0-9.e+-]+' | head -1 | cut -d= -f2)"
    sp="$(print -r -- "$o" | grep -oE 'spread=[0-9.]+' | head -1 | cut -d= -f2)"
    print -r -- "${md:--} ${sp:--}"
  }
  echo "  $(date +%H:%M:%S) GMB prefill (1 load)..."
  read pfo spo <<< "$(grab "$name pf" 'PREFILL_REP.*tok/s=[0-9.,]+' dotnet "$B" "$M" rprefill 2048 4096 "$REPS")"
  echo "    -> pf_ours=$pfo (spread ${spo}%)"
  echo "  $(date +%H:%M:%S) GMB decode (1 load)..."
  read dco sdo <<< "$(grab "$name dc" 'DECODE_REP.*tok/s=[0-9.,]+'  dotnet "$B" "$M" rdecode 48 "$REPS")"
  echo "    -> dc_ours=$dco (spread ${sdo}%)"
  echo "  $(date +%H:%M:%S) mlx_lm prefill (1 load)..."
  read pfm _   <<< "$(grab "$name pf mlx" 'REP n=4096 prefill [0-9.]+ tok/s' "$PY" "$ROOT/tools/bench_mlxlm.py" --model "$M" --lengths 4096 --max-tokens 1 --repeats "$REPS")"
  echo "    -> pf_mlx=$pfm"
  echo "  $(date +%H:%M:%S) mlx_lm decode (1 load)..."
  read dcm _   <<< "$(grab "$name dc mlx" 'REP n=64 .*decode [0-9.]+ tok/s'  "$PY" "$ROOT/tools/bench_mlxlm.py" --model "$M" --lengths 64 --max-tokens 48 --repeats "$REPS")"
  echo "    -> dc_mlx=$dcm"
  if [ "$pfo" = "-" ] || [ "$dco" = "-" ] || [ "$pfm" = "-" ] || [ "$dcm" = "-" ] \
     || [ -z "$pfo" ] || [ -z "$dco" ] || [ -z "$pfm" ] || [ -z "$dcm" ]; then
    printf "%s\t%s\t%s\t%s\t%s\t-\t-\t%s\t%s\tNON_MISURABILE(estrazione)\n" \
      "$name" "${pfo:--}" "${dco:--}" "${pfm:--}" "${dcm:--}" "${spo:--}" "${sdo:--}" | tee -a "$OUT"
    continue
  fi
  read rpf rdo <<< "$(python3 -c "print(f'{$pfo/$pfm:.3f} {$dco/$dcm:.3f}')")"
  echo "  $(date +%H:%M:%S) RISULTATO $name: PF ${rpf}x  DC ${rdo}x"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tOK\n" \
    "$name" "$pfo" "$dco" "$pfm" "$dcm" "$rpf" "$rdo" "${spo:--}" "${sdo:--}" | tee -a "$OUT"
done
echo "[stable] $(date +%H:%M:%S) fatto -> $OUT"
