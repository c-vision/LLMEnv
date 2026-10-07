#!/usr/bin/env bash
# scale_sweep.sh - misura lo SCALING a batch/ampiezza crescente e segnala i cliff non monotoni.
#
# Motivazione (oracoli.md 5.26): il throughput non cresce necessariamente in modo monotono; soglie
# di selezione del KERNEL in funzione del batch/forma possono creare salti e cali. Un peggioramento
# di ms/token passando a un batch piu' grande va misurato a piu' ampiezze, non assunto.
#
# Modalita' A (default): ampiezza = NP (token di prefill = batch M del matmul), via decode_profile.
# Modalita' B (--cmd): comando arbitrario con {W} (es. concorrenza verso il server); estrai la metrica
#   con --grep/--field (default: piu' basso e' meglio; usa --higher-better per tok/s).
#
# Uso:
#   MODEL=<dir MLX> [DYLIB=..] [NG=64] [REPS=1] LLMEngine/tools/scale_sweep.sh 128,256,512,1024,2048,4096 [--opt k=v ...]
#   LLMEngine/tools/scale_sweep.sh --cmd "client --conc {W}" --grep tok/s --field 3 --higher-better 1,2,4,8
#
# Soglia cliff: env THRESH (default 15, %).
set -eo pipefail
export LC_ALL=C   # decimal point '.', non ',' (altrimenti i TSV/awk si rompono in locale IT)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
THRESH="${THRESH:-15}"
CMD=""; GREP=""; FIELD=""; LOWER=1
WIDTHS=""; EXTRA_STR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cmd) CMD="$2"; shift 2;;
    --grep) GREP="$2"; shift 2;;
    --field) FIELD="$2"; shift 2;;
    --higher-better) LOWER=0; shift;;
    --lower-better) LOWER=1; shift;;
    *) if [ -z "$WIDTHS" ]; then WIDTHS="$1"; else EXTRA_STR="$EXTRA_STR $1"; fi; shift;;
  esac
done
[ -n "$WIDTHS" ] || { echo "uso: $0 <csv_ampiezze> [--opt k=v ...] | --cmd ... --grep RE --field N <csv>"; exit 2; }

TMP=$(mktemp -d /tmp/scale_sweep.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

run_width() {
  W="$1"
  if [ -n "$CMD" ]; then
    C="${CMD/\{W\}/$W}"
    OUT=$(eval "$C" 2>/dev/null || true)
    V=$(printf '%s\n' "$OUT" | grep -E "$GREP" | head -1 | awk -v f="${FIELD:-1}" '{print $f}')
    printf '%s %s\n' "$W" "${V:-nan}"
    return
  fi
  DYL="${DYLIB:-$ROOT/build/dist/osx-arm64/libllm_ai_mlx.dylib}"
  MDL="${MODEL:?MODEL}"
  NG="${NG:-64}"
  REPS="${REPS:-1}"
  BIN=/tmp/decode_profile
  if [ ! -x "$BIN" ]; then cc -O2 -o "$BIN" "$ROOT/LLMEngine/tools/decode_profile.c"; fi
  ROW=$("$BIN" "$DYL" "$MDL" "$W" "$NG" --reps "$REPS" --quiet $EXTRA_STR 2>/dev/null | awk '/^ *[0-9]+ /{print $2; exit}')
  PF=$(printf '%s\n' "$ROW" | awk -v w="$W" 'BEGIN{v=""} NF{if($1>0)v=$1/w} END{if(v=="")print "nan"; else printf "%.4f", v}')
  printf '%s %s\n' "$W" "${PF:-nan}"
}

: > "$TMP/res.tsv"
for W in $(printf '%s' "$WIDTHS" | tr ',' ' '); do
  run_width "$W" >> "$TMP/res.tsv"
done

cat > "$TMP/analyze.awk" <<'AWK'
{
  w[NR]=$1; v[NR]=$2+0
}
END {
  cliff=0
  for (i=1; i<=NR; i++) {
    d=""
    if (i>1 && v[i-1]>0) { d=sprintf("%+.1f%%", 100.0*(v[i]-v[i-1])/v[i-1]) }
    printf("  %-10s %-16.4f %s\n", w[i], v[i], d)
    if (i>1 && v[i-1]>0) {
      bad = lower ? (100.0*(v[i]-v[i-1])/v[i-1] >  thresh) : (100.0*(v[i]-v[i-1])/v[i-1] < -thresh)
      if (bad) { printf("     ^^ CLIFF NON MONOTONO: %.4f -> %.4f oltre %d%%\n", v[i-1], v[i], thresh); cliff=1 }
    }
  }
  if (cliff) print "=> ATTENZIONE: scaling NON monotono (sospetta soglia di selezione kernel). Ripeti con REPS>1 e verifica acceptance."
  else       print "=> scaling monotono entro la soglia."
}
AWK

echo "== risultati (soglia cliff ${THRESH}%) =="
printf '  %-10s %-16s %s\n' "ampiezza" "metrica" "delta%"
awk -v thresh="$THRESH" -v lower="$LOWER" -f "$TMP/analyze.awk" "$TMP/res.tsv"
