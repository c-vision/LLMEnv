#!/bin/zsh
# =============================================================================
# ab_engine_mlxlm.sh — A/B motore GMBasic vs mlx_lm (reference eager), STESSA finestra.
#
# STRATEGIA (2026-09-13): l'obiettivo primario e' la PARITA' con mlx_lm su ogni
# modello (prefill + decode). rapidMLX = strato mlx_lm + strato server: finche' non
# si pareggia mlx_lm NON si testa rapid. Questo script misura SOLO GMB vs mlx_lm.
#
# Metodo (come ab_engine_rapid.sh): fase GMB e fase mlx_lm in sequenza nello stesso
# intervallo quiete; load1 registrato prima/dopo ogni fase. Su macchina condivisa
# valgono i RAPPORTI. CONTROLLO A ORDINE INVERTITO obbligatorio: --order mlxlm-first
# esegue mlx_lm prima di GMB (il warm-up della 2a fase produce falsi 5-10%).
#
# Uso:
#   tools/ab_engine_mlxlm.sh --model <dir> [--ns 4096] [--m 64] [--reps 2] \
#                            [--order gmb-first|mlxlm-first]
# Log: /tmp/ab_engine_mlxlm_<ts>/
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DLL="$ROOT/test_mlx_bench/bin/Release/net10.0/test_mlx_bench.dll"
export DYLD_LIBRARY_PATH="$ROOT/test_mlx_bench/bin/Release/net10.0"
PY="${GMB_AB_PY:-/Users/gaetano/ai/env312/bin/python}"   # override: GMB_AB_PY=~/ai/env-mlxlm/bin/python (mlx-lm 0.29.1 + transformers 4.57.6)
export PATH="/Users/gaetano/ai/env312/bin:$PATH"
DOTNET="$(command -v dotnet 2>/dev/null || echo /usr/local/share/dotnet/dotnet)"

MODEL=""; NS="4096"; M=64; REPS=2; ORDER="gmb-first"
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --ns) NS="$2"; shift 2;;
  --m) M="$2"; shift 2;; --reps) REPS="$2"; shift 2;;
  --order) ORDER="$2"; shift 2;;
  *) echo "?$1"; exit 2;; esac; done
[ -n "$MODEL" ] || { echo "uso: tools/ab_engine_mlxlm.sh --model <dir> [--ns ..] [--m ..] [--reps ..]"; exit 2; }
NAME="$(basename "$MODEL")"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="/tmp/ab_engine_mlxlm_$TS"; mkdir -p "$LOG"
l1() { uptime | sed 's/.*load averages: //' | awk '{print $1}' | tr ',' '.'; }

echo "[ab] $NAME  N=$NS  M=$M  reps=$REPS  order=$ORDER  log=$LOG"

phase_gmb() {
  echo "[ab] === FASE GMB (load1=$(l1)) ==="
  gmb_out="$LOG/gmb.txt"; : > "$gmb_out"
  for N in ${(s:,:)NS}; do
    # UNA sola load per N: `rdecodectx` prefill N + decode M, REPS volte, warm-up interno ->
    # per-rep per il MAD senza REPS ricaricamenti. Prefill a N>=4096 (a N corto la finestra
    # misura il boost GPU, non il throughput).
    "$DOTNET" "$DLL" "$MODEL" rdecodectx "$N" "$M" "$REPS" 2>/dev/null \
      | grep DECODECTX_REP \
      | sed -E 's/.*N=([0-9]+) prefill tok\/s=([0-9.,]+) decode tok\/s=([0-9.,]+)/\1 \2 \3/' \
      | tr ',' '.' >> "$gmb_out"
    grep "^$N " "$gmb_out" | while read -r n p v; do
      printf "  GMB    N=%-6s prefill=%s  decode=%s tok/s  (load1=%s)\n" "$n" "$p" "$v" "$(l1)"
    done
  done
}

phase_mlxlm() {
  echo "[ab] === FASE MLX_LM (load1=$(l1)) ==="
  mlx_out="$LOG/mlxlm.txt"; : > "$mlx_out"
  # GMB_BENCH_PERREP=1 -> bench_mlxlm.py stampa i valori PER-REP (per il MAD); una sola load.
  GMB_BENCH_PERREP=1 "$PY" "$ROOT/tools/bench_mlxlm.py" --model "$MODEL" --lengths "$NS" \
        --max-tokens "$M" --repeats "$REPS" 2>&1 \
    | tee "$LOG/mlxlm_bench.log" \
    | grep -oE 'REP n=[0-9]+ prefill [0-9.]+ tok/s decode [0-9.]+ tok/s' \
    | sed -E 's/^REP n=([0-9]+) prefill ([0-9.]+) tok\/s decode ([0-9.]+) tok\/s/\1 \2 \3/' >> "$mlx_out"
  while read -r N p v; do printf "  MLX_LM N=%-6s prefill=%s  decode=%s tok/s  (load1=%s)\n" "$N" "$p" "$v" "$(l1)"; done < "$mlx_out"
}

if [ "$ORDER" = "mlxlm-first" ]; then phase_mlxlm; phase_gmb; else phase_gmb; phase_mlxlm; fi

echo
echo "# GMB vs mlx_lm — mediane + MAD + rapporto (parita' = 1.000x). order=$ORDER  load finale $(l1)"
echo "#   MAD <= 5% = misura ripetibile; altrimenti NON citare il rapporto."
for N in ${(s:,:)NS}; do
  python3 - "$N" "$gmb_out" "$mlx_out" <<'PY'
import sys, statistics as st
N, gf, rf = sys.argv[1], sys.argv[2], sys.argv[3]
def col(path, n, c):
    out=[]
    for line in open(path):
        p=line.split()
        if len(p)>=3 and p[0]==n:
            try: out.append(float(p[c]))
            except ValueError: pass
    return out
def stat(v):
    if not v: return (0.0, 0.0)
    m=st.median(v); mad=(st.median([abs(x-m) for x in v])/m*100) if (m and len(v)>1) else 0.0
    return (m, mad)
def fmt(t): return f"{t[0]:.0f}(MAD {t[1]:.1f}%)"
gpf, gdc = stat(col(gf,N,1)), stat(col(gf,N,2))
rpf, rdc = stat(col(rf,N,1)), stat(col(rf,N,2))
if gpf[0] and rpf[0] and gdc[0] and rdc[0]:
    ok = "OK" if max(gpf[1],gdc[1],rpf[1],rdc[1]) <= 5.0 else "RUMORE"
    print(f"  N={N:<6} PREFILL gmb={fmt(gpf)} mlx_lm={fmt(rpf)} -> {gpf[0]/rpf[0]:.3f}x | "
          f"DECODE gmb={fmt(gdc)} mlx_lm={fmt(rdc)} -> {gdc[0]/rdc[0]:.3f}x  [{ok}]")
else:
    print(f"  N={N:<6} dati incompleti (gmb={gpf[0]}/{gdc[0]} mlx_lm={rpf[0]}/{rdc[0]})")
PY
done
echo "[ab] log in $LOG"
