#!/bin/zsh
# =============================================================================
# ab_engine_rapid.sh — A/B motore GMBasic vs rapidMLX nella STESSA finestra.
# Metodo (kvf16_idle_bench / prompt): fase GMB e fase rapid ESEGUITE IN SEQUENZA
# nello stesso intervallo quiete (NON col server rapid residente durante la gamba GMB:
# tenerlo vivo penalizza GMB del 15-25%). Registra load1 prima/dopo ogni fase.
# Su macchina condivisa valgono i RAPPORTI, non gli assoluti; load>10 = non pubblicabile.
#
# CONTROLLO A ORDINE INVERTITO: --order rapid-first esegue la fase rapid PRIMA di quella
# GMB. Servono ENTRAMBI gli ordini per cancellare il bias di warm-up (la 2a fase beneficia
# di GPU/page-cache caldi): a load 5-8 il warm-up della 2a meta' produce falsi 5-10%.
#
# Uso:
#   tools/ab_engine_rapid.sh --model <dir> [--ns 4096,16384] [--m 64] [--reps 3] \
#                            [--port 8041] [--order gmb-first|rapid-first]
# Log: /tmp/ab_engine_rapid_<ts>/
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DLL="$ROOT/test_mlx_bench/bin/Release/net10.0/test_mlx_bench.dll"
export DYLD_LIBRARY_PATH="$ROOT/test_mlx_bench/bin/Release/net10.0"
PY="${GMB_AB_PY:-/Users/gaetano/ai/env312/bin/python}"   # override: GMB_AB_PY=~/ai/env-mlxbench/bin/python (mlx_lm 0.31.3 + transformers>=5)
export PATH="$(dirname "$PY"):$PATH"   # segue PY (default: env312)
DOTNET="$(command -v dotnet 2>/dev/null || echo /usr/local/share/dotnet/dotnet)"

MODEL=""; NS="4096,16384"; M=64; REPS=3; PORT=8041; ORDER="gmb-first"
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --ns) NS="$2"; shift 2;;
  --m) M="$2"; shift 2;; --reps) REPS="$2"; shift 2;; --port) PORT="$2"; shift 2;;
  --order) ORDER="$2"; shift 2;;
  *) echo "?$1"; exit 2;; esac; done
[ -n "$MODEL" ] || { echo "uso: tools/ab_engine_rapid.sh --model <dir> [--ns ..] [--m ..] [--reps ..]"; exit 2; }
NAME="$(basename "$MODEL")"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="/tmp/ab_engine_rapid_$TS"; mkdir -p "$LOG"
l1() { uptime | sed 's/.*load averages: //' | awk '{print $1}' | tr ',' '.'; }

echo "[ab] $NAME  N=$NS  M=$M  reps=$REPS  order=$ORDER  log=$LOG"

phase_gmb() {
  echo "[ab] === FASE GMB (load1=$(l1)) ==="
  gmb_out="$LOG/gmb.txt"; : > "$gmb_out"
  for N in ${(s:,:)NS}; do
    # UNA sola load del modello per N: `rdecodectx` fa prefill N + decode M, REPS volte, con
    # warm-up interno -> valori per-rep per il MAD, senza REPS ricaricamenti (minuti per modello
    # grande). Prefill a N>=4096 (a N corto la finestra misura il boost GPU, non il throughput).
    "$DOTNET" "$DLL" "$MODEL" rdecodectx "$N" "$M" "$REPS" 2>/dev/null \
      | grep DECODECTX_REP \
      | sed -E 's/.*N=([0-9]+) prefill tok\/s=([0-9.,]+) decode tok\/s=([0-9.,]+)/\1 \2 \3/' \
      | tr ',' '.' >> "$gmb_out"
    grep "^$N " "$gmb_out" | while read -r n p v; do
      printf "  GMB   N=%-6s prefill=%s  decode=%s tok/s  (load1=%s)\n" "$n" "$p" "$v" "$(l1)"
    done
  done
}

phase_rapid() {
  echo "[ab] === FASE RAPID (load1=$(l1)) ===  server :$PORT"
  rapid-mlx serve "$MODEL" --served-model-name "$NAME" --port "$PORT" --disable-prefix-cache \
    --kv-cache-dtype bf16 >"$LOG/rapid_server.log" 2>&1 &
  SRV=$!
  trap 'kill -9 $SRV 2>/dev/null' EXIT INT TERM
  for i in $(seq 1 180); do
    curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 1
  done
  rapid_out="$LOG/rapid.txt"; : > "$rapid_out"
  "$PY" "$ROOT/tools/bench_rapidmlx.py" --url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
        --lengths ${(s:,:)NS} --max-tokens "$M" --repeats "$REPS" 2>&1 \
    | tee "$LOG/rapid_bench.log" \
    | grep -oE 'N=[0-9]+ rep[0-9]+: prefill [0-9.]+ tok/s \(TTFT [0-9.]+s\) decode [0-9.]+ tok/s' \
    | sed -E 's/^N=([0-9]+) rep[0-9]+: prefill ([0-9.]+) tok\/s.* decode ([0-9.]+) tok\/s/\1 \2 \3/' >> "$rapid_out"
  while read -r N p v; do printf "  RAPID N=%-6s prefill=%s  decode=%s tok/s  (load1=%s)\n" "$N" "$p" "$v" "$(l1)"; done < "$rapid_out"
  kill -9 $SRV 2>/dev/null; trap - EXIT
}

if [ "$ORDER" = "rapid-first" ]; then phase_rapid; phase_gmb; else phase_gmb; phase_rapid; fi

echo
echo "# GMB vs rapidMLX — mediane + MAD + rapporto. order=$ORDER  load finale $(l1)"
echo "#   MAD <= 5% = misura ripetibile; altrimenti NON citare il rapporto."
for N in ${(s:,:)NS}; do
  python3 - "$N" "$gmb_out" "$rapid_out" <<'PY'
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
    print(f"  N={N:<6} PREFILL gmb={fmt(gpf)} rapid={fmt(rpf)} -> {gpf[0]/rpf[0]:.3f}x | "
          f"DECODE gmb={fmt(gdc)} rapid={fmt(rdc)} -> {gdc[0]/rdc[0]:.3f}x  [{ok}]")
else:
    print(f"  N={N:<6} dati incompleti (gmb={gpf[0]}/{gdc[0]} rapid={rpf[0]}/{rdc[0]})")
PY
done
echo "[ab] log in $LOG"
