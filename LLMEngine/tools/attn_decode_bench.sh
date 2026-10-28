#!/bin/zsh
# attn_decode_bench.sh — GMB_DECODE_ATTENTION_BENCH (load-free): banda/costo del path
# attenzione+KV del decode al crescere del contesto, per 4 braccia, e verdetto se il residuo
# "pendenza +31% vs mlx_lm" (Bugs/(open) - MLX-Decode-Costo-Contesto-KV-Attention.md) sta qui.
#
# 1) KERNEL default : wrapper reale llm_mlx_gqa_sdpa vs call cruda MLX + tetto di lettura (sum K)
# 2) COLD           : 36 cache DISTINTE (=i layer) -> working set vero, fuori dalla L2
# 3) PATTERN        : append in-place + vista + SDPA per layer, UNA eval/token.
#                     append (donatable) vs ALIASED (viste del passo k vive durante l'append k+1)
#                     -> isola il costo di copia di mlx_slice_update_inplace (O(ctx)/token)
# 4) INTERF         : KV e pesi letti insieme -> superlinearita' (x_interf>1 = interferenza)
#
# Uso: tools/attn_decode_bench.sh [nh nkv hd] [NK "4096 8192 16384 32768"] [JSON out]
#   preset: qwen3-8b "32 8 128"; mixtral "32 8 128" (attn); GQA in generale "H Hkv 128".
# Output: tabella a stdout + JSON (default /tmp/attn_decode_bench.json) per W2 (risultati come dati).
set -u
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
NH=${1:-32}; NKV=${2:-8}; HD=${3:-128}
NKS=${4:-"4096 8192 16384 32768"}
JSON=${5:-/tmp/attn_decode_bench.json}
BIN=/tmp/gmb_attn_roof_probe_bench
export DYLD_LIBRARY_PATH=$SRC/build

echo "== build $BIN =="
/usr/bin/cc -std=gnu11 -O2 -Wno-unused-function -c \
  -I"$SRC" -I"$ROOT/libs/mlx-deps/mlx-c-src" -I"$ROOT/libs/mlx-deps/mlx-src" \
  -I"$SRC/build/_deps/metal_cpp-src/include" "$SRC/llm_mlx_attn_roof_probe.c" -o /tmp/attn_bench.o || exit 1
# Lo shim C++ della «foglia alias» (MODE I/J) serve a LINK: il probe la dichiara extern.
# Senza questo oggetto il link fallisce con `undefined symbols: _gmb_probe_alias_leaf*`.
if [ -f "$SRC/llm_mlx_attn_roof_probe_alias.cpp" ]; then
  /usr/bin/c++ -std=gnu++20 -O2 -w -c \
    -I"$SRC" -I"$ROOT/libs/mlx-deps/mlx-c-src" -I"$ROOT/libs/mlx-deps/mlx-src" \
    -I"$SRC/build/_deps/metal_cpp-src/include" \
    "$SRC/llm_mlx_attn_roof_probe_alias.cpp" -o /tmp/attn_bench_alias.o || exit 1
  SHIM=/tmp/attn_bench_alias.o
fi
/usr/bin/c++ -O2 /tmp/attn_bench.o ${SHIM:-} \
  "$SRC"/build/_deps/mlx_c-build/libmlxc.a "$SRC"/build/_deps/mlx-build/libmlx.a \
  "$SRC"/build/_deps/cjson-build/libcjson.a "$SRC"/build/_deps/mlx-build/mlx/io/libgguflib.a \
  "$SRC"/build/jaccl/libjaccl.a -framework Metal -framework Foundation -framework QuartzCore \
  -framework Accelerate -o "$BIN" || exit 1
echo "== geometry nh=$NH nkv=$NKV hd=$HD  NK=\"$NKS\" =="

python3 - "$BIN" "$NH" "$NKV" "$HD" "$NKS" "$JSON" <<'PY'
import json, re, subprocess, sys
bin_, nh, nkv, hd, nks, out = sys.argv[1:7]
env = {"LLM_ATTN_PROBE_NK": nks, "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
def run(extra, a5="0", a6="0", a7="4"):
    e = dict(env); e.update(extra)
    r = subprocess.run([bin_, nh, nkv, hd, "5", a5, a6, a7], env=e,
                       capture_output=True, text=True)
    return r.stdout
res = {"geometry": {"nh": int(nh), "nkv": int(nkv), "hd": int(hd)}, "nk": [int(x) for x in nks.split()], "modes": {}}
# COLD
cold = []
for ln in run({"LLM_ATTN_PROBE_COLD": "36"}).splitlines():
    m = re.match(r"\s*(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", ln)
    if m: cold.append(dict(nk=int(m[1]), ms=float(m[2]), gbs=int(m[3]), x_passi=float(m[6])))
res["modes"]["cold"] = cold
# PATTERN (aliased + concat)
pat = []
for ln in run({"LLM_ATTN_PROBE_PATTERN": "1", "LLM_ATTN_PROBE_ALIASED": "1", "LLM_ATTN_PROBE_CONCAT": "1"},
              a5="36", a6="24").splitlines():
    m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+\d+\s+[-\d.]+\s+[-\d.]+\s+([\d.]+)", ln)
    if m: pat.append(dict(nk=int(m[1]), append_ms=float(m[2]), noapp_ms=float(m[3]),
                          alias_ms=float(m[5]), alias_over_append=round(float(m[5])/float(m[2]), 3) if float(m[2]) else 0))
res["modes"]["pattern"] = pat
# INTERF
interf = []
for ln in run({"LLM_ATTN_PROBE_INTERF": "36", "LLM_ATTN_PROBE_WMB": "82", "LLM_ATTN_PROBE_NWB": "4"}).splitlines():
    m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", ln)
    if m: interf.append(dict(nk=int(m[1]), sdpa_ms=float(m[2]), w_ms=float(m[3]), both_ms=float(m[5]), x_interf=float(m[6])))
res["modes"]["interf"] = interf
with open(out, "w") as f: json.dump(res, f, indent=2)
print(f"JSON -> {out}")
# verdetto rapido
for row in cold:
    if row["nk"] >= 16384:
        print(f"  COLD Nk={row['nk']}: {row['ms']:.3f} ms/layer  {row['gbs']} GB/s  x_passi={row['x_passi']} "
              f"(1.00 = legge K+V una volta = al tetto)")
for row in pat:
    print(f"  PATTERN Nk={row['nk']}: append={row['append_ms']:.4f}  alias={row['alias_ms']:.4f}  "
          f"x{row['alias_over_append']} (aliased = viste vive = copia O(ctx))")
for row in interf:
    print(f"  INTERF Nk={row['nk']}: sdpa={row['sdpa_ms']:.4f} w={row['w_ms']:.4f} both={row['both_ms']:.4f}  "
          f"x_interf={row['x_interf']} (1.0 = additivo, nessuna interferenza)")
PY
echo "== done =="
