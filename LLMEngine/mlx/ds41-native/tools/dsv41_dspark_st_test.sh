#!/usr/bin/env bash
# dsv41_dspark_st_test.sh — test UNITARIO MTP/DSPARK di DeepSeek V4.1 (fixture sintetica "tiny").
#
# NON richiede i pesi reali DS4.1: genera /tmp/dsv41_st/mtp.safetensors +
# gmb_dsv41_dspark_st_ref.h (via gen_dsv41_dspark_st_ref.py, venv ~/ai/env-omlx) e verifica
# llm_dsv41_load_dspark (loader `mtp.N.*`) + forward_spec contro il riferimento (proposed/
# biases/confidence). Adattato al layout post-migrazione (LLMEnv). Metodo "unità-first" (§0sexies).
#
# Uso: bash tools/dsv41_dspark_st_test.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)                       # LLMEnv
GEN="${DSV41_GEN:-$HOME/llmenv-engine/tools/gen_dsv41_dspark_st_ref.py}"
PY="${OMLX_PY:-$HOME/ai/env-omlx/bin/python}"
MX="$ROOT/LLMEngine/mlx/build-llmenv"
OUT=/tmp/dsv41_st

[ -f "$GEN" ] || { echo "generator assente: $GEN (imposta DSV41_GEN)"; exit 2; }
[ -x "$PY" ]  || { echo "venv omlx assente: $PY (imposta OMLX_PY)"; exit 2; }

# 1) fixture sintetica: patcha il generator per scrivere l'header ref in $OUT (layout migrato).
mkdir -p "$OUT"
TMPGEN="$(mktemp -d)/gen_st.py"
cp "$GEN" "$TMPGEN"
python3 - "$TMPGEN" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "gmb_dsv41_dspark_st_ref.h")',
            'dst = os.path.join(outdir, "gmb_dsv41_dspark_st_ref.h")')
open(p,"w").write(s)
PY
"$PY" "$TMPGEN" >/dev/null

# 2) compila + linka + esegue (layout LLMEnv).
INC="-I $ROOT/LLMEngine/mlx -I $OUT -I $ROOT/LLMCodec/include -I $ROOT/LLMEngine/include -I $ROOT/LLMEngine/moe/include -I $ROOT/LLMToken/include -I $ROOT/LLMEngine/Options -I $ROOT/libs/mlx-deps/mlx-c-src -I $ROOT/libs/mlx-deps/mlx-src -I $MX/_deps/metal_cpp-src/include -I $MX/_deps/cjson-src"
cc -std=gnu11 -O2 -Wno-unused-function $INC -c "$ROOT/LLMEngine/mlx/tests/llm_dsv41_dspark_st_test.c" -o /tmp/dsv41_st_test.o
cc -O2 -I "$ROOT/LLMEngine/Options" -c "$ROOT/LLMEngine/Options/llm_engine_opts.c" -o /tmp/dsv41_opts.o
c++ -O2 /tmp/dsv41_st_test.o /tmp/dsv41_opts.o \
  "$MX"/_deps/mlx_c-build/libmlxc.a "$MX"/_deps/mlx-build/libmlx.a \
  "$MX"/_deps/cjson-build/libcjson.a "$MX"/_deps/mlx-build/mlx/io/libgguflib.a "$MX"/jaccl/libjaccl.a \
  -L"$MX" -lllmcodec_fp8 -Wl,-rpath,"$MX" \
  -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
  -o /tmp/gmb_dsv41_dspark_st_test
exec /tmp/gmb_dsv41_dspark_st_test
