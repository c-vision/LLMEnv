#!/bin/bash
# ── MUTEX di build: i comandi di build (cc/clang/cmake/make/dotnet) prendono un lock ──
# esclusivo su mutex.txt SOLO per la durata del SINGOLO comando, poi lo rilasciano subito
# (lo script puo proseguire a lungo senza tenere il lock). Scadenza 15 min con heartbeat:
# un processo in hang non blocca le altre AI. Vedi tools/gmb_with_lock.sh e Knowledge/
# Build-Mutex-Lock-AI.md.
if [ -z "${GMB_BUILDBIN_ON:-}" ]; then
  export GMB_BUILDBIN_ON=1
  GMB_BUILDBIN="$(cd "$(dirname "$0")" && pwd)/buildbin"
  [ -d "$GMB_BUILDBIN" ] && export PATH="$GMB_BUILDBIN:$PATH"
fi

# dsv41_stream_e2e_test.sh — E2E: motore DS4.1 materializzato (safetensors) vs STREAMING (.gmoe).
# Prerequisiti: /tmp/dsv41_st_np/sharded (gen_dsv41_st_ref.py --no-probes --out /tmp/dsv41_st_np)
# e /tmp/dsv41_st_np/tiny.gmoe (gmb_model_convert --generic).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; SRC="$ROOT/../LLMEnv/LLMEngine/mlx"; D="$SRC/build"
cc -std=gnu11 -O2 -Wno-unused-function -c -I"$SRC" -I "$ROOT/libs/mlx-deps/mlx-c-src" -I "$ROOT/libs/mlx-deps/mlx-src" \
   -I "$D/_deps/metal_cpp-src/include" -I "$D/_deps/cjson-src" -I "$ROOT/../LLMEnv/LLMEngine/moe/include" \
   -I "$ROOT/../LLMEnv/LLMCodec/include" -I "$ROOT/native/gmb_runtime_c/include" \
   -I "$ROOT/../LLMEnv/LLMMetal_oq_a8" -I "$ROOT/../LLMEnv/LLMMetal_mpp_q4" \
   "$SRC/tests/llm_dsv41_stream_e2e_test.c" -o /tmp/dsv41_stream_e2e_test.o
cc -std=gnu11 -O2 -Wno-unused-function -c -I"$SRC" -I "$ROOT/../LLMEnv/LLMEngine/moe/include" \
   -I "$ROOT/../LLMEnv/LLMCodec/include" -I "$ROOT/native/gmb_runtime_c/include" \
   "$SRC/gmb_engine_container.c" -o /tmp/dsv41_stream_e2e_container.o
c++ -O2 /tmp/dsv41_stream_e2e_test.o /tmp/dsv41_stream_e2e_container.o \
   "$D/_deps/mlx_c-build/libmlxc.a" "$D/_deps/mlx-build/libmlx.a" "$D/_deps/cjson-build/libcjson.a" \
   "$D/_deps/mlx-build/mlx/io/libgguflib.a" "$ROOT/../LLMEnv/LLMEngine/moe/build/libgmb_moe_runtime.a" \
   "$ROOT/../LLMEnv/LLMCodec/build/libgmb_ggml_codec.a" \
   -L "$ROOT/../LLMEnv/LLMCodec/build" -lgmb_codec_ggml -lgmb_codec_fp8 -lgmb_codec_mlx_affine \
   -Wl,-rpath,"$ROOT/../LLMEnv/LLMCodec/build" -framework Metal -framework Foundation \
   -framework QuartzCore -framework Accelerate -lcompression -lm -o /tmp/dsv41_stream_e2e_test
/tmp/dsv41_stream_e2e_test
