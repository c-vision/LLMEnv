#!/bin/bash
# build_probes.sh — compila le SONDE-ORACOLO dai sorgenti LOCALI (in questa cartella, non
# tracciata). Sono "model-free": NON caricano un modello, costruiscono grafi sintetici e
# misurano banda/nodi/costo-closure. Nessun peso, nessuna run modello, nessun rischio RAM.
#
# Uso:  build_probes.sh [build_dir]
#   default: FastLLM/build-release (fallback LLMEnv/build).
#   Richiede libmlxc.a/libmlx.a in <build>/_deps e la dylib engine in <build>/llmengine_mlx.
#
# NOTE DI COMPILAZIONE (imparate a spese nostre):
#  - I driver vanno compilati in C (clang -std=c17), NON C++: ridefiniscono simboli mlx-c
#    dentro main senza extern "C" -> in C++ il nome è MANGLED e il link fallisce.
#  - Le sonde che chiamano funzioni dell'engine (llm_ai_mlx_gemv_bench, llm_ai_mlx_bw_test, ...)
#    linkano la dylib `llmengine_mlx/libllm_ai_mlx.dylib` (+ le statiche mlx).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HOME/Documents/Projects/LLMEnv"
MLXC="$ROOT/libs/mlx-deps/mlx-c-src"
MLXS="$ROOT/libs/mlx-deps/mlx-src"
B="${1:-$HOME/Documents/Projects/FastLLM/build-release}"
[ -e "$B/_deps/mlx_c-build/libmlxc.a" ] || B="$ROOT/build"
if [ ! -e "$B/_deps/mlx_c-build/libmlxc.a" ]; then
  echo "ERRORE: libmlxc.a non trovata in $B/_deps/mlx_c-build. Passa il build_dir giusto." >&2
  exit 2
fi
DY="$B/llmengine_mlx/libllm_ai_mlx.dylib"
INC="-I$MLXC -I$MLXS -I$B/_deps/metal_cpp-src -I$ROOT/LLMEngine/mlx -I$ROOT/LLMEngine/include -I$ROOT/LLMEngine/Options"
LINK=(); [ -e "$DY" ] && LINK=("$DY" -Wl,-rpath,"$B/llmengine_mlx")
STATIC=("$B/_deps/mlx_c-build/libmlxc.a" "$B/_deps/mlx-build/libmlx.a" \
        "$B/_deps/cjson-build/libcjson.a" "$B/_deps/mlx-build/mlx/io/libgguflib.a" \
        "$B/jaccl/libjaccl.a")
FW=(-framework Metal -framework Foundation -framework QuartzCore -framework Accelerate)

for p in llm_gemv_bench_test llm_bw_test llm_mlx_compile_cost_probe llm_qwen4_build_probe llm_gather_qmm_sorted_probe kv_append_probe llm_moe_forms_probe llm_glm_router_compile_probe llm_shared_expert_probe llm_glm_qkv_probe llm_q4_gate_probe llm_m5_tensor_path_probe; do
  [ -f "$HERE/$p.c" ] || { echo "skip $p"; continue; }
  if clang -std=c17 -O2 -Wno-deprecated-declarations $INC "$HERE/$p.c" "${LINK[@]}" \
       "${STATIC[@]}" "${FW[@]}" -o "$HERE/$p" 2>"/tmp/build_$p.err"; then
    echo "OK   $p"
  else echo "FAIL $p:"; sed -n '1,4p' "/tmp/build_$p.err"; fi
done

# llm_moe_guard_test: include l'header interno dell'engine -> set di include completo + dylib.
if [ -f "$HERE/llm_moe_guard_test.c" ]; then
  EINC=(-I"$ROOT/LLMEngine/mlx/../plugin" -I"$ROOT/LLMEngine/mlx/../../LLMDesc" \
        -I"$ROOT/LLMEngine/mlx/../include" -I"$B/_deps/cjson-src" \
        -I"$ROOT/LLMEngine/mlx/../../LLMMetal/oq_a8" -I"$ROOT/LLMEngine/mlx/../../LLMMetal/mpp_q4" \
        -I"$ROOT/LLMEngine/mlx/." -I"$ROOT/LLMEngine/mlx/../Options" \
        -I"$ROOT/LLMEngine/mlx/../moe/include" -I"$ROOT/LLMEngine/moe/include" \
        -I"$ROOT/LLMEngine/Options" -I"$ROOT/LLMCodec/include" -I"$ROOT/LLMDesc/.")
  if clang -std=c17 -O2 -Wno-deprecated-declarations $INC "${EINC[@]}" "$HERE/llm_moe_guard_test.c" \
       "${LINK[@]}" "${STATIC[@]}" "${FW[@]}" -o "$HERE/llm_moe_guard_test" 2>"/tmp/build_guard.err"; then
    echo "OK   llm_moe_guard_test"
  else echo "FAIL llm_moe_guard_test:"; sed -n '1,6p' "/tmp/build_guard.err"; fi
  if clang -std=c17 -O2 -Wno-deprecated-declarations $INC "${EINC[@]}" "$HERE/llm_moe_fused_parity_probe.c" \
       "${LINK[@]}" "${STATIC[@]}" "${FW[@]}" -o "$HERE/llm_moe_fused_parity_probe" 2>"/tmp/build_fused.err"; then
    echo "OK   llm_moe_fused_parity_probe"
  else echo "FAIL llm_moe_fused_parity_probe:"; sed -n '1,8p' "/tmp/build_fused.err"; fi
fi

# llm_dsv4_batch_moe_oracle: include l'header dello store DS4 -> set include ds4-slow + ds41-native.
if [ -f "$HERE/llm_dsv4_batch_moe_oracle.c" ]; then
  DSINC=(-I"$ROOT/LLMEngine/mlx/ds4-slow" -I"$ROOT/LLMEngine/mlx/ds41-native")
  if clang -std=c17 -O2 -Wno-deprecated-declarations $INC "${DSINC[@]}" "$HERE/llm_dsv4_batch_moe_oracle.c" \
       "${LINK[@]}" "${STATIC[@]}" "${FW[@]}" -o "$HERE/llm_dsv4_batch_moe_oracle" 2>"/tmp/build_ds4moe.err"; then
    echo "OK   llm_dsv4_batch_moe_oracle"
  else echo "FAIL llm_dsv4_batch_moe_oracle:"; sed -n '1,8p' "/tmp/build_ds4moe.err"; fi
fi

# attn_roof: il driver .c (C) + alias .cpp (C++) -> link insieme.
if [ -f "$HERE/llm_mlx_attn_roof_probe.c" ]; then
  if clang -std=c17 -O2 -Wno-deprecated-declarations $INC -c "$HERE/llm_mlx_attn_roof_probe.c" -o /tmp/attn_roof.o 2>/tmp/build_attn.err \
     && clang++ -std=c++20 -O2 -Wno-deprecated-declarations $INC -I"$MLXS" -c "$HERE/llm_mlx_attn_roof_probe_alias.cpp" -o /tmp/attn_alias.o 2>>/tmp/build_attn.err \
     && clang++ /tmp/attn_roof.o /tmp/attn_alias.o "${LINK[@]}" "${STATIC[@]}" "${FW[@]}" -o "$HERE/llm_mlx_attn_roof_probe" 2>>/tmp/build_attn.err; then
    echo "OK   llm_mlx_attn_roof_probe"
  else echo "FAIL llm_mlx_attn_roof_probe:"; sed -n '1,5p' /tmp/build_attn.err; fi
fi
