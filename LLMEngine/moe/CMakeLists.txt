cmake_minimum_required(VERSION 3.20)
project(llmengine_moe C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
endif()

option(LLMENGINE_MOE_ASAN "Build with AddressSanitizer (gate: ASAN pulito)" OFF)
option(LLMENGINE_MOE_TSAN "Build with ThreadSanitizer (gate M4: dedup concorrente)" OFF)
option(LLMENGINE_MOE_UBSAN "Build with UndefinedBehaviorSanitizer (gate M19)" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../LLMDesc ${CMAKE_CURRENT_SOURCE_DIR}/../include)
# codec MLX affine (llmcodec_moe_affine_quantize/dequantize) ora nel progetto llmcodec.
# GUARD: se un altro consumatore della stessa build-tree (es. llm_runtime in llm_vm_c) lo ha già
# aggiunto, i target esistono già → un secondo add_subdirectory collide (CMP0002). Stesso pattern
# del guard più sotto (riga ~188).
if(NOT TARGET llmcodec)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../LLMCodec llmcodec_moe)
endif()
# Registro opzioni condiviso (parametri senza env): core, dipendenza anche di MoE.
if(NOT TARGET llmengine_opts)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Options llmengine_opts_moe)
endif()
# NIENTE LLMDesc/llmdesc.c qui: il MoE non referenzia alcun llmdesc_* (0 simboli U) e llmdesc
# e' gia' un target a se' (LLMDesc/, add_library(llmdesc STATIC llmdesc.c)). Compilarlo anche
# qui era duplicazione sorgente: lo stesso llmdesc.c.o finiva in libllmengine_moe.a E in
# libllmdesc.a E in libllm_ai_mlx.
add_library(llmengine_moe STATIC
    llm_engine_moe.c   # adapter all'ABI comune llmengine
    src/llmengine_moe_arch.c
    src/llmengine_moe_json.c
    src/llmengine_moe_ggml.c
    src/llmengine_moe_gguf.c
    src/llmengine_moe_gguf_expert.c
    src/llmengine_moe_mla.c
    src/llmengine_moe_quant_alloc.c
    src/llmengine_moe_discover.c
    src/llmengine_moe_adapter_gguf.c
    src/llmengine_moe.c
    src/llmengine_moe_cache.c
    src/llmengine_moe_async.c
    src/llmengine_moe_prefetch.c
    src/llmengine_moe_slot.c
    src/llmengine_moe_mem.c
    src/llmengine_moe_kv.c
    src/llmengine_moe_prof.c
    src/llmengine_moe_sim.c
    src/llmengine_moe_provider.c
    src/llmengine_moe_qwen3.c
    src/llmengine_moe_diverge.c
    src/llmengine_moe_diverge_metal.mm
    src/llmengine_moe_qwen3_forward.c
)
target_include_directories(llmengine_moe PUBLIC include)
target_link_libraries(llmengine_moe PUBLIC llmengine_opts)
target_compile_options(llmengine_moe PRIVATE -Wall -Wextra -Wpedantic)


# llm-discover (Auto-Discovery: descriptor da modello su disco, senza conversione)
# [GGUF DEPRECATO] target llm-discover rimosso









# test
# [GGUF DEPRECATO] test_moe_runtime rimosso

# [GGUF DEPRECATO] target test_moe_format rimosso

# [GGUF DEPRECATO] target test_moe_fuzz rimosso

# [GGUF DEPRECATO] target test_moe_stats rimosso

# [GGUF DEPRECATO] target test_moe_residency rimosso

# [GGUF DEPRECATO] target test_moe_adaptive rimosso

# [GGUF DEPRECATO] target test_moe_stream rimosso

add_executable(test_moe_cache tests/test_moe_cache.c)
target_link_libraries(test_moe_cache PRIVATE llmengine_moe)
target_compile_options(test_moe_cache PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_async tests/test_moe_async.c)
target_link_libraries(test_moe_async PRIVATE llmengine_moe)
target_compile_options(test_moe_async PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_prefetch tests/test_moe_prefetch.c)
target_link_libraries(test_moe_prefetch PRIVATE llmengine_moe)
target_compile_options(test_moe_prefetch PRIVATE -Wall -Wextra -Wpedantic)

# [GGUF DEPRECATO] target test_moe_compress rimosso

add_executable(test_moe_slot tests/test_moe_slot.c)
target_link_libraries(test_moe_slot PRIVATE llmengine_moe)
target_compile_options(test_moe_slot PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_cache_bench tests/test_moe_cache_bench.c)
target_link_libraries(test_moe_cache_bench PRIVATE llmengine_moe)
target_compile_options(test_moe_cache_bench PRIVATE -Wall -Wextra -Wpedantic)

# D19: expert cache adattiva (hotness + profilo di routing)
add_executable(test_moe_cache_adaptive tests/test_moe_cache_adaptive.c)
target_link_libraries(test_moe_cache_adaptive PRIVATE llmengine_moe)
target_compile_options(test_moe_cache_adaptive PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_mem tests/test_moe_mem.c)
target_link_libraries(test_moe_mem PRIVATE llmengine_moe)
target_compile_options(test_moe_mem PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_kv tests/test_moe_kv.c)
target_link_libraries(test_moe_kv PRIVATE llmengine_moe)
target_compile_options(test_moe_kv PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_prof tests/test_moe_prof.c)
target_link_libraries(test_moe_prof PRIVATE llmengine_moe)
target_compile_options(test_moe_prof PRIVATE -Wall -Wextra -Wpedantic)

# K1: replay offline della cache da trace (moe_sim) + test di parità col runtime.
add_executable(moe_sim tools/moe_sim.c)
target_link_libraries(moe_sim PRIVATE llmengine_moe)
target_compile_options(moe_sim PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_sim tests/test_moe_sim.c)
target_link_libraries(test_moe_sim PRIVATE llmengine_moe)
target_compile_options(test_moe_sim PRIVATE -Wall -Wextra -Wpedantic)

# [GGUF DEPRECATO] target test_moe_provider rimosso

# [GGUF DEPRECATO] target test_moe_qwen3 rimosso

# [GGUF DEPRECATO] target test_moe_qwen3_fwd rimosso

# [GGUF DEPRECATO] target test_moe_adapter_qwen3 rimosso

add_executable(test_moe_discover tests/test_moe_discover.c)
target_link_libraries(test_moe_discover PRIVATE llmengine_moe)
target_compile_options(test_moe_discover PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_gguf_adapter tests/test_moe_gguf_adapter.c)
target_link_libraries(test_moe_gguf_adapter PRIVATE llmengine_moe m)
target_compile_options(test_moe_gguf_adapter PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_gguf_store tests/test_moe_gguf_store.c)
target_link_libraries(test_moe_gguf_store PRIVATE llmengine_moe m)
target_compile_options(test_moe_gguf_store PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_mla tests/test_moe_mla.c)
target_link_libraries(test_moe_mla PRIVATE llmengine_moe m)
target_compile_options(test_moe_mla PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_moe_quant tests/test_moe_quant.c)
target_link_libraries(test_moe_quant PRIVATE llmengine_moe m)
target_compile_options(test_moe_quant PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_quant_alloc tests/test_quant_alloc.c)
target_link_libraries(test_quant_alloc PRIVATE llmengine_moe m)
target_compile_options(test_quant_alloc PRIVATE -Wall -Wextra -Wpedantic)
find_package(Threads REQUIRED)
target_link_libraries(llmengine_moe PUBLIC Threads::Threads)
if(NOT TARGET llmcodec)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../LLMCodec llmcodec_moe2)
endif()
target_link_libraries(llmengine_moe PUBLIC llmcodec llmcodec_mlx_affine llmcodec_fp8 llmcodec_ggml)   # plugin codec (ggml: llmcodec_ggml_type_bytes usato dal reader del codec)

if(APPLE)
    target_link_libraries(llmengine_moe PUBLIC compression)  # libcompression (LZ4/LZFSE, M9)
    target_sources(llmengine_moe PRIVATE src/llmengine_moe_backend_metal.mm)
    target_link_libraries(llmengine_moe PUBLIC "-framework Metal" "-framework Foundation")
    add_executable(test_moe_backend_metal tests/test_moe_backend_metal.c)
    target_link_libraries(test_moe_backend_metal PRIVATE llmengine_moe)
    target_compile_options(test_moe_backend_metal PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME moe_backend_metal COMMAND test_moe_backend_metal)
endif()

# PIC per llmengine_moe (richiesto dalle dylib consumer).
if(NOT LLMENGINE_MOE_ASAN AND NOT LLMENGINE_MOE_TSAN AND NOT LLMENGINE_MOE_UBSAN)
    set_target_properties(llmengine_moe PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # ABI C stabile del provider MoE out-of-core (ex "moestream"), consumata dal C# via P/Invoke.
    # Ripristinata 2026-09-25 (era stata rimossa insieme al formato .gmoe; il runtime e' GGUF).
    add_library(llmengine_moe_stream SHARED src/llmengine_ai_moestream.c)
    target_link_libraries(llmengine_moe_stream PRIVATE llmengine_moe)
    target_compile_options(llmengine_moe_stream PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(llmengine_moe_stream PROPERTIES OUTPUT_NAME llmengine_moe_stream)
endif()

if(LLMENGINE_MOE_ASAN)
    foreach(t llmengine_moe test_moe_cache test_moe_async test_moe_prefetch test_moe_cache_bench test_moe_slot test_moe_mem test_moe_kv test_moe_prof test_moe_discover test_moe_gguf_adapter test_moe_gguf_store test_moe_mla test_moe_quant test_quant_alloc)
        target_compile_options(${t} PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
        target_link_options(${t} PRIVATE -fsanitize=address)
    endforeach()
    if(APPLE AND TARGET test_moe_backend_metal)
        target_compile_options(test_moe_backend_metal PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
        target_link_options(test_moe_backend_metal PRIVATE -fsanitize=address)
    endif()
endif()

if(LLMENGINE_MOE_TSAN)
    foreach(t llmengine_moe test_moe_cache test_moe_async test_moe_prefetch test_moe_cache_bench test_moe_slot test_moe_mem test_moe_kv test_moe_prof test_moe_discover test_moe_gguf_adapter test_moe_gguf_store test_moe_mla test_moe_quant test_quant_alloc)
        target_compile_options(${t} PRIVATE -fsanitize=thread -fno-omit-frame-pointer -g)
        target_link_options(${t} PRIVATE -fsanitize=thread)
    endforeach()
    if(APPLE AND TARGET test_moe_backend_metal)
        target_compile_options(test_moe_backend_metal PRIVATE -fsanitize=thread -fno-omit-frame-pointer -g)
        target_link_options(test_moe_backend_metal PRIVATE -fsanitize=thread)
    endif()
endif()

if(LLMENGINE_MOE_UBSAN)
    foreach(t llmengine_moe test_moe_cache test_moe_async test_moe_prefetch test_moe_cache_bench test_moe_slot test_moe_mem test_moe_kv test_moe_prof test_moe_discover test_moe_gguf_adapter test_moe_gguf_store test_moe_mla test_moe_quant test_quant_alloc)
        target_compile_options(${t} PRIVATE -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
        target_link_options(${t} PRIVATE -fsanitize=undefined)
    endforeach()
    if(APPLE AND TARGET test_moe_backend_metal)
        target_compile_options(test_moe_backend_metal PRIVATE -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
        target_link_options(test_moe_backend_metal PRIVATE -fsanitize=undefined)
    endif()
endif()

enable_testing()
add_test(NAME moe_runtime COMMAND test_moe_runtime)
add_test(NAME moe_format COMMAND test_moe_format)
add_test(NAME moe_fuzz COMMAND test_moe_fuzz)
add_test(NAME moe_stats COMMAND test_moe_stats)
add_test(NAME moe_residency COMMAND test_moe_residency)
add_test(NAME moe_adaptive COMMAND test_moe_adaptive)
add_test(NAME moe_stream COMMAND test_moe_stream)
add_test(NAME moe_cache COMMAND test_moe_cache)
add_test(NAME moe_async COMMAND test_moe_async)
add_test(NAME moe_prefetch COMMAND test_moe_prefetch)
add_test(NAME moe_cache_bench COMMAND test_moe_cache_bench)
add_test(NAME moe_cache_adaptive COMMAND test_moe_cache_adaptive)
add_test(NAME moe_slot COMMAND test_moe_slot)
add_test(NAME moe_compress COMMAND test_moe_compress)
add_test(NAME moe_mem COMMAND test_moe_mem)
add_test(NAME moe_kv COMMAND test_moe_kv)
add_test(NAME moe_prof COMMAND test_moe_prof)
add_test(NAME moe_provider COMMAND test_moe_provider)
add_test(NAME moe_qwen3 COMMAND test_moe_qwen3)
add_test(NAME moe_qwen3_fwd COMMAND test_moe_qwen3_fwd)
add_test(NAME moe_adapter_qwen3 COMMAND test_moe_adapter_qwen3)
add_test(NAME moe_discover COMMAND test_moe_discover)
add_test(NAME moe_gguf_adapter COMMAND test_moe_gguf_adapter)
add_test(NAME moe_gguf_store COMMAND test_moe_gguf_store)
add_test(NAME moe_mla COMMAND test_moe_mla)
add_test(NAME moe_quant COMMAND test_moe_quant)
add_test(NAME quant_alloc COMMAND test_quant_alloc)
set_target_properties(llmengine_moe PROPERTIES INSTALL_RPATH "@loader_path")

add_executable(test_moe_diverge tests/test_moe_diverge.c src/llmengine_moe_diverge.c)
