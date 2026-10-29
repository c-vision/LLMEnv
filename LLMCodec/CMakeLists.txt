# Progetto C AUTONOMO: codec GGML (blocco<->f32) + (futuro) conversioni GGUF<->MLX.
# Separato dal runtime LLMEnv: nessuna dipendenza dal resto (solo libm). Chi ne ha bisogno
# (llm_runtime, il converter) lo aggancia come subdirectory e ne linka la libreria.
cmake_minimum_required(VERSION 3.16)
project(llmcodec C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CORE: contenitori + descrizione + registry + reader/parser. NESSUNA implementazione codec.
# Il CONVERTER (produce/trasforma pesi) NON vive qui: e' in GMBasic/gmbllm (PUL), che ci linka.
add_library(llmcodec STATIC
    src/llmcodec_gguf_writer.c     # contenitore GGUF (I/O) + tabella byte-per-tipo
    src/llmcodec_safetensors.c     # contenitore safetensors (I/O)
    src/llmcodec_gguf_parse.c      # PARSER GGUF canonico buffer-based (reader unico)
    src/llmcodec_gguf_status.c     # TASSONOMIA D'ERRORE GGUF (asset assorbito da lane_gguf)
    src/llmcodec_gguf_meta.c       # METADATA GGUF (hyperparametri + tokenizer) per il converter
    src/llmcodec_gguf_compare.c    # confronto numerico per-tensore tra due GGUF (lineage/breakdown)
    src/llmcodec_rolemap.c         # descrizione .llmarch
    src/llmcodec.c           # registry PLUGIN (dlopen + cache)
    src/llmcodec_quant_awq.c       # decoder AWQ int4 GEMM (autoawq gemm layout) per l'engine MLX
    src/llmcodec_quant_gptq.c      # decoder GPTQ int2/4/8 (autogptq/gptqmodel) per l'engine MLX
    src/llmcodec_quant_nf4.c       # decoder bitsandbytes NF4 (+double-quant) per l'engine MLX
    # ENCODE/DEQUANT ggml (CONDIVISO: actquant in inferenza + quant PUL). Nel core STATICO così
    # i consumatori (gmbllm) li LINKANO invece di ricompilarli (niente fork stale).
    src/llmcodec_ggml_quants.c src/llmcodec_ggml_iq_grids.c src/llmcodec_make_qp.c
)

# Suffisso dei plugin per piattaforma: `.dylib` su macOS, `.so` su Linux (glibc/ld).
# Il loader (LLMCodec/src/llmcodec.c) prova {.dylib, .so, ""}: entrambi funzionano, ma il nome
# nativo evita sorprese con strumenti che filtrano per estensione (bundle, sha, ldd).
if(APPLE)
    set(_codec_suffix ".dylib")
else()
    set(_codec_suffix ".so")
endif()

# PLUGIN CODEC (dylib): le IMPLEMENTAZIONI vivono QUI, mai nel core.
add_library(llmcodec_ggml SHARED
    src/llmcodec_ggml_quants.c src/llmcodec_ggml_iq_grids.c src/llmcodec_make_qp.c
    plugins/ggml/codec_ggml.c)
add_library(llmcodec_mlx_affine SHARED src/llmcodec_moe_quant.c plugins/mlx_affine/codec_mlx_affine.c)
add_library(llmcodec_fp8 SHARED src/llmcodec_fp8_fp4.c plugins/fp8/codec_fp8.c)
add_library(llmcodec_plain SHARED plugins/plain/codec_plain.c)
add_library(llmcodec_mxfp SHARED src/llmcodec_fp8_fp4.c plugins/mxfp/codec_mxfp.c)
add_library(llmcodec_bitnet SHARED plugins/bitnet/codec_bitnet.c)
add_library(llmcodec_turboquant SHARED plugins/turboquant/codec_turboquant.c)
add_library(llmcodec_fp4 SHARED src/llmcodec_fp8_fp4.c plugins/fp4/codec_fp4.c)
add_library(llmcodec_nvfp SHARED src/llmcodec_fp8_fp4.c plugins/nvfp/codec_nvfp.c)
add_library(llmcodec_bnb SHARED src/llmcodec_fp8_fp4.c plugins/bnb/codec_bnb.c)
add_library(llmcodec_hqq SHARED plugins/hqq/codec_hqq.c)
add_library(llmcodec_aqlm SHARED plugins/aqlm/codec_aqlm.c)
add_library(llmcodec_vptq SHARED plugins/vptq/codec_vptq.c)
add_library(llmcodec_spqr SHARED plugins/spqr/codec_spqr.c)
add_library(llmcodec_higgs SHARED plugins/higgs/codec_higgs.c)
add_library(llmcodec_awq SHARED plugins/awq/codec_awq.c)
add_library(llmcodec_gptq SHARED plugins/gptq/codec_gptq.c)
foreach(pl llmcodec_ggml llmcodec_mlx_affine llmcodec_fp8 llmcodec_plain llmcodec_mxfp llmcodec_gptq llmcodec_awq llmcodec_bitnet llmcodec_turboquant llmcodec_fp4 llmcodec_nvfp llmcodec_bnb llmcodec_hqq llmcodec_aqlm llmcodec_vptq llmcodec_spqr llmcodec_higgs)
    target_include_directories(${pl} PRIVATE include src)
    target_link_libraries(${pl} PRIVATE m)
    set_target_properties(${pl} PROPERTIES
        PREFIX "lib" OUTPUT_NAME "${pl}" SUFFIX "${_codec_suffix}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        BUILD_RPATH "${CMAKE_BINARY_DIR}" INSTALL_RPATH "@loader_path")
endforeach()
target_include_directories(llmcodec PUBLIC include)
# include/ per i consumatori; src/ per l'header interno delle grid
target_include_directories(llmcodec PRIVATE src)
target_link_libraries(llmcodec PUBLIC ${CMAKE_DL_LIBS} m)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    option(LLMCODEC_TESTS "Build codec tests" ON)
else()
    option(LLMCODEC_TESTS "Build codec tests" OFF)
endif()

if(LLMCODEC_TESTS)
    # NOTA: i self-check del CONVERTER (ggml_quants, conv_gguf_mlx, conv_sharded, rolemap,
    # recipe, expert3d, dsv41_source, gguf_map, engram, plugin, a1_fp8, quantcfg) sono stati
    # spostati in GMBasic/gmbllm/tests/ — il converter appartiene al confine PUL.

    add_executable(llmcodec_structs_selfcheck tests/test_structs.c)
    target_link_libraries(llmcodec_structs_selfcheck PRIVATE llmcodec)
    target_compile_definitions(llmcodec_structs_selfcheck PRIVATE LLMCODEC_ABI_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ABI")
    add_test(NAME llmcodec_structs_selfcheck COMMAND llmcodec_structs_selfcheck)

    add_executable(llmcodec_fp8_fp4_selfcheck tests/test_fp8_fp4.c)
    target_link_libraries(llmcodec_fp8_fp4_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_fp8_fp4_selfcheck COMMAND llmcodec_fp8_fp4_selfcheck)

    add_executable(llmcodec_qwen4_rolemap_selfcheck tests/test_qwen4_rolemap.c)
    target_link_libraries(llmcodec_qwen4_rolemap_selfcheck PRIVATE llmcodec)
    target_compile_definitions(llmcodec_qwen4_rolemap_selfcheck PRIVATE
        LLMCODEC_Q4_ABI="${CMAKE_CURRENT_SOURCE_DIR}/../ABI/qwen4_exp.gmbarch")
    add_test(NAME llmcodec_qwen4_rolemap_selfcheck COMMAND llmcodec_qwen4_rolemap_selfcheck)

    add_executable(llmcodec_abi_all_selfcheck tests/test_abi_all.c)
    target_link_libraries(llmcodec_abi_all_selfcheck PRIVATE llmcodec)
    target_compile_definitions(llmcodec_abi_all_selfcheck PRIVATE LLMCODEC_ABI_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ABI")
    add_test(NAME llmcodec_abi_all_selfcheck COMMAND llmcodec_abi_all_selfcheck)

    add_executable(llmcodec_mlx_mxfp4_selfcheck tests/test_mlx_mxfp4.c)
    target_link_libraries(llmcodec_mlx_mxfp4_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_mlx_mxfp4_selfcheck COMMAND llmcodec_mlx_mxfp4_selfcheck llmcodec_quantcfg_selfcheck llmcodec_formats_p0_selfcheck llmcodec_real_fixtures_selfcheck)

    add_executable(llmcodec_formats_p0_selfcheck tests/test_formats_p0.c)
    target_include_directories(llmcodec_formats_p0_selfcheck PRIVATE plugins/higgs)
    target_link_libraries(llmcodec_formats_p0_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_formats_p0_selfcheck COMMAND llmcodec_formats_p0_selfcheck llmcodec_real_fixtures_selfcheck)

    add_executable(llmcodec_real_fixtures_selfcheck tests/test_real_fixtures.c)
    target_link_libraries(llmcodec_real_fixtures_selfcheck PRIVATE llmcodec)
    target_compile_definitions(llmcodec_real_fixtures_selfcheck PRIVATE LLMCODEC_FIX_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tools/Fixtures/codec")
    add_test(NAME llmcodec_real_fixtures_selfcheck COMMAND llmcodec_real_fixtures_selfcheck)

    # TASSONOMIA D'ERRORE GGUF: ogni corruzione porta il SUO codice (asset assorbito da lane_gguf).
    add_executable(llmcodec_gguf_status_selfcheck tests/test_gguf_status.c)
    target_link_libraries(llmcodec_gguf_status_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_gguf_status_selfcheck COMMAND llmcodec_gguf_status_selfcheck)

    # CONFORMANCE dequant: llmcodec_ggml_dequant_row vs un secondo implementatore indipendente dei
    # bit-layout GGUF (port verbatim di antirez/gguf-tools gguflib.c, MIT). Una divergenza = bug.
    add_executable(llmcodec_dequant_conformance_selfcheck tests/test_dequant_conformance.c)
    target_link_libraries(llmcodec_dequant_conformance_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_dequant_conformance_selfcheck COMMAND llmcodec_dequant_conformance_selfcheck)
endif()

# I test esercitano le funzioni dei codec: linkano i PLUGIN (le impl non sono piu' nel core).
set(LLMCODEC_CODEC_PLUGIN_LINK llmcodec_ggml llmcodec_mlx_affine llmcodec_fp8 llmcodec_plain llmcodec_mxfp llmcodec_gptq llmcodec_awq llmcodec_bitnet llmcodec_turboquant llmcodec_fp4 llmcodec_nvfp llmcodec_bnb llmcodec_hqq llmcodec_aqlm llmcodec_vptq llmcodec_spqr llmcodec_higgs)
foreach(t llmcodec_structs_selfcheck
        llmcodec_fp8_fp4_selfcheck llmcodec_abi_all_selfcheck
        llmcodec_mlx_mxfp4_selfcheck llmcodec_formats_p0_selfcheck llmcodec_real_fixtures_selfcheck
        llmcodec_gguf_status_selfcheck)
    if(TARGET ${t})
        target_link_libraries(${t} PRIVATE ${LLMCODEC_CODEC_PLUGIN_LINK})
    endif()
endforeach()

# CLI di conversione MLX/safetensors -> GGUF: SPOSTATA in GMBasic/gmbllm (PUL). Il converter
# produce/trasforma pesi e appartiene al confine PUL — vedi
# Decisions/ADR-Confine-Licenza-Engine-Apache-Training-Quant-PUL.md.
