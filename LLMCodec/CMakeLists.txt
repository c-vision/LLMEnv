# Progetto C AUTONOMO: codec GGML (blocco<->f32) + (futuro) conversioni GGUF<->MLX.
# Separato dal runtime LLMEnv: nessuna dipendenza dal resto (solo libm). Chi ne ha bisogno
# (llm_runtime, il converter) lo aggancia come subdirectory e ne linka la libreria.
cmake_minimum_required(VERSION 3.16)
project(llmcodec C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CORE: contenitori + descrizione + registry + converter + bridge. NESSUNA implementazione codec.
add_library(llmcodec STATIC
    src/llmcodec_gguf_writer.c     # contenitore GGUF (I/O) + tabella byte-per-tipo
    src/llmcodec_safetensors.c     # contenitore safetensors (I/O)
    src/llmcodec_gguf_mlx.c        # converter (orchestrazione via registry)
    src/llmcodec_gguf_parse.c      # PARSER GGUF canonico buffer-based (reader unico)
    src/llmcodec_gguf_meta.c       # METADATA GGUF (hyperparametri + tokenizer) per il converter
    src/llmcodec_rolemap.c         # descrizione .llmarch
    src/llmcodec.c           # registry PLUGIN (dlopen + cache)
    src/llmcodec_quant_awq.c       # decoder AWQ int4 GEMM (autoawq gemm layout) per l'engine MLX
    src/llmcodec_quant_gptq.c      # decoder GPTQ int2/4/8 (autogptq/gptqmodel) per l'engine MLX
    src/llmcodec_quant_nf4.c       # decoder bitsandbytes NF4 (+double-quant) per l'engine MLX
)

# PLUGIN CODEC (dylib): le IMPLEMENTAZIONI vivono QUI, mai nel core.
add_library(llmcodec_ggml SHARED
    src/llmcodec_ggml_quants.c src/llmcodec_ggml_iq_grids.c src/llmcodec_iq2_xxs.c
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
        PREFIX "lib" OUTPUT_NAME "${pl}" SUFFIX ".dylib"
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
    add_executable(llmcodec_ggml_quants_selfcheck tests/test_ggml_quants.c)
    target_link_libraries(llmcodec_ggml_quants_selfcheck PRIVATE llmcodec llmcodec_ggml)   # encoder IQ vivono nel plugin ggml
    add_test(NAME llmcodec_ggml_quants_selfcheck COMMAND llmcodec_ggml_quants_selfcheck)

    add_executable(llmcodec_conv_gguf_mlx_selfcheck tests/test_conv_gguf_mlx.c)
    target_link_libraries(llmcodec_conv_gguf_mlx_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_conv_gguf_mlx_selfcheck COMMAND llmcodec_conv_gguf_mlx_selfcheck)

    add_executable(llmcodec_conv_sharded_selfcheck tests/test_conv_sharded.c)
    target_link_libraries(llmcodec_conv_sharded_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_conv_sharded_selfcheck COMMAND llmcodec_conv_sharded_selfcheck)

    add_executable(llmcodec_rolemap_selfcheck tests/test_rolemap.c)
    target_link_libraries(llmcodec_rolemap_selfcheck PRIVATE llmcodec)
    target_compile_definitions(llmcodec_rolemap_selfcheck PRIVATE LLMCODEC_ABI_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ABI")
    add_test(NAME llmcodec_rolemap_selfcheck COMMAND llmcodec_rolemap_selfcheck)

    add_executable(llmcodec_recipe_selfcheck tests/test_recipe.c)
    target_link_libraries(llmcodec_recipe_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_recipe_selfcheck COMMAND llmcodec_recipe_selfcheck)

    add_executable(llmcodec_expert3d_selfcheck tests/test_expert3d.c)
    target_link_libraries(llmcodec_expert3d_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_expert3d_selfcheck COMMAND llmcodec_expert3d_selfcheck)

    add_executable(llmcodec_structs_selfcheck tests/test_structs.c)
    target_link_libraries(llmcodec_structs_selfcheck PRIVATE llmcodec)
    target_compile_definitions(llmcodec_structs_selfcheck PRIVATE LLMCODEC_ABI_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ABI")
    add_test(NAME llmcodec_structs_selfcheck COMMAND llmcodec_structs_selfcheck)

    add_executable(llmcodec_fp8_fp4_selfcheck tests/test_fp8_fp4.c)
    target_link_libraries(llmcodec_fp8_fp4_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_fp8_fp4_selfcheck COMMAND llmcodec_fp8_fp4_selfcheck)

    add_executable(llmcodec_dsv41_source_selfcheck tests/test_dsv41_source.c)
    target_link_libraries(llmcodec_dsv41_source_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_dsv41_source_selfcheck COMMAND llmcodec_dsv41_source_selfcheck)

    add_executable(llmcodec_gguf_map_selfcheck tests/test_gguf_map.c)
    target_link_libraries(llmcodec_gguf_map_selfcheck PRIVATE llmcodec llmcodec_ggml llmcodec_fp8)
    add_test(NAME llmcodec_gguf_map_selfcheck COMMAND llmcodec_gguf_map_selfcheck)

    add_executable(llmcodec_abi_all_selfcheck tests/test_abi_all.c)
    target_link_libraries(llmcodec_abi_all_selfcheck PRIVATE llmcodec)
    target_compile_definitions(llmcodec_abi_all_selfcheck PRIVATE LLMCODEC_ABI_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ABI")
    add_test(NAME llmcodec_abi_all_selfcheck COMMAND llmcodec_abi_all_selfcheck)

    add_executable(llmcodec_engram_selfcheck tests/test_engram.c)
    target_link_libraries(llmcodec_engram_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_engram_selfcheck COMMAND llmcodec_engram_selfcheck)

    add_library(llmcodec_test_id MODULE tests/plugin_test_id.c)
    target_include_directories(llmcodec_test_id PRIVATE include)
    set_target_properties(llmcodec_test_id PROPERTIES PREFIX "lib" OUTPUT_NAME "llmcodec_test_id")

    add_executable(llmcodec_plugin_selfcheck tests/test_plugin.c)
    target_link_libraries(llmcodec_plugin_selfcheck PRIVATE llmcodec)
    add_dependencies(llmcodec_plugin_selfcheck llmcodec_test_id)
    add_test(NAME llmcodec_plugin_selfcheck COMMAND llmcodec_plugin_selfcheck)

    add_executable(llmcodec_a1_fp8_selfcheck tests/test_a1_fp8.c)
    target_link_libraries(llmcodec_a1_fp8_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_a1_fp8_selfcheck COMMAND llmcodec_a1_fp8_selfcheck)

    add_executable(llmcodec_mlx_mxfp4_selfcheck tests/test_mlx_mxfp4.c)
    target_link_libraries(llmcodec_mlx_mxfp4_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_mlx_mxfp4_selfcheck COMMAND llmcodec_mlx_mxfp4_selfcheck llmcodec_quantcfg_selfcheck llmcodec_formats_p0_selfcheck llmcodec_real_fixtures_selfcheck)

    add_executable(llmcodec_quantcfg_selfcheck tests/test_quantcfg.c)
    target_link_libraries(llmcodec_quantcfg_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_quantcfg_selfcheck COMMAND llmcodec_quantcfg_selfcheck llmcodec_formats_p0_selfcheck llmcodec_real_fixtures_selfcheck)

    add_executable(llmcodec_formats_p0_selfcheck tests/test_formats_p0.c)
    target_include_directories(llmcodec_formats_p0_selfcheck PRIVATE plugins/higgs)
    target_link_libraries(llmcodec_formats_p0_selfcheck PRIVATE llmcodec)
    add_test(NAME llmcodec_formats_p0_selfcheck COMMAND llmcodec_formats_p0_selfcheck llmcodec_real_fixtures_selfcheck)

    add_executable(llmcodec_real_fixtures_selfcheck tests/test_real_fixtures.c)
    target_link_libraries(llmcodec_real_fixtures_selfcheck PRIVATE llmcodec)
    target_compile_definitions(llmcodec_real_fixtures_selfcheck PRIVATE LLMCODEC_FIX_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../samples/Fixtures/codec")
    add_test(NAME llmcodec_real_fixtures_selfcheck COMMAND llmcodec_real_fixtures_selfcheck)
endif()

# I test esercitano le funzioni dei codec: linkano i PLUGIN (le impl non sono piu' nel core).
set(LLMCODEC_CODEC_PLUGIN_LINK llmcodec_ggml llmcodec_mlx_affine llmcodec_fp8 llmcodec_plain llmcodec_mxfp llmcodec_gptq llmcodec_awq llmcodec_bitnet llmcodec_turboquant llmcodec_fp4 llmcodec_nvfp llmcodec_bnb llmcodec_hqq llmcodec_aqlm llmcodec_vptq llmcodec_spqr llmcodec_higgs)
foreach(t llmcodec_ggml_quants_selfcheck llmcodec_conv_gguf_mlx_selfcheck llmcodec_conv_sharded_selfcheck
        llmcodec_rolemap_selfcheck llmcodec_recipe_selfcheck llmcodec_expert3d_selfcheck llmcodec_structs_selfcheck
        llmcodec_fp8_fp4_selfcheck llmcodec_dsv41_source_selfcheck llmcodec_abi_all_selfcheck llmcodec_engram_selfcheck
        llmcodec_plugin_selfcheck llmcodec_a1_fp8_selfcheck llmcodec_mlx_mxfp4_selfcheck llmcodec_quantcfg_selfcheck llmcodec_formats_p0_selfcheck llmcodec_real_fixtures_selfcheck)
    if(TARGET ${t})
        target_link_libraries(${t} PRIVATE ${LLMCODEC_CODEC_PLUGIN_LINK})
    endif()
endforeach()

# CLI di conversione MLX/safetensors -> GGUF (rename canonico + stacking esperti via .llmarch).
# La conversione richiede ANCHE il tokenizer per i metadata (tokenizer.ggml.*): il target
# LIBRERIA resta senza dipendenza dal tokenizer, solo il tool lo linka. Opzionale: di default
# OFF cosi' LLMCodec builda STANDALONE (solo libm + i suoi plugin dlopen).
option(LLMCODEC_BUILD_CONVERTER "Build gguf_convert CLI (richiede LLMToken)" OFF)
if(LLMCODEC_BUILD_CONVERTER)
    if(NOT TARGET llmtoken)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../LLMToken llmtoken_build)
    endif()
    add_executable(gguf_convert tools/gguf_convert.c)
    target_link_libraries(gguf_convert PRIVATE llmcodec ${LLMCODEC_CODEC_PLUGIN_LINK} llmtoken)
endif()
