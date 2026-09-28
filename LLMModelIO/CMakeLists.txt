# Progetto C autonomo: MODEL IO — lettura/scrittura/quantizzazione/conversione dei modelli
# (superficie `LLM MODEL SAVE/QUANTIZE/CONVERT/PLAN/POLICY`), calibrazione imatrix e forward
# di calibrazione. Dipende da: llmcodec (formati+quant), llmtoken, llmengine_plugin,
# llmdesc (e llmengine_moe_quant_alloc via llmengine_moe). NON dipende dal runtime BASIC.
cmake_minimum_required(VERSION 3.16)
project(llmmodelio C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Sottoprogetti: usa i target gia' presenti nel tree, altrimenti li aggiunge (build autonomo).
if(NOT TARGET llmcodec)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../LLMCodec llmcodec_build)
endif()
if(NOT TARGET llmtoken)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../LLMToken llmtoken_build)
endif()
if(NOT TARGET llmengine_plugin)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../LLMEngine/plugin llmengine_plugin_build)
endif()
if(NOT TARGET llmdesc)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../LLMDesc llmdesc_build)
endif()

add_library(llmmodelio STATIC
    src/llmmodelio_model.c    # MODEL COMPILER: save/quantize/convert/plan/policy
    src/llmmodelio_imatrix.c      # imatrix (calibrazione) read/write
    src/llmmodelio_infer.c    # forward di calibrazione (conteggio routing per LLM PRUNE)
    ${CMAKE_CURRENT_SOURCE_DIR}/../LLMEngine/moe/src/llmengine_moe_quant_alloc.c)   # llmengine_moe_quant_alloc (allocatore) — self-contained
target_include_directories(llmmodelio PUBLIC include
    ${CMAKE_CURRENT_SOURCE_DIR}/../LLMEngine/moe/include)
target_link_libraries(llmmodelio PUBLIC llmcodec llmtoken llmengine_plugin llmdesc ${CMAKE_DL_LIBS} m)
