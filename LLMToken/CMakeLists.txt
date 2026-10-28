cmake_minimum_required(VERSION 3.16)
project(llmtoken C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Vendoring via FetchContent -- stesso pattern "no submodule, nessun checkout
# manuale, pin a un tag di release reale (mai master)" già usato per
# llama.cpp (Fase 0) e mlx-c (Fase 1).
include(FetchContent)
include(CTest)

# PCRE2: motore regex con supporto Unicode property escapes (\p{L}, \p{N})
# -- necessario per replicare esattamente il pre-tokenizer regex di Qwen3
# (`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
# letto da tokenizer.json reale, stesso schema byte-level BPE di GPT-2/GPT-4).
# Senza supporto Unicode reale, i confini dei token su testo non-ASCII
# (accenti, CJK, emoji) divergerebbero silenziosamente da HuggingFace/mlx-lm.
set(PCRE2_SUPPORT_UNICODE ON CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_STATIC_PIC ON CACHE BOOL "" FORCE)  # embeddiamo la lib statica in llmtoken (SHARED)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    pcre2
    GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
    GIT_TAG pcre2-10.47  # PIN ESPLICITO -- verificato reale via GitHub tags API, 2026-08-17
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pcre2)

# cJSON: parser JSON C puro, header+sorgente unici (DaveGamble/cJSON), usato
# per tokenizer.json (~11MB, vocab+merges reali del modello). Sostituisce
# nlohmann/json (C++ header-only) nella riscrittura in C puro del 2026-08-24
# -- vedi Fase0b-LlamaCpp-Wrapper-C-Rewrite-Plan.md per il piano complessivo
# (questo file era il primo target, essendo il più piccolo e autonomo).
set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
set(ENABLE_CJSON_UTILS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_AND_STATIC_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG v1.7.19  # PIN ESPLICITO -- verificato reale via GitHub tags API, 2026-08-24
    GIT_SHALLOW TRUE
)
# v1.7.19 dichiara cmake_minimum_required troppo vecchio per CMake attuale
# (rimossa la compatibilità <3.5 nelle versioni recenti di CMake).
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(cjson)

# sds: Simple Dynamic Strings (antirez/Redis) -- stringhe C binary-safe con
# capacità pre-allocata (append ammortizzato O(1)), usate al posto di
# std::string. Nessun CMakeLists upstream (solo Makefile): vendorizzata come
# sorgente singola aggiunta direttamente al target sotto.
FetchContent_Declare(
    sds
    GIT_REPOSITORY https://github.com/antirez/sds.git
    GIT_TAG 2.0.0  # PIN ESPLICITO -- verificato reale via GitHub tags API, 2026-08-24
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(sds)

add_library(llmtoken SHARED llmtoken.c llmtoken_sp.c llmtoken_wordpiece.c "${sds_SOURCE_DIR}/sds.c")
target_include_directories(llmtoken PRIVATE "${sds_SOURCE_DIR}" "${cjson_SOURCE_DIR}")
target_link_libraries(llmtoken PRIVATE pcre2-8 cjson)
target_include_directories(llmtoken PUBLIC .)
set_target_properties(llmtoken PROPERTIES OUTPUT_NAME "llmtoken" POSITION_INDEPENDENT_CODE ON)

# Test del loader TIKTOKEN (Kimi) contro id golden del riferimento `tiktoken`.
add_executable(llmtoken_tiktoken_test llmtoken_tiktoken_test.c)
target_link_libraries(llmtoken_tiktoken_test PRIVATE llmtoken)

# Conformance del loader SentencePiece nativo (llmtoken_sp.c) contro sentencepiece/golden id.
add_executable(llmtoken_sp_selfcheck llmtoken_sp_selfcheck.c)
target_link_libraries(llmtoken_sp_selfcheck PRIVATE llmtoken)

# Conformance del tokenizer BERT WordPiece C contro il golden HF (tokenizer_golden.json),
# gia' in questo progetto: TUTTI i tokenizer vivono qui (BPE byte-level, metaspace/SPM,
# tiktoken, WordPiece) -- nessun tokenizer nel runtime.
add_executable(llmtoken_wordpiece_selfcheck llmtoken_wordpiece_selfcheck.c)
target_link_libraries(llmtoken_wordpiece_selfcheck PRIVATE llmtoken)
if(BUILD_TESTING)
    add_test(NAME llmtoken_wordpiece_selfcheck COMMAND llmtoken_wordpiece_selfcheck
        ${CMAKE_CURRENT_SOURCE_DIR}/../tools/Fixtures/tokenizer)
endif()
