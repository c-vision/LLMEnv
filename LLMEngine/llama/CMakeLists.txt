cmake_minimum_required(VERSION 3.14)
project(llmengine_llama C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendoring llama.cpp via FetchContent -- stesso approccio "no submodule,
# nessun checkout manuale" già usato per gli altri riferimenti di terze
# parti del progetto. Pin a un tag di release, non a `master`, per build
# riproducibili -- aggiornare deliberatamente, mai seguire HEAD.
include(FetchContent)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
# Niente OpenSSL: serve solo al download HTTPS di modelli da parte di llama.cpp, che qui non si usa (FastLLM scarica con curl).
# Con OpenSSL attivo la libreria si collega a un percorso Homebrew assoluto (openssl@4) e non si carica su un Mac che non lo ha.
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
# FASE 5B: mtmd (multimodal tool-kit) -- OCR/visione/audio embedded. Vendored in
# llama.cpp's tools/mtmd/, built as its own static lib. See
# Native-LLM-Provider-Plan.md FASE 5 Parte B. paddleocr.cpp/deepseekocr.cpp are
# compiled into it, so PaddleOCR-VL GGUF+mmproj works out of the box.
# LLAMA_BUILD_COMMON is required: mtmd-cli (and our wrapper) format the prompt
# with the model's Jinja chat template via common_chat_templates -- llama_chat_apply_template
# only supports a pre-defined list, not arbitrary Jinja (PaddleOCR needs Jinja to
# emit <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>). LLAMA_BUILD_TOOLS
# brings mtmd in via tools/ (the standalone hook fires only when TOOLS is OFF).
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)

# Backend: su macOS Metal è on-by-default nel build system di llama.cpp
# quando GGML_METAL non è esplicitamente disattivato. Su Linux/Windows
# questa stessa build produce comunque il backend CPU; CUDA/ROCm/Vulkan/
# SYCL vanno abilitati esplicitamente per piattaforma (vedi
# Native-LLM-Provider-Plan.md, tabella "Selezione backend per piattaforma"
# -- non tutti abilitati di default per non forzare toolchain CUDA/ROCm/
# SYCL su ogni macchina di build).
#
# Opzioni esplicite per i backend non-Apple (default OFF). Attivare con:
#   -DCMAKE_CUDA_COMPILER=<path> -DLLMENGINE_LLAMA_CUDA=ON        (NVIDIA)
#   -DLLMENGINE_LLAMA_HIP=ON                                      (AMD ROCm)
#   -DLLMENGINE_LLAMA_VULKAN=ON                                   (cross-vendor)
#   -DLLMENGINE_LLAMA_SYCL=ON -DCMAKE_CXX_COMPILER=icpx          (Intel)
# Ogni flag mappa direttamente alla variabile GGML_* di llama.cpp.
option(LLMENGINE_LLAMA_CUDA "Enable CUDA backend (NVIDIA)" OFF)
option(LLMENGINE_LLAMA_HIP "Enable HIP/ROCm backend (AMD)" OFF)
option(LLMENGINE_LLAMA_VULKAN "Enable Vulkan backend (cross-vendor)" OFF)
option(LLMENGINE_LLAMA_SYCL "Enable SYCL backend (Intel)" OFF)

if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
else()
    if(LLMENGINE_LLAMA_CUDA)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
    endif()
    if(LLMENGINE_LLAMA_HIP)
        set(GGML_HIP ON CACHE BOOL "" FORCE)
    endif()
    if(LLMENGINE_LLAMA_VULKAN)
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    endif()
    if(LLMENGINE_LLAMA_SYCL)
        set(GGML_SYCL ON CACHE BOOL "" FORCE)
    endif()
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b10448  # PIN ESPLICITO -- verificato reale (release GitHub più recente al
                     # 2026-08-16, https://github.com/ggml-org/llama.cpp/releases/latest).
                     # Aggiornare a mano dopo aver verificato compatibilità, mai seguire master.
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama_cpp)

# llmengine_llama.c/llmengine_llama_ocr.c: riscritti in C puro (2026-08-24,
# ULTIMO modulo del piano di conversione C++ -> C, vedi
# Fase0b-LlamaCpp-Wrapper-C-Rewrite-Plan.md). llama.cpp/mtmd restano C++
# upstream, invariati -- project(... C CXX) sopra abilita entrambi i
# linguaggi cosi' il linker finale puo' risolvere i simboli C++ di
# libllama/libmtmd anche se nessun file .cpp e' compilato in QUESTO target.
add_library(llm_ai_llamacpp SHARED
    llmengine_llama.c
    llmengine_llama_ocr.c
    llm_engine_llama.c   # adapter all'ABI comune llmengine
)

target_link_libraries(llm_ai_llamacpp PRIVATE llama mtmd llama-common)
target_include_directories(llm_ai_llamacpp PUBLIC . PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/llama_cpp-src
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/llama_cpp-src/common)

set_target_properties(llm_ai_llamacpp PROPERTIES
    OUTPUT_NAME "llm_ai_llamacpp"
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE CXX
)

# FASE 5 Parte B (opzionale): shim Apple Vision OCR (VNRecognizeTextRequest),
# macOS-only, zero-download. Compilato solo su Apple; il dylib finisce accanto
# a llm_ai_llamacpp così DYLD_LIBRARY_PATH lo trova.
if(APPLE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libllm_vision_apple.dylib
        COMMAND clang -fobjc-arc -framework Vision -framework Foundation
                -framework AppKit -dynamiclib
                -o ${CMAKE_CURRENT_BINARY_DIR}/libllm_vision_apple.dylib
                ${CMAKE_CURRENT_SOURCE_DIR}/apple_vision/llm_vision_apple.m
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/apple_vision/llm_vision_apple.m
        COMMENT "Building Apple Vision OCR shim (libllm_vision_apple.dylib)"
        VERBATIM)
    add_custom_target(llm_vision_apple ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libllm_vision_apple.dylib)
    add_dependencies(llm_ai_llamacpp llm_vision_apple)
endif()
