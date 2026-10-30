cmake_minimum_required(VERSION 3.16)
project(llm_asr C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Engine ASR Qwen3-ASR (speech-to-text) — INFERENCE-ONLY, self-contained (libc/libm/pthread/NEON).
# Port di antirez/qwen-asr (MIT). Vive nel lato Apache (LLMEnv) perche' e' inferenza: la stessa
# libreria e' consumata da GMBasic (runtime + superficie BASIC `ASR.RECOGNIZE`) e da FastLLM
# (`fastllm asr`). Nessuna copia su entrambi i lati.

add_library(llm_asr STATIC
    src/gmb_asr_audio.c
    src/gmb_asr_kernels.c
    src/gmb_asr_encoder.c
    src/gmb_asr_decoder.c
    src/gmb_asr_model.c
    src/gmb_asr_tok.c
    src/gmb_asr_transcribe.c
    src/gmb_asr_pipeline.c)

target_include_directories(llm_asr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llm_asr PUBLIC m)

find_package(Threads REQUIRED)
target_link_libraries(llm_asr PUBLIC Threads::Threads)

# Parakeet-TDT (NVIDIA FastConformer + transducer con durate) da GGUF: `llm_parakeet.h`. Legge il GGUF con il parser
# canonico di LLMCodec; matrici con Accelerate su macOS.
if(NOT TARGET llmcodec)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../LLMCodec ${CMAKE_BINARY_DIR}/llmcodec_for_asr)
endif()
target_sources(llm_asr PRIVATE src/llm_parakeet.c)
target_link_libraries(llm_asr PUBLIC llmcodec)
if(APPLE)
    target_link_libraries(llm_asr PUBLIC "-framework Accelerate")
endif()

# Test: trascrive un WAV noto con il GGUF locale (SKIPPED, exit 77, se mancano).
add_executable(llm_parakeet_test tests/llm_parakeet_test.c)
target_link_libraries(llm_parakeet_test PRIVATE llm_asr)
# I percorsi WAV/testo atteso si leggono dall'ambiente AL MOMENTO DELLA CONFIGURAZIONE (LLM_PARAKEET_WAV, LLM_PARAKEET_TXT).
enable_testing()
add_test(NAME llm_parakeet_test
         COMMAND llm_parakeet_test "$ENV{HOME}/ai/Models/gguf/parakeet-tdt-0.6b-v3-q8_0.gguf" "$ENV{LLM_PARAKEET_WAV}" "$ENV{LLM_PARAKEET_TXT}")
set_tests_properties(llm_parakeet_test PROPERTIES SKIP_RETURN_CODE 77)
