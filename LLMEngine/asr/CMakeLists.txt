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
