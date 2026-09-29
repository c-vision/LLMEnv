# LLMEnv — superbuild. Un progetto CMake per layer, grafo aciclico (vedi README).
# Ogni layer compila anche STANDALONE (`cmake -S LLMxxx -B build`); qui li si aggancia in ordine.
cmake_minimum_required(VERSION 3.20)
project(LLMEnv C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Layer sempre costruibili (nessuna dipendenza da framework LLM pesanti).
add_subdirectory(LLMToken)    # 1
add_subdirectory(LLMCodec)    # 2
add_subdirectory(LLMJinja)    # 2b (motore Jinja chat-template, port di GMBasic/gmb_jinja)
add_subdirectory(LLMDesc)     # 3
add_subdirectory(LLMMetal)    # 4
add_subdirectory(LLMModelIO)  # 6 (usa 1,2,3 + il plugin engine)

# LLMEngine richiede MLX (mlx-c) e/o llama.cpp via FetchContent: pesante e opzionale.
# Attivare con -DLLMENV_BUILD_ENGINE=ON (sub-layer: plugin, moe, llama, mlx).
option(LLMENV_BUILD_ENGINE "Build LLMEngine (mlx/llama/moe/plugin)" OFF)
if(LLMENV_BUILD_ENGINE)
  # `LLMEngine/llama` contiene sorgenti Objective-C (`llm_vision_apple.m`, Vision/OCR): senza
  # `enable_language(OBJC)` la generazione fallisce con `CMAKE_OBJC_COMPILE_OBJECT` non definito.
  if(APPLE)
    enable_language(OBJC)
  endif()
  # `LLMModelIO` (aggiunto sopra) porta già `llmengine_plugin` con la sua guardia: senza questa
  # guardia il superbuild chiamerebbe `project()`/`add_library()` una SECONDA volta e CMake
  # fallirebbe con "target already exists". Lo standalone `cmake -S LLMEngine/plugin` resta valido.
  if(NOT TARGET llmengine_plugin)
    add_subdirectory(LLMEngine/plugin)
  endif()
  add_subdirectory(LLMEngine/moe)
  add_subdirectory(LLMEngine/llama)
  add_subdirectory(LLMEngine/mlx)
endif()
