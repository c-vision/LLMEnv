# LLMEnv — superbuild (SCHELETRO). Un progetto CMake per layer, grafo aciclico (vedi README).
# Stato: la scomposizione e' gia' implementata DENTRO GMBasic (nomi gmb_*); qui i layer arriveranno
# dopo rename + esternalizzazione. Fino ad allora NESSUN add_subdirectory (le cartelle sono vuote).
cmake_minimum_required(VERSION 3.16)
project(LLMEnv C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "LLMEnv: scheletro — i layer (LLMToken/LLMCodec/LLMDesc/LLMMetal/LLMEngine/LLMModelIO)")
message(STATUS "verranno agganciati qui dopo il rename gmb_*->llm* e l'esternalizzazione da GMBasic.")

# TODO(estrazione): un add_subdirectory per layer, in ordine di dipendenza:
#   add_subdirectory(LLMToken)   # 1 (cJSON, PCRE2, sds)
#   add_subdirectory(LLMCodec)   # 2 (solo formato)
#   add_subdirectory(LLMDesc)    # 3
#   add_subdirectory(LLMMetal)   # 4 (Metal)
#   add_subdirectory(LLMEngine)  # 5 (ABI) + 5a/b/c (mlx/llama/moe)
#   add_subdirectory(LLMModelIO) # 6 (1,2,3)
