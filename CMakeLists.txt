# LLMEnv — superbuild. Due progetti C separati, librerie statiche.
# Vedi README.md. Il codice NON è ancora estratto (scheletro).
cmake_minimum_required(VERSION 3.16)
project(LLMEnv C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# I due progetti sono indipendenti fra loro salvo tramite common/.
add_subdirectory(LLMCodec)
add_subdirectory(LLMEngine)
