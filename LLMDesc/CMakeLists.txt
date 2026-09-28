# Progetto C autonomo: descrizione d'architettura `.llmarch` (lettura, risoluzione per un
# modello, applicazione delle regole ruoli/quant/strutture/codecs). Vedi llmdesc.h.
# Nessuna dipendenza dal runtime: chi ne ha bisogno (llm_runtime, llmmodelio,
# llmengine_mlx) lo linka.
cmake_minimum_required(VERSION 3.16)
project(llmdesc C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(llmdesc STATIC llmdesc.c)
target_include_directories(llmdesc PUBLIC .)
