cmake_minimum_required(VERSION 3.16)
# LLMEngine/Options — registro di opzioni CONDIVISO: parametri del motore con default compilati,
# impostabili via API (BASIC/CLI), MAI via variabile d'ambiente. Core a basso livello: mlx, moe,
# llama e i plugin vi dipendono (nessun ciclo).
project(llmengine_opts C)

add_library(llmengine_opts STATIC llm_engine_opts.c)
target_include_directories(llmengine_opts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(llmengine_opts PROPERTIES POSITION_INDEPENDENT_CODE ON)
