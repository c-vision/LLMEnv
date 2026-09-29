# LLMJinja — motore Jinja (subset chat-template LLM) in C, port di GMBasic/native/gmb_jinja.
# Layer autonomo: nessuna dipendenza. Chi ne ha bisogno (llm_host_ai, FastLLM, GMBasic) lo
# linka e gli passa template + vars. §0quater: due domini (`llmjinja.c` parser/eval/render,
# `llmjinja_builtins.c` filtri/metodi/funzioni/valori) + header interno condiviso.
cmake_minimum_required(VERSION 3.16)
project(llmjinja C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(llmjinja STATIC llmjinja.c llmjinja_builtins.c)
target_include_directories(llmjinja PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
