# Progetto C autonomo: ABI + registry dei PLUGIN nativi (CODEC di formato + PRIMITIVE
# d'architettura), caricati via dlopen. Vedi llmengine_plugin.h. Nessuna dipendenza dal runtime:
# chi ne ha bisogno (llm_runtime, llmmodelio, llmengine_mlx, i plugin d'esempio) lo linka.
cmake_minimum_required(VERSION 3.16)
project(llmengine_plugin C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(llmengine_plugin STATIC llmengine_plugin.c)
target_include_directories(llmengine_plugin PUBLIC .)
target_link_libraries(llmengine_plugin PUBLIC ${CMAKE_DL_LIBS})
