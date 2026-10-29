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

# Test: per ogni modello locale il descrittore risolto dalla cartella ABI/ ha la geometria di config.json.
# La radice dei modelli e' $HOME/ai/Models; senza modelli il test risulta SKIPPED (exit 77), non verde.
enable_testing()
add_executable(llmdesc_geometry_selfcheck llmdesc_geometry_selfcheck.c)
target_link_libraries(llmdesc_geometry_selfcheck PRIVATE llmdesc)
add_test(NAME llmdesc_geometry_selfcheck
         COMMAND llmdesc_geometry_selfcheck "$ENV{HOME}/ai/Models" "${CMAKE_CURRENT_SOURCE_DIR}/../ABI")
set_tests_properties(llmdesc_geometry_selfcheck PROPERTIES SKIP_RETURN_CODE 77)
