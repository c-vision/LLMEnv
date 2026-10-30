# Progetto C autonomo: descrizione d'architettura `.llmarch` (lettura, risoluzione per un
# modello, applicazione delle regole ruoli/quant/strutture/codecs). Vedi llmdesc.h.
# Nessuna dipendenza dal runtime: chi ne ha bisogno (llm_runtime, llmmodelio,
# llmengine_mlx) lo linka.
cmake_minimum_required(VERSION 3.16)
project(llmdesc C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(llmdesc STATIC llmdesc.c llmdesc_gen.c)
target_include_directories(llmdesc PUBLIC .)

# Test: per ogni modello locale il descrittore risolto dalla cartella ABI/ ha la geometria di config.json.
# La radice dei modelli e' $HOME/ai/Models; senza modelli il test risulta SKIPPED (exit 77), non verde.
enable_testing()
add_executable(llmdesc_geometry_selfcheck llmdesc_geometry_selfcheck.c)
target_link_libraries(llmdesc_geometry_selfcheck PRIVATE llmdesc)
add_test(NAME llmdesc_geometry_selfcheck
         COMMAND llmdesc_geometry_selfcheck "$ENV{HOME}/ai/Models" "${CMAKE_CURRENT_SOURCE_DIR}/../ABI")
set_tests_properties(llmdesc_geometry_selfcheck PROPERTIES SKIP_RETURN_CODE 77)

# Test: validita', localizzazione e politica della cartella ABI (autonomo, nessun modello).
add_executable(llmdesc_abi_selfcheck llmdesc_abi_selfcheck.c)
target_link_libraries(llmdesc_abi_selfcheck PRIVATE llmdesc)
add_test(NAME llmdesc_abi_selfcheck COMMAND llmdesc_abi_selfcheck "${CMAKE_CURRENT_SOURCE_DIR}/../ABI")

# Strumento: crea i `.gmbarch` mancanti da cartelle modello HF (llmdesc_ensure_for_dir).
add_executable(llmdesc_gen tools/llmdesc_gen.c)
target_link_libraries(llmdesc_gen PRIVATE llmdesc ${CMAKE_DL_LIBS})
