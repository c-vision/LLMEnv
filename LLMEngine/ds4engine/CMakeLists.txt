# ds4engine — vendorizzazione del motore ds4 (antirez) come libreria C/ObjC.
# Il motore ds4 e' COMPLETO e standalone: include modello, tokenizer, attenzione, MoE, Metal.
# TEMPORANEAMENTE SCOLLEGATO da LLMEnv/FastLLM (si veda la nota di knowledge nel vault):
# non e' un add_subdirectory del build principale. Il C tipo/api resta `ds4_engine`.
cmake_minimum_required(VERSION 3.16)
project(ds4engine C OBJC)

set(CMAKE_C_STANDARD 99)
set(CMAKE_OBJC_STANDARD 99)

set(DS4_CORE
    ds4.c
    ds4_image.c
    ds4_distributed.c
    ds4_tp.c
    ds4_ssd.c
    ds4_layer_pack.c
    ds4_engram.c
)
if(APPLE)
    list(APPEND DS4_CORE ds4_metal.m)
endif()

add_library(ds4engine STATIC ${DS4_CORE} ds4_shim.c)

add_executable(ds4_shim_test ds4_shim_test.c)
target_link_libraries(ds4_shim_test PRIVATE ds4engine)

if(APPLE)
    set_source_files_properties(ds4_metal.m PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

target_include_directories(ds4engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(ds4engine PRIVATE -O3 -ffast-math)
if(APPLE)
    find_library(FW_METAL Metal REQUIRED)
    find_library(FW_FOUNDATION Foundation REQUIRED)
    find_library(FW_MPS MetalPerformanceShaders REQUIRED)
    target_link_libraries(ds4engine PUBLIC ${FW_METAL} ${FW_FOUNDATION} ${FW_MPS} "-lm" "-pthread")
else()
    target_compile_options(ds4engine PRIVATE -march=native -Wall -Wextra -D_GNU_SOURCE -fno-finite-math-only)
    target_link_libraries(ds4engine PUBLIC -lm -pthread)
endif()
