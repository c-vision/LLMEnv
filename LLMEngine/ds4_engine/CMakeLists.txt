# ds4_engine — vendorizzazione del motore ds4 (antirez) come libreria C/ObjC.
# Il motore ds4 e' COMPLETO e standalone: include modello, tokenizer, attenzione, MoE, Metal.
# Questo target lo compila cosi' com'e', cosi' FastLLM lo usa come backend DS4 (senza reimplementarlo).
cmake_minimum_required(VERSION 3.16)
project(ds4_engine C OBJC)

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

add_library(ds4_engine STATIC ${DS4_CORE})

if(APPLE)
    set_source_files_properties(ds4_metal.m PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

target_include_directories(ds4_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(ds4_engine PRIVATE -O3 -ffast-math)
if(APPLE)
    target_compile_options(ds4_engine PRIVATE -mcpu=native -Wall -Wextra)
    target_link_libraries(ds4_engine PUBLIC
        -framework Foundation -framework Metal -framework MetalPerformanceShaders
        -lm -pthread)
else()
    target_compile_options(ds4_engine PRIVATE -march=native -Wall -Wextra -D_GNU_SOURCE -fno-finite-math-only)
    target_link_libraries(ds4_engine PUBLIC -lm -pthread)
endif()
