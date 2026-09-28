# Progetto Objective-C autonomo: kernel Metal di LLMEnv — GEMV (Q4 affine + MXFP4-native),
# KV cache (pagine + decode online-softmax), oQ A8 (int8-act × Q4/Q5, tensor unit M5, port oMLX),
# MPP q4 (tensor unit). Solo framework di sistema Metal/Foundation, nessuna dipendenza esterna.
# I kernel vivono come header (Metal source embedded) e shader .metal, consumati dall'engine via
# include; qui si costruiscono i self-check di correttezza.
cmake_minimum_required(VERSION 3.16)
project(llmmetal OBJC C)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_OBJC_STANDARD 11)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# GEMV Q4 affine: self-check vs riferimento CPU (quantizza una matrice random in Q4 affine).
add_executable(llmmetal_gemv_q4_selfcheck gemv/tests/gemv_q4_selfcheck.m)
target_link_libraries(llmmetal_gemv_q4_selfcheck PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
target_compile_options(llmmetal_gemv_q4_selfcheck PRIVATE -fobjc-arc -O2)

# Kernel #36: GEMV decode MXFP4-native (gpt-oss), dequant e8m0*e2m1 che replica MLX fp4.h/fp8.h.
add_executable(llmmetal_gemv_mxfp4_selfcheck gemv/tests/gemv_mxfp4_selfcheck.m)
target_link_libraries(llmmetal_gemv_mxfp4_selfcheck PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
target_compile_options(llmmetal_gemv_mxfp4_selfcheck PRIVATE -fobjc-arc -O2)

# KV cache Metal reale (pagine fisiche, scrittura in-place, decode attention paged online-softmax).
add_executable(llmmetal_kv_selfcheck kv/tests/kv_metal_selfcheck.m)
target_link_libraries(llmmetal_kv_selfcheck PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
target_compile_options(llmmetal_kv_selfcheck PRIVATE -fobjc-arc -O2)
