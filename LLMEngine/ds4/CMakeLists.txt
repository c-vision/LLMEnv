# ds4engine — motore ds4 (MIT) vendorizzato in LLMEngine/ds4/vendor.
# NON è ancora collegato a llm_ai_mlx: l'integrazione è la fase F3 del piano.
# Aggiornamento upstream: tools/ds4_sync.sh (vendor/ non si modifica a mano).
cmake_minimum_required(VERSION 3.16)
project(ds4engine C OBJC)

set(CMAKE_C_STANDARD 99)
set(CMAKE_OBJC_STANDARD 99)

option(DS4ENGINE_BUILD "Compila il target ds4engine" OFF)

if(DS4ENGINE_BUILD)
  set(DS4_CORE
      vendor/ds4.c vendor/ds4_image.c vendor/ds4_distributed.c vendor/ds4_tp.c
      vendor/ds4_ssd.c vendor/ds4_layer_pack.c vendor/ds4_engram.c)
  set(DS4_BRIDGE bridge/ds4_bridge.c bridge/llm_ds4_engine_abi.c)   # l'adattatore ABI LlmEngine arriva in F3 (bridge/README)
  if(APPLE)
    list(APPEND DS4_CORE vendor/ds4_metal.m)
  endif()
  add_library(ds4engine STATIC ${DS4_CORE} ${DS4_BRIDGE})
  add_executable(ds4_inspect tools/ds4_inspect.c)
  target_link_libraries(ds4_inspect PRIVATE ds4engine)
  target_include_directories(ds4engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor ${CMAKE_CURRENT_SOURCE_DIR}/bridge ${CMAKE_CURRENT_SOURCE_DIR}/../include)
  target_compile_options(ds4engine PRIVATE -O3 -ffast-math)
  if(APPLE)
    set_source_files_properties(vendor/ds4_metal.m PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    find_library(FW_METAL Metal REQUIRED)
    find_library(FW_FOUNDATION Foundation REQUIRED)
    find_library(FW_MPS MetalPerformanceShaders REQUIRED)
    target_link_libraries(ds4engine PUBLIC ${FW_METAL} ${FW_FOUNDATION} ${FW_MPS} -lm -pthread)
    # gli shader .metal sono caricati a runtime: copia accanto al binario
    add_custom_command(TARGET ds4engine POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/vendor/metal $<TARGET_FILE_DIR:ds4engine>/metal)
  else()
    target_link_libraries(ds4engine PUBLIC -lm -pthread)
  endif()
endif()
