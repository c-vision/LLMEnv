#!/bin/zsh
set -eu
ROOT=/Users/gaetano/Documents/Projects/GMBasic
SRC=$ROOT/../LLMEnv/LLMEngine/mlx
/usr/bin/cc -std=gnu11 -O2 -Wno-unused-function -c -I"$SRC" -I"$ROOT/libs/mlx-deps/mlx-c-src" -I"$ROOT/libs/mlx-deps/mlx-src" -I"$SRC/build/_deps/metal_cpp-src/include" "$SRC/tests/llm_dsv41_vision_test.c" -o /tmp/dsv41_vision_test.o
/usr/bin/c++ -O2 /tmp/dsv41_vision_test.o "$SRC"/build/_deps/mlx_c-build/libmlxc.a "$SRC"/build/_deps/mlx-build/libmlx.a "$SRC"/build/_deps/cjson-build/libcjson.a "$SRC"/build/_deps/mlx-build/mlx/io/libgguflib.a "$SRC"/build/jaccl/libjaccl.a -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate -o /tmp/llm_dsv41_vision_test
/tmp/llm_dsv41_vision_test
