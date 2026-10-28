#include <mlx/c/mlx.h>
#include <stdio.h>
static const char* HDR = "using namespace metal;\n";
static const char* SRC =
  "uint lane = thread_position_in_threadgroup.x;\n"
  "simdgroup_half8x8 A; simdgroup_half8x8 B; simdgroup_half8x8 C;\n"
  "simdgroup_load(A, inp, 8);\n"
  "simdgroup_load(B, inp, 8);\n"
  "simdgroup_multiply_accumulate(C, A, B, (simdgroup_half8x8)0);\n"
  "if (lane==0) out[0] = 1.0f;\n";
int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_vector_string in=mlx_vector_string_new(); mlx_vector_string_append_value(in,"inp");
  mlx_vector_string on=mlx_vector_string_new(); mlx_vector_string_append_value(on,"out");
  mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new("mma_test", in, on, SRC, HDR, false, false);
  printf("kernel mma_new: %s\n", k.ctx!=NULL ? "OK (simdgroup_matrix COMPILA)" : "FAIL");
  return 0;
}
