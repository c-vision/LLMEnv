/* llm_qgemv_fast_probe.c — parita' e prestazioni: la nostra GEMV 4-bit f16 registrata come
 * mlx_fast_metal_kernel nel grafo MLX, confrontata con mlx_quantized_matmul.
 * Layout affine MLX: w uint32 (N, K*bits/32), scales/biases f16 (N, K/group), group=64, bits=4. */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mach/mach_time.h>

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / (double)tb.denom / 1e6;
}

static const char* SRC =
"\n"
"int N = dims[0], K = dims[1];\n"
"int n = (int)threadgroup_position_in_grid.x;\n"
"if (n >= N) return;\n"
"int Kw = K >> 3;\n"
"const device uint* wr = w + (size_t)n * Kw;\n"
"const device half* sc = scales + (size_t)n * (K >> 6);\n"
"const device half* bs = biases + (size_t)n * (K >> 6);\n"
"int tid = (int)thread_index_in_threadgroup;\n"
"float acc = 0.0f;\n"
"for (int u = tid; u < Kw; u += 256) {\n"
"    uint pk = wr[u];\n"
"    int base = u << 3, g = base >> 6;\n"
"    float s = (float)sc[g], b = (float)bs[g];\n"
"    #pragma unroll\n"
"    for (int j = 0; j < 8; ++j)\n"
"        acc += (float((pk >> (j << 2)) & 0xFu) * s + b) * (float)x[base + j];\n"
"}\n"
"for (int off = 16; off > 0; off >>= 1) acc += simd_shuffle_down(acc, off);\n"
"threadgroup float sm[8];\n"
"int lane = (int)thread_index_in_simdgroup, sid = (int)simdgroup_index_in_threadgroup;\n"
"if (lane == 0) sm[sid] = acc;\n"
"threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"if (tid == 0) { float s2 = 0; for (int i = 0; i < 8; i++) s2 += sm[i]; out[n] = (half)s2; }\n";

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 4096;
    int K = argc > 2 ? atoi(argv[2]) : 4096;
    int reps = argc > 3 ? atoi(argv[3]) : 20;
    int group = 64, bits = 4;

    mlx_stream s = mlx_default_gpu_stream_new();
    mlx_optional_int gsz = {group, true}, bts = {bits, true};
    mlx_array empty = mlx_array_new();

    mlx_array lo = mlx_array_new_float(-1.f), hi = mlx_array_new_float(1.f);
    uint32_t kv1[2] = {7, 0}, kv2[2] = {11, 0};
    mlx_array k1 = mlx_array_new_data(kv1, (int[]){2}, 1, MLX_UINT32);
    mlx_array k2 = mlx_array_new_data(kv2, (int[]){2}, 1, MLX_UINT32);
    mlx_array wf = mlx_array_new(), xf = mlx_array_new();
    mlx_random_uniform(&wf, lo, hi, (int[]){N, K}, 2, MLX_FLOAT16, k1, s);
    mlx_random_uniform(&xf, lo, hi, (int[]){1, K}, 2, MLX_FLOAT16, k2, s);

    mlx_vector_array q = mlx_vector_array_new();
    if (mlx_quantize(&q, wf, gsz, bts, "affine", empty, s) != 0) { fprintf(stderr, "quantize fail\n"); return 1; }
    mlx_array wq = mlx_array_new(), sc = mlx_array_new(), bi = mlx_array_new();
    mlx_vector_array_get(&wq, q, 0);
    mlx_vector_array_get(&sc, q, 1);
    mlx_vector_array_get(&bi, q, 2);

    /* riferimento MLX (x @ w.T) */
    mlx_array ref = mlx_array_new();
    mlx_quantized_matmul(&ref, xf, wq, sc, bi, true, gsz, bts, "affine", s);
    mlx_array_eval(ref);

    /* nostro kernel */
    const char* in_names[] = {"w", "scales", "biases", "x", "dims"};
    const char* out_names[] = {"out"};
    mlx_vector_string vin = mlx_vector_string_new_data(in_names, 5);
    mlx_vector_string vout = mlx_vector_string_new_data(out_names, 1);
    mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new("qgemv4_f16", vin, vout, SRC, "", false, false);
    mlx_vector_string_free(vin); mlx_vector_string_free(vout);
    if (!k.ctx) { fprintf(stderr, "kernel_new fail\n"); return 2; }

    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_output_arg(cfg, (int[]){1, N}, 2, MLX_FLOAT16);
    mlx_fast_metal_kernel_config_set_grid(cfg, N * 256, 1, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);

    mlx_array dims = mlx_array_new_data((int[]){N, K}, (int[]){2}, 1, MLX_INT32);
    const mlx_array ins[] = {wq, sc, bi, xf, dims};
    mlx_vector_array vin2 = mlx_vector_array_new_data(ins, 5);
    mlx_vector_array outs = mlx_vector_array_new();
    if (mlx_fast_metal_kernel_apply(&outs, k, vin2, cfg, s) != 0) { fprintf(stderr, "apply fail\n"); return 3; }
    mlx_array out = mlx_array_new();
    mlx_vector_array_get(&out, outs, 0);
    mlx_array_eval(out);

    /* confronto in f32 */
    mlx_array rf = mlx_array_new(), of = mlx_array_new();
    mlx_astype(&rf, ref, MLX_FLOAT32, s);
    mlx_astype(&of, out, MLX_FLOAT32, s);
    mlx_array_eval(rf); mlx_array_eval(of);
    const float* rp = mlx_array_data_float32(rf);
    const float* op = mlx_array_data_float32(of);
    double md = 0; int worst = 0;
    for (int i = 0; i < N; i++) { double d = (double)rp[i] - (double)op[i]; if (d < 0) d = -d; if (d > md) { md = d; worst = i; } }
    printf("parita': maxabs=%.5f (i=%d ref=%.4f ours=%.4f) | ref[0]=%.4f ours[0]=%.4f\n",
           md, worst, rp[worst], op[worst], rp[0], op[0]);

    /* timing (eval per rep, come il voce del decode) */
    double t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        mlx_vector_array o = mlx_vector_array_new();
        mlx_fast_metal_kernel_apply(&o, k, vin2, cfg, s);
        mlx_array a = mlx_array_new(); mlx_vector_array_get(&a, o, 0); mlx_array_eval(a);
        mlx_array_free(a); mlx_vector_array_free(o);
    }
    double dt = (now_ms() - t0) / reps;
    printf("OURS qgemv4_f16 N=%d K=%d: %.4f ms/call  %.1f GB/s(W)\n", N, K, dt, (double)N * K * 0.5 / dt / 1e6);

    t0 = now_ms();
    for (int r = 0; r < reps; r++) {
        mlx_array o = mlx_array_new();
        mlx_quantized_matmul(&o, xf, wq, sc, bi, true, gsz, bts, "affine", s);
        mlx_array_eval(o);
        mlx_array_free(o);
    }
    dt = (now_ms() - t0) / reps;
    printf("MLX  qmm      N=%d K=%d: %.4f ms/call  %.1f GB/s(W)\n", N, K, dt, (double)N * K * 0.5 / dt / 1e6);
    return 0;
}
