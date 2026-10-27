/* Misura la crescita RSS delle due famiglie di kernel usate nel blocco:
 *  A) gemv Q8 wide (view no-copy per chiamata)      B) kernel MLX (rmsnorm/gate) via gdn layer f32.
 * Nessun modello: pesi sintetici piccoli, 200 iterazioni. */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_gpu.h"
#include "qwen4_native/qwen4_native_layer.h"
#include "qwen4_native/qwen4_native_trunk.h"
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static double rss_mb(void) {
    mach_task_basic_info_data_t info; mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &cnt) != KERN_SUCCESS) return -1;
    return (double)info.resident_size / 1048576.0;
}
static float frand(void) { return (float)((int)(rand() % 2001) - 1000) / 1000.0f; }

int main(void) {
    const int n_in = 2560, n_out = 512, nblk = n_in / 32;
    const size_t rb = (size_t)nblk * 34;
    uint8_t* W = malloc(rb * (size_t)n_out);
    float* x = malloc((size_t)n_in * sizeof(float));
    float* y = malloc((size_t)n_out * sizeof(float));
    srand(3);
    for (int o = 0; o < n_out; o++) { uint8_t* row = W + (size_t)o*rb;
        for (int b = 0; b < nblk; b++) { uint8_t* blk = row + (size_t)b*34;
            _Float16 d = (_Float16)0.001f; __builtin_memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) blk[2+i] = (uint8_t)(int8_t)(rand()%255-127); } }
    for (int i = 0; i < n_in; i++) x[i] = frand();
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (!dev.ctx || mlx_set_default_device(dev) != 0) return 1;
    Q4nGpu g; if (q4n_gpu_init(&g) != 0) return 1;
    Q4nMat mat = { W, 8, rb };
    double r0 = rss_mb();
    for (int it = 0; it < 200; it++) {
        uint8_t* Wn = malloc(rb * (size_t)n_out); memcpy(Wn, W, rb * (size_t)n_out);
        Q4nMat mn = { Wn, 8, rb };
        if (q4n_mat_mv_gpu(&g, &mn, x, y, n_in, n_out) != 0) { printf("gemv fail %d\n", it); return 1; }
        free(Wn);
        if ((it & 63) == 63) q4n_gpu_reset(&g);
    }
    double r1 = rss_mb();
    printf("A) gemv Q8 wide: 200 iter  RSS %.1f -> %.1f MB (delta %.1f, %.3f MB/iter)\n", r0, r1, r1-r0, (r1-r0)/200.0);
    /* C) gemv Q4_K (ds4 batch, vista cached) su pesi a zero */
    { const int nb_q4k = (n_in + 255) / 256; const size_t rb4 = (size_t)nb_q4k * 144;
      uint8_t* W4 = calloc(rb4 * (size_t)n_out, 1);
      Q4nMat m4 = { W4, 12, rb4 };
      double c0 = rss_mb();
      for (int it = 0; it < 200; it++) if (q4n_mat_mv_gpu(&g, &m4, x, y, n_in, n_out) != 0) { printf("q4k fail %d\n", it); return 1; }
      double c1 = rss_mb();
      printf("C) gemv Q4_K ds4: 200 iter  RSS %.1f -> %.1f MB (delta %.1f, %.3f MB/iter)\n", c0, c1, c1-c0, (c1-c0)/200.0);
      free(W4); }
    /* B) GDN layer f32 (kernel MLX) con mv CPU */
    { const int E = 256, Hk = 16, Hv = 48, D = 128, F = 640;
      int conv_dim = 2*Hk*D + Hv*D, v_dim = Hv*D;
      float* qkv = malloc((size_t)conv_dim*E*sizeof(float));
      float* gate = malloc((size_t)v_dim*E*sizeof(float));
      float* beta = malloc((size_t)Hv*E*sizeof(float));
      float* alpha = malloc((size_t)Hv*E*sizeof(float));
      float* outw = malloc((size_t)E*v_dim*sizeof(float));
      float* conv = malloc((size_t)conv_dim*4*sizeof(float));
      float* a = malloc((size_t)Hv*sizeof(float));
      float* dt = malloc((size_t)Hv*sizeof(float));
      float* nrm = malloc((size_t)D*sizeof(float));
      if (!qkv||!gate||!beta||!alpha||!outw||!conv||!a||!dt||!nrm) return 1;
      for (size_t i = 0; i < (size_t)conv_dim*E; i++) qkv[i] = frand()*0.01f;
      for (size_t i = 0; i < (size_t)v_dim*E; i++) gate[i] = frand()*0.01f;
      for (size_t i = 0; i < (size_t)Hv*E; i++) { beta[i] = frand()*0.01f; alpha[i] = frand()*0.01f; }
      for (size_t i = 0; i < (size_t)E*v_dim; i++) outw[i] = frand()*0.01f;
      for (int i = 0; i < conv_dim*4; i++) conv[i] = frand()*0.1f;
      for (int i = 0; i < Hv; i++) { a[i] = -0.1f; dt[i] = 0.01f; }
      for (int i = 0; i < D; i++) nrm[i] = 1.0f;
      Q4nGdnW gw = { {qkv,-1,0},{gate,-1,0},{beta,-1,0},{alpha,-1,0},{outw,-1,0}, conv, a, dt, nrm,
                     conv_dim, v_dim, Hk, Hv, D };
      float* xh = malloc((size_t)E*sizeof(float)); for (int i = 0; i < E; i++) xh[i] = frand();
      int xsh[2] = {1, E};
      mlx_stream s = mlx_stream_new_device(dev);
      mlx_array xa = mlx_array_new_data(xh, xsh, 2, MLX_FLOAT32);
      float* chz = calloc((size_t)(4-1)*conv_dim, sizeof(float));
      float* lsz = calloc((size_t)Hv*D*D, sizeof(float));
      mlx_array ch = mlx_array_new_data(chz, (int[]){4-1, conv_dim}, 2, MLX_FLOAT32);
      mlx_array ls = mlx_array_new_data(lsz, (int[]){Hv, D, D}, 3, MLX_FLOAT32);
      double b0 = rss_mb();
      for (int it = 0; it < 48; it++) {
          mlx_array out = mlx_array_new();
          if (qwen4_native_gdn_layer(&out, xa, &ch, &ls, &gw, q4n_mat_mv_gpu, NULL, s) != 0) { printf("gdn fail %d\n", it); return 1; }
          mlx_vector_array ev = mlx_vector_array_new(); mlx_vector_array_append_value(ev, out);
          mlx_eval(ev); mlx_vector_array_free(ev); mlx_array_free(out);
      }
      double b1 = rss_mb();
      printf("B) GDN f32 (MLX): 48 iter  RSS %.1f -> %.1f MB (delta %.1f, %.1f MB/iter)\n", b0, b1, b1-b0, (b1-b0)/48.0);
    }
    /* D) hc_mix con pesi quantizzati (Q4_0, zero) e dimensioni reali */
    { const int E = 2560, hc = 4, rank = 320, hcE = hc * E;
      size_t rb_dn = (size_t)(10240/32)*18, rb_up = (size_t)(320/32)*18;
      uint8_t* dn = calloc(rb_dn * (size_t)rank, 1);
      uint8_t* up = calloc(rb_up * (size_t)hcE, 1);
      float* norm = malloc((size_t)hcE*4), *inj = malloc((size_t)hc*4);
      for (int i = 0; i < hcE; i++) norm[i] = 1.0f;
      for (int i = 0; i < hc; i++) inj[i] = 0.01f;
      Q4nHcW w = { norm, {dn, 2, rb_dn}, {up, 2, rb_up}, {inj, -1, 0}, rank };
      float* xf = malloc((size_t)hcE*4); for (int i=0;i<hcE;i++) xf[i]=frand();
      mlx_stream ss = mlx_stream_new_device(dev);
      mlx_array xa = mlx_array_new_data(xf, (int[]){1, hcE}, 2, MLX_FLOAT32);
      double d0 = rss_mb();
      for (int it = 0; it < 20; it++) {
          mlx_array mo = mlx_array_new(), iv = mlx_array_new();
          if (qwen4_native_hc_mix(&mo, &iv, xa, &w, E, hc, 1e-6f, q4n_mat_mv_gpu, &g, ss) != 0) { printf("hc fail\n"); return 1; }
          mlx_vector_array ev = mlx_vector_array_new();
          mlx_vector_array_append_value(ev, mo); mlx_vector_array_append_value(ev, iv);
          mlx_eval(ev); mlx_vector_array_free(ev);
          mlx_array_free(mo); mlx_array_free(iv);
      }
      double d1 = rss_mb();
      printf("D) hc_mix Q4_0: 20 iter  RSS %.1f -> %.1f MB (delta %.1f, %.1f MB/iter)\n", d0, d1, d1-d0, (d1-d0)/20.0);
    }
    /* E) hc_mix con pesi Q8 e dimensioni reali (down [320,10240], up [10240,320]) */
    { const int E = 2560, hc = 4, rank = 320, hcE = hc * E;
      size_t rb_dn = (size_t)(10240/32)*34, rb_up = (size_t)(320/32)*34;
      uint8_t* dn = calloc(rb_dn * (size_t)rank, 1);
      uint8_t* up = calloc(rb_up * (size_t)hcE, 1);
      float* norm = malloc((size_t)hcE*4), *inj = malloc((size_t)hc*4);
      for (int i = 0; i < hcE; i++) norm[i] = 1.0f;
      for (int i = 0; i < hc; i++) inj[i] = 0.01f;
      Q4nHcW w = { norm, {dn, 8, rb_dn}, {up, 8, rb_up}, {inj, -1, 0}, rank };
      float* xf = malloc((size_t)hcE*4); for (int i=0;i<hcE;i++) xf[i]=frand();
      mlx_stream ss = mlx_stream_new_device(dev);
      mlx_array xa = mlx_array_new_data(xf, (int[]){1, hcE}, 2, MLX_FLOAT32);
      double e0 = rss_mb();
      struct timespec t0, t1; timespec_get(&t0, TIME_UTC);
      for (int it = 0; it < 48; it++) {
          mlx_array mo = mlx_array_new(), iv = mlx_array_new();
          if (qwen4_native_hc_mix(&mo, &iv, xa, &w, E, hc, 1e-6f, q4n_mat_mv_gpu, &g, ss) != 0) { printf("hc8 fail\n"); return 1; }
          mlx_vector_array ev = mlx_vector_array_new();
          mlx_vector_array_append_value(ev, mo); mlx_vector_array_append_value(ev, iv);
          mlx_eval(ev); mlx_vector_array_free(ev);
          mlx_array_free(mo); mlx_array_free(iv);
      }
      timespec_get(&t1, TIME_UTC);
      double e1 = rss_mb();
      printf("E) hc_mix Q8 reali: 48 iter  %.2f ms/iter  RSS delta %.1f MB\n",
             ((t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6)/48.0, e1-e0);
    }
    q4n_gpu_free(&g);
    return 0;
}
