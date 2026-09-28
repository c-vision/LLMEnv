/* llm_mlx_pagesel_test.c — test dell'op sparse page-level (§G-bis leva 1).
 * Verifica: selezionando TUTTE le pagine l'output e' (near) identico all'attention piena;
 * selezionando un sottoinsieme l'op gira. Linka mlx-c + libmlx. */
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static mlx_array mk(const float* d, int shape[4], int nd) {
    return mlx_array_new_data(d, shape, nd, MLX_FLOAT32);
}

static float maxdiff(mlx_array a, mlx_array b) {
    mlx_array_eval(a); mlx_array_eval(b);
    const float* x = mlx_array_data_float32(a);
    const float* y = mlx_array_data_float32(b);
    size_t n = mlx_array_size(a);
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) { float d = x[i] - y[i]; if (d < 0) d = -d; if (d > m) m = d; }
    return m;
}

static void fill(float* p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) { s = s * 1103515245u + 12345u; p[i] = ((s >> 16) & 0x7fff) / 16384.0f - 1.0f; }
}

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_CPU, 0);
    mlx_stream stream = mlx_stream_new_device(dev);
    const int B = 1, nh = 4, nkv = 2, N = 8, D = 4, page = 4;   /* GQA: nh=4, nkv=2 */
    const float scale = 1.0f / 4.0f;

    float qd[1 * 4 * 1 * 4], kd[1 * 2 * 8 * 4], vd[1 * 2 * 8 * 4];
    fill(qd, 1 * 4 * 1 * 4, 7); fill(kd, 1 * 2 * 8 * 4, 11); fill(vd, 1 * 2 * 8 * 4, 13);
    int qs[4] = {B, nh, 1, D}, ks[4] = {B, nkv, N, D};
    mlx_array q = mk(qd, qs, 4), k = mk(kd, ks, 4), v = mk(vd, ks, 4);

    mlx_array full = mlx_array_new();
    CHECK(llm_mlx_gqa_sdpa(&full, q, k, v, scale, 0, stream));

    /* tutte le pagine (0,1) -> deve coincidere con l'attention piena */
    int all[2] = {0, 1};
    mlx_array a_all = mlx_array_new();
    CHECK(llm_mlx_pagesel_decode_sdpa(&a_all, q, k, v, scale, page, all, 2, stream));
    float d_all = maxdiff(full, a_all);
    CHECK(d_all < 1e-6f);

    /* sottoinsieme (pagina 1, gli ultimi 4 token) -> gira e differisce (lossy) */
    int one[1] = {1};
    mlx_array a_one = mlx_array_new();
    CHECK(llm_mlx_pagesel_decode_sdpa(&a_one, q, k, v, scale, page, one, 1, stream));
    float d_one = maxdiff(full, a_one);
    CHECK(d_one > 1e-6f);

    printf("maxdiff full-vs-all=%.3g  full-vs-page1=%.3g\n", d_all, d_one);
    mlx_array_free(full); mlx_array_free(a_all); mlx_array_free(a_one);
    mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
    mlx_stream_free(stream); mlx_device_free(dev);
    if (fails == 0) printf("llm_mlx_pagesel_test: OK\n");
    return fails ? 1 : 0;
}
