/* llm_mlx_compile_cost_probe.c — LOAD-FREE: il compile per-layer a n_tokens=1 PAGA?
 *
 * Motivazione (M1 "Decode-Compilato"): compilare i blocchi GDN (~40-70 nodi/layer, 36 layer)
 * ridurrebbe i nodi/token da ~3125 a poche centinaia. MA la contro-evidenza interna dice che
 * `mlx_closure_apply` ha un costo fisso CPU per applicazione che a n_tokens==1 NON si
 * ammortizza: GEGLU compilata -57% a 3 nodi (commento LLM_FUSE_GEGLU), SwiGLU compilata
 * utile solo nel prefill (n_tokens>1), closure del decode-step intero (LLM_COMPILE) +2,7%
 * su llama-1B. Questo probe misura il BREAK-EVEN senza modello: simula un passo-token =
 * 36 blocchi in catena (shapes fisse, stato I/O) + UNA eval, e confronta eager vs closure
 * compilata al crescere dei nodi per blocco (10/40/80).
 *
 * Cosa misura: ms per passo-token (mediana di 3 gruppi di 50 iterazioni, warmup scartato) e
 * il rapporto compiled/eager. Se il rapporto > 1 ai nodi reali del blocco GDN (~40-70),
 * M1 va FERMATO (la closure non paga a n_tokens=1) — e' il criterio di stop del todo.
 *
 * Build (dal build/ gia' configurato, come la roof probe — NON tocca la dylib condivisa):
 *   cd native/llmengine_mlx && /usr/bin/c++ -std=c++20 -O2 -Wno-unused-function \
 *     -I. -I../../libs/mlx-deps/mlx-c-src -I../../libs/mlx-deps/mlx-src \
 *     -Ibuild/_deps/metal_cpp-src/include \
 *     llm_mlx_compile_cost_probe.c build/_deps/mlx_c-build/libmlxc.a \
 *     build/_deps/mlx-build/libmlx.a build/_deps/cjson-build/libcjson.a \
 *     build/_deps/mlx-build/mlx/io/libgguflib.a build/jaccl/libjaccl.a \
 *     -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
 *     -o /tmp/llm_compile_cost_probe
 */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NLAYERS 36
#define NITER 50
#define NREP 3
#define NOPS_LIST {10, 40, 80}

static mlx_stream g_st;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

typedef struct { int nops; } BlockCtx;

/* Blocco "GDN-shaped": catena di nops op elementwise su x [1,1,8192] + aggiornamento
 * dello stato (s1 [1,32,128,128], s2 [1,4,8192]) come I/O — nessuna mutazione in-place. */
static int block_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    BlockCtx* c = (BlockCtx*)payload;
    mlx_array x = mlx_array_new(), s1 = mlx_array_new(), s2 = mlx_array_new();
    if (mlx_vector_array_get(&x, in, 0) != 0 || mlx_vector_array_get(&s1, in, 1) != 0 ||
        mlx_vector_array_get(&s2, in, 2) != 0) return 1;
    mlx_array two = mlx_array_new_float32(2.0f), half = mlx_array_new_float32(0.5f);
    mlx_array t = mlx_array_new();
    mlx_array_set(&t, x);
    for (int i = 0; i < c->nops; i++) {
        mlx_array r = mlx_array_new();
        int rc = 1;
        switch (i % 5) {
            case 0: rc = mlx_multiply(&r, t, two, g_st); break;
            case 1: rc = mlx_add(&r, t, two, g_st); break;
            case 2: rc = mlx_tanh(&r, t, g_st); break;
            case 3: rc = mlx_sigmoid(&r, t, g_st); break;
            default: rc = mlx_multiply(&r, t, half, g_st); break;
        }
        if (rc != 0) { mlx_array_free(r); mlx_array_free(t); return 1; }
        mlx_array_free(t);
        t = r;
    }
    mlx_array s1n = mlx_array_new(), s2n = mlx_array_new();
    if (mlx_multiply(&s1n, s1, two, g_st) != 0 || mlx_multiply(&s2n, s2, two, g_st) != 0) return 1;
    mlx_array outs[3] = { t, s1n, s2n };
    mlx_vector_array built = mlx_vector_array_new_data(outs, 3);
    int rc = mlx_vector_array_set(out, built) == 0 ? 0 : 1;
    mlx_vector_array_free(built);
    mlx_array_free(t); mlx_array_free(s1n); mlx_array_free(s2n);
    mlx_array_free(two); mlx_array_free(half);
    mlx_array_free(x); mlx_array_free(s1); mlx_array_free(s2);
    return rc;
}

/* Un passo-token: NLAYERS blocchi in catena, UNA eval finale (come il decode dell'engine). */
static int token_step(mlx_closure* cl, BlockCtx* ctx, mlx_array x0,
                      mlx_array* states1, mlx_array* states2, int compiled, mlx_array* out_y) {
    mlx_array y = mlx_array_new();
    mlx_array_set(&y, x0);
    for (int l = 0; l < NLAYERS; l++) {
        mlx_array ins[3] = { y, states1[l], states2[l] };
        mlx_vector_array in = mlx_vector_array_new_data(ins, 3);
        mlx_vector_array res = mlx_vector_array_new();
        int rc;
        if (compiled) rc = mlx_closure_apply(&res, *cl, in);
        else rc = block_fn(&res, in, ctx);
        mlx_vector_array_free(in);
        if (rc != 0 || mlx_vector_array_size(res) != 3) { mlx_vector_array_free(res); mlx_array_free(y); return 1; }
        mlx_array_free(y);
        y = mlx_array_new();
        mlx_vector_array_get(&y, res, 0);
        mlx_vector_array_free(res);
    }
    if (out_y) mlx_array_set(out_y, y);
    int rc = mlx_array_eval(y);
    mlx_array_free(y);
    return rc == 0 ? 0 : 1;
}

static double median3(double a, double b, double c) {
    if (a > b) { double t = a; a = b; b = t; }
    if (b > c) { double t = b; b = c; c = t; }
    if (a > b) { double t = a; a = b; b = t; }
    return b;
}

int main(void) {
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(dev);
    g_st = mlx_default_gpu_stream_new();

    /* shapes fisse (decode n_tokens=1) */
    int xs[3] = {1, 1, 8192};
    int s1s[4] = {1, 32, 128, 128};
    int s2s[3] = {1, 4, 8192};
    mlx_array x0 = mlx_array_new(), one = mlx_array_new_float32(1.0f);
    mlx_full(&x0, xs, 3, one, MLX_FLOAT32, g_st);
    mlx_array states1[NLAYERS], states2[NLAYERS];
    for (int l = 0; l < NLAYERS; l++) {
        states1[l] = mlx_array_new(); states2[l] = mlx_array_new();
        mlx_full(&states1[l], s1s, 4, one, MLX_FLOAT32, g_st);
        mlx_full(&states2[l], s2s, 3, one, MLX_FLOAT32, g_st);
    }
    mlx_array_eval(x0);

    printf("llm_compile_cost_probe — passo-token simulato: %d blocchi, 1 eval, shapes fisse\n",
           NLAYERS);
    printf("%-6s %-14s %-14s %-8s\n", "nops", "eager ms/tok", "compiled ms/tok", "ratio");
    int nops_list[] = NOPS_LIST;
    for (size_t k = 0; k < sizeof(nops_list) / sizeof(nops_list[0]); k++) {
        BlockCtx ctx = { nops_list[k] };
        mlx_closure raw = mlx_closure_new_func_payload(block_fn, &ctx, NULL);
        mlx_closure cl = mlx_closure_new();
        mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
        int rc = mlx_compile(&cl, raw, /*shapeless=*/false);
        mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
        mlx_closure_free(raw);
        if (rc != 0) { printf("nops=%d: mlx_compile FALLITO\n", nops_list[k]); continue; }

        double e[NREP], c[NREP];
        for (int r = 0; r < NREP; r++) {
            /* warmup 5 + misura NITER */
            for (int i = 0; i < 5; i++) (void)token_step(&cl, &ctx, x0, states1, states2, 0, NULL);
            double t0 = now_ms();
            for (int i = 0; i < NITER; i++) (void)token_step(&cl, &ctx, x0, states1, states2, 0, NULL);
            e[r] = (now_ms() - t0) / NITER;

            for (int i = 0; i < 5; i++) (void)token_step(&cl, &ctx, x0, states1, states2, 1, NULL);
            t0 = now_ms();
            for (int i = 0; i < NITER; i++) (void)token_step(&cl, &ctx, x0, states1, states2, 1, NULL);
            c[r] = (now_ms() - t0) / NITER;
        }
        double me = median3(e[0], e[1], e[2]), mc = median3(c[0], c[1], c[2]);
        printf("%-6d %-14.3f %-14.3f %-8.2f%s\n", nops_list[k], me, mc, mc / me,
               mc > me ? "  <- compiled NON paga" : "  <- compiled paga");
        mlx_closure_free(cl);
    }

    mlx_array_free(x0); mlx_array_free(one);
    for (int l = 0; l < NLAYERS; l++) { mlx_array_free(states1[l]); mlx_array_free(states2[l]); }
    mlx_stream_free(g_st);
    mlx_device_free(dev);
    return 0;
}
