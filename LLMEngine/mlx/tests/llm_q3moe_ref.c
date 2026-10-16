/* llm_q3moe_ref.c — riferimento C (mlx-c puro, nessun ramo dell'engine) di Qwen3-MoE: stessa sequenza di op di
 * mlx_lm/models/qwen3_moe.py (embed quantizzato, rms_norm, quantized_matmul, rope, SDPA causal, softmax precise,
 * argpartition, gather_qmm, SwiGLU compilato, lm_head). Serve a confrontare l'engine layer per layer:
 * l'engine con LLM_DBG_DUMP_DIR=<dir> LLM_DBG_DUMP_LAYERS=<n> scrive gli stessi file L<l>_<stadio>.bin.
 * Verificato (2026-10-16, Coder-30B-A3B 8bit, MLX 0.32.0): 48/48 layer bit-exact vs engine e vs mlx_lm reale.
 * Opzioni di debug: REF_NOCOMPILE=1 (SwiGLU eager), REF_F32SWIGLU=1 (intermedi f32), REF_VERBOSE=1.
 *
 * Build (dal build dir dell'engine, linkando la stessa libmlx del dylib):
 *   cc -std=gnu99 -O1 -I<libs/mlx-deps/mlx-c-src> -o q3ref llm_q3moe_ref.c -L<build> -lllm_ai_mlx -Wl,-rpath,<build>
 * Uso: q3ref <model_dir> <dump_dir> <ndump_layers> <nlayers> tok0 tok1 ...
 * Oracolo Python: mlx_lm (env /Users/gaetano/ai/env312 = MLX 0.32.0) — confrontare SOLO a pari versione MLX. */
#include "mlx/c/mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#define CK(x) do { if ((x) != 0) { fprintf(stderr, "ERR %s:%d %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static mlx_stream S;
static mlx_map_string_to_array SH[16]; static int NSH = 0;
static const char* DUMP; static int NDUMP;
static const int GS = 64, BITS = 8;
static mlx_optional_int OG = {64, true}, OB = {8, true};

static mlx_array get(const char* name) {
    for (int i = 0; i < NSH; i++) {
        mlx_array a = mlx_array_new();
        if (mlx_map_string_to_array_get(&a, SH[i], name) == 0 && a.ctx) return a;
    }
    fprintf(stderr, "tensore mancante: %s\n", name); exit(2);
}
static mlx_array getf(const char* fmt, int l, const char* suf) {
    char n[256]; snprintf(n, sizeof n, fmt, l, suf); return get(n);
}
static void dump(const char* nm, int l, mlx_array a) {
    if (l >= NDUMP) return;
    mlx_array f = mlx_array_new();
    CK(mlx_astype(&f, a, MLX_FLOAT32, S)); CK(mlx_array_eval(f));
    char p[1024]; snprintf(p, sizeof p, "%s/L%d_%s.bin", DUMP, l, nm);
    FILE* fp = fopen(p, "wb"); fwrite(mlx_array_data_float32(f), 4, mlx_array_size(f), fp); fclose(fp);
    mlx_array_free(f);
}
/* nn.QuantizedLinear: mx.quantized_matmul(x, w, scales, biases, transpose=True, gs, bits) */
static mlx_array qlin(mlx_array x, const char* base) {
    char n[256]; mlx_array r = mlx_array_new();
    snprintf(n, sizeof n, "%s.weight", base); mlx_array w = get(n);
    snprintf(n, sizeof n, "%s.scales", base); mlx_array sc = get(n);
    snprintf(n, sizeof n, "%s.biases", base); mlx_array bi = get(n);
    CK(mlx_quantized_matmul(&r, x, w, sc, bi, true, OG, OB, "affine", S));
    return r;
}
static mlx_array rms(mlx_array x, mlx_array w, float eps) {
    mlx_array r = mlx_array_new(); CK(mlx_fast_rms_norm(&r, x, w, eps, S)); return r;
}
/* swiglu compilato: silu(gate) * x */
static int swiglu_fn(mlx_vector_array* res, const mlx_vector_array in) {
    mlx_array g = mlx_array_new(), x = mlx_array_new(), sg = mlx_array_new(), sl = mlx_array_new(), o = mlx_array_new();
    mlx_vector_array_get(&g, in, 0); mlx_vector_array_get(&x, in, 1);
    mlx_sigmoid(&sg, g, S); mlx_multiply(&sl, g, sg, S); mlx_multiply(&o, sl, x, S);
    *res = mlx_vector_array_new_data(&o, 1);
    return 0;
}
static mlx_closure SWIGLU;
static mlx_array swiglu(mlx_array g, mlx_array x) {
    if (getenv("REF_F32SWIGLU")) { mlx_array gf=mlx_array_new(), xf=mlx_array_new(), sg=mlx_array_new(), sl=mlx_array_new(), o=mlx_array_new(), ob=mlx_array_new();
        mlx_astype(&gf,g,MLX_FLOAT32,S); mlx_astype(&xf,x,MLX_FLOAT32,S); mlx_sigmoid(&sg,gf,S); mlx_multiply(&sl,gf,sg,S); mlx_multiply(&o,sl,xf,S); mlx_astype(&ob,o,mlx_array_dtype(g),S); return ob; }
    if (getenv("REF_NOCOMPILE")) { mlx_vector_array in; mlx_array a[2] = {g, x}; in = mlx_vector_array_new_data(a, 2);
        mlx_vector_array out = mlx_vector_array_new(); swiglu_fn(&out, in); mlx_array r = mlx_array_new(); mlx_vector_array_get(&r, out, 0); return r; }
    mlx_array a[2] = {g, x}; mlx_vector_array in = mlx_vector_array_new_data(a, 2);
    mlx_vector_array out = mlx_vector_array_new(); CK(mlx_closure_apply(&out, SWIGLU, in));
    mlx_array r = mlx_array_new(); mlx_vector_array_get(&r, out, 0); return r;
}
static mlx_array gqmm(mlx_array x, const char* base, mlx_array inds) {
    char n[256]; mlx_array r = mlx_array_new(), none = mlx_array_new();
    snprintf(n, sizeof n, "%s.weight", base); mlx_array w = get(n);
    snprintf(n, sizeof n, "%s.scales", base); mlx_array sc = get(n);
    snprintf(n, sizeof n, "%s.biases", base); mlx_array bi = get(n);
    CK(mlx_gather_qmm(&r, x, w, sc, bi, none, inds, true, OG, OB, "affine", false, S));
    return r;
}

int main(int argc, char** argv) {
    const char* dir = argv[1]; DUMP = argv[2]; NDUMP = atoi(argv[3]); int NL = atoi(argv[4]);
    int toks[256], T = 0; for (int i = 5; i < argc; i++) toks[T++] = atoi(argv[i]);
    S = mlx_default_gpu_stream_new(); mlx_set_default_stream(S);
    mlx_stream cpu = mlx_default_cpu_stream_new();
    DIR* d = opendir(dir); struct dirent* e;
    while ((e = readdir(d))) { size_t n = strlen(e->d_name);
        if (n > 12 && !strcmp(e->d_name + n - 12, ".safetensors")) {
            char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
            SH[NSH] = mlx_map_string_to_array_new(); mlx_map_string_to_string mm = mlx_map_string_to_string_new();
            CK(mlx_load_safetensors(&SH[NSH], &mm, p, cpu)); NSH++; } }
    closedir(d);
    SWIGLU = mlx_closure_new(); { mlx_closure c = mlx_closure_new_func(swiglu_fn); CK(mlx_compile(&SWIGLU, c, true)); }
    const float eps = 1e-6f; const int H = 32, KV = 4, HD = 128, E = 128, K = 8, D = 2048;

    /* embed: QuantizedEmbedding = dequantize(w[x], scales[x], biases[x]) */
    mlx_array ids = mlx_array_new_data(toks, (int[]){T}, 1, MLX_INT32);
    mlx_array ew = get("model.embed_tokens.weight"), es = get("model.embed_tokens.scales"), eb = get("model.embed_tokens.biases");
    mlx_array w_ = mlx_array_new(), s_ = mlx_array_new(), b_ = mlx_array_new(), h = mlx_array_new(), none = mlx_array_new();
    CK(mlx_take_axis(&w_, ew, ids, 0, S)); CK(mlx_take_axis(&s_, es, ids, 0, S)); CK(mlx_take_axis(&b_, eb, ids, 0, S));
    mlx_optional_dtype nodt = {MLX_FLOAT32, false};
    CK(mlx_dequantize(&h, w_, s_, b_, OG, OB, "affine", none, nodt, S));
    { int sh[3] = {1, T, D}; mlx_array r = mlx_array_new(); CK(mlx_reshape(&r, h, sh, 3, S)); h = r; }
    dump("x", 0, h);

    for (int l = 0; l < NL; l++) {
        char base[256], nm[256];
        /* attention */
        mlx_array lnw = getf("model.layers.%d.%s", l, "input_layernorm.weight");
        mlx_array normed = rms(h, lnw, eps); dump("normed", l, normed);
        snprintf(base, sizeof base, "model.layers.%d.self_attn.q_proj", l); mlx_array q = qlin(normed, base); dump("q", l, q);
        snprintf(base, sizeof base, "model.layers.%d.self_attn.k_proj", l); mlx_array k = qlin(normed, base);
        snprintf(base, sizeof base, "model.layers.%d.self_attn.v_proj", l); mlx_array v = qlin(normed, base);
        int qs[4] = {1, T, H, HD}, ks[4] = {1, T, KV, HD};
        mlx_array q4 = mlx_array_new(), k4 = mlx_array_new(), v4 = mlx_array_new();
        CK(mlx_reshape(&q4, q, qs, 4, S)); CK(mlx_reshape(&k4, k, ks, 4, S)); CK(mlx_reshape(&v4, v, ks, 4, S));
        mlx_array qnw = getf("model.layers.%d.%s", l, "self_attn.q_norm.weight"), knw = getf("model.layers.%d.%s", l, "self_attn.k_norm.weight");
        mlx_array qn = rms(q4, qnw, eps), kn = rms(k4, knw, eps); dump("qn", l, qn);
        int perm[4] = {0, 2, 1, 3};
        mlx_array qt = mlx_array_new(), kt = mlx_array_new(), vt = mlx_array_new();
        CK(mlx_transpose_axes(&qt, qn, perm, 4, S)); CK(mlx_transpose_axes(&kt, kn, perm, 4, S)); CK(mlx_transpose_axes(&vt, v4, perm, 4, S));
        mlx_optional_float base_th = {10000000.0f, true};
        mlx_array qr = mlx_array_new(), kr = mlx_array_new();
        CK(mlx_fast_rope(&qr, qt, HD, false, base_th, 1.0f, 0, none, S)); CK(mlx_fast_rope(&kr, kt, HD, false, base_th, 1.0f, 0, none, S));
        dump("qrope", l, qr);
        mlx_array att = mlx_array_new();
        CK(mlx_fast_scaled_dot_product_attention(&att, qr, kr, vt, 1.0f / sqrtf((float)HD), T > 1 ? "causal" : "", none, none, S));
        dump("sdpa", l, att);
        mlx_array at = mlx_array_new(), af = mlx_array_new(); int fs[3] = {1, T, H * HD};
        CK(mlx_transpose_axes(&at, att, perm, 4, S)); CK(mlx_reshape(&af, at, fs, 3, S));
        snprintf(base, sizeof base, "model.layers.%d.self_attn.o_proj", l); mlx_array r = qlin(af, base); dump("oproj", l, r);
        mlx_array hh = mlx_array_new(); CK(mlx_add(&hh, h, r, S)); dump("h_attn", l, hh);

        /* MoE */
        mlx_array pw = getf("model.layers.%d.%s", l, "post_attention_layernorm.weight");
        mlx_array n2 = rms(hh, pw, eps); dump("ffn_normed", l, n2);
        snprintf(base, sizeof base, "model.layers.%d.mlp.gate", l); mlx_array gates = qlin(n2, base);
        mlx_array gsm = mlx_array_new(); CK(mlx_softmax_axis(&gsm, gates, -1, true, S)); dump("gates", l, gsm);
        mlx_array part = mlx_array_new(), inds = mlx_array_new(), scores = mlx_array_new();
        CK(mlx_argpartition_axis(&part, gsm, -K, -1, S));
        { int a[3] = {0, 0, E - K}, b[3] = {1, T, E}, st[3] = {1, 1, 1}; CK(mlx_slice(&inds, part, a, 3, b, 3, st, 3, S)); }
        CK(mlx_take_along_axis(&scores, gsm, inds, -1, S));
        { mlx_array ss = mlx_array_new(), sc2 = mlx_array_new(); int ax[1] = {-1};
          CK(mlx_sum_axes(&ss, scores, ax, 1, true, S)); CK(mlx_divide(&sc2, scores, ss, S)); scores = sc2; }
        dump("inds", l, inds); dump("scores", l, scores);
        /* SwitchGLU: x expand_dims (-2,-3) -> [1,T,1,1,D] */
        mlx_array x5 = mlx_array_new(); { int sh[5] = {1, T, 1, 1, D}; CK(mlx_reshape(&x5, n2, sh, 5, S)); }
        mlx_array inds_u = mlx_array_new(); CK(mlx_astype(&inds_u, inds, MLX_UINT32, S));
        snprintf(base, sizeof base, "model.layers.%d.mlp.switch_mlp.up_proj", l); mlx_array xu = gqmm(x5, base, inds_u);
        snprintf(base, sizeof base, "model.layers.%d.mlp.switch_mlp.gate_proj", l); mlx_array xg = gqmm(x5, base, inds_u);
        dump("g_out", l, xg);
        mlx_array act = swiglu(xg, xu); dump("act", l, act);
        snprintf(base, sizeof base, "model.layers.%d.mlp.switch_mlp.down_proj", l); mlx_array xd = gqmm(act, base, inds_u);
        { mlx_array sq = mlx_array_new(); int ax[1] = {-2}; CK(mlx_squeeze_axes(&sq, xd, ax, 1, S)); xd = sq; }
        dump("d_out", l, xd);
        mlx_array sce = mlx_array_new(), wd = mlx_array_new(), y = mlx_array_new(); int ax2[1] = {-2};
        { int sh[4] = {1, T, K, 1}; CK(mlx_reshape(&sce, scores, sh, 4, S)); }
        CK(mlx_multiply(&wd, xd, sce, S)); CK(mlx_sum_axes(&y, wd, ax2, 1, false, S)); dump("moe_out", l, y);
        mlx_array out = mlx_array_new(); CK(mlx_add(&out, hh, y, S)); dump("out", l, out);
        CK(mlx_array_eval(out));
        h = out;
        if (getenv("REF_VERBOSE")) fprintf(stderr, "layer %d ok\n", l);
    }
    /* final norm + lm_head, logprobs dell'ultima posizione */
    mlx_array nw = get("model.norm.weight"); mlx_array fn = rms(h, nw, eps);
    mlx_array lg = qlin(fn, "lm_head");
    mlx_array last = mlx_array_new(); { int a[3] = {0, T - 1, 0}, b[3] = {1, T, 151936}, st[3] = {1, 1, 1}; CK(mlx_slice(&last, lg, a, 3, b, 3, st, 3, S)); }
    mlx_array lse = mlx_array_new(), lp = mlx_array_new(), lpf = mlx_array_new();
    CK(mlx_logsumexp_axis(&lse, last, -1, true, S)); CK(mlx_subtract(&lp, last, lse, S));   /* bf16 come mlx_lm */
    CK(mlx_astype(&lpf, lp, MLX_FLOAT32, S)); CK(mlx_array_eval(lpf));
    mlx_array lf = mlx_array_new(), lse2 = mlx_array_new(), lp2 = mlx_array_new();
    CK(mlx_astype(&lf, last, MLX_FLOAT32, S)); CK(mlx_logsumexp_axis(&lse2, lf, -1, true, S)); CK(mlx_subtract(&lp2, lf, lse2, S)); CK(mlx_array_eval(lp2));
    const float* p = mlx_array_data_float32(lpf); const float* p2 = mlx_array_data_float32(lp2);
    { FILE* fp = fopen("/dev/stdout", "w"); (void)fp; }
    for (int pass = 0; pass < 2; pass++) {
        const float* q = pass ? p2 : p; int used[8] = {0}; printf("REF%s n=%d:", pass ? "(f32 logprob)" : "(bf16 logprob)", T);
        for (int r = 0; r < 8; r++) { int bi = -1; float bv = -1e30f;
            for (int i = 0; i < 151936; i++) if (q[i] > bv) { int u = 0; for (int j = 0; j < r; j++) if (used[j] == i) u = 1; if (!u) { bv = q[i]; bi = i; } }
            used[r] = bi; printf(" %d(%.4f)", bi, bv); }
        printf("\n");
    }
    return 0;
}
