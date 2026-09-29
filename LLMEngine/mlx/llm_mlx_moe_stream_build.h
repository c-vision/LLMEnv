/* llm_mlx_moe_stream_build.h — costruisce i tensori STACKED [k, d1, d2] dei k esperti unici
 * a partire dalle SLICE dell'ingest canonico safetensors (`llmengine_moe_safetensors_ingest.h`).
 *
 * NON un formato nuovo: il descrittore `(layer,expert,proj,kind) -> slice` e' UNO (l'ingest);
 * qui c'e' solo il consumo lato MLX (costruzione degli mlx_array per gather_qmm). I bundle sono
 * copiati in buffer RESIDENTI (mlock) -> Metal non fa fault (no Command buffer). */
#ifndef LLM_MLX_MOE_STREAM_BUILD_H
#define LLM_MLX_MOE_STREAM_BUILD_H
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "../moe/include/llmengine_moe_safetensors_ingest.h"
#include "llm_mlx_tq_decode.h"
#include "llm_mlx_tq_qmv_kernel.h"
#if defined(__APPLE__)
#include <dispatch/dispatch.h>   /* popolamento parallelo della cache esperto->affine-2bit */
#endif

/* Costruisce+materializza [k, d1, d2] per (layer, proj, kind) dagli esperti `uniq[]`.
 * kind: 0=weight(U32) 1=scales(BF16) 2=biases(BF16). Ritorna 0 e *out, oppure -1. */
static inline int llm_mlx_xs_stack(const LlmEngineMoeTsIngest* s, int layer, int proj, int kind,
                                   const int* uniq, int k, mlx_stream str, mlx_array* out) {
    (void)str;
    if (k <= 0 || !out) return -1;
    const LlmEngineMoeTsSlice* t = llmengine_moe_ts_slice(s, layer, proj, kind);
    if (!t || t->d1 <= 0 || t->d2 <= 0) return -1;
    const mlx_dtype dt = (kind == 0) ? MLX_UINT32 : MLX_BFLOAT16;
    const size_t per = (size_t)t->d1 * (size_t)t->d2 * ((kind == 0) ? 4u : 2u);
    /* UN buffer contiguo [k,d1,d2] + MANAGED (MLX prende ownership, chiama `free` a fine grafo):
     * una sola copia mmap->buffer, un solo array. (Prima: k array + mlx_stack che ricopiava, e un
     * eval per-chiamata -> ~1300 sync/token.) */
    uint8_t* buf = (uint8_t*)malloc(per * (size_t)k);
    if (!buf) return -1;
    for (int j = 0; j < k; j++) {
        const uint8_t* p = llmengine_moe_ts_get(s, layer, proj, kind, uniq[j], NULL);
        if (!p) { free(buf); return -1; }
        memcpy(buf + (size_t)j * per, p, per);
    }
    int shp[3] = { k, t->d1, t->d2 };
    *out = mlx_array_new_data_managed(buf, shp, 3, dt, free);
    return 0;
}

/* I 9 tensori di un layer (gate/up/down × weight/scales/biases) per gli esperti `uniq[]`. */
typedef struct {
    mlx_array w[LLMENGINE_MOE_TS_NPROJ]; /* 0=gate 1=up 2=down, weight  */
    mlx_array s[LLMENGINE_MOE_TS_NPROJ]; /* scales */
    mlx_array b[LLMENGINE_MOE_TS_NPROJ]; /* biases */
} LlmMlxStreamTriples;

/* ── Ternary TurboQuant → MLX affine 2-bit RESIDENTE (GPU gather_qmm, indici GLOBALI) ───────────
 * Gli esperti tqTe sono ternary {-c,0,+c} (codebook[3]), = affine 2-bit esatto:
 * w = q·(rms·c) + (−rms·c), q∈{0,1,2}. Si ripacca trit→2-bit (16/U32 LSB) e si derivano
 * scales/biases, poi la matmul gira sul kernel quantizzato GPU di MLX. I pesi restano in dominio
 * RUOTATO (l'input va ruotato: `llm_mlx_tq_rotate_input`).
 *
 * Per (layer,proj) si materializza UNA VOLTA l'array COMPLETO [n,out,p2] (+ scales/biases), in
 * parallelo (dispatch_apply). Cosi' il hot path per-token usa gli indici GLOBALI: NIENTE
 * `unique`/remap, NIENTE `mlx_array_eval(idxs)` (sync per-layer) e NIENTE memcpy di stacking —
 * esattamente come il path affine residente (95 tok/s vs 21 dello streaming). */
static inline void llm_mlx_tq_store(void* p, size_t i, mlx_dtype dt, float v) {
    if (dt == MLX_FLOAT32) ((float*)p)[i] = v;
    else if (dt == MLX_BFLOAT16) ((uint16_t*)p)[i] = llmcodec_bf16_from_f32(v);
    else ((uint16_t*)p)[i] = llmcodec_f16_from_f32(v);
}

/* Ripacca una riga di trit (base-3, 20/U32) in 2-bit (16/U32, LSB-first come MLX). Estrazione per
 * parola a 20 trit INDIPENDENTI (tabella 3^k) invece della catena `%3 /3` seriale. */
static inline void llm_mlx_tq_repack_trit_row(const uint32_t* wrow, int in_dim, uint32_t* out_p2) {
    static const uint32_t PW[20] = { 1u,3u,9u,27u,81u,243u,729u,2187u,6561u,19683u,59049u,177147u,
                                     531441u,1594323u,4782969u,14348907u,43046721u,129140163u,387420489u,1162261467u };
    const int p2 = (in_dim + 15) >> 4;
    for (int i = 0; i < p2; i++) out_p2[i] = 0;
    const int nwords = (in_dim + 19) / 20;
    for (int wi = 0; wi < nwords; wi++) {
        const uint32_t word = wrow[wi];
        const int base = wi * 20;
        for (int k = 0; k < 20; k++) {
            const int c = base + k;
            if (c >= in_dim) break;
            out_p2[c >> 4] |= (((word / PW[k]) % 3u) << (2 * (c & 15)));
        }
    }
}

typedef struct { mlx_dtype dt; int n; mlx_array w, sc, bi; } LlmTqFull;
static LlmTqFull g_tq_full[LLMENGINE_MOE_TS_MAXLAYER][LLMENGINE_MOE_TS_NPROJ];

typedef struct { const LlmEngineMoeTsIngest* s; int layer, proj, in_dim, out, packed, ngrp, p2;
                 mlx_dtype dt; const uint16_t* cbd; uint32_t* wb; uint8_t* sb; uint8_t* bb; } LlmTqFullCtx;
static void llm_mlx_tq_full_worker(void* c, size_t i) {
    LlmTqFullCtx* x = (LlmTqFullCtx*)c;
    const int e = (int)i;
    const uint8_t* wp = llmengine_moe_ts_get(x->s, x->layer, x->proj, 0, e, NULL);
    const uint8_t* scp = llmengine_moe_ts_get(x->s, x->layer, x->proj, 1, e, NULL);
    if (!wp || !scp) return;
    const float cb0 = llmcodec_f16_to_f32(x->cbd[0]);
    const float d = 0.5f * (llmcodec_f16_to_f32(x->cbd[2]) - cb0); /* passo affine (= c per {-c,0,+c}) */
    const size_t esz = (x->dt == MLX_FLOAT32) ? 4u : 2u;
    uint32_t* wo = x->wb + (size_t)e * x->out * x->p2;
    uint8_t* so = x->sb + (size_t)e * x->out * x->ngrp * esz;
    uint8_t* bo = x->bb + (size_t)e * x->out * x->ngrp * esz;
    for (int r = 0; r < x->out; r++) {
        const uint32_t* wrow = (const uint32_t*)(const void*)wp + (size_t)r * x->packed;
        llm_mlx_tq_repack_trit_row(wrow, x->in_dim, wo + (size_t)r * x->p2);
        const uint16_t* scrow = (const uint16_t*)(const void*)scp + (size_t)r * x->ngrp;
        for (int g = 0; g < x->ngrp; g++) {
            const float rms = llmcodec_f16_to_f32(scrow[g]);
            llm_mlx_tq_store(so, (size_t)r * x->ngrp + g, x->dt, d * rms);
            llm_mlx_tq_store(bo, (size_t)r * x->ngrp + g, x->dt, cb0 * rms);
        }
    }
}

/* Costruisce (una volta per layer/proj/dtype) gli array RESIDENTI [n,out,p2]/[n,out,ngrp]. */
static inline int llm_mlx_tq_affine_full(const LlmEngineMoeTsIngest* s, int layer, int proj, mlx_dtype dt,
                                         mlx_array* w, mlx_array* sc, mlx_array* bi) {
    LlmTqFull* f = &g_tq_full[layer][proj];
    const LlmEngineMoeTsSlice* tw = llmengine_moe_ts_slice(s, layer, proj, 0);
    const LlmEngineMoeTsSlice* ts = llmengine_moe_ts_slice(s, layer, proj, 1);
    uint32_t cbb = 0; int cbn = 0;
    const uint8_t* cbp = llmengine_moe_ts_aux_get(s, layer, proj, 1, &cbb, &cbn);
    if (!tw || !ts || !cbp || cbn != 3) return -1;
    if (f->w.ctx != NULL && f->dt == dt && f->n == tw->n) { *w = f->w; *sc = f->sc; *bi = f->bi; return 0; }
    if (f->w.ctx != NULL) { mlx_array_free(f->w); mlx_array_free(f->sc); mlx_array_free(f->bi); f->w = mlx_array_new(); }
    uint32_t sgb = 0; int in_dim = 0;
    if (!llmengine_moe_ts_aux_get(s, layer, proj, 0, &sgb, &in_dim)) return -1;
    const int n = tw->n, out = tw->d1, packed = tw->d2, ngrp = ts->d2, p2 = (in_dim + 15) >> 4;
    if (n <= 0 || out <= 0 || in_dim <= 0 || ngrp <= 0 || p2 <= 0) return -1;
    const size_t esz = (dt == MLX_FLOAT32) ? 4u : 2u;
    uint32_t* wb = (uint32_t*)calloc((size_t)n * out * p2, sizeof(uint32_t));
    uint8_t* sb = (uint8_t*)malloc((size_t)n * out * ngrp * esz);
    uint8_t* bb = (uint8_t*)malloc((size_t)n * out * ngrp * esz);
    if (!wb || !sb || !bb) { free(wb); free(sb); free(bb); return -1; }
    LlmTqFullCtx ctx; ctx.s = s; ctx.layer = layer; ctx.proj = proj; ctx.in_dim = in_dim; ctx.out = out;
    ctx.packed = packed; ctx.ngrp = ngrp; ctx.p2 = p2; ctx.dt = dt;
    ctx.cbd = (const uint16_t*)(const void*)cbp; ctx.wb = wb; ctx.sb = sb; ctx.bb = bb;
#if defined(__APPLE__)
    dispatch_apply_f((size_t)n, DISPATCH_APPLY_AUTO, &ctx, llm_mlx_tq_full_worker);
#else
    for (int i = 0; i < n; i++) llm_mlx_tq_full_worker(&ctx, (size_t)i);
#endif
    int wshp[3] = { n, out, p2 }, sshp[3] = { n, out, ngrp };
    f->w  = mlx_array_new_data_managed(wb, wshp, 3, MLX_UINT32, free);
    f->sc = mlx_array_new_data_managed(sb, sshp, 3, dt, free);
    f->bi = mlx_array_new_data_managed(bb, sshp, 3, dt, free);
    f->dt = dt; f->n = n;
    *w = f->w; *sc = f->sc; *bi = f->bi;
    return 0;
}

/* ── Array PACKED residenti [E,O,PW] (trit) per il kernel qmv (niente repack affine) ────────────
 * Il tensore safetensors degli esperti e' gia' [E,O,PW] contiguo: una sola copia mmap->mlx. */
typedef struct { mlx_array w, sc, cb; int E, O, PW, NG, grp, trit, bits, cbn, built; } LlmTqPackedFull;
static LlmTqPackedFull g_tq_pk[LLMENGINE_MOE_TS_MAXLAYER][LLMENGINE_MOE_TS_NPROJ];

static inline int llm_mlx_tq_packed_full(const LlmEngineMoeTsIngest* s, int layer, int proj,
        mlx_array* w, mlx_array* sc, mlx_array* cb,
        int* E, int* O, int* PW, int* NG, int* grp, int* trit, int* bits) {
    LlmTqPackedFull* f = &g_tq_pk[layer][proj];
    const LlmEngineMoeTsSlice* tw = llmengine_moe_ts_slice(s, layer, proj, 0);
    const LlmEngineMoeTsSlice* ts = llmengine_moe_ts_slice(s, layer, proj, 1);
    uint32_t cbb = 0, sgb = 0; int cbn = 0, in_dim = 0;
    const uint8_t* cbp = llmengine_moe_ts_aux_get(s, layer, proj, 1, &cbb, &cbn);
    const uint8_t* sgp = llmengine_moe_ts_aux_get(s, layer, proj, 0, &sgb, &in_dim);
    if (!tw || !ts || !cbp || !sgp || cbn != 3) return -1;
    if (!f->built) {
        const int E_ = tw->n, O_ = tw->d1, PW_ = tw->d2, NG_ = ts->d2;
        if (E_ <= 0 || O_ <= 0 || PW_ <= 0 || NG_ <= 0 || in_dim <= 0) return -1;
        const uint8_t* w0 = llmengine_moe_ts_get(s, layer, proj, 0, 0, NULL);
        const uint8_t* s0 = llmengine_moe_ts_get(s, layer, proj, 1, 0, NULL);
        if (!w0 || !s0) return -1;
        int wshp[3] = { E_, O_, PW_ };
        int sshp[3] = { E_, O_, NG_ };
        int cshp[1] = { cbn };
        mlx_array W = mlx_array_new_data((const void*)w0, wshp, 3, MLX_UINT32); /* copia [E,O,PW] */
        mlx_array S = mlx_array_new_data((const void*)s0, sshp, 3, MLX_FLOAT16);
        mlx_array C = mlx_array_new_data((const void*)cbp, cshp, 1, MLX_FLOAT16);
        if (W.ctx == NULL || S.ctx == NULL || C.ctx == NULL) { mlx_array_free(W); mlx_array_free(S); mlx_array_free(C); return -1; }
        f->w = W; f->sc = S; f->cb = C;
        f->E = E_; f->O = O_; f->PW = PW_; f->NG = NG_;
        f->grp = in_dim / NG_; f->trit = 1; f->bits = 3; f->cbn = cbn; f->built = 1;
    }
    *w = f->w; *sc = f->sc; *cb = f->cb;
    *E = f->E; *O = f->O; *PW = f->PW; *NG = f->NG;
    *grp = f->grp; *trit = f->trit; *bits = f->bits;
    return 0;
}

/* Gather+matvec decode (batch=1): UN SOLO launch del kernel (mode kg) su [E,O,PW] + indices.
 * y[kk,row] = dequant(W[indices[kk]])·x[kk]; output [1,tk,1,O]. Niente sort/tile/eval-host. */
static inline int llm_mlx_tq_gather_qmv(mlx_array* res, mlx_array pw, mlx_array sc, mlx_array cb,
        mlx_array x_rot, mlx_array idxs, int O, int PW, int NG, int grp, int trit, int bits, mlx_stream st) {
    const int ind = (int)mlx_array_ndim(idxs);
    const int tk = (int)mlx_array_dim(idxs, ind - 1);
    const int xnd = (int)mlx_array_ndim(x_rot);
    const int rows = (xnd >= 2) ? (int)mlx_array_dim(x_rot, 1) : 1;
    const int in = (int)mlx_array_dim(x_rot, xnd - 1);
    if (tk <= 0 || in <= 0) return -1;
    /* x -> [tk,in] contiguo (broadcast se gate/up: stessa x per tutti i k). */
    int shp2[2] = { rows, in };
    mlx_array x2 = mlx_array_new();
    if (mlx_reshape(&x2, x_rot, shp2, 2, st) != 0) { mlx_array_free(x2); return -1; }
    mlx_array xb = mlx_array_new();
    if (rows == tk) {
        if (mlx_contiguous(&xb, x2, false, st) != 0) { mlx_array_free(x2); mlx_array_free(xb); return -1; }
    } else {
        int bsh[2] = { tk, in };
        mlx_array xbc = mlx_array_new();
        if (mlx_broadcast_to(&xbc, x2, bsh, 2, st) != 0) { mlx_array_free(x2); mlx_array_free(xbc); return -1; }
        if (mlx_contiguous(&xb, xbc, false, st) != 0) { mlx_array_free(x2); mlx_array_free(xbc); mlx_array_free(xb); return -1; }
        mlx_array_free(xbc);
    }
    mlx_array_free(x2);
    int ish[1] = { tk };
    mlx_array idxf = mlx_array_new();
    if (mlx_reshape(&idxf, idxs, ish, 1, st) != 0) { mlx_array_free(xb); mlx_array_free(idxf); return -1; }
    mlx_array y2 = mlx_array_new();
    int rc = llm_mlx_tq_qmv_kg(&y2, pw, sc, cb, xb, idxf, tk, in, O, NG, PW, bits, grp, trit, st);
    mlx_array_free(xb); mlx_array_free(idxf);
    if (rc != 0) { mlx_array_free(y2); return -1; }
    int osh[4] = { 1, tk, 1, O };
    mlx_array out = mlx_array_new();
    if (mlx_reshape(&out, y2, osh, 4, st) != 0) { mlx_array_free(y2); mlx_array_free(out); return -1; }
    mlx_array_free(y2);
    mlx_array_set(res, out);
    mlx_array_free(out);
    return 0;
}

/* Rotazione ONLINE dell'input: x_rot = Hadamard(x · signs) (dominio dei pesi, non li srotola).
 * Usa `mlx_hadamard_transform` (stessa op della libreria di riferimento). Cast finale a `cdt`
 * per far combaciare x_rot con scales/biases di gather_qmm. Il vettore signs (F32) e' costruito
 * una volta per (layer,proj) e messo in cache: era ricreato a ogni chiamata (960x/token). */
static mlx_array g_tq_sign[LLMENGINE_MOE_TS_MAXLAYER][LLMENGINE_MOE_TS_NPROJ];
static inline int llm_mlx_tq_rotate_input(const LlmEngineMoeTsIngest* s, int layer, int proj,
                                          mlx_array x, mlx_stream st, mlx_dtype cdt, mlx_array* out) {
    uint32_t sgb = 0; int sn = 0;
    const uint8_t* sp = llmengine_moe_ts_aux_get(s, layer, proj, 0, &sgb, &sn);
    if (!sp || sn <= 0) return -1;
    mlx_array signs = g_tq_sign[layer][proj];
    if (signs.ctx == NULL) {
        float* sf = (float*)malloc((size_t)sn * sizeof(float));
        if (!sf) return -1;
        const uint16_t* sgd = (const uint16_t*)(const void*)sp;
        for (int i = 0; i < sn; i++) sf[i] = llmcodec_f16_to_f32(sgd[i]);
        int shp[1] = { sn };
        signs = mlx_array_new_data(sf, shp, 1, MLX_FLOAT32); /* copia */
        free(sf);
        if (signs.ctx == NULL) return -1;
        g_tq_sign[layer][proj] = signs;
    }
    mlx_array mul = mlx_array_new();
    int r = mlx_multiply(&mul, x, signs, st);
    if (r != 0) { mlx_array_free(mul); return -1; }
    mlx_optional_float scale = { 1.0f / sqrtf((float)sn), true };
    mlx_array h = mlx_array_new();
    r = mlx_hadamard_transform(&h, mul, scale, st);
    mlx_array_free(mul);
    if (r != 0) { mlx_array_free(h); return -1; }
    if (mlx_array_dtype(h) == cdt) { *out = h; return 0; }
    r = mlx_astype(out, h, cdt, st);
    mlx_array_free(h);
    return r == 0 ? 0 : -1;
}

/* Self-test ISOLATO del kernel mode 1 (`kg`, gather multi-esperto): y[kk,row] deve coincidere col
 * decode di riferimento dell'esperto indices[kk]. Env LLM_TQ_KG_TEST=1. Isola il kernel dal resto. */
static inline void llm_mlx_tq_kg_selftest(const LlmEngineMoeTsIngest* s, int layer, int proj, mlx_stream st) {
    static int done[8] = { 0 };
    if (proj < 0 || proj >= 8 || done[proj]) return;
    done[proj] = 1;
    const LlmEngineMoeTsSlice* tw = llmengine_moe_ts_slice(s, layer, proj, 0);
    const LlmEngineMoeTsSlice* ts = llmengine_moe_ts_slice(s, layer, proj, 1);
    uint32_t cbb = 0, sgb = 0; int cbn = 0, in_dim = 0;
    const uint8_t* cbp = llmengine_moe_ts_aux_get(s, layer, proj, 1, &cbb, &cbn);
    const uint8_t* sgp = llmengine_moe_ts_aux_get(s, layer, proj, 0, &sgb, &in_dim);
    if (!tw || !ts || !cbp || !sgp || in_dim <= 0) { fprintf(stderr, "[tq-kg-test] dati mancanti\n"); return; }
    const int E = tw->n, O = tw->d1, PW = tw->d2, NG = ts->d2;
    const int trit = (cbn == 3);
    const int bits = trit ? 3 : (cbn == 8 ? 3 : (cbn == 4 ? 2 : 4));
    const int grp = (NG > 0) ? in_dim / NG : 64;
    const int tk = (E >= 2) ? 2 : 1;
    const uint8_t* w0 = llmengine_moe_ts_get(s, layer, proj, 0, 0, NULL);
    const uint8_t* s0 = llmengine_moe_ts_get(s, layer, proj, 1, 0, NULL);
    if (!w0 || !s0) { fprintf(stderr, "[tq-kg-test] base mancante\n"); return; }
    int wsh[3] = { E, O, PW }, ssh[3] = { E, O, NG }, csh[1] = { cbn };
    mlx_array W = mlx_array_new_data((const void*)w0, wsh, 3, MLX_UINT32);
    mlx_array S = mlx_array_new_data((const void*)s0, ssh, 3, MLX_FLOAT16);
    mlx_array CB = mlx_array_new_data((const void*)cbp, csh, 1, MLX_FLOAT16);
    float* x0 = (float*)malloc((size_t)tk * in_dim * sizeof(float));
    for (int kk = 0; kk < tk; kk++) for (int i = 0; i < in_dim; i++)
        x0[kk * in_dim + i] = ((float)(((unsigned)(kk * 7919 + i) * 2654435761u) >> 9 & 0x1ffu) / 256.0f) - 1.0f;
    int xsh[2] = { tk, in_dim };
    mlx_array X0 = mlx_array_new_data(x0, xsh, 2, MLX_FLOAT32);
    uint32_t* idb = (uint32_t*)malloc((size_t)tk * sizeof(uint32_t));
    for (int kk = 0; kk < tk; kk++) idb[kk] = (uint32_t)(kk % E);
    int ish[1] = { tk };
    mlx_array IDX = mlx_array_new_data(idb, ish, 1, MLX_UINT32);
    mlx_array Xr = mlx_array_new(), Y = mlx_array_new();
    int rc = llm_mlx_tq_rotate_input(s, layer, proj, X0, st, MLX_FLOAT32, &Xr);
    if (rc == 0) rc = llm_mlx_tq_qmv_kg(&Y, W, S, CB, Xr, IDX, tk, in_dim, O, NG, PW, bits, grp, trit, st);
    if (rc != 0 || mlx_array_eval(Y) != 0 || (int)mlx_array_size(Y) != tk * O) {
        fprintf(stderr, "[tq-kg-test] kernel fallito rc=%d size=%d\n", rc, (int)mlx_array_size(Y));
    } else {
        float cbf[16];
        for (int i = 0; i < cbn; i++) cbf[i] = llmcodec_f16_to_f32(((const uint16_t*)(const void*)cbp)[i]);
        float* sgf = (float*)malloc((size_t)in_dim * sizeof(float));
        for (int i = 0; i < in_dim; i++) sgf[i] = llmcodec_f16_to_f32(((const uint16_t*)(const void*)sgp)[i]);
        float* f = (float*)malloc((size_t)in_dim * sizeof(float));
        const float* yv = mlx_array_data_float32(Y);
        double maxd = 0, ref2 = 0;
        for (int kk = 0; kk < tk; kk++) {
            const uint32_t* we = (const uint32_t*)(const void*)w0 + (size_t)idb[kk] * O * PW;
            const uint16_t* se = (const uint16_t*)(const void*)s0 + (size_t)idb[kk] * O * NG;
            for (int r = 0; r < O; r++) {
                llm_mlx_tq_decode_row(we + (size_t)r * PW, se + (size_t)r * NG, cbf, sgf, in_dim, NG, trit, bits, f);
                double acc = 0; for (int c = 0; c < in_dim; c++) acc += (double)f[c] * x0[kk * in_dim + c];
                double d = fabs(acc - (double)yv[kk * O + r]); if (d > maxd) maxd = d; ref2 += acc * acc;
            }
        }
        fprintf(stderr, "[tq-kg-test] layer=%d proj=%d E=%d O=%d in=%d trit=%d grp=%d max_abs_diff=%.6g rms_ref=%.4g\n",
                layer, proj, E, O, in_dim, trit, grp, maxd, sqrt(ref2 / (tk * O)));
        free(sgf); free(f);
    }
    mlx_array_free(W); mlx_array_free(S); mlx_array_free(CB); mlx_array_free(X0); mlx_array_free(IDX);
    mlx_array_free(Xr); mlx_array_free(Y); free(x0); free(idb);
}

/* Stack TurboQuant degli esperti `uniq[]` per (layer,proj): decodifica on-the-fly da packed
 * (indici→codebook→scala→Hadamard inversa→signs) nel dominio NON ruotato, F16 [k, out, in].
 * Riusa il decode CONDIVISO (`llm_mlx_tq_decode.h`, stessa sorgente del path residente). */
static inline int llm_mlx_tq_stack(const LlmEngineMoeTsIngest* s, int layer, int proj,
                                   const int* uniq, int k, mlx_stream str, mlx_array* out) {
    (void)str;
    if (k <= 0 || !out) return -1;
    const LlmEngineMoeTsSlice* tw = llmengine_moe_ts_slice(s, layer, proj, 0); /* weight U32 */
    const LlmEngineMoeTsSlice* ts = llmengine_moe_ts_slice(s, layer, proj, 1); /* scales F16 */
    uint32_t sgb = 0, cbb = 0; int sn = 0, cbn = 0;
    const uint8_t* sgp = llmengine_moe_ts_aux_get(s, layer, proj, 0, &sgb, &sn); /* signs [in] */
    const uint8_t* cbp = llmengine_moe_ts_aux_get(s, layer, proj, 1, &cbb, &cbn); /* codebook */
    if (!tw || !ts || !sgp || !cbp) return -1;
    const int out_dim = tw->d1, in_dim = sn, ngrp = ts->d2;
    if (out_dim <= 0 || in_dim <= 0 || ngrp <= 0 || ts->d1 != out_dim) return -1;
    if (!llm_mlx_tq_pow2(in_dim)) return -1;   /* come il path residente: solo dim pow2 */
    const int trit = (cbn == 3);
    const int bits = trit ? 0 : (cbn == 8 ? 3 : (cbn == 4 ? 2 : 4));
    const uint16_t* sgd = (const uint16_t*)(const void*)sgp;
    const uint16_t* cbd = (const uint16_t*)(const void*)cbp;
    /* Hoist F32 di codebook (ncode) e signs (in_dim), condivisi da tutti gli esperti del proj:
     * evita la conversione f16->f32 per-elemento nel hot path del decode. */
    float cbf[16];
    if (cbn < 1 || cbn > 16) return -1;
    for (int i = 0; i < cbn; i++) cbf[i] = llmcodec_f16_to_f32(cbd[i]);
    float* sgf = (float*)malloc((size_t)in_dim * sizeof(float));
    if (!sgf) return -1;
    for (int i = 0; i < in_dim; i++) sgf[i] = llmcodec_f16_to_f32(sgd[i]);
    const size_t per = (size_t)out_dim * (size_t)in_dim;
    uint16_t* buf = (uint16_t*)malloc(per * (size_t)k * sizeof(uint16_t));
    float* f = (float*)malloc(per * sizeof(float));
    if (!buf || !f) { free(buf); free(f); free(sgf); return -1; }
    for (int j = 0; j < k; j++) {
        const uint8_t* wp = llmengine_moe_ts_get(s, layer, proj, 0, uniq[j], NULL);
        const uint8_t* scp = llmengine_moe_ts_get(s, layer, proj, 1, uniq[j], NULL);
        if (!wp || !scp) { free(buf); free(f); free(sgf); return -1; }
        const uint32_t* wd = (const uint32_t*)(const void*)wp;
        const uint16_t* scd = (const uint16_t*)(const void*)scp;
        for (int r = 0; r < out_dim; r++) {
            llm_mlx_tq_decode_row(wd + (size_t)r * tw->d2, scd + (size_t)r * ngrp,
                                  cbf, sgf, in_dim, ngrp, trit, bits, f + (size_t)r * in_dim);
        }
        uint16_t* dst = buf + (size_t)j * per;
        for (size_t q = 0; q < per; q++) dst[q] = llmcodec_f16_from_f32(f[q]);
    }
    free(f); free(sgf);
    int shp[3] = { k, out_dim, in_dim };
    *out = mlx_array_new_data_managed(buf, shp, 3, MLX_FLOAT16, free);
    return 0;
}

/* Self-test (env LLM_TQ_QMV_TEST=1, una volta): il kernel polar_qmv deve coincidere col decode di
 * riferimento nel dominio ruotato — kernel(W_rot, x_rot) == dot(unrotate(W_rot), x0). */
static int g_tq_qmv_tested = 0;
static inline void llm_mlx_tq_qmv_selftest(const LlmEngineMoeTsIngest* s, int layer, int proj, mlx_stream st) {
    if (g_tq_qmv_tested) return;
    g_tq_qmv_tested = 1;
    const LlmEngineMoeTsSlice* tw = llmengine_moe_ts_slice(s, layer, proj, 0);
    const LlmEngineMoeTsSlice* ts = llmengine_moe_ts_slice(s, layer, proj, 1);
    uint32_t cbb = 0, sgb = 0; int cbn = 0, in_dim = 0;
    const uint8_t* cbp = llmengine_moe_ts_aux_get(s, layer, proj, 1, &cbb, &cbn);
    const uint8_t* sgp = llmengine_moe_ts_aux_get(s, layer, proj, 0, &sgb, &in_dim);
    if (!tw || !ts || !cbp || !sgp || in_dim <= 0) { fprintf(stderr, "[tq-qmv-test] dati mancanti\n"); return; }
    const int O = tw->d1, packed = tw->d2, ngrp = ts->d2;
    const int trit = (cbn == 3);
    const int bits = trit ? 3 : (cbn == 8 ? 3 : (cbn == 4 ? 2 : 4));
    const int group = (ngrp > 0) ? in_dim / ngrp : 64;
    const uint8_t* wp = llmengine_moe_ts_get(s, layer, proj, 0, 0, NULL);
    const uint8_t* scp = llmengine_moe_ts_get(s, layer, proj, 1, 0, NULL);
    if (!wp || !scp) { fprintf(stderr, "[tq-qmv-test] expert mancante\n"); return; }
    int wshp[3] = { 1, O, packed }, sshp[3] = { 1, O, ngrp }, cshp[1] = { cbn };
    mlx_array W = mlx_array_new_data((const void*)wp, wshp, 3, MLX_UINT32);   /* copia */
    mlx_array S = mlx_array_new_data((const void*)scp, sshp, 3, MLX_FLOAT16);
    mlx_array C = mlx_array_new_data((const void*)cbp, cshp, 1, MLX_FLOAT16);
    float* x0 = (float*)malloc((size_t)in_dim * sizeof(float));
    for (int i = 0; i < in_dim; i++) x0[i] = ((float)(((unsigned)i * 2654435761u) >> 9 & 0x1ffu) / 256.0f) - 1.0f;
    int xshp[1] = { in_dim };
    mlx_array X0 = mlx_array_new_data(x0, xshp, 1, MLX_FLOAT32);
    mlx_array Xrot = mlx_array_new();
    mlx_array Y = mlx_array_new();
    if (llm_mlx_tq_rotate_input(s, layer, proj, X0, st, MLX_FLOAT32, &Xrot) != 0) {
        fprintf(stderr, "[tq-qmv-test] rotate fallita\n");
    } else if (llm_mlx_tq_qmv(&Y, W, S, C, Xrot, /*e=*/0, in_dim, O, ngrp, packed, bits, group, trit, st) != 0) {
        fprintf(stderr, "[tq-qmv-test] kernel apply fallita\n");
    } else if (mlx_array_eval(Y) != 0 || (int)mlx_array_size(Y) != O) {
        fprintf(stderr, "[tq-qmv-test] eval/output inattesi\n");
    } else {
        float cbf[16];
        for (int i = 0; i < cbn; i++) cbf[i] = llmcodec_f16_to_f32(((const uint16_t*)(const void*)cbp)[i]);
        float* sgf = (float*)malloc((size_t)in_dim * sizeof(float));
        float* f = (float*)malloc((size_t)in_dim * sizeof(float));
        for (int i = 0; i < in_dim; i++) sgf[i] = llmcodec_f16_to_f32(((const uint16_t*)(const void*)sgp)[i]);
        const float* yv = mlx_array_data_float32(Y);
        double maxd = 0, ref2 = 0;
        for (int r = 0; r < O; r++) {
            const uint32_t* wrow = (const uint32_t*)(const void*)wp + (size_t)r * packed;
            const uint16_t* scrow = (const uint16_t*)(const void*)scp + (size_t)r * ngrp;
            llm_mlx_tq_decode_row(wrow, scrow, cbf, sgf, in_dim, ngrp, trit, bits, f);
            double acc = 0; for (int c = 0; c < in_dim; c++) acc += (double)f[c] * x0[c];
            double d = fabs(acc - (double)yv[r]); if (d > maxd) maxd = d; ref2 += acc * acc;
        }
        fprintf(stderr, "[tq-qmv-test] layer=%d proj=%d O=%d in=%d trit=%d group=%d cbn=%d max_abs_diff=%.6g rms_ref=%.4g\n",
                layer, proj, O, in_dim, trit, group, cbn, maxd, sqrt(ref2 / (O > 0 ? O : 1)));
        free(sgf); free(f);
    }
    mlx_array_free(W); mlx_array_free(S); mlx_array_free(C); mlx_array_free(X0); mlx_array_free(Xrot);
    mlx_array_free(Y); free(x0);
}

/* Prepara lo stacked [k,d1,d2] del SOLO `proj` richiesto (w/scales/biases). Prima si impilavano
 * TUTTI e 3 i proj e il chiamante ne usava uno: 6 stack su 9 sprecati a ogni chiamata (2/3 del
 * lavoro e altrettanti sync). 0 ok / -1. */
static inline int llm_mlx_moe_swap_prepare(const LlmEngineMoeTsIngest* ing, int layer, int proj,
                                           const int* uniq, int k, mlx_stream str,
                                           LlmMlxStreamTriples* out) {
    if (proj < 0 || proj >= LLMENGINE_MOE_TS_NPROJ) return -1;
    for (int p = 0; p < LLMENGINE_MOE_TS_NPROJ; p++) {
        out->w[p] = mlx_array_new();
        out->s[p] = mlx_array_new();
        out->b[p] = mlx_array_new();
    }
    /* TurboQuant NON-ternary (tq 3/4-bit): decode F16 in dominio NON ruotato + ramo PLAIN
     * gather_mm (scales/biases vuoti). Il tq TERNARY non passa di qui: usa array pieni residenti
     * con indici globali, gestiti direttamente in llm_mlx_moe_gather_streamed. */
    if (llmengine_moe_ts_is_tq(ing, layer, proj)) {
        if (llm_mlx_tq_stack(ing, layer, proj, uniq, k, str, &out->w[proj]) != 0) return -1;
        return 0;
    }
    for (int kd = 0; kd < LLMENGINE_MOE_TS_NKIND; kd++) {
        mlx_array a = mlx_array_new();
        if (llm_mlx_xs_stack(ing, layer, proj, kd, uniq, k, str, &a) != 0) return -1;
        if (kd == 0) out->w[proj] = a; else if (kd == 1) out->s[proj] = a; else out->b[proj] = a;
    }
    return 0;
}

#endif /* LLM_MLX_MOE_STREAM_BUILD_H */
