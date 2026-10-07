/* llm_mlx_moe_stream_build.h — costruisce i tensori STACKED [k, d1, d2] dei k esperti unici
 * a partire dalle SLICE dell'ingest canonico safetensors (`llmengine_moe_safetensors_ingest.h`).
 *
 * NON un formato nuovo: il descrittore `(layer,expert,proj,kind) -> slice` e' UNO (l'ingest);
 * qui c'e' solo il consumo lato MLX (costruzione degli mlx_array per gather_qmm). I bundle sono
 * copiati in buffer RESIDENTI da un POOL riusabile e `mlock`-ato (dtor di MLX = riciclo) -> Metal
 * non fa fault e niente churn di malloc/free nel hot path. Il routing predice il set del prossimo
 * token per il prefetch (`MADV_WILLNEED`) e, sotto pressione, evince i sorgenti freddi. */
#ifndef LLM_MLX_MOE_STREAM_BUILD_H
#define LLM_MLX_MOE_STREAM_BUILD_H
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "../moe/include/llmengine_moe_safetensors_ingest.h"
#include "llm_mlx_tq_decode.h"
#include "llm_mlx_tq_qmv_kernel.h"
#include "llm_mlx_memgov.h"   /* pressure-gate per l'eviction predittiva (governor) */
#if defined(__APPLE__)
#include <dispatch/dispatch.h>   /* popolamento parallelo della cache esperto->affine-2bit */
#endif

#include "llm_mlx_xs_pool.h"   /* pool riusabile + mlock dei bundle streamed */

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
    const size_t need = per * (size_t)k;
    /* UN buffer contiguo [k,d1,d2] + MANAGED (MLX prende ownership): una sola copia mmap->buffer. Il
     * buffer è RICICLATO dal pool (diversamente da prima: un malloc per chiamata); se il pool è pieno
     * si ricade sul malloc singolo con dtor `free`. */
    LlmXsSlot* sl = NULL;
    uint8_t* buf = llm_xs_pool_acquire(need, &sl);
    if (buf == NULL) { buf = (uint8_t*)malloc(need); sl = NULL; }
    if (buf == NULL) return -1;
    for (int j = 0; j < k; j++) {
        const uint8_t* p = llmengine_moe_ts_get(s, layer, proj, kind, uniq[j], NULL);
        if (!p) { if (sl != NULL) llm_xs_slot_dtor(sl); else free(buf); return -1; }
        memcpy(buf + (size_t)j * per, p, per);
    }
    int shp[3] = { k, t->d1, t->d2 };
    if (sl != NULL)
        *out = mlx_array_new_data_managed_payload(buf, shp, 3, dt, sl, llm_xs_slot_dtor);
    else
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

/* Esperti tq PACKED diretti (opzione memoria): usa i `w`/`scales` packed + codebook/signs dall'ingest
 * e UN launch del kernel `kg`. Output [M,1,O] (come il gather_qmm affine: routing appiattiti a M).
 * Elimina gli array affine-2bit residenti (~11.5 GB sul 35B). */
static inline int llm_mlx_tq_gather_packed(const LlmEngineMoeTsIngest* s, int layer, int proj,
        mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array idxs,
        bool transpose, mlx_stream st) {
    if (!transpose) return -1;
    const int wnd = (int)mlx_array_ndim(w);
    const int O = (int)mlx_array_shape(w)[wnd >= 2 ? wnd - 2 : 0];
    const int PW = (int)mlx_array_shape(w)[wnd - 1];
    const int snd = (int)mlx_array_ndim(scales);
    const int NG = (int)mlx_array_shape(scales)[snd - 1];
    const int xnd = (int)mlx_array_ndim(x);
    const int K = (int)mlx_array_shape(x)[xnd - 1];
    const int M = (int)mlx_array_size(idxs);                  /* idxs flat [M] = n_tok*top_k */
    const int T = (xnd >= 1) ? (int)mlx_array_dim(x, 0) : M;  /* n_tokens da x */
    uint32_t cbb = 0; int cbn = 0;
    const uint8_t* cbp = llmengine_moe_ts_aux_get(s, layer, proj, 1, &cbb, &cbn);
    if (!cbp || cbn < 1 || M <= 0 || K <= 0 || O <= 0 || NG <= 0) return -1;
    const int trit = (cbn == 3);
    const int bits = trit ? 3 : (cbn == 8 ? 3 : (cbn == 4 ? 2 : 4));
    const int grp = (K % NG == 0) ? K / NG : 64;
    int csh[1] = { cbn };
    mlx_array CB = mlx_array_new_data((const void*)cbp, csh, 1, MLX_FLOAT16);
    const mlx_dtype cdt = mlx_array_dtype(x);
    mlx_array xr = mlx_array_new();
    if (llm_mlx_tq_rotate_input(s, layer, proj, x, st, cdt, &xr) != 0) {
        mlx_array_free(CB); return -1;
    }
    mlx_array xm = mlx_array_new();
    int r = 0;
    const int rows_x = (int)(mlx_array_size(xr) / (size_t)K);
    int s2[2] = { M, K };
    if (rows_x == T && M > T) {
        int tk = M / T;
        int s3[3] = { T, 1, K }, b3[3] = { T, tk, K };
        mlx_array x3 = mlx_array_new(), xb = mlx_array_new();
        if (mlx_reshape(&x3, xr, s3, 3, st) != 0) r = -1;
        if (r == 0 && mlx_broadcast_to(&xb, x3, b3, 3, st) != 0) r = -1;
        if (r == 0 && mlx_reshape(&xm, xb, s2, 2, st) != 0) r = -1;
        mlx_array_free(x3); mlx_array_free(xb);
    } else {
        if (mlx_reshape(&xm, xr, s2, 2, st) != 0) r = -1;
    }
    mlx_array_free(xr);
    if (r != 0) {
        mlx_array_free(CB); mlx_array_free(xm); return -1;
    }
    mlx_array y = mlx_array_new();
    r = llm_mlx_tq_qmv_kg(&y, w, scales, CB, xm, idxs, M, K, O, NG, PW, bits, grp, trit, st);
    mlx_array_free(CB); mlx_array_free(xm);
    if (r != 0) {
        mlx_array_free(y); return -1;
    }
    int osh[3] = { M, 1, O };
    mlx_array out = mlx_array_new();
    if (mlx_reshape(&out, y, osh, 3, st) != 0) {
        mlx_array_free(y); mlx_array_free(out); return -1;
    }
    mlx_array_free(y);
    mlx_array_set(res, out); mlx_array_free(out);
    return 0;
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
    const size_t bneed = per * (size_t)k * sizeof(uint16_t);
    LlmXsSlot* bsl = NULL;
    uint16_t* buf = (uint16_t*)llm_xs_pool_acquire(bneed, &bsl);
    if (buf == NULL) { buf = (uint16_t*)malloc(bneed); bsl = NULL; }
    float* f = (float*)malloc(per * sizeof(float));
    if (!buf || !f) { if (bsl) llm_xs_slot_dtor(bsl); else free(buf); free(f); free(sgf); return -1; }
    for (int j = 0; j < k; j++) {
        const uint8_t* wp = llmengine_moe_ts_get(s, layer, proj, 0, uniq[j], NULL);
        const uint8_t* scp = llmengine_moe_ts_get(s, layer, proj, 1, uniq[j], NULL);
        if (!wp || !scp) { if (bsl) llm_xs_slot_dtor(bsl); else free(buf); free(f); free(sgf); return -1; }
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
    if (bsl != NULL)
        *out = mlx_array_new_data_managed_payload(buf, shp, 3, MLX_FLOAT16, bsl, llm_xs_slot_dtor);
    else
        *out = mlx_array_new_data_managed(buf, shp, 3, MLX_FLOAT16, free);
    return 0;
}

/* Self-test (env LLM_TQ_QMV_TEST=1, una volta): il kernel polar_qmv deve coincidere col decode di
 * riferimento nel dominio ruotato — kernel(W_rot, x_rot) == dot(unrotate(W_rot), x0). */
static int g_tq_qmv_tested = 0;

/* ── PREFETCH / EVICT predittivo (streaming esperti) ─────────────────────────────────────────────
 * Il routing MoE è temporalmente STABILE (stessa conversazione -> stessi esperti): il set del token
 * precedente per (layer,proj) predice bene il prossimo. Uso:
 *   - prefetch (MADV_WILLNEED) del set predetto all'inizio di ogni prepare -> maschera la latenza di
 *     ri-lettura quando le pagine erano state evincite;
 *   - sotto PRESSIONE (governor) si rilasciano (MADV_DONTNEED) le pagine SORGENTE del set corrente che
 *     NON erano nel set predetto: l'RSS resta limitato agli esperti "hot" (riusati), i freddi si
 *     ri-leggono. Fuori pressione non si evince (si lasciano calde per il riuso). Hint puri. */
#define LLM_XS_PRED_MAXK 32
static int32_t g_xs_pred[LLMENGINE_MOE_TS_MAXLAYER][LLMENGINE_MOE_TS_NPROJ][LLM_XS_PRED_MAXK];
static int g_xs_pred_n[LLMENGINE_MOE_TS_MAXLAYER][LLMENGINE_MOE_TS_NPROJ];

static inline void llm_xs_prefetch_predicted(const LlmEngineMoeTsIngest* ing, int layer, int proj) {
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER) return;
    const int n = g_xs_pred_n[layer][proj];
    for (int i = 0; i < n; i++)
        llmengine_moe_ts_advise_expert(ing, layer, proj, (int)g_xs_pred[layer][proj][i], MADV_WILLNEED);
}

static inline void llm_xs_record_predicted(int layer, int proj, const int* uniq, int k) {
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER) return;
    if (k > LLM_XS_PRED_MAXK) k = LLM_XS_PRED_MAXK;
    for (int i = 0; i < k; i++) g_xs_pred[layer][proj][i] = (int32_t)uniq[i];
    g_xs_pred_n[layer][proj] = k;
}

static inline int llm_xs_in_pred(int layer, int proj, int e) {
    const int n = g_xs_pred_n[layer][proj];
    for (int i = 0; i < n; i++) if (g_xs_pred[layer][proj][i] == e) return 1;
    return 0;
}

static inline void llm_xs_evict_cold(const LlmEngineMoeTsIngest* ing, int layer, int proj,
                                     const int* uniq, int k) {
    if (ing == NULL || layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER) return;
    LlmMemGov g = llm_memgov_from_registry();
    if (!llm_memgov_should_spill(&g, llm_memgov_system_free_bytes())) return;
    for (int i = 0; i < k; i++)
        if (!llm_xs_in_pred(layer, proj, uniq[i]))
            llmengine_moe_ts_advise_expert(ing, layer, proj, uniq[i], MADV_DONTNEED);
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
    llm_xs_prefetch_predicted(ing, layer, proj);   /* hint puro, nessun cambio di valori */
    /* TurboQuant NON-ternary (tq 3/4-bit): decode F16 in dominio NON ruotato + ramo PLAIN
     * gather_mm (scales/biases vuoti). Il tq TERNARY non passa di qui: usa array pieni residenti
     * con indici globali, gestiti direttamente in llm_mlx_moe_gather_streamed. */
    if (llmengine_moe_ts_is_tq(ing, layer, proj)) {
        if (llm_mlx_tq_stack(ing, layer, proj, uniq, k, str, &out->w[proj]) != 0) return -1;
    } else {
        for (int kd = 0; kd < LLMENGINE_MOE_TS_NKIND; kd++) {
            mlx_array a = mlx_array_new();
            if (llm_mlx_xs_stack(ing, layer, proj, kd, uniq, k, str, &a) != 0) return -1;
            if (kd == 0) out->w[proj] = a; else if (kd == 1) out->s[proj] = a; else out->b[proj] = a;
        }
    }
    /* Sotto pressione: rilascia le pagine SORGENTE del set corrente non-predetto (gli hot restano
     * caldi). Poi aggiorna il predetto col set corrente. */
    llm_xs_evict_cold(ing, layer, proj, uniq, k);
    llm_xs_record_predicted(layer, proj, uniq, k);
    return 0;
}

#endif /* LLM_MLX_MOE_STREAM_BUILD_H */
