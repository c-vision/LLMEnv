/* ============================================================================
 * llm_mlx_selftest_eval.c — self-test unitari/bench model-free (KV view, attn bw, mask cache, oQ A8, sliding, drift SDPA/forma).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"   /* llm_mlx_gqa_sdpa */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* llm_ai_mlx_mask_cache_selftest (2026-09-16) — self-test UNITARIO della cache della mask
 * sliding (P0-bis), SENZA modello. Va chiamato DENTRO la dylib perche' MLX va inizializzato
 * (device+stream) come in load_model: un eseguibile esterno che chiama le API MLX direttamente
 * da' SIGBUS (verificato). Costruisce q/k/v sintetici e invoca llm_mlx_gqa_sdpa con window>0,
 * poi scrive in `out` un hash dell'output. Confrontando gli hash di due processi con
 * LLM_MASK_CACHE=0 (percorso non cached) e senza (percorso cached, se la mask sta sotto il cap)
 * si verifica che i due percorsi diano lo STESSO risultato. Ritorna 0 su successo. */
/* SELF-TEST UNITARIO (2026-09-17) — KV view vs slice, SENZA MODELLO.
 * Premessa strutturale del fix LLM_KV_STRIDED: in questa C API `mlx_slice` MATERIALIZZA (alloca una
 * copia), `mlx_as_strided` no (vista zero-copy sul buffer del parent). Verifica anche che i contenuti
 * della vista coincidano con quelli della slice (strides corretti).
 * Driver: `llm_kv_view_test <nkv> <cap> <need> <hd>` (usa f32: il test e' sull'ALLOCAZIONE). */
/* SELF-TEST UNITARIO (2026-09-17, esteso) — strumenti KV, SENZA MODELLO.
 * Tre controlli deterministici (nessun modello, nessun carico: si eseguono SEMPRE dopo una modifica
 * all'engine che tocchi la KV):
 *  C1 vista-vs-copia: `mlx_slice` NON deve allocare (e' una vista strided), `mlx_as_strided` nemmeno;
 *     i contenuti dei due percorsi devono coincidere (strides corretti).
 *  C2 write in-place: `mlx_slice_update_inplace` deve MODIFICARE il parent (non una copia) e non
 *     allocare il buffer: si verifica leggendo indietro il parent nella regione scritta.
 *  C3 pattern di decode: N append (update in-place nel buffer `cap`) + lettura della vista `need`
 *     NON devono far crescere la memoria attiva (era l'ipotesi "copiamo la KV per token": qui si
 *     misura direttamente sul pattern reale, a costo zero e senza modello).
 * Driver: `llm_kv_view_test <nkv> <cap> <need> <hd> [iters]`. */
int llm_ai_mlx_kv_view_selftest(int nkv, int cap, int need, int hd, char* out, int cap_out) {
    if (nkv <= 0 || cap <= 0 || need <= 0 || need > cap || hd <= 0) {
        if (out && cap_out > 0) snprintf(out, (size_t)cap_out, "FAIL: parametri non validi");
        return -1;
    }
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL) { if (out && cap_out > 0) snprintf(out, (size_t)cap_out, "FAIL: device GPU"); return -1; }
    if (mlx_set_default_device(dev) != 0) { mlx_device_free(dev); if (out && cap_out > 0) snprintf(out, (size_t)cap_out, "FAIL: set_default_device"); return -1; }
    mlx_stream str = mlx_stream_new();
    if (mlx_get_default_stream(&str, dev) != 0 || str.ctx == NULL) { mlx_device_free(dev); if (out && cap_out > 0) snprintf(out, (size_t)cap_out, "FAIL: default_stream"); return -1; }
    const int sh[4] = {1, nkv, cap, hd};
    const size_t n = (size_t)nkv * (size_t)cap * (size_t)hd;
    float* host = (float*)malloc(n * sizeof(float));
    if (host == NULL) { mlx_stream_free(str); mlx_device_free(dev); if (out) snprintf(out, (size_t)cap_out, "FAIL: malloc"); return -1; }
    for (size_t i = 0; i < n; i++) host[i] = (float)(i % 1024) * 0.25f;
    mlx_array parent = mlx_array_new_data(host, sh, 4, MLX_FLOAT32);
    mlx_array sliced = mlx_array_new(), view = mlx_array_new(), eq = mlx_array_new();
    size_t m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0;
    bool ok = parent.ctx != NULL;
    /* ---- C1: vista-vs-copia ---- */
    (void)mlx_get_active_memory(&m0);
    const int p0[4] = {0,0,0,0}, ps[4] = {1,1,1,1}, pk[4] = {1,nkv,need,hd};
    if (ok) ok = mlx_slice(&sliced, parent, p0, 4, pk, 4, ps, 4, str) == 0 && mlx_array_eval(sliced) == 0;
    (void)mlx_get_active_memory(&m1);
    const int64_t st[4] = {(int64_t)nkv * cap * hd, (int64_t)cap * hd, (int64_t)hd, 1};
    int vsh[4] = {1, nkv, need, hd};
    if (ok) ok = mlx_as_strided(&view, parent, vsh, 4, st, 4, 0, str) == 0 && mlx_array_eval(view) == 0;
    (void)mlx_get_active_memory(&m2);
    bool eq1 = false;
    if (ok) { bool b = false; eq1 = mlx_allclose(&eq, sliced, view, 1e-6, 1e-6, false, str) == 0 &&
                                              mlx_array_eval(eq) == 0 && mlx_array_item_bool(&b, eq) == 0 && b; }
    const double slice_mb = (double)(m1 - m0) / 1048576.0, view_mb = (double)(m2 - m1) / 1048576.0;
    const double copy_mb = (double)(nkv * need * hd * 4) / 1048576.0;
    /* ---- C2: write in-place (parent MODIFICATO, nessuna allocazione) ---- */
    bool inplace_ok = false; double c2_mb = 0.0;
    if (ok) {
        const int off = (need > 4) ? 4 : 0;
        mlx_array knew = mlx_array_new();
        const int ksh[4] = {1, nkv, 1, hd};
        float* kh = (float*)malloc((size_t)nkv * (size_t)hd * sizeof(float));
        mlx_array k7 = mlx_array_new_float32(7.0f);
        if (kh != NULL && mlx_full(&knew, ksh, 4, k7, MLX_FLOAT32, str) == 0) {
            (void)mlx_get_active_memory(&m3);
            const int s0[4] = {0,0,off,0}, sk[4] = {1,nkv,off+1,hd}, stp[4] = {1,1,1,1};
            bool w = mlx_slice_update_inplace(&parent, knew, s0, 4, sk, 4, stp, 4, str) == 0 &&
                     mlx_array_eval(parent) == 0;
            (void)mlx_get_active_memory(&m4);
            /* C2 (handoff 2026-09-17): m3/m4 sono size_t e l'allocatore puo' riportare MENO
             * dopo il free -> la sottrazione underflowava stampando un numero assurdo.
             * Aritmetica signed in entrambe le direzioni (non influiva sul PASS, ma era un
             * falso segnale di memoria). */
            c2_mb = (m4 >= m3) ? (double)(m4 - m3) / 1048576.0
                               : -(double)(m3 - m4) / 1048576.0;
            /* rileggo la regione scritta dal PARENT: se l'update e' una copia, il parent e' invariato */
            mlx_array back = mlx_array_new(), beq = mlx_array_new();
            const int b0[4] = {0,0,off,0}, bk[4] = {1,nkv,off+1,hd};
            if (w && mlx_slice(&back, parent, b0, 4, bk, 4, ps, 4, str) == 0 && mlx_array_eval(back) == 0) {
                bool bb = false;
                inplace_ok = mlx_allclose(&beq, back, knew, 1e-6, 1e-6, false, str) == 0 &&
                             mlx_array_eval(beq) == 0 && mlx_array_item_bool(&bb, beq) == 0 && bb;
            }
            mlx_array_free(back); mlx_array_free(beq);
        }
        mlx_array_free(k7);
        free(kh); mlx_array_free(knew);
    }
    /* ---- C3: pattern di decode (N append + vista) ---- */
    double c3_mb = 0.0;
    if (ok) {
        const int iters = 64;
        mlx_array zk = mlx_array_new(), zv = mlx_array_new();
        int zsh[4] = {1, nkv, cap, hd};
        if (mlx_zeros(&zk, zsh, 4, MLX_FLOAT32, str) == 0 && mlx_zeros(&zv, zsh, 4, MLX_FLOAT32, str) == 0 &&
            mlx_array_eval(zk) == 0 && mlx_array_eval(zv) == 0) {
            size_t a0 = 0, a1 = 0;
            (void)mlx_get_active_memory(&a0);
            for (int t = 0; t < iters; t++) {
                mlx_array kn = mlx_array_new(), vn = mlx_array_new(), kv = mlx_array_new(), vv = mlx_array_new();
                const int nsh[4] = {1, nkv, 1, hd};
                mlx_array s1v = mlx_array_new_float32(1.0f), s2v = mlx_array_new_float32(2.0f);
                int fk = mlx_full(&kn, nsh, 4, s1v, MLX_FLOAT32, str); int fv = mlx_full(&vn, nsh, 4, s2v, MLX_FLOAT32, str);
                mlx_array_free(s1v); mlx_array_free(s2v);
                if (fk != 0 || fv != 0) { mlx_array_free(kn); mlx_array_free(vn); break; }
                const int w0[4] = {0,0,t,0}, wk[4] = {1,nkv,t+1,hd};
                if (mlx_slice_update_inplace(&zk, kn, w0, 4, wk, 4, ps, 4, str) != 0 ||
                    mlx_slice_update_inplace(&zv, vn, w0, 4, wk, 4, ps, 4, str) != 0) { mlx_array_free(kn); mlx_array_free(vn); break; }
                const int rk[4] = {1,nkv,t+1,hd};
                if (mlx_slice(&kv, zk, p0, 4, rk, 4, ps, 4, str) != 0 ||
                    mlx_slice(&vv, zv, p0, 4, rk, 4, ps, 4, str) != 0) { mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(kv); mlx_array_free(vv); break; }
                if (mlx_array_eval(kv) != 0 || mlx_array_eval(vv) != 0) { mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(kv); mlx_array_free(vv); break; }
                mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(kv); mlx_array_free(vv);
            }
            (void)mlx_get_active_memory(&a1);
            c3_mb = (double)(a1 - a0) / 1048576.0;
            /* atteso: nessuna crescita proporzionale alle iterazioni (cap e' fisso) */
        }
        mlx_array_free(zk); mlx_array_free(zv);
    }
    const bool pass = ok && eq1 && inplace_ok && slice_mb < copy_mb * 0.25 + 1.0 && c3_mb < 32.0;
    if (out && cap_out > 0)
        snprintf(out, (size_t)cap_out,
                 "%s C1[slice_alloca=%.2f view_alloca=%.2f atteso_slice~%.2fMB uguali=%s] "
                 "C2[inplace=%s alloca=%.2fMB] C3[pattern_decode iters=64 crescita=%.2fMB]",
                 pass ? "PASS" : "FAIL", slice_mb, view_mb, copy_mb, eq1 ? "si" : "NO",
                 inplace_ok ? "si" : "NO", c2_mb, c3_mb);
    mlx_array_free(parent); mlx_array_free(sliced); mlx_array_free(view); mlx_array_free(eq);
    mlx_stream_free(str); mlx_device_free(dev); free(host);
    return pass ? 0 : -1;
}

/* SELF-TEST UNITARIO (2026-09-17) — BANDA DELL'ATTENZIONE vs lunghezza KV (nessun modello).
 * Misura la banda effettiva della lettura K/V in SDPA decode (q=[1,H,1,hd], K/V=[1,nkv,nk,hd])
 * alle geometrie reali dei nostri modelli (K/V f16, q bf16 come nel decode). Serve a decidere se il
 * residuo di contesto lungo e' NELL'ATTENZIONE o altrove: se la banda e' al muro (~500 GB/s su M5
 * Max) l'attenzione e' ottimale; se e' molto sotto, il collo e' li'.
 * Driver: `llm_attn_bw_test <nkv> <n_heads> <hd> <nk> <reps>` (nk singolo, si invoca per piu' Nk). */
int llm_ai_mlx_attention_bw_selftest(int nkv, int n_heads, int hd, int nk, int reps, char* out, int cap_out) {
    const int kv_f16 = (getenv("LLM_ATTN_TEST_KVF16") != NULL) ? 1 : 0; /* default: K/V nello stesso dtype di q (come in produzione su qwen3) */
    if (nkv <= 0 || n_heads <= 0 || hd <= 0 || nk <= 0 || reps <= 0) {
        if (out && cap_out > 0) snprintf(out, (size_t)cap_out, "FAIL: parametri");
        return -1;
    }
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL) { if (out) snprintf(out, (size_t)cap_out, "FAIL: device GPU"); return -1; }
    if (mlx_set_default_device(dev) != 0) { mlx_device_free(dev); if (out) snprintf(out, (size_t)cap_out, "FAIL: set_default_device"); return -1; }
    mlx_stream str = mlx_stream_new();
    if (mlx_get_default_stream(&str, dev) != 0 || str.ctx == NULL) { mlx_device_free(dev); if (out) snprintf(out, (size_t)cap_out, "FAIL: default_stream"); return -1; }
    const int qsh[4] = {1, n_heads, 1, hd}, ksh[4] = {1, nkv, nk, hd};
    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new(), o = mlx_array_new();
    bool ok = true;
    int failstep = 0;
    if (ok && mlx_zeros(&q, qsh, 4, MLX_BFLOAT16, str) != 0) { ok = false; failstep = 1; }
    if (ok && mlx_zeros(&k, ksh, 4, kv_f16 ? MLX_FLOAT16 : MLX_BFLOAT16, str) != 0) { ok = false; failstep = 2; }
    if (ok && mlx_zeros(&v, ksh, 4, kv_f16 ? MLX_FLOAT16 : MLX_BFLOAT16, str) != 0) { ok = false; failstep = 3; }
    if (ok && (mlx_array_eval(q) != 0 || mlx_array_eval(k) != 0 || mlx_array_eval(v) != 0)) { ok = false; failstep = 4; }
    double ms = 0.0;
    if (ok) {
        const float scale = 1.0f / sqrtf((float)hd);
        for (int w = 0; w < 3; w++) { if (!llm_mlx_gqa_sdpa(&o, q, k, v, scale, 0, str)) { ok = false; failstep = 5; break; } if (mlx_array_eval(o) != 0) { ok = false; failstep = 6; break; } }
        /* NB: `llm_mlx_gqa_sdpa` ritorna bool TRUE = successo. Il test `!= 0` qui e sotto
         * trattava il SUCCESSO come errore -> il self-test ha sempre riportato il falso
         * "FAIL step_fail=5" (Bugs/(open) - MLX-Decode-Costo-Contesto-KV-Attention.md,
         * 2026-09-17: "NON funzionante ... non attribuito"). Verificato col probe
         * llm_mlx_attn_roof_probe.c (stessa chiamata, esito positivo).
         * RIPRISTINATO 2026-09-17 (sessione MLX): questo fix era nel working tree di un'altra
         * sessione ed e' stato sovrascritto per errore da un ripristino "git show HEAD"; il testo
         * e' stato ripreso dal diff. NON risovrascrivere il file con HEAD: usare backup nominati. */
        if (ok) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int r = 0; r < reps; r++) { if (!llm_mlx_gqa_sdpa(&o, q, k, v, scale, 0, str)) { ok = false; failstep = 7; break; } }
            if (ok && mlx_array_eval(o) != 0) { ok = false; failstep = 8; }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ms = ((double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9) * 1e3 / (double)reps;
        }
    }
    double gbs = 0.0;
    if (ok && ms > 0.0) {
        const double bytes = 2.0 * (double)nkv * (double)nk * (double)hd * 2.0; /* K+V, f16 */
        gbs = bytes / (ms * 1e-3) / 1e9;
    }
    if (out && cap_out > 0)
        snprintf(out, (size_t)cap_out, "%s Nk=%d ms/call=%.3f banda_KV=%.0f GB/s step_fail=%d",
                 (ok ? "INFO" : "FAIL"), nk, ms, gbs, failstep);
    mlx_array_free(q); mlx_array_free(k); mlx_array_free(v); mlx_array_free(o);
    mlx_stream_free(str); mlx_device_free(dev);
    return ok ? 0 : -1;
}

int llm_ai_mlx_mask_cache_selftest(int Lq, int Nk, int win, char* out, int cap) {
    if (out == NULL || cap < 32 || Lq <= 0 || Nk <= 0) return -1;
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    bool gpu_ok = false;
    mlx_device_is_available(&gpu_ok, gpu);
    mlx_device chosen = gpu_ok ? gpu : mlx_device_new_type(MLX_CPU, 0);
    if (!gpu_ok) mlx_device_free(gpu);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }

    const int H = 2, Hkv = 1, D = 32;
    float* qd = (float*)malloc(sizeof(float) * (size_t)H * Lq * D);
    float* kd = (float*)malloc(sizeof(float) * (size_t)Hkv * Nk * D);
    float* vd = (float*)malloc(sizeof(float) * (size_t)Hkv * Nk * D);
    int rc = -1;
    if (qd && kd && vd) {
        uint32_t seed = 12345u;
        for (size_t i = 0; i < (size_t)H * Lq * D; i++) { seed = seed * 1664525u + 1013904223u; qd[i] = ((seed >> 8) & 1023) / 512.0f - 1.0f; }
        for (size_t i = 0; i < (size_t)Hkv * Nk * D; i++) { seed = seed * 1664525u + 1013904223u; kd[i] = ((seed >> 8) & 1023) / 512.0f - 1.0f; }
        for (size_t i = 0; i < (size_t)Hkv * Nk * D; i++) { seed = seed * 1664525u + 1013904223u; vd[i] = ((seed >> 8) & 1023) / 512.0f - 1.0f; }
        int qsh[4] = {1, H, Lq, D}, kvsh[4] = {1, Hkv, Nk, D};
        mlx_array q = mlx_array_new_data(qd, qsh, 4, MLX_FLOAT32);
        mlx_array k = mlx_array_new_data(kd, kvsh, 4, MLX_FLOAT32);
        mlx_array v = mlx_array_new_data(vd, kvsh, 4, MLX_FLOAT32);
        mlx_array outa = mlx_array_new();
        bool ok = true;
        for (int rep = 0; rep < 3 && ok; rep++) ok = llm_mlx_gqa_sdpa(&outa, q, k, v, 1.0f, win, st) ? true : false;
        if (ok) {
            mlx_array of = mlx_array_new();
            if (mlx_astype(&of, outa, MLX_FLOAT32, st) == 0 && mlx_array_eval(of) == 0) {
                const float* od = mlx_array_data_float32(of);
                const size_t nel = mlx_array_size(of);
                uint64_t h = 1469598103934665603ULL;
                for (size_t i = 0; i < nel; i++) {
                    const unsigned char* b = (const unsigned char*)&od[i];
                    for (int j = 0; j < 4; j++) { h ^= b[j]; h *= 1099511628211ULL; }
                }
                snprintf(out, (size_t)cap, "HASH %016llx SAMPLES %zu FIRST %.6f",
                         (unsigned long long)h, nel, nel ? od[0] : 0.0f);
                rc = 0;
            }
            mlx_array_free(of);
        }
        mlx_array_free(outa); mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
    }
    free(qd); free(kd); free(vd);
    mlx_device_free(chosen);
    return rc;
}

/* llm_ai_mlx_oq_a8_selftest (2026-09-16) — self-test UNITARIO del primitive oQ A8, SENZA modello.
 * Verifica le due cose introdotte dalla promozione: (1) la RISOLUZIONE AUTOMATICA delle metallib
 * accanto alla dylib (dladdr, senza env); (2) il FAIL-OPEN: `quantized_linear_cfg` su input
 * eleggibile (affine, g64, bits4, M>=16, N%64==0, K%64==0) deve tornare TRUE sia col ctx creato
 * sia quando le metallib mancano (prima del fix faceva `return false` = prefill FALLITO).
 * I dati sono zeri: qui conta solo che la funzione NON fallisca. Scrive l'esito in `out`. */
int llm_ai_mlx_oq_a8_selftest(int m, int n, int k, char* out, int cap) {
    if (out == NULL || cap < 32 || m < 16 || n <= 0 || k <= 0 || (n % 64) != 0 || (k % 64) != 0) return -1;
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    bool gpu_ok = false;
    mlx_device_is_available(&gpu_ok, gpu);
    mlx_device chosen = gpu_ok ? gpu : mlx_device_new_type(MLX_CPU, 0);
    if (!gpu_ok) mlx_device_free(gpu);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }

    const int wcols = k / 8;      /* 4-bit packed: k * 4 / 32 */
    const int gcols = k / 64;     /* group 64 */
    uint16_t* xb = (uint16_t*)calloc((size_t)m * k, sizeof(uint16_t));
    uint32_t* wb = (uint32_t*)calloc((size_t)n * wcols, sizeof(uint32_t));
    uint16_t* sb = (uint16_t*)calloc((size_t)n * gcols, sizeof(uint16_t));
    uint16_t* bb = (uint16_t*)calloc((size_t)n * gcols, sizeof(uint16_t));
    int rc = -1;
    if (xb && wb && sb && bb) {
        int xsh[2] = {m, k}, wsh[2] = {n, wcols}, gsh[2] = {n, gcols};
        mlx_array x = mlx_array_new_data(xb, xsh, 2, MLX_FLOAT16);
        mlx_array w = mlx_array_new_data(wb, wsh, 2, MLX_UINT32);
        mlx_array sc = mlx_array_new_data(sb, gsh, 2, MLX_FLOAT16);
        mlx_array bi = mlx_array_new_data(bb, gsh, 2, MLX_FLOAT16);
        const bool ctx_ok = (llm_mlx_oq_a8_ctx() != NULL);
        mlx_array res = mlx_array_new();
        const bool cfg_ok = quantized_linear_cfg(&res, x, w, sc, bi, 64, 4, "affine", st);
        const bool res_ok = (res.ctx != NULL);
        snprintf(out, (size_t)cap, "CTX %s CFG_OK %d RES %d (m=%d n=%d k=%d: affine g64 bits4)",
                 ctx_ok ? "metallib-risolte" : "metallib-assenti", cfg_ok ? 1 : 0, res_ok ? 1 : 0, m, n, k);
        rc = cfg_ok ? 0 : -2;   /* -2 = FAIL-OPEN ROTTO: quantized_linear_cfg ha rifiutato */
        mlx_array_free(res);
        mlx_array_free(x); mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bi);
    }
    free(xb); free(wb); free(sb); free(bb);
    mlx_device_free(chosen);
    return rc;
}

/* llm_ai_mlx_sliding_selftest (2026-09-16) — P0-quinquies: un BLOCCO di query che
 * ATTRAVERSA la finestra deve dare lo STESSO risultato del calcolo INCREMENTALE token-per-token
 * con la stessa finestra. E' la proprieta' che un "trim" di solo bookkeeping romperebbe
 * (buffer vs maschera desincronizzati); qui e' verificata sul percorso finestrato di
 * llm_mlx_gqa_sdpa.  mode=1: KV incrementale ricostruita dai dati host (nessun mlx_slice);
 * mode=0: KV incrementale via mlx_slice.  Ritorna 0 se coerente (tolleranza bf16), -2 se no.
 *
 * ⚠️ TRAPPOLA (trovata scrivendo questo test, l'hang che mi ha fatto perdere tempo): NON usare
 * `mlx_device_is_available` + `chosen = gpu ? gpu : new(CPU)` + `mlx_device_free(gpu)` prima di
 * `mlx_get_default_stream`: con quel pattern la chiamata a get_default_stream si BLOCCA (device
 * aliasing/free). L'init che funziona e' quello qui sotto, identico al mask-selftest:
 * device_new_type(GPU) -> set_default_device -> get_default_stream. */
int llm_ai_mlx_sliding_selftest(int pre, int Lq, int win, int D, int mode, char* out, int cap) {
    if (out == NULL || cap < 32 || pre < 1 || Lq < 1 || win < 1 || win > pre + 1 || D < 8) return -1;
    mlx_device chosen = mlx_device_new_type(MLX_GPU, 0);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }

    const int Nk = pre + Lq;
    float* qd = (float*)malloc(sizeof(float) * (size_t)Lq * D);
    float* kd = (float*)malloc(sizeof(float) * (size_t)Nk * D);
    float* vd = (float*)malloc(sizeof(float) * (size_t)Nk * D);
    int rc = -1;
    if (qd && kd && vd) {
        uint32_t seed = 999u;
        for (size_t i = 0; i < (size_t)Lq * D; i++) { seed = seed * 1664525u + 1013904223u; qd[i] = ((seed >> 8) & 1023) / 512.0f - 1.0f; }
        for (size_t i = 0; i < (size_t)Nk * D; i++) { seed = seed * 1664525u + 1013904223u; kd[i] = ((seed >> 8) & 1023) / 512.0f - 1.0f; }
        for (size_t i = 0; i < (size_t)Nk * D; i++) { seed = seed * 1664525u + 1013904223u; vd[i] = ((seed >> 8) & 1023) / 512.0f - 1.0f; }
        int qsh[4] = {1, 1, Lq, D}, kvsh[4] = {1, 1, Nk, D};
        mlx_array q = mlx_array_new_data(qd, qsh, 4, MLX_FLOAT32);
        mlx_array k = mlx_array_new_data(kd, kvsh, 4, MLX_FLOAT32);
        mlx_array v = mlx_array_new_data(vd, kvsh, 4, MLX_FLOAT32);
        mlx_array blk = mlx_array_new(), blkf = mlx_array_new();
        bool ok = llm_mlx_gqa_sdpa(&blk, q, k, v, 1.0f, win, st) ? true : false;
        ok = ok && mlx_astype(&blkf, blk, MLX_FLOAT32, st) == 0 && mlx_array_eval(blkf) == 0;
        if (ok) {
            const float* bd = mlx_array_data_float32(blkf);
            double mx = 0.0;
            for (int i = 0; i < Lq && ok; i++) {
                const int nki = pre + i + 1;
                mlx_array ki = mlx_array_new(), vi = mlx_array_new();
                if (mode == 1) {
                    int ki_sh[4] = {1, 1, nki, D};
                    ki = mlx_array_new_data(kd, ki_sh, 4, MLX_FLOAT32);
                    vi = mlx_array_new_data(vd, ki_sh, 4, MLX_FLOAT32);
                } else {
                    int s0[4] = {0,0,0,0}, s1[4] = {1,1,nki,D}, stp[4] = {1,1,1,1};
                    ok = mlx_slice(&ki, k, s0, 4, s1, 4, stp, 4, st) == 0 &&
                         mlx_slice(&vi, v, s0, 4, s1, 4, stp, 4, st) == 0;
                }
                int q1sh[4] = {1, 1, 1, D};
                mlx_array qi = mlx_array_new_data(qd + (size_t)i * D, q1sh, 4, MLX_FLOAT32);
                mlx_array oi = mlx_array_new(), of = mlx_array_new();
                if (ok) ok = llm_mlx_gqa_sdpa(&oi, qi, ki, vi, 1.0f, win, st) ? true : false;
                if (ok) ok = (mlx_astype(&of, oi, MLX_FLOAT32, st) == 0 && mlx_array_eval(of) == 0);
                if (ok) {
                    const float* od = mlx_array_data_float32(of);
                    for (int d = 0; d < D; d++) { double e = fabs((double)bd[(size_t)i*D+d] - (double)od[d]); if (e > mx) mx = e; }
                }
                mlx_array_free(oi); mlx_array_free(of); mlx_array_free(qi); mlx_array_free(ki); mlx_array_free(vi);
            }
            snprintf(out, (size_t)cap, "maxdiff=%.6g mode=%d (pre=%d Lq=%d win=%d D=%d, %s)",
                     mx, mode, pre, Lq, win, D, (ok && mx <= 0.02) ? "COERENTE" : "*** INCOERENTE ***");
            rc = (ok && mx <= 0.02) ? 0 : -2;
        }
        mlx_array_free(blkf); mlx_array_free(blk);
        mlx_array_free(q); mlx_array_free(k); mlx_array_free(v);
    }
    free(qd); free(kd); free(vd);
    mlx_device_free(chosen);
    return rc;
}

/* Self-test (2026-09-16, deciso dal dato LLM_ASYNC_TIME): separa BUILD e ENCODING per nodo.
 * Perche': in decode il wall di Llama-1B e' CPU (build 0,26 + submit 1,15 ms/token vs GPU ~1,0)
 * e il grafo ha 363 nodi/token -> l'ipotesi e' che il costo sia l'ENCODING Metal per nodo
 * (~3 us), non la costruzione. Qui si misura: catena di `n_ops` Add elementwise su array
 * piccoli [1,1,1024] (lavoro GPU trascurabile), su `reps` ripetizioni INTERLEAVED async/eager,
 * e si riporta il MIN per (a) costruzione, (b) mlx_async_eval (submit, CPU), (c) attesa GPU,
 * (d) eager mlx_eval (build+submit+wait). */
/* Self-test DRIFT della SDPA (2026-09-16): e' l'op del decode che riceve una forma NUOVA a ogni
 * token (la slice della KV a `need` crescente). `grow>0`: Nk cresce a ogni iterazione (com'e'
 * oggi); `grow==0`: Nk COSTANTE (com'e' con la cache padded + maschera, fix B). Se il tempo
 * cresce solo con grow>0, la degradazione per-iterazione e' la forma nuova passata alla SDPA. */
int llm_ai_mlx_eval_sdpadrift_selftest(int nkv_start, int nkv_grow, int iters, char* out, int cap) {
    if (out == NULL || cap < 64 || nkv_start < 8 || nkv_grow < 0 || iters < 128) return -1;
    mlx_device chosen = mlx_device_new_type(MLX_GPU, 0);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }
    const int H = 8, D = 64, MAXK = 4096;
    float* qd = (float*)malloc(sizeof(float) * (size_t)H * D);
    float* kd = (float*)malloc(sizeof(float) * (size_t)H * MAXK * D);
    float* vd = (float*)malloc(sizeof(float) * (size_t)H * MAXK * D);
    if (qd == NULL || kd == NULL || vd == NULL) { free(qd); free(kd); free(vd); mlx_device_free(chosen); return -1; }
    for (size_t i = 0; i < (size_t)H * D; i++) qd[i] = 0.01f * (float)(i % 13);
    for (size_t i = 0; i < (size_t)H * MAXK * D; i++) { kd[i] = 0.01f * (float)(i % 11); vd[i] = 0.01f * (float)(i % 7); }
    const int qsh[4] = {1, H, 1, D};
    mlx_array q = mlx_array_new_data(qd, qsh, 4, MLX_FLOAT32);
    double first = 0, last = 0, worst = 0;
    for (int it = 0; it < iters; it++) {
        int N = nkv_start + it * nkv_grow;
        if (N > MAXK) N = MAXK;
        const int ksh[4] = {1, H, N, D};
        mlx_array k = mlx_array_new_data(kd, ksh, 4, MLX_FLOAT32);
        mlx_array v = mlx_array_new_data(vd, ksh, 4, MLX_FLOAT32);
        mlx_array no_mask = mlx_array_new(), outa = mlx_array_new();
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int rc = mlx_fast_scaled_dot_product_attention(&outa, q, k, v, 1.0f / 8.0f, "", no_mask, no_mask, st);
        if (rc == 0) {
            mlx_vector_array vec = mlx_vector_array_new();
            mlx_vector_array_append_value(vec, outa);
            mlx_async_eval(vec);
            mlx_vector_array_free(vec);
            mlx_synchronize(st);
        }
        mlx_array_free(no_mask); mlx_array_free(outa); mlx_array_free(k); mlx_array_free(v);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
        if (rc != 0) { snprintf(out, (size_t)cap, "sdpadrift: SDPA fallita (rc=%d)", rc); mlx_array_free(q); free(qd); free(kd); free(vd); mlx_device_free(chosen); return -1; }
        if (it == 0) first = ms;
        if (it == iters - 1) last = ms;
        if (it >= 64 && ms > worst) worst = ms;
        if ((it % 64) == 63 && llm_env_get("LLM_DRIFT_TRACE"))
            fprintf(stderr, "[sdpadrift] it=%d Nk=%d  %.3f ms\n", it + 1, N, ms);
    }
    snprintf(out, (size_t)cap, "sdpadrift: nkv=%d grow=%d iters=%d | primo=%.3f ultimo=%.3f worst=%.3f -> %s",
             nkv_start, nkv_grow, iters, first, last, worst, (last > first * 2.0) ? "*** CRESCE ***" : "stabile");
    mlx_array_free(q); free(qd); free(kd); free(vd);
    mlx_device_free(chosen);
    return 0;
}

/* Self-test di DRIFT CON FORMA CRESCENTE (2026-09-16): a ogni iterazione il grafo ha una forma
 * DIVERSA (N cresce di `step`), come la slice della cache KV nel decode (need cresce a ogni
 * token). Isola la voce "forma nuova = specializzazione/compilazione/caching" di MLX dal loop
 * del motore. Se il tempo CRESCE con l'indice di iterazione, il degrado del decode (7,7 -> 22
 * ms/token, indipendente dal CONTESTO ma non dalle ITERAZIONI) e' questa. */
int llm_ai_mlx_eval_shapedrift_selftest(int base_n, int grow, int iters, char* out, int cap) {
    if (out == NULL || cap < 64 || base_n < 8 || grow < 0 || iters < 128) return -1;
    mlx_device chosen = mlx_device_new_type(MLX_GPU, 0);
    if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); return -1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, chosen) != 0) { mlx_device_free(chosen); return -1; }
    const int MAXN = 65536;
    float* d = (float*)malloc(sizeof(float) * (size_t)MAXN);
    if (d == NULL) { mlx_device_free(chosen); return -1; }
    for (int i = 0; i < MAXN; i++) d[i] = (float)(i % 17) * 0.0625f;
    double first = 0, last = 0, worst = 0;
    for (int it = 0; it < iters; it++) {
        int N = base_n + it * grow;
        if (N > MAXN) N = MAXN;
        const int sh[2] = {1, N};
        mlx_array a = mlx_array_new_data(d, sh, 2, MLX_FLOAT32);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        mlx_array y = mlx_array_new();
        mlx_add(&y, a, a, st);
        mlx_vector_array vec = mlx_vector_array_new();
        mlx_vector_array_append_value(vec, y);
        mlx_async_eval(vec);
        mlx_vector_array_free(vec);
        mlx_synchronize(st);
        mlx_array_free(y); mlx_array_free(a);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
        if (it == 0) first = ms;
        if (it == iters - 1) last = ms;
        if (it >= 64 && ms > worst) worst = ms;
        if ((it % 64) == 63 && getenv("LLM_DRIFT_TRACE"))
            fprintf(stderr, "[shapedrift] it=%d N=%d  %.3f ms\n", it + 1, N, ms);
    }
    snprintf(out, (size_t)cap, "shapedrift: base=%d grow=%d iters=%d | primo=%.3f ultimo=%.3f worst=%.3f -> %s",
             base_n, grow, iters, first, last, worst, (last > first * 2.0) ? "*** cresce con la FORMA nuova ***" : "stabile");
    free(d);
    mlx_device_free(chosen);
    return 0;
}
