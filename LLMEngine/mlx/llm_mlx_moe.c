/* ============================================================================
 * llm_mlx_moe.c — dominio MoE: gather gate/up fuso, gather_mm,
 * cache pesi FFN MoE per-layer (lazy). Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_moe_fused_kernel.h"
#include "llm_mlx_swiglu_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── SWAP/offload esperti (streaming da disco) ─────────────────────────────────────────────────
 * Prima era un env (`LLM_MOE_SWAP_DIR`); ora è una SCELTA del consumer via API (nessun env):
 * `llm_ai_mlx_set_moe_swap_dir(dir)` imposta la dir del modello da cui streamare gli esperti,
 * chiamata PRIMA del load. Con dir non vuota lo streaming è ATTIVO. `llm_mlx_moe_swap_on()` è il
 * gate letto da tutti i path (fused, pretouch, gather_streamed). */
static char g_moe_swap_dir[1024];
int llm_mlx_moe_swap_on(void) { return g_moe_swap_dir[0] != '\0'; }
const char* llm_mlx_moe_swap_dir(void) { return g_moe_swap_dir; }

/* ── TurboQuant: scelta MEMORIA vs VELOCITÀ per gli esperti ─────────────────────────────────────
 * API del motore (NON env): il consumer (gli host, via ABI) la imposta.
 *   on=1 (default): esperti PACKED (kernel TQ diretto) -> RSS basso, un po' piu' lento.
 *   on=0          : array affine-2bit residenti + gather_qmm MLX -> velocita' massima, piu' RAM. */
static int g_tq_mem_opt = 1;
void llm_ai_mlx_set_tq_memory_opt(int on) { g_tq_mem_opt = on ? 1 : 0; }

/* Registra il setter TurboQuant nel registro opzioni (decoupling: `llmengine_opts` non
 * referenzia direttamente la lib mlx). Il constructor gira al load della dylib. */
__attribute__((constructor)) static void llm_tq_opt_register_c(void) {
    llm_ai_mlx_opt_register_tq_setter(llm_ai_mlx_set_tq_memory_opt);
}
int llm_mlx_tq_mem_opt(void) { return g_tq_mem_opt; }


/* LLMENGINE_MOE_GATEUP_FUSE (default OFF): stessa tecnica di rapidMLX vllm_mlx/moe_fusion.py
 * (+7.0% decode misurato DA LORO su Qwen3.6-35B-A3B) -- un gather_qmm sul peso gate+up
 * concatenato (fq->guw/gus/gub, calcolato una volta al caricamento, vedi GmoeFfnQuant) invece
 * di due gather_qmm separati, split del risultato in due meta' lungo l'ultimo asse. */
static int g_moe_gateup_fuse_state = -1;
static bool llm_mlx_moe_gateup_fuse_enabled(void) {
    if (g_moe_gateup_fuse_state < 0) {
        /* DEFAULT ON (2026-10-01). Misura deterministica glm4_moe: 3554 -> 3509 nodi/token
         * (-45, -1.3%) con output BIT-IDENTICO; la reference (rapidMLX) misura +7% decode.
         * Fallback automatico se i pesi gate+up fusi non esistono o in streaming esperti.
         * Opt-out: LLMENGINE_MOE_GATEUP_FUSE=0. */
        const char* e = llm_ai_mlx_opt_env("LLMENGINE_MOE_GATEUP_FUSE");
        g_moe_gateup_fuse_state = (e != NULL && e[0] == '0') ? 0 : 1;
    }
    return g_moe_gateup_fuse_state == 1;
}

/* Tutti i gather_qmm degli esperti passano DIRETTAMENTE da mlx_gather_qmm: l'allineamento
 * dtype(x) vs dtype(scales) e' deciso UNA volta a load-time da llm_mlx_auto_pair_act_dtype
 * (in MEGLIO: li' x coincide con le scale per tutti i tensori, qui sarebbe un cast
 * per-chiamata che cambia i bit dei NON-promossi -- l'fp32 con scale BF16 li' accanto resta
 * fp32, che e' il comportamento giusto). Il flag sperimentale che faceva il cast al gather
 * (LLMENGINE_MOE_QQ16, L28) e' stato ELIMINATO il 2026-09-09 per quello: misurato x1,80 decode su
 * Mixtral con x BF16 e scale F16, ma era la PROMUZIONE MANCANTE la causa, non il cast --
 * la stessa lezione gia' arbitrata sul GLM (flavor attivo -> il flag valeva 0). */

/* Guardia NAX #3856 (MLX 0.32.x su M5): gather_qmm(sorted_indices=true) con M>32768 e
 * M%64!=0 corrompe la coda (ultimo blocco parziale di 64 righe). Riprodotto OFFLINE senza
 * modello da `LLMEngine/tools/oracoli/llm_gather_qmm_sorted_probe` (forme reali esperti GLM-4.5): M<=32768
 * sempre bit-exact; M>32768 bit-exact sse M%64==0; altrimenti un blocco esperto corrotto
 * (max|diff| ~1.5e3, deterministico su piu' seed). Fix locale: segmentare in slice di
 * <=32768 righe (multiplo di 64) e concatenare. Vale per OGNI MoE sorted (GLM/DS/qwen3/
 * mixtral): tutte passano da qui. */
#define LLM_GATHER_SORTED_MAX_ROWS 32768

static int llm_mlx_gather_qmm_guarded(mlx_array* res, mlx_array x, mlx_array w, mlx_array sc,
                                      mlx_array bi, mlx_array lhs, mlx_array rhs, bool transpose,
                                      mlx_optional_int gsz, mlx_optional_int bts, const char* mode,
                                      bool sorted, mlx_stream st) {
    const int64_t rows = (rhs.ctx != NULL) ? (int64_t)mlx_array_size(rhs) : 0;
    const bool need_slice = sorted && rows > LLM_GATHER_SORTED_MAX_ROWS && (rows % 64) != 0 &&
                            mlx_array_ndim(rhs) == 1 && mlx_array_ndim(x) >= 1 &&
                            (int64_t)mlx_array_shape(x)[0] == rows;
    static int g_guard_dbg = -1;
    if (g_guard_dbg < 0) g_guard_dbg = getenv("LLM_MOE_GUARD_DBG") != NULL ? 1 : 0;
    if (g_guard_dbg) fprintf(stderr, "[moe-guard] rows=%lld sorted=%d slice=%d\n",
                             (long long)rows, (int)sorted, (int)need_slice);
    if (!need_slice)
        return mlx_gather_qmm(res, x, w, sc, bi, lhs, rhs, transpose, gsz, bts, mode, sorted, st);
    mlx_vector_array parts = mlx_vector_array_new();
    int rc = 0;
    for (int64_t off = 0; off < rows && rc == 0; off += LLM_GATHER_SORTED_MAX_ROWS) {
        int64_t stop = off + LLM_GATHER_SORTED_MAX_ROWS;
        if (stop > rows) stop = rows;
        int s0[8], s1[8], st1[8];
        const int ndx = mlx_array_ndim(x);
        const int* shp = mlx_array_shape(x);
        for (int d = 0; d < ndx; d++) { s0[d] = 0; s1[d] = shp[d]; st1[d] = 1; }
        s0[0] = (int)off; s1[0] = (int)stop;
        const int r0[1] = {(int)off}, r1[1] = {(int)stop}, rst[1] = {1};
        mlx_array xs = mlx_array_new(), rs = mlx_array_new(), o = mlx_array_new();
        rc = mlx_slice(&xs, x, s0, ndx, s1, ndx, st1, ndx, st);
        if (rc == 0) rc = mlx_slice(&rs, rhs, r0, 1, r1, 1, rst, 1, st);
        if (rc == 0) rc = mlx_gather_qmm(&o, xs, w, sc, bi, lhs, rs, transpose, gsz, bts, mode, true, st);
        if (rc == 0) mlx_vector_array_append_value(parts, o);
        mlx_array_free(o); mlx_array_free(rs); mlx_array_free(xs);
    }
    if (rc == 0) rc = mlx_concatenate_axis(res, parts, 0, st);
    mlx_vector_array_free(parts);
    return rc;
}

/* Prova il gather_qmm fuso; ritorna false SENZA scrivere out_gate/out_up se non applicabile
 * (fq NULL, fusione non riuscita al caricamento, flag OFF, o l'apply stesso fallisce) -- il
 * chiamante deve fare fallback ai due gather_qmm separati in quel caso. */
bool llm_mlx_moe_gateup_gather(GmoeFfnQuant* fq, mlx_array x, mlx_array lhs_indices,
                                       mlx_array rhs_indices, bool sorted_indices,
                                       mlx_optional_int gsz_opt, mlx_optional_int bts_opt,
                                       const char* mode,
                                       mlx_array* out_gate, mlx_array* out_up, mlx_stream stream) {
    if (fq == NULL || !fq->has_gate_up_fused || !llm_mlx_moe_gateup_fuse_enabled()) return false;
    if (llm_mlx_moe_swap_on()) return false; /* streaming esperti: niente fused (peso pieno) */
    mlx_array gate_up = mlx_array_new();
    bool ok = llm_mlx_gather_qmm_guarded(&gate_up, x, fq->guw, fq->gus, fq->gub, lhs_indices, rhs_indices,
                             /*transpose=*/true, gsz_opt, bts_opt, mode, sorted_indices, stream) == 0;
    if (ok) {
        mlx_vector_array parts = mlx_vector_array_new();
        ok = mlx_split(&parts, gate_up, 2, /*axis=*/-1, stream) == 0 && mlx_vector_array_size(parts) == 2;
        if (ok) {
            mlx_vector_array_get(out_gate, parts, 0);
            mlx_vector_array_get(out_up, parts, 1);
            /* Le due meta' della split sono VISTE non-contigue: un gather_qmm successivo
             * (down_proj) su input non-contiguo degrada ~40x (lezione gia' trovata dal vivo
             * su mixtral_ffn_block, 2026-09-03, stesso fix li'). Forzare la densita' qui,
             * prima che il chiamante usi gate/up in sigmoid/multiply/gather_qmm. */
            mlx_array gc = mlx_array_new(), uc = mlx_array_new();
            ok = mlx_contiguous(&gc, *out_gate, false, stream) == 0 &&
                 mlx_contiguous(&uc, *out_up, false, stream) == 0;
            if (ok) { mlx_array_set(out_gate, gc); mlx_array_set(out_up, uc); }
            mlx_array_free(gc); mlx_array_free(uc);
        }
        mlx_vector_array_free(parts);
    }
    mlx_array_free(gate_up);
    /* Branch-first (todo MLX-Decode-Composizione §7): una riga UNA volta per processo che
     * prova che il ramo fuso e' davvero in uso (senza, "IDS identici" non distingue fused da
     * fallback). LLMENGINE_MOE_GATEUP_DBG=1, zero overhead quando non impostata. */
    if (ok) {
        static int s_dbg = -1, s_printed = 0;
        if (s_dbg < 0) s_dbg = (getenv("LLMENGINE_MOE_GATEUP_DBG") != NULL) ? 1 : 0;
        if (s_dbg && !s_printed) { s_printed = 1; fprintf(stderr, "[moe-gateup] fusione ATTIVA (gather_qmm fuso gate+up)\n"); }
    }
    return ok;
}
/* MoE: matmul per-esperto. Percorso GGUF: pesi dequantizzati F16 (quant_mode=""), NIENTE
 * scales -> mlx_gather_mm (non quantizzato). Percorso MLX: mlx_gather_qmm (affine). */
int llm_mlx_moe_gather_mm(const ModelConfig* cfg, mlx_array* res, mlx_array x, mlx_array w,
                                 mlx_array scales, mlx_array biases, mlx_array lhs, mlx_array idxs,
                                 bool transpose, mlx_optional_int gsz, mlx_optional_int bts,
                                 const char* mode, bool sorted, mlx_stream s) {
    if (cfg->quant_mode[0] == 0 || strcmp(cfg->quant_mode, "none") == 0 || scales.ctx == NULL) {
        /* gather_mm calcola a @ b con b = (E, K, N): il peso (E, out, in) va trasposto (ultimi 2 assi). */
        (void)transpose; (void)gsz; (void)bts; (void)mode;
        int nd = mlx_array_ndim(w);
        if (nd < 2) return -1;
        int perm[8];
        for (int i = 0; i < nd; i++) perm[i] = i;
        int tmp = perm[nd - 1]; perm[nd - 1] = perm[nd - 2]; perm[nd - 2] = tmp;
        mlx_array wt = mlx_array_new();
        int r = mlx_transpose_axes(&wt, w, perm, nd, s);
        if (r == 0) r = mlx_gather_mm(res, x, wt, lhs, idxs, sorted, s);
        mlx_array_free(wt);
        return r;
    }
    return llm_mlx_gather_qmm_guarded(res, x, w, scales, biases, lhs, idxs, transpose, gsz, bts, mode, sorted, s);
}
/* ── SWAP esperti (streaming): ingest safetensors + remap + builder. Env-gated, default OFF
 * offload esperti (dir via API). Include relativo: nessuna modifica al CMakeLists. ── */
#include "llm_mlx_moe_stream.h"
#include "llm_mlx_moe_stream_build.h"

static LlmEngineMoeTsIngest g_moe_ts;
static int g_moe_ts_state = -1; /* -1 non init, 0 off, 1 ok */

/* API del consumer (nessun env): imposta la dir degli esperti da streamare. Va chiamata PRIMA
 * del load del modello. `dir` NULL/vuota disattiva. Resetta lo stato dell'ingest così una nuova
 * dir viene riaperta alla prima chiamata. */
/* ── Profiler routing + hotlist (port ds4 `ee7e840`/hotlist) ───────────────────────────────── */
static uint16_t* g_prof = NULL;
static int g_prof_layers = 0, g_prof_experts = 0;
void llm_mlx_moe_prof_reset(void) { free(g_prof); g_prof = NULL; g_prof_layers = g_prof_experts = 0; }
void llm_mlx_moe_prof_count(int layer, int e) {
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER || e < 0 || e >= 8192) return;
    if (g_prof == NULL) {
        g_prof_layers = LLMENGINE_MOE_TS_MAXLAYER; g_prof_experts = 8192;
        g_prof = (uint16_t*)calloc((size_t)g_prof_layers * (size_t)g_prof_experts, sizeof(uint16_t));
        if (g_prof == NULL) return;
    }
    uint16_t* v = &g_prof[(size_t)layer * (size_t)g_prof_experts + (size_t)e];
    if (*v < 0xFFFFu) (*v)++;
}
int llm_ai_mlx_moe_prof_write(const char* path) {
    if (path == NULL || path[0] == 0 || g_prof == NULL) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "# gemma/gmb expert hotlist v1 (layer expert count)\n");
    for (int l = 0; l < g_prof_layers; ++l)
        for (int e = 0; e < g_prof_experts; ++e) {
            uint16_t c = g_prof[(size_t)l * (size_t)g_prof_experts + (size_t)e];
            if (c) fprintf(f, "%d %d %u\n", l, e, (unsigned)c);
        }
    fclose(f);
    return 0;
}
int llm_ai_mlx_moe_hotlist_load(const char* path) { return llm_xsc_hot_load(path, LLMENGINE_MOE_TS_MAXLAYER, 8192); }

void llm_ai_mlx_set_moe_swap_dir(const char* dir) {
    llm_xsc_reset();   /* modello diverso -> cache esperti non valida */
    llm_xsc_hot_clear();
    llm_mlx_moe_prof_reset();
    if (dir) {
        snprintf(g_moe_swap_dir, sizeof g_moe_swap_dir, "%s", dir);
    } else {
        g_moe_swap_dir[0] = '\0';
    }
    g_moe_ts_state = -1;
    if (dir != NULL && dir[0] != '\0') {
        char hp[1200];
        if (llm_ai_mlx_opt_get_str("moe_hotlist", hp, sizeof hp) != 0 || hp[0] == '\0') {
            snprintf(hp, sizeof hp, "%s/hotlist.txt", dir);
        }
        if (llm_xsc_hot_load(hp, LLMENGINE_MOE_TS_MAXLAYER, 8192))
            fprintf(stderr, "[moe-hotlist] caricata %s\n", hp);
    } else {
        char pp[1200];
        if (llm_ai_mlx_opt_get_str("moe_profile", pp, sizeof pp) == 0 && pp[0] != '\0' &&
            llm_ai_mlx_moe_prof_write(pp) == 0)
            fprintf(stderr, "[moe-profile] scritto %s\n", pp);
        llm_mlx_moe_prof_reset();
    }
}

/* Statistiche della cache HOT/LRU (per metriche e test dal consumer). */
void llm_ai_mlx_moe_cache_stats(unsigned long long* hits, unsigned long long* misses,
                                unsigned long long* bytes) {
    if (hits) *hits = g_xsc.hits;
    if (misses) *misses = g_xsc.misses;
    if (bytes) *bytes = (unsigned long long)g_xsc.bytes;
}

static const LlmEngineMoeTsIngest* llm_moe_swap_ingest(void) {
    if (g_moe_ts_state < 0) {
        const char* d = llm_mlx_moe_swap_dir();
        g_moe_ts_state = (d && *d && llmengine_moe_ts_open(&g_moe_ts, d) > 0) ? 1 : 0;
    }
    return g_moe_ts_state == 1 ? &g_moe_ts : NULL;
}
/* true solo se lo streaming e' DAVVERO utilizzabile (ingest aperto con successo). Diverso da
 * `llm_mlx_moe_swap_on()` (solo dir impostata): e' la condizione da usare per decidere se il synth
 * puo' SALTARE la materializzazione degli esperti. Se l'ingest non apre (dir errata), si resta
 * resident F16 e il modello NON produce spazzatura. */
int llm_mlx_moe_swap_active(void) { return llm_moe_swap_ingest() != NULL; }

/* Byte TOTALI degli esperti (weight+scales+biases di tutti i layer) dall'ingest attivo.
 * Serve a sottrarli dal conto dell'active memory: con lo streaming gli esperti NON sono
 * materializzati -> il budget KV non deve contarli (altrimenti 0 su modelli >wired-limit). */
uint64_t llm_mlx_moe_swap_expert_bytes(void) {
    if (llm_moe_swap_ingest() == NULL) return 0;
    uint64_t tot = 0;
    for (int l = 0; l < LLMENGINE_MOE_TS_MAXLAYER; l++)
        for (int p = 0; p < LLMENGINE_MOE_TS_NPROJ; p++)
            for (int k = 0; k < LLMENGINE_MOE_TS_NKIND; k++) {
                const LlmEngineMoeTsSlice* t = &g_moe_ts.slice[l][p][k];
                if (t->map >= 0) tot += (uint64_t)t->stride * (uint64_t)t->n;
            }
    return tot;
}

/* Wrapper GENERICO (UNICO punto) per lo streaming esperti: TUTTE le ffn MoE chiamano questo al
 * posto di llm_mlx_moe_gather_mm. Se l'ingest e' attivo, rimappa gli indici GLOBALI -> LOCALI e
 * rimpiazza w/s/b con lo stacked dei k esperti attivi (dall'ingest, per layer+proj); altrimenti
 * DELEGA al gather esistente (path invariato). `proj`: 0=gate 1=up 2=down. */
int llm_mlx_moe_gather_streamed(const ModelConfig* cfg, int layer, int proj,
        mlx_array* res, mlx_array x, mlx_array w, mlx_array s, mlx_array b, mlx_array lhs, mlx_array idxs,
        bool transpose, mlx_optional_int gsz, mlx_optional_int bts, const char* mode, bool sorted, mlx_stream st) {
    const LlmEngineMoeTsIngest* ing = llm_moe_swap_ingest();
    if (ing == NULL)
        return llm_mlx_moe_gather_mm(cfg, res, x, w, s, b, lhs, idxs, transpose, gsz, bts, mode, sorted, st);
    /* TurboQuant TERNARY: array pieni RESIDENTI + indici GLOBALI. Niente remap (`unique`/`pos`),
     * niente `mlx_array_eval(idxs)` (sync per-layer) e niente memcpy di stacking: stesso percorso
     * del resident affine (95 tok/s) + rotazione online dell'input. */
    if (llmengine_moe_ts_is_tq(ing, layer, proj)) {
        uint32_t cbb = 0; int cbn = 0;
        llmengine_moe_ts_aux_get(ing, layer, proj, 1, &cbb, &cbn);
        if (cbn == 3) {
            /* Esperti PACKED (scelta API del motore `llm_ai_mlx_set_tq_memory_opt`, NON env): kernel TQ
             * diretto sui pesi packed -> RSS basso. on=0 -> affine-2bit residenti + gather_qmm MLX. */
            if (llm_mlx_tq_mem_opt()) {
                if (llm_mlx_tq_gather_packed(ing, layer, proj, res, x, w, s, idxs, transpose, st) == 0)
                    return 0;
            }
            /* PREFILL / fallback: array pieni affine-2bit residenti + gather_qmm. */
            const mlx_dtype cdt = mlx_array_dtype(x);
            mlx_array fw, fs, fb;
            if (llm_mlx_tq_affine_full(ing, layer, proj, cdt, &fw, &fs, &fb) == 0) {
                int grp = 64;
                const LlmEngineMoeTsSlice* sts = llmengine_moe_ts_slice(ing, layer, proj, 1);
                uint32_t sb2 = 0; int in2 = 0;
                llmengine_moe_ts_aux_get(ing, layer, proj, 0, &sb2, &in2);
                if (sts && sts->d2 > 0 && in2 > 0 && in2 % sts->d2 == 0) grp = in2 / sts->d2;
                mlx_array x_rot = mlx_array_new();
                mlx_array x_use = x;
                if (llm_mlx_tq_rotate_input(ing, layer, proj, x, st, cdt, &x_rot) == 0) x_use = x_rot;
                int rc = llm_mlx_gather_qmm_guarded(res, x_use, fw, fs, fb, lhs, idxs, transpose,
                                        (mlx_optional_int){grp, true}, (mlx_optional_int){2, true},
                                        "affine", sorted, st);
                mlx_array_free(x_rot);
                return rc;
            }
        }
    }
    if (mlx_array_eval(idxs) != 0) return -1;
    const uint32_t* idp = mlx_array_data_uint32(idxs);
    if (idp == NULL) return -1;
    /* Forma ORIGINALE di idxs (es. [n_tokens, top_k] nel ramo unsorted, dove x ha batch shape
     * (n_tokens, top_k, ...)): va preservata su pos_arr, altrimenti mlx_gather_qmm broadcasta x
     * contro un pos_arr 1-D [M] invece che contro la forma [n_tokens, top_k], producendo un
     * output di batch shape SBAGLIATA (silenzioso su gate/up, esplode in broadcast_shapes al
     * down). Port del fix originale acacd768a. */
    const int idxs_nd = (int)mlx_array_ndim(idxs);
    int idxs_shape[8];
    for (int i = 0; i < idxs_nd && i < 8; i++) idxs_shape[i] = (int)mlx_array_dim(idxs, i);
    const int M = (int)mlx_array_size(idxs);
    int* ids  = (int*)malloc((size_t)M * sizeof(int));
    int* pos  = (int*)malloc((size_t)M * sizeof(int));
    int* uniq = (int*)malloc((size_t)M * sizeof(int));
    if (!ids || !pos || !uniq) { free(ids); free(pos); free(uniq); return -1; }
    static int seen[8192];
    { int ne = cfg->n_experts; if (ne > 8192) ne = 8192; for (int i = 0; i < ne; i++) seen[i] = -1; }
    for (int i = 0; i < M; i++) ids[i] = (int)idp[i];
    int nu = llm_mlx_stream_unique(ids, M, cfg->n_experts, seen, pos, uniq);
    int rc = -1;
    LlmMlxStreamTriples tr;
    mlx_array pos_arr = mlx_array_new();
    if (nu > 0 && llm_mlx_moe_swap_prepare(ing, layer, proj, uniq, nu, st, &tr) == 0) {
        if (mlx_array_set(&pos_arr, mlx_array_new_data(pos, idxs_shape, idxs_nd, MLX_UINT32)) == 0)
            rc = llm_mlx_moe_gather_mm(cfg, res, x, tr.w[proj], tr.s[proj], tr.b[proj], lhs, pos_arr,
                                       transpose, gsz, bts, mode, /*sorted=*/false, st);
        /* Libera i 3 tensori del proj corrente (gli altri slot sono array vuoti). */
        for (int pj = 0; pj < LLMENGINE_MOE_TS_NPROJ; pj++) {
            mlx_array_free(tr.w[pj]);
            mlx_array_free(tr.s[pj]);
            mlx_array_free(tr.b[pj]);
        }
    }
    mlx_array_free(pos_arr);
    /* `mlx_array_new_data` COPIA (mlx::core::array::init -> std::copy): il commento storico
     * "no-copy" era errato e `pos` veniva LEAKATO ad ogni chiamata (per layer/proj/token).
     * Ora si libera come ids/uniq. */
    free(ids); free(pos); free(uniq);
    return rc;
}

/* ── Kernel MoE-fuso STREAM-AWARE (unifica ON/OFF) ─────────────────────────────────────────────
 * Il kernel fuso (llm_mlx_moe_fused_decode) indicizza i tensori COMPLETI [E,...]. Con lo swap i
 * tensori completi NON sono caricati: invece di DISATTIVARE il fuso (guardia per-modello), si
 * costruiscono gli stacked [nu,...] dei soli esperti attivi e si rimappano gli indici a LOCALI
 * (pos), poi si chiama LO STESSO kernel. Cosi' ON e OFF condividono il percorso: nessuna
 * divergenza per-modello, un solo kernel. OFF: pesi completi + indici globali (identico a prima). */
bool llm_mlx_moe_fused_stream_aware(
    const ModelConfig* cfg, int layer, mlx_array x1, mlx_array top_inds_arr,
    mlx_array gw, mlx_array gs, mlx_array gb,
    mlx_array uw, mlx_array us, mlx_array ub,
    mlx_array dw, mlx_array ds, mlx_array db,
    int gu_bits, int gu_group, int d_bits, int d_group, int act,
    mlx_array scores_flat, mlx_array* out_summed, mlx_stream st) {
    const int din = cfg->hidden_size, dff = cfg->moe_intermediate_size, k = cfg->top_k;
    const LlmEngineMoeTsIngest* ing = llm_moe_swap_ingest();
    if (ing == NULL) /* OFF: percorso identico a prima */
        return llm_mlx_moe_fused_decode(out_summed, x1, top_inds_arr, gw, gs, gb, uw, us, ub,
                                        dw, ds, db, scores_flat, din, dff, k, gu_bits, gu_group,
                                        d_bits, d_group, act, st);
    if (mlx_array_eval(top_inds_arr) != 0) return false;
    const uint32_t* idp = mlx_array_data_uint32(top_inds_arr);
    if (idp == NULL) return false;
    int* ids = (int*)malloc((size_t)k * sizeof(int));
    int* pos = (int*)malloc((size_t)k * sizeof(int));
    int* uniq = (int*)malloc((size_t)k * sizeof(int));
    uint32_t* posu = (uint32_t*)malloc((size_t)k * sizeof(uint32_t));
    if (!ids || !pos || !uniq || !posu) { free(ids); free(pos); free(uniq); free(posu); return false; }
    static int seen[8192];
    { int ne = cfg->n_experts; if (ne > 8192) ne = 8192; for (int i = 0; i < ne; i++) seen[i] = -1; }
    for (int i = 0; i < k; i++) ids[i] = (int)idp[i];
    int nu = llm_mlx_stream_unique(ids, k, cfg->n_experts, seen, pos, uniq);
    bool ok = false;
    if (nu > 0) {
        mlx_array sw[3][3]; /* [proj][kind] */
        for (int p = 0; p < 3; p++) for (int kd = 0; kd < 3; kd++) sw[p][kd] = mlx_array_new();
        int built = 0;
        for (int p = 0; p < 3 && built >= 0; p++)
            for (int kd = 0; kd < 3; kd++)
                if (llm_mlx_xs_stack(ing, layer, p, kd, uniq, nu, st, &sw[p][kd]) != 0) { built = -1; break; }
        if (built == 0) {
            for (int i = 0; i < k; i++) posu[i] = (uint32_t)pos[i];
            const int shp[1] = { k };
            mlx_array lidx = mlx_array_new_data(posu, shp, 1, MLX_UINT32); /* copia */
            ok = llm_mlx_moe_fused_decode(out_summed, x1, lidx,
                    sw[0][0], sw[0][1], sw[0][2], sw[1][0], sw[1][1], sw[1][2],
                    sw[2][0], sw[2][1], sw[2][2], scores_flat,
                    din, dff, k, gu_bits, gu_group, d_bits, d_group, act, st);
            mlx_array_free(lidx);
        }
        for (int p = 0; p < 3; p++) for (int kd = 0; kd < 3; kd++) mlx_array_free(sw[p][kd]);
    }
    free(ids); free(pos); free(uniq); free(posu);
    return ok;
}


/* FASE 1b.1: FFN denso SwiGLU singolo

/* Cache pesi/quant FFN MoE (bug P1/P3): risolve UNA volta per layer le triple degli esperti
 * routed (gate/up/down) e la quant per-tensore; i forward successivi leggono dalla cache.
 * Gate: LLM_WCACHE=0 disabilita (fallback alla risoluzione per chiamata). */
static int g_wcache_state = -1;
bool llm_mlx_wcache_enabled(void) {
    if (g_wcache_state < 0) g_wcache_state = (llm_ai_mlx_opt_env("LLM_WCACHE") != NULL && strcmp(llm_ai_mlx_opt_env("LLM_WCACHE"), "0") == 0) ? 0 : 1;
    return g_wcache_state == 1;
}
static int g_reshape_canonical_state = -1;
bool llm_mlx_reshape_canonical_enabled(void) {
    if (g_reshape_canonical_state < 0) {
        const char* v = llm_ai_mlx_opt_env("LLM_RESHAPE_CANONICAL");
        g_reshape_canonical_state = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return g_reshape_canonical_state == 1;
}
GmoeFfnQuant* llm_mlx_moe_ffn_quant(LlmMlxModel* m, int layer_idx) {
    if (m == NULL || !llm_mlx_wcache_enabled()) return NULL;
    /* Streaming esperti: la cache per-layer materializza gli esperti FULL -> la disattivo, cosi'
     * la ffn usa il lookup lazy (mai eval-ato) e il wrapper streamed la rimpiazza. */
    if (llm_mlx_moe_swap_on()) return NULL;
    /* Percorso GGUF: pesi dequantizzati F16 (quant_mode="") senza scales -> la cache affine
     * (llm_mlx_quant_derive) non e' applicabile; il ffn usa il lookup inline + llm_mlx_moe_gather_mm. */
    if (m->cfg.quant_mode[0] == 0) return NULL;
    if (m->ffn_quant == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->ffn_quant = (GmoeFfnQuant*)calloc((size_t)n, sizeof(GmoeFfnQuant));
        if (m->ffn_quant == NULL) return NULL;
        m->ffn_quant_n = n;
    }
    if (layer_idx < 0 || layer_idx >= m->ffn_quant_n) return NULL;
    GmoeFfnQuant* c = &m->ffn_quant[layer_idx];
    if (c->valid) return c;
    char nb[256];
    bool ok = true;
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &c->ln2);
    /* Prefissi MoE: "mlp.*" (qwen3_5_moe/qwen3_moe/glm4_moe) o "block_sparse_moe.*" (Mixtral). */
    ok = false;
    for (int pi = 0; pi < 2 && !ok; pi++) {
        const char* base = (pi == 0) ? "mlp" : "block_sparse_moe";
        char su[160];
        bool rok = true;
        snprintf(su, sizeof(su), "%s.gate", base);
        tname(m->cfg.layer_prefix, layer_idx, su, nb, sizeof(nb));
        rok = rok && multishard_get_triple(&m->weights, nb, &c->rw, &c->rs, &c->rb);
        snprintf(su, sizeof(su), "%s.switch_mlp.gate_proj", base);
        tname(m->cfg.layer_prefix, layer_idx, su, nb, sizeof(nb));
        rok = rok && multishard_get_triple(&m->weights, nb, &c->gw, &c->gs, &c->gb);
        snprintf(su, sizeof(su), "%s.switch_mlp.up_proj", base);
        tname(m->cfg.layer_prefix, layer_idx, su, nb, sizeof(nb));
        rok = rok && multishard_get_triple(&m->weights, nb, &c->uw, &c->us, &c->ub);
        snprintf(su, sizeof(su), "%s.switch_mlp.down_proj", base);
        tname(m->cfg.layer_prefix, layer_idx, su, nb, sizeof(nb));
        rok = rok && multishard_get_triple(&m->weights, nb, &c->dw, &c->ds, &c->db);
        ok = rok;
    }
    if (!ok) return NULL;
    llm_mlx_quant_derive(c->gw, c->gs, m->cfg.quant_group_size, m->cfg.quant_bits, "affine",
                         &c->gg, &c->gbts, c->gmode, sizeof(c->gmode));
    llm_mlx_quant_derive(c->uw, c->us, m->cfg.quant_group_size, m->cfg.quant_bits, "affine",
                         &c->ug, &c->ubts, c->umode, sizeof(c->umode));
    llm_mlx_quant_derive(c->dw, c->ds, m->cfg.quant_group_size, m->cfg.quant_bits, "affine",
                         &c->dg, &c->dbts, c->dmode, sizeof(c->dmode));
    c->valid = true;
    /* LLMENGINE_MOE_GATEUP_FUSE: precondizione strutturale (stessa di rapidMLX _can_fuse) -- stessa
     * quantizzazione gate/up e stessa presenza di bias -- prima di concatenare. Asse 1 =
     * out_features (verificato: mlx_gather_qmm chiamato con transpose=true su w[n_experti,
     * out_features,in_features_packed], stesso asse usato da rapidMLX in moe_fusion.py). */
    c->has_gate_up_fused = llm_mlx_moe_gateup_fuse_enabled() &&
                           (c->gg == c->ug && c->gbts == c->ubts &&
                            strcmp(c->gmode, c->umode) == 0 &&
                            (c->gb.ctx != NULL) == (c->ub.ctx != NULL));
    if (c->has_gate_up_fused) {
        mlx_vector_array wv = mlx_vector_array_new();
        mlx_vector_array_append_value(wv, c->gw);
        mlx_vector_array_append_value(wv, c->uw);
        bool fok = mlx_concatenate_axis(&c->guw, wv, 1, m->stream) == 0;
        mlx_vector_array_free(wv);
        mlx_vector_array sv = mlx_vector_array_new();
        mlx_vector_array_append_value(sv, c->gs);
        mlx_vector_array_append_value(sv, c->us);
        fok = fok && mlx_concatenate_axis(&c->gus, sv, 1, m->stream) == 0;
        mlx_vector_array_free(sv);
        if (fok && c->gb.ctx != NULL) {
            mlx_vector_array bv = mlx_vector_array_new();
            mlx_vector_array_append_value(bv, c->gb);
            mlx_vector_array_append_value(bv, c->ub);
            fok = mlx_concatenate_axis(&c->gub, bv, 1, m->stream) == 0;
            mlx_vector_array_free(bv);
        }
        if (fok) {
            fok = mlx_array_eval(c->guw) == 0 && mlx_array_eval(c->gus) == 0 &&
                  (c->gub.ctx == NULL || mlx_array_eval(c->gub) == 0);
        }
        c->has_gate_up_fused = fok;
        if (!fok) fprintf(stderr, "[wcache] layer %d: fusione gate_up fallita, fallback ai due gather_qmm\n", layer_idx);
    }
    /* shared_expert (best-effort: se manca, has_shared=false e la funzione usa il fallback) */
    {
        bool sok = true;
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.gate_proj", nb, sizeof(nb));
        sok = sok && multishard_get_triple(&m->weights, nb, &c->se_w, &c->se_s, &c->se_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.up_proj", nb, sizeof(nb));
        sok = sok && multishard_get_triple(&m->weights, nb, &c->su_w, &c->su_s, &c->su_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert.down_proj", nb, sizeof(nb));
        sok = sok && multishard_get_triple(&m->weights, nb, &c->sd_w, &c->sd_s, &c->sd_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.shared_expert_gate", nb, sizeof(nb));
        sok = sok && multishard_get_triple(&m->weights, nb, &c->sg_w, &c->sg_s, &c->sg_b);
        c->has_shared = sok;
    }
    if (llm_env_get("LLM_WDEBUG")) fprintf(stderr, "[wcache] layer %d risolto (g=%d/%d/%d b=%d/%d/%d shared=%d)\n",
                                      layer_idx, c->gg, c->ug, c->dg, c->gbts, c->ubts, c->dbts, c->has_shared ? 1 : 0);
    return c;
}
