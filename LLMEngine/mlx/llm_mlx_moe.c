/* ============================================================================
 * llm_mlx_moe.c — dominio MoE: gather gate/up fuso, gather_mm, qwen3_moe_ffn_block,
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
 * API del motore (NON env): il consumer (GMBasic via BASIC / FastLLM) la imposta.
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
 * modello da `oracoli/llm_gather_qmm_sorted_probe` (forme reali esperti GLM-4.5): M<=32768
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
void llm_ai_mlx_set_moe_swap_dir(const char* dir) {
    if (dir) {
        snprintf(g_moe_swap_dir, sizeof g_moe_swap_dir, "%s", dir);
    } else {
        g_moe_swap_dir[0] = '\0';
    }
    g_moe_ts_state = -1;
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
     * down). Port del fix GMBasic acacd768a. */
    const int idxs_nd = (int)mlx_array_ndim(idxs);
    int idxs_shape[8];
    for (int i = 0; i < idxs_nd && i < 8; i++) idxs_shape[i] = (int)mlx_array_dim(idxs, i);
    const int M = (int)mlx_array_size(idxs);
    int* ids  = (int*)malloc((size_t)M * sizeof(int));
    int* pos  = (int*)malloc((size_t)M * sizeof(int));
    int* uniq = (int*)malloc((size_t)M * sizeof(int));
    if (!ids || !pos || !uniq) { free(ids); free(pos); free(uniq); return -1; }
    static int seen[8192];
    for (int i = 0; i < 8192; i++) seen[i] = -1;
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
    free(ids); free(uniq); /* pos vivo: referenziato da pos_arr (new_data no-copy) */
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
    for (int i = 0; i < 8192; i++) seen[i] = -1;
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

bool qwen3_moe_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen3_moe_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    /* Cache pesi per-layer (P1/P3): norm e router risolti una volta al load; fallback legacy. */
    GmoeFfnQuant* fq = llm_mlx_moe_ffn_quant(m, layer_idx);
    mlx_array ln2_w = mlx_array_new();
    if (fq != NULL) {
        ok = ok && mlx_array_set(&ln2_w, fq->ln2) == 0;
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &ln2_w);
    }
    mlx_array normed2 = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed2, h, ln2_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "post_attention_layernorm"); }

    mlx_array gate_w = mlx_array_new(), gate_s = mlx_array_new(), gate_b = mlx_array_new();
    if (fq != NULL) {
        ok = ok && mlx_array_set(&gate_w, fq->rw) == 0 && mlx_array_set_optional(&gate_s, fq->rs) &&
                   mlx_array_set_optional(&gate_b, fq->rb);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "mlp.gate", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_w, &gate_s, &gate_b);
    }
    mlx_array gates = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&gates, normed2, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "router matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "mlp.gate"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&gates, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }

    mlx_array gates_sm = mlx_array_new();
    /* bfloat16 refactor: mlx_lm calcola i logit del router in bfloat16 (input bf16). Il softmax
     * precise in float32 sui logit float32 amplifica la differenza di selezione esperti vs la
     * reference (root cause della divergenza MoE Qwen3.6-35B-A3B, §9). Se bfloat16_input e' attivo,
     * abbassa i logit a bf16 PRIMA del softmax per allineare la selezione. */
    if (ok && m->cfg.bfloat16_input)
        ok = (mlx_array_dtype(gates) == g_act16) || (mlx_astype(&gates, gates, g_act16, m->stream) == 0);
    if (ok) { ok = mlx_softmax_axis(&gates_sm, gates, -1, /*precise=*/true, m->stream) == 0; CHK(ok, "router softmax"); }

    mlx_array part = mlx_array_new(), top_inds = mlx_array_new();
    if (ok) {
        ok = mlx_argpartition_axis(&part, gates_sm, /*kth=*/-m->cfg.top_k, /*axis=*/-1, m->stream) == 0;
        CHK(ok, "router argpartition");
        if (ok) {
            const int num_experts = (int)mlx_array_dim(part, 2);
            const int start[3] = {0, 0, num_experts - m->cfg.top_k};
            const int stop[3] = {1, n_tokens, num_experts};
            const int strides[3] = {1, 1, 1};
            mlx_array top_inds_view = mlx_array_new();
            ok = mlx_slice(&top_inds_view, part, start, 3, stop, 3, strides, 3, m->stream) == 0;
            CHK(ok, "top-k slice");
            if (ok) {
                /* mlx_slice restituisce una VIEW non contigua (bug reale trovato dal
                 * vivo: per n_tokens>1 gli stride restano quelli del tensore PART
                 * originale -- (n_tokens*num_experts, num_experts, 1) invece di
                 * (n_tokens*top_k, top_k, 1) -- il readback host flat che segue
                 * assumeva erroneamente un buffer denso row-major, leggendo la
                 * posizione j=0 per coincidenza corretta ma OGNI riga successiva da
                 * un offset sbagliato. mlx_contiguous forza una copia densa prima
                 * del memcpy in host memory -- con L=1 (com'era prima del batched
                 * prefill) il bug non si manifestava mai: una sola riga rende
                 * l'offset "sbagliato" comunque coincidente col corretto. */
                ok = mlx_contiguous(&top_inds, top_inds_view, /*allow_col_major=*/false, m->stream) == 0;
                CHK(ok, "top-k contiguous copy");
            }
            mlx_array_free(top_inds_view);
        }
    }

    /* Leva A: niente readback host del router. top_inds (lazy) e gates_sm restano su
     * device; gli indici per gather_qmm e i pesi softmax (con eventuale rinormalizzazione
     * norm_topk_prob) si derivano con reshape/astype/take_along_axis. Un solo sync per
     * token (il final_step), come in gpt_oss L<64. */
    mlx_array gate_proj_w = mlx_array_new(), gate_proj_s = mlx_array_new(), gate_proj_b = mlx_array_new();
    mlx_array up_proj_w = mlx_array_new(), up_proj_s = mlx_array_new(), up_proj_b = mlx_array_new();
    mlx_array down_proj_w = mlx_array_new(), down_proj_s = mlx_array_new(), down_proj_b = mlx_array_new();
    if (fq != NULL) {
        ok = ok && mlx_array_set(&gate_proj_w, fq->gw) == 0 && mlx_array_set_optional(&gate_proj_s, fq->gs) &&
                   mlx_array_set_optional(&gate_proj_b, fq->gb);
        ok = ok && mlx_array_set(&up_proj_w, fq->uw) == 0 && mlx_array_set_optional(&up_proj_s, fq->us) &&
                   mlx_array_set_optional(&up_proj_b, fq->ub);
        ok = ok && mlx_array_set(&down_proj_w, fq->dw) == 0 && mlx_array_set_optional(&down_proj_s, fq->ds) &&
                   mlx_array_set_optional(&down_proj_b, fq->db);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.gate_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_proj_w, &gate_proj_s, &gate_proj_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.up_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &up_proj_w, &up_proj_s, &up_proj_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.down_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &down_proj_w, &down_proj_s, &down_proj_b);
    }

    /* FASE B: dispatch MoE raggruppato per esperto via mlx_gather_qmm --
     * sostituisce il loop per-token/per-esperto della Fase A (mlx_take_axis +
     * matmul a riga singola, O(n_tokens*top_k) chiamate) con un numero fisso
     * di chiamate MLX per layer, indipendentemente da n_tokens. */
    mlx_array x1 = mlx_array_new();
    mlx_array moe_out = mlx_array_new();
    mlx_array no_lhs = mlx_array_new(); /* "may be null": .ctx==NULL -> std::nullopt lato mlx-c */
    mlx_array top_inds_arr = mlx_array_new();
    uint32_t* im_expert = NULL; uint32_t* im_src = NULL; int im_rows = 0; /* §E2 imatrix collector */
    if (ok) {
        const int ti2_shape[2] = {n_tokens, m->cfg.top_k};
        mlx_array tt = mlx_array_new();
        ok = mlx_reshape(&tt, top_inds, ti2_shape, 2, m->stream) == 0;
        if (ok) ok = mlx_astype(&top_inds_arr, tt, MLX_UINT32, m->stream) == 0;
        mlx_array_free(tt);
        CHK(ok, "top_inds device reshape/astype");
    }

    if (ok) {
        const int x1_shape[4] = {n_tokens, 1, 1, m->cfg.hidden_size};
        ok = mlx_reshape(&x1, normed2, x1_shape, 4, m->stream) == 0;
        CHK(ok, "moe x1 reshape");
    }

    mlx_optional_int gsz_opt = {gsz, true};
    mlx_optional_int bts_opt = {bts, true};
    bool do_sort = ok && ((int64_t)n_tokens * (int64_t)m->cfg.top_k) >= 64;
    /* LLMENGINE_MOE_NOSORT=1 (2026-09-17): il ramo sorted aggiunge argsort x2 + floor_divide + astype +
     * take + take_along_axis + contiguous (8 op con confine host) per layer/token. Misurato su
     * Mixtral-8x22B: quelle op costano 12,4% di GPU (GatherAxis [1,64,2] = 3,3 ms per 128 elementi)
     * + quota CPU -> sospetta serializzazione. Il ramo unsorted e' quello gia' usato dal decode. */
    /* Mixtral (2026-09-17): il ramo sorted del prefill costa 8 op/layer (argsort x2, floor_divide,
     * astype, take, take_along_axis, contiguous) misurate PATOLOGICHE (GatherAxis [1,64,2] = 3,3 ms
     * per 128 elementi, 12,4% del GPU; Softmax [1,64,2] 0,14 ms; Matmul [64,8] 0,12 ms = ~1000x la
     * loro taglia -> firma di sync/serializzazione). A/B (back-to-back, load 5,15 => da riconfermare
     * in finestra quieta): prefill 6,6 -> **8,3 tok/s (+26%)** con **IDS IDENTICI** (92675131).
     * DEFAULT unsorted per mixtral; kill-switch LLMENGINE_MOE_SORT=1; LLMENGINE_MOE_NOSORT=1 forzato resta valido. */
    if (strcmp(m->cfg.model_type, "mixtral") == 0 && llm_ai_mlx_opt_env("LLMENGINE_MOE_SORT") == NULL) do_sort = false;
    if (llm_ai_mlx_opt_env("LLMENGINE_MOE_NOSORT") != NULL) do_sort = false;
    if (llm_moe_swap_ingest() != NULL) do_sort = false; /* streaming esperti: gather locale (unsorted) */
    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new();
    mlx_array sig = mlx_array_new(), silu = mlx_array_new(), act = mlx_array_new();
    mlx_array down_out = mlx_array_new();
    mlx_array probs = mlx_array_new();
    mlx_array scores_arr = mlx_array_new(), scaled = mlx_array_new();
    mlx_array summed = mlx_array_new();
    if (ok && do_sort) {
        const int M = n_tokens * m->cfg.top_k;
        mlx_array tt = mlx_array_new();
        { const int ts[1] = {M}; if (ok) ok = mlx_reshape(&tt, top_inds_arr, ts, 1, m->stream) == 0; }
        mlx_array order = mlx_array_new();
        if (ok) ok = mlx_argsort(&order, tt, m->stream) == 0;
        mlx_array inv = mlx_array_new();
        if (ok) ok = mlx_argsort(&inv, order, m->stream) == 0;
        mlx_array df = mlx_array_new();
        { mlx_array kk = mlx_array_new_int(m->cfg.top_k);
          if (ok) ok = mlx_floor_divide(&df, order, kk, m->stream) == 0; mlx_array_free(kk); }
        mlx_array dv = mlx_array_new();
        if (ok) ok = mlx_astype(&dv, df, MLX_UINT32, m->stream) == 0;
        mlx_array idxs = mlx_array_new();
        if (ok) ok = mlx_take(&idxs, tt, order, m->stream) == 0;
        mlx_array dvm4 = mlx_array_new();
        { const int ms[4] = {M, 1, 1, 1}; if (ok) ok = mlx_reshape(&dvm4, dv, ms, 4, m->stream) == 0; }
        mlx_array rn = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&rn, x1, dvm4, 0, m->stream) == 0;
        mlx_array rnc = mlx_array_new();
        if (ok) ok = mlx_contiguous(&rnc, rn, false, m->stream) == 0;
        mlx_array xs = mlx_array_new();
        { const int xs3[3] = {M, 1, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&xs, rnc, xs3, 3, m->stream) == 0; }
        if (ok && !llm_mlx_moe_gateup_gather(fq, xs, no_lhs, idxs, /*sorted=*/true, gsz_opt, bts_opt,
                                             "affine", &gate_out, &up_out, m->stream)) {
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, xs, gate_proj_w, gate_proj_s, gate_proj_b, no_lhs, idxs,
                                        true, gsz_opt, bts_opt, "affine", true, m->stream) == 0;
            if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, xs, up_proj_w, up_proj_s, up_proj_b, no_lhs, idxs,
                                        true, gsz_opt, bts_opt, "affine", true, m->stream) == 0;
        }
        /* LLM_FUSE_SWIGLU_PREFILL (roadmap P3, default OFF): closure compilata SOLO qui,
         * nel ramo sorted (M=n_tokens*top_k>=64, prefill batched) -- mai nel ramo unsorted
         * sotto (decode a n_tokens==1, dove la stessa tecnica e' gia' falsificata per GeGLU). */
        if (ok && llm_mlx_fuse_swiglu_prefill_enabled()) {
            ok = llm_mlx_swiglu_compiled(gate_out, up_out, &act, m->stream);
            CHK(ok, "moe swiglu (compiled prefill)");
        } else if (ok) {
            ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0;
            if (ok) ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0;
            if (ok) ok = mlx_multiply(&act, silu, up_out, m->stream) == 0;
        }
        if (ok) ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, down_proj_w, down_proj_s, down_proj_b, no_lhs, idxs,
                                    true, gsz_opt, bts_opt, "affine", true, m->stream) == 0;
        /* §E2 imatrix collector (path SORTED): readback idxs (esperto per riga) + inv (riga sorted
         * -> riga token-major). Solo in calibrazione (im.active). */
        if (ok && m->im.active) {
            const int Mi = n_tokens * m->cfg.top_k; im_rows = Mi;
            if (mlx_array_eval(idxs) == 0 && mlx_array_eval(inv) == 0 &&
                mlx_array_data_uint32(idxs) != NULL && mlx_array_data_uint32(inv) != NULL) {
                im_expert = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
                im_src = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
                if (im_expert && im_src) {
                    memcpy(im_expert, mlx_array_data_uint32(idxs), (size_t)Mi * sizeof(uint32_t));
                    memcpy(im_src, mlx_array_data_uint32(inv), (size_t)Mi * sizeof(uint32_t));
                }
            }
        }
        mlx_array invm = mlx_array_new();
        { const int is3[3] = {M, 1, 1}; if (ok) ok = mlx_reshape(&invm, inv, is3, 3, m->stream) == 0; }
        mlx_array duns = mlx_array_new();
        if (ok) ok = mlx_take_along_axis(&duns, down_out, invm, 0, m->stream) == 0;
        mlx_array_free(invm);
        mlx_array dh3 = mlx_array_new();
        { const int d3[3] = {n_tokens, m->cfg.top_k, m->cfg.hidden_size};
          if (ok) ok = mlx_reshape(&dh3, duns, d3, 3, m->stream) == 0; }
        if (ok) ok = mlx_take_along_axis(&probs, gates_sm, top_inds, -1, m->stream) == 0;
        if (ok && m->cfg.norm_topk_prob) {
            mlx_array psum = mlx_array_new();
            if (ok) ok = mlx_sum_axis(&psum, probs, 2, true, m->stream) == 0;
            if (ok) ok = mlx_divide(&probs, probs, psum, m->stream) == 0;
            mlx_array_free(psum);
        }
        {
            mlx_array pr = mlx_array_new();
            const int pr2[2] = {n_tokens, m->cfg.top_k};
            if (ok) ok = mlx_reshape(&pr, probs, pr2, 2, m->stream) == 0;
            mlx_array sc3 = mlx_array_new();
            const int sc3s[3] = {n_tokens, m->cfg.top_k, 1};
            if (ok) ok = mlx_reshape(&sc3, pr, sc3s, 3, m->stream) == 0;
            mlx_array scaled3 = mlx_array_new();
            if (ok) ok = mlx_multiply(&scaled3, dh3, sc3, m->stream) == 0;
            mlx_array sumt = mlx_array_new();
            if (ok) ok = mlx_sum_axis(&sumt, scaled3, 1, false, m->stream) == 0;
            const int o3[3] = {1, n_tokens, m->cfg.hidden_size};
            if (ok) ok = mlx_reshape(&moe_out, sumt, o3, 3, m->stream) == 0;
            mlx_array_free(pr); mlx_array_free(sc3); mlx_array_free(scaled3); mlx_array_free(sumt);
        }
        mlx_array_free(tt); mlx_array_free(order); mlx_array_free(inv); mlx_array_free(idxs); mlx_array_free(dvm4);
        mlx_array_free(rn); mlx_array_free(rnc); mlx_array_free(xs); mlx_array_free(duns);
        mlx_array_free(dh3);
    } else
    {

    /* Score per (token,esperto) su device, spostato qui (prima serviva solo dopo down_out,
     * ma non dipende da gate/up/down) perche' il kernel fuso sotto ne ha bisogno subito.
     * gates_sm [1,T,E], top_inds [1,T,K] -> probs [1,T,K]. */
    if (ok) ok = mlx_take_along_axis(&probs, gates_sm, top_inds, /*axis=*/-1, m->stream) == 0;
    if (ok && m->cfg.norm_topk_prob) {
        mlx_array psum = mlx_array_new();
        if (ok) ok = mlx_sum_axis(&psum, probs, /*axis=*/2, /*keepdims=*/true, m->stream) == 0;
        if (ok) ok = mlx_divide(&probs, probs, psum, m->stream) == 0;
        mlx_array_free(psum);
    }

    /* Kernel Metal MoE decode fuso (LLMENGINE_MOE_FUSE_MOE_DECODE, sperimentale, default OFF):
     * sostituisce 2-3 gather_qmm + sigmoid/multiply con due dispatch fusi (gate+up+SwiGLU,
     * poi down pesata per score). Solo per n_tokens==1 (il caso per cui il kernel e' scritto
     * -- K esperti di UN token) e bits in {4,8} (il kernel non porta il caso 6-bit).
     * Fallback silenzioso al path esistente sotto se una condizione non e' soddisfatta o il
     * kernel stesso fallisce -- mai un errore bloccante. Vedi llm_mlx_moe_fused_kernel.h. */
    bool moe_fused_done = false;
    /* Streaming esperti (offload): il kernel fuso usa DIRETTAMENTE i
     * tensori COMPLETI gate/up/down, non gli stacked locali di llm_mlx_moe_gather_streamed ->
     * sotto SWAP non sono i pesi giusti. Stessa guardia gia' in llm_mlx_moe_gateup_gather
     * (altra fusione, stesso motivo). Port del fix GMBasic acacd768a. */
    if (ok && n_tokens == 1 && (bts == 4 || bts == 8) && llm_mlx_fuse_moe_decode_enabled() &&
        !llm_mlx_moe_swap_on()) {
        const int din = m->cfg.hidden_size, dff = m->cfg.moe_intermediate_size;
        const int k = m->cfg.top_k;
        mlx_array x_flat = mlx_array_new(), idx_flat = mlx_array_new();
        mlx_array pr1 = mlx_array_new(), sc_flat = mlx_array_new();
        const int xsh[1] = {din};
        bool prep = mlx_reshape(&x_flat, x1, xsh, 1, m->stream) == 0;
        const int ksh[1] = {k};
        prep = prep && mlx_reshape(&idx_flat, top_inds_arr, ksh, 1, m->stream) == 0;
        prep = prep && mlx_reshape(&pr1, probs, ksh, 1, m->stream) == 0;
        prep = prep && mlx_astype(&sc_flat, pr1, MLX_FLOAT32, m->stream) == 0;
        if (prep) {
            mlx_array fused_summed = mlx_array_new();
            bool fok = llm_mlx_moe_fused_decode(&fused_summed, x_flat, idx_flat,
                gate_proj_w, gate_proj_s, gate_proj_b, up_proj_w, up_proj_s, up_proj_b,
                down_proj_w, down_proj_s, down_proj_b, sc_flat,
                din, dff, k, bts, gsz, bts, gsz, /*act=SwiGLU*/0, m->stream);
            if (fok) {
                mlx_array cast = mlx_array_new();
                fok = mlx_astype(&cast, fused_summed, mlx_array_dtype(h), m->stream) == 0;
                if (fok) {
                    const int o3[3] = {1, n_tokens, m->cfg.hidden_size};
                    fok = mlx_reshape(&moe_out, cast, o3, 3, m->stream) == 0;
                }
                mlx_array_free(cast);
            }
            if (fok && llm_mlx_fuse_moe_decode_verify_enabled()) {
                /* _VERIFY: non usato per calcolare il risultato (moe_out resta quello del
                 * kernel fuso), solo per stampare il diff -- zero rischio, misura fiducia. */
                mlx_array ref_gate = mlx_array_new(), ref_up = mlx_array_new(), ref_act = mlx_array_new();
                mlx_array ref_down = mlx_array_new(), ref_sig = mlx_array_new();
                bool rok = mlx_gather_qmm(&ref_gate, x1, gate_proj_w, gate_proj_s, gate_proj_b,
                        no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0 &&
                    mlx_gather_qmm(&ref_up, x1, up_proj_w, up_proj_s, up_proj_b,
                        no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0;
                if (rok) rok = mlx_sigmoid(&ref_sig, ref_gate, m->stream) == 0;
                mlx_array ref_silu = mlx_array_new();
                if (rok) rok = mlx_multiply(&ref_silu, ref_gate, ref_sig, m->stream) == 0;
                if (rok) rok = mlx_multiply(&ref_act, ref_silu, ref_up, m->stream) == 0;
                if (rok) rok = mlx_gather_qmm(&ref_down, ref_act, down_proj_w, down_proj_s, down_proj_b,
                        no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0;
                mlx_array ref_scaled = mlx_array_new(), ref_scores4 = mlx_array_new(), ref_summed = mlx_array_new();
                if (rok) { const int sc4[4] = {1, k, 1, 1}; rok = mlx_reshape(&ref_scores4, pr1, sc4, 4, m->stream) == 0; }
                if (rok) rok = mlx_multiply(&ref_scaled, ref_down, ref_scores4, m->stream) == 0;
                if (rok) rok = mlx_sum_axis(&ref_summed, ref_scaled, 1, false, m->stream) == 0;
                if (rok) {
                    mlx_array af = mlx_array_new(), bf = mlx_array_new();
                    mlx_array ref_summed_flat = mlx_array_new();
                    const int rsh[1] = {din};
                    if (mlx_reshape(&ref_summed_flat, ref_summed, rsh, 1, m->stream) == 0 &&
                        mlx_astype(&af, fused_summed, MLX_FLOAT32, m->stream) == 0 &&
                        mlx_astype(&bf, ref_summed_flat, MLX_FLOAT32, m->stream) == 0 &&
                        mlx_array_eval(af) == 0 && mlx_array_eval(bf) == 0) {
                        const float* ad = mlx_array_data_float32(af);
                        const float* bd = mlx_array_data_float32(bf);
                        if (ad && bd) {
                            double maxdiff = 0, sumsq = 0, refsq = 0;
                            for (int vi = 0; vi < din; vi++) {
                                double d = fabs((double)ad[vi] - (double)bd[vi]);
                                if (d > maxdiff) maxdiff = d;
                                sumsq += d * d; refsq += (double)bd[vi] * (double)bd[vi];
                            }
                            double rms = refsq > 0 ? sqrt(sumsq / refsq) : sqrt(sumsq);
                            fprintf(stderr, "[fuse-moe-verify] layer=%d max_abs_diff=%.6g rel_rms=%.6g\n",
                                    layer_idx, maxdiff, rms);
                        }
                    }
                    mlx_array_free(af); mlx_array_free(bf); mlx_array_free(ref_summed_flat);
                }
                mlx_array_free(ref_gate); mlx_array_free(ref_up); mlx_array_free(ref_act);
                mlx_array_free(ref_down); mlx_array_free(ref_sig); mlx_array_free(ref_silu);
                mlx_array_free(ref_scaled); mlx_array_free(ref_scores4); mlx_array_free(ref_summed);
            }
            moe_fused_done = fok;
            mlx_array_free(fused_summed);
        }
        mlx_array_free(x_flat); mlx_array_free(idx_flat); mlx_array_free(pr1); mlx_array_free(sc_flat);
    }

    /* SWAP esperti: spostato nel wrapper generico llm_mlx_moe_gather_streamed (sotto),
     * usato da TUTTE le ffn MoE. Qui nessuna logica per-ffn. */

    if (ok && !moe_fused_done && !llm_mlx_moe_gateup_gather(fq, x1, no_lhs, top_inds_arr, /*sorted=*/false, gsz_opt, bts_opt,
                                         "affine", &gate_out, &up_out, m->stream)) {
        if (ok) {
            ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 0, &gate_out, x1, gate_proj_w, gate_proj_s, gate_proj_b,
                                 no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0;
            CHK(ok, "gather_qmm gate_proj");
        }
        if (ok) {
            ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 1, &up_out, x1, up_proj_w, up_proj_s, up_proj_b,
                                 no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0;
            CHK(ok, "gather_qmm up_proj");
        }
    }
    /* gate_out/up_out: [n_tokens, top_k, 1, moe_intermediate] */

    /* Replica della vittoria densa sulla MoE: kernel Metal fuso silu(g)*u (bit-fedele,
     * 1 dispatch invece di 3) gated da LLMMETAL_FUSE_SWIGLU_METAL (default ON); fallback 3-op. */
    bool moe_swiglu_done = false;
    if (ok && !moe_fused_done && llm_mlx_fuse_swiglu_metal_enabled())
        moe_swiglu_done = llm_mlx_swiglu_metal(gate_out, up_out, &act, m->stream);
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0; CHK(ok, "moe sigmoid"); }
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0; CHK(ok, "moe silu"); }
    if (ok && !moe_fused_done && !moe_swiglu_done) { ok = mlx_multiply(&act, silu, up_out, m->stream) == 0; CHK(ok, "moe swiglu"); }
    /* act: [n_tokens, top_k, 1, moe_intermediate] -- batch shape (n_tokens,
     * top_k) combacia ESATTAMENTE con top_inds_arr, quindi lhs_indices
     * omesso qui e' gia' l'indice identita' giusto (nessun broadcast). */
    /* §E2 imatrix collector (path UNSORTED): row_expert = top_inds_arr, row_src = identita'. */
    if (ok && m->im.active && !do_sort && top_inds_arr.ctx != NULL) {
        const int Mi = n_tokens * m->cfg.top_k; im_rows = Mi;
        if (mlx_array_eval(top_inds_arr) == 0 && mlx_array_data_uint32(top_inds_arr) != NULL) {
            im_expert = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
            im_src = (uint32_t*)malloc((size_t)Mi * sizeof(uint32_t));
            if (im_expert && im_src) {
                memcpy(im_expert, mlx_array_data_uint32(top_inds_arr), (size_t)Mi * sizeof(uint32_t));
                for (int r = 0; r < Mi; r++) im_src[r] = (uint32_t)r;
            }
        }
    }

    if (ok && !moe_fused_done) {
        ok = llm_mlx_moe_gather_streamed(&m->cfg, layer_idx, 2, &down_out, act, down_proj_w, down_proj_s, down_proj_b,
                             no_lhs, top_inds_arr, true, gsz_opt, bts_opt, "affine", false, m->stream) == 0;
        CHK(ok, "gather_qmm down_proj");
    }
    if (ok && !moe_fused_done) {
        /* Il gather TQ (llm_mlx_tq_gather_packed) ritorna [n_tokens*top_k, 1, H]; il gather affine
         * NON-TQ ritorna [n_tokens, top_k, 1, H]. Normalizza a [n_tokens, top_k, H] cosi' il
         * broadcast con scores [n_tokens, top_k, 1] e la somma su axis=1 sono corretti per OGNI
         * n_tokens (con n_tokens==1 il flatten coincide con top_k -> funzionava per caso). */
        mlx_array d3 = mlx_array_new();
        const int dsh[3] = { n_tokens, m->cfg.top_k, m->cfg.hidden_size };
        if (mlx_reshape(&d3, down_out, dsh, 3, m->stream) == 0) mlx_array_set(&down_out, d3);
        else ok = false;
        mlx_array_free(d3);
    }
    /* down_out: [n_tokens, top_k, hidden]. probs (score per token/esperto, con eventuale
     * rinormalizzazione norm_topk_prob) e' gia' stato calcolato piu' sopra, prima del
     * tentativo di kernel fuso -- non ricalcolarlo qui. */
    if (ok && !moe_fused_done) {
        const int pr_shape[2] = {n_tokens, m->cfg.top_k};
        const int sc_shape[3] = {n_tokens, m->cfg.top_k, 1};
        ok = mlx_reshape(&scores_arr, probs, sc_shape, 3, m->stream) == 0;   /* canonico: [T,K] intermedio ridondante */
        if (ok) ok = mlx_multiply(&scaled, down_out, scores_arr, m->stream) == 0;
        CHK(ok, "moe score scale");
    }

    if (ok && !moe_fused_done) { ok = mlx_sum_axis(&summed, scaled, /*axis=*/1, /*keepdims=*/false, m->stream) == 0; CHK(ok, "moe sum over top_k"); }
    /* summed: [n_tokens, 1, hidden] -- reshape a [1, n_tokens, hidden] (se il kernel fuso ha
     * gia' scritto moe_out, questo blocco e' saltato). */
    if (ok && !moe_fused_done) {
        const int out_shape[3] = {1, n_tokens, m->cfg.hidden_size};
        ok = mlx_reshape(&moe_out, summed, out_shape, 3, m->stream) == 0;
        CHK(ok, "moe_out reshape");
    }    }


    if (ok) { ok = mlx_add(out, h, moe_out, m->stream) == 0; CHK(ok, "final residual add"); }

    /* §E2 imatrix collector (calibrazione): accumula l'energia per (layer, expert routato, colonna)
     * sull'input FFN-normato + sulla riga SwiGLU, sia col path sorted sia unsorted (im_*). */
    if (m->im.active && ok && im_expert && im_src && im_rows > 0 && act.ctx != NULL)
        llm_mlx_imatrix_accumulate_rows(m, layer_idx, normed2, act, im_expert, im_src, im_rows);
    free(im_expert); free(im_src);

    mlx_array all[] = {
        ln2_w, normed2, gate_w, gate_s, gate_b, gates, gates_sm, part, top_inds,
        gate_proj_w, gate_proj_s, gate_proj_b, up_proj_w, up_proj_s, up_proj_b,
        down_proj_w, down_proj_s, down_proj_b, moe_out,
        no_lhs, top_inds_arr, x1, gate_out, up_out, sig, silu, act, down_out,
        scores_arr, scaled, summed, probs,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
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
