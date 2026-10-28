/* llm_mlx_qwen4_moe_packed.h — MoE qwen4 su pesi PACKED zero-copy (routed + shared).
 * v1: router/top-k e somma in CPU; GEMV per esperto col kernel Q4 del bridge.
 * 0 = gestito, 1 = non gestito (il chiamante usa il percorso MLX). */
#ifndef LLM_MLX_QWEN4_MOE_PACKED_H
#define LLM_MLX_QWEN4_MOE_PACKED_H

#include "llm_mlx_internal.h"
#include "llm_mlx_weights_backend.h"
#include "llm_mlx_wpacked.h"
#include "llm_mlx_linear_packed.h"
int llm_mlx_qp_name_get(const char* name, int* group, int* bits);
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <Accelerate/Accelerate.h>

static int q4mp_shard_of(const LlmMlxModel* m, const char* base, char* path, size_t cap) {
    char ip[1200]; snprintf(ip, sizeof ip, "%s/model.safetensors.index.json", m->model_dir);
    snprintf(path, cap, "%s/model.safetensors", m->model_dir);
    const size_t rootlen = strlen(m->model_dir);
    FILE* f = fopen(ip, "rb"); if (!f) return 0;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if (n<=0) { fclose(f); return 0; }
    char* j=(char*)malloc((size_t)n+1); if(!j){fclose(f); return 0;}
    if (fread(j,1,(size_t)n,f)!=(size_t)n){free(j);fclose(f);return 0;} fclose(f); j[n]=0;
    char key[700]; snprintf(key,sizeof key,"%s.weight",base);
    char* p = strstr(j, key);
    if (p) { char* q = strstr(p, ".safetensors"); if (q) { char* b=q; while (b>j && b[-1]!='"') --b; size_t len=(size_t)(q+12-b); if (rootlen+1+len<cap){memcpy(path,m->model_dir,rootlen);path[rootlen]='/';memcpy(path+rootlen+1,b,len);path[rootlen+1+len]=0;} } }
    free(j); return 0;
}

static void q4mp_eval(mlx_array a) { mlx_vector_array ev=mlx_vector_array_new(); mlx_vector_array_append_value(ev,a); mlx_eval(ev); mlx_vector_array_free(ev); }

/* ── Cache per-modello dei descrittori packed ─────────────────────────────────────────────────────────
 * Prima ogni gemv di esperto rileggeva l'indice JSON e gli header, rimappava i file e ricopiava scale/bias in un
 * nuovo contesto Metal (migliaia di volte per token). Ora per tensore si fa tutto UNA volta: path dello shard,
 * vista 3D con UN contesto Metal (esperto scelto a run-time), lineare 2D, e i pesi piccoli non-Q4 (router,
 * shared_expert_gate) dequantizzati in f32. Le entry hanno puntatori stabili; l'uso e' serializzato da q4mp_mu. */
typedef struct {
    char base[300]; char path[1600];
    int st3; LlmMlxPackedW3 w3; LlmGemvQ4Ctx* g3;      /* st: 0 = da aprire, 1 = pronto, -1 = fallito (cache negativa) */
    int stl; LlmMlxPackedLinear lin;
    int std; float* dq; int dq_rows, dq_cols;
} Q4mpEnt;
static Q4mpEnt** q4mp_tab; static int q4mp_n, q4mp_cap; static char q4mp_dir[1200];
static pthread_mutex_t q4mp_mu = PTHREAD_MUTEX_INITIALIZER;

static void q4mp_cache_reset(void) {
    for (int i = 0; i < q4mp_n; i++) {
        Q4mpEnt* e = q4mp_tab[i];
        if (e->st3 == 1) { if (e->g3) llm_gemvq4_destroy(e->g3); llm_mlx_packed_close3d(&e->w3); }
        if (e->stl == 1) llm_mlx_linear_packed_close(&e->lin);
        free(e->dq); free(e);
    }
    free(q4mp_tab); q4mp_tab = NULL; q4mp_n = q4mp_cap = 0;
}

static Q4mpEnt* q4mp_ent(const LlmMlxModel* m, const char* base) {
    if (strcmp(q4mp_dir, m->model_dir) != 0) { q4mp_cache_reset(); snprintf(q4mp_dir, sizeof q4mp_dir, "%s", m->model_dir); }
    for (int i = 0; i < q4mp_n; i++) if (strcmp(q4mp_tab[i]->base, base) == 0) return q4mp_tab[i];
    if (q4mp_n >= q4mp_cap) {
        int nc = q4mp_cap ? q4mp_cap * 2 : 512;
        Q4mpEnt** nt = (Q4mpEnt**)realloc(q4mp_tab, (size_t)nc * sizeof *nt);
        if (!nt) return NULL;
        q4mp_tab = nt; q4mp_cap = nc;
    }
    Q4mpEnt* e = (Q4mpEnt*)calloc(1, sizeof *e); if (!e) return NULL;
    snprintf(e->base, sizeof e->base, "%s", base);
    q4mp_shard_of(m, base, e->path, sizeof e->path);
    q4mp_tab[q4mp_n++] = e;
    return e;
}

/* Vista 3D degli esperti: apre una volta, un solo contesto Metal con offset per esperto. NULL se non disponibile. */
static Q4mpEnt* q4mp_ent3(const LlmMlxModel* m, const char* base, int group) {
    Q4mpEnt* e = q4mp_ent(m, base); if (!e || e->st3 < 0) return NULL;
    if (e->st3 == 1) return e;
    e->st3 = -1;
    { int qg=0,qb=0; if (llm_mlx_qp_name_get(base,&qg,&qb) && qb>0) {
        if (llm_mlx_packed_open3d(e->path, base, qb, qg>0?qg:0, &e->w3) != 0) return NULL; }
      else if (llm_mlx_packed_open3d(e->path, base, 4, 0, &e->w3) != 0) return NULL; }
    e->g3 = llm_gemvq4_create();
    if (!e->g3 || !e->w3.biases ||
        llm_gemvq4_load_weights_3d_nocopy(e->g3, (const uint8_t*)e->w3.w, e->w3.scales, e->w3.biases, e->w3.cols, e->w3.rows,
                                          (uint32_t)e->w3.group, e->w3.experts, e->w3.w_stride, e->w3.s_stride) != 0) {
        if (e->g3) { llm_gemvq4_destroy(e->g3); e->g3 = NULL; }
        llm_mlx_packed_close3d(&e->w3); return NULL;
    }
    llm_gemvq4_set_scale_bf16(e->g3, e->w3.scale_bf16);
    e->st3 = 1; return e;
}

static Q4mpEnt* q4mp_entl(const LlmMlxModel* m, const char* base, int group) {
    Q4mpEnt* e = q4mp_ent(m, base); if (!e || e->stl < 0) return NULL;
    if (e->stl == 1) return e;
    e->stl = -1;
    { int qg=0,qb=0; int r = 0;
      if (llm_mlx_qp_name_get(base,&qg,&qb) && qb>0) r = llm_mlx_linear_packed_open(e->path, base, qb, qg>0?qg:0, &e->lin);
      else r = llm_mlx_linear_packed_open(e->path, base, 4, 0, &e->lin);
      if (r != 0) return NULL; }
    if (e->lin.g) llm_gemvq4_set_scale_bf16(e->lin.g, e->lin.w.scale_bf16);
    e->stl = 1; return e;
}
/* Router e shared_expert_gate sono Q8/gs64 (override per-modulo nel config): il kernel packed e' solo Q4. Li dequantizzo UNA
 * volta in f32 dal map MLX (tensori piccoli, non rilasciati) e li applico con BLAS: nessun dispatch/eval MLX per token. */
static const float* q4mp_dense(LlmMlxModel* m, const char* base, int* rows, int* cols) {
    Q4mpEnt* e = q4mp_ent(m, base); if (!e || e->std < 0) return NULL;
    if (e->std == 1) { *rows = e->dq_rows; *cols = e->dq_cols; return e->dq; }
    e->std = -1;
    mlx_array w = mlx_array_new(), sc = mlx_array_new(), b = mlx_array_new(), d = mlx_array_new();
    int ok = 0;
    if (multishard_get_triple(&m->weights, base, &w, &sc, &b)) {
        int g = m->cfg.quant_group_size, bt = m->cfg.quant_bits; char mode[16];
        llm_mlx_quant_derive(w, sc, g, bt, "affine", &g, &bt, mode, sizeof mode);
        mlx_optional_dtype dt = { MLX_FLOAT32, true };
        if (mlx_dequantize(&d, w, sc, b, (mlx_optional_int){g, true}, (mlx_optional_int){bt, true}, mode, g_mlx_empty_arr, dt, m->stream) == 0) {
            q4mp_eval(d);
            const float* p = mlx_array_data_float32(d);
            if (p && mlx_array_ndim(d) == 2) {
                size_t n = mlx_array_size(d);
                e->dq = (float*)malloc(n * sizeof(float));
                if (e->dq) { memcpy(e->dq, p, n * sizeof(float)); e->dq_rows = mlx_array_dim(d, 0); e->dq_cols = mlx_array_dim(d, 1); ok = 1; }
            }
        }
    }
    mlx_array_free(w); mlx_array_free(sc); mlx_array_free(b); mlx_array_free(d);
    if (!ok) return NULL;
    e->std = 1; *rows = e->dq_rows; *cols = e->dq_cols; return e->dq;
}

/* Rilascia dal map gli array dei tensori MoE del layer: il forward packed non li usa.
 * Chiamata SOLO dopo un calcolo packed riuscito (mai prima), così il fallback resta sicuro. */
#undef Q4J
static void q4mp_release_layer(LlmMlxModel* m, int layer_idx) {
    static const char* suf[] = {
        ".mlp.switch_mlp.gate_proj", ".mlp.switch_mlp.up_proj", ".mlp.switch_mlp.down_proj",
        ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj", ".mlp.shared_expert.down_proj"
    };
    static const char* aux[] = {".weight",".scales",".biases"};
    for (size_t b=0;b<sizeof suf/sizeof suf[0];b++)
        for (int a=0;a<3;a++) {
            char nm[300]; snprintf(nm,sizeof nm,"%s.%d%s%s", m->cfg.layer_prefix, layer_idx, suf[b], aux[a]);
            mlx_array e = mlx_array_new();
            for (int si=0; si<m->weights.count; si++) (void)mlx_map_string_to_array_insert(m->weights.shards[si], nm, e);
            mlx_array_free(e);
        }
}


static float q4mp_silu(float v){ return v>=0? v/(1.f+expf(-v)) : v*expf(v)/(1.f+expf(v)); }

static int qwen4_exp_moe_packed_try_locked(mlx_array* out, LlmMlxModel* m, int layer_idx, mlx_array x);

/* 0 = calcolato in packed; 1 = non gestito. */
static int qwen4_exp_moe_packed_try(mlx_array* out, LlmMlxModel* m, int layer_idx, mlx_array x) {
    pthread_mutex_lock(&q4mp_mu);
    int rc = qwen4_exp_moe_packed_try_locked(out, m, layer_idx, x);
    pthread_mutex_unlock(&q4mp_mu);
    return rc;
}

static int qwen4_exp_moe_packed_try_locked(mlx_array* out, LlmMlxModel* m, int layer_idx, mlx_array x) {
    if (!out || !m) return 1;
    if (!llm_mlx_weights_backend_packed() || m->cfg.quant_bits != 4) return 1;
    if (m->cfg.model_type[0] && !strstr(m->cfg.model_type, "qwen4")) return 1;
    const int nd0 = mlx_array_ndim(x);
    if (nd0 != 2 && nd0 != 3) return 1;
    const int* xs = mlx_array_shape(x);
    int nrow0 = 1; for (int d = 0; d < nd0-1; d++) nrow0 *= xs[d];
    const char* pre = m->cfg.layer_prefix; const int g = m->cfg.quant_group_size;
    char pg[256], pu[256], pd[256], rg[256], rgt[256], sg[256], su[256], sd[256];
    #define Q4J(name,tail) snprintf(name,sizeof name,"%s%d.%s", pre, layer_idx, tail)
    Q4J(pg,"mlp.switch_mlp.gate_proj"); Q4J(pu,"mlp.switch_mlp.up_proj"); Q4J(pd,"mlp.switch_mlp.down_proj");
    Q4J(rg,"mlp.gate"); Q4J(rgt,"mlp.shared_expert_gate");
    Q4J(sg,"mlp.shared_expert.gate_proj"); Q4J(su,"mlp.shared_expert.up_proj"); Q4J(sd,"mlp.shared_expert.down_proj");
    #undef Q4J
    Q4mpEnt *Eg = q4mp_ent3(m, pg, g), *Eu = q4mp_ent3(m, pu, g), *Ed = q4mp_ent3(m, pd, g);
    Q4mpEnt *Sg = q4mp_entl(m, sg, g), *Su = q4mp_entl(m, su, g), *Sd = q4mp_entl(m, sd, g);
    int rrows = 0, rcols = 0, grows = 0, gcols = 0;
    const float* RW = q4mp_dense(m, rg, &rrows, &rcols);
    const float* GW = q4mp_dense(m, rgt, &grows, &gcols);
    if (!Eg || !Eu || !Ed || !Sg || !Su || !Sd || !RW || !GW) return 1;
    const int F = (int)Eg->w3.rows, E = (int)Eg->w3.cols, NE = (int)Eg->w3.experts, K = m->cfg.top_k;
    if (F <= 0 || E <= 0 || NE <= 0 || K <= 0 || K > 16 || rrows != NE || rcols != E || grows != 1 || gcols != E) return 1;
    if ((int)Eu->w3.rows != F || (int)Eu->w3.cols != E || (int)Ed->w3.rows != E || (int)Ed->w3.cols != F || (int)Eu->w3.experts != NE || (int)Ed->w3.experts != NE ||
        (int)Sg->lin.w.rows != F || (int)Sg->lin.w.cols != E || (int)Su->lin.w.rows != F || (int)Su->lin.w.cols != E ||
        (int)Sd->lin.w.rows != E || (int)Sd->lin.w.cols != F) return 1;

    mlx_array xflat = mlx_array_new(), x32 = mlx_array_new();
    { int sh2[2] = {nrow0, xs[nd0-1]};
      mlx_array t = mlx_array_new();
      int ok = (nd0 == 3 ? mlx_reshape(&t, x, sh2, 2, m->stream) : mlx_array_set(&t, x)) == 0 && mlx_astype(&xflat, t, MLX_FLOAT32, m->stream) == 0;
      mlx_array_free(t); (void)x32;
      if (!ok) { mlx_array_free(xflat); mlx_array_free(x32); return 1; } }
    mlx_array_free(x32);
    q4mp_eval(xflat);
    const float* xd = mlx_array_data_float32(xflat);
    float* gates = (float*)malloc((size_t)NE * 4 * (size_t)nrow0);
    float* sgate = (float*)malloc(sizeof(float) * (size_t)nrow0);
    uint16_t *xh16 = (uint16_t*)malloc((size_t)E*2), *oG = (uint16_t*)malloc((size_t)(K+1)*F*2), *oU = (uint16_t*)malloc((size_t)(K+1)*F*2);
    float *oA = (float*)malloc(sizeof(float)*(size_t)(K+1)*F), *oD = (float*)malloc(sizeof(float)*(size_t)(K+1)*E);
    double* acc = (double*)calloc((size_t)E, sizeof(double));
    float* obuf = (float*)malloc((size_t)E*4*(size_t)nrow0);
    char* used = (char*)malloc((size_t)NE);
    int rc = (xd && gates && sgate && xh16 && oG && oU && oA && oD && acc && obuf && used) ? 0 : 1;
    if (!rc) {   /* router e gate dello shared expert su tutte le righe: gates[nrow][NE] = X[nrow][E] . RW^T */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nrow0, NE, E, 1.0f, xd, E, RW, E, 0.0f, gates, NE);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nrow0, 1, E, 1.0f, xd, E, GW, E, 0.0f, sgate, 1);
    }
    /* DECODE (1 riga): esperto-major come ds4 — gate, up e down in UN dispatch ciascuno (K threadgroup
     * in parallelo), niente loop CPU per esperto e niente 30 dispatch seriali. */
    if (!rc && nrow0 == 1) {
        const float* gr = gates;
        double mx = -1e300; for (int i = 0; i < NE; i++) if (gr[i] > mx) mx = gr[i];
        double sum = 0; for (int i = 0; i < NE; i++) sum += exp((double)gr[i] - mx);
        double inv = sum > 0 ? 1.0/sum : 0.0;
        memset(used, 0, (size_t)NE);
        int sel[16]; double wts[16]; double wsum = 0;
        for (int k = 0; k < K; k++) {
            int bi = -1; double bp = -1;
            for (int e = 0; e < NE; e++) { if (used[e]) continue; double pr = exp((double)gr[e] - mx) * inv; if (pr > bp) { bp = pr; bi = e; } }
            if (bi < 0) { rc = 1; break; }
            used[bi] = 1; sel[k] = bi; wts[k] = bp; wsum += bp;
        }
        if (!rc && wsum > 0) for (int k = 0; k < K; k++) wts[k] /= wsum;
        int32_t sel32[16]; for (int k = 0; k < K; k++) sel32[k] = sel[k];
        for (int i = 0; i < E; i++) { _Float16 h = (_Float16)xd[i]; memcpy(&xh16[i], &h, 2); }
        /* FUSO come ds4: lo SHARED e' lo slot n_slots dello stesso kernel (mid e down), quindi
         * per layer restano DUE dispatch (prima erano 4: routed mid/down + shared mid/down). */
        float wtsf[17]; for (int k = 0; k < K; k++) wtsf[k] = (float)wts[k];
        wtsf[K] = 1.f/(1.f+expf(-sgate[0]));   /* gate dello shared expert (sigmoid) */
        sel32[K] = 0;
        if (!rc) { rc = llm_gemvq4_run_moe_mid(Eg->g3, Eu->g3, Sg->lin.g, Su->lin.g, xh16, oA, sel32, K);
                   if (rc) fprintf(stderr, "[moe-fused] mid rc=%d\n", rc); }
        if (!rc) { rc = llm_gemvq4_run_moe_down(Ed->g3, Sd->lin.g, oA, oD, sel32, K, (uint32_t)F);
                   if (rc) fprintf(stderr, "[moe-fused] down rc=%d\n", rc); }
        if (!rc) {
            memset(acc, 0, (size_t)E*sizeof(double));
            for (int k = 0; k <= K; k++)   /* combine con peso (shared = slot K), come ds4 */
                for (int d = 0; d < E; d++) acc[d] += (double)wtsf[k] * (double)oD[(size_t)k*E + d];
            for (int d = 0; d < E; d++) obuf[d] = (float)acc[d];
        }
    }
    for (int r = 0; r < nrow0 && !rc && nrow0 > 1; r++) {
        const float* gr = gates + (size_t)r*NE;
        const float* xr = xd + (size_t)r*E;
        double mx = -1e300; for (int i = 0; i < NE; i++) if (gr[i] > mx) mx = gr[i];
        double sum = 0; for (int i = 0; i < NE; i++) sum += exp((double)gr[i] - mx);
        double inv = sum > 0 ? 1.0/sum : 0.0;
        memset(used, 0, (size_t)NE);
        int sel[16]; double wts[16]; double wsum = 0;
        for (int k = 0; k < K; k++) {
            int bi = -1; double bp = -1;
            for (int e = 0; e < NE; e++) { if (used[e]) continue; double pr = exp((double)gr[e] - mx) * inv; if (pr > bp) { bp = pr; bi = e; } }
            if (bi < 0) { fprintf(stderr, "[moe-packed] riga %d: router senza esperto selezionabile (NaN nei gates?)\n", r); rc = 1; break; }
            used[bi] = 1; sel[k] = bi; wts[k] = bp; wsum += bp;
        }
        if (rc) break;
        if (wsum > 0) for (int k = 0; k < K; k++) wts[k] /= wsum;
        /* prefill: stessi kernel fusi del decode, una riga per volta (2 dispatch per riga). */
        int32_t sel32[16]; for (int k = 0; k < K; k++) sel32[k] = sel[k];
        float wtsf[17]; for (int k = 0; k < K; k++) wtsf[k] = (float)wts[k];
        wtsf[K] = 1.f/(1.f+expf(-sgate[r])); sel32[K] = 0;
        for (int i = 0; i < E; i++) { _Float16 h = (_Float16)xr[i]; memcpy(&xh16[i], &h, 2); }
        if (!rc) { rc = llm_gemvq4_run_moe(Eg->g3, Eu->g3, Sg->lin.g, Su->lin.g, Ed->g3, Sd->lin.g,
                                           xh16, oA, oD, sel32, K, (uint32_t)F);
                   if (rc) fprintf(stderr,"[moe-row] r=%d rc=%d\n", r, rc); }
        memset(acc, 0, (size_t)E*sizeof(double));
        for (int k = 0; k <= K; k++)   /* combine con peso (shared = slot K) */
            for (int d = 0; d < E; d++) acc[d] += (double)wtsf[k] * (double)oD[(size_t)k*E + d];
        for (int d = 0; d < E; d++) obuf[(size_t)r*E + d] = (float)acc[d];
    }
    if (!rc) {
        int sh[2] = {nrow0, E};
        mlx_array flatout = mlx_array_new_data(obuf, sh, 2, MLX_FLOAT32);
        if (nd0 == 3) { int osh[3] = {xs[0], xs[1], E}; rc = mlx_reshape(out, flatout, osh, 3, m->stream); mlx_array_free(flatout); }
        else { *out = flatout; }
    }
    free(obuf); free(acc); free(xh16); free(oG); free(oU); free(oA); free(oD); free(gates); free(sgate); free(used);
    mlx_array_free(xflat);
    return rc;
}

#endif
