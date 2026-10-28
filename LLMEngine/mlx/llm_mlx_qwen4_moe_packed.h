/* llm_mlx_qwen4_moe_packed.h — MoE qwen4 su pesi PACKED zero-copy, come NODI LAZY del grafo MLX.
 *
 * Come ds4: router -> softmax/top-K -> mid (gate+up+silu) -> down -> combine pesata, con lo shared expert come slot K dello
 * stesso kernel. Cinque mlx_fast_metal_kernel concatenati nel grafo: nessun eval, nessun command buffer dedicato, nessuna
 * attesa fra i layer. I pesi degli esperti sono viste mmap zero-copy (llm_mlx_packed_lazy.h); router e gate dello shared
 * expert (Q8/gs64 nel checkpoint, non packabili col kernel Q4) sono dequantizzati una volta in f32.
 * 0 = gestito (out lazy), 1 = non gestito (il chiamante usa il percorso MLX). */
#ifndef LLM_MLX_QWEN4_MOE_PACKED_H
#define LLM_MLX_QWEN4_MOE_PACKED_H

#include "llm_mlx_internal.h"
#include "llm_mlx_weights_backend.h"
#include "llm_mlx_wpacked.h"
#include "llm_mlx_packed_lazy.h"
#include "llm_mlx_q4_msl.h"
int llm_mlx_qp_name_get(const char* name, int* group, int* bits);
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

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

/* ── Stato per layer ─────────────────────────────────────────────────────────────────────────────────── */
#define Q4MP_NT 6   /* tensori: 0 gate 1 up 2 down (3D esperti); 3 sg 4 su 5 sd (2D shared) */
typedef struct {
    char layer_key[300];                   /* "<prefix><layer>." */
    int ready, failed;
    int F, E, NE, K;
    LlmMlxPackedW3 w3[3];                  /* esperti */
    LlmMlxPackedW  w2[3];                  /* shared */
    mlx_array v[Q4MP_NT][3];               /* viste weight/scales/biases */
    mlx_array P;                           /* parametri int32 */
    mlx_array RW;                          /* [NE+1][E] f32: router + gate shared */
} Q4mpLayer;
static Q4mpLayer** q4mp_tab; static int q4mp_n, q4mp_cap; static char q4mp_dir[1200];
static pthread_mutex_t q4mp_mu = PTHREAD_MUTEX_INITIALIZER;
static mlx_fast_metal_kernel q4mp_k[5]; static int q4mp_kready;

static void q4mp_cache_reset(void) {
    for (int i = 0; i < q4mp_n; i++) {
        Q4mpLayer* l = q4mp_tab[i];
        if (l->ready) {
            for (int t = 0; t < Q4MP_NT; t++) for (int c = 0; c < 3; c++) mlx_array_free(l->v[t][c]);
            mlx_array_free(l->P); mlx_array_free(l->RW);
            for (int t = 0; t < 3; t++) { llm_mlx_packed_close3d(&l->w3[t]); llm_mlx_packed_close(&l->w2[t]); }
        }
        free(l);
    }
    free(q4mp_tab); q4mp_tab = NULL; q4mp_n = q4mp_cap = 0;
}

/* Router e gate shared: dequantizzati in f32 dal map MLX (tensori piccoli, non rilasciati). */
static int q4mp_dense_f32(LlmMlxModel* m, const char* base, float** out, int* rows, int* cols) {
    mlx_array w = mlx_array_new(), sc = mlx_array_new(), b = mlx_array_new(), d = mlx_array_new();
    int ok = 0;
    char skey[330]; snprintf(skey, sizeof skey, "%s.scales", base);
    mlx_array probe = mlx_array_new();
    const int quantized = multishard_get(&m->weights, skey, &probe);
    mlx_array_free(probe);
    if (!quantized) {   /* router/gate non quantizzati (checkpoint 2 bit): peso bf16/f32 piano */
        char wkey[330]; snprintf(wkey, sizeof wkey, "%s.weight", base);
        if (multishard_get(&m->weights, wkey, &w) && mlx_astype(&d, w, MLX_FLOAT32, m->stream) == 0) {
            q4mp_eval(d);
            const float* p = mlx_array_data_float32(d);
            if (p && mlx_array_ndim(d) == 2) {
                size_t n = mlx_array_size(d);
                *out = (float*)malloc(n * sizeof(float));
                if (*out) { memcpy(*out, p, n * sizeof(float)); *rows = mlx_array_dim(d, 0); *cols = mlx_array_dim(d, 1); ok = 1; }
            }
        }
    } else if (multishard_get_triple(&m->weights, base, &w, &sc, &b)) {
        int g = m->cfg.quant_group_size, bt = m->cfg.quant_bits; char mode[16];
        llm_mlx_quant_derive(w, sc, g, bt, "affine", &g, &bt, mode, sizeof mode);
        mlx_optional_dtype dt = { MLX_FLOAT32, true };
        if (mlx_dequantize(&d, w, sc, b, (mlx_optional_int){g, true}, (mlx_optional_int){bt, true}, mode, g_mlx_empty_arr, dt, m->stream) == 0) {
            q4mp_eval(d);
            const float* p = mlx_array_data_float32(d);
            if (p && mlx_array_ndim(d) == 2) {
                size_t n = mlx_array_size(d);
                *out = (float*)malloc(n * sizeof(float));
                if (*out) { memcpy(*out, p, n * sizeof(float)); *rows = mlx_array_dim(d, 0); *cols = mlx_array_dim(d, 1); ok = 1; }
            }
        }
    }
    mlx_array_free(w); mlx_array_free(sc); mlx_array_free(b); mlx_array_free(d);
    return ok;
}

static int q4mp_bits_of(const LlmMlxModel* m, const char* base) {
    int qg = 0, qb = 0;
    if (llm_mlx_qp_name_get(base, &qg, &qb) && qb > 0) return qb;
    return m->cfg.quant_bits;
}

static void q4mp_pset(int32_t* P, int t, int group, int bf, int lw, int ls, int lb, size_t wst, size_t sst, int bits) {
    P[t*8+7] = bits;
    P[t*8+0] = group; P[t*8+1] = bf; P[t*8+2] = lw; P[t*8+3] = ls; P[t*8+4] = lb; P[t*8+5] = (int32_t)wst; P[t*8+6] = (int32_t)sst;
}

/* Costruisce (una volta) lo stato lazy del layer. 0 = pronto. */
static int q4mp_layer_build(LlmMlxModel* m, Q4mpLayer* l, int layer_idx) {
    const char* pre = m->cfg.layer_prefix; const int g = m->cfg.quant_group_size;
    static const char* const leaf3[3] = { "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj" };
    static const char* const leaf2[3] = { "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj" };
    char base[300], path[1600];
    int32_t P[96]; memset(P, 0, sizeof P);
    for (int t = 0; t < 3; t++) {
        snprintf(base, sizeof base, "%s%d.%s", pre, layer_idx, leaf3[t]); q4mp_shard_of(m, base, path, sizeof path);
        if (llm_mlx_packed_open3d(path, base, q4mp_bits_of(m, base), g, &l->w3[t]) != 0 || !l->w3[t].biases) return -1;
        if (l->w3[t].bits != 4 && l->w3[t].bits != 2) return -1;
        const LlmMlxPackedW3* w = &l->w3[t];
        size_t wb = (size_t)w->experts * w->w_stride, sb = (size_t)w->experts * w->s_stride;
        int lw, ls, lb;
        l->v[t][0] = llm_mlx_lazy_view(w->w, wb, &lw); l->v[t][1] = llm_mlx_lazy_view(w->scales, sb, &ls); l->v[t][2] = llm_mlx_lazy_view(w->biases, sb, &lb);
        q4mp_pset(P, t, w->group, w->scale_bf16, lw, ls, lb, w->w_stride, w->s_stride, w->bits);
    }
    for (int t = 0; t < 3; t++) {
        snprintf(base, sizeof base, "%s%d.%s", pre, layer_idx, leaf2[t]); q4mp_shard_of(m, base, path, sizeof path);
        if (llm_mlx_packed_open(path, base, q4mp_bits_of(m, base), g, &l->w2[t]) != 0 || !l->w2[t].biases) return -1;
        if (l->w2[t].bits != 4 && l->w2[t].bits != 2) return -1;
        const LlmMlxPackedW* w = &l->w2[t];
        size_t wb = (size_t)w->rows * (w->cols * (uint32_t)w->bits / 8), sb = (size_t)w->rows * (w->cols / (uint32_t)w->group) * 2;
        int lw, ls, lb;
        l->v[3+t][0] = llm_mlx_lazy_view(w->w, wb, &lw); l->v[3+t][1] = llm_mlx_lazy_view(w->scales, sb, &ls); l->v[3+t][2] = llm_mlx_lazy_view(w->biases, sb, &lb);
        q4mp_pset(P, 3+t, w->group, w->scale_bf16, lw, ls, lb, 0, 0, w->bits);
    }
    l->F = (int)l->w3[0].rows; l->E = (int)l->w3[0].cols; l->NE = (int)l->w3[0].experts; l->K = m->cfg.top_k;
    if (l->F <= 0 || l->E <= 0 || l->NE <= 0 || l->NE > 512 || l->K <= 0 || l->K > 15) return -1;
    if ((int)l->w3[1].rows != l->F || (int)l->w3[1].cols != l->E || (int)l->w3[2].rows != l->E || (int)l->w3[2].cols != l->F ||
        (int)l->w3[1].experts != l->NE || (int)l->w3[2].experts != l->NE ||
        (int)l->w2[0].rows != l->F || (int)l->w2[0].cols != l->E || (int)l->w2[1].rows != l->F || (int)l->w2[1].cols != l->E ||
        (int)l->w2[2].rows != l->E || (int)l->w2[2].cols != l->F) return -1;
    P[64] = l->E; P[65] = l->F; P[66] = l->NE; P[67] = l->K;
    int psh[1] = { 96 };
    l->P = mlx_array_new_data(P, psh, 1, MLX_INT32);
    /* router (NE righe) + gate dello shared expert (1 riga) */
    char rb[300], gb[300]; float *rw = NULL, *gw = NULL; int rr = 0, rc = 0, gr = 0, gc = 0;
    snprintf(rb, sizeof rb, "%s%d.mlp.gate", pre, layer_idx); snprintf(gb, sizeof gb, "%s%d.mlp.shared_expert_gate", pre, layer_idx);
    if (!q4mp_dense_f32(m, rb, &rw, &rr, &rc) || !q4mp_dense_f32(m, gb, &gw, &gr, &gc)) { free(rw); free(gw); return -1; }
    if (rr != l->NE || rc != l->E || gr != 1 || gc != l->E) { free(rw); free(gw); return -1; }
    float* W = (float*)malloc((size_t)(l->NE + 1) * (size_t)l->E * sizeof(float));
    if (!W) { free(rw); free(gw); return -1; }
    memcpy(W, rw, (size_t)l->NE * l->E * sizeof(float)); memcpy(W + (size_t)l->NE * l->E, gw, (size_t)l->E * sizeof(float));
    int rsh[2] = { l->NE + 1, l->E };
    l->RW = mlx_array_new_data(W, rsh, 2, MLX_FLOAT32);   /* copia in un buffer MLX */
    free(W); free(rw); free(gw);
    return 0;
}

static Q4mpLayer* q4mp_layer(LlmMlxModel* m, int layer_idx) {
    if (strcmp(q4mp_dir, m->model_dir) != 0) { q4mp_cache_reset(); snprintf(q4mp_dir, sizeof q4mp_dir, "%s", m->model_dir); }
    char key[300]; snprintf(key, sizeof key, "%s%d.", m->cfg.layer_prefix, layer_idx);
    Q4mpLayer* l = NULL;
    for (int i = 0; i < q4mp_n; i++) if (strcmp(q4mp_tab[i]->layer_key, key) == 0) { l = q4mp_tab[i]; break; }
    if (!l) {
        if (q4mp_n >= q4mp_cap) {
            int nc = q4mp_cap ? q4mp_cap * 2 : 64;
            Q4mpLayer** nt = (Q4mpLayer**)realloc(q4mp_tab, (size_t)nc * sizeof *nt);
            if (!nt) return NULL;
            q4mp_tab = nt; q4mp_cap = nc;
        }
        l = (Q4mpLayer*)calloc(1, sizeof *l); if (!l) return NULL;
        snprintf(l->layer_key, sizeof l->layer_key, "%s", key);
        q4mp_tab[q4mp_n++] = l;
    }
    if (l->failed) return NULL;
    if (!l->ready) {
        if (q4mp_layer_build(m, l, layer_idx) != 0) { l->failed = 1; return NULL; }
        l->ready = 1;
    }
    return l;
}

/* ── Kernel ───────────────────────────────────────────────────────────────────────────────────────────── */
static const char q4mp_src_gates[] =
"uint tg = threadgroup_position_in_grid.x; uint t = threadgroup_position_in_grid.y;\n"
"uint sg = simdgroup_index_in_threadgroup; uint lane = thread_index_in_simdgroup;\n"
"uint E = P[64], NE = P[66];\n"
"uint row = tg * 8 + sg;\n"
"if (row > NE) return;\n"
"const device float* w = RW + (ulong)row * E; const device float* x = X + (ulong)t * E;\n"
"float acc = 0.0f;\n"
"for (uint i = lane; i < E; i += 32) acc += w[i] * x[i];\n"
"float s = simd_sum(acc);\n"
"if (lane == 0) GATES[(ulong)t * (NE + 1) + row] = s;\n";

static const char q4mp_src_topk[] =
"threadgroup float sv[512]; threadgroup int si[512]; threadgroup int chosen[16]; threadgroup float chosenv[16];\n"
"uint tid = thread_position_in_threadgroup.x; uint t = threadgroup_position_in_grid.y;\n"
"uint NE = P[66], K = P[67];\n"
"const device float* gates = GATES + (ulong)t * (NE + 1);\n"
"float mx = -INFINITY;\n"
"for (uint i = tid; i < NE; i += 512) mx = max(mx, gates[i]);\n"
"sv[tid] = mx; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"for (uint s = 256; s > 0; s >>= 1) { if (tid < s) sv[tid] = max(sv[tid], sv[tid + s]); threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
"mx = sv[0]; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"for (uint k = 0; k < K; k++) {\n"
"    float best = -INFINITY; int bi = 0x7fffffff;\n"
"    for (uint i = tid; i < NE; i += 512) {\n"
"        bool used = false; for (uint j = 0; j < k; j++) used = used || (chosen[j] == (int)i);\n"
"        if (used) continue;\n"
"        float v = gates[i]; if (v > best) { best = v; bi = (int)i; }\n"
"    }\n"
"    sv[tid] = best; si[tid] = bi; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint s = 256; s > 0; s >>= 1) {\n"
"        if (tid < s) { float ov = sv[tid + s]; int oi = si[tid + s]; if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) { sv[tid] = ov; si[tid] = oi; } }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0) { chosen[k] = si[0]; chosenv[k] = sv[0]; }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"}\n"
"if (tid == 0) {\n"
"    float ws = 0.0f; for (uint k = 0; k < K; k++) ws += exp(chosenv[k] - mx);\n"
"    for (uint k = 0; k < K; k++) { SEL[t * 16 + k] = chosen[k]; WTS[t * 16 + k] = exp(chosenv[k] - mx) / ws; }\n"
"    SEL[t * 16 + K] = 0; WTS[t * 16 + K] = 1.0f / (1.0f + exp(-gates[NE]));\n"
"}\n";

static const char q4mp_src_mid[] =
"uint tgx = threadgroup_position_in_grid.x, slot = threadgroup_position_in_grid.y, t = threadgroup_position_in_grid.z;\n"
"uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;\n"
"uint E = P[64], F = P[65], K = P[67];\n"
"bool sh = slot >= K; int e = sh ? 0 : SEL[t * 16 + slot];\n"
"const device float* x = X + (ulong)t * E;\n"
"uint base = (tgx * 16 + sg) * 2;\n"
"if (base >= F) return;\n"
"uint nr = min(2u, F - base); uint r1 = min(base + 1u, F - 1u);\n"
"uint gi = sh ? 3u : 0u, ui = sh ? 4u : 1u;\n"
"ulong gow = (ulong)P[gi*8+2] + (ulong)e * (uint)P[gi*8+5], gos = (ulong)P[gi*8+3] + (ulong)e * (uint)P[gi*8+6], gob = (ulong)P[gi*8+4] + (ulong)e * (uint)P[gi*8+6];\n"
"ulong uow = (ulong)P[ui*8+2] + (ulong)e * (uint)P[ui*8+5], uos = (ulong)P[ui*8+3] + (ulong)e * (uint)P[ui*8+6], uob = (ulong)P[ui*8+4] + (ulong)e * (uint)P[ui*8+6];\n"
"uint gg = P[gi*8+0], gbf = P[gi*8+1], gbt = P[gi*8+7], ug = P[ui*8+0], ubf = P[ui*8+1], ubt = P[ui*8+7];\n"
"const device uchar* gW = (sh ? WSG : WG) + gow; const device uchar* gS = (sh ? SSG : SG) + gos; const device uchar* gB = (sh ? BSG : BG) + gob;\n"
"const device uchar* uW = (sh ? WSU : WU) + uow; const device uchar* uS = (sh ? SSU : SU) + uos; const device uchar* uB = (sh ? BSU : BU) + uob;\n"
"ulong grw = (ulong)(E * gbt / 8), urw = (ulong)(E * ubt / 8), gsr = (ulong)(E / gg) * 2, usr = (ulong)(E / ug) * 2;\n"
"float og[4], ou[4];\n"
"q4_dotn(og, nr, gW + base * grw, gW + r1 * grw, gW, gW, gS + base * gsr, gS + r1 * gsr, gS, gS, gB + base * gsr, gB + r1 * gsr, gB, gB, (uint)(gow & 3), x, E, gg, gbf, gbt, lane);\n"
"q4_dotn(ou, nr, uW + base * urw, uW + r1 * urw, uW, uW, uS + base * usr, uS + r1 * usr, uS, uS, uB + base * usr, uB + r1 * usr, uB, uB, (uint)(uow & 3), x, E, ug, ubf, ubt, lane);\n"
"if (lane == 0) for (uint i = 0; i < nr; i++) ACT[((ulong)t * (K + 1) + slot) * F + base + i] = og[i] / (1.0f + exp(-og[i])) * ou[i];\n";

static const char q4mp_src_down[] =
"uint tgx = threadgroup_position_in_grid.x, slot = threadgroup_position_in_grid.y, t = threadgroup_position_in_grid.z;\n"
"uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;\n"
"uint E = P[64], F = P[65], K = P[67];\n"
"bool sh = slot >= K; int e = sh ? 0 : SEL[t * 16 + slot];\n"
"const device float* a = ACT + ((ulong)t * (K + 1) + slot) * F;\n"
"uint base = (tgx * 16 + sg) * 4;\n"
"if (base >= E) return;\n"
"uint nr = min(4u, E - base); uint r1 = min(base + 1u, E - 1u), r2 = min(base + 2u, E - 1u), r3 = min(base + 3u, E - 1u);\n"
"uint di = sh ? 5u : 2u;\n"
"ulong dow = (ulong)P[di*8+2] + (ulong)e * (uint)P[di*8+5], dos = (ulong)P[di*8+3] + (ulong)e * (uint)P[di*8+6], dob = (ulong)P[di*8+4] + (ulong)e * (uint)P[di*8+6];\n"
"uint dg = P[di*8+0], dbf = P[di*8+1], dbt = P[di*8+7];\n"
"const device uchar* dW = (sh ? WSD : WD) + dow; const device uchar* dS = (sh ? SSD : SD) + dos; const device uchar* dB = (sh ? BSD : BD) + dob;\n"
"ulong rw = (ulong)(F * dbt / 8), sr = (ulong)(F / dg) * 2;\n"
"float od[4];\n"
"q4_dotn(od, nr, dW + base * rw, dW + r1 * rw, dW + r2 * rw, dW + r3 * rw, dS + base * sr, dS + r1 * sr, dS + r2 * sr, dS + r3 * sr,\n"
"        dB + base * sr, dB + r1 * sr, dB + r2 * sr, dB + r3 * sr, (uint)(dow & 3), a, F, dg, dbf, dbt, lane);\n"
"if (lane == 0) for (uint i = 0; i < nr; i++) DOWN[((ulong)t * (K + 1) + slot) * E + base + i] = od[i];\n";

static const char q4mp_src_comb[] =
"uint d = thread_position_in_grid.x; uint t = thread_position_in_grid.y;\n"
"uint E = P[64], K = P[67];\n"
"if (d >= E) return;\n"
"float a = 0.0f;\n"
"for (uint k = 0; k <= K; k++) a += WTS[t * 16 + k] * DOWN[((ulong)t * (K + 1) + k) * E + d];\n"
"OUT[(ulong)t * E + d] = a;\n";

static void q4mp_kernels_init(void) {
    if (q4mp_kready) return;
    static const char* const i0[] = { "X", "RW", "P" };                       static const char* const o0[] = { "GATES" };
    static const char* const i1[] = { "GATES", "P" };                         static const char* const o1[] = { "SEL", "WTS" };
    static const char* const i2[] = { "X", "WG","SG","BG", "WU","SU","BU", "WSG","SSG","BSG", "WSU","SSU","BSU", "SEL", "P" }; static const char* const o2[] = { "ACT" };
    static const char* const i3[] = { "ACT", "WD","SD","BD", "WSD","SSD","BSD", "SEL", "P" };                                  static const char* const o3[] = { "DOWN" };
    static const char* const i4[] = { "DOWN", "WTS", "P" };                  static const char* const o4[] = { "OUT" };
    const char* const* ins[5] = { i0, i1, i2, i3, i4 }; const int nin[5] = { 3, 2, 15, 9, 3 };
    const char* const* outs[5] = { o0, o1, o2, o3, o4 }; const int nout[5] = { 1, 2, 1, 1, 1 };
    const char* srcs[5] = { q4mp_src_gates, q4mp_src_topk, q4mp_src_mid, q4mp_src_down, q4mp_src_comb };
    const char* names[5] = { "llm_moe_gates", "llm_moe_topk", "llm_moe_mid", "llm_moe_down", "llm_moe_comb" };
    for (int k = 0; k < 5; k++) {
        mlx_vector_string vi = mlx_vector_string_new(), vo = mlx_vector_string_new();
        for (int i = 0; i < nin[k]; i++) mlx_vector_string_append_value(vi, ins[k][i]);
        for (int i = 0; i < nout[k]; i++) mlx_vector_string_append_value(vo, outs[k][i]);
        q4mp_k[k] = mlx_fast_metal_kernel_new(names[k], vi, vo, srcs[k], llm_q4_msl_helpers, true, false);
        mlx_vector_string_free(vi); mlx_vector_string_free(vo);
    }
    q4mp_kready = 1;
}

/* Applica un kernel: ins (array di mlx_array), n; out_shapes; restituisce gli output in outs[]. 0 = ok. */
static int q4mp_apply(int k, const mlx_array* in, int nin, const int (*shape)[3], const int* ndim, const mlx_dtype* dt, int nout,
                      int g0, int g1, int g2, int t0, mlx_array* outs, mlx_stream s) {
    mlx_vector_array vi = mlx_vector_array_new();
    for (int i = 0; i < nin; i++) mlx_vector_array_append_value(vi, in[i]);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    for (int o = 0; o < nout; o++) mlx_fast_metal_kernel_config_add_output_arg(cfg, shape[o], (size_t)ndim[o], dt[o]);
    mlx_fast_metal_kernel_config_set_grid(cfg, g0, g1, g2);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, t0, 1, 1);
    mlx_vector_array vo = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&vo, q4mp_k[k], vi, cfg, s);
    if (rc == 0) for (int o = 0; o < nout; o++) mlx_vector_array_get(&outs[o], vo, (size_t)o);
    mlx_vector_array_free(vo); mlx_fast_metal_kernel_config_free(cfg); mlx_vector_array_free(vi);
    return rc;
}

static int qwen4_exp_moe_packed_try_locked(mlx_array* out, LlmMlxModel* m, int layer_idx, mlx_array x) {
    if (!out || !m) return 1;
    if (!llm_mlx_weights_backend_packed() || (m->cfg.quant_bits != 4 && m->cfg.quant_bits != 2)) return 1;
    if (m->cfg.model_type[0] && !strstr(m->cfg.model_type, "qwen4")) return 1;
    const int nd0 = mlx_array_ndim(x);
    if (nd0 != 2 && nd0 != 3) return 1;
    Q4mpLayer* l = q4mp_layer(m, layer_idx);
    if (!l) return 1;
    q4mp_kernels_init();
    const int E = l->E, F = l->F, NE = l->NE, K = l->K;
    if ((int)mlx_array_dim(x, nd0 - 1) != E) return 1;
    int nt = 1; for (int d = 0; d < nd0 - 1; d++) nt *= (int)mlx_array_dim(x, d);
    mlx_stream s = m->stream;

    mlx_array x32 = mlx_array_new(), X = mlx_array_new();
    int sh2[2] = { nt, E };
    if (mlx_astype(&x32, x, MLX_FLOAT32, s) != 0 || mlx_reshape(&X, x32, sh2, 2, s) != 0) { mlx_array_free(x32); mlx_array_free(X); return 1; }
    mlx_array_free(x32);

    mlx_array gates = mlx_array_new(), sel = mlx_array_new(), wts = mlx_array_new(), act = mlx_array_new(), down = mlx_array_new(), comb = mlx_array_new(), tmp[2];
    tmp[0] = mlx_array_new(); tmp[1] = mlx_array_new();
    int rc = 1;
    int shp[2][3]; int ndm[2]; mlx_dtype dts[2];
    /* 1) gates [nt][NE+1] */
    { mlx_array in[3] = { X, l->RW, l->P };
      shp[0][0] = nt; shp[0][1] = NE + 1; ndm[0] = 2; dts[0] = MLX_FLOAT32;
      if (q4mp_apply(0, in, 3, shp, ndm, dts, 1, ((NE + 1 + 7) / 8) * 256, nt, 1, 256, &gates, s) != 0) goto done; }
    /* 2) softmax + top-K -> sel/wts [nt][16] */
    { mlx_array in[2] = { gates, l->P };
      shp[0][0] = nt; shp[0][1] = 16; ndm[0] = 2; dts[0] = MLX_INT32;
      shp[1][0] = nt; shp[1][1] = 16; ndm[1] = 2; dts[1] = MLX_FLOAT32;
      if (q4mp_apply(1, in, 2, shp, ndm, dts, 2, 512, nt, 1, 512, tmp, s) != 0) goto done;
      mlx_array_set(&sel, tmp[0]); mlx_array_set(&wts, tmp[1]); }
    /* 3) mid -> act [nt][K+1][F] */
    { mlx_array in[15] = { X, l->v[0][0], l->v[0][1], l->v[0][2], l->v[1][0], l->v[1][1], l->v[1][2],
                           l->v[3][0], l->v[3][1], l->v[3][2], l->v[4][0], l->v[4][1], l->v[4][2], sel, l->P };
      int shm[1][3] = { { nt, K + 1, F } }; int nm[1] = { 3 }; mlx_dtype dm[1] = { MLX_FLOAT32 };
      if (q4mp_apply(2, in, 15, shm, nm, dm, 1, ((F + 31) / 32) * 512, K + 1, nt, 512, &act, s) != 0) goto done; }
    /* 4) down -> [nt][K+1][E] */
    { mlx_array in[9] = { act, l->v[2][0], l->v[2][1], l->v[2][2], l->v[5][0], l->v[5][1], l->v[5][2], sel, l->P };
      int shm[1][3] = { { nt, K + 1, E } }; int nm[1] = { 3 }; mlx_dtype dm[1] = { MLX_FLOAT32 };
      if (q4mp_apply(3, in, 9, shm, nm, dm, 1, ((E + 63) / 64) * 512, K + 1, nt, 512, &down, s) != 0) goto done; }
    /* 5) combine pesata -> [nt][E] */
    { mlx_array in[3] = { down, wts, l->P };
      shp[0][0] = nt; shp[0][1] = E; ndm[0] = 2; dts[0] = MLX_FLOAT32;
      if (q4mp_apply(4, in, 3, shp, ndm, dts, 1, ((E + 255) / 256) * 256, nt, 1, 256, &comb, s) != 0) goto done; }
    if (nd0 == 3) { int osh[3] = { (int)mlx_array_dim(x, 0), (int)mlx_array_dim(x, 1), E }; rc = mlx_reshape(out, comb, osh, 3, s) == 0 ? 0 : 1; }
    else rc = mlx_array_set(out, comb) == 0 ? 0 : 1;
done:
    mlx_array_free(X); mlx_array_free(gates); mlx_array_free(sel); mlx_array_free(wts); mlx_array_free(act); mlx_array_free(down); mlx_array_free(comb);
    mlx_array_free(tmp[0]); mlx_array_free(tmp[1]);
    return rc;
}

/* 0 = calcolato in packed (lazy); 1 = non gestito. */
static int qwen4_exp_moe_packed_try(mlx_array* out, LlmMlxModel* m, int layer_idx, mlx_array x) {
    pthread_mutex_lock(&q4mp_mu);
    int rc = qwen4_exp_moe_packed_try_locked(out, m, layer_idx, x);
    pthread_mutex_unlock(&q4mp_mu);
    return rc;
}

#endif
