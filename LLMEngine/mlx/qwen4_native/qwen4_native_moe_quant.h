/* qwen4_native_moe_quant.h — MoE con esperti QUANTIZZATI (Q8_0/F16/...): router f32
 * via kernel, poi dequant dello slab del singolo esperto (solo i 10 selezionati).
 * Layout GGUF 3D: dims [in, out, E] -> slab esperto contiguo (out*in). */
#ifndef LLM_QWEN4_NATIVE_MOE_QUANT_H
#define LLM_QWEN4_NATIVE_MOE_QUANT_H

#include "mlx/c/mlx.h"
#include "llm_ggml.h"
#include "qwen4_native_gguf.h"
#include "qwen4_native_router_kernel.h"
#include <time.h>
extern double g_q4n_prof[12];
static double q4nm_now(void){struct timespec t;timespec_get(&t,TIME_UTC);return (double)t.tv_sec+1e-9*(double)t.tv_nsec;}
#include "qwen4_native_layer.h"
#include "qwen4_native_gpu.h"

static int q4n_slab_f32(const LlmEngineMoeGguf* g, const char* name, int e, float* out) {
    int i = llm_gguf_find_tensor(g, name);
    if (i < 0) return -1;
    const int type = g->ti[i].type;
    const uint64_t in = g->ti[i].dims[0], outd = g->ti[i].dims[1];
    const int rb = (int)llm_ggml_nbytes(type, (int64_t)in);
    if (rb <= 0) return -1;
    const uint8_t* base = llm_gguf_tensor_data(g, i, NULL);
    if (!base) return -1;
    base += (uint64_t)e * (uint64_t)rb * outd;
    for (uint64_t r = 0; r < outd; r++)
        if (llmcodec_ggml_dequant_row(type, base + r * (uint64_t)rb, out + r * in, (int64_t)in) != 0) return -1;
    return 0;
}

static int qwen4_native_moe_quant_cpu(mlx_array* out, mlx_array x, const LlmEngineMoeGguf* g, int il,
                                      const Q4nLayerRefs* r, int E, int NE, int K, int F, mlx_stream s) {
    if (!out || !r || !g || mlx_array_ndim(x) != 2 || mlx_array_shape(x)[0] != 1) return -1;
    const float* xh = mlx_array_data_float32(x);
    float* logits = malloc((size_t)NE * sizeof(float));
    float* gate = malloc((size_t)F * sizeof(float));      /* attivazioni (F) */
    float* gw = malloc((size_t)F * E * sizeof(float));    /* slab gate (F x E) */
    float* uw = malloc((size_t)F * E * sizeof(float));    /* slab up   (F x E) */
    float* down = malloc((size_t)E * F * sizeof(float));  /* slab down (E x F) */
    float* op = malloc((size_t)E * sizeof(float));
    if (!logits || !gate || !gw || !uw || !down || !op) return -1;
    if (q4n_default_mv(NULL, r->gate_inp, xh, logits, E, NE) != 0) return -1;
    int lsh[2] = {1, NE}, xsh[2] = {1, E}, wsh[1] = {E};
    mlx_array aL = mlx_array_new_data(logits, lsh, 2, MLX_FLOAT32);
    mlx_array aX = mlx_array_new_data(xh, xsh, 2, MLX_FLOAT32);
    mlx_array aW = mlx_array_new_data(r->gate_inp_sh, wsh, 1, MLX_FLOAT32);
    mlx_array aSel = mlx_array_new(), aWt = mlx_array_new(), aSG = mlx_array_new();
    if (llm_q4n_router_topk(&aSel, &aWt, &aSG, aL, aX, aW, K, s) != 0) return -1;
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, aSel);
    mlx_vector_array_append_value(ev, aWt);
    mlx_vector_array_append_value(ev, aSG);
    mlx_eval(ev); mlx_vector_array_free(ev);
    const int* sel = mlx_array_data_int32(aSel);
    const float* wt = mlx_array_data_float32(aWt);
    const float sg = mlx_array_data_float32(aSG)[0];
    char nm[128];
    for (int d = 0; d < E; d++) op[d] = 0.0f;
    for (int k = 0; k < K; k++) {
        int e = sel[k];
        if (e < 0 || e >= NE) continue;
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", il);
        if (q4n_slab_f32(g, nm, e, gw) != 0) return -1;
        snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", il);
        if (q4n_slab_f32(g, nm, e, uw) != 0) return -1;
        for (int f = 0; f < F; f++) {
            double ga = 0.0, ua = 0.0;
            for (int d = 0; d < E; d++) { ga += (double)gw[f * E + d] * xh[d]; ua += (double)uw[f * E + d] * xh[d]; }
            float gv = (float)ga, sv = gv >= 0 ? 1.0f/(1.0f+expf(-gv)) : expf(gv)/(1.0f+expf(gv));
            gate[f] = gv * sv * (float)ua;
        }
        snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", il);
        if (q4n_slab_f32(g, nm, e, down) != 0) return -1;
        for (int d = 0; d < E; d++) {
            double acc = 0.0;
            for (int f = 0; f < F; f++) acc += (double)down[d * F + f] * gate[f];
            op[d] += wt[k] * (float)acc;
        }
    }
    { /* shared (f32) */
        for (int f = 0; f < F; f++) {
            double ga = 0.0, ua = 0.0;
            for (int d = 0; d < E; d++) { const float xd = xh[d]; ga += (double)r->gate_sh[f * E + d] * xd; ua += (double)r->up_sh[f * E + d] * xd; }
            float gv = (float)ga, sv = gv >= 0 ? 1.0f/(1.0f+expf(-gv)) : expf(gv)/(1.0f+expf(gv));
            gate[f] = gv * sv * (float)ua;
        }
        float sv = sg >= 0 ? 1.0f/(1.0f+expf(-sg)) : expf(sg)/(1.0f+expf(sg));
        for (int d = 0; d < E; d++) {
            double acc = 0.0;
            for (int f = 0; f < F; f++) acc += (double)r->down_sh[d * F + f] * gate[f];
            op[d] += sv * (float)acc;
        }
    }
    { int osh[2] = {1, E}; *out = mlx_array_new_data(op, osh, 2, MLX_FLOAT32); }
    mlx_array_free(aSel); mlx_array_free(aWt); mlx_array_free(aSG);
    mlx_array_free(aL); mlx_array_free(aX); mlx_array_free(aW);
    free(logits); free(gate); free(gw); free(uw); free(down); free(op);
    return 0;
}

#endif

/* Variante GPU (kernel ds4, Q8/Q2K/IQ2): gate+up in un batch, swiglu CPU, down in un batch. */
static int qwen4_native_moe_quant_gpu(mlx_array* out, mlx_array x, const LlmEngineMoeGguf* g, int il,
                                      const Q4nLayerRefs* r, int E, int NE, int K, int F,
                                      Q4nGpu* gpu, mlx_stream s) {
    if (!gpu || !gpu->b) return qwen4_native_moe_quant_cpu(out, x, g, il, r, E, NE, K, F, s);
    if (!out || !r || mlx_array_ndim(x) != 2) return -1;
    const float* xh = mlx_array_data_float32(x);
    float logits[512]; float wts[16]; int sel[16]; float sg;
    double m0 = q4nm_now(), m1, m2, m3, m4;
    if (NE > 512 || K > 16) return -1;
    if (q4n_default_mv(NULL, r->gate_inp, xh, logits, E, NE) != 0) return -1;
    { g_q4n_prof[11] += q4nm_now() - m0; }
    { /* router in CPU puro (evita il round-trip MLX per layer) */
        float mx = -3.0e38f; for (int e = 0; e < NE; e++) if (logits[e] > mx) mx = logits[e];
        float p[512]; float sum = 0.0f;
        for (int e = 0; e < NE; e++) { p[e] = expf(logits[e] - mx); sum += p[e]; }
        if (sum > 0.0f) { float inv = 1.0f / sum; for (int e = 0; e < NE; e++) p[e] *= inv; }
        signed char used[512]; memset(used, 0, (size_t)NE);
        float wsum = 0.0f;
        for (int r = 0; r < K; r++) {
            int bi = -1; float bp = -1.0f;
            for (int e = 0; e < NE; e++) if (!used[e] && p[e] > bp) { bp = p[e]; bi = e; }
            if (bi < 0) return -1;
            used[bi] = 1; sel[r] = bi; wts[r] = bp; wsum += bp;
        }
        if (wsum > 0.0f) { float invw = 1.0f / wsum; for (int r = 0; r < K; r++) wts[r] *= invw; }
        double sgd = 0.0;
        for (int i = 0; i < E; i++) sgd += (double)r->gate_inp_sh[i] * xh[i];
        sg = (float)sgd;
    }
    m1 = q4nm_now(); g_q4n_prof[4] += m1 - m0;
    int ig = llm_gguf_find_tensor(g, "blk.%d.ffn_gate_exps.weight") >= 0 ? 0 : 0; (void)ig;
    char nm[96];
    int gi, ui, di;
    snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", il); gi = llm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", il);   ui = llm_gguf_find_tensor(g, nm);
    snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", il); di = llm_gguf_find_tensor(g, nm);
    if (gi < 0 || ui < 0 || di < 0) return -1;
    const int gt = g->ti[gi].type, ut = g->ti[ui].type, dt = g->ti[di].type;
    if (gt != 8 || ut != 8 || dt != 8) return qwen4_native_moe_quant_cpu(out, x, g, il, r, E, NE, K, F, s);
    const uint8_t* gbase = llm_gguf_tensor_data(g, gi, NULL);
    const uint8_t* ubase = llm_gguf_tensor_data(g, ui, NULL);
    const uint8_t* dbase = llm_gguf_tensor_data(g, di, NULL);
    const size_t g_rb = (size_t)llm_ggml_nbytes(8, (int64_t)g->ti[gi].dims[0]);
    const size_t d_rb = (size_t)llm_ggml_nbytes(8, (int64_t)g->ti[di].dims[0]);
    const size_t u_rb = (size_t)llm_ggml_nbytes(8, (int64_t)g->ti[ui].dims[0]);
    const size_t g_slab = g_rb * (size_t)g->ti[gi].dims[1];
    const size_t d_slab = d_rb * (size_t)g->ti[di].dims[1];
    float* dw = malloc((size_t)K * E * sizeof(float));
    float* mid = malloc((size_t)K * F * sizeof(float));
    if (!dw || !mid) return -1;
    const void* gates[16]; const void* ups[16]; float* mids[16];
    int ne = 0;
    for (int k = 0; k < K; k++) {
        int e = sel[k]; if (e < 0 || e >= NE) continue;
        gates[ne] = gbase + (size_t)e * g_slab;
        ups[ne]   = ubase + (size_t)e * g_slab;
        mids[ne]  = mid + (size_t)k * F;
        ne++;
    }
    (void)gi; (void)ui;
    int nj = 0;
    if (q4n_gpu_pair_swiglu_q8(gpu, gates, ups, xh, mids, ne, F, E, g_rb, u_rb) != 0) return -1;
    m2 = q4nm_now(); g_q4n_prof[5] += m2 - m1; g_q4n_prof[6] += 0.0;
    m3 = m2;
    LlmDsv4WideJob wjobs[32];
    nj = 0;
    for (int k = 0; k < K; k++) {
        int e = sel[k]; if (e < 0 || e >= NE) continue;
        wjobs[nj++] = (LlmDsv4WideJob){ .w = dbase + (size_t)e * d_slab,
                                        .row_bytes = (uint32_t)d_rb, .n = (uint32_t)F, .out_dim = (uint32_t)E,
                                        .x = mid + (size_t)k * F, .out = dw + (size_t)k * E };
    }
    if (q4n_gpu_gemv_q8_wide_batch(gpu, wjobs, nj) != 0) return -1;
    m4 = q4nm_now(); g_q4n_prof[7] += m4 - m3;
    float outv[4096];
    if (E > 4096) return -1;
    for (int d = 0; d < E; d++) outv[d] = 0.0f;
    for (int k = 0; k < K; k++) for (int d = 0; d < E; d++) outv[d] += wts[k] * dw[(size_t)k*E+d];
    /* shared expert (CPU, f32 piccoli) */
    {
        float gsh[1024], ush[1024], y[4096];
        if (F > 1024 || E > 4096) return -1;
        for (int f = 0; f < F; f++) {
            double ga = 0, ua = 0;
            for (int d = 0; d < E; d++) { ga += (double)r->gate_sh[f*E+d]*xh[d]; ua += (double)r->up_sh[f*E+d]*xh[d]; }
            float gv = (float)ga, sv = gv>=0?1.f/(1.f+expf(-gv)):expf(gv)/(1.f+expf(gv));
            gsh[f] = gv*sv*(float)ua;
        }
        for (int d = 0; d < E; d++) { double acc=0; for (int f=0; f<F; f++) acc += (double)r->down_sh[d*F+f]*gsh[f]; y[d]=(float)acc; }
        float sv = sg>=0?1.f/(1.f+expf(-sg)):expf(sg)/(1.f+expf(sg));
        for (int d = 0; d < E; d++) outv[d] += sv*y[d];
    }
    g_q4n_prof[8] += q4nm_now() - m4;
    { int osh[2] = {1, E}; *out = mlx_array_new_data(outv, osh, 2, MLX_FLOAT32); }
    free(dw); free(mid);
    return 0;
}

static int qwen4_native_moe_quant(mlx_array* out, mlx_array x, const LlmEngineMoeGguf* g, int il,
                                  const Q4nLayerRefs* r, int E, int NE, int K, int F, mlx_stream s) {
    return qwen4_native_moe_quant_cpu(out, x, g, il, r, E, NE, K, F, s);
}
