/* qwen4_native_moe_quant.h — MoE con esperti QUANTIZZATI (Q8_0/F16/...): router f32
 * via kernel, poi dequant dello slab del singolo esperto (solo i 10 selezionati).
 * Layout GGUF 3D: dims [in, out, E] -> slab esperto contiguo (out*in). */
#ifndef LLM_QWEN4_NATIVE_MOE_QUANT_H
#define LLM_QWEN4_NATIVE_MOE_QUANT_H

#include "mlx/c/mlx.h"
#include "llm_ggml.h"
#include "qwen4_native_gguf.h"
#include "qwen4_native_router_kernel.h"
#include "qwen4_native_layer.h"

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

static int qwen4_native_moe_quant(mlx_array* out, mlx_array x, const LlmEngineMoeGguf* g, int il,
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
