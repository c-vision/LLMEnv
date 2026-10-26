/* qwen4_native_model.h — P7f: orchestrazione del forward completo (primo giro).
 * Catena ds4 `qwen4_ref_forward_token`:
 *   embed -> replica su hc stream -> per layer [PLE] -> hc_mix -> linear|attn -> combine
 *   -> hc_mix -> MoE -> combine -> ... -> output_hc_mix -> lm_head -> argmax.
 * PLE e MoE sono opzionali in questa fase (block f32/CPU); i pesi restano quelli
 * letti dal GGUF (q4n_layer_load). Stati (conv/GDN/KV/idx) per layer.
 */
#ifndef LLM_QWEN4_NATIVE_MODEL_H
#define LLM_QWEN4_NATIVE_MODEL_H

#include "mlx/c/mlx.h"
#include "qwen4_native_gguf.h"
#include "qwen4_native_attn_layer.h"
#include "qwen4_native_trunk.h"
#include "qwen4_native_moe_layer.h"
#include "qwen4_native_moe_quant.h"
#include "qwen4_native_ple.h"
#include "llm_ggml.h"

typedef struct {
    int cap, pos;
    mlx_array conv_hist[64];   /* [K-1][conv_dim] */
    mlx_array lin_state[64];   /* [Hv][D][D] */
    float* k_cache;            /* [L][cap][Hkv*D] */
    float* v_cache;
    float* idx_cache;          /* [L][cap][Di] */
} Q4nState;

typedef struct {
    const char* path;
    LlmEngineMoeGguf gguf; void* map; size_t maplen;
    Q4nHp hp;
    float* token_embd;   /* [V][E] f32 */
    float* output;       /* [V][E] f32 */
    float* out_norm, *out_down, *out_up;
    Q4nState st;
    int use_moe;         /* 0 = salta il blocco MoE (smoke/f32) */
    int* layer_linear;   /* [L] da model.gmb.config layer_types */
    float* rope_freq;    /* [n_rot/2] */
    Q4nPleMeta ple_meta;
    Q4nPleState ple_st;
    Q4nLayerRefs* lrefs;      /* [L] caricati una volta */
    uint8_t* lrefs_ok;
} Q4nModel;

static void q4n_state_free(Q4nState* st, int L) {
    for (int i = 0; i < L; i++) { mlx_array_free(st->conv_hist[i]); mlx_array_free(st->lin_state[i]); }
    free(st->k_cache); free(st->v_cache); free(st->idx_cache);
    memset(st, 0, sizeof *st);
}

/* riga `row` di un tensore GGUF [cols,in ...]: dequant in out (cols valori). */
static int q4n_tensor_row(const LlmEngineMoeGguf* g, const char* name, uint64_t row, float* out) {
    int i = llm_gguf_find_tensor(g, name);
    if (i < 0) return -1;
    int type = g->ti[i].type;
    uint64_t cols = (uint64_t)g->ti[i].dims[0];
    int rb = (int)llm_ggml_nbytes(type, (int64_t)cols);
    const uint8_t* base = llm_gguf_tensor_data(g, i, NULL);
    if (!base || rb <= 0) return -1;
    return llmcodec_ggml_dequant_row(type, base + row * (uint64_t)rb, out, (int64_t)cols);
}

static int q4n_model_open(Q4nModel* m, const char* path, int cap, int use_moe) {
    if (!m || !path || cap <= 0) return -1;
    memset(m, 0, sizeof *m);
    m->path = path;
    if (!llm_gguf_open_file(path, &m->gguf, &m->map, &m->maplen)) return -1;
    char err[256] = {0};
    if (q4n_hparams(&m->gguf, &m->hp, err, sizeof err) != 0) return -1;
    const Q4nHp* h = &m->hp;
    m->token_embd = NULL; m->output = NULL;   /* letti a riga (RAM costante) */
    bool ok = true;
    m->out_norm = q4n_f32_global(&m->gguf, "output_hc_norm.weight", true, &ok);
    m->out_down = q4n_f32_global(&m->gguf, "output_hc_down.weight", true, &ok);
    m->out_up = q4n_f32_global(&m->gguf, "output_hc_up.weight", true, &ok);
    if (!m->out_norm || !m->out_down || !m->out_up) return -1;
    m->use_moe = use_moe;
    m->layer_linear = calloc((size_t)h->L, sizeof(int));
    { const LlmGgufVal* cv = llm_gguf_find_val(&m->gguf, "model.gmb.config");
      const char* js = cv ? cv->s : NULL;
      const char* p = js ? strstr(js, "\"layer_types\"") : NULL;
      if (p) { p = strchr(p, '['); }
      for (int il = 0; p && il < h->L; il++) {
          const char* a = strstr(p, "\"linear_attention\"");
          const char* b = strstr(p, "\"full_attention\"");
          const char* n = NULL;
          if (a && (!b || a < b)) { m->layer_linear[il] = 1; n = a + 16; }
          else if (b) { m->layer_linear[il] = 0; n = b + 14; }
          if (!n) break;
          p = n;
      }
    }
    m->lrefs = calloc((size_t)h->L, sizeof(Q4nLayerRefs));
    m->lrefs_ok = calloc((size_t)h->L, 1);
    m->rope_freq = malloc((size_t)(h->n_rot / 2) * sizeof(float));
    for (int i = 0; i < h->n_rot / 2; i++) m->rope_freq[i] = (float)pow((double)h->rope_base, -2.0 * i / (double)h->n_rot);
    if (q4n_ple_meta(&m->gguf, &m->hp, &m->ple_meta) != 0) { m->ple_meta.n_heads = 0; }
    if (m->ple_meta.n_heads > 0) {
        int ph[2] = {(h->ple_conv - 1) * h->ple_ng, h->hc * h->E};
        float* z = calloc((size_t)ph[0] * ph[1], sizeof(float));
        m->ple_st.hist = mlx_array_new_data(z, ph, 2, MLX_FLOAT32);
        free(z);
        for (int i = 0; i < 4; i++) m->ple_st.prev[i] = h->ple_eos;
    }
    m->st.cap = cap;
    m->st.k_cache = calloc((size_t)h->L * cap * h->Hkv * h->D, sizeof(float));
    m->st.v_cache = calloc((size_t)h->L * cap * h->Hkv * h->D, sizeof(float));
    m->st.idx_cache = calloc((size_t)h->L * cap * h->Di, sizeof(float));
    for (int il = 0; il < h->L; il++) {
        int hsz[2] = {h->lin_conv - 1, 2 * h->lin_Hk * h->lin_D + h->lin_Hv * h->lin_D};
        int ssz[3] = {h->lin_Hv, h->lin_D, h->lin_D};
        float* z = calloc((size_t)hsz[0] * hsz[1], sizeof(float));
        float* zs = calloc((size_t)ssz[0] * ssz[1] * ssz[2], sizeof(float));
        m->st.conv_hist[il] = mlx_array_new_data(z, hsz, 2, MLX_FLOAT32);
        m->st.lin_state[il] = mlx_array_new_data(zs, ssz, 3, MLX_FLOAT32);
        free(z); free(zs);
    }
    return 0;
}

static void q4n_model_free(Q4nModel* m) {
    if (!m) return;
    q4n_state_free(&m->st, m->hp.L);
    if (m->lrefs) for (int i = 0; i < m->hp.L; i++) if (m->lrefs_ok && m->lrefs_ok[i]) q4n_layer_free(&m->lrefs[i]);
    free(m->lrefs); free(m->lrefs_ok);
    free(m->out_norm); free(m->out_down); free(m->out_up); free(m->layer_linear); free(m->rope_freq);
    llm_gguf_close_file(&m->gguf, m->map, m->maplen);
}

/* matvec f32 row-major (W:[n_out][n_in]) */
typedef struct { const float* W; int n_in, n_out; } Q4nMvJob;

/* forward di un token; logits [V] nel buffer del chiamante. 0 = ok. */
static int q4n_model_forward(Q4nModel* m, int token, float* logits, mlx_stream s) {
    if (!m || !logits || token < 0) return -1;
    const Q4nHp* h = &m->hp;
    const int E = h->E, hc = h->hc, hd = hc * E;
    if (m->st.pos >= m->st.cap) return -1;
    const int pos = m->st.pos;
    float* R = malloc((size_t)hd * sizeof(float));
    if (!R) return -1;
    if (q4n_tensor_row(&m->gguf, "token_embd.weight", (uint64_t)token, R) != 0) { free(R); return -1; }
    for (int st = 1; st < hc; st++) memcpy(R + (size_t)st * E, R, (size_t)E * sizeof(float));
    int rsh[2] = {1, hd};
    mlx_array aR = mlx_array_new_data(R, rsh, 2, MLX_FLOAT32);
    for (int il = 0; il < h->L; il++) {
        char pkn[64]; snprintf(pkn, sizeof pkn, "blk.%d.ple_key.weight", il);
        int is_ple = llm_gguf_find_tensor(&m->gguf, pkn) >= 0;
        int is_linear = (il == 0);
        if (m->layer_linear) is_linear = m->layer_linear[il];
        Q4nLayerRefs refs; char err[256] = {0};
        if (q4n_layer_load(&m->gguf, il, is_linear, is_ple, &refs, err, sizeof err) != 0) continue;
        if (is_ple && m->ple_meta.n_heads > 0) {
            if (qwen4_native_ple_layer(&aR, &m->gguf, il, &refs, &m->ple_meta, &m->ple_st, token,
                                       E, hc, h->ple_ng, h->ple_eos, h->ple_conv, h->eps,
                                       q4n_default_mv, s) != 0) return -1;
        }
        Q4nHcW wa = {refs.hc_attn_norm, refs.hc_attn_down, refs.hc_attn_up, refs.hc_attn_inject, h->rank};
        mlx_array m1 = mlx_array_new(), i1 = mlx_array_new();
        if (qwen4_native_hc_mix(&m1, &i1, aR, &wa, E, hc, h->eps, q4n_default_mv, NULL, s) != 0) return -1;
        mlx_array blk = mlx_array_new();
        if (!is_linear) {
            const float* iq = refs.idx_q_proj;
            const float* ik = refs.idx_k_proj;
            if (!iq && refs.idx_qk) { iq = refs.idx_qk; ik = refs.idx_qk + (size_t)h->Hi * h->Di * h->E; }
            Q4nAttnW aw = {refs.attn_q, refs.attn_k, refs.attn_v, refs.attn_q_norm, refs.attn_k_norm,
                           refs.attn_out, iq, ik, refs.idx_q_norm, refs.idx_k_norm,
                           h->H, h->Hkv, h->D, h->Hi, h->Di};
            size_t stride = (size_t)h->Hkv * h->D;
            float* kc = m->st.k_cache + (size_t)il * m->st.cap * stride;
            float* vc = m->st.v_cache + (size_t)il * m->st.cap * stride;
            float* ic = m->st.idx_cache + (size_t)il * m->st.cap * h->Di;
            if (qwen4_native_attn_layer(&blk, m1, kc, vc, ic, m->st.cap, pos, &aw,
                                        m->rope_freq, h->n_rot, h->eps, q4n_default_mv, NULL, s) != 0) return -1;
        } else {
        Q4nGdnW gw = {refs.attn_qkv, refs.attn_gate, refs.ssm_beta, refs.ssm_alpha, refs.ssm_conv1d,
                      refs.ssm_a, refs.ssm_dt, refs.ssm_norm, refs.ssm_out,
                      2 * h->lin_Hk * h->lin_D + h->lin_Hv * h->lin_D, h->lin_Hv * h->lin_D,
                      h->lin_Hk, h->lin_Hv, h->lin_D};
        if (qwen4_native_gdn_layer(&blk, m1, &m->st.conv_hist[il], &m->st.lin_state[il], &gw,
                                   q4n_default_mv, NULL, s) != 0) return -1;
        }
        mlx_array R1 = mlx_array_new();
        if (llm_q4n_hc_combine(&R1, aR, blk, i1, s) != 0) return -1;
        Q4nHcW wf = {refs.hc_ffn_norm, refs.hc_ffn_down, refs.hc_ffn_up, refs.hc_ffn_inject, h->rank};
        mlx_array m2 = mlx_array_new(), i2 = mlx_array_new();
        if (qwen4_native_hc_mix(&m2, &i2, R1, &wf, E, hc, h->eps, q4n_default_mv, NULL, s) != 0) return -1;
        mlx_array b2 = mlx_array_new();
        if (m->use_moe) {
            if (qwen4_native_moe_quant(&b2, m2, &m->gguf, il, &refs, E, h->NE, h->K, h->F, s) != 0) return -1;
        } else {
            float* z = calloc((size_t)E, sizeof(float));
            int zsh[2] = {1, E};
            b2 = mlx_array_new_data(z, zsh, 2, MLX_FLOAT32);
            free(z);
        }
        mlx_array R2 = mlx_array_new();
        if (llm_q4n_hc_combine(&R2, R1, b2, i2, s) != 0) return -1;
        aR = R2;
        mlx_array_free(R1); mlx_array_free(m1); mlx_array_free(i1);
        mlx_array_free(m2); mlx_array_free(i2); mlx_array_free(blk); mlx_array_free(b2);
        q4n_layer_free(&refs);
#if defined(__APPLE__)
        { extern void malloc_zone_pressure_relief(void*, unsigned long); malloc_zone_pressure_relief((void*)0, 0); }
#endif
    }
    /* output_hc_mix (no inject) -> mixed [E] */
    Q4nHcW wo = {m->out_norm, m->out_down, m->out_up, NULL, h->rank};
    mlx_array mixed = mlx_array_new(), dummy = mlx_array_new();
    if (qwen4_native_hc_mix(&mixed, &dummy, aR, &wo, E, hc, h->eps, q4n_default_mv, NULL, s) != 0) return -1;
    mlx_vector_array ev = mlx_vector_array_new();
    mlx_vector_array_append_value(ev, mixed);
    mlx_eval(ev); mlx_vector_array_free(ev);
    const float* mh = mlx_array_data_float32(mixed);
    const int V = (int)(h->E > 0 ? 248320 : 0);   /* vocab dal GGUF non in hp: output rows */
    (void)V;
    /* lm_head: quante righe? dedotto dal metadata vocab_size */
    const uint64_t vs = llm_gguf_u(&m->gguf, "qwen4exp.vocab_size", 0, "qwen4_exp.vocab_size");
    if (vs == 0) return -1;
    {   /* lm_head a blocchi: dequant di poche righe per volta (nessun array VxE in RAM) */
        int oi = llm_gguf_find_tensor(&m->gguf, "output.weight");
        if (oi < 0) return -1;
        int ot = m->gguf.ti[oi].type;
        int rb = (int)llm_ggml_nbytes(ot, (int64_t)E);
        const uint8_t* obase = llm_gguf_tensor_data(&m->gguf, oi, NULL);
        if (!obase || rb <= 0) return -1;
        enum { CH = 64 };
        float buf[64 * 4096];
        if (E > 4096) return -1;
        for (uint64_t v0 = 0; v0 < vs; v0 += CH) {
            uint64_t nc = vs - v0; if (nc > CH) nc = CH;
            for (uint64_t r = 0; r < nc; r++)
                if (llmcodec_ggml_dequant_row(ot, obase + (v0 + r) * (uint64_t)rb, buf + r * E, (int64_t)E) != 0) return -1;
            for (uint64_t r = 0; r < nc; r++) {
                const float* row = buf + r * E; double acc = 0.0;
                for (int d = 0; d < E; d++) acc += (double)row[d] * mh[d];
                logits[v0 + r] = (float)acc;
            }
        }
    }
    mlx_array_free(mixed); mlx_array_free(dummy); mlx_array_free(aR);
    free(R);
    m->st.pos = pos + 1;
    return 0;
}

#endif /* LLM_QWEN4_NATIVE_MODEL_H */
