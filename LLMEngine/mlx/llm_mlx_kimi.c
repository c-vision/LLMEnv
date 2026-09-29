/* ============================================================================
 * llm_mlx_kimi.c — dominio Kimi Linear: MLA assorbita, KDA (Gated DeltaNet) conv weight,
 * delta-attention e dispatch ibrido attn/ffn. Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_kv_quant_sdpa_kernel.h"
#include "llm_mlx_gdn_gate_kernel.h"
#include "llm_mlx_gdn_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Kimi Linear — MLA ASSORBITA (KimiMLAAttention). Vedi mlx_lm/models/kimi_linear.py + mla.py.
 * Cache: `kv_latent` [1,1,S,kv_lora] in `k_cache`; `k_pe` [1,1,S,qk_rope] in `m->aux_cache[layer]`.
 * Decode (L==1): Q assorbito con `embed_q` -> SDPA(q_emb, kv_latent, kv_latent) + mask pe_scores ->
 * `unembed_out`. Prefill (L>1): k_nope=embed_q(kv_latent, transpose=false), v=unembed_out(...).
 * `embed_q`/`unembed_out` sono MultiLinear per-head: quantized_matmul con peso 3D [H, out, in]. */
bool kimi_mla_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens,
                                     mlx_array x, mlx_array* k_cache, mlx_array* v_cache, bool has_cache,
                                     mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const char* mode = m->cfg.quant_mode;
    const int H = m->cfg.n_heads;
    const int qn = m->cfg.qk_nope_head_dim, qr = m->cfg.qk_rope_head_dim;
    const int vhd = m->cfg.v_head_dim, kvl = m->cfg.kv_lora_rank;
    const int qh = qn + qr;
    char nb[256];
    mlx_array ln=mlx_array_new(), qw=mlx_array_new(),qs=mlx_array_new(),qb=mlx_array_new();
    mlx_array aw=mlx_array_new(),as=mlx_array_new(),ab=mlx_array_new(),aln=mlx_array_new();
    mlx_array ew=mlx_array_new(),es=mlx_array_new(),eb=mlx_array_new();
    mlx_array uw=mlx_array_new(),us=mlx_array_new(),ub=mlx_array_new();
    mlx_array ow=mlx_array_new(),os=mlx_array_new(),ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &ln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &qw,&qs,&qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_a_proj_with_mqa", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &aw,&as,&ab);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_a_layernorm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &aln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.embed_q", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ew,&es,&eb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.unembed_out", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &uw,&us,&ub);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&os,&ob);

    mlx_array normed = mlx_array_new();
    if (ok) ok = rms_norm_cfg(&normed, x, ln, m->cfg.rms_norm_eps, m->stream);
    mlx_array q=mlx_array_new(), q4=mlx_array_new(), qt=mlx_array_new(), q_nope=mlx_array_new(), q_pe=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&q, normed, qw,qs,qb, gsz,bts, mode, m->stream);
    /* §D R6 (kimi_mla): LoRA su q_proj. */
    { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { int sh[4]={1,n_tokens,H,qh}; ok = mlx_reshape(&q4, q, sh, 4, m->stream)==0; }
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&qt, q4, perm, 4, m->stream)==0; }
    if (ok) { int _sec[1]={qn}; mlx_vector_array _sp=mlx_vector_array_new();
              ok = mlx_split_sections(&_sp, qt, _sec, 1, -1, m->stream)==0 &&
                   mlx_vector_array_get(&q_nope, _sp, 0)==0 && mlx_vector_array_get(&q_pe, _sp, 1)==0;
              mlx_vector_array_free(_sp); }

    mlx_array ckv=mlx_array_new(), ckv3=mlx_array_new(), compressed=mlx_array_new(), kpe3=mlx_array_new(), kpe=mlx_array_new(), kvl4=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&ckv, normed, aw,as,ab, gsz,bts, mode, m->stream);
    /* §D R6 (kimi_mla): LoRA su kv_a_proj_with_mqa. */
    { int li = llm_mlx_model_lora_get(m, layer_idx, "kv_a_proj_with_mqa"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&ckv, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { int sh[3]={1,n_tokens,kvl+qr}; ok = mlx_reshape(&ckv3, ckv, sh, 3, m->stream)==0; }
    if (ok) { int _sec[1]={kvl}; mlx_vector_array _sp=mlx_vector_array_new();
              ok = mlx_split_sections(&_sp, ckv3, _sec, 1, -1, m->stream)==0 &&
                   mlx_vector_array_get(&compressed, _sp, 0)==0 && mlx_vector_array_get(&kpe3, _sp, 1)==0;
              mlx_vector_array_free(_sp); }
    if (ok) { int sh[4]={1,1,n_tokens,qr}; ok = mlx_reshape(&kpe, kpe3, sh, 4, m->stream)==0; }
    if (ok) { int sh[4]={1,1,n_tokens,kvl}; mlx_array kvl3=mlx_array_new();
              ok = rms_norm_cfg(&kvl3, compressed, aln, m->cfg.rms_norm_eps, m->stream) && mlx_reshape(&kvl4, kvl3, sh, 4, m->stream)==0; mlx_array_free(kvl3); }

    /* NIENTE RoPE: KimiMLAAttention usa q_pe/k_pe GREZZI come bias ADDITIVO
     * (pe_scores = (q_pe*scale) @ k_pe^T). Verificato sulla reference
     * mlx_lm/models/kimi_linear.py: il suo __call__ NON chiama mx.fast.rope. */
    mlx_array q_pe_r=mlx_array_new(), kpe_r=mlx_array_new();
    if (ok) ok = mlx_array_set(&q_pe_r, q_pe)==0 && mlx_array_set(&kpe_r, kpe)==0;
    /* Append kv_latent a k_cache, k_pe a aux_cache[layer] (entrambi [1,1,S,*]) */
    mlx_array k_full=mlx_array_new(), aux_old=mlx_array_new(), aux_full=mlx_array_new();
    if (ok) {
        if (has_cache) {
            mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,*k_cache); mlx_vector_array_append_value(v,kvl4);
            ok = mlx_concatenate_axis(&k_full, v, 2, m->stream)==0; mlx_vector_array_free(v);
        } else ok = mlx_array_set(&k_full, kvl4)==0;
    }
    if (ok) {
        mlx_array* ac = (m->aux_cache && layer_idx < m->cfg.n_layers) ? &m->aux_cache[layer_idx] : NULL;
        if (ac && (*ac).ctx != NULL) { mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,*ac); mlx_vector_array_append_value(v,kpe_r);
            ok = mlx_concatenate_axis(&aux_full, v, 2, m->stream)==0; mlx_vector_array_free(v); }
        else ok = mlx_array_set(&aux_full, kpe_r)==0;
        /* Store dell'aux_cache per-layer (k_pe cumulativo): indispensabile per il decode MLA. */
        if (ok && ac) ok = mlx_array_set(ac, aux_full)==0;
    }
    if (ok) ok = mlx_array_set(k_cache, k_full)==0;   /* v_cache NON usata per l'MLA assorbita */

    /* scores_pe = (q_pe*scale) @ k_pe^T + mascherina causale additiva */
    mlx_array kpe_c = mlx_array_new(), kpe_ct=mlx_array_new(), qpes=mlx_array_new();
    mlx_array spe=mlx_array_new(), addmask=mlx_array_new(), spe_m=mlx_array_new();
    const float scale = 1.0f / sqrtf((float)qh);
    if (ok) {
        mlx_array* ac = (m->aux_cache && layer_idx < m->cfg.n_layers) ? &m->aux_cache[layer_idx] : NULL;
        mlx_array kc = (ac && (*ac).ctx != NULL) ? *ac : kpe_r;
        int nd=(int)mlx_array_ndim(kc); int perm[8]; for(int i=0;i<nd;i++)perm[i]=i; int t=perm[nd-1];perm[nd-1]=perm[nd-2];perm[nd-2]=t;
        ok = mlx_transpose_axes(&kpe_ct, kc, perm, nd, m->stream)==0;
        mlx_array sf = mlx_array_new_float32(scale), s = mlx_array_new();
        if (ok) ok = mlx_astype(&s, sf, mlx_array_dtype(q_pe_r), m->stream)==0;   /* scalare nel dtype di q_pe */
        if (ok) ok = mlx_multiply(&qpes, q_pe_r, s, m->stream)==0;
        mlx_array_free(sf); mlx_array_free(s);
        if (ok) ok = mlx_matmul(&spe, qpes, kpe_ct, m->stream)==0;   /* [1,H,L,S] */
    }
    if (ok) {   /* mask causale additiva [1,1,L,S]: 0 se j<=cache_offset+i, else -inf */
        int S = (int)mlx_array_shape(*k_cache)[2];
        mlx_array iq=mlx_array_new(), ik=mlx_array_new(), iqe=mlx_array_new(), ike=mlx_array_new(), cmp=mlx_array_new();
        mlx_arange(&iq, 0, n_tokens, 1, MLX_INT32, m->stream);
        mlx_arange(&ik, 0, S, 1, MLX_INT32, m->stream);
        mlx_array coff = mlx_array_new_int(cache_offset);
        mlx_array ip = mlx_array_new(); ok = mlx_add(&ip, iq, coff, m->stream)==0; mlx_array_free(coff);
        if (ok) { int sh[2]={n_tokens,1}; ok = mlx_reshape(&iqe, ip, sh, 2, m->stream)==0; }
        if (ok) { int sh[2]={1,S}; ok = mlx_reshape(&ike, ik, sh, 2, m->stream)==0; }
        if (ok) ok = mlx_greater_equal(&cmp, iqe, ike, m->stream)==0;   /* iqe>=ike -> j<=pos */
        mlx_array zerof=mlx_array_new_float32(0.0f), ninff=mlx_array_new_float32(-INFINITY);
        mlx_array zero=mlx_array_new(), ninf=mlx_array_new();
        if (ok) ok = mlx_astype(&zero, zerof, mlx_array_dtype(spe), m->stream)==0 &&
                     mlx_astype(&ninf, ninff, mlx_array_dtype(spe), m->stream)==0;   /* mask nel dtype di spe (niente f32) */
        if (ok) ok = mlx_where(&addmask, cmp, zero, ninf, m->stream)==0;
        mlx_array_free(zerof); mlx_array_free(ninff); mlx_array_free(zero); mlx_array_free(ninf);
        if (ok) { int sh[4]={1,1,n_tokens,S}; ok = mlx_reshape(&addmask, addmask, sh, 4, m->stream)==0; }
        mlx_array_free(iq);mlx_array_free(ik);mlx_array_free(iqe);mlx_array_free(ike);mlx_array_free(cmp);mlx_array_free(ip);
    }
    if (ok) ok = mlx_add(&spe_m, spe, addmask, m->stream)==0;

    /* SDPA assorbita */
    mlx_array attn=mlx_array_new(), k_attn=mlx_array_new(), v_attn=mlx_array_new(), out_attn=mlx_array_new();
    mlx_optional_int gsz_o={gsz,true}, bts_o={bts,true};
    mlx_array no_sinks = mlx_array_new();
    if (ok && n_tokens == 1) {
        ok = mlx_quantized_matmul(&k_attn, q_nope, ew, es, eb, /*transpose=*/true, gsz_o, bts_o, mode, m->stream)==0;   /* q_emb [1,H,1,kvl] */
        if (ok) ok = mlx_fast_scaled_dot_product_attention(&attn, k_attn, *k_cache, *k_cache, scale, "", spe_m, no_sinks, m->stream)==0;
        if (ok) ok = mlx_quantized_matmul(&out_attn, attn, uw, us, ub, true, gsz_o, bts_o, mode, m->stream)==0;
    } else if (ok) {
        /* prefill: k_nope = embed_q(kv_latent, transpose=false) [peso H,kvl,qn];
         * v = unembed_out(kv_latent) con transpose=TRUE [peso H,vhd,kvl come nn.Linear]. */
        if (ok) ok = mlx_quantized_matmul(&k_attn, kvl4, ew, es, eb, /*transpose=*/false, gsz_o, bts_o, mode, m->stream)==0;
        if (ok) ok = mlx_quantized_matmul(&v_attn, kvl4, uw, us, ub, /*transpose=*/true, gsz_o, bts_o, mode, m->stream)==0;
        if (ok) ok = mlx_fast_scaled_dot_product_attention(&attn, q_nope, k_attn, v_attn, scale, "", spe_m, no_sinks, m->stream)==0;
        if (ok) ok = mlx_array_set(&out_attn, attn)==0;
    }
    mlx_array_free(no_sinks);

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new(), proj=mlx_array_new();
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, out_attn, perm, 4, m->stream)==0; }
    if (ok) { int sh[3]={1,n_tokens,H*vhd}; ok = mlx_reshape(&attn_flat, attn_bt, sh, 3, m->stream)==0; }
    if (ok) ok = quantized_linear_cfg(&proj, attn_flat, ow,os,ob, gsz,bts, mode, m->stream);
    /* §D R6 (kimi_mla): LoRA su o_proj. */
    { int li = llm_mlx_model_lora_get(m, layer_idx, "o_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&proj, attn_flat, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) ok = mlx_add(out_h, x, proj, m->stream)==0;

    mlx_array_free(ln);mlx_array_free(qw);mlx_array_free(qs);mlx_array_free(qb);
    mlx_array_free(aw);mlx_array_free(as);mlx_array_free(ab);mlx_array_free(aln);
    mlx_array_free(ew);mlx_array_free(es);mlx_array_free(eb);mlx_array_free(uw);mlx_array_free(us);mlx_array_free(ub);
    mlx_array_free(ow);mlx_array_free(os);mlx_array_free(ob);mlx_array_free(normed);
    mlx_array_free(q);mlx_array_free(q4);mlx_array_free(qt);mlx_array_free(q_nope);mlx_array_free(q_pe);
    mlx_array_free(ckv);mlx_array_free(ckv3);mlx_array_free(compressed);mlx_array_free(kpe3);mlx_array_free(kpe);mlx_array_free(kvl4);
    mlx_array_free(q_pe_r);mlx_array_free(kpe_r);mlx_array_free(k_full);mlx_array_free(aux_old);mlx_array_free(aux_full);
    mlx_array_free(kpe_c);mlx_array_free(kpe_ct);mlx_array_free(qpes);mlx_array_free(spe);mlx_array_free(addmask);mlx_array_free(spe_m);
    mlx_array_free(attn);mlx_array_free(k_attn);mlx_array_free(v_attn);mlx_array_free(out_attn);
    mlx_array_free(attn_bt);mlx_array_free(attn_flat);mlx_array_free(proj);
    return ok;
}

/* Cache per-layer del peso conv KDA fuso (concat dei 3 ShortConv1d + cast al dtype di
 * attivazione), costante nel tempo: ricostruirlo a ogni token era colla di dispatch.
 * Condivisa tra `kimi_delta_attention_block` (path singolo) e il path batched. */
static mlx_array g_kda_cw_cache[512];
static const void* g_kda_cw_model = NULL;
static bool g_kda_cw_ready[512];
bool kimi_kda_conv_weight(LlmMlxModel* m, int layer_idx, mlx_stream stream,
                                 mlx_array cqw, mlx_array ckw, mlx_array cvw,
                                 mlx_dtype act_dt, mlx_array* out) {
    if (g_kda_cw_model != (const void*)m) {
        for (int i = 0; i < 512; i++) {
            if (g_kda_cw_cache[i].ctx) mlx_array_free(g_kda_cw_cache[i]);
            g_kda_cw_cache[i] = mlx_array_new();
            g_kda_cw_ready[i] = false;
        }
        g_kda_cw_model = (const void*)m;
    }
    if (layer_idx < 0 || layer_idx >= 512) return false;
    if (g_kda_cw_ready[layer_idx]) return mlx_array_set(out, g_kda_cw_cache[layer_idx]) == 0;
    mlx_array cwcat = mlx_array_new();
    mlx_array ca[3] = {cqw, ckw, cvw};
    mlx_vector_array cv = mlx_vector_array_new_data(ca, 3);
    bool ok = mlx_concatenate_axis(&cwcat, cv, 0, stream) == 0;
    mlx_vector_array_free(cv);
    if (ok) ok = mlx_astype(out, cwcat, act_dt, stream) == 0;
    if (ok) { mlx_array_set(&g_kda_cw_cache[layer_idx], *out); g_kda_cw_ready[layer_idx] = true; }
    mlx_array_free(cwcat);
    return ok;
}

/* Kimi Linear — KDA (Gated DeltaNet) per i layer `kda_layers`. Rif:
 * mlx_lm/models/kimi_linear.py (KimiDeltaAttention) + gated_delta.py.
 * Differenze dal GDN qwen (gdn_attention_block):
 *  - pesi: q_proj/k_proj/v_proj separati; a = f_b_proj(f_a_proj(x)); b = b_proj(x);
 *    gate = g_b_proj(g_a_proj(x)); A_log [H], dt_bias [H*D].
 *  - conv: 3 ShortConv1d (q_conv/k_conv/v_conv) fusi in UN depthwise (come qwen): pesi
 *    concat [3P,kk,1], input concat([state|qkv]) con stato unico [B,kk-1,3P].
 *  - g VETTORIALE [B,S,H,D] = exp(-exp(A_log[H,1])*softplus(a+dt_bias[H,D])) -> helper _ext.
 *  - uscita: o_norm(out)*sigmoid(gate) -> o_proj (NON silu(z) come qwen).
 * Stato: k_cache[layer] = ssm [B,H,D,D]; v_cache[layer] = conv [B,kk-1,3P]. */
bool kimi_delta_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                       mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const char* mode = m->cfg.quant_mode;
    char nb[256];
    const int B = 1, S = n_tokens;
    const int H = m->cfg.linear_attn_num_heads > 0 ? m->cfg.linear_attn_num_heads : m->cfg.n_heads;
    const int D = m->cfg.linear_attn_head_dim > 0 ? m->cfg.linear_attn_head_dim : m->cfg.head_dim;
    const int kk = m->cfg.linear_attn_conv_kernel > 0 ? m->cfg.linear_attn_conv_kernel : 4;
    const int P = H * D, CD = 3 * P, csl = kk - 1;
    mlx_stream s = m->stream;
#define KCHK(cond, what) do { if (!(cond)) { fprintf(stderr, "kimi_kda[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } } while (0)
/* Workaround MLX (indagare): la PRIMA quantized_matmul dopo una sequenza di op lazy puo'
 * fallire (rc!=0) mentre un identico retry riesce. Riproviamo una volta. */
#define KLIN(dst, x, w, sc, b) do { \
        if (ok) ok = quantized_linear_cfg(&(dst), (x), (w), (sc), (b), gsz, bts, mode, s); \
    } while (0)

    mlx_array ln=mlx_array_new(), h=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &ln);
    if (ok) ok = rms_norm_cfg(&h, x, ln, m->cfg.rms_norm_eps, s);
    KCHK(ok, "input_layernorm");

    mlx_array qw=mlx_array_new(),qs=mlx_array_new(),qb=mlx_array_new();
    mlx_array kw=mlx_array_new(),ks=mlx_array_new(),kb=mlx_array_new();
    mlx_array vw=mlx_array_new(),vs=mlx_array_new(),vb=mlx_array_new();
    mlx_array faw=mlx_array_new(),fas=mlx_array_new(),fab=mlx_array_new();
    mlx_array fbw=mlx_array_new(),fbs=mlx_array_new(),fbb=mlx_array_new();
    mlx_array bw=mlx_array_new(),bs_=mlx_array_new(),bb=mlx_array_new();
    mlx_array gaw=mlx_array_new(),gas=mlx_array_new(),gab=mlx_array_new();
    mlx_array gbw=mlx_array_new(),gbs=mlx_array_new(),gbb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &qw,&qs,&qb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &kw,&ks,&kb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &vw,&vs,&vb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.f_a_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &faw,&fas,&fab);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.f_b_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &fbw,&fbs,&fbb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.b_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &bw,&bs_,&bb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.g_a_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &gaw,&gas,&gab);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.g_b_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &gbw,&gbs,&gbb);
    KCHK(ok, "proj lookups");

    mlx_array qp=mlx_array_new(),kp=mlx_array_new(),vp=mlx_array_new();
    KLIN(qp,h,qw,qs,qb); KLIN(kp,h,kw,ks,kb); KLIN(vp,h,vw,vs,vb);
    KCHK(ok, "q/k/v proj");

    mlx_array cqw=mlx_array_new(),ckw=mlx_array_new(),cvw=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_conv.conv.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &cqw);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_conv.conv.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &ckw);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_conv.conv.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &cvw);
    KCHK(ok, "conv weight lookups");
    /* Peso conv fuso (concat 3 + cast) CACHATO per layer (helper condiviso col path batched). */
    mlx_array cw32 = mlx_array_new();
    if (ok) ok = kimi_kda_conv_weight(m, layer_idx, s, cqw, ckw, cvw, mlx_array_dtype(qp), &cw32);
    mlx_array qkv=mlx_array_new();
    if (ok) { mlx_array qa[3]={qp,kp,vp}; mlx_vector_array qv=mlx_vector_array_new_data(qa,3); ok = mlx_concatenate_axis(&qkv,qv,-1,s)==0; mlx_vector_array_free(qv); }
    KCHK(ok, "conv weight/qkv concat");

    int cshape[3]={B,csl,CD};
    mlx_array cstate=mlx_array_new();
    if (ok) {
        if (has_cache && v_cache && v_cache->ctx!=NULL && mlx_array_ndim(*v_cache)>0) ok = mlx_array_set(&cstate,*v_cache)==0;
        else ok = mlx_zeros(&cstate, cshape, 3, mlx_array_dtype(qkv), s)==0;   /* stesso dtype di qkv -> conv in bf16 come la reference */
        KCHK(ok, "conv state init");
    }
    mlx_array cin=mlx_array_new();
    if (ok) { mlx_array ca[2]={cstate,qkv}; mlx_vector_array cv=mlx_vector_array_new_data(ca,2); ok = mlx_concatenate_axis(&cin,cv,1,s)==0; mlx_vector_array_free(cv); }
    KCHK(ok, "conv input concat");
    mlx_array cout=mlx_array_new();
    if (ok) {
        int str[1]={1}, pl[1]={0}, ph[1]={0}, kd[1]={1}, idil[1]={1};
        ok = mlx_conv_general(&cout, cin, cw32, str,1, pl,1, ph,1, kd,1, idil,1, CD, /*flip=*/false, s)==0;
        KCHK(ok, "depthwise conv fused");
    }
    mlx_array csig=mlx_array_new(), csilu=mlx_array_new();
    if (ok) ok = mlx_sigmoid(&csig,cout,s)==0 && mlx_multiply(&csilu,cout,csig,s)==0;
    KCHK(ok, "silu conv");
    int secs[2]={P,2*P};
    mlx_vector_array spl=mlx_vector_array_new();
    if (ok) ok = mlx_split_sections(&spl, csilu, secs, 2, -1, s)==0;
    mlx_array qp2=mlx_array_new(),kp2=mlx_array_new(),vp2=mlx_array_new();
    if (ok) ok = mlx_vector_array_get(&qp2,spl,0)==0 && mlx_vector_array_get(&kp2,spl,1)==0 && mlx_vector_array_get(&vp2,spl,2)==0;
    mlx_vector_array_free(spl);
    KCHK(ok, "split conv q/k/v");

    int qsh[4]={B,S,H,D};
    mlx_array q4=mlx_array_new(),k4=mlx_array_new(),v4=mlx_array_new();
    if (ok) ok = mlx_reshape(&q4,qp2,qsh,4,s)==0 && mlx_reshape(&k4,kp2,qsh,4,s)==0 && mlx_reshape(&v4,vp2,qsh,4,s)==0;
    KCHK(ok, "reshape q/k/v");
    const double inv = pow((double)D, -0.5);
    mlx_array inv2f=mlx_array_new_float32((float)(inv*inv)), inv1f=mlx_array_new_float32((float)inv);
    mlx_array inv2=mlx_array_new(), inv1=mlx_array_new();
    if (ok) ok = mlx_astype(&inv2, inv2f, mlx_array_dtype(q4), s)==0 && mlx_astype(&inv1, inv1f, mlx_array_dtype(q4), s)==0;
    mlx_array qn=mlx_array_new(),kn=mlx_array_new(),qs2=mlx_array_new(),ks2=mlx_array_new();
    if (ok) ok = mlx_fast_rms_norm_no_scale(&qn,q4,D,1e-6f,s) && mlx_fast_rms_norm_no_scale(&kn,k4,D,1e-6f,s);
    if (ok) ok = mlx_multiply(&qs2,qn,inv2,s)==0 && mlx_multiply(&ks2,kn,inv1,s)==0;
    if (ok) ok = mlx_array_dtype(qs2)==mlx_array_dtype(v4);  /* kernel GVEC: stessi dtype */
    mlx_array_free(inv2f); mlx_array_free(inv1f);
    KCHK(ok, "rms_norm scaled q/k");

    mlx_array fah=mlx_array_new(),araw=mlx_array_new(),braw=mlx_array_new();
    KLIN(fah,h,faw,fas,fab); KLIN(araw,fah,fbw,fbs,fbb); KLIN(braw,h,bw,bs_,bb);
    KCHK(ok, "a/b proj");
    mlx_array a4=mlx_array_new();
    if (ok) ok = mlx_reshape(&a4, araw, qsh, 4, s)==0;
    mlx_array beta=mlx_array_new();
    if (ok) ok = mlx_sigmoid(&beta, braw, s)==0;
    KCHK(ok, "beta");
    mlx_array Alog=mlx_array_new(), dtb=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.A_log", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &Alog);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.dt_bias", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &dtb);
    KCHK(ok, "A_log/dt_bias");
    int alsh[2]={H,1}, dtsh[2]={H,D};
    mlx_array Alog2=mlx_array_new(), dtb2=mlx_array_new();
    if (ok) ok = mlx_reshape(&Alog2, Alog, alsh, 2, s)==0 && mlx_reshape(&dtb2, dtb, dtsh, 2, s)==0;
    mlx_array apd=mlx_array_new(), spexp=mlx_array_new(), sp1=mlx_array_new(), splog=mlx_array_new();
    mlx_array onef=mlx_array_new_float32(1.0f), one=mlx_array_new();
    if (ok) ok = mlx_astype(&one, onef, mlx_array_dtype(a4), s)==0;   /* softplus in bf16 come la reference */
    mlx_array_free(onef);
    if (ok) ok = mlx_add(&apd, a4, dtb2, s)==0;
    if (ok) ok = mlx_exp(&spexp, apd, s)==0 && mlx_add(&sp1, spexp, one, s)==0 && mlx_log(&splog, sp1, s)==0;
    mlx_array eAlog=mlx_array_new();
    if (ok) ok = mlx_exp(&eAlog, Alog2, s)==0;
    mlx_array neg=mlx_array_new(), g=mlx_array_new();
    if (ok) ok = mlx_multiply(&neg, eAlog, splog, s)==0;
    if (ok) ok = mlx_negative(&neg, neg, s)==0 && mlx_exp(&g, neg, s)==0;
    KCHK(ok, "g vectorized");

    /* gate = g_b_proj(g_a_proj(h)) — dipende SOLO da h: calcolato qui (prima della
     * ricorrenza) per non avere proiezioni quantizzate dopo il grande grafo del loop. */
    mlx_array gah=mlx_array_new(), graw=mlx_array_new(), gate4=mlx_array_new();
    KLIN(gah,h,gaw,gas,gab); KLIN(graw,gah,gbw,gbs,gbb);
    if (ok) ok = mlx_reshape(&gate4, graw, qsh, 4, s)==0;
    KCHK(ok, "gate proj");

    int ssh[4]={B,H,D,D};
    mlx_array state=mlx_array_new();
    if (ok) {
        if (has_cache && k_cache && k_cache->ctx!=NULL && mlx_array_ndim(*k_cache)>0) ok = mlx_array_set(&state,*k_cache)==0;
        else ok = mlx_zeros(&state, ssh, 4, MLX_FLOAT32, s)==0;
        KCHK(ok, "ssm state init");
    }
    mlx_vector_array ys=mlx_vector_array_new();
    /* Kernel Metal GDN fuso con g VETTORIALE (GVEC=1): stessa ricorrenza del loop ops,
     * ma ordine-op del kernel Metal come mlx_lm -> parita' bit. Fallback al loop ops se
     * ineleggibile (Dk!=128) o LLM_GDN_KERNEL_OFF. */
    mlx_array y_stack=mlx_array_new();
    bool kda_kernel = false;
    mlx_array g32 = mlx_array_new();
    if (ok) ok = mlx_astype(&g32, g, MLX_FLOAT32, s)==0;   /* il kernel staziona g in float */
    if (ok && !llm_env_isset("LLM_GDN_KERNEL_OFF")) {
        mlx_array state_nk = mlx_array_new();
        if (llm_mlx_gdn_prefill(&y_stack, &state_nk, qs2, ks2, v4, g32, beta, state, S, /*g_vector=*/1, s) == 0) {
            mlx_array_set(&state, state_nk);
            kda_kernel = true;
        }
        mlx_array_free(state_nk);
    }
    if (!kda_kernel) for (int t=0; ok && t<S; t++) {
        int t0[4]={0,t,0,0}, t1[4]={1,t+1,H,D}, on4[4]={1,1,1,1};
        mlx_array qt=mlx_array_new(),kt=mlx_array_new(),vt=mlx_array_new(),gt=mlx_array_new();
        if (ok) ok = mlx_slice(&qt,qs2,t0,4,t1,4,on4,4,s)==0 && mlx_slice(&kt,ks2,t0,4,t1,4,on4,4,s)==0 &&
                     mlx_slice(&vt,v4,t0,4,t1,4,on4,4,s)==0 && mlx_slice(&gt,g,t0,4,t1,4,on4,4,s)==0;
        int b0[3]={0,t,0}, b1[3]={1,t+1,H}, on3[3]={1,1,1};
        mlx_array bt=mlx_array_new();
        if (ok) ok = mlx_slice(&bt,beta,b0,3,b1,3,on3,3,s)==0;
        mlx_array qsq=mlx_array_new(),ksq=mlx_array_new(),vsq=mlx_array_new(),gsq=mlx_array_new(),bsq=mlx_array_new();
        if (ok) ok = mlx_squeeze_axis(&qsq,qt,1,s)==0 && mlx_squeeze_axis(&ksq,kt,1,s)==0 && mlx_squeeze_axis(&vsq,vt,1,s)==0 &&
                     mlx_squeeze_axis(&gsq,gt,1,s)==0 && mlx_squeeze_axis(&bsq,bt,1,s)==0;
        mlx_array yt=mlx_array_new();
        if (ok) ok = llm_mlx_gated_delta_step_ext(&yt,&state,qsq,ksq,vsq,gsq,bsq,/*g_vector=*/true,s);
        KCHK(ok, "gated_delta_step vec");
        if (ok) ok = mlx_vector_array_append_value(ys, yt)==0;
        KCHK(ok, "collect y");
        mlx_array_free(qt);mlx_array_free(kt);mlx_array_free(vt);mlx_array_free(gt);mlx_array_free(bt);
        mlx_array_free(qsq);mlx_array_free(ksq);mlx_array_free(vsq);mlx_array_free(gsq);mlx_array_free(bsq);
        mlx_array_free(yt);
    }
    if (!kda_kernel && ok) ok = mlx_stack_axis(&y_stack, ys, 1, s)==0;
    KCHK(ok, "stack y");
    if (ok) {
        mlx_array cnew=mlx_array_new();
        int c0[3]={0,S,0}, c1[3]={1,S+csl,CD}, cst[3]={1,1,1};
        ok = mlx_slice(&cnew, cin, c0,3,c1,3,cst,3,s)==0;
        if (ok) { mlx_array_free(cstate); cstate=cnew; } else mlx_array_free(cnew);
        /* Store degli stati: k_cache = ssm [B,H,D,D], v_cache = conv [B,kk-1,CD].
         * Indispensabile per il decode (senza, lo stato riparte da zero ogni token). */
        if (ok) ok = mlx_array_set(k_cache, state) == 0 && mlx_array_set(v_cache, cstate) == 0;
        KCHK(ok, "state store");
    }

    /* gate4 (g_b(g_a(h))) e' gia' stato calcolato PRIMA della ricorrenza. */
    mlx_array onorm=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_norm.weight", nb, sizeof nb); ok = ok && multishard_get(&m->weights, nb, &onorm);
    KCHK(ok, "o_norm weight");
    mlx_array yn=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&yn, y_stack, onorm, m->cfg.rms_norm_eps, s);
    mlx_array gsig=mlx_array_new(), gated=mlx_array_new();
    if (ok) ok = mlx_sigmoid(&gsig, gate4, s)==0 && mlx_multiply(&gated, yn, gsig, s)==0;
    KCHK(ok, "o_norm*sigmoid(gate)");
    int flat[3]={B,S,P};
    mlx_array gf=mlx_array_new();
    if (ok) ok = mlx_reshape(&gf, gated, flat, 3, s)==0;
    mlx_array ow=mlx_array_new(),os_=mlx_array_new(),ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb); ok = ok && multishard_get_triple(&m->weights, nb, &ow,&os_,&ob);
    mlx_array r=mlx_array_new();
    KLIN(r, gf, ow,os_,ob);
    if (ok) ok = mlx_add(out_h, x, r, s)==0;
    KCHK(ok, "o_proj+residual");

    mlx_array_free(ln);mlx_array_free(h);
    mlx_array_free(qw);mlx_array_free(qs);mlx_array_free(qb);
    mlx_array_free(kw);mlx_array_free(ks);mlx_array_free(kb);
    mlx_array_free(vw);mlx_array_free(vs);mlx_array_free(vb);
    mlx_array_free(faw);mlx_array_free(fas);mlx_array_free(fab);
    mlx_array_free(fbw);mlx_array_free(fbs);mlx_array_free(fbb);
    mlx_array_free(bw);mlx_array_free(bs_);mlx_array_free(bb);
    mlx_array_free(gaw);mlx_array_free(gas);mlx_array_free(gab);
    mlx_array_free(gbw);mlx_array_free(gbs);mlx_array_free(gbb);
    mlx_array_free(qp);mlx_array_free(kp);mlx_array_free(vp);
    mlx_array_free(cqw);mlx_array_free(ckw);mlx_array_free(cvw);mlx_array_free(cw32);
    mlx_array_free(qkv);mlx_array_free(cstate);mlx_array_free(cin);mlx_array_free(cout);mlx_array_free(csig);mlx_array_free(csilu);
    mlx_array_free(qp2);mlx_array_free(kp2);mlx_array_free(vp2);
    mlx_array_free(q4);mlx_array_free(k4);mlx_array_free(v4);
    mlx_array_free(inv2);mlx_array_free(inv1);mlx_array_free(qn);mlx_array_free(kn);mlx_array_free(qs2);mlx_array_free(ks2);
    mlx_array_free(fah);mlx_array_free(araw);mlx_array_free(braw);mlx_array_free(a4);mlx_array_free(beta);
    mlx_array_free(Alog);mlx_array_free(dtb);mlx_array_free(Alog2);mlx_array_free(dtb2);
    mlx_array_free(apd);mlx_array_free(spexp);mlx_array_free(sp1);mlx_array_free(splog);mlx_array_free(one);
    mlx_array_free(eAlog);mlx_array_free(neg);mlx_array_free(g); mlx_array_free(g32);
    mlx_array_free(state);mlx_vector_array_free(ys);mlx_array_free(y_stack);
    mlx_array_free(gah);mlx_array_free(graw);mlx_array_free(gate4);mlx_array_free(onorm);mlx_array_free(yn);
    mlx_array_free(gsig);mlx_array_free(gated);mlx_array_free(gf);mlx_array_free(ow);mlx_array_free(os_);mlx_array_free(ob);mlx_array_free(r);
    return ok;
#undef KCHK
}

bool kimi_linear_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                           mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool is_kda = (m->cfg.is_linear_layer != NULL && layer_idx >= 0 && layer_idx < m->cfg.n_layers &&
                   m->cfg.is_linear_layer[layer_idx]);
    if (is_kda) {
        return kimi_delta_attention_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
    }
    return kimi_mla_attention_block(m, layer_idx, cache_offset, n_tokens, x, k_cache, v_cache, has_cache, out_h);
}

bool kimi_linear_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    /* Kimi Linear MLP: primi `first_k_dense_replace` layer densi (SwiGLU: mlp.{gate,up,down}_proj),
     * poi MoE con lo STESSO layout/nomi di GLM-4.5 (router mlp.gate + e_score_correction_bias a
     * livello mlp., switch_mlp, shared_experts): riusa glm4_moe_ffn_block (sigmoid noaux_tc).
     * Kimi usa solo il router sigmoid (moe_router_activation_func=sigmoid). */
    bool use_moe = (m->cfg.n_experts > 0 && layer_idx >= m->cfg.first_k_dense_replace &&
                    (m->cfg.moe_layer_freq <= 1 || (layer_idx % m->cfg.moe_layer_freq) == 0));
    if (!use_moe) return dense_swiglu_ffn(m, layer_idx, n_tokens, h, out);
    if (strncmp(m->cfg.moe_router_activation, "sigmoid", 7) != 0) {
        fprintf(stderr, "kimi_linear[layer=%d]: router '%s' non supportato (solo sigmoid). "
                        "Vedi Todo/(open) - MLX-Engine-Lavoro-Residuo.md §C.\n",
                layer_idx, m->cfg.moe_router_activation);
        return false;
    }
    return glm4_moe_ffn_block(m, layer_idx, n_tokens, h, out);
}
