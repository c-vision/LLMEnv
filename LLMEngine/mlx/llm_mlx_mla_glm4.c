/* ============================================================================
 * Dominio MLA DeepSeek-V2 + GLM-4.5 attenzione: llm_mla_inplace_enabled, deepseek_v2_attention_block, glm4_moe_attention_block.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_lora.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* DeepSeek-V2 (MLA) attention: riferimento mlx_lm/models/deepseek_v2.py.
 * q_proj -> [H, qk_nope+qk_rope]; kv_a_proj_with_mqa -> [kv_lora_rank+qk_rope]; kv_a_layernorm;
 * kv_b_proj -> [H, qk_nope+v_head]; RoPE (yarn, traditional=true) SOLO sui qk_rope_head_dim; cache
 * keys [H, qk_nope+qk_rope], values [H, v_head]; scale 1/sqrt(qk_nope+qk_rope); v_dim != k_dim. */
/* MLA decode con cache in-place preallocata (2026-09-13): elimina la copia O(pos) del
 * concat per token. Usa `mlx_slice_update_inplace` (estensione mlx-c, 2026-09-13) che
 * replica `keys[...]=x` di mlx_lm: overwrite_descriptor(slice_update(...)) nel core ->
 * la base resta di proprieta' della sola primitive, append O(1). Vedi
 * Knowledge/LLMEnv-vs-mlxlm-Comparison.md §3.3. Default ON; LLM_MLA_INPLACE=0 per l'A/B. */
#define MLA_KVSTEP 256
static int g_mla_inplace_state = -1;
static bool llm_mla_inplace_enabled(void) {
    if (g_mla_inplace_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_MLA_INPLACE");
        g_mla_inplace_state = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1;
    }
    return g_mla_inplace_state == 1;
}
bool deepseek_v2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens,
                                        mlx_array x, mlx_array* k_cache, mlx_array* v_cache,
                                        bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int H = m->cfg.n_heads;
    if (layer_idx == 1 && n_tokens > 1) {
        mlx_array f=mlx_array_new();
        mlx_array_free(f);
    }
    const int qn = m->cfg.qk_nope_head_dim, qr = m->cfg.qk_rope_head_dim;
    const int vhd = m->cfg.v_head_dim, kvl = m->cfg.kv_lora_rank;
    const int qh = qn + qr;
    const char* mode = m->cfg.quant_mode;
    char nb[256];
    mlx_array ln=mlx_array_new(), qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    mlx_array aw=mlx_array_new(), as=mlx_array_new(), ab=mlx_array_new(), aln=mlx_array_new();
    mlx_array bw=mlx_array_new(), bs=mlx_array_new(), bb=mlx_array_new();
    mlx_array ow=mlx_array_new(), os=mlx_array_new(), ob=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &ln);
    /* MLA: con q_lora_rank>0 (DeepSeek-V3) il Q passa da q_a_proj -> q_a_layernorm ->
     * q_b_proj; con q_lora_rank==0 (V2-Lite) da un singolo q_proj. */
    mlx_array qaw=mlx_array_new(), qas=mlx_array_new(), qab=mlx_array_new(), qaln=mlx_array_new();
    mlx_array qbw=mlx_array_new(), qbs=mlx_array_new(), qbb=mlx_array_new();
    if (m->cfg.q_lora_rank > 0) {
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_a_proj", nb, sizeof nb);
        ok = ok && multishard_get_triple(&m->weights, nb, &qaw, &qas, &qab);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_a_layernorm.weight", nb, sizeof nb);
        ok = ok && multishard_get(&m->weights, nb, &qaln);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_b_proj", nb, sizeof nb);
        ok = ok && multishard_get_triple(&m->weights, nb, &qbw, &qbs, &qbb);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", nb, sizeof nb);
        ok = ok && multishard_get_triple(&m->weights, nb, &qw, &qs, &qb);
    }
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_a_proj_with_mqa", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &aw, &as, &ab);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_a_layernorm.weight", nb, sizeof nb);
    ok = ok && multishard_get(&m->weights, nb, &aln);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.kv_b_proj", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &bw, &bs, &bb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", nb, sizeof nb);
    ok = ok && multishard_get_triple(&m->weights, nb, &ow, &os, &ob);

    mlx_array normed = mlx_array_new();
    if (ok) ok = rms_norm_cfg(&normed, x, ln, m->cfg.rms_norm_eps, m->stream);

    /* q -> [1,L,H,qh] -> [1,H,L,qh] -> q_nope (:qn) / q_pe (qn:) */
    mlx_array q=mlx_array_new(), q4=mlx_array_new(), qt=mlx_array_new();
    mlx_array q_nope=mlx_array_new(), q_pe=mlx_array_new();
    if (ok && m->cfg.q_lora_rank > 0) {
        mlx_array qa=mlx_array_new(), qan=mlx_array_new();
        if (ok) ok = quantized_linear_cfg(&qa, normed, qaw, qas, qab, gsz, bts, mode, m->stream);
        if (ok) ok = rms_norm_cfg(&qan, qa, qaln, m->cfg.rms_norm_eps, m->stream);
        if (ok) ok = quantized_linear_cfg(&q, qan, qbw, qbs, qbb, gsz, bts, mode, m->stream);
        mlx_array_free(qa); mlx_array_free(qan);
    } else if (ok) {
        ok = quantized_linear_cfg(&q, normed, qw, qs, qb, gsz, bts, mode, m->stream);
    }
    /* §D R6 (MLA): LoRA su q_proj. */
    { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { int sh[4]={1,n_tokens,H,qh}; ok = mlx_reshape(&q4, q, sh, 4, m->stream)==0; }
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&qt, q4, perm, 4, m->stream)==0; }
    if (ok) { int _sec[1]={qn}; mlx_vector_array _sp=mlx_vector_array_new();
              ok = mlx_split_sections(&_sp, qt, _sec, 1, -1, m->stream)==0 &&
                   mlx_vector_array_get(&q_nope, _sp, 0)==0 && mlx_vector_array_get(&q_pe, _sp, 1)==0;
              mlx_vector_array_free(_sp); }

    /* compressed_kv -> compressed [1,L,kvl] / k_pe [1,1,L,qr] */
    mlx_array ckv=mlx_array_new(), ckv3=mlx_array_new(), compressed=mlx_array_new();
    mlx_array kpe3=mlx_array_new(), kpe=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&ckv, normed, aw, as, ab, gsz, bts, mode, m->stream);
    if (ok) { int sh[3]={1,n_tokens,kvl+qr}; ok = mlx_reshape(&ckv3, ckv, sh, 3, m->stream)==0; }
    if (ok) { int _sec[1]={kvl}; mlx_vector_array _sp=mlx_vector_array_new();
              ok = mlx_split_sections(&_sp, ckv3, _sec, 1, -1, m->stream)==0 &&
                   mlx_vector_array_get(&compressed, _sp, 0)==0 && mlx_vector_array_get(&kpe3, _sp, 1)==0;
              mlx_vector_array_free(_sp); }
    if (ok) { int sh[4]={1,1,n_tokens,qr}; ok = mlx_reshape(&kpe, kpe3, sh, 4, m->stream)==0; }

    /* kv -> [1,H,L,qn+vhd] -> k_nope (:qn) / values (qn:) */
    mlx_array ckn=mlx_array_new(), kv=mlx_array_new(), kv4=mlx_array_new(), kvt=mlx_array_new();
    mlx_array k_nope=mlx_array_new(), values=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&ckn, compressed, aln, m->cfg.rms_norm_eps, m->stream);
    if (ok) ok = quantized_linear_cfg(&kv, ckn, bw, bs, bb, gsz, bts, mode, m->stream);
    /* §D R6 (MLA): LoRA su kv_b_proj. */
    { int li = llm_mlx_model_lora_get(m, layer_idx, "kv_b_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&kv, ckn, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) { int sh[4]={1,n_tokens,H,qn+vhd}; ok = mlx_reshape(&kv4, kv, sh, 4, m->stream)==0; }
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&kvt, kv4, perm, 4, m->stream)==0; }
    if (ok) { int _sec[1]={qn}; mlx_vector_array _sp=mlx_vector_array_new();
              ok = mlx_split_sections(&_sp, kvt, _sec, 1, -1, m->stream)==0 &&
                   mlx_vector_array_get(&k_nope, _sp, 0)==0 && mlx_vector_array_get(&values, _sp, 1)==0;
              mlx_vector_array_free(_sp); }

    /* RoPE yarn, traditional=true, solo i qk_rope_head_dim; mscale = ratio (V2-Lite 0.707/0.707=1) */
    mlx_array q_pe_r=mlx_array_new(), kpe_r=mlx_array_new();
    if (ok) {
        mlx_array freqs = llm_mlx_yarn_rope_freqs_cached(qr, m->cfg.rope_theta, m->cfg.yarn_factor,
                                                          m->cfg.yarn_orig_max, m->cfg.yarn_beta_fast, m->cfg.yarn_beta_slow);
        mlx_optional_float no_base = {0.0f, false};
        ok = freqs.ctx != NULL &&
             mlx_fast_rope(&q_pe_r, q_pe, qr, true, no_base, 1.0f, cache_offset, freqs, m->stream)==0 &&
             mlx_fast_rope(&kpe_r, kpe, qr, true, no_base, 1.0f, cache_offset, freqs, m->stream)==0;
        /* NB: freqs appartiene a *_cached (array STATICO) -> NON liberarlo qui. */
        float num = llm_mlx_yarn_mscale(m->cfg.yarn_factor, m->cfg.yarn_mscale);
        float den = llm_mlx_yarn_mscale(m->cfg.yarn_factor, m->cfg.yarn_mscale_all_dim);
        float msc = (den > 1e-9f) ? num/den : 1.0f;
        if (ok && msc != 1.0f) {
            mlx_array s = mlx_array_new_float32(msc), q2=mlx_array_new(), k2=mlx_array_new();
            ok = mlx_multiply(&q2, q_pe_r, s, m->stream)==0 && mlx_multiply(&k2, kpe_r, s, m->stream)==0;
            if (ok) { mlx_array_set(&q_pe_r, q2); mlx_array_set(&kpe_r, k2); }
            mlx_array_free(s); mlx_array_free(q2); mlx_array_free(k2);
        }
    }
    mlx_array kpe_h=mlx_array_new(), keys=mlx_array_new(), queries=mlx_array_new();
    if (ok) { int sh[4]={1,H,n_tokens,qr}; ok = mlx_broadcast_to(&kpe_h, kpe_r, sh, 4, m->stream)==0; }
    if (ok) { mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,k_nope); mlx_vector_array_append_value(v,kpe_h);
              ok = mlx_concatenate_axis(&keys, v, 3, m->stream)==0; mlx_vector_array_free(v); }
    if (ok) { mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,q_nope); mlx_vector_array_append_value(v,q_pe_r);
              ok = mlx_concatenate_axis(&queries, v, 3, m->stream)==0; mlx_vector_array_free(v); }

    { static double _dse=0; static int _dsn2=0; if (llm_env_isset("LLM_DS_TIME")) { struct timespec _a,_b; clock_gettime(CLOCK_MONOTONIC,&_a);
        mlx_array_eval(keys); mlx_array_eval(values); clock_gettime(CLOCK_MONOTONIC,&_b);
        _dse += (_b.tv_sec-_a.tv_sec)+(_b.tv_nsec-_a.tv_nsec)*1e-9; _dsn2++;
        if(_dsn2%27==0) fprintf(stderr,"[DS_EXP] n=%d exp=%.3fs\n",_dsn2,_dse); } }
    mlx_array k_full=mlx_array_new(), v_full=mlx_array_new();
    mlx_array k_attn_src=mlx_array_new(), v_attn_src=mlx_array_new();
    if (ok && has_cache && n_tokens == 1 && llm_mla_inplace_enabled()) {
        /* Decode MLA: cache in-place preallocata. off = posizione del token (lunghezza
         * corrente); se la cache e' "esatta" (dim2==off) la si migra a un buffer con
         * capacita' arrotondata; altrimenti si scrive in-place con slice_update+rebind
         * (MLX dona il buffer: niente copia O(pos)) e si legge la vista troncata. */
        const int off = cache_offset;
        const int cap = (int)mlx_array_shape(*k_cache)[2];
        const int kdim = (int)mlx_array_shape(*k_cache)[3];
        const int vdim = (int)mlx_array_shape(*v_cache)[3];
        const int sv4[4] = {1,1,1,1};
        if (cap > off) {
            if (!llm_env_isset("LLM_MLA_NOWRITE")) {
            int st4[4]={0,0,off,0}, sk[4]={1,H,off+1,kdim}, svv[4]={1,H,off+1,vdim};
            ok = mlx_slice_update_inplace(k_cache, keys, st4,4, sk,4, sv4,4, m->stream)==0 &&
                 mlx_slice_update_inplace(v_cache, values, st4,4, svv,4, sv4,4, m->stream)==0;
            }
        } else {
            const int newcap = ((off + 1 + MLA_KVSTEP - 1) / MLA_KVSTEP) * MLA_KVSTEP;
            int zk[4]={1,H,newcap,kdim}, zv[4]={1,H,newcap,vdim};
            mlx_array zK=mlx_array_new(), zV=mlx_array_new();
            ok = mlx_zeros(&zK, zk,4, mlx_array_dtype(*k_cache), m->stream)==0 &&
                 mlx_zeros(&zV, zv,4, mlx_array_dtype(*v_cache), m->stream)==0;
            mlx_array K1=mlx_array_new(), V1=mlx_array_new();
            if (ok) { int s0[4]={0,0,0,0}, sk[4]={1,H,off,kdim}, sv[4]={1,1,1,1};
                      ok = mlx_slice_update(&K1, zK, *k_cache, s0,4, sk,4, sv,4, m->stream)==0; }
            if (ok) { int s0[4]={0,0,0,0}, sv[4]={1,H,off,vdim}, ss[4]={1,1,1,1};
                      ok = mlx_slice_update(&V1, zV, *v_cache, s0,4, sv,4, ss,4, m->stream)==0; }
            mlx_array K2=mlx_array_new(), V2=mlx_array_new();
            if (ok) { int s0[4]={0,0,off,0}, sk[4]={1,H,off+1,kdim}, sv[4]={1,1,1,1};
                      ok = mlx_slice_update(&K2, K1, keys, s0,4, sk,4, sv,4, m->stream)==0; }
            if (ok) { int s0[4]={0,0,off,0}, sv[4]={1,H,off+1,vdim}, ss[4]={1,1,1,1};
                      ok = mlx_slice_update(&V2, V1, values, s0,4, sv,4, ss,4, m->stream)==0; }
            if (ok) { mlx_array_set(k_cache, K2); mlx_array_set(v_cache, V2); }
            mlx_array_free(zK); mlx_array_free(zV); mlx_array_free(K1); mlx_array_free(V1);
            mlx_array_free(K2); mlx_array_free(V2);
        }
        if (llm_env_isset("LLM_MLA_NOSLICE")) {
            if (ok) ok = mlx_array_set(&k_attn_src, *k_cache)==0 && mlx_array_set(&v_attn_src, *v_cache)==0;
        } else {
        if (ok) { int p0[4]={0,0,0,0}, p1[4]={1,H,off+1,kdim};
                  ok = mlx_slice(&k_attn_src, *k_cache, p0,4, p1,4, sv4,4, m->stream)==0; }
        if (ok) { int p0[4]={0,0,0,0}, p1[4]={1,H,off+1,vdim};
                  ok = mlx_slice(&v_attn_src, *v_cache, p0,4, p1,4, sv4,4, m->stream)==0; }
        }
    } else if (ok) {
        if (has_cache) {
            mlx_vector_array v=mlx_vector_array_new(); mlx_vector_array_append_value(v,*k_cache); mlx_vector_array_append_value(v,keys);
            ok = mlx_concatenate_axis(&k_full, v, 2, m->stream)==0; mlx_vector_array_free(v);
            if (ok) { mlx_vector_array w=mlx_vector_array_new(); mlx_vector_array_append_value(w,*v_cache); mlx_vector_array_append_value(w,values);
                      ok = mlx_concatenate_axis(&v_full, w, 2, m->stream)==0; mlx_vector_array_free(w); }
        } else { ok = mlx_array_set(&k_full, keys)==0 && mlx_array_set(&v_full, values)==0; }
        if (ok) { ok = mlx_array_set(k_cache, k_full)==0 && mlx_array_set(v_cache, v_full)==0; }
        if (ok) { ok = mlx_array_set(&k_attn_src, *k_cache)==0 && mlx_array_set(&v_attn_src, *v_cache)==0; }
    }

    mlx_array attn=mlx_array_new();
    /* LLM_DS_TIME: diagnostica temporanea (con eval) append-cache vs SDPA. */
    static double _ds_ap = 0.0, _ds_sd = 0.0; static int _ds_n = 0;
    const bool _ds_tm = llm_env_isset("LLM_DS_TIME");
    struct timespec _ds_t0, _ds_t1;
    if (_ds_tm && ok) {
        clock_gettime(CLOCK_MONOTONIC, &_ds_t0);
        mlx_array_eval(k_attn_src); mlx_array_eval(v_attn_src);
        clock_gettime(CLOCK_MONOTONIC, &_ds_t1);
        _ds_ap += (_ds_t1.tv_sec-_ds_t0.tv_sec) + (_ds_t1.tv_nsec-_ds_t0.tv_nsec)*1e-9;
    }
    /* mlx_lm deepseek_v2.py: `if mscale_all_dim: mscale=yarn_get_mscale(factor,mscale_all_dim);
     * self.scale = self.scale * mscale * mscale` -> la scala attenzione include mscale^2. */
    float amsc = llm_mlx_yarn_mscale(m->cfg.yarn_factor, m->cfg.yarn_mscale_all_dim);
    float attn_scale = (1.0f/sqrtf((float)qh)) * amsc * amsc;
    if (ok) ok = llm_mlx_gqa_sdpa(&attn, queries, k_attn_src, v_attn_src, attn_scale, 0, m->stream);
    if (_ds_tm && ok) {
        clock_gettime(CLOCK_MONOTONIC, &_ds_t0);
        mlx_array_eval(attn);
        clock_gettime(CLOCK_MONOTONIC, &_ds_t1);
        _ds_sd += (_ds_t1.tv_sec-_ds_t0.tv_sec) + (_ds_t1.tv_nsec-_ds_t0.tv_nsec)*1e-9;
        _ds_n++;
        if (_ds_n % 27 == 0)
            fprintf(stderr, "[DS_TIME] layers=%d append=%.3fs sdpa=%.3fs\n", _ds_n, _ds_ap, _ds_sd);
    }

    mlx_array attn_bt=mlx_array_new(), attn_flat=mlx_array_new(), proj=mlx_array_new();
    if (ok) { int perm[4]={0,2,1,3}; ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, m->stream)==0; }
    if (ok) { int sh[3]={1,n_tokens,H*vhd}; ok = mlx_reshape(&attn_flat, attn_bt, sh, 3, m->stream)==0; }
    if (ok) ok = quantized_linear_cfg(&proj, attn_flat, ow, os, ob, gsz, bts, mode, m->stream);
    /* §D R6 (MLA): LoRA su o_proj. */
    { int li = llm_mlx_model_lora_get(m, layer_idx, "o_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&proj, attn_flat, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok) ok = mlx_add(out_h, x, proj, m->stream)==0;
    if (ok && layer_idx == 0 && n_tokens > 1) {
        mlx_array f=mlx_array_new();
        mlx_array_free(f);
    }

    mlx_array_free(ln); mlx_array_free(qw); mlx_array_free(qs); mlx_array_free(qb);
    mlx_array_free(qaw); mlx_array_free(qas); mlx_array_free(qab); mlx_array_free(qaln);
    mlx_array_free(qbw); mlx_array_free(qbs); mlx_array_free(qbb);
    mlx_array_free(aw); mlx_array_free(as); mlx_array_free(ab); mlx_array_free(aln);
    mlx_array_free(bw); mlx_array_free(bs); mlx_array_free(bb);
    mlx_array_free(ow); mlx_array_free(os); mlx_array_free(ob);
    mlx_array_free(normed);
    mlx_array_free(q); mlx_array_free(q4); mlx_array_free(qt); mlx_array_free(q_nope); mlx_array_free(q_pe);
    mlx_array_free(ckv); mlx_array_free(ckv3); mlx_array_free(compressed); mlx_array_free(kpe3); mlx_array_free(kpe);
    mlx_array_free(ckn); mlx_array_free(kv); mlx_array_free(kv4); mlx_array_free(kvt); mlx_array_free(k_nope); mlx_array_free(values);
    mlx_array_free(q_pe_r); mlx_array_free(kpe_r); mlx_array_free(kpe_h); mlx_array_free(keys); mlx_array_free(queries);
    mlx_array_free(k_full); mlx_array_free(v_full); mlx_array_free(attn);
    mlx_array_free(k_attn_src); mlx_array_free(v_attn_src);
    mlx_array_free(attn_bt); mlx_array_free(attn_flat); mlx_array_free(proj);
    return ok;
}

/* GLM-4.5-Air (glm4_moe) attention: GQA standard con bias lineari su q/k/v (o_proj
 * senza bias), RoPE PARZIALE (partial_rotary_factor 0.5: ruota meta' head_dim), scale
 * 1/sqrt(head_dim), niente sinks. Reference: mlx_lm/models/glm4_moe.py Attention. */
bool glm4_moe_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                     mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "glm4_moe_attention[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array ln_w = mlx_array_new();
    mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
    mlx_array kw = mlx_array_new(), ks = mlx_array_new(), kb = mlx_array_new();
    mlx_array vw = mlx_array_new(), vs = mlx_array_new(), vb = mlx_array_new();
    GattnQuant* ac = llm_mlx_attn_quant(m, layer_idx);
    if (ac != NULL) {
        ok = ok && mlx_array_set(&ln_w, ac->ln1) == 0;
        ok = ok && mlx_array_set(&qw, ac->qw) == 0 && mlx_array_set_optional(&qs, ac->qs) && mlx_array_set_optional(&qb, ac->qb);
        ok = ok && mlx_array_set(&kw, ac->kw) == 0 && mlx_array_set_optional(&ks, ac->ks) && mlx_array_set_optional(&kb, ac->kb);
        ok = ok && mlx_array_set(&vw, ac->vw) == 0 && mlx_array_set_optional(&vs, ac->vs) && mlx_array_set_optional(&vb, ac->vb);
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &ln_w);
        if (!ok) CHK(false, "input_layernorm lookup");
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &qw, &qs, &qb);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &kw, &ks, &kb);
        tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &vw, &vs, &vb);
    }
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, x, ln_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "input_layernorm"); }
    /* Q2-b: energia dell'input condiviso di q/k/v (riga attention, expert=-1). Solo in
     * calibrazione (m->im.active): in inferenza costo zero (nessuna lettura host). */
    if (m->im.active && m->im.attn_qkv != NULL && layer_idx >= 0 && layer_idx < m->im.n_layers) {
        int ndq = mlx_array_ndim(normed), lcq = ndq >= 1 ? (int)mlx_array_dim(normed, ndq - 1) : 0;
        int nrq = lcq > 0 ? (int)(mlx_array_size(normed) / (size_t)lcq) : 0;  /* righe: firme diverse fra blocchi */
        if (lcq == m->im.gate_cols && nrq > 0) {
            llm_mlx_imatrix_energy_add(m, m->im.attn_qkv + (size_t)layer_idx * m->im.gate_cols,
                                       m->im.gate_cols, normed, nrq);
            m->im.attn_cnt[layer_idx]++;
        }
    }

    /* Bias lineari reali (q/k/v_proj.bias) */
    mlx_array q_lb = mlx_array_new(), k_lb = mlx_array_new(), v_lb = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.q_proj.bias", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &q_lb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.k_proj.bias", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &k_lb);
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.v_proj.bias", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &v_lb);
    CHK(ok, "q/k/v_proj.bias lookup");

    mlx_array q0 = mlx_array_new(), k0 = mlx_array_new(), v0 = mlx_array_new();
    /* LLMENGINE_GLM_QKV_FUSE: q/k/v in UNA matmul (concat asse 0) + 3 slice. Non applicabile
     * se c'e' LoRA su q/k/v_proj (il path separato la applica per-proiezione). */
    const bool qkv_fused = (ac != NULL && ac->has_qkv_fused) &&
                           llm_mlx_model_lora_get(m, layer_idx, "q_proj") < 0 &&
                           llm_mlx_model_lora_get(m, layer_idx, "k_proj") < 0 &&
                           llm_mlx_model_lora_get(m, layer_idx, "v_proj") < 0;
    if (ok && qkv_fused) {
        const int NQ = m->cfg.n_heads * m->cfg.head_dim;
        const int NKV = m->cfg.n_kv_heads * m->cfg.head_dim;
        mlx_array qkv = mlx_array_new();
        ok = quantized_linear_cfg(&qkv, normed, ac->qkvw, ac->qkvs, ac->qkvb, gsz, bts, m->cfg.quant_mode, m->stream);
        CHK(ok, "qkv_proj matmul fused");
        if (ok) {
            const int sv[3] = {1, 1, 1};
            const int sa[3] = {0, 0, 0}, sb[3] = {0, 0, NQ}, sc2[3] = {0, 0, NQ + NKV};
            const int ea[3] = {1, n_tokens, NQ}, eb[3] = {1, n_tokens, NQ + NKV}, ec[3] = {1, n_tokens, NQ + 2 * NKV};
            ok = mlx_slice(&q0, qkv, sa, 3, ea, 3, sv, 3, m->stream) == 0 &&
                 mlx_slice(&k0, qkv, sb, 3, eb, 3, sv, 3, m->stream) == 0 &&
                 mlx_slice(&v0, qkv, sc2, 3, ec, 3, sv, 3, m->stream) == 0;
            CHK(ok, "qkv_proj slice");
        }
        mlx_array_free(qkv);
    } else {
        if (ok) { ok = quantized_linear_cfg(&q0, normed, qw, qs, qb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "q_proj matmul"); }
        if (ok) { ok = quantized_linear_cfg(&k0, normed, kw, ks, kb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "k_proj matmul"); }
        if (ok) { ok = quantized_linear_cfg(&v0, normed, vw, vs, vb, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "v_proj matmul"); }
        /* §D R6 (multi-arch): LoRA su q/k/v_proj. */
        { int li = llm_mlx_model_lora_get(m, layer_idx, "q_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&q0, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
        { int li = llm_mlx_model_lora_get(m, layer_idx, "k_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&k0, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
        { int li = llm_mlx_model_lora_get(m, layer_idx, "v_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&v0, normed, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    }
    mlx_array q = mlx_array_new(), k = mlx_array_new(), v = mlx_array_new();
    if (ok) {
        ok = mlx_add(&q, q0, q_lb, m->stream) == 0 &&
             mlx_add(&k, k0, k_lb, m->stream) == 0 &&
             mlx_add(&v, v0, v_lb, m->stream) == 0;
        CHK(ok, "q/k/v linear bias add");
    }

    mlx_array q_heads = mlx_array_new(), k_heads = mlx_array_new(), v_heads = mlx_array_new();
    if (ok) {
        const int q_shape[4] = {1, n_tokens, m->cfg.n_heads, m->cfg.head_dim};
        const int kv_shape[4] = {1, n_tokens, m->cfg.n_kv_heads, m->cfg.head_dim};
        ok = mlx_reshape(&q_heads, q, q_shape, 4, m->stream) == 0 &&
             mlx_reshape(&k_heads, k, kv_shape, 4, m->stream) == 0 &&
             mlx_reshape(&v_heads, v, kv_shape, 4, m->stream) == 0;
        CHK(ok, "head reshape");
    }
    mlx_array q_t = mlx_array_new(), k_t = mlx_array_new(), v_t = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&q_t, q_heads, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&k_t, k_heads, perm, 4, m->stream) == 0 &&
             mlx_transpose_axes(&v_t, v_heads, perm, 4, m->stream) == 0;
        CHK(ok, "qkv transpose");
    }
    /* RoPE PARZIALE: ruota solo i primi head_dim*partial_rotary_factor dims (0.5) */
    mlx_array q_rope = mlx_array_new(), k_rope = mlx_array_new();
    if (ok) {
        float prf = m->cfg.partial_rotary_factor;
        int rope_dims = (prf > 0.0f && prf < 1.0f) ? (int)(m->cfg.head_dim * prf) : m->cfg.head_dim;
        if (rope_dims <= 0) rope_dims = m->cfg.head_dim;
        mlx_optional_float base = {m->cfg.rope_theta, true};
        mlx_array no_freqs = mlx_array_new();
        ok = mlx_fast_rope(&q_rope, q_t, rope_dims, /*traditional=*/false, base, /*scale=*/1.0f,
                            cache_offset, no_freqs, m->stream) == 0 &&
             mlx_fast_rope(&k_rope, k_t, rope_dims, false, base, 1.0f, cache_offset, no_freqs, m->stream) == 0;
        CHK(ok, "rope (partial)");
        mlx_array_free(no_freqs);
    }

    mlx_array k_full = mlx_array_new(), v_full = mlx_array_new();
    mlx_array attn = mlx_array_new();
    /* LLM_KVSLICE esteso a GLM-4.5-Air (2026-09-07 notte): stesso pattern verificato e
     * misurato su Llama in llama_attention_block (buffer persistente a capacita' fissa +
     * slice_update a owner unico per la donazione MLX + SDPA diretta sulla slice strided,
     * niente mlx_contiguous) -- vedi i commenti nel ramo gemello per il dettaglio completo
     * dei due bug pre-esistenti trovati e fissati, e la scoperta che mlx_contiguous non
     * serve affatto. GLM eligibile alle stesse condizioni di Llama (head_dim=128 %32==0,
     * n_heads=96, n_kv_heads=8, 96%8==0).
     *
     * VERDETTO MISURATO (2026-09-07, macchina scarica, TODO KvPagePool §1): a differenza di
     * Llama (dove vince chiaramente a contesto lungo), su GLM questo path e' CONTROPRODUCENTE
     * e peggiora con N -- N=9000-15000 0.81-0.97x, N=30000 0.45x (2.2x PIU' LENTO). Ipotesi:
     * il GQA ratio molto piu' alto di GLM (96 query heads / 8 kv heads = 12x) rende la SDPA
     * strided su un buffer fisico grande quanto `cap` (non `needed`) molto piu' costosa quando
     * ci sono 12 query head da rileggere sullo stesso buffer per ogni head KV. Corretto
     * (output byte-identico) ma NON promuovere/raccomandare per GLM -- resta dietro
     * LLM_KVSLICE=1 (default OFF) solo per compatibilita' col path Llama, non perche' aiuti
     * qui. */
    bool kvsl = llm_mlx_kvslice_enabled() && n_tokens == 1 && has_cache && m->cfg.kv_max_tokens <= 0;
    mlx_array k_src = mlx_array_new(), v_src = mlx_array_new();
    bool kv_inplace = false;
    if (ok && kvsl) {
        const int nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
        int have_len = 0;
        if ((*k_cache).ctx != NULL && mlx_array_ndim(*k_cache) == 4) { const int* csh = mlx_array_shape(*k_cache); if (csh) have_len = csh[2]; }
        int needed = cache_offset + n_tokens;
        int cap = have_len >= needed ? have_len : llm_mlx_kv_cap_for(needed);
        mlx_array k_base = mlx_array_new(), v_base = mlx_array_new();
        if (have_len > 0 && have_len < cap) {
            int zsh[4] = {1, nkv, cap - have_len, hd};
            mlx_array kz = mlx_array_new(), vz = mlx_array_new();
            bool zok = mlx_zeros(&kz, zsh, 4, mlx_array_dtype(*k_cache), m->stream) == 0 &&
                       mlx_zeros(&vz, zsh, 4, mlx_array_dtype(*v_cache), m->stream) == 0;
            if (zok) {
                mlx_vector_array kv = mlx_vector_array_new();
                mlx_vector_array_append_value(kv, *k_cache); mlx_vector_array_append_value(kv, kz);
                zok = mlx_concatenate_axis(&k_base, kv, 2, m->stream) == 0; mlx_vector_array_free(kv);
                mlx_vector_array vv = mlx_vector_array_new();
                mlx_vector_array_append_value(vv, *v_cache); mlx_vector_array_append_value(vv, vz);
                zok = zok && mlx_concatenate_axis(&v_base, vv, 2, m->stream) == 0; mlx_vector_array_free(vv);
            }
            mlx_array_free(kz); mlx_array_free(vz);
            if (!zok) { ok = false; }
        } else if (have_len == cap && cap > 0) {
            ok = ok && mlx_array_set(&k_base, *k_cache) == 0 && mlx_array_set(&v_base, *v_cache) == 0;
            if (ok) {
                mlx_array_free(*k_cache); mlx_array_free(*v_cache);
                *k_cache = mlx_array_new(); *v_cache = mlx_array_new();
            }
        } else if (cap > 0) {
            int zsh[4] = {1, nkv, cap, hd};
            mlx_dtype dt = (llm_mlx_act_dt(m->cfg.bfloat16_input));
            ok = ok && mlx_zeros(&k_base, zsh, 4, dt, m->stream) == 0 &&
                       mlx_zeros(&v_base, zsh, 4, dt, m->stream) == 0;
        }
        mlx_array k_buf = mlx_array_new(), v_buf = mlx_array_new();
        if (ok && cap > 0 && cache_offset >= 0 && cache_offset + 1 <= cap) {
            int st4[4] = {0, 0, cache_offset, 0}, sp4[4] = {1, nkv, cache_offset + 1, hd}, sv4[4] = {1, 1, 1, 1};
            ok = mlx_slice_update_inplace(&k_base, k_rope, st4, 4, sp4, 4, sv4, 4, m->stream) == 0 &&
                 mlx_slice_update_inplace(&v_base, v_t, st4, 4, sp4, 4, sv4, 4, m->stream) == 0;
        } else { ok = false; }
        if (ok) { ok = mlx_array_set(k_cache, k_base) == 0 && mlx_array_set(v_cache, v_base) == 0; }
        mlx_array_free(k_base); mlx_array_free(v_base); mlx_array_free(k_buf); mlx_array_free(v_buf);
        if (ok) {
            mlx_array kp = mlx_array_new(), vp = mlx_array_new();
            int p0[4] = {0, 0, 0, 0}, p1[4] = {1, nkv, needed, hd}, ps[4] = {1, 1, 1, 1};
            ok = mlx_slice(&kp, *k_cache, p0, 4, p1, 4, ps, 4, m->stream) == 0 &&
                 mlx_slice(&vp, *v_cache, p0, 4, p1, 4, ps, 4, m->stream) == 0;
            if (ok) {
                ok = llm_mlx_gqa_sdpa(&attn, q_rope, kp, vp, 1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);
                CHK(ok, "scaled_dot_product_attention (kvslice)");
            }
            mlx_array_free(kp); mlx_array_free(vp);
        }
    } else {
    /* KVF16 solo con attivazione fp32: li' dimezza DAVVERO i byte (4->2). Con attivazioni
     * bf16/f16 la cache e' gia' a 2 byte/elemento: il cast non risparmia nulla e aggiunge
     * 4 AsType per layer (144/token su Qwen3-8B, 12% dei nodi; misurato LLM_DISPATCH
     * 2026-09-19) allontanando le numeriche dal riferimento mlx_lm (cache bf16). */
    if (ok && llm_mlx_kv_f16_enabled() && mlx_array_dtype(k_rope) == MLX_FLOAT32) {
        /* LLM_KVF16 (#61): cache KV in fp16 per dimezzare il Concatenate O(pos). k_rope/v_t
         * riassegnati a fp16: il concat/init sotto li usa cosi' com'e'. q->fp16 e ritorno
         * attn->act sono armonizzati in llm_mlx_gqa_sdpa_sinks. Default-OFF: mai in prod. */
        mlx_array kr16 = mlx_array_new(), vt16 = mlx_array_new();
        if (mlx_astype(&kr16, k_rope, MLX_FLOAT16, m->stream) == 0 &&
            mlx_astype(&vt16, v_t,   MLX_FLOAT16, m->stream) == 0) {
            /* FIX ownership (2026-09-13): `k_rope = kr16` sovrascriveva l'handle raw perdendo
             * l'originale FP32/bf16 (buffer mai liberato). mlx_array_set condivide e sostituisce
             * in-place, poi si libera il nuovo handle. Vedi Bugs/(open) - MLX-Core-Prefill-Buffer-Retention.md. */
            mlx_array_set(&k_rope, kr16); mlx_array_set(&v_t, vt16);
            mlx_array_free(kr16); mlx_array_free(vt16);
        }
        else { mlx_array_free(kr16); mlx_array_free(vt16); }
    }
    if (ok && has_cache && n_tokens == 1 && !m->cfg.nsa_sparse && llm_mlx_kv_inplace_enabled() &&
        !(m->cfg.is_sliding_layer != NULL && layer_idx >= 0 && layer_idx < m->cfg.n_layers &&
          m->cfg.is_sliding_layer[layer_idx]))
        kv_inplace = llm_mlx_kv_append_inplace(k_cache, v_cache, k_rope, v_t,
                                               cache_offset, n_tokens, /*step=*/256,
                                               &k_src, &v_src, m->stream);
    if (ok && !kv_inplace) {
    if (ok) {
        if (has_cache) {
            mlx_vector_array kv = mlx_vector_array_new();
            mlx_vector_array_append_value(kv, *k_cache);
            mlx_vector_array_append_value(kv, k_rope);
            ok = mlx_concatenate_axis(&k_full, kv, 2, m->stream) == 0;
            mlx_vector_array_free(kv);
            CHK(ok, "k-cache concat");
            if (ok) {
                mlx_vector_array vv = mlx_vector_array_new();
                mlx_vector_array_append_value(vv, *v_cache);
                mlx_vector_array_append_value(vv, v_t);
                ok = mlx_concatenate_axis(&v_full, vv, 2, m->stream) == 0;
                mlx_vector_array_free(vv);
                CHK(ok, "v-cache concat");
            }
        } else {
            ok = mlx_array_set(&k_full, k_rope) == 0 && mlx_array_set(&v_full, v_t) == 0;
            CHK(ok, "k/v cache init");
        }
    }
        if (ok) { ok = mlx_array_set(&k_src, k_full) == 0 && mlx_array_set(&v_src, v_full) == 0; }
    }
    if (ok && !kv_inplace) {
        ok = mlx_array_set(k_cache, k_full) == 0;
        CHK(ok, "k cache store");
        if (ok) { ok = mlx_array_set(v_cache, v_full) == 0; CHK(ok, "v cache store"); }
    }

    if (ok && n_tokens > 1) {
        /* LLMMETAL_KVMETAL esteso a GLM-4.5-Air (2026-09-07): stesso wiring di
         * llama_attention_block (unica famiglia coperta finora) -- popola il pool per
         * un forward BATCH (prefill/append), fallback silenzioso se una condizione non
         * e' soddisfatta. GLM ha head_dim=128 (%32==0), n_heads=96, n_kv_heads=8
         * (96%8==0): eligibile alle stesse condizioni di Llama. */
        bool kvmetal_populate = (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_verify_enabled())
            && g_kvmetal_ctx != NULL && !g_kvmetal_disabled
            && !llm_mlx_is_act16(mlx_array_dtype(k_rope))
            && m->cfg.kv_max_tokens > 0 && m->cfg.n_kv_heads > 0
            && m->cfg.head_dim % 32 == 0 && m->cfg.head_dim <= 256
            && m->cfg.n_heads % m->cfg.n_kv_heads == 0
            && layer_idx >= 0 && layer_idx < g_kvmetal_n_layers;
        if (kvmetal_populate)
            llm_mlx_kvmetal_populate_batch(g_kvmetal_ctx, layer_idx, m, cache_offset, n_tokens, k_rope, v_t);
    }

    if (ok) {
        /* LLMMETAL_KVMETAL (sperimentale, default OFF): eligibilita' runtime -- ogni condizione
         * che fallisce fa fallback silenzioso alla SDPA normale sotto, mai un errore.
         * kvsl e' gia' garantito false qui (siamo nel ramo else di "if (ok && kvsl)"). */
        bool kvmetal_active = (llm_mlx_kvmetal_enabled() || llm_mlx_kvmetal_verify_enabled())
            && g_kvmetal_ctx != NULL && !g_kvmetal_disabled
            && n_tokens == 1 && has_cache
            && !llm_mlx_is_act16(mlx_array_dtype(q_rope))
            && m->cfg.kv_max_tokens > 0 && m->cfg.n_kv_heads > 0
            && m->cfg.head_dim % 32 == 0 && m->cfg.head_dim <= 256
            && m->cfg.n_heads % m->cfg.n_kv_heads == 0
            && layer_idx >= 0 && layer_idx < g_kvmetal_n_layers;
        mlx_array attn_metal = mlx_array_new();
        bool have_attn_metal = false;
        if (kvmetal_active)
            have_attn_metal = llm_mlx_kvmetal_step(g_kvmetal_ctx, layer_idx, m, cache_offset, q_rope, k_rope, v_t, &attn_metal);
        if (have_attn_metal && llm_da_enabled()) g_da.kv_metal++;

        if (have_attn_metal && !llm_mlx_kvmetal_verify_enabled()) {
            /* LLMMETAL_KVMETAL puro: usa il risultato del kernel Metal al posto della SDPA. */
            ok = mlx_array_set(&attn, attn_metal) == 0;
            CHK(ok, "kvmetal decode attention");
        } else {
            ok = llm_mlx_gqa_sdpa(&attn, q_rope, k_src, v_src,
                                  1.0f / sqrtf((float)m->cfg.head_dim), /*window=*/0, m->stream);
            CHK(ok, "scaled_dot_product_attention");
            if (ok && have_attn_metal && llm_mlx_kvmetal_verify_enabled()) {
                /* LLMMETAL_KVMETAL_VERIFY: confronta col risultato Metal, stampa il diff su
                 * stderr, NON cambia il risultato usato -- zero rischio. */
                mlx_array af = mlx_array_new(), amf = mlx_array_new();
                if (mlx_astype(&af, attn, MLX_FLOAT32, m->stream) == 0 &&
                    mlx_astype(&amf, attn_metal, MLX_FLOAT32, m->stream) == 0 &&
                    mlx_array_eval(af) == 0 && mlx_array_eval(amf) == 0) {
                    const float* ad = mlx_array_data_float32(af);
                    const float* bd = mlx_array_data_float32(amf);
                    size_t nel = (size_t)m->cfg.n_heads * m->cfg.head_dim;
                    if (ad && bd) {
                        double maxdiff = 0;
                        for (size_t vi = 0; vi < nel; vi++) {
                            double d = fabs((double)ad[vi] - (double)bd[vi]);
                            if (d > maxdiff) maxdiff = d;
                        }
                        fprintf(stderr, "[kvmetal-verify] layer=%d cache_offset=%d max_abs_diff=%.6g\n",
                                layer_idx, cache_offset, maxdiff);
                    }
                }
                mlx_array_free(af); mlx_array_free(amf);
            }
        }
        mlx_array_free(attn_metal);
    }
    }

    mlx_array attn_bt = mlx_array_new(), attn_flat = mlx_array_new();
    if (ok) {
        const int perm[4] = {0, 2, 1, 3};
        ok = mlx_transpose_axes(&attn_bt, attn, perm, 4, m->stream) == 0;
        CHK(ok, "attn transpose back");
        if (ok) {
            const int flat_shape[3] = {1, n_tokens, m->cfg.n_heads * m->cfg.head_dim};
            ok = mlx_reshape(&attn_flat, attn_bt, flat_shape, 3, m->stream) == 0;
            CHK(ok, "attn flatten");
        }
    }

    mlx_array ow = mlx_array_new(), os = mlx_array_new(), ob = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "self_attn.o_proj", namebuf, sizeof(namebuf));
    if (ac != NULL) { ok = ok && mlx_array_set(&ow, ac->ow) == 0 && mlx_array_set_optional(&os, ac->os) && mlx_array_set_optional(&ob, ac->ob); }
    else { ok = ok && multishard_get_triple(&m->weights, namebuf, &ow, &os, &ob); }
    CHK(ok, "o_proj lookup");
    /* Q2-b: energia dell'input di o_proj (uscita attention). Solo in calibrazione e solo se lo
     * stesso layer ha gia' accumulato la sezione q/k/v (contatore condiviso dal writer). */
    if (ok && m->im.active && m->im.attn_o != NULL && m->im.attn_cnt != NULL &&
        layer_idx >= 0 && layer_idx < m->im.n_layers && m->im.attn_cnt[layer_idx] > 0) {
        int ndo = mlx_array_ndim(attn_flat), lco = ndo >= 1 ? (int)mlx_array_dim(attn_flat, ndo - 1) : 0;
        int nro = lco > 0 ? (int)(mlx_array_size(attn_flat) / (size_t)lco) : 0;
        if (lco == m->im.attn_o_cols && nro > 0)
            llm_mlx_imatrix_energy_add(m, m->im.attn_o + (size_t)layer_idx * m->im.attn_o_cols,
                                       m->im.attn_o_cols, attn_flat, nro);
    }
    mlx_array attn_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&attn_out, attn_flat, ow, os, ob, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "o_proj matmul"); }

    if (ok) { ok = mlx_add(out_h, x, attn_out, m->stream) == 0; CHK(ok, "residual add (attn)"); }

    mlx_array all[] = {
        ln_w, normed, qw, qs, qb, kw, ks, kb, vw, vs, vb,
        q_lb, k_lb, v_lb, q0, k0, v0, q, k, v,
        q_heads, k_heads, v_heads, q_t, k_t, v_t, q_rope, k_rope,
        k_full, v_full, k_src, v_src, attn, attn_bt, attn_flat, ow, os, ob, attn_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
