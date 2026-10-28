/* ============================================================================
 * llm_mlx_gdn.c — dominio Gated DeltaNet: gdn_attention_block (attn lineare) e qwen4_linear_gdn_block.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_gdn_gate_kernel.h"
#include "llm_mlx_gdn_kernel.h"
#include "llm_mlx_swiglu_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

bool gdn_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    /* M1 (decode compilato): closure dell'INTERO blocco GDN di decode. Se il gate e' soddisfatto
     * il blocco eager non viene eseguito affatto: la closure produce out_h + i due stati nuovi.
     * Bail-out conservativo (si torna al path eager) su ogni condizione che la closure non
     * replica: LoRA su una qualunque proiezione, spec-capture, kernel GDN spento, A8 a M=1. */
    if (n_tokens == 1 && llm_mlx_m1_block_enabled() && llm_mlx_compile_enabled() &&
        llm_mlx_compile_qwen35_enabled() && !g_spec_capture &&
        !llm_env_isset("LLM_GDN_KERNEL_OFF") && !llm_env_isset("LLMMETAL_OQ_A8_MINTOK") &&
        llm_mlx_model_lora_get(m, layer_idx, "in_proj_qkv") < 0 &&
        llm_mlx_model_lora_get(m, layer_idx, "in_proj_z") < 0 &&
        llm_mlx_model_lora_get(m, layer_idx, "in_proj_a") < 0 &&
        llm_mlx_model_lora_get(m, layer_idx, "in_proj_b") < 0 &&
        llm_mlx_model_lora_get(m, layer_idx, "out_proj") < 0) {
        if (llm_mlx_gdn_block_compiled(m, layer_idx, x, k_cache, v_cache, has_cache, out_h))
            return true;
        if (getenv("LLM_M1_DBG"))
            fprintf(stderr, "[m1blk] fallback eager li=%d\n", layer_idx);
    }
    /* §13.5: k_cache[layer] = stato ricorrente GDN S [B,Hv,Dv,Dk]; v_cache[layer] =
     * conv_state [B,kk-1,conv_dim]; has_cache[layer] = true se lo stato esiste gia'
     * (decode che continua il prefill). Riuso i due slot perche' i layer lineari GDN non
     * usano la KV-cache dell'attention (solo i layer full la usano). */
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];
    const int B = 1;
    const int Hk = m->cfg.linear_num_key_heads;
    const int Hv = m->cfg.linear_num_value_heads;
    const int Dk = m->cfg.linear_key_head_dim;
    const int Dv = m->cfg.linear_value_head_dim;
    const int key_dim = Hk * Dk;
    const int value_dim = Hv * Dv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int S = n_tokens;
    mlx_stream s = m->stream;

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "gdn_attention[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array ln_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "input_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln_w);
    mlx_array h = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&h, x, ln_w, m->cfg.rms_norm_eps, s); CHK(ok, "input_layernorm"); }
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, h, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { double sm=0; size_t n=mlx_array_size(vf); for (size_t q=0;q<(size_t)m->cfg.hidden_size && q<n;q++) sm+=d[q]; fprintf(stderr, "[gdn] h_inputln tok0sum=%.5f first6= %.6f %.6f %.6f %.6f %.6f %.6f\n", sm, d[0],d[1],d[2],d[3],d[4],d[5]); }
        }
        mlx_array_free(vf);
    }
    /* Ternary-Bonsai: in_proj_qkv e in_proj_z sono elencati, in_proj_a/b NO -> copia ruotata
     * per qkv/z, `h` inalterato per a/b. */
    mlx_array h_rot = mlx_array_new();
    if (ok) { ok = mlx_array_set(&h_rot, h) == 0; }
    if (ok) ok = llm_bonsai_rotate(&h_rot, m, layer_idx, "linear_attn.in_proj_qkv.weight", false);

    /* in_proj_qkv / in_proj_z / in_proj_a / in_proj_b (quantizzati) */
    mlx_array qkv_w=mlx_array_new(), qkv_sc=mlx_array_new(), qkv_b=mlx_array_new();
    mlx_array z_w=mlx_array_new(), z_sc=mlx_array_new(), z_b=mlx_array_new();
    mlx_array a_w=mlx_array_new(), a_sc=mlx_array_new(), a_b=mlx_array_new();
    mlx_array bw=mlx_array_new(), b_sc=mlx_array_new(), b_b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_qkv", namebuf, sizeof(namebuf));
    /* PACKED (sperimentale): qkv sui pesi zero-copy, salta load/Stage A/closure per qkv. */
    mlx_array qkv_pk = mlx_array_new(); int qkv_packed = 0;
    if (llm_mlx_linear_packed_model(&m->packed, m->model_dir, namebuf, m->cfg.quant_bits,
                                    m->cfg.quant_group_size, h_rot, &qkv_pk, s) == 0) qkv_packed = 1;
    else mlx_array_free(qkv_pk);
    if (!qkv_packed) ok = ok && multishard_get_triple(&m->weights, namebuf, &qkv_w, &qkv_sc, &qkv_b);
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_z", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &z_w, &z_sc, &z_b);
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_a", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &a_w, &a_sc, &a_b);
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_b", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &bw, &b_sc, &b_b);
    CHK(ok, "in_proj lookups");
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, bw, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] bw dtype=%d dims=(", (int)mlx_array_dtype(bw));
                     for (int k=0;k<mlx_array_ndim(bw);k++) fprintf(stderr,"%d,", (int)mlx_array_dim(bw,k));
                     fprintf(stderr, ") [0:6] ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }

    mlx_array qkv=mlx_array_new(), z=mlx_array_new(), av=mlx_array_new(), bv=mlx_array_new();
    /* FEDELTA' oMLX (oq_a8_gdn_projections): Stage A UNA volta, condivisa fra qkv/z/a/b.
     * oMLX: "qkv e' Q4 mentre z/a/b possono essere Q5, ed e' proprio il caso per cui la
     * Stage A condivisa e' pensata" (Qa non dipende dai bit del peso). */
    bool a8_shared_gdn = false;
    if (ok && !qkv_packed && m->bonsai_had == NULL) {   /* Ternary-Bonsai: usa h NON ruotato */
        mlx_array s_qa, s_sa, s_ra; s_qa.ctx = NULL; s_sa.ctx = NULL; s_ra.ctx = NULL;
        if (llm_mlx_oq_a8_stage_try(h, qkv_w, qkv_sc, qkv_b, gsz, bts, m->cfg.quant_mode, s, &s_qa, &s_sa, &s_ra)) {
            bool k1 = llm_mlx_oq_a8_qmm_stage_try(&qkv, s_qa, s_sa, s_ra, h, qkv_w, qkv_sc, qkv_b, gsz, bts, m->cfg.quant_mode, s);
            bool k2 = k1 && llm_mlx_oq_a8_qmm_stage_try(&z, s_qa, s_sa, s_ra, h, z_w, z_sc, z_b, gsz, bts, m->cfg.quant_mode, s);
            bool k3 = k2 && llm_mlx_oq_a8_qmm_stage_try(&av, s_qa, s_sa, s_ra, h, a_w, a_sc, a_b, gsz, bts, m->cfg.quant_mode, s);
            bool k4 = k3 && llm_mlx_oq_a8_qmm_stage_try(&bv, s_qa, s_sa, s_ra, h, bw, b_sc, b_b, gsz, bts, m->cfg.quant_mode, s);
            if (k4) { a8_shared_gdn = true; }
            else { mlx_array_free(qkv); qkv = mlx_array_new(); mlx_array_free(z); z = mlx_array_new();
                   mlx_array_free(av); av = mlx_array_new(); mlx_array_free(bv); bv = mlx_array_new(); }
            mlx_array_free(s_qa); mlx_array_free(s_sa); mlx_array_free(s_ra);
        }
    }
        /* M1 iter-1: le 4 in_proj in UNA closure compilata (solo decode M=1, niente OQ-A8 shared/LoRA). */
    bool gdn_ip_compiled = false;
    if (getenv("LLM_M1_DBG") && n_tokens == 1)
        fprintf(stderr, "[m1dbg] layer=%d ok=%d a8_shared=%d compile=%d q35=%d lora=%d\n", layer_idx, (int)ok,
                (int)a8_shared_gdn, (int)llm_mlx_compile_enabled(), (int)llm_mlx_compile_qwen35_enabled(),
                (int)(llm_mlx_model_lora_get(m, layer_idx, "in_proj_qkv") >= 0));
    if (ok && !qkv_packed && !a8_shared_gdn && n_tokens == 1 && llm_mlx_compile_qwen35_enabled() &&
        m->bonsai_had == NULL &&   /* Ternary-Bonsai: la closure M1 non ruota l'input */
        llm_mlx_compile_enabled() &&
        llm_mlx_model_lora_get(m, layer_idx, "in_proj_qkv") < 0 &&
        llm_mlx_model_lora_get(m, layer_idx, "in_proj_z") < 0 &&
        llm_mlx_model_lora_get(m, layer_idx, "in_proj_a") < 0 &&
        llm_mlx_model_lora_get(m, layer_idx, "in_proj_b") < 0) {
        gdn_ip_compiled = llm_mlx_gdn_inproj_compiled(layer_idx, h, &qkv, &z, &av, &bv,
            qkv_w, qkv_sc, qkv_b, z_w, z_sc, z_b, a_w, a_sc, a_b, bw, b_sc, b_b, gsz, bts, m->cfg.quant_mode);
        if (!gdn_ip_compiled) {
            mlx_array_free(qkv); qkv = mlx_array_new();
            mlx_array_free(z);   z   = mlx_array_new();
            mlx_array_free(av);  av  = mlx_array_new();
            mlx_array_free(bv);  bv  = mlx_array_new();
        }
    }
    if (ok && !qkv_packed && !a8_shared_gdn && !gdn_ip_compiled) { ok = quantized_linear_cfg(&qkv, h_rot, qkv_w, qkv_sc, qkv_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_qkv");
        if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
            mlx_array vf = mlx_array_new();
            if (mlx_astype(&vf, qkv, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
                const float* d = mlx_array_data_float32(vf);
                if (d) { fprintf(stderr, "[gdn] qkv[0,:6] ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
            }
            mlx_array_free(vf);
        } }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "in_proj_qkv"); if (li >= 0 && ok && !qkv_packed) ok = llm_mlx_lora_apply_mlx(&qkv, h, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    if (qkv_packed) { mlx_array_free(qkv); qkv = qkv_pk; }
    if (ok && !a8_shared_gdn && !gdn_ip_compiled) { ok = quantized_linear_cfg(&z, h_rot, z_w, z_sc, z_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_z"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "in_proj_z"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&z, h, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    if (ok && !a8_shared_gdn && !gdn_ip_compiled) { ok = quantized_linear_cfg(&av, h, a_w, a_sc, a_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_a"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "in_proj_a"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&av, h, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    if (ok && !a8_shared_gdn && !gdn_ip_compiled) { ok = quantized_linear_cfg(&bv, h, bw, b_sc, b_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_b"); }
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, bv, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] bv_raw ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
        vf = mlx_array_new();
        if (mlx_astype(&vf, av, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] av_raw ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "in_proj_b"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&bv, h, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }

    /* conv1d depthwise su concat([conv_state, qkv]) */
    mlx_array conv_w = mlx_array_new();
    mlx_array* cw32p = llm_mlx_gdn_conv_cached(m, layer_idx);
    if (cw32p != NULL) {
        ok = ok && mlx_array_set(&conv_w, *cw32p) == 0;
        CHK(ok, "conv1d cast cache");
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.conv1d.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &conv_w);
        CHK(ok, "conv1d lookup");
        /* conv1d.weight e' BF16 (verificato sull'header safetensors); mlx_conv_general richiede
         * input e weight con lo stesso dtype. qkv e' float32 -> cast esplicito del weight. */
        if (ok) {
            mlx_array cw32 = mlx_array_new();
            ok = mlx_astype(&cw32, conv_w, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0;
            if (ok) { mlx_array_free(conv_w); conv_w = cw32; }
            else mlx_array_free(cw32);
            CHK(ok, "conv1d weight cast");
        }
    }
    int kk = m->cfg.linear_conv_kernel_dim;
    int cstate_len = kk - 1;
    int cshape[3] = { B, cstate_len, conv_dim };
    mlx_array conv_state = mlx_array_new();
    /* §13.5: conv_state persistente (v_cache[layer]). Se has_cache e v_cache ha un valore
     * (decode che continua), riuso; altrimenti zero (primo prefill del layer). */
    if (ok) {
        if (has_cache && v_cache && v_cache->ctx != NULL && mlx_array_ndim(*v_cache) > 0) {
            ok = mlx_array_set(&conv_state, *v_cache) == 0;
        } else {
            ok = mlx_zeros(&conv_state, cshape, 3, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0;
        }
        CHK(ok, "conv_state init");
    }
    mlx_array conv_input = mlx_array_new();
    /* concat [conv_state, qkv] lungo l'asse 1 (sequenza), come il riferimento
     * mx.concatenate([...], axis=1). mlx_concatenate senza axis unirebbe sull'ultimo
     * asse (conv_dim) -> forma sbagliata. */
    if (ok) { mlx_array cv_arr[2] = { conv_state, qkv }; mlx_vector_array cv = mlx_vector_array_new_data(cv_arr, 2); ok = mlx_concatenate_axis(&conv_input, cv, /*axis=*/1, s) == 0; CHK(ok, "concat conv_state+qkv"); mlx_vector_array_free(cv); }
    /* conv1d depthwise (groups=conv_dim) via mlx_conv_general: mlx_conv1d non supporta
     * groups != 1 (errore reale osservato: "Can only handle groups != 1 in 1D or 2D").
     * Il conv1d GDN e' depthwise (groups=conv_dim=10240, kernel kk), stride 1, padding 0.
     * Weight [conv_dim, kk, 1]; input [B, S+kk-1, conv_dim] -> output [B, S, conv_dim]. */
    mlx_array conv_out = mlx_array_new();
    if (ok) {
        int c_stride[1] = {1};
        int c_pad_lo[1] = {0};
        int c_pad_hi[1] = {0};
        int c_kdil[1] = {1};
        int c_idil[1] = {1};
        ok = mlx_conv_general(&conv_out, conv_input, conv_w, c_stride, 1, c_pad_lo, 1, c_pad_hi, 1,
                              c_kdil, 1, c_idil, 1, /*groups=*/conv_dim, /*flip=*/false, s) == 0;
        CHK(ok, "conv1d (depthwise)");
    }
    /* silu(conv_out) */
    mlx_array sig = mlx_array_new();
    mlx_array silu = mlx_array_new();
    if (ok) { ok = llm_mlx_silu_parity(&silu, &sig, conv_out, s) == 0; CHK(ok, "silu"); }

    /* split in q,k,v per dim (mlx_split_sections: [0,key), [key,2key), [2key,conv)) lungo
     * l'ULTIMO asse (conv_dim), come il riferimento verificato (mx.split(..., -1)). */
    int secs[2] = { key_dim, 2 * key_dim };
    mlx_vector_array kv_split = mlx_vector_array_new();
    mlx_array q_part=mlx_array_new(), k_part=mlx_array_new(), v_part=mlx_array_new();
    if (ok) { ok = mlx_split_sections(&kv_split, silu, secs, 2, /*axis=*/-1, s) == 0; CHK(ok, "split q/k/v"); }
    if (ok) { ok = mlx_vector_array_get(&q_part, kv_split, 0) == 0 &&
                    mlx_vector_array_get(&k_part, kv_split, 1) == 0 &&
                    mlx_vector_array_get(&v_part, kv_split, 2) == 0; CHK(ok, "get q/k/v"); }

    /* reshape a teste: q,k [B,S,Hk,Dk], v [B,S,Hv,Dv] */
    int q_shape[4] = { B, S, Hk, Dk };
    int v_shape[4] = { B, S, Hv, Dv };
    mlx_array q4=mlx_array_new(), k4=mlx_array_new(), v4=mlx_array_new();
    if (ok) { ok = mlx_reshape(&q4, q_part, q_shape, 4, s) == 0 &&
                    mlx_reshape(&k4, k_part, q_shape, 4, s) == 0 &&
                    mlx_reshape(&v4, v_part, v_shape, 4, s) == 0; CHK(ok, "reshape q/k/v"); }
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, q4, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] q4 ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
        vf = mlx_array_new();
        if (mlx_astype(&vf, v4, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] v4 ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }

    /* inv_scale = Dk**-0.5; q = inv^2 * rms_norm(q); k = inv * rms_norm(k). eps=1e-6
     * come il riferimento (mx.fast.rms_norm(..., None, 1e-6)). */
    const double inv_scale = pow((double)Dk, -0.5);
    mlx_array inv2 = llm_mlx_act_scalar((float)(inv_scale * inv_scale));
    mlx_array inv1 = llm_mlx_act_scalar((float)inv_scale);
    mlx_array q_n=mlx_array_new(), k_n=mlx_array_new();
    if (ok) { ok = mlx_fast_rms_norm_no_scale(&q_n, q4, Dk, 1e-6f, s) &&
                    mlx_fast_rms_norm_no_scale(&k_n, k4, Dk, 1e-6f, s); CHK(ok, "rms_norm q/k"); }
    mlx_array q_s=mlx_array_new(), k_s=mlx_array_new();
    if (ok) { ok = mlx_multiply(&q_s, q_n, inv2, s) == 0 &&
                    mlx_multiply(&k_s, k_n, inv1, s) == 0; CHK(ok, "inv_scale q/k"); }
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, q_s, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] q_s ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }

    /* beta = sigmoid(b); g = compute_g(A_log, a, dt_bias) = exp(-exp(A_log)*softplus(a+dt_bias)) */
    mlx_array A_log=mlx_array_new(), dt_bias=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.A_log", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &A_log);
    mlx_array* _dt = llm_mlx_gdn_dt_bias(m, layer_idx);
    if (_dt != NULL) { ok = ok && mlx_array_set(&dt_bias, *_dt) == 0; }
    else {
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.dt_bias", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &dt_bias);
    }
    CHK(ok, "A_log/dt_bias lookup");
    mlx_array beta=mlx_array_new();
    mlx_array g=mlx_array_new();
    mlx_array eAlog=mlx_array_new();
    mlx_array a_plus_dt=mlx_array_new();
    mlx_array sp_exp=mlx_array_new();
    mlx_array zero_arr=llm_mlx_act_scalar(0.0f);
    mlx_array one_arr=llm_mlx_act_scalar(1.0f);
    mlx_array sp_1p=mlx_array_new();
    mlx_array sp_log=mlx_array_new();
    mlx_array neg_exp=mlx_array_new();
    mlx_array neg_g=mlx_array_new();
    {
        mlx_array* _ae = llm_mlx_gdn_alog_exp(m, layer_idx);
        if (_ae != NULL) { ok = ok && mlx_array_set(&eAlog, *_ae) == 0; CHK(ok, "exp A_log cache"); }
        else if (ok) { ok = mlx_exp(&eAlog, A_log, s) == 0; CHK(ok, "exp A_log"); }
    }
    /* LLMMETAL_GDN_GATE_METAL: beta + g in UN dispatch (8 op di grafo -> 1). Fallback: catena MLX. */
    bool gate_fused = false;
    if (ok && llm_mlx_gdn_gate_metal_enabled() && eAlog.ctx != NULL &&
        mlx_array_dtype(bv) == mlx_array_dtype(av) &&
        mlx_array_dtype(eAlog) == MLX_FLOAT32 &&
        mlx_array_dtype(dt_bias) == mlx_array_dtype(av)) {
        gate_fused = (llm_mlx_gdn_gate(&g, &beta, av, bv, eAlog, dt_bias, s) == 0);
        if (!gate_fused) { mlx_array_free(g); g = mlx_array_new(); mlx_array_free(beta); beta = mlx_array_new(); }
    }
    if (ok && !gate_fused) {
        if (ok) { ok = mlx_sigmoid(&beta, bv, s) == 0; CHK(ok, "beta sigmoid"); }
        /* softplus(a+dt_bias) = logaddexp(a+dt_bias, 0); a: [B,S,Hv], dt_bias: [Hv] -> broadcast */
        if (ok) { ok = mlx_add(&a_plus_dt, av, dt_bias, s) == 0; CHK(ok, "a+dt_bias"); }
        /* nn.softplus di mlx-lm = logaddexp(x, 0): UN solo arrotondamento bf16 (non exp/+1/log a 3 passi) */
        if (ok) { ok = mlx_logaddexp(&sp_log, a_plus_dt, zero_arr, s) == 0; CHK(ok, "softplus logaddexp"); }
        if (ok) { ok = mlx_multiply(&neg_exp, eAlog, sp_log, s) == 0; CHK(ok, "exp(A_log)*softplus"); }
        if (ok) { ok = mlx_negative(&neg_g, neg_exp, s) == 0; CHK(ok, "negate exp(A_log)*softplus"); }
        if (ok) { ok = mlx_exp(&g, neg_g, s) == 0; CHK(ok, "g = exp(-...)"); }
    }
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, beta, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] beta ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
        vf = mlx_array_new();
        if (mlx_astype(&vf, g, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] g ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }
    /* Diagnostica: confronta kernel vs catena di riferimento (max|diff|). Stampa una volta. */
    if (gate_fused && ok && llm_ai_mlx_opt_env("LLMMETAL_GDN_GATE_METAL_VERIFY") != NULL) {
        mlx_array gr=mlx_array_new(), br=mlx_array_new();
        mlx_array apd=mlx_array_new(), spe=mlx_array_new(), sp1=mlx_array_new(), spl=mlx_array_new();
        mlx_array ne=mlx_array_new(), ng2=mlx_array_new();
        bool vok = mlx_add(&apd, av, dt_bias, s) == 0 && mlx_logaddexp(&spl, apd, zero_arr, s) == 0 &&
                   mlx_multiply(&ne, eAlog, spl, s) == 0 && mlx_negative(&ng2, ne, s) == 0 &&
                   mlx_exp(&gr, ng2, s) == 0 && mlx_sigmoid(&br, bv, s) == 0;
        double dg = -1.0, db = -1.0;
        if (vok) {
            mlx_array df=mlx_array_new(), dff=mlx_array_new();
            if (mlx_subtract(&df, g, gr, s) == 0 && mlx_astype(&dff, df, MLX_FLOAT32, s) == 0) {
                mlx_array_eval(dff);
                size_t n = mlx_array_size(dff);
                const float* pd = mlx_array_data_float32(dff);
                if (pd != NULL) { dg = 0.0; for (size_t q = 0; q < n; q++) { double v = fabs((double)pd[q]); if (v > dg) dg = v; } }
            }
            mlx_array df2=mlx_array_new(), df2f=mlx_array_new();
            if (mlx_subtract(&df2, beta, br, s) == 0 && mlx_astype(&df2f, df2, MLX_FLOAT32, s) == 0) {
                mlx_array_eval(df2f);
                size_t n = mlx_array_size(df2f);
                const float* pd = mlx_array_data_float32(df2f);
                if (pd != NULL) { db = 0.0; for (size_t q = 0; q < n; q++) { double v = fabs((double)pd[q]); if (v > db) db = v; } }
            }
            mlx_array_free(df); mlx_array_free(dff); mlx_array_free(df2); mlx_array_free(df2f);
            static int once = 0;
            if (!once++) fprintf(stderr, "[gdn-gate] max|g_kernel-g_ref|=%.6g  max|beta_kernel-beta_ref|=%.6g\n", dg, db);
        }
        mlx_array_free(gr); mlx_array_free(br); mlx_array_free(apd); mlx_array_free(spe);
        mlx_array_free(sp1); mlx_array_free(spl); mlx_array_free(ne); mlx_array_free(ng2);
    }

    /* repeat q,k da Hk a Hv teste (Hv//Hk = 3) lungo l'asse teste (-2), come
     * gated_delta_ops (mx.repeat(x, repeat_factor, -2)). */
    const int repeat_factor = Hv / Hk;
    mlx_array q_rep=mlx_array_new(), k_rep=mlx_array_new();
    if (ok) {
        if (repeat_factor > 1) {
            ok = mlx_repeat_axis(&q_rep, q_s, repeat_factor, /*axis=*/2, s) == 0 &&
                 mlx_repeat_axis(&k_rep, k_s, repeat_factor, /*axis=*/2, s) == 0;
        } else {
            ok = mlx_array_set(&q_rep, q_s) == 0 && mlx_array_set(&k_rep, k_s) == 0;
        }
        CHK(ok, "repeat q/k to Hv");
    }

    /* Stato ricorrente S [B,Hv,Dv,Dk] (k_cache[layer]). §13.5: se has_cache e k_cache ha
     * un valore (decode che continua il prefill), riuso lo stato finale del passo
     * precedente; altrimenti zero (primo prefill del layer). Il loop lo evolve poi di S
     * token (prefill: l'intero batch; decode: 1 step). */
    int state_shape[4] = { B, Hv, Dv, Dk };
    mlx_array state = mlx_array_new();
    if (ok) {
        if (has_cache && k_cache && k_cache->ctx != NULL && mlx_array_ndim(*k_cache) > 0) {
            ok = mlx_array_set(&state, *k_cache) == 0;
        } else {
            ok = mlx_zeros(&state, state_shape, 4, MLX_FLOAT32, s) == 0;
        }
        CHK(ok, "state init");
    }

    /* §2026-09-01 KERNEL METAL GDN FUSO: per il PREFILL (S>1) la ricorrenza viene calcolata in
     * un solo kernel Metal (llm_mlx_gdn_prefill, modulo llm_mlx_gdn_kernel.h) invece del loop
     * per-token (~32 chiamate MLX per step, il collo del prefill). Stessa ricorrenza esatta.
     * Se il kernel non e' disponibile (dimensioni/Dk) o fallisce, fallback al loop qui sotto. */
    mlx_array y_stack = mlx_array_new();
    bool kernel_done = false;
    if (ok && S >= 1 && !llm_env_isset("LLM_GDN_KERNEL_OFF")) {
      if (g_spec_capture && S > 1 && S <= LLM_SPEC_CAP_MAXS) {
        /* S5(b): UN SOLO call del kernel FUSO con l'output `state_seq` (stato dopo OGNI
         * token) -> rollback O(1) senza l'overhead di launch per-token. */
        mlx_array st_n = mlx_array_new(), sseq_arr = mlx_array_new();
        if (llm_mlx_gdn_prefill_sseq(&y_stack, &st_n, &sseq_arr, q_s, k_s, v4, g, beta, state, S, 0, s) == 0) {
            (void)mlx_array_set(&state, st_n);
            if (layer_idx < LLM_SPEC_MAXL) {
                (void)mlx_array_set(&g_spec_snap_k[layer_idx], sseq_arr);   /* [B,S,Hv,Dv,Dk] */
                (void)mlx_array_set(&g_spec_snap_v[layer_idx], conv_input); /* [B,cstate_len+S,conv_dim] */
            }
            kernel_done = true;
        }
        mlx_array_free(st_n); mlx_array_free(sseq_arr);
        CHK(ok, "gdn kernel sseq (capture)");
      } else {
        mlx_array state_new_k = mlx_array_new();
        if (llm_mlx_gdn_prefill(&y_stack, &state_new_k, q_s, k_s, v4, g, beta, state, S, /*g_vector=*/0, s) == 0) {
            mlx_array_set(&state, state_new_k);
            kernel_done = true;
        }
        mlx_array_free(state_new_k);
      }
    }
    if (getenv("LLM_GDN_TIME")) {
        struct timespec _g0,_g1; clock_gettime(CLOCK_MONOTONIC,&_g0);
        if (ok) (void)mlx_array_eval(y_stack);
        clock_gettime(CLOCK_MONOTONIC,&_g1);
        fprintf(stderr, "[gdn-time] layer=%d S=%d %.2fms\n", layer_idx, S,
                (_g1.tv_sec-_g0.tv_sec)*1e3+(_g1.tv_nsec-_g0.tv_nsec)/1e6);
    }
    if (llm_mlx_gdn_kernel_dbg_enabled())
        fprintf(stderr, "[gdn-kernel] layer=%d S=%d -> %s\n", layer_idx, S,
                kernel_done ? "KERNEL FUSO" : "LOOP per-token");

    /* Loop ricorrente su S token (gated_delta_update, reference mlx-lm). Per ogni t:
     *   decay = g[...,None,None]; state *= decay
     *   kv_mem = (state * k[...,None,:]).sum(-1)
     *   delta = (v - kv_mem) * beta[...,None]
     *   state += k[...,None,:] * delta[...,None]
     *   y_t   = (state * q[...,None,:]).sum(-1)
     * Raccoglie y_t in un vettore e li stack lungo l'asse 1 -> [B,S,Hv,Dv]. */
    mlx_vector_array ys = mlx_vector_array_new();
    if (!kernel_done) for (int t = 0; ok && t < S; t++) {
        int st0[4] = {0, t, 0, 0};
        int st1[4] = {1, t + 1, Hv, Dk};
        int stv[4] = {1, t + 1, Hv, Dv};
        int ones4[4] = {1, 1, 1, 1};
        int ones3[3] = {1, 1, 1};
        mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
        mlx_array g_t=mlx_array_new(), beta_t=mlx_array_new();
        if (ok) {
            ok = mlx_slice(&q_t, q_rep, st0, 4, st1, 4, ones4, 4, s) == 0 &&
                 mlx_slice(&k_t, k_rep, st0, 4, st1, 4, ones4, 4, s) == 0 &&
                 mlx_slice(&v_t, v4, st0, 4, stv, 4, ones4, 4, s) == 0;
            int gt0[3] = {0, t, 0}, gt1[3] = {1, t + 1, Hv};
            ok = ok && mlx_slice(&g_t, g, gt0, 3, gt1, 3, ones3, 3, s) == 0 &&
                       mlx_slice(&beta_t, beta, gt0, 3, gt1, 3, ones3, 3, s) == 0;
        }
        /* squeeze assi 1 (singleton) */
        mlx_array q_sq=mlx_array_new(), k_sq=mlx_array_new(), v_sq=mlx_array_new();
        mlx_array g_sq=mlx_array_new(), beta_sq=mlx_array_new();
        if (ok) {
            ok = mlx_squeeze_axis(&q_sq, q_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&k_sq, k_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&v_sq, v_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&g_sq, g_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&beta_sq, beta_t, 1, s) == 0;
        }
        /* §refactor: il passo ricorrente GDN ora delega al modulo componibile
         * llm_mlx_gated_delta_step (stessa matematica, unica sorgente). */
        mlx_array y_t = mlx_array_new();
        if (ok) ok = llm_mlx_gated_delta_step(&y_t, &state, q_sq, k_sq, v_sq, g_sq, beta_sq, s);
        CHK(ok, "gated_delta_step");
        /* y_t: [B,Hv,Dv] viene stackato DIRETTAMENTE lungo l'asse 1 (stack di array
         * 3D [B,Hv,Dv] su axis=1 -> [B,S,Hv,Dv]). NON espandere prima: espandere a
         * 4D [B,1,Hv,Dv] e poi stack su axis=1 produce 5D [B,S,1,Hv,Dv] (bug reale
         * osservato). */
        if (ok) { ok = mlx_vector_array_append_value(ys, y_t) == 0; CHK(ok, "collect y"); }
        mlx_array_free(q_t); mlx_array_free(k_t); mlx_array_free(v_t);
        mlx_array_free(g_t); mlx_array_free(beta_t);
        mlx_array_free(q_sq); mlx_array_free(k_sq); mlx_array_free(v_sq);
        mlx_array_free(g_sq); mlx_array_free(beta_sq);
        mlx_array_free(y_t);
    }
    if (!kernel_done && ok) { ok = mlx_stack_axis(&y_stack, ys, /*axis=*/1, s) == 0; CHK(ok, "stack y"); }

    /* §13.5: salva lo stato ricorrente finale in k_cache[layer] e il conv_state aggiornato
     * in v_cache[layer], perche' il decode (forward_one_token) continui da dove ha lasciato
     * il prefill. conv_state aggiornato = ultimi cstate_len token di conv_input (come
     * mlx-lm GatedDeltaNet.__call__: cache[0] = conv_input[:, -n_keep:, :]). Eval prima di
     * salvare (state/conv_input sono lazy). */
    if (ok) {
        mlx_array cstate_new = mlx_array_new();
        /* conv_input: [B, cstate_len+S, conv_dim]; prendi gli ultimi cstate_len lungo l'asse 1 */
        int cst0[3] = {0, S, 0};
        int cst1[3] = {1, S + cstate_len, conv_dim};
        int cstrides[3] = {1, 1, 1};
        ok = mlx_slice(&cstate_new, conv_input, cst0, 3, cst1, 3, cstrides, 3, s) == 0;
        if (ok) { mlx_array_free(conv_state); conv_state = cstate_new; }
        else mlx_array_free(cstate_new);
        CHK(ok, "conv_state update");
        /* RMLX/bug P0 (2026-09-03): NIENTE eval per-layer GDN. state/conv_state restano LAZY
         * (refcount) e vengono materializzati dall' eval a fine token/chunk (dipendenza
         * dell'hidden finale); LLM_GDN_EVAL=1 ripristina il vecchio comportamento (2 sync/layer). */
        if (ok && llm_mlx_gdn_eval_forced())
            ok = mlx_array_eval(state) == 0 && mlx_array_eval(conv_state) == 0;
        if (ok) ok = mlx_array_set(k_cache, state) == 0 && mlx_array_set(v_cache, conv_state) == 0;
        CHK(ok, "gdn state store");
    }

    /* RMSNormGated(y, z) = silu(z) * rms_norm(y, norm_w, eps). norm_w = linear_attn.norm.weight [Dv]. */
    mlx_array norm_w=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &norm_w);
    CHK(ok, "norm.weight lookup");
    /* z [B,S,value_dim] -> [B,S,Hv,Dv] */
    int z_shape[4] = { B, S, Hv, Dv };
    mlx_array z4=mlx_array_new();
    if (ok) { ok = mlx_reshape(&z4, z, z_shape, 4, s) == 0; CHK(ok, "reshape z"); }
    mlx_array y_norm=mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&y_norm, y_stack, norm_w, m->cfg.rms_norm_eps, s); CHK(ok, "rms_norm y"); }
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, y_stack, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) {
                double tot=0, hs[4]={0,0,0,0};
                for (int q=0;q<6144;q++) { tot += d[q]; if (q<512) hs[q/128] += d[q]; }
                fprintf(stderr, "[gdn] y_stack tok0: sum=%.5f head0=%.5f h1=%.5f h2=%.5f h3=%.5f | [0,:6]=", tot, hs[0],hs[1],hs[2],hs[3]);
                for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n");
            }
        }
        mlx_array_free(vf);
    }
    /* D11 (2026-10-02): gated norm in FP32 come il reference mlx-lm `_precise_swiglu`
     * (`Qwen3NextRMSNormGated`): `(silu(gate.astype(f32)) * x.astype(f32)).astype(h.dtype)`.
     * Il precedente "rompe la parita'" era il BUG del 2-forward (B su stato post-A), NON l'fp32:
     * ora che e' risolto (a5592a9) questo fix e' verificabile. */
    mlx_array z32=mlx_array_new(), yn32=mlx_array_new();
    if (ok) { ok = mlx_astype(&z32, z4, MLX_FLOAT32, s) == 0; }
    if (ok) { ok = mlx_astype(&yn32, y_norm, MLX_FLOAT32, s) == 0; }
    mlx_array z32_sig=mlx_array_new(), silu32=mlx_array_new(), g32=mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&z32_sig, z32, s) == 0; }
    if (ok) { ok = mlx_multiply(&silu32, z32, z32_sig, s) == 0; CHK(ok, "silu(z) fp32"); }
    if (ok) { ok = mlx_multiply(&g32, yn32, silu32, s) == 0; CHK(ok, "RMSNormGated fp32"); }
    mlx_array gated=mlx_array_new();
    if (ok) { ok = mlx_astype(&gated, g32, mlx_array_dtype(y_stack), s) == 0; CHK(ok, "gated -> h.dtype"); }
    mlx_array_free(z32); mlx_array_free(yn32); mlx_array_free(z32_sig); mlx_array_free(silu32); mlx_array_free(g32);
    /* out_proj: [B,S,value_dim] -> [B,S,hidden_size] */
    int flat_shape[3] = { B, S, value_dim };
    mlx_array gated_flat=mlx_array_new();
    if (ok) { ok = mlx_reshape(&gated_flat, gated, flat_shape, 3, s) == 0; CHK(ok, "flatten gated"); }
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, gated_flat, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { double sm=0; for (int q=0;q<6144;q++) sm+=d[q]; fprintf(stderr, "[gdn] out_proj_in sum=%.5f [0,:6]=", sm); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }
    /* NOTA: il pack ha `gdn_v_grouped=true` ma il reference mlx-vlm NON permuta l'attivazione
     * di out_proj (`self.out_proj(out.reshape(B,S,-1))`): la GDN produce GIA' il layout corretto.
     * Una precedente permutazione [hd,nk,rep]->[hd,rep,nk] era ERRATA (rompeva la parita' col
     * reference). Rimossa. */
    /* Ternary-Bonsai: linear_attn.out_proj e' elencato (input width = value_dim). */
    if (ok) ok = llm_bonsai_rotate(&gated_flat, m, layer_idx, "linear_attn.out_proj.weight", false);
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, gated_flat, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] gated_rot[0,0,:6] ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }
    mlx_array out_w=mlx_array_new(), out_sc=mlx_array_new(), out_b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.out_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &out_w, &out_sc, &out_b);
    CHK(ok, "out_proj lookup");
    mlx_array r=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&r, gated_flat, out_w, out_sc, out_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "out_proj");
        if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
            mlx_array vf = mlx_array_new();
            if (mlx_astype(&vf, r, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
                const float* d = mlx_array_data_float32(vf);
                if (d) { fprintf(stderr, "[gdn] r_out[:6] ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
            }
            mlx_array_free(vf);
        } }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "out_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&r, gated_flat, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    /* residual: h = x + r (il layer lineare ritorna la somma, come DecoderLayer.__call__). */
    mlx_array h_out=mlx_array_new();
    if (ok) { ok = mlx_add(&h_out, x, r, s) == 0; CHK(ok, "residual add (gdn)"); }
    mlx_array_set(out_h, h_out);
    mlx_array_free(h_out);

    mlx_array_free(q_part); mlx_array_free(k_part); mlx_array_free(v_part);
    mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4);
    mlx_array_free(q_n); mlx_array_free(k_n); mlx_array_free(q_s); mlx_array_free(k_s);
    mlx_array_free(A_log); mlx_array_free(dt_bias); mlx_array_free(beta);
    mlx_array_free(a_plus_dt); mlx_array_free(sp_exp); mlx_array_free(sp_1p); mlx_array_free(sp_log);
    mlx_array_free(eAlog); mlx_array_free(neg_exp); mlx_array_free(g);
    mlx_array_free(q_rep); mlx_array_free(k_rep);
    mlx_array_free(conv_state); mlx_array_free(state); mlx_vector_array_free(ys); mlx_array_free(y_stack);
    mlx_array_free(norm_w); mlx_array_free(z4); mlx_array_free(y_norm);
    mlx_array_free(gated); mlx_array_free(gated_flat);
    mlx_array_free(out_w); mlx_array_free(out_sc); mlx_array_free(out_b); mlx_array_free(r);
    mlx_array_free(inv2); mlx_array_free(inv1); mlx_array_free(one_arr); mlx_array_free(zero_arr);
    mlx_vector_array_free(kv_split);
    /* FIX LEAK (2026-09-02): questi handle non erano mai liberati -> ogni forward GDN
     * (prefill o decode) tratteneva le attivazioni (h, qkv, z, av, bv, conv_input,
     * conv_out, sig, silu, neg_g) e gli handle dei pesi caricati -> memoria MLX "active"
     * cresceva di ~GB per eval su qwen3_5 (Jetsam a lungo termine / saturazione
     * imprevedibile). Tutti sono INPUT LAZY a questo punto (i consumer li referenziano
     * gia' nel grafo): liberare l'handle e' sicuro, nessun doppio free (non compaiono
     * nella cleanup esistente). */
    mlx_array_free(ln_w); mlx_array_free(h); mlx_array_free(h_rot);
    mlx_array_free(qkv_w); mlx_array_free(qkv_sc); mlx_array_free(qkv_b);
    mlx_array_free(z_w); mlx_array_free(z_sc); mlx_array_free(z_b);
    mlx_array_free(a_w); mlx_array_free(a_sc); mlx_array_free(a_b);
    mlx_array_free(bw); mlx_array_free(b_sc); mlx_array_free(b_b);
    mlx_array_free(conv_w);
    mlx_array_free(qkv); mlx_array_free(z); mlx_array_free(av); mlx_array_free(bv);
    mlx_array_free(conv_input); mlx_array_free(conv_out);
    mlx_array_free(sig); mlx_array_free(silu); mlx_array_free(neg_g);
    return ok;
}

bool qwen4_linear_gdn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    (void)cache_offset;
    /* §13.5: k_cache[layer] = stato ricorrente GDN S [B,Hv,Dv,Dk]; v_cache[layer] =
     * conv_state [B,kk-1,conv_dim]; has_cache[layer] = true se lo stato esiste gia'
     * (decode che continua il prefill). Riuso i due slot perche' i layer lineari GDN non
     * usano la KV-cache dell'attention (solo i layer full la usano). */
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];
    const int B = 1;
    const int Hk = m->cfg.linear_num_key_heads;
    const int Hv = m->cfg.linear_num_value_heads;
    const int Dk = m->cfg.linear_key_head_dim;
    const int Dv = m->cfg.linear_value_head_dim;
    const int key_dim = Hk * Dk;
    const int value_dim = Hv * Dv;
    const int conv_dim = 2 * key_dim + value_dim;
    const int S = n_tokens;
    mlx_stream s = m->stream;

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen4_linear_gdn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array h = mlx_array_new();
    if (ok) ok = mlx_array_set(&h, x) == 0;

    /* in_proj_qkv / in_proj_z / in_proj_a / in_proj_b (quantizzati) */
    mlx_array qkv_w=mlx_array_new(), qkv_sc=mlx_array_new(), qkv_b=mlx_array_new();
    mlx_array z_w=mlx_array_new(), z_sc=mlx_array_new(), z_b=mlx_array_new();
    mlx_array a_w=mlx_array_new(), a_sc=mlx_array_new(), a_b=mlx_array_new();
    mlx_array bw=mlx_array_new(), b_sc=mlx_array_new(), b_b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_qkv", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &qkv_w, &qkv_sc, &qkv_b);
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_z", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &z_w, &z_sc, &z_b);
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_a", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &a_w, &a_sc, &a_b);
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.in_proj_b", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &bw, &b_sc, &b_b);
    CHK(ok, "in_proj lookups");
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, bw, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] bw dtype=%d dims=(", (int)mlx_array_dtype(bw));
                     for (int k=0;k<mlx_array_ndim(bw);k++) fprintf(stderr,"%d,", (int)mlx_array_dim(bw,k));
                     fprintf(stderr, ") [0:6] ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }

    mlx_array qkv=mlx_array_new(), z=mlx_array_new(), av=mlx_array_new(), bv=mlx_array_new();
    /* FEDELTA' oMLX (oq_a8_gdn_projections): Stage A UNA volta, condivisa fra qkv/z/a/b.
     * oMLX: "qkv e' Q4 mentre z/a/b possono essere Q5, ed e' proprio il caso per cui la
     * Stage A condivisa e' pensata" (Qa non dipende dai bit del peso). */
    bool a8_shared_gdn = false;
    if (ok) {
        mlx_array s_qa, s_sa, s_ra; s_qa.ctx = NULL; s_sa.ctx = NULL; s_ra.ctx = NULL;
        if (llm_mlx_oq_a8_stage_try(h, qkv_w, qkv_sc, qkv_b, gsz, bts, m->cfg.quant_mode, s, &s_qa, &s_sa, &s_ra)) {
            bool k1 = llm_mlx_oq_a8_qmm_stage_try(&qkv, s_qa, s_sa, s_ra, h, qkv_w, qkv_sc, qkv_b, gsz, bts, m->cfg.quant_mode, s);
            bool k2 = k1 && llm_mlx_oq_a8_qmm_stage_try(&z, s_qa, s_sa, s_ra, h, z_w, z_sc, z_b, gsz, bts, m->cfg.quant_mode, s);
            bool k3 = k2 && llm_mlx_oq_a8_qmm_stage_try(&av, s_qa, s_sa, s_ra, h, a_w, a_sc, a_b, gsz, bts, m->cfg.quant_mode, s);
            bool k4 = k3 && llm_mlx_oq_a8_qmm_stage_try(&bv, s_qa, s_sa, s_ra, h, bw, b_sc, b_b, gsz, bts, m->cfg.quant_mode, s);
            if (k4) { a8_shared_gdn = true; }
            else { mlx_array_free(qkv); qkv = mlx_array_new(); mlx_array_free(z); z = mlx_array_new();
                   mlx_array_free(av); av = mlx_array_new(); mlx_array_free(bv); bv = mlx_array_new(); }
            mlx_array_free(s_qa); mlx_array_free(s_sa); mlx_array_free(s_ra);
        }
    }
    if (ok && !a8_shared_gdn) { ok = quantized_linear_cfg(&qkv, h, qkv_w, qkv_sc, qkv_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_qkv"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "in_proj_qkv"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&qkv, h, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    if (ok && !a8_shared_gdn) { ok = quantized_linear_cfg(&z, h, z_w, z_sc, z_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_z"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "in_proj_z"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&z, h, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    if (ok && !a8_shared_gdn) { ok = quantized_linear_cfg(&av, h, a_w, a_sc, a_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_a"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "in_proj_a"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&av, h, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    if (ok && !a8_shared_gdn) { ok = quantized_linear_cfg(&bv, h, bw, b_sc, b_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "in_proj_b"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "in_proj_b"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&bv, h, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }

    /* conv1d depthwise su concat([conv_state, qkv]) */
    mlx_array conv_w = mlx_array_new();
    mlx_array* cw32p = llm_mlx_gdn_conv_cached(m, layer_idx);
    if (cw32p != NULL) {
        ok = ok && mlx_array_set(&conv_w, *cw32p) == 0;
        CHK(ok, "conv1d cast cache");
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.conv1d.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &conv_w);
        CHK(ok, "conv1d lookup");
        /* conv1d.weight e' BF16 (verificato sull'header safetensors); mlx_conv_general richiede
         * input e weight con lo stesso dtype. qkv e' float32 -> cast esplicito del weight. */
        if (ok) {
            mlx_array cw32 = mlx_array_new();
            ok = mlx_astype(&cw32, conv_w, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0;
            if (ok) { mlx_array_free(conv_w); conv_w = cw32; }
            else mlx_array_free(cw32);
            CHK(ok, "conv1d weight cast");
        }
    }
    int kk = m->cfg.linear_conv_kernel_dim;
    int cstate_len = kk - 1;
    int cshape[3] = { B, cstate_len, conv_dim };
    mlx_array conv_state = mlx_array_new();
    /* §13.5: conv_state persistente (v_cache[layer]). Se has_cache e v_cache ha un valore
     * (decode che continua), riuso; altrimenti zero (primo prefill del layer). */
    if (ok) {
        if (has_cache && v_cache && v_cache->ctx != NULL && mlx_array_ndim(*v_cache) > 0) {
            ok = mlx_array_set(&conv_state, *v_cache) == 0;
        } else {
            ok = mlx_zeros(&conv_state, cshape, 3, llm_mlx_act_dt(m->cfg.bfloat16_input), s) == 0;
        }
        CHK(ok, "conv_state init");
    }
    mlx_array conv_input = mlx_array_new();
    /* concat [conv_state, qkv] lungo l'asse 1 (sequenza), come il riferimento
     * mx.concatenate([...], axis=1). mlx_concatenate senza axis unirebbe sull'ultimo
     * asse (conv_dim) -> forma sbagliata. */
    if (ok) { mlx_array cv_arr[2] = { conv_state, qkv }; mlx_vector_array cv = mlx_vector_array_new_data(cv_arr, 2); ok = mlx_concatenate_axis(&conv_input, cv, /*axis=*/1, s) == 0; CHK(ok, "concat conv_state+qkv"); mlx_vector_array_free(cv); }
    /* conv1d depthwise (groups=conv_dim) via mlx_conv_general: mlx_conv1d non supporta
     * groups != 1 (errore reale osservato: "Can only handle groups != 1 in 1D or 2D").
     * Il conv1d GDN e' depthwise (groups=conv_dim=10240, kernel kk), stride 1, padding 0.
     * Weight [conv_dim, kk, 1]; input [B, S+kk-1, conv_dim] -> output [B, S, conv_dim]. */
    mlx_array conv_out = mlx_array_new();
    if (ok) {
        int c_stride[1] = {1};
        int c_pad_lo[1] = {0};
        int c_pad_hi[1] = {0};
        int c_kdil[1] = {1};
        int c_idil[1] = {1};
        ok = mlx_conv_general(&conv_out, conv_input, conv_w, c_stride, 1, c_pad_lo, 1, c_pad_hi, 1,
                              c_kdil, 1, c_idil, 1, /*groups=*/conv_dim, /*flip=*/false, s) == 0;
        CHK(ok, "conv1d (depthwise)");
    }
    /* silu(conv_out) */
    mlx_array sig = mlx_array_new();
    mlx_array silu = mlx_array_new();
    if (ok) { ok = llm_mlx_silu_parity(&silu, &sig, conv_out, s) == 0; CHK(ok, "silu"); }

    /* split in q,k,v per dim (mlx_split_sections: [0,key), [key,2key), [2key,conv)) lungo
     * l'ULTIMO asse (conv_dim), come il riferimento verificato (mx.split(..., -1)). */
    int secs[2] = { key_dim, 2 * key_dim };
    mlx_vector_array kv_split = mlx_vector_array_new();
    mlx_array q_part=mlx_array_new(), k_part=mlx_array_new(), v_part=mlx_array_new();
    if (ok) { ok = mlx_split_sections(&kv_split, silu, secs, 2, /*axis=*/-1, s) == 0; CHK(ok, "split q/k/v"); }
    if (ok) { ok = mlx_vector_array_get(&q_part, kv_split, 0) == 0 &&
                    mlx_vector_array_get(&k_part, kv_split, 1) == 0 &&
                    mlx_vector_array_get(&v_part, kv_split, 2) == 0; CHK(ok, "get q/k/v"); }

    /* reshape a teste: q,k [B,S,Hk,Dk], v [B,S,Hv,Dv] */
    int q_shape[4] = { B, S, Hk, Dk };
    int v_shape[4] = { B, S, Hv, Dv };
    mlx_array q4=mlx_array_new(), k4=mlx_array_new(), v4=mlx_array_new();
    if (ok) { ok = mlx_reshape(&q4, q_part, q_shape, 4, s) == 0 &&
                    mlx_reshape(&k4, k_part, q_shape, 4, s) == 0 &&
                    mlx_reshape(&v4, v_part, v_shape, 4, s) == 0; CHK(ok, "reshape q/k/v"); }
    if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, q4, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] q4 ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
        vf = mlx_array_new();
        if (mlx_astype(&vf, v4, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[gdn] v4 ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }

    /* inv_scale = Dk**-0.5; q = inv^2 * rms_norm(q); k = inv * rms_norm(k). eps=1e-6
     * come il riferimento (mx.fast.rms_norm(..., None, 1e-6)). */
    const double inv_scale = pow((double)Dk, -0.5);
    mlx_array inv1 = llm_mlx_act_scalar((float)inv_scale);
    mlx_array eps6 = llm_mlx_act_scalar(1e-6f);
    mlx_array q_sq=mlx_array_new(), q_ss=mlx_array_new(), q_rs=mlx_array_new(), q_l2=mlx_array_new();
    mlx_array k_sq=mlx_array_new(), k_ss=mlx_array_new(), k_rs=mlx_array_new(), k_l2=mlx_array_new();
    if (ok) ok = mlx_square(&q_sq, q4, s) == 0;
    if (ok) ok = mlx_sum_axis(&q_ss, q_sq, -1, /*keepdims=*/true, s) == 0;
    if (ok) ok = mlx_add(&q_ss, q_ss, eps6, s) == 0;
    if (ok) ok = mlx_rsqrt(&q_rs, q_ss, s) == 0;
    if (ok) ok = mlx_multiply(&q_l2, q4, q_rs, s) == 0;
    if (ok) ok = mlx_square(&k_sq, k4, s) == 0;
    if (ok) ok = mlx_sum_axis(&k_ss, k_sq, -1, /*keepdims=*/true, s) == 0;
    if (ok) ok = mlx_add(&k_ss, k_ss, eps6, s) == 0;
    if (ok) ok = mlx_rsqrt(&k_rs, k_ss, s) == 0;
    if (ok) ok = mlx_multiply(&k_l2, k4, k_rs, s) == 0;
    mlx_array q_s=mlx_array_new(), k_s=mlx_array_new();
    if (ok) { ok = mlx_multiply(&q_s, q_l2, inv1, s) == 0 && mlx_array_set(&k_s, k_l2) == 0; CHK(ok, "l2 scale q/k"); }
    mlx_array_free(q_sq); mlx_array_free(q_rs); mlx_array_free(q_l2);
    mlx_array_free(k_sq); mlx_array_free(k_rs); mlx_array_free(k_l2); mlx_array_free(eps6);

    /* beta = sigmoid(b); g = compute_g(A_log, a, dt_bias) = exp(-exp(A_log)*softplus(a+dt_bias)) */
    mlx_array A_log=mlx_array_new(), dt_bias=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.A_log", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &A_log);
    mlx_array* _dt = llm_mlx_gdn_dt_bias(m, layer_idx);
    if (_dt != NULL) { ok = ok && mlx_array_set(&dt_bias, *_dt) == 0; }
    else {
        tname(m->cfg.layer_prefix, layer_idx, "linear_attn.dt_bias", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &dt_bias);
    }
    CHK(ok, "A_log/dt_bias lookup");
    mlx_array beta=mlx_array_new();
    if (ok) { ok = mlx_sigmoid(&beta, bv, s) == 0; CHK(ok, "beta sigmoid"); }
    /* softplus(a+dt_bias) = logaddexp(a+dt_bias, 0); a: [B,S,Hv], dt_bias: [Hv] -> broadcast */
    mlx_array a_plus_dt=mlx_array_new();
    if (ok) { ok = mlx_add(&a_plus_dt, av, dt_bias, s) == 0; CHK(ok, "a+dt_bias"); }
    mlx_array zero_arr=llm_mlx_act_scalar(0.0f);
    mlx_array sp_log=mlx_array_new();
    if (ok) { ok = mlx_logaddexp(&sp_log, a_plus_dt, zero_arr, s) == 0; CHK(ok, "softplus logaddexp"); }
    mlx_array eAlog=mlx_array_new();
        mlx_array* _ae = llm_mlx_gdn_alog_exp(m, layer_idx);
        if (_ae != NULL) { ok = ok && mlx_array_set(&eAlog, *_ae) == 0; CHK(ok, "exp A_log cache"); }
        else if (ok) { ok = mlx_exp(&eAlog, A_log, s) == 0; CHK(ok, "exp A_log"); }
    mlx_array neg_exp=mlx_array_new();
    if (ok) { ok = mlx_multiply(&neg_exp, eAlog, sp_log, s) == 0; CHK(ok, "exp(A_log)*softplus"); }
    mlx_array neg_g=mlx_array_new();
    if (ok) { ok = mlx_negative(&neg_g, neg_exp, s) == 0; CHK(ok, "negate exp(A_log)*softplus"); }
    mlx_array g=mlx_array_new();
    if (ok) { ok = mlx_exp(&g, neg_g, s) == 0; CHK(ok, "g = exp(-...)"); }

    /* repeat q,k da Hk a Hv teste (Hv//Hk = 3) lungo l'asse teste (-2), come
     * gated_delta_ops (mx.repeat(x, repeat_factor, -2)). */
    const int repeat_factor = Hv / Hk;
    mlx_array q_rep=mlx_array_new(), k_rep=mlx_array_new();
    if (ok) {
        if (repeat_factor > 1) {
            ok = mlx_repeat_axis(&q_rep, q_s, repeat_factor, /*axis=*/2, s) == 0 &&
                 mlx_repeat_axis(&k_rep, k_s, repeat_factor, /*axis=*/2, s) == 0;
        } else {
            ok = mlx_array_set(&q_rep, q_s) == 0 && mlx_array_set(&k_rep, k_s) == 0;
        }
        CHK(ok, "repeat q/k to Hv");
    }

    /* Stato ricorrente S [B,Hv,Dv,Dk] (k_cache[layer]). §13.5: se has_cache e k_cache ha
     * un valore (decode che continua il prefill), riuso lo stato finale del passo
     * precedente; altrimenti zero (primo prefill del layer). Il loop lo evolve poi di S
     * token (prefill: l'intero batch; decode: 1 step). */
    int state_shape[4] = { B, Hv, Dv, Dk };
    mlx_array state = mlx_array_new();
    if (ok) {
        if (has_cache && k_cache && k_cache->ctx != NULL && mlx_array_ndim(*k_cache) > 0) {
            ok = mlx_array_set(&state, *k_cache) == 0;
        } else {
            ok = mlx_zeros(&state, state_shape, 4, MLX_FLOAT32, s) == 0;
        }
        CHK(ok, "state init");
    }

    /* §2026-09-01 KERNEL METAL GDN FUSO: per il PREFILL (S>1) la ricorrenza viene calcolata in
     * un solo kernel Metal (llm_mlx_gdn_prefill, modulo llm_mlx_gdn_kernel.h) invece del loop
     * per-token (~32 chiamate MLX per step, il collo del prefill). Stessa ricorrenza esatta.
     * Se il kernel non e' disponibile (dimensioni/Dk) o fallisce, fallback al loop qui sotto. */
    mlx_array y_stack = mlx_array_new();
    bool kernel_done = false;
    if (ok && S >= 1 && !llm_env_isset("LLM_GDN_KERNEL_OFF")) {
      if (g_spec_capture && S > 1 && S <= LLM_SPEC_CAP_MAXS) {
        /* S5(b): UN SOLO call del kernel FUSO con l'output `state_seq` (stato dopo OGNI
         * token) -> rollback O(1) senza l'overhead di launch per-token. */
        mlx_array st_n = mlx_array_new(), sseq_arr = mlx_array_new();
        if (llm_mlx_gdn_prefill_sseq(&y_stack, &st_n, &sseq_arr, q_s, k_s, v4, g, beta, state, S, 0, s) == 0) {
            (void)mlx_array_set(&state, st_n);
            if (layer_idx < LLM_SPEC_MAXL) {
                (void)mlx_array_set(&g_spec_snap_k[layer_idx], sseq_arr);   /* [B,S,Hv,Dv,Dk] */
                (void)mlx_array_set(&g_spec_snap_v[layer_idx], conv_input); /* [B,cstate_len+S,conv_dim] */
            }
            kernel_done = true;
        }
        mlx_array_free(st_n); mlx_array_free(sseq_arr);
        CHK(ok, "gdn kernel sseq (capture)");
      } else {
        mlx_array state_new_k = mlx_array_new();
        if (llm_mlx_gdn_prefill(&y_stack, &state_new_k, q_s, k_s, v4, g, beta, state, S, /*g_vector=*/0, s) == 0) {
            mlx_array_set(&state, state_new_k);
            kernel_done = true;
        }
        mlx_array_free(state_new_k);
      }
    }
    if (getenv("LLM_GDN_TIME")) {
        struct timespec _g0,_g1; clock_gettime(CLOCK_MONOTONIC,&_g0);
        if (ok) (void)mlx_array_eval(y_stack);
        clock_gettime(CLOCK_MONOTONIC,&_g1);
        fprintf(stderr, "[gdn-time] layer=%d S=%d %.2fms\n", layer_idx, S,
                (_g1.tv_sec-_g0.tv_sec)*1e3+(_g1.tv_nsec-_g0.tv_nsec)/1e6);
    }
    if (llm_mlx_gdn_kernel_dbg_enabled())
        fprintf(stderr, "[gdn-kernel] layer=%d S=%d -> %s\n", layer_idx, S,
                kernel_done ? "KERNEL FUSO" : "LOOP per-token");

    /* Loop ricorrente su S token (gated_delta_update, reference mlx-lm). Per ogni t:
     *   decay = g[...,None,None]; state *= decay
     *   kv_mem = (state * k[...,None,:]).sum(-1)
     *   delta = (v - kv_mem) * beta[...,None]
     *   state += k[...,None,:] * delta[...,None]
     *   y_t   = (state * q[...,None,:]).sum(-1)
     * Raccoglie y_t in un vettore e li stack lungo l'asse 1 -> [B,S,Hv,Dv]. */
    mlx_vector_array ys = mlx_vector_array_new();
    if (!kernel_done) for (int t = 0; ok && t < S; t++) {
        int st0[4] = {0, t, 0, 0};
        int st1[4] = {1, t + 1, Hv, Dk};
        int stv[4] = {1, t + 1, Hv, Dv};
        int ones4[4] = {1, 1, 1, 1};
        int ones3[3] = {1, 1, 1};
        mlx_array q_t=mlx_array_new(), k_t=mlx_array_new(), v_t=mlx_array_new();
        mlx_array g_t=mlx_array_new(), beta_t=mlx_array_new();
        if (ok) {
            ok = mlx_slice(&q_t, q_rep, st0, 4, st1, 4, ones4, 4, s) == 0 &&
                 mlx_slice(&k_t, k_rep, st0, 4, st1, 4, ones4, 4, s) == 0 &&
                 mlx_slice(&v_t, v4, st0, 4, stv, 4, ones4, 4, s) == 0;
            int gt0[3] = {0, t, 0}, gt1[3] = {1, t + 1, Hv};
            ok = ok && mlx_slice(&g_t, g, gt0, 3, gt1, 3, ones3, 3, s) == 0 &&
                       mlx_slice(&beta_t, beta, gt0, 3, gt1, 3, ones3, 3, s) == 0;
        }
        /* squeeze assi 1 (singleton) */
        mlx_array q_sq=mlx_array_new(), k_sq=mlx_array_new(), v_sq=mlx_array_new();
        mlx_array g_sq=mlx_array_new(), beta_sq=mlx_array_new();
        if (ok) {
            ok = mlx_squeeze_axis(&q_sq, q_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&k_sq, k_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&v_sq, v_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&g_sq, g_t, 1, s) == 0 &&
                 mlx_squeeze_axis(&beta_sq, beta_t, 1, s) == 0;
        }
        /* §refactor: il passo ricorrente GDN ora delega al modulo componibile
         * llm_mlx_gated_delta_step (stessa matematica, unica sorgente). */
        mlx_array y_t = mlx_array_new();
        if (ok) ok = llm_mlx_gated_delta_step(&y_t, &state, q_sq, k_sq, v_sq, g_sq, beta_sq, s);
        CHK(ok, "gated_delta_step");
        /* y_t: [B,Hv,Dv] viene stackato DIRETTAMENTE lungo l'asse 1 (stack di array
         * 3D [B,Hv,Dv] su axis=1 -> [B,S,Hv,Dv]). NON espandere prima: espandere a
         * 4D [B,1,Hv,Dv] e poi stack su axis=1 produce 5D [B,S,1,Hv,Dv] (bug reale
         * osservato). */
        if (ok) { ok = mlx_vector_array_append_value(ys, y_t) == 0; CHK(ok, "collect y"); }
        mlx_array_free(q_t); mlx_array_free(k_t); mlx_array_free(v_t);
        mlx_array_free(g_t); mlx_array_free(beta_t);
        mlx_array_free(q_sq); mlx_array_free(k_sq); mlx_array_free(v_sq);
        mlx_array_free(g_sq); mlx_array_free(beta_sq);
        mlx_array_free(y_t);
    }
    if (!kernel_done && ok) { ok = mlx_stack_axis(&y_stack, ys, /*axis=*/1, s) == 0; CHK(ok, "stack y"); }

    /* §13.5: salva lo stato ricorrente finale in k_cache[layer] e il conv_state aggiornato
     * in v_cache[layer], perche' il decode (forward_one_token) continui da dove ha lasciato
     * il prefill. conv_state aggiornato = ultimi cstate_len token di conv_input (come
     * mlx-lm GatedDeltaNet.__call__: cache[0] = conv_input[:, -n_keep:, :]). Eval prima di
     * salvare (state/conv_input sono lazy). */
    if (ok) {
        mlx_array cstate_new = mlx_array_new();
        /* conv_input: [B, cstate_len+S, conv_dim]; prendi gli ultimi cstate_len lungo l'asse 1 */
        int cst0[3] = {0, S, 0};
        int cst1[3] = {1, S + cstate_len, conv_dim};
        int cstrides[3] = {1, 1, 1};
        ok = mlx_slice(&cstate_new, conv_input, cst0, 3, cst1, 3, cstrides, 3, s) == 0;
        if (ok) { mlx_array_free(conv_state); conv_state = cstate_new; }
        else mlx_array_free(cstate_new);
        CHK(ok, "conv_state update");
        /* RMLX/bug P0 (2026-09-03): NIENTE eval per-layer GDN. state/conv_state restano LAZY
         * (refcount) e vengono materializzati dall' eval a fine token/chunk (dipendenza
         * dell'hidden finale); LLM_GDN_EVAL=1 ripristina il vecchio comportamento (2 sync/layer). */
        if (ok && llm_mlx_gdn_eval_forced())
            ok = mlx_array_eval(state) == 0 && mlx_array_eval(conv_state) == 0;
        if (ok) ok = mlx_array_set(k_cache, state) == 0 && mlx_array_set(v_cache, conv_state) == 0;
        CHK(ok, "gdn state store");
    }

    /* RMSNormGated(y, z) = silu(z) * rms_norm(y, norm_w, eps). norm_w = linear_attn.norm.weight [Dv]. */
    mlx_array norm_w=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &norm_w);
    CHK(ok, "norm.weight lookup");
    /* z [B,S,value_dim] -> [B,S,Hv,Dv] */
    int z_shape[4] = { B, S, Hv, Dv };
    mlx_array z4=mlx_array_new();
    if (ok) { ok = mlx_reshape(&z4, z, z_shape, 4, s) == 0; CHK(ok, "reshape z"); }
    mlx_array gated=mlx_array_new();
    if (ok) ok = q4_sigmoid_rmsnorm_gated(&gated, y_stack, z4, norm_w, m->cfg.rms_norm_eps, m->cfg.q4_gate_silu, s);
    CHK(ok, "SigmoidRMSNormGated");
    /* out_proj: [B,S,value_dim] -> [B,S,hidden_size] */
    int flat_shape[3] = { B, S, value_dim };
    mlx_array gated_flat=mlx_array_new();
    if (ok) { ok = mlx_reshape(&gated_flat, gated, flat_shape, 3, s) == 0; CHK(ok, "flatten gated"); }
    mlx_array out_w=mlx_array_new(), out_sc=mlx_array_new(), out_b=mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "linear_attn.out_proj", namebuf, sizeof(namebuf));
    ok = ok && multishard_get_triple(&m->weights, namebuf, &out_w, &out_sc, &out_b);
    CHK(ok, "out_proj lookup");
    mlx_array r=mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&r, gated_flat, out_w, out_sc, out_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "out_proj");
        if (ok && layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
            mlx_array vf = mlx_array_new();
            if (mlx_astype(&vf, r, MLX_FLOAT32, s) == 0 && mlx_array_eval(vf) == 0) {
                const float* d = mlx_array_data_float32(vf);
                if (d) { fprintf(stderr, "[gdn] r_out[:6] ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
            }
            mlx_array_free(vf);
        } }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "out_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&r, gated_flat, m->lora_A[li], m->lora_B[li], m->lora_scale[li], s); }
    mlx_array_set(out_h, r);

    mlx_array_free(q_part); mlx_array_free(k_part); mlx_array_free(v_part);
    mlx_array_free(q4); mlx_array_free(k4); mlx_array_free(v4);
    mlx_array_free(q_ss); mlx_array_free(k_ss); mlx_array_free(q_s); mlx_array_free(k_s);
    mlx_array_free(A_log); mlx_array_free(dt_bias); mlx_array_free(beta);
    mlx_array_free(a_plus_dt); mlx_array_free(sp_log);
    mlx_array_free(eAlog); mlx_array_free(neg_exp); mlx_array_free(g);
    mlx_array_free(q_rep); mlx_array_free(k_rep);
    mlx_array_free(conv_state); mlx_array_free(state); mlx_vector_array_free(ys); mlx_array_free(y_stack);
    mlx_array_free(norm_w); mlx_array_free(z4); mlx_array_free(gated); mlx_array_free(gated_flat);
    mlx_array_free(out_w); mlx_array_free(out_sc); mlx_array_free(out_b); mlx_array_free(r);
    mlx_array_free(inv1); mlx_array_free(zero_arr);
    mlx_vector_array_free(kv_split);
    /* FIX LEAK (2026-09-02): questi handle non erano mai liberati -> ogni forward GDN
     * (prefill o decode) tratteneva le attivazioni (h, qkv, z, av, bv, conv_input,
     * conv_out, sig, silu, neg_g) e gli handle dei pesi caricati -> memoria MLX "active"
     * cresceva di ~GB per eval su qwen3_5 (Jetsam a lungo termine / saturazione
     * imprevedibile). Tutti sono INPUT LAZY a questo punto (i consumer li referenziano
     * gia' nel grafo): liberare l'handle e' sicuro, nessun doppio free (non compaiono
     * nella cleanup esistente). */
    mlx_array_free(h);
    mlx_array_free(qkv_w); mlx_array_free(qkv_sc); mlx_array_free(qkv_b);
    mlx_array_free(z_w); mlx_array_free(z_sc); mlx_array_free(z_b);
    mlx_array_free(a_w); mlx_array_free(a_sc); mlx_array_free(a_b);
    mlx_array_free(bw); mlx_array_free(b_sc); mlx_array_free(b_b);
    mlx_array_free(conv_w);
    mlx_array_free(qkv); mlx_array_free(z); mlx_array_free(av); mlx_array_free(bv);
    mlx_array_free(conv_input); mlx_array_free(conv_out);
    mlx_array_free(sig); mlx_array_free(silu); mlx_array_free(neg_g);
    return ok;
}

