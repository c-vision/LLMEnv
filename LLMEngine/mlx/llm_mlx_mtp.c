/* ============================================================================
 * llm_mlx_mtp.c — dominio MTP head (MTPLX / qwen3-next-mtp).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * MTP head (MTPLX / qwen3-next-mtp). Rif: Rapid-MLX vllm_mlx/patches/qwen3_next_mtp.py.
 * x = fc(cat([norm_emb(embed(t)), norm_hid(hidden)])) -> 1 layer full-attention
 * (q+gate, q/k norm, GQA) -> norm -> lm_head. depth-1: 1 posizione, attn=v.
 * ============================================================================ */

/* Copia un array MTP dallo stream CPU (dove l'ha messo mlx_load_safetensors, che
 * non supporta lo stream GPU) alla GPU, cosi' il forward MTP gira su GPU. */
static void llm_mlx_mtp_to_gpu(mlx_array* a, mlx_stream s) {
    if (a == NULL || a->ctx == NULL) return;
    mlx_array g = mlx_array_new();
    int rc = mlx_copy(&g, *a, s);
    if (rc == 0) { (void)mlx_array_eval(g); mlx_array_free(*a); *a = g; }
    else mlx_array_free(g);
}

static void llm_mlx_mtp_take(mlx_array* dst, mlx_map_string_to_array map, const char* name) {
    *dst = mlx_array_new();
    (void)mlx_map_string_to_array_get(dst, map, name);
}
static void llm_mlx_mtp_take3(mlx_array* w, mlx_array* sc, mlx_array* bi, mlx_map_string_to_array map, const char* base) {
    char nm[256];
    snprintf(nm, sizeof(nm), "%s.weight", base); llm_mlx_mtp_take(w, map, nm);
    snprintf(nm, sizeof(nm), "%s.scales", base); llm_mlx_mtp_take(sc, map, nm);
    snprintf(nm, sizeof(nm), "%s.biases", base); llm_mlx_mtp_take(bi, map, nm);
}
LlmMlxMtp* llm_mlx_mtp_get(LlmMlxModel* m) {
    static LlmMlxMtp s_mtp; static int s_state = 0;   /* 0=untried 1=ok 2=no */
    if (s_state) return s_state == 1 ? &s_mtp : NULL;
    s_state = 2;
    if (m == NULL || m->model_dir[0] == '\0') return NULL;
    char path[1200]; snprintf(path, sizeof(path), "%s/mtp.safetensors", m->model_dir);
    FILE* f = fopen(path, "rb"); if (!f) return NULL; fclose(f);
    mlx_map_string_to_array map = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    /* NB: `mlx_load_safetensors` NON supporta lo stream GPU ("Load::eval_gpu Not
     * implemented") -> i pesi MTP restano in memoria CPU. E' una delle cause del costo
     * del forward MTP (~35ms, ≈ un token target); per portarli su GPU serve una copia
     * esplicita via op su m->stream (da fare col verify-path veloce, leva S5). */
    { mlx_stream cst = mlx_default_cpu_stream_new();
      int lrc = mlx_load_safetensors(&map, &meta, path, cst);
      mlx_stream_free(cst);
      if (lrc != 0) { mlx_map_string_to_string_free(meta); mlx_map_string_to_array_free(map); return NULL; } }
    mlx_map_string_to_string_free(meta);
    llm_mlx_mtp_take3(&s_mtp.fc_w,&s_mtp.fc_s,&s_mtp.fc_b, map, "mtp.fc");
    llm_mlx_mtp_take(&s_mtp.pne, map, "mtp.pre_fc_norm_embedding.weight");
    llm_mlx_mtp_take(&s_mtp.pnh, map, "mtp.pre_fc_norm_hidden.weight");
    llm_mlx_mtp_take(&s_mtp.iln, map, "mtp.layers.0.input_layernorm.weight");
    llm_mlx_mtp_take(&s_mtp.pln, map, "mtp.layers.0.post_attention_layernorm.weight");
    llm_mlx_mtp_take(&s_mtp.qn,  map, "mtp.layers.0.self_attn.q_norm.weight");
    llm_mlx_mtp_take(&s_mtp.kn,  map, "mtp.layers.0.self_attn.k_norm.weight");
    llm_mlx_mtp_take(&s_mtp.mnorm, map, "mtp.norm.weight");
    llm_mlx_mtp_take3(&s_mtp.q_w,&s_mtp.q_s,&s_mtp.q_b, map, "mtp.layers.0.self_attn.q_proj");
    llm_mlx_mtp_take3(&s_mtp.k_w,&s_mtp.k_s,&s_mtp.k_b, map, "mtp.layers.0.self_attn.k_proj");
    llm_mlx_mtp_take3(&s_mtp.v_w,&s_mtp.v_s,&s_mtp.v_b, map, "mtp.layers.0.self_attn.v_proj");
    llm_mlx_mtp_take3(&s_mtp.o_w,&s_mtp.o_s,&s_mtp.o_b, map, "mtp.layers.0.self_attn.o_proj");
    llm_mlx_mtp_take3(&s_mtp.g_w,&s_mtp.g_s,&s_mtp.g_b, map, "mtp.layers.0.mlp.gate_proj");
    llm_mlx_mtp_take3(&s_mtp.u_w,&s_mtp.u_s,&s_mtp.u_b, map, "mtp.layers.0.mlp.up_proj");
    llm_mlx_mtp_take3(&s_mtp.d_w,&s_mtp.d_s,&s_mtp.d_b, map, "mtp.layers.0.mlp.down_proj");
    mlx_map_string_to_array_free(map);
    if (s_mtp.fc_w.ctx == NULL || s_mtp.q_w.ctx == NULL || s_mtp.d_w.ctx == NULL || s_mtp.mnorm.ctx == NULL) return NULL;
    /* Porta i pesi MTP su GPU (mlx_load_safetensors li lascia su CPU). */
    { mlx_array* ws[] = { &s_mtp.fc_w,&s_mtp.fc_s,&s_mtp.fc_b, &s_mtp.pne,&s_mtp.pnh,&s_mtp.iln,&s_mtp.pln,
        &s_mtp.qn,&s_mtp.kn,&s_mtp.mnorm, &s_mtp.q_w,&s_mtp.q_s,&s_mtp.q_b, &s_mtp.k_w,&s_mtp.k_s,&s_mtp.k_b,
        &s_mtp.v_w,&s_mtp.v_s,&s_mtp.v_b, &s_mtp.o_w,&s_mtp.o_s,&s_mtp.o_b, &s_mtp.g_w,&s_mtp.g_s,&s_mtp.g_b,
        &s_mtp.u_w,&s_mtp.u_s,&s_mtp.u_b, &s_mtp.d_w,&s_mtp.d_s,&s_mtp.d_b };
      for (size_t i = 0; i < sizeof(ws)/sizeof(ws[0]); i++) llm_mlx_mtp_to_gpu(ws[i], m->stream); }
    /* Contract MTPLX (config.json mtplx_mtp_contract): `hidden_variant`
     * = post_norm -> l'hidden in input e' il POST-final-norm del target (si
     * ricostruisce applicando final_norm_w); pre_norm -> hidden grezzo. Il
     * modello pubblicato Qwen3.8-27B-MTPLX dichiara post_norm (verificato:
     * draft==greedy al depth-1). Env override per il debug. */
    s_mtp.post_norm = 1;
    { char cp[1200]; snprintf(cp, sizeof(cp), "%s/config.json", m->model_dir);
      FILE* cf = fopen(cp, "rb");
      if (cf) { if (fseek(cf,0,SEEK_END)==0) { long sz=ftell(cf); rewind(cf);
                  if (sz>0 && sz<(1<<22)) { char* b=(char*)malloc((size_t)sz+1);
                    if (b) { size_t rd=fread(b,1,(size_t)sz,cf); b[rd]='\0';
                      const char* hv=strstr(b,"hidden_variant");
                      if (hv) { const char* pn=strstr(hv,"post_norm"), *pr=strstr(hv,"pre_norm");
                                if (pr && (!pn || pr<pn)) s_mtp.post_norm=0; }
                      free(b); } } }
                fclose(cf); } }
    if (llm_ai_mlx_opt_env("LLM_MTP_POSTNORM")) s_mtp.post_norm = 1;
    if (llm_ai_mlx_opt_env("LLM_MTP_PRENORM"))  s_mtp.post_norm = 0;
    s_mtp.loaded = true; s_state = 1;
    return &s_mtp;
}

/* Un passo del modulo MTP con CONTESTO DI BLOCCO. from_target=1 al primo passo
 * (hidden = post-final-norm del target, contract hidden_variant=post_norm); 0 ai passi
 * successivi (hidden = output del MTP gia' normato con mtp.norm). pos = posizione LOCALE
 * nel blocco (RoPE). k_cache/v_cache = KV del blocco (aggiornate in-place). out_hidden
 * (opzionale) = hidden post mtp.norm per il passo successivo. Ritorna il token draftato. */
static int32_t llm_mlx_mtp_step_impl(LlmMlxModel* m, LlmMlxMtp* q, OutputTensors* tp,
                                     int32_t last_token, mlx_array hidden, int from_target, int pos,
                                     mlx_array* k_cache, mlx_array* v_cache, mlx_array* out_hidden) {
    OutputTensors t = *tp;
    mlx_stream s = m->stream;
    const float eps = m->cfg.rms_norm_eps;
    const int H = m->cfg.hidden_size, NH = m->cfg.n_heads, NKV = m->cfg.n_kv_heads, HD = m->cfg.head_dim;
    const char* mode = m->cfg.quant_mode;
    bool ok = true;
    mlx_array emb = mlx_array_new();
    ok = embed_tokens_batch(m, t.embed_w, t.embed_s, t.embed_b, &last_token, 1, &emb);
    mlx_array he=mlx_array_new(), hh=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&he, emb, q->pne, eps, s);
    if (ok && from_target && q->post_norm) {
        /* contract hidden_variant=post_norm: hidden = post-final-norm del target
         * (ricostruito dal pre-norm pending_h applicando final_norm_w). */
        mlx_array hn = mlx_array_new();
        if (rms_norm_cfg(&hn, hidden, t.final_norm_w, eps, s)) ok = rms_norm_cfg(&hh, hn, q->pnh, eps, s);
        else ok = false;
        mlx_array_free(hn);
    } else if (ok) ok = rms_norm_cfg(&hh, hidden, q->pnh, eps, s);
    mlx_array cat=mlx_array_new();
    if (ok) { mlx_array cv[2]; if (llm_ai_mlx_opt_env("LLM_MTP_HIDFIRST")) { cv[0]=hh; cv[1]=he; } else { cv[0]=he; cv[1]=hh; }
              mlx_vector_array v=mlx_vector_array_new_data(cv,2); ok = mlx_concatenate_axis(&cat, v, -1, s)==0; mlx_vector_array_free(v); }
    mlx_array x=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&x, cat, q->fc_w,q->fc_s,q->fc_b, 64,4,mode, s);
    mlx_array res=mlx_array_new();
    if (ok) ok = mlx_array_set(&res, x)==0;
    mlx_array xn=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&xn, x, q->iln, eps, s);
    mlx_array qg=mlx_array_new(), kk=mlx_array_new(), vv=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&qg, xn, q->q_w,q->q_s,q->q_b, 64,4,mode, s);
    if (ok) ok = quantized_linear_cfg(&kk, xn, q->k_w,q->k_s,q->k_b, 64,4,mode, s);
    if (ok) ok = quantized_linear_cfg(&vv, xn, q->v_w,q->v_s,q->v_b, 64,4,mode, s);
    int hn = NH*HD;   /* 6144 */
    /* Output gate (Qwen3NextAttention, come il blocco full-attn del motore):
     * q_proj = [1,1,nh*head_dim*2] -> reshape [1,1,nh,2*hd] -> split axis -1 in
     * (queries, gate): INTERLEAVED per testa, NON due meta' piatte. Bug fix 2026-09-21. */
    mlx_array qfull=mlx_array_new(), gate=mlx_array_new();
    if (ok) {
        int q4[4]={1,1,NH,2*HD}; mlx_array qresh=mlx_array_new();
        ok = mlx_reshape(&qresh, qg, q4,4,s)==0;
        int sec[2]={HD,2*HD}; mlx_vector_array qsp=mlx_vector_array_new();
        if (ok) ok = mlx_split_sections(&qsp, qresh, sec,2,-1,s)==0;
        if (ok) ok = mlx_vector_array_get(&qfull,qsp,0)==0 && mlx_vector_array_get(&gate,qsp,1)==0;
        mlx_vector_array_free(qsp); mlx_array_free(qresh);
    }
    mlx_array qr=mlx_array_new(), kr=mlx_array_new();
    if (ok) { int sh[3]={1,NH,HD}, sk[3]={1,NKV,HD}; ok = mlx_reshape(&qr, qfull, sh,3,s)==0 && mlx_reshape(&kr, kk, sk,3,s)==0; }
    mlx_array qn=mlx_array_new(), kn=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&qn, qr, q->qn, eps, s);
    if (ok) ok = rms_norm_cfg(&kn, kr, q->kn, eps, s);
    /* RoPE alla posizione LOCALE `pos` (Qwen3Next full-attn, 4D [1,H,1,D]). */
    mlx_array q4=mlx_array_new(), k4=mlx_array_new(), v3=mlx_array_new(), v4=mlx_array_new();
    if (ok) { int a4[4]={1,NH,1,HD}, b4[4]={1,NKV,1,HD}, c3[3]={1,NKV,HD}, c4[4]={1,NKV,1,HD};
              ok = mlx_reshape(&q4, qn, a4,4,s)==0 && mlx_reshape(&k4, kn, b4,4,s)==0 &&
                   mlx_reshape(&v3, vv, c3,3,s)==0 && mlx_reshape(&v4, v3, c4,4,s)==0; }
    if (ok) { mlx_optional_float rbase={m->cfg.rope_theta,true}; mlx_array nf=mlx_array_new();
              ok = mlx_fast_rope(&q4, q4, HD, false, rbase, 1.0f, pos, nf, s)==0 &&
                   mlx_fast_rope(&k4, k4, HD, false, rbase, 1.0f, pos, nf, s)==0;
              mlx_array_free(nf); }
    /* Appendi k/v alla cache del BLOCCO (asse 2 = posizioni) e SDPA su [0..pos]
     * (causale implicita: q e' alla posizione pos, k copre 0..pos). */
    if (ok) {
        if (k_cache->ctx == NULL || mlx_array_size(*k_cache) == 0) {
            ok = mlx_array_set(k_cache, k4)==0 && mlx_array_set(v_cache, v4)==0;
        } else {
            mlx_vector_array kc=mlx_vector_array_new_data((mlx_array[]){*k_cache,k4},2);
            mlx_vector_array vc=mlx_vector_array_new_data((mlx_array[]){*v_cache,v4},2);
            mlx_array nk=mlx_array_new(), nv=mlx_array_new();
            ok = mlx_concatenate_axis(&nk,kc,2,s)==0 && mlx_concatenate_axis(&nv,vc,2,s)==0;
            if (ok) ok = mlx_array_set(k_cache,nk)==0 && mlx_array_set(v_cache,nv)==0;
            mlx_array_free(nk); mlx_array_free(nv); mlx_vector_array_free(kc); mlx_vector_array_free(vc);
        }
    }
    mlx_array attn4=mlx_array_new(), attn=mlx_array_new();
    if (ok) { mlx_array nomask=mlx_array_new(), nosink=mlx_array_new();
              ok = mlx_fast_scaled_dot_product_attention(&attn4, q4, *k_cache, *v_cache,
                     1.0f/sqrtf((float)HD), "", nomask, nosink, s)==0;
              mlx_array_free(nomask); mlx_array_free(nosink); }
    mlx_array gr=mlx_array_new();
    if (ok) { int sg[3]={1,NH,HD}; ok = mlx_reshape(&attn, attn4, sg,3,s)==0 && mlx_reshape(&gr, gate, sg,3,s)==0; }
    mlx_array gs=mlx_array_new(); if (ok) ok = mlx_sigmoid(&gs, gr, s)==0;
    mlx_array oin=mlx_array_new(); if (ok) ok = mlx_multiply(&oin, attn, gs, s)==0;
    mlx_array oif=mlx_array_new(); if (ok) { int so[3]={1,1,hn}; ok = mlx_reshape(&oif, oin, so,3,s)==0; }
    mlx_array oo=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&oo, oif, q->o_w,q->o_s,q->o_b, 64,4,mode, s);
    mlx_array x2=mlx_array_new(); if (ok) ok = mlx_add(&x2, res, oo, s)==0;
    
    mlx_array xn2=mlx_array_new(); if (ok) ok = rms_norm_cfg(&xn2, x2, q->pln, eps, s);
    mlx_array gg=mlx_array_new(), uu=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&gg, xn2, q->g_w,q->g_s,q->g_b, 64,4,mode, s);
    if (ok) ok = quantized_linear_cfg(&uu, xn2, q->u_w,q->u_s,q->u_b, 64,4,mode, s);
    mlx_array sg2=mlx_array_new(); if (ok) ok = mlx_sigmoid(&sg2, gg, s)==0;
    mlx_array sil=mlx_array_new(); if (ok) ok = mlx_multiply(&sil, gg, sg2, s)==0;
    mlx_array act=mlx_array_new(); if (ok) ok = mlx_multiply(&act, sil, uu, s)==0;
    mlx_array dn=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&dn, act, q->d_w,q->d_s,q->d_b, 64,4,mode, s);
    mlx_array x3=mlx_array_new(); if (ok) ok = mlx_add(&x3, x2, dn, s)==0;
    
    mlx_array fn=mlx_array_new(); if (ok) ok = rms_norm_cfg(&fn, x3, q->mnorm, eps, s);
    /* lm_head: bits derivati (8) dai tensori del target */
    struct timespec _tl0; clock_gettime(CLOCK_MONOTONIC,&_tl0);
    int ob=0, og=0; char omode[16];
    llm_mlx_quant_derive(t.out_w, t.out_s, m->cfg.quant_group_size, m->cfg.quant_bits, mode, &og,&ob,omode,sizeof(omode));
    mlx_array logits=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&logits, fn, t.out_w,t.out_s,t.out_b, og,ob,omode, s);
    
    mlx_array am=mlx_array_new(); if (ok) ok = mlx_argmax_axis(&am, logits, -1, false, s)==0;
    int32_t out = -1;
    if (ok && mlx_array_eval(am)==0) { const uint32_t* d=mlx_array_data_uint32(am); if (d) out=(int32_t)d[0]; else out=-2; }
    
    if (ok && out_hidden) { (void)mlx_array_set(out_hidden, fn); }
    mlx_array_free(emb);mlx_array_free(he);mlx_array_free(hh);mlx_array_free(cat);mlx_array_free(x);
    mlx_array_free(res);mlx_array_free(xn);mlx_array_free(qg);mlx_array_free(kk);mlx_array_free(vv);
    mlx_array_free(qfull);mlx_array_free(gate);mlx_array_free(qr);mlx_array_free(kr);mlx_array_free(qn);
    mlx_array_free(kn);mlx_array_free(q4);mlx_array_free(k4);mlx_array_free(v3);mlx_array_free(v4);mlx_array_free(attn4);mlx_array_free(attn);mlx_array_free(gr);mlx_array_free(gs);
    mlx_array_free(oin);mlx_array_free(oif);mlx_array_free(oo);mlx_array_free(x2);mlx_array_free(xn2);
    mlx_array_free(gg);mlx_array_free(uu);mlx_array_free(sg2);mlx_array_free(sil);mlx_array_free(act);
    mlx_array_free(dn);mlx_array_free(x3);mlx_array_free(fn);mlx_array_free(logits);mlx_array_free(am);
    return out;
}

/* depth-1 BATCHED su N sessioni (K=1): UN forward MTP per N richieste. Posizione 0 per
 * tutte -> RoPE = identita'; attenzione self -> attn = v per riga. */
static int llm_mlx_mtp_draft1_batch_impl(LlmMlxModel* m, LlmMlxMtp* q, OutputTensors* tp,
                                         const int32_t* toks, int N, mlx_array hidden_cat, int32_t* out) {
    OutputTensors t = *tp;
    mlx_stream s = m->stream;
    const float eps = m->cfg.rms_norm_eps;
    const int NH = m->cfg.n_heads, NKV = m->cfg.n_kv_heads, HD = m->cfg.head_dim;
    const char* mode = m->cfg.quant_mode;
    bool ok = true;
    mlx_array emb = mlx_array_new();
    ok = embed_tokens_batch(m, t.embed_w, t.embed_s, t.embed_b, toks, N, &emb);
    mlx_array he=mlx_array_new(), hh=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&he, emb, q->pne, eps, s);
    if (ok && q->post_norm) {
        mlx_array hn = mlx_array_new();
        if (rms_norm_cfg(&hn, hidden_cat, t.final_norm_w, eps, s)) ok = rms_norm_cfg(&hh, hn, q->pnh, eps, s);
        else ok = false;
        mlx_array_free(hn);
    } else if (ok) ok = rms_norm_cfg(&hh, hidden_cat, q->pnh, eps, s);
    mlx_array cat=mlx_array_new();
    if (ok) { mlx_array cv[2]={he,hh}; mlx_vector_array v=mlx_vector_array_new_data(cv,2); ok = mlx_concatenate_axis(&cat, v, -1, s)==0; mlx_vector_array_free(v); }
    mlx_array x=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&x, cat, q->fc_w,q->fc_s,q->fc_b, 64,4,mode, s);
    mlx_array res=mlx_array_new(); if (ok) ok = mlx_array_set(&res,x)==0;
    mlx_array xn=mlx_array_new(); if (ok) ok = rms_norm_cfg(&xn, x, q->iln, eps, s);
    mlx_array qg=mlx_array_new(), kk=mlx_array_new(), vv=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&qg, xn, q->q_w,q->q_s,q->q_b, 64,4,mode, s);
    if (ok) ok = quantized_linear_cfg(&kk, xn, q->k_w,q->k_s,q->k_b, 64,4,mode, s);
    if (ok) ok = quantized_linear_cfg(&vv, xn, q->v_w,q->v_s,q->v_b, 64,4,mode, s);
    const int hn_ = NH*HD;
    mlx_array qfull=mlx_array_new(), gate=mlx_array_new();
    if (ok) {
        int q4[4]={1,N,NH,2*HD}; mlx_array qresh=mlx_array_new();
        ok = mlx_reshape(&qresh, qg, q4,4,s)==0;
        int sec[2]={HD,2*HD}; mlx_vector_array qsp=mlx_vector_array_new();
        if (ok) ok = mlx_split_sections(&qsp, qresh, sec,2,-1,s)==0;
        if (ok) ok = mlx_vector_array_get(&qfull,qsp,0)==0 && mlx_vector_array_get(&gate,qsp,1)==0;
        mlx_vector_array_free(qsp); mlx_array_free(qresh);
    }
    mlx_array qr=mlx_array_new(), kr=mlx_array_new();
    if (ok) { int sh[4]={1,N,NH,HD}, sk[4]={1,N,NKV,HD}; ok = mlx_reshape(&qr,qfull,sh,4,s)==0 && mlx_reshape(&kr,kk,sk,4,s)==0; }
    mlx_array qn=mlx_array_new(), kn=mlx_array_new();
    if (ok) ok = rms_norm_cfg(&qn, qr, q->qn, eps, s);
    if (ok) ok = rms_norm_cfg(&kn, kr, q->kn, eps, s);
    mlx_array vr=mlx_array_new(), v4=mlx_array_new();
    if (ok) { int sv[4]={1,N,NKV,HD}; ok = mlx_reshape(&vr, vv, sv,4,s)==0; }
    if (ok) ok = (NH==NKV) ? (mlx_array_set(&v4,vr)==0) : (mlx_repeat_axis(&v4, vr, NH/NKV, 2, s)==0);
    mlx_array gs=mlx_array_new(); if (ok) ok = mlx_sigmoid(&gs, gate, s)==0;
    mlx_array oin=mlx_array_new(); if (ok) ok = mlx_multiply(&oin, v4, gs, s)==0;
    mlx_array oif=mlx_array_new(); if (ok) { int so[3]={1,N,hn_}; ok = mlx_reshape(&oif, oin, so,3,s)==0; }
    mlx_array oo=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&oo, oif, q->o_w,q->o_s,q->o_b, 64,4,mode, s);
    mlx_array x2=mlx_array_new(); if (ok) ok = mlx_add(&x2, res, oo, s)==0;
    mlx_array xn2=mlx_array_new(); if (ok) ok = rms_norm_cfg(&xn2, x2, q->pln, eps, s);
    mlx_array gg=mlx_array_new(), uu=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&gg, xn2, q->g_w,q->g_s,q->g_b, 64,4,mode, s);
    if (ok) ok = quantized_linear_cfg(&uu, xn2, q->u_w,q->u_s,q->u_b, 64,4,mode, s);
    mlx_array sg2=mlx_array_new(); if (ok) ok = mlx_sigmoid(&sg2, gg, s)==0;
    mlx_array sil=mlx_array_new(); if (ok) ok = mlx_multiply(&sil, gg, sg2, s)==0;
    mlx_array act=mlx_array_new(); if (ok) ok = mlx_multiply(&act, sil, uu, s)==0;
    mlx_array dn=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&dn, act, q->d_w,q->d_s,q->d_b, 64,4,mode, s);
    mlx_array x3=mlx_array_new(); if (ok) ok = mlx_add(&x3, x2, dn, s)==0;
    mlx_array fn=mlx_array_new(); if (ok) ok = rms_norm_cfg(&fn, x3, q->mnorm, eps, s);
    int ob=0, og=0; char omode[16];
    llm_mlx_quant_derive(t.out_w, t.out_s, m->cfg.quant_group_size, m->cfg.quant_bits, mode, &og,&ob,omode,sizeof(omode));
    mlx_array logits=mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&logits, fn, t.out_w,t.out_s,t.out_b, og,ob,omode, s);
    mlx_array am=mlx_array_new(); if (ok) ok = mlx_argmax_axis(&am, logits, -1, false, s)==0;
    if (ok && mlx_array_eval(am)==0) {
        const uint32_t* d=mlx_array_data_uint32(am);
        if (d) for (int i=0;i<N;i++) out[i]=(int32_t)d[i]; else ok=false;
    } else ok=false;
    mlx_array_free(emb);mlx_array_free(he);mlx_array_free(hh);mlx_array_free(cat);mlx_array_free(x);
    mlx_array_free(res);mlx_array_free(xn);mlx_array_free(qg);mlx_array_free(kk);mlx_array_free(vv);
    mlx_array_free(qfull);mlx_array_free(gate);mlx_array_free(qr);mlx_array_free(kr);mlx_array_free(qn);
    mlx_array_free(kn);mlx_array_free(vr);mlx_array_free(v4);mlx_array_free(gs);mlx_array_free(oin);
    mlx_array_free(oif);mlx_array_free(oo);mlx_array_free(x2);mlx_array_free(xn2);mlx_array_free(gg);
    mlx_array_free(uu);mlx_array_free(sg2);mlx_array_free(sil);mlx_array_free(act);mlx_array_free(dn);
    mlx_array_free(x3);mlx_array_free(fn);mlx_array_free(logits);mlx_array_free(am);
    return ok ? 0 : -1;
}

/* API: draft depth-1 BATCHED su N sessioni (K=1). Scrive N id in out_drafts. */
int llm_ai_mlx_mtp_draft1_multi(llm_ai_mlx_session_handle* handles, int N, int32_t* out_drafts) {
    if (handles == NULL || N <= 0 || out_drafts == NULL || N > 32) return -1;
    LlmMlxSession* s0 = (LlmMlxSession*)handles[0];
    if (s0 == NULL || s0->model == NULL) return -1;
    LlmMlxModel* m = s0->model;
    LlmMlxMtp* q = llm_mlx_mtp_get(m);
    if (q == NULL) return -2;
    OutputTensors t; if (!load_output_tensors(m, &t)) return -1;
    int32_t toks[32]; mlx_array hv[32];
    for (int i = 0; i < N; i++) {
        LlmMlxSession* s = (LlmMlxSession*)handles[i];
        if (s == NULL || s->model != m || !s->has_pending || s->pending_h.ctx == NULL) { free_output_tensors(&t); return -1; }
        toks[i] = s->pending_next_id; hv[i] = s->pending_h;
    }
    mlx_vector_array v = mlx_vector_array_new_data(hv, N);
    mlx_array cat = mlx_array_new();
    int rc = mlx_concatenate_axis(&cat, v, 1, m->stream);
    mlx_vector_array_free(v);
    if (rc == 0) rc = llm_mlx_mtp_draft1_batch_impl(m, q, &t, toks, N, cat, out_drafts);
    mlx_array_free(cat);
    free_output_tensors(&t);
    return rc == 0 ? 0 : -3;
}

/* depth-1 (compat): un solo passo, nessun contesto di blocco. */
static int32_t llm_mlx_mtp_draft1_impl(LlmMlxModel* m, int32_t last_token, mlx_array hidden) {
    LlmMlxMtp* q = llm_mlx_mtp_get(m);
    if (q == NULL) return -1;
    struct timespec _w0,_w1,_w2; clock_gettime(CLOCK_MONOTONIC,&_w0);
    OutputTensors t; if (!load_output_tensors(m, &t)) return -1;
    clock_gettime(CLOCK_MONOTONIC,&_w1);
    mlx_array kc=mlx_array_new(), vc=mlx_array_new();
    int32_t out = llm_mlx_mtp_step_impl(m, q, &t, last_token, hidden, 1, 0, &kc, &vc, NULL);
    clock_gettime(CLOCK_MONOTONIC,&_w2);
    if (0) fprintf(stderr,"[mtp-wrap] load=%.1fms step=%.1fms\n",
        (_w1.tv_sec-_w0.tv_sec)*1e3+(_w1.tv_nsec-_w0.tv_nsec)/1e6,
        (_w2.tv_sec-_w1.tv_sec)*1e3+(_w2.tv_nsec-_w1.tv_nsec)/1e6);
    mlx_array_free(kc); mlx_array_free(vc);
    free_output_tensors(&t);
    return out;
}

/* depth-K: drafta K token riusando l'MTP in ricorsione (blocco [0..K-1], RoPE locale). */
static int llm_mlx_mtp_draftk_impl(LlmMlxModel* m, int32_t last_token, mlx_array hidden, int K, int32_t* out) {
    LlmMlxMtp* q = llm_mlx_mtp_get(m);
    if (q == NULL) return -1;
    OutputTensors t; if (!load_output_tensors(m, &t)) return -1;
    mlx_array kc=mlx_array_new(), vc=mlx_array_new();
    mlx_array cur=mlx_array_new(), nxt=mlx_array_new();
    (void)mlx_array_set(&cur, hidden);
    int32_t tok = last_token; int rc = 0;
    for (int i = 0; i < K; i++) {
        int32_t d = llm_mlx_mtp_step_impl(m, q, &t, tok, cur, i==0, i, &kc, &vc, &nxt);
        if (d < 0) { rc = -1; break; }
        out[i] = d;
        (void)mlx_array_set(&cur, nxt);
        tok = d;
    }
    mlx_array_free(kc); mlx_array_free(vc); mlx_array_free(cur); mlx_array_free(nxt);
    free_output_tensors(&t);
    return rc;
}

int llm_ai_mlx_session_pending_id(llm_ai_mlx_session_handle sh, int32_t* out) {
    LlmMlxSession* ss = (LlmMlxSession*)sh;
    if (ss == NULL || out == NULL || !ss->has_pending) return -1;
    *out = ss->pending_next_id; return 0;
}

int llm_ai_mlx_mtp_draft1(llm_ai_mlx_session_handle sh, int32_t* out_draft, int32_t* out_greedy) {
    LlmMlxSession* ss = (LlmMlxSession*)sh;
    if (ss == NULL || ss->model == NULL) return -1;
    LlmMlxModel* m = ss->model;
    if (out_greedy) *out_greedy = ss->pending_next_id;
    if (ss->history == NULL || ss->history_len <= 0 || ss->pending_h.ctx == NULL) return -2;
    const char* emb = llm_ai_mlx_opt_env("LLM_MTP_EMBED");
    int32_t last = (emb != NULL && strcmp(emb, "last") == 0) ? ss->history[ss->history_len-1] : ss->pending_next_id;
    int32_t d = llm_mlx_mtp_draft1_impl(m, last, ss->pending_h);
    if (out_draft) *out_draft = d;
    return d >= 0 ? 0 : -3;
}

/* Drafta K token col head MTP (ricorrenza sul blocco [0..K-1] con RoPE locale). */
int llm_ai_mlx_mtp_draftk(llm_ai_mlx_session_handle sh, int K, int32_t* out_drafts, int32_t* out_greedy) {
    LlmMlxSession* ss = (LlmMlxSession*)sh;
    if (ss == NULL || ss->model == NULL || out_drafts == NULL || K <= 0) return -1;
    if (out_greedy) *out_greedy = ss->pending_next_id;
    if (ss->history == NULL || ss->history_len <= 0 || ss->pending_h.ctx == NULL) return -2;
    return llm_mlx_mtp_draftk_impl(ss->model, ss->pending_next_id, ss->pending_h, K, out_drafts) == 0 ? 0 : -3;
}

/* MTP speculative generate (schema MTPLX M(K+1)=[primary, draft_0..draft_{K-1}]).
 * Per round: emette il primary (pending_next_id), drafta K token col head MTP, verifica
 * il blocco in UN forward batch e usa il BONUS dagli stessi logits (nessun forward di
 * commit). Il token pending (`next`) resta in sessione ed e' emesso al giro successivo.
 * Output IDENTICO all'AR greedy (parita' algoritmica). Depth da LLM_MTP_DEPTH (1..8, def 1).
 * Ritorna #token emessi, -1 errore, -2 modello senza head MTP. */
int llm_ai_mlx_session_generate_mtp(llm_ai_mlx_session_handle sh, int max_new_tokens,
                                    int32_t eos_token_id, int32_t* out_ids) {
    LlmMlxSession* s = (LlmMlxSession*)sh;
    if (s == NULL || s->model == NULL || out_ids == NULL || max_new_tokens <= 0) return -1;
    if (llm_mlx_mtp_get(s->model) == NULL) return -2;
    int32_t* drafts = (int32_t*)malloc(sizeof(int32_t) * 8);
    int32_t* blk = (int32_t*)malloc(sizeof(int32_t) * 9);
    if (!drafts || !blk) { free(drafts); free(blk); return -1; }
    int n = 0; int done = 0;
    while (n < max_new_tokens && !done) {
        int32_t x1 = -1;
        if (llm_ai_mlx_session_pending_id(sh, &x1) != 0) break;
        out_ids[n++] = x1;
        if (eos_token_id >= 0 && x1 == eos_token_id) break;
        if (n >= max_new_tokens) break;
        /* Profondità del draft MTP: ADATTIVA (Gufo MtpLengthController: acceptance EWMA + sonda
         * full-accept) con override da registro (`LLM_MTP_DEPTH`). */
        int K = llm_mlx_spec_adaptive_width(s, 7);
        { const char* d = llm_ai_mlx_opt_env("LLM_MTP_DEPTH"); if (d) { int v = atoi(d); if (v >= 1 && v <= 8) K = v; } }
        if (K > 7) K = 7; if (K < 1) K = 1;
        int32_t g1 = -1;
        if (llm_ai_mlx_mtp_draftk(sh, K, drafts, &g1) != 0) for (int i = 0; i < K; i++) drafts[i] = x1;
        blk[0] = x1;
        for (int i = 0; i < K; i++) blk[i + 1] = drafts[i];
        int32_t acc = 0, next = -1;
        if (llm_ai_mlx_session_verify_draft_batch(sh, blk, K + 1, &acc, &next) != 0) break;
        /* Feedback al controllore adattivo: draft accettati = acc-1 (il primary e' sempre acc=1). */
        { int m_acc = acc > 0 ? acc - 1 : 0; if (m_acc > K) m_acc = K;
          s->spec_acc_ewma = (s->spec_acc_ewma <= 0.0f) ? (float)m_acc : 0.75f * s->spec_acc_ewma + 0.25f * (float)m_acc;
          s->spec_full_streak = (m_acc == K) ? s->spec_full_streak + 1 : 0; }
        /* acc >= 1 (il primary e' sempre il greedy corrente): emetti i draft accettati. */
        for (int i = 1; i < acc && n < max_new_tokens; i++) {
            out_ids[n++] = blk[i];
            if (eos_token_id >= 0 && blk[i] == eos_token_id) { done = 1; break; }
        }
    }
    free(drafts); free(blk);
    return n;
}
