/* ============================================================================
 * llm_mlx_moe_v2.c — PERCORSO MoE PARAMETRIZZATO (unico FFN MoE dopo il ritiro di
 * qwen3_moe_ffn_block, 2026-10-17). Obiettivo: un solo FFN MoE che copre
 * piu' famiglie via config (nome shared-expert, presenza gate, gate-act, selezione
 * densi/MoE). Correttezza-prima: gather esperti PER-TOKEN/PER-ESPERTO (loop), niente
 * gather_qmm sorted per ora (perf = step successivo). Prima famiglia: ernie4_5_moe.
 *
 * Stadio 1 (2026-10-17) — indirezione nomi + router plain + expert_bias (default = invariato):
 *   moe_norm_name / moe_router_name / moe_experts_base / moe_dense_base (vuoti = nomi storici);
 *   moe_router_plain (1 = plain_linear sul router); use_expert_bias + moe_expert_bias_name
 *   (bias SOLO nella selezione top-k, mai nei pesi — semantica mlx_lm LFM2-MoE).
 *
 * Layer: norm = <moe_norm_name>(h)
 *   denso: <moe_dense_base>.{gate,up,down}_proj (SwiGLU)                 out = h + mlp
 *   MoE:   router <moe_router_name> -> act(softmax|sigmoid) -> (+bias) -> top-k -> renorm
 *          -> <moe_experts_base>.{gate,up,down}_proj [E,...]  (+ shared expert opz.)  out = h + moe
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

bool moe_v2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* lineare su un expert slice: prende la riga `e` dell'asse 0 di w/sc/b (stacked esperti). */
static bool expert_lin(mlx_array* res, mlx_array x, mlx_array w, mlx_array sc, mlx_array b,
                       int e, int gsz, int bts, const char* mode, mlx_stream s) {
    mlx_array idx = mlx_array_new_int(e);
    mlx_array we = mlx_array_new(), se = mlx_array_new(), be = mlx_array_new();
    bool ok = mlx_take_axis(&we, w, idx, 0, s) == 0;
    if (ok && sc.ctx != NULL && mlx_array_size(sc) > 0) ok = mlx_take_axis(&se, sc, idx, 0, s) == 0;
    if (ok && b.ctx != NULL && mlx_array_size(b) > 0) ok = mlx_take_axis(&be, b, idx, 0, s) == 0;
    if (ok) ok = quantized_linear_cfg(res, x, we, se, be, gsz, bts, mode, s);
    mlx_array_free(idx); mlx_array_free(we); mlx_array_free(se); mlx_array_free(be);
    return ok;
}

/* lineare batched da un peso (triple) con nome. */
static bool named_lin(mlx_array* res, LlmMlxModel* m, int layer_idx, const char* suf, mlx_array x,
                      int gsz, int bts, mlx_stream s) {
    char nb[256];
    tname(m->cfg.layer_prefix, layer_idx, suf, nb, sizeof(nb));
    mlx_array w = mlx_array_new(), sc = mlx_array_new(), b = mlx_array_new();
    if (!multishard_get_triple(&m->weights, nb, &w, &sc, &b)) { mlx_array_free(w); mlx_array_free(sc); mlx_array_free(b); return false; }
    bool ok = quantized_linear_cfg(res, x, w, sc, b, gsz, bts, m->cfg.quant_mode, s);
    mlx_array_free(w); mlx_array_free(sc); mlx_array_free(b);
    return ok;
}

/* true se questo layer e' DENSE (non MoE). */
static bool moe_v2_is_dense(const LlmMlxModel* m, int layer_idx) {
    if (m->cfg.moe_layer_interval > 0) {
        int end = m->cfg.moe_layer_end >= 0 ? m->cfg.moe_layer_end : m->cfg.n_layers - 1;
        bool is_moe = ((layer_idx + 1) % m->cfg.moe_layer_interval == 0) &&
                      layer_idx >= m->cfg.moe_layer_start && layer_idx <= end;
        return !is_moe;
    }
    return layer_idx < m->cfg.first_k_dense_replace;   /* fallback: primi N densi */
}

bool moe_v2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const float rm = m->cfg.moe_residual_multiplier != 0.0f ? m->cfg.moe_residual_multiplier : 1.0f;
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "moe_v2[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    const char* normname = m->cfg.moe_norm_name[0] ? m->cfg.moe_norm_name : "post_attention_layernorm.weight";
    mlx_array pal_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, normname, nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &pal_w);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, pal_w, m->cfg.rms_norm_eps, s); CHK(ok, "post_attn_norm"); }

    /* ---- layer DENSO: SwiGLU <dense_base>.{gate,up,down}_proj (default mlp) ---- */
    if (ok && moe_v2_is_dense(m, layer_idx)) {
        const char* dbase = m->cfg.moe_dense_base[0] ? m->cfg.moe_dense_base : "mlp";
        char dsuf[96];
        mlx_array g = mlx_array_new(), u = mlx_array_new(), sw = mlx_array_new(), dn = mlx_array_new();
        snprintf(dsuf, sizeof dsuf, "%s.gate_proj", dbase);
        ok = named_lin(&g, m, layer_idx, dsuf, normed, gsz, bts, s);
        snprintf(dsuf, sizeof dsuf, "%s.up_proj", dbase);
        if (ok) ok = named_lin(&u, m, layer_idx, dsuf, normed, gsz, bts, s);
        if (ok) { bool fused = n_tokens > 1 && llm_mlx_fuse_swiglu_prefill_enabled() && llm_mlx_swiglu_compiled(g, u, &sw, s);
                  if (!fused) { mlx_array sg=mlx_array_new(), sl=mlx_array_new();
                  ok = mlx_sigmoid(&sg,g,s)==0 && mlx_multiply(&sl,g,sg,s)==0 && mlx_multiply(&sw,sl,u,s)==0;
                  mlx_array_free(sg); mlx_array_free(sl); } }
        snprintf(dsuf, sizeof dsuf, "%s.down_proj", dbase);
        if (ok) ok = named_lin(&dn, m, layer_idx, dsuf, sw, gsz, bts, s);
        if (ok && rm != 1.0f) { mlx_array sc=mlx_array_new_float32(rm), sdn=mlx_array_new();
                                ok = mlx_multiply(&sdn, dn, sc, s)==0; if (ok) mlx_array_set(&dn, sdn);
                                mlx_array_free(sc); mlx_array_free(sdn); }
        if (ok) ok = mlx_add(out, h, dn, s) == 0;
        mlx_array_free(g); mlx_array_free(u); mlx_array_free(sw); mlx_array_free(dn);
        mlx_array_free(pal_w); mlx_array_free(normed);
        return ok;
    }

    /* ---- layer MoE ---- */
    const int E = m->cfg.n_experts, K = m->cfg.top_k;
    /* pesi esperti stacked: mlp.switch_mlp.{gate,up,down}_proj [E, ...] */
    mlx_array gw=mlx_array_new(),gs=mlx_array_new(),gb=mlx_array_new();
    mlx_array uw=mlx_array_new(),us=mlx_array_new(),ub=mlx_array_new();
    mlx_array dw=mlx_array_new(),ds=mlx_array_new(),db=mlx_array_new();
    { const char* ebase = m->cfg.moe_experts_base[0] ? m->cfg.moe_experts_base : "mlp.switch_mlp";
      char esuf[96];
      snprintf(esuf, sizeof esuf, "%s.gate_proj", ebase);
      tname(m->cfg.layer_prefix, layer_idx, esuf, nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &gw, &gs, &gb);
      snprintf(esuf, sizeof esuf, "%s.up_proj", ebase);
      tname(m->cfg.layer_prefix, layer_idx, esuf, nb, sizeof(nb));   ok = ok && multishard_get_triple(&m->weights, nb, &uw, &us, &ub);
      snprintf(esuf, sizeof esuf, "%s.down_proj", ebase);
      tname(m->cfg.layer_prefix, layer_idx, esuf, nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &dw, &ds, &db);
    }
    CHK(ok, "switch_mlp lookup");

    /* router: <router_name> (default mlp.gate) -> act */
    const char* rname = m->cfg.moe_router_name[0] ? m->cfg.moe_router_name : "mlp.gate";
    mlx_array gates = mlx_array_new();
    if (m->cfg.moe_router_plain) {
        /* Router NON quantizzato (es. mixtral block_sparse_moe.gate): solo .weight, plain matmul. */
        char rwsuf[96], rb[256];
        snprintf(rwsuf, sizeof rwsuf, "%s.weight", rname);
        tname(m->cfg.layer_prefix, layer_idx, rwsuf, rb, sizeof(rb));
        mlx_array rw = mlx_array_new();
        ok = ok && multishard_get(&m->weights, rb, &rw);
        if (ok) ok = plain_linear(&gates, normed, rw, s);
        mlx_array_free(rw);
        CHK(ok, "router matmul (plain)");
    } else if (m->cfg.moe_quant_derive) {
        /* Router con quant PER-TENSORE (es. LFM2-MoE: 8-bit in modello 4-bit). */
        char rb[256]; tname(m->cfg.layer_prefix, layer_idx, rname, rb, sizeof(rb));
        mlx_array rw = mlx_array_new(), rs = mlx_array_new(), rbb = mlx_array_new();
        ok = ok && multishard_get_triple(&m->weights, rb, &rw, &rs, &rbb);
        if (ok) {
            int rg = gsz, rbits = bts; char rm[16];
            llm_mlx_quant_derive(rw, rs, gsz, bts, "affine", &rg, &rbits, rm, sizeof rm);
            ok = quantized_linear_cfg(&gates, normed, rw, rs, rbb, rg, rbits, rm, s);
        }
        mlx_array_free(rw); mlx_array_free(rs); mlx_array_free(rbb);
        CHK(ok, "router matmul (derive)");
    } else {
        ok = ok && named_lin(&gates, m, layer_idx, rname, normed, gsz, bts, s);
    }
    if (ok) {
        mlx_array act = mlx_array_new();
        /* Gate in f32 (mlx_lm: gates.astype(mx.float32) PRIMA dell'act, sia softmax sia sigmoid).
         * Guardato da moe_gate_f32 (glm4_moe/ernie): il default resta il dtype del router. */
        if (m->cfg.moe_gate_f32 && mlx_array_dtype(gates) != MLX_FLOAT32) {
            mlx_array gf = mlx_array_new();
            if (mlx_astype(&gf, gates, MLX_FLOAT32, s) == 0) mlx_array_set(&gates, gf);
            mlx_array_free(gf);
        }
        if (m->cfg.moe_gate_sigmoid) ok = mlx_sigmoid(&act, gates, s) == 0;
        else ok = mlx_softmax_axis(&act, gates, -1, /*precise=*/true, s) == 0;
        if (ok) mlx_array_set(&gates, act);
        mlx_array_free(act);
        CHK(ok, "router act");
    }

    llm_mlx_dbg_dump("ffn_normed", layer_idx, normed); llm_mlx_dbg_dump("gates", layer_idx, gates);
    /* expert_bias di SELEZIONE (opzionale, es. LFM2-MoE): il bias entra SOLO nell'argpartition,
     * mai nei pesi; i pesi si prendono da `gates` (attivati, senza bias). Default: assente =>
     * selection == gates (comportamento invariato). */
    mlx_array select = mlx_array_new();
    if (ok && m->cfg.use_expert_bias && m->cfg.moe_expert_bias_name[0]) {
        char bb[256]; tname(m->cfg.layer_prefix, layer_idx, m->cfg.moe_expert_bias_name, bb, sizeof(bb));
        mlx_array eb = mlx_array_new();
        bool got = multishard_get(&m->weights, bb, &eb);
        if (!got && m->cfg.moe_expert_bias_alt[0]) {
            tname(m->cfg.layer_prefix, layer_idx, m->cfg.moe_expert_bias_alt, bb, sizeof(bb));
            got = multishard_get(&m->weights, bb, &eb);
        }
        ok = ok && got;
        if (ok) {
            mlx_array gf = mlx_array_new(), sum = mlx_array_new();
            ok = mlx_astype(&gf, gates, MLX_FLOAT32, s) == 0;
            if (ok) ok = mlx_add(&sum, gf, eb, s) == 0;
            if (ok) mlx_array_set(&select, sum);
            mlx_array_free(gf); mlx_array_free(sum);
        }
        mlx_array_free(eb);
        CHK(ok, "expert_bias");
    }
    /* group-pruning router (GLM-4.5): su select (o gates), tiene i topk_group gruppi per
     * group_score = somma dei top-2 esperti del gruppo; azzera gli altri. [1,T,n_group,epg] -> [1,T,E]. */
    mlx_array pruned = mlx_array_new();
    if (ok && m->cfg.moe_group_prune && m->cfg.n_group > 1 && m->cfg.n_experts > 0 &&
        m->cfg.n_experts % m->cfg.n_group == 0 && m->cfg.topk_group > 0 && m->cfg.topk_group < m->cfg.n_group) {
        const int ng = m->cfg.n_group, epg = m->cfg.n_experts / ng;
        mlx_array base = select.ctx != NULL ? select : gates;
        int us4[4] = {1, n_tokens, ng, epg};
        mlx_array bu = mlx_array_new();
        ok = mlx_reshape(&bu, base, us4, 4, s) == 0;
        mlx_array gs2 = mlx_array_new();
        if (ok) {
            mlx_array gtop = mlx_array_new();
            ok = mlx_topk_axis(&gtop, bu, 2, -1, s) == 0;
            if (ok) ok = mlx_sum_axis(&gs2, gtop, -1, /*keepdims=*/true, s) == 0;
            mlx_array_free(gtop);
        }
        if (ok) {
            const int k = ng - m->cfg.topk_group;
            mlx_array gpart = mlx_array_new();
            ok = mlx_argpartition_axis(&gpart, gs2, k - 1, /*axis=*/2, s) == 0;
            if (ok) {
                int st4[4]={0,0,0,0}, sp4[4]={1,n_tokens,k,1}, sv4[4]={1,1,1,1};
                mlx_array gview = mlx_array_new();
                ok = mlx_slice(&gview, gpart, st4,4, sp4,4, sv4,4, s) == 0;
                mlx_array gidx = mlx_array_new();
                if (ok) ok = mlx_contiguous(&gidx, gview, /*allow_col_major=*/false, s) == 0;
                mlx_array_free(gview);
                if (ok) {
                    mlx_array zero = mlx_array_new_float32(0.0f), bu2 = mlx_array_new();
                    ok = mlx_put_along_axis(&bu2, bu, gidx, zero, /*axis=*/2, s) == 0;
                    mlx_array_free(zero);
                    if (ok) mlx_array_set(&bu, bu2);
                    mlx_array_free(bu2);
                }
                mlx_array_free(gidx);
            }
            mlx_array_free(gpart);
        }
        mlx_array_free(gs2);
        if (ok) { int fs3[3]={1,n_tokens,m->cfg.n_experts}; ok = mlx_reshape(&pruned, bu, fs3, 3, s) == 0; }
        mlx_array_free(bu);
        CHK(ok, "group-prune");
    }
    /* top-k + renorm (device, batched) */
    mlx_array part = mlx_array_new(), inds = mlx_array_new(), scores = mlx_array_new();
    if (ok) {
        mlx_array arg = pruned.ctx != NULL ? pruned : (select.ctx != NULL ? select : gates);
        ok = mlx_argpartition_axis(&part, arg, /*kth=*/-K, /*axis=*/-1, s) == 0;
        if (ok) { int a[3]={0,0,E-K}, b2[3]={1,n_tokens,E}, st[3]={1,1,1}; ok = mlx_slice(&inds, part, a,3, b2,3, st,3, s)==0; }
        if (ok) ok = mlx_take_along_axis(&scores, gates, inds, -1, s)==0;
        if (ok && m->cfg.norm_topk_prob) {
            mlx_array ssum=mlx_array_new(), rs=mlx_array_new(), sc2=mlx_array_new(); int ax[1]={-1};
            ok = mlx_sum_axes(&ssum, scores, ax, 1, true, s)==0
              && mlx_maximum(&rs, ssum, llm_mlx_act_scalar(1e-12f), s)==0
              && mlx_divide(&sc2, scores, rs, s)==0;
            if (ok) mlx_array_set(&scores, sc2);
            mlx_array_free(ssum); mlx_array_free(rs); mlx_array_free(sc2);
        }
        /* routed_scaling_factor (es. LFM2-MoE; default 1.0 -> no-op): scala i pesi dopo la renorm. */
        if (ok && m->cfg.routed_scaling_factor != 1.0f) {
            mlx_array scf=mlx_array_new_float32(m->cfg.routed_scaling_factor), sc=mlx_array_new(), wsc=mlx_array_new();
            ok = mlx_astype(&sc, scf, mlx_array_dtype(scores), s)==0
              && mlx_multiply(&wsc, scores, sc, s)==0;
            if (ok) mlx_array_set(&scores, wsc);
            mlx_array_free(scf); mlx_array_free(sc); mlx_array_free(wsc);
        }
        CHK(ok, "top-k/renorm");
    }
    /* gather_qmm batched (come mlx_lm SwitchGLU): x [T,1,1,D] + inds [T,K] -> [T,K,1,*] */
    mlx_array x1=mlx_array_new(), inds_u=mlx_array_new();
    if (ok) { int sh[4]={n_tokens,1,1,m->cfg.hidden_size}; ok = mlx_reshape(&x1, normed, sh, 4, s)==0; }
    if (ok) ok = mlx_astype(&inds_u, inds, MLX_UINT32, s)==0;
    mlx_array no_lhs=mlx_array_new();
    /* quant PER-TENSORE degli esperti (es. LFM2-MoE): deriva group/bits/mode dalle scales
     * degli esperti stacked; default = quant globale (comportamento invariato). */
    int gg = gsz, gbts = bts; char gmode[16];
    llm_mlx_mode_set(gmode, sizeof gmode, m->cfg.quant_mode);
    if (m->cfg.moe_quant_derive)
        llm_mlx_quant_derive(gw, gs, gsz, bts, "affine", &gg, &gbts, gmode, sizeof gmode);
    mlx_optional_int gsz_o={gg,true}, bts_o={gbts,true};
    mlx_array g_out=mlx_array_new(), u_out=mlx_array_new(), act=mlx_array_new(), d_out=mlx_array_new();
    if (ok) ok = llm_mlx_moe_gather_mm(&m->cfg, &g_out, x1, gw, gs, gb, no_lhs, inds_u, true, gsz_o, bts_o, gmode, false, s)==0;
    if (ok) ok = llm_mlx_moe_gather_mm(&m->cfg, &u_out, x1, uw, us, ub, no_lhs, inds_u, true, gsz_o, bts_o, gmode, false, s)==0;
    /* mlx_lm: swiglu = @mx.compile(shapeless) silu(gate)*up (fusa): differisce di 1 ulp bf16 dall'eager su alcuni
     * elementi. Prefill (n_tokens>1): closure compilata (stessa politica di qwen3_moe_ffn_block); decode:
     * eager (il costo fisso di mlx_closure_apply non si ammortizza a n_tokens==1). */
    if (ok) {
        bool fused = n_tokens > 1 && llm_mlx_fuse_swiglu_prefill_enabled() && llm_mlx_swiglu_compiled(g_out, u_out, &act, s);
        if (!fused) { mlx_array sg=mlx_array_new(), sl=mlx_array_new(); ok = mlx_sigmoid(&sg,g_out,s)==0 && mlx_multiply(&sl,g_out,sg,s)==0 && mlx_multiply(&act,sl,u_out,s)==0; mlx_array_free(sg); mlx_array_free(sl); }
    }
    if (ok) ok = llm_mlx_moe_gather_mm(&m->cfg, &d_out, act, dw, ds, db, no_lhs, inds_u, true, gsz_o, bts_o, gmode, false, s)==0;
    CHK(ok, "gather_qmm experts");
    mlx_array moe_out = mlx_array_new();
    if (ok) {
        int scs[5]={1,n_tokens,K,1,1}; mlx_array sce=mlx_array_new(); ok = mlx_reshape(&sce, scores, scs, 5, s)==0;
        mlx_array wd=mlx_array_new(); int ax1[1]={2}; mlx_array sm=mlx_array_new();
        if (ok) ok = mlx_multiply(&wd, d_out, sce, s)==0;
        if (ok) ok = mlx_sum_axes(&sm, wd, ax1, 1, false, s)==0;
        if (ok) { int os[3]={1,n_tokens,m->cfg.hidden_size}; ok = mlx_reshape(&moe_out, sm, os, 3, s)==0; }
        mlx_array_free(sce); mlx_array_free(wd); mlx_array_free(sm);
        CHK(ok, "moe weighted sum");
    }
    llm_mlx_dbg_dump("inds", layer_idx, inds); llm_mlx_dbg_dump("scores", layer_idx, scores); llm_mlx_dbg_dump("g_out", layer_idx, g_out); llm_mlx_dbg_dump("act", layer_idx, act); llm_mlx_dbg_dump("d_out", layer_idx, d_out); llm_mlx_dbg_dump("moe_out", layer_idx, moe_out);
    mlx_array_free(part); mlx_array_free(inds); mlx_array_free(scores); mlx_array_free(x1); mlx_array_free(inds_u);
    mlx_array_free(select); mlx_array_free(pruned);
    mlx_array_free(no_lhs); mlx_array_free(g_out); mlx_array_free(u_out); mlx_array_free(act); mlx_array_free(d_out);

    /* shared expert (opzionale): base cfg (default "mlp.shared_expert"); gate opzionale */
    if (ok && m->cfg.moe_shared_base[0]) {
        char suf[64]; mlx_array sg_w=mlx_array_new(), su=mlx_array_new(), sd=mlx_array_new(), sh=mlx_array_new();
        snprintf(suf,sizeof suf,"%s.gate_proj", m->cfg.moe_shared_base);
        bool sok = named_lin(&sg_w, m, layer_idx, suf, normed, gsz, bts, s);
        snprintf(suf,sizeof suf,"%s.up_proj", m->cfg.moe_shared_base);
        if (sok) sok = named_lin(&su, m, layer_idx, suf, normed, gsz, bts, s);
        if (sok) { mlx_array g=mlx_array_new(), sl=mlx_array_new(); sok = mlx_sigmoid(&g,sg_w,s)==0 && mlx_multiply(&sl,sg_w,g,s)==0 && mlx_multiply(&sh,sl,su,s)==0; mlx_array_free(g); mlx_array_free(sl); }
        snprintf(suf,sizeof suf,"%s.down_proj", m->cfg.moe_shared_base);
        if (sok) sok = named_lin(&sd, m, layer_idx, suf, sh, gsz, bts, s);
        if (sok && m->cfg.moe_shared_gate) {
            char gsuf[64]; snprintf(gsuf,sizeof gsuf,"%s_gate", m->cfg.moe_shared_base);
            mlx_array ggate=mlx_array_new();
            if (named_lin(&ggate, m, layer_idx, gsuf, normed, gsz, bts, s)) {
                mlx_array gg=mlx_array_new(), sd2=mlx_array_new();
                if (mlx_sigmoid(&gg,ggate,s)==0 && mlx_multiply(&sd2,sd,gg,s)==0) mlx_array_set(&sd, sd2);
                mlx_array_free(gg); mlx_array_free(sd2);
            }
            mlx_array_free(ggate);
        }
        if (sok) { mlx_array na=mlx_array_new(); ok = mlx_add(&na, moe_out, sd, s)==0; if (ok) mlx_array_set(&moe_out, na); mlx_array_free(na); }
        mlx_array_free(sg_w); mlx_array_free(su); mlx_array_free(sd); mlx_array_free(sh);
    }

    if (ok && rm != 1.0f) { mlx_array sc=mlx_array_new_float32(rm), smo=mlx_array_new();
                            ok = mlx_multiply(&smo, moe_out, sc, s)==0; if (ok) mlx_array_set(&moe_out, smo);
                            mlx_array_free(sc); mlx_array_free(smo); }
    if (ok) { ok = mlx_add(out, h, moe_out, s) == 0; CHK(ok, "residual"); }
    mlx_array_free(pal_w); mlx_array_free(normed);
    mlx_array_free(gw);mlx_array_free(gs);mlx_array_free(gb);mlx_array_free(uw);mlx_array_free(us);mlx_array_free(ub);
    mlx_array_free(dw);mlx_array_free(ds);mlx_array_free(db);mlx_array_free(gates);mlx_array_free(moe_out);
    return ok;
#undef CHK
}
