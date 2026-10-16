/* ============================================================================
 * llm_mlx_moe_v2.c — PERCORSO MoE PARAMETRIZZATO (nuovo, affiancato al vecchio
 * qwen3_moe_ffn_block che resta PARCHEGGIATO). Obiettivo: un solo FFN MoE che copre
 * piu' famiglie via config (nome shared-expert, presenza gate, gate-act, selezione
 * densi/MoE). Correttezza-prima: gather esperti PER-TOKEN/PER-ESPERTO (loop), niente
 * gather_qmm sorted per ora (perf = step successivo). Prima famiglia: ernie4_5_moe.
 *
 * Layer: norm = post_attention_layernorm(h)
 *   denso: mlp.{gate,up,down}_proj (SwiGLU)                          out = h + mlp
 *   MoE:   router mlp.gate -> act(softmax|sigmoid) -> top-k -> renorm -> switch_mlp
 *          (+ shared expert opzionale)                               out = h + moe
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
    char nb[256];
    mlx_stream s = m->stream;
#define CHK(c, w) do { if (!(c)) { fprintf(stderr, "moe_v2[layer=%d]: %s failed\n", layer_idx, (w)); ok = false; } } while (0)

    mlx_array pal_w = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", nb, sizeof(nb));
    ok = ok && multishard_get(&m->weights, nb, &pal_w);
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, pal_w, m->cfg.rms_norm_eps, s); CHK(ok, "post_attn_norm"); }

    /* ---- layer DENSO: SwiGLU mlp.{gate,up,down}_proj ---- */
    if (ok && moe_v2_is_dense(m, layer_idx)) {
        mlx_array g = mlx_array_new(), u = mlx_array_new(), sw = mlx_array_new(), dn = mlx_array_new();
        ok = named_lin(&g, m, layer_idx, "mlp.gate_proj", normed, gsz, bts, s);
        if (ok) ok = named_lin(&u, m, layer_idx, "mlp.up_proj", normed, gsz, bts, s);
        if (ok) { bool fused = n_tokens > 1 && llm_mlx_fuse_swiglu_prefill_enabled() && llm_mlx_swiglu_compiled(g, u, &sw, s);
                  if (!fused) { mlx_array sg=mlx_array_new(), sl=mlx_array_new();
                  ok = mlx_sigmoid(&sg,g,s)==0 && mlx_multiply(&sl,g,sg,s)==0 && mlx_multiply(&sw,sl,u,s)==0;
                  mlx_array_free(sg); mlx_array_free(sl); } }
        if (ok) ok = named_lin(&dn, m, layer_idx, "mlp.down_proj", sw, gsz, bts, s);
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
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.gate_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &gw, &gs, &gb);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.up_proj", nb, sizeof(nb));   ok = ok && multishard_get_triple(&m->weights, nb, &uw, &us, &ub);
    tname(m->cfg.layer_prefix, layer_idx, "mlp.switch_mlp.down_proj", nb, sizeof(nb)); ok = ok && multishard_get_triple(&m->weights, nb, &dw, &ds, &db);
    CHK(ok, "switch_mlp lookup");

    /* router: mlp.gate -> act */
    mlx_array gates = mlx_array_new();
    ok = ok && named_lin(&gates, m, layer_idx, "mlp.gate", normed, gsz, bts, s);
    if (ok) {
        mlx_array act = mlx_array_new();
        if (m->cfg.moe_gate_sigmoid) ok = mlx_sigmoid(&act, gates, s) == 0;
        else ok = mlx_softmax_axis(&act, gates, -1, /*precise=*/true, s) == 0;
        if (ok) mlx_array_set(&gates, act);
        mlx_array_free(act);
        CHK(ok, "router act");
    }

    llm_mlx_dbg_dump("ffn_normed", layer_idx, normed); llm_mlx_dbg_dump("gates", layer_idx, gates);
    /* top-k + renorm (device, batched) */
    mlx_array part = mlx_array_new(), inds = mlx_array_new(), scores = mlx_array_new();
    if (ok) {
        ok = mlx_argpartition_axis(&part, gates, /*kth=*/-K, /*axis=*/-1, s) == 0;
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
        CHK(ok, "top-k/renorm");
    }
    /* gather_qmm batched (come mlx_lm SwitchGLU): x [T,1,1,D] + inds [T,K] -> [T,K,1,*] */
    mlx_array x1=mlx_array_new(), inds_u=mlx_array_new();
    if (ok) { int sh[4]={n_tokens,1,1,m->cfg.hidden_size}; ok = mlx_reshape(&x1, normed, sh, 4, s)==0; }
    if (ok) ok = mlx_astype(&inds_u, inds, MLX_UINT32, s)==0;
    mlx_array no_lhs=mlx_array_new();
    mlx_optional_int gsz_o={gsz,true}, bts_o={bts,true};
    mlx_array g_out=mlx_array_new(), u_out=mlx_array_new(), act=mlx_array_new(), d_out=mlx_array_new();
    if (ok) ok = llm_mlx_moe_gather_mm(&m->cfg, &g_out, x1, gw, gs, gb, no_lhs, inds_u, true, gsz_o, bts_o, m->cfg.quant_mode, false, s)==0;
    if (ok) ok = llm_mlx_moe_gather_mm(&m->cfg, &u_out, x1, uw, us, ub, no_lhs, inds_u, true, gsz_o, bts_o, m->cfg.quant_mode, false, s)==0;
    /* mlx_lm: swiglu = @mx.compile(shapeless) silu(gate)*up (fusa): differisce di 1 ulp bf16 dall'eager su alcuni
     * elementi. Prefill (n_tokens>1): closure compilata (stessa politica di qwen3_moe_ffn_block); decode:
     * eager (il costo fisso di mlx_closure_apply non si ammortizza a n_tokens==1). */
    if (ok) {
        bool fused = n_tokens > 1 && llm_mlx_fuse_swiglu_prefill_enabled() && llm_mlx_swiglu_compiled(g_out, u_out, &act, s);
        if (!fused) { mlx_array sg=mlx_array_new(), sl=mlx_array_new(); ok = mlx_sigmoid(&sg,g_out,s)==0 && mlx_multiply(&sl,g_out,sg,s)==0 && mlx_multiply(&act,sl,u_out,s)==0; mlx_array_free(sg); mlx_array_free(sl); }
    }
    if (ok) ok = llm_mlx_moe_gather_mm(&m->cfg, &d_out, act, dw, ds, db, no_lhs, inds_u, true, gsz_o, bts_o, m->cfg.quant_mode, false, s)==0;
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

    if (ok) { ok = mlx_add(out, h, moe_out, s) == 0; CHK(ok, "residual"); }
    mlx_array_free(pal_w); mlx_array_free(normed);
    mlx_array_free(gw);mlx_array_free(gs);mlx_array_free(gb);mlx_array_free(uw);mlx_array_free(us);mlx_array_free(ub);
    mlx_array_free(dw);mlx_array_free(ds);mlx_array_free(db);mlx_array_free(gates);mlx_array_free(moe_out);
    return ok;
#undef CHK
}
