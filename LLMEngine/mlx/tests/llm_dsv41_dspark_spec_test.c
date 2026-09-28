/* llm_dsv41_dspark_spec_test.c — verifica DSparkBlock/proposal_forward/forward_spec
 * (tiny mtp) vs riferimento. Build: tools/dsv41_dspark_spec_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_dspark_spec.h"
#include "llm_dsv41_dspark_spec_ref.h"
#define M(name) llm_dsv41_array_f32(REF_DS_##name, REF_DS_##name##_ND, (const int[]) REF_DS_##name##_SH)
static int nclose(const float* a, const float* b, int n, float tol) {
    for (int i=0;i<n;i++){ float d=a[i]-b[i]; if(d<0)d=-d; float m=b[i]<0?-b[i]:b[i]; if(m<1.0f)m=1.0f;
        if (d>tol*m){ printf("  [%d] got=%.6g exp=%.6g\n", i,a[i],b[i]); return 0; } }
    return 1;
}
static int eq_int(const int* a, const int* b, int n) {
    for (int i=0;i<n;i++) if (a[i]!=b[i]) { printf("  [%d] got=%d exp=%d\n", i,a[i],b[i]); return 0; }
    return 1;
}
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("llm_dsv41_dspark_spec — DSpark forward_spec (tiny mtp)\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Config c; memset(&c, 0, sizeof c);
    c.dim=REF_DS_DIM; c.n_heads=REF_DS_H; c.head_dim=REF_DS_D; c.rope_head_dim=REF_DS_DROPE;
    c.o_groups=REF_DS_OG; c.o_lora_rank=REF_DS_OL; c.q_lora_rank=REF_DS_QD;
    c.moe_inter_dim=REF_DS_DI; c.n_routed_experts=REF_DS_E; c.n_activated_experts=REF_DS_ACT;
    c.vocab_size=REF_DS_VOCAB; c.hc_mult=REF_DS_HC; c.window_size=REF_DS_WIN;
    c.norm_eps=1e-20f; c.hc_eps=1e-6f; c.hc_sinkhorn_iters=20;
    c.rope_theta=10000.0f; c.original_seq_len=0; c.rope_factor=1.0f; c.beta_fast=32.0f; c.beta_slow=1.0f;
    c.route_scale=1.0f; c.gate_temp=1.0f; c.swiglu_limit=0.0f; c.score_func=LLM_DSV41_SCORE_SQRTSOFTPLUS;
    c.norm_topk_prob=1; c.dspark_block_size=REF_DS_BLOCK; c.dspark_noise_token_id=REF_DS_NOISE;
    c.dspark_n_routed_experts=REF_DS_E; c.dspark_n_activated_experts=REF_DS_ACT;
    LlmDsv41Config dcfg = c;

    LlmDsv41LayerWeights blk; memset(&blk, 0, sizeof blk);
    blk.hc_attn_fn=M(HCAFN); blk.hc_attn_scale=M(HCAS); blk.hc_attn_base=M(HCAB);
    blk.hc_ffn_fn=M(HCFFN); blk.hc_ffn_scale=M(HCFFS); blk.hc_ffn_base=M(HCFFB);
    blk.attn_norm=M(ANORM); blk.ffn_norm=M(FNORM);
    blk.wq_a=M(WQA); blk.q_norm=M(QN); blk.wq_b=M(WQB); blk.wkv=M(WKV); blk.kv_norm=M(KVN);
    blk.wo_a=M(WOA); blk.wo_b=M(WOB); blk.attn_sink=M(SINK);
    blk.gate_w=M(GW); blk.gate_b=M(GB);
    blk.experts_w1=M(EW1); blk.experts_w3=M(EW3); blk.experts_w2=M(EW2);
    blk.shared_w1=M(SW1); blk.shared_w3=M(SW3); blk.shared_w2=M(SW2);

    LlmDsv41DSparkStage st; memset(&st, 0, sizeof st); st.block=blk;
    st.main_proj=M(MAINPROJ); st.main_norm=M(MAINNORM);
    st.out_norm=M(OUTNORM); st.markov_embed=M(MKEMB); st.markov_head=M(MKHEAD); st.conf_proj=M(CONFPROJ);

    LlmDsv41DSpark m; memset(&m, 0, sizeof m);
    m.cfg=c; m.draft_cfg=dcfg; m.embed=M(EMBED); m.head=M(HEAD); m.stages=&st; m.n_stages=1;

    LlmDsv41DSparkCache ck; memset(&ck, 0, sizeof ck); ck.max_size=REF_DS_WIN;

    mlx_array ids_p = mlx_array_new_data(REF_DS_IDSP, (const int[]){1,1}, 2, MLX_INT32);
    mlx_array ids_s = mlx_array_new_data(REF_DS_IDSS, (const int[]){1,1}, 2, MLX_INT32);
    mlx_array mh_p = M(MHP), mh_s = M(MHS);
    mlx_array op=mlx_array_new(), ob=mlx_array_new(), oc=mlx_array_new();

    int fail = 0;
    bool r0 = llm_dsv41_dspark_forward_spec(&m, ids_p, mh_p, &ck, &op, &ob, &oc, s);
    if (!r0 && ck.offset == (int)mlx_array_shape(mh_p)[1])
        printf("PASS %-46s\n", "forward_spec priming -> None + context");
    else { printf("FAIL %-46s\n", "forward_spec priming"); fail++; }

    bool r1 = llm_dsv41_dspark_forward_spec(&m, ids_s, mh_s, &ck, &op, &ob, &oc, s);
    if (r1 && ck.offset == REF_DS_CKOFF) printf("PASS %-46s\n", "forward_spec step -> output");
    else { printf("FAIL %-46s (offset=%d exp=%d)\n", "forward_spec step", ck.offset, REF_DS_CKOFF); fail++; }
    if (r1) {
        mlx_array_eval(op); mlx_array_eval(ob); mlx_array_eval(oc); mlx_array_eval(ck.keys);
        int nprop = (REF_DS_BLOCK + 1);
        if (eq_int(mlx_array_data_int32(op), REF_DS_PROP, nprop)) printf("PASS %-46s\n", "forward_spec -> proposed");
        else { printf("FAIL %-46s\n", "forward_spec -> proposed"); fail++; }
        if (nclose(mlx_array_data_float32(ob), REF_DS_BIAS, REF_DS_BLOCK*REF_DS_VOCAB, 2e-3f))
            printf("PASS %-46s\n", "forward_spec -> biases");
        else { printf("FAIL %-46s\n", "forward_spec -> biases"); fail++; }
        if (nclose(mlx_array_data_float32(oc), REF_DS_CONF, REF_DS_BLOCK, 2e-3f))
            printf("PASS %-46s\n", "forward_spec -> confidence");
        else { printf("FAIL %-46s\n", "forward_spec -> confidence"); fail++; }
        int nk = (int)mlx_array_shape(ck.keys)[2]*REF_DS_D;
        if (nclose(mlx_array_data_float32(ck.keys), REF_DS_CKEYS, nk, 2e-3f))
            printf("PASS %-46s\n", "forward_spec -> context keys");
        else { printf("FAIL %-46s\n", "forward_spec -> context keys"); fail++; }
    }
    printf(fail?"FAILURES: %d\n":"ALL PASS\n", fail);
    return fail;
}
