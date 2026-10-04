/* llm_dsv41_dspark_st_test.c — verifica del loader DSpark (`mtp.*`) + forward_spec.
 * Richiede il fixture /tmp/dsv41_st/mtp.safetensors (tools/gen_dsv41_dspark_st_ref.py).
 * Build: tools/dsv41_dspark_st_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_loader.h"
#include "llm_dsv41_dspark_st_ref.h"
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
    printf("llm_dsv41_dspark_st — loader mtp.* + forward_spec\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Config c; memset(&c, 0, sizeof c);
    c.dim=DSR_DIM; c.n_layers=1; c.n_heads=DSR_H; c.head_dim=DSR_D; c.rope_head_dim=DSR_DROPE;
    c.o_groups=DSR_OG; c.o_lora_rank=DSR_OL; c.q_lora_rank=DSR_QD; c.moe_inter_dim=DSR_DI;
    c.n_routed_experts=DSR_E; c.n_activated_experts=DSR_ACT; c.n_shared_experts=1;
    c.vocab_size=DSR_VOCAB; c.hc_mult=DSR_HC; c.window_size=DSR_WIN;
    c.norm_eps=1e-20f; c.hc_eps=1e-6f; c.hc_sinkhorn_iters=20;
    c.rope_theta=10000.0f; c.rope_factor=1.0f; c.beta_fast=32.0f; c.beta_slow=1.0f;
    c.route_scale=1.0f; c.gate_temp=1.0f; c.score_func=LLM_DSV41_SCORE_SQRTSOFTPLUS; c.norm_topk_prob=1;
    c.n_mtp_layers=1; c.dspark_block_size=DSR_BLOCK; c.dspark_noise_token_id=DSR_NOISE;
    c.dspark_n_routed_experts=DSR_E; c.dspark_n_activated_experts=DSR_ACT; c.dspark_markov_rank=DSR_RANK;
    c.n_dspark_targets=DSR_NT; c.dspark_target_layer_ids[0]=0;

    int fail=0;
    LlmDsv41DSpark d;
    if (llm_dsv41_load_dspark(&d, &c, "/tmp/dsv41_st/mtp.safetensors", s)) {
        printf("PASS %-46s\n", "load_dspark mtp.*");
    } else { printf("FAIL %-46s\n", "load_dspark mtp.*"); return 1; }

    LlmDsv41DSparkCache caches[1]; memset(&caches[0], 0, sizeof caches[0]); caches[0].max_size=DSR_WIN;
    mlx_array ids_p = mlx_array_new_data(DSR_IDS_P, (const int[]){1,1}, 2, MLX_INT32);
    mlx_array ids_s = mlx_array_new_data(DSR_IDS_S, (const int[]){1,1}, 2, MLX_INT32);
    mlx_array mh_p = llm_dsv41_array_f32(DSR_MHP, 3, (const int[]){1, DSR_P, DSR_M});
    mlx_array mh_s = llm_dsv41_array_f32(DSR_MHS, 3, (const int[]){1, DSR_S, DSR_M});
    mlx_array op=mlx_array_new(), ob=mlx_array_new(), oc=mlx_array_new();
    bool r0 = llm_dsv41_dspark_forward_spec(&d, ids_p, mh_p, caches, &op, &ob, &oc, s);
    if (!r0) printf("PASS %-46s\n", "priming -> None"); else { printf("FAIL priming\n"); fail++; }
    bool r1 = llm_dsv41_dspark_forward_spec(&d, ids_s, mh_s, caches, &op, &ob, &oc, s);
    if (r1) {
        mlx_array_eval(op); mlx_array_eval(ob); mlx_array_eval(oc);
        if (eq_int(mlx_array_data_int32(op), DSR_PROP, DSR_BLOCK+1)) printf("PASS %-46s\n", "loaded forward_spec -> proposed");
        else { printf("FAIL proposed\n"); fail++; }
        if (nclose(mlx_array_data_float32(ob), DSR_BIAS, DSR_BLOCK*DSR_VOCAB, 2e-3f)) printf("PASS %-46s\n", "loaded forward_spec -> biases");
        else { printf("FAIL biases\n"); fail++; }
        if (nclose(mlx_array_data_float32(oc), DSR_CONF, DSR_BLOCK, 2e-3f)) printf("PASS %-46s\n", "loaded forward_spec -> confidence");
        else { printf("FAIL confidence\n"); fail++; }
    } else { printf("FAIL loaded forward_spec step\n"); fail++; }
    printf(fail?"FAILURES: %d\n":"ALL PASS\n", fail);
    return fail;
}
