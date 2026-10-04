/* llm_dsv41_engine_mtp_test.c — engine MTP wiring: parità MTP vs greedy (tiny, fixture mtp.*).
 * Build: tools/dsv41_engine_mtp_test.sh */
#include <stdio.h>
#include <string.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_engine.h"
#include "llm_dsv41_dspark_st_ref.h"
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("llm_dsv41_engine_mtp — parità MTP vs greedy\n");
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

    const char* path = "/tmp/dsv41_st/mtp.safetensors";
    LlmDsv41Engine e; memset(&e, 0, sizeof e);
    int unloaded=0;
    if (!llm_dsv41_load_checkpoint(&e.w, &c, path, s, &unloaded)) { printf("FAIL load backbone\n"); return 1; }
    e.vocab = c.vocab_size;
    e.has_dspark = llm_dsv41_load_dspark(&e.dspark, &c, path, s);
    if (!e.has_dspark) { printf("FAIL load dspark\n"); return 1; }

    LlmDsv41Sampling sm; memset(&sm, 0, sizeof sm); sm.active = false;  /* greedy */
    int32_t prompt[3] = { 3, 7, 5 };
    int fail = 0;

    /* greedy */
    LlmDsv41Cache* c1 = NULL; int off1 = 0;
    int32_t pend1 = 0; bool hp1 = false;
    if (llm_dsv41_engine_append(&e, (void**)&c1, prompt, 3, &off1, &pend1, &hp1, &sm, s) != 0) { printf("FAIL append greedy\n"); return 1; }
    int32_t g1[600]; int n1 = llm_dsv41_engine_generate(&e, (void**)&c1, &off1, pend1, hp1, 400, -1, g1, &sm, s);
    if (n1 < 0) { printf("FAIL greedy generate\n"); return 1; }
    llm_dsv41_cache_free(c1); free(c1);

    /* mtp */
    e.mtp_enabled = true;
    LlmDsv41Cache *c2 = NULL; int off2 = 0; int32_t pend2 = 0; bool hp2 = false;
    if (llm_dsv41_engine_append(&e, (void**)&c2, prompt, 3, &off2, &pend2, &hp2, &sm, s) != 0) { printf("FAIL append mtp\n"); return 1; }
    int32_t g2[600]; int n2 = llm_dsv41_engine_generate(&e, (void**)&c2, &off2, pend2, hp2, 400, -1, g2, &sm, s);
    if (n2 < 0) { printf("FAIL mtp generate\n"); return 1; }

    printf("mtp stats: prop_ok=%ld speculated=%ld accepted=%ld rollbacks=%ld\n", e.mtp_prop_ok, e.mtp_proposed, e.mtp_accepted, e.mtp_rollbacks);
    printf("greedy[%d]:", n1); for (int i=0;i<n1;i++) printf(" %d", g1[i]); printf("\n");
    printf("mtp[%d]  :", n2); for (int i=0;i<n2;i++) printf(" %d", g2[i]); printf("\n");
    if (n1 == n2 && memcmp(g1, g2, (size_t)n1*sizeof(int32_t)) == 0)
        printf("PASS %-46s\n", "MTP accepted/greedy parity");
    else { printf("FAIL %-46s\n", "MTP parity"); fail++; }
    printf(fail?"FAILURES: %d\n":"ALL PASS\n", fail);
    return fail;
}
