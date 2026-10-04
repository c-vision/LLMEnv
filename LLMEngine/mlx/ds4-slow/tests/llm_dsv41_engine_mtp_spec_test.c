/* llm_dsv41_engine_mtp_spec_test.c — test mirato del batch/rollback MTP:
 * costruisce una proposta = continuazione greedy reale del target e ne verifica
 * l'accettazione completa; poi corrompe l'ultimo token per forzare il rollback.
 * Build: tools/dsv41_engine_mtp_spec_test.sh */
#include <stdio.h>
#include <string.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_engine.h"
#include "llm_dsv41_dspark_st_ref.h"

static int32_t argmax_row(const float* p, int V) {
    int bi = 0; for (int i = 1; i < V; i++) if (p[i] > p[bi]) bi = i; return bi;
}
/* step greedy su una cache: ritorna argmax dei logits e libera. */
static int32_t step_argmax(LlmDsv41Engine* e, LlmDsv41Cache** c, int32_t tok, int* off, mlx_stream s) {
    mlx_array lg = mlx_array_new(), mh = mlx_array_new();
    if (!llm_dsv41_engine_step_logits(e, c, &tok, 1, off, &lg, &mh, s)) return -1;
    mlx_array_eval(lg);
    int32_t n = argmax_row(mlx_array_data_float32(lg), e->vocab);
    mlx_array_free(lg); mlx_array_free(mh);
    return n;
}
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("llm_dsv41_engine_mtp_spec — batch + rollback\n");
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

    LlmDsv41Sampling sm; memset(&sm, 0, sizeof sm); sm.active = false;
    int32_t prompt[3] = { 3, 7, 5 };
    LlmDsv41Cache* c0 = NULL; int off = 0; int32_t pend = 0; bool hp = false;
    if (llm_dsv41_engine_append(&e, (void**)&c0, prompt, 3, &off, &pend, &hp, &sm, s) != 0) { printf("FAIL append\n"); return 1; }
    int32_t cur = pend;
    int32_t next = step_argmax(&e, &c0, cur, &off, s);
    if (next < 0) { printf("FAIL step\n"); return 1; }
    /* snapshot dello stato dopo il commit di `cur` */
    LlmDsv41Cache* base = (LlmDsv41Cache*)malloc(sizeof *base);
    llm_dsv41_cache_clone(base, c0);
    int base_off = off;

    /* proposta = continuazione greedy reale del target */
    const int k = DSR_BLOCK;
    int32_t P[8]; P[0] = cur;
    LlmDsv41Cache* look = (LlmDsv41Cache*)malloc(sizeof *look);
    llm_dsv41_cache_clone(look, base);
    int loff = base_off; int32_t t = next;
    for (int j = 1; j <= k; j++) { P[j] = t; t = step_argmax(&e, &look, t, &loff, s); }
    int32_t t_after = t;   /* token target dopo il blocco */
    llm_dsv41_cache_free(look); free(look);

    int fail = 0;
    /* 1) proposta accurata -> accept = k, pending = token dopo il blocco */
    LlmDsv41Cache* c1 = (LlmDsv41Cache*)malloc(sizeof *c1);
    llm_dsv41_cache_clone(c1, base);
    int o1 = base_off; int32_t out1[8], pend1 = 0;
    int em1 = llm_dsv41_engine_mtp_speculate(&e, (void**)&c1, &o1, next, P, k, 8, -1, out1, &pend1, s);
    int ok1 = (em1 == k) && (pend1 == t_after);
    for (int i = 0; i < k && ok1; i++) if (out1[i] != P[i+1]) ok1 = 0;
    printf("%s %-46s (em=%d pend=%d exp=%d)\n", ok1?"PASS":"FAIL", "speculate accept=k", em1, pend1, t_after);
    if (!ok1) fail++;
    /* 2) corrompi l'ultimo token -> accept = k-1, rollback, pending = P[k] vero */
    int32_t bad = P[k]; P[k] = (P[k] + 1) % e.vocab;
    LlmDsv41Cache* c2 = (LlmDsv41Cache*)malloc(sizeof *c2);
    llm_dsv41_cache_clone(c2, base);
    int o2 = base_off; int32_t out2[8], pend2 = 0; long rb = e.mtp_rollbacks;
    int em2 = llm_dsv41_engine_mtp_speculate(&e, (void**)&c2, &o2, next, P, k, 8, -1, out2, &pend2, s);
    int ok2 = (em2 == k-1) && (pend2 == bad) && (e.mtp_rollbacks == rb + 1);
    for (int i = 0; i < k-1 && ok2; i++) if (out2[i] != P[i+1]) ok2 = 0;
    printf("%s %-46s (em=%d pend=%d exp=%d)\n", ok2?"PASS":"FAIL", "speculate rollback accept=k-1", em2, pend2, bad);
    if (!ok2) fail++;
    printf("mtp stats: accepted=%ld rollbacks=%ld\n", e.mtp_accepted, e.mtp_rollbacks);
    printf(fail?"FAILURES: %d\n":"ALL PASS\n", fail);
    return fail;
}
