/* llm_dsv41_dspark_attn_test.c — verifica DSparkContextCache + DSparkAttention vs riferimento.
 * Build: tools/dsv41_dspark_attn_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_dspark_attn.h"
#include "llm_dsv41_dspark_attn_ref.h"
static int nclose(const float* a, const float* b, int n, float tol) {
    for (int i=0;i<n;i++){ float d=a[i]-b[i]; if(d<0)d=-d; float m=b[i]<0?-b[i]:b[i]; if(m<1.0f)m=1.0f;
        if (d>tol*m){ printf("  [%d] got=%.6g exp=%.6g\n", i,a[i],b[i]); return 0; } }
    return 1;
}
#define ARR(p, nd, ...) llm_dsv41_array_f32(p, nd, (const int[]){ __VA_ARGS__ })
static int check(int fail, const char* name, const float* got, const float* exp, int n) {
    if (nclose(got, exp, n, 2e-3f)) { printf("PASS %-46s\n", name); return fail; }
    printf("FAIL %-46s\n", name); return fail + 1;
}
int main(void) {
    printf("llm_dsv41_dspark_attn — context cache + draft attention\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmDsv41Config c; memset(&c, 0, sizeof c);
    c.dim=REF_DA_DIM; c.n_heads=REF_DA_H; c.head_dim=REF_DA_D; c.rope_head_dim=REF_DA_DR;
    c.o_groups=REF_DA_OG; c.o_lora_rank=REF_DA_OL; c.q_lora_rank=REF_DA_QD;
    c.norm_eps=1e-20f; c.rope_theta=10000.0f; c.original_seq_len=0; c.rope_factor=1.0f;
    c.beta_fast=32.0f; c.beta_slow=1.0f;
    int GRP=REF_DA_H*REF_DA_D/REF_DA_OG;
    LlmDsv41DSparkCache cache; memset(&cache,0,sizeof cache); cache.max_size=REF_DA_CAP;
    cache.keys = ARR(REF_DA_CTX, 4, 1,1,REF_DA_CTX_N,REF_DA_D); cache.offset=REF_DA_OFF; cache.has=true;
    mlx_array x = ARR(REF_DA_X, 3, 1,REF_DA_L,REF_DA_DIM);
    mlx_array wqa=ARR(REF_DA_WQA,2,REF_DA_QD,REF_DA_DIM), qn=ARR(REF_DA_QN,1,REF_DA_QD);
    mlx_array wqb=ARR(REF_DA_WQB,2,REF_DA_H*REF_DA_D,REF_DA_QD), wkv=ARR(REF_DA_WKV,2,REF_DA_D,REF_DA_DIM);
    mlx_array kvn=ARR(REF_DA_KVN,1,REF_DA_D), woa=ARR(REF_DA_WOA,2,REF_DA_OG*REF_DA_OL,GRP);
    mlx_array wob=ARR(REF_DA_WOB,2,REF_DA_DIM,REF_DA_OG*REF_DA_OL), sink=ARR(REF_DA_SINK,1,REF_DA_H);
    bool ok=true;
    int fail=0;

    mlx_array out = llm_dsv41_dspark_attention(x, &cache, &c, wqa,qn,wqb,wkv,kvn,woa,wob,sink, s, &ok);

    /* intermedi: q (pre/post rope), draft kv (post fp8), kv concat */
    mlx_array positions = mlx_array_new();
    mlx_arange(&positions, (double)cache.offset, (double)(cache.offset+REF_DA_L), 1.0, MLX_INT32, s);
    mlx_array q = llm_dsv41_mlx_linear(x, wqa, s, &ok);
    q = llm_dsv41_mlx_rmsnorm(q, qn, c.norm_eps, s, &ok);
    q = llm_dsv41_mlx_linear(q, wqb, s, &ok);
    mlx_reshape(&q, q, (const int[]){1,REF_DA_L,REF_DA_H,REF_DA_D}, 4, s);
    mlx_array_eval(q);
    fail = check(fail, "DSparkAttention -> q pre-rope", mlx_array_data_float32(q), REF_DA_QPRE, REF_DA_L*REF_DA_H*REF_DA_D);
    q = llm_dsv41_mlx_rope(q, positions, &c, 0, 0, s, &ok);
    mlx_array dk = llm_dsv41_mlx_linear(x, wkv, s, &ok);
    dk = llm_dsv41_mlx_rmsnorm(dk, kvn, c.norm_eps, s, &ok);
    dk = llm_dsv41_mlx_rope(dk, positions, &c, 0, 0, s, &ok);
    dk = llm_dsv41_mlx_fp8_roundtrip(dk, REF_DA_D, s, &ok);
    mlx_array_eval(q); mlx_array_eval(dk);
    fail = check(fail, "DSparkAttention -> q post-rope", mlx_array_data_float32(q), REF_DA_Q, REF_DA_L*REF_DA_H*REF_DA_D);
    fail = check(fail, "DSparkAttention -> draft kv (fp8)", mlx_array_data_float32(dk), REF_DA_DK, REF_DA_L*REF_DA_D);
    mlx_array ctx = llm_dsv41_mlx_slice_axis(cache.keys, 1, 0, 1, s, &ok);
    mlx_reshape(&ctx, ctx, (const int[]){1,-1,REF_DA_D}, 3, s);
    mlx_array kv = mlx_array_new();
    mlx_vector_array ops = mlx_vector_array_new();
    mlx_vector_array_append_value(ops, ctx); mlx_vector_array_append_value(ops, dk);
    mlx_concatenate_axis(&kv, ops, 1, s); mlx_vector_array_free(ops);
    mlx_array_eval(kv);
    fail = check(fail, "DSparkAttention -> kv concat", mlx_array_data_float32(kv), REF_DA_KV, (REF_DA_CTX_N+REF_DA_L)*REF_DA_D);

    mlx_array_eval(out);
    fail = check(fail, "DSparkAttention -> output", mlx_array_data_float32(out), REF_DA_OUT, REF_DA_L*REF_DA_DIM);

    /* ring fisico: 3 append da 2 su capacity 4, offset finale 6 */
    LlmDsv41DSparkCache c2; memset(&c2,0,sizeof c2); c2.max_size=REF_DA_CAP;
    llm_dspark_cache_append(&c2, ARR(REF_DA_CK0,4,1,1,2,REF_DA_D), s, &ok);
    llm_dspark_cache_append(&c2, ARR(REF_DA_CK1,4,1,1,2,REF_DA_D), s, &ok);
    llm_dspark_cache_append(&c2, ARR(REF_DA_CK2,4,1,1,2,REF_DA_D), s, &ok);
    mlx_array_eval(c2.keys);
    if (ok && c2.offset==REF_DA_CKOFF)
        fail = check(fail, "DSparkContextCache ring -> keys",
                     mlx_array_data_float32(c2.keys), REF_DA_CKEYS,
                     (int)mlx_array_shape(c2.keys)[2]*REF_DA_D);
    else { printf("FAIL %-46s\n", "DSparkContextCache ring -> offset"); fail++; }

    printf(fail?"FAILURES: %d\n":"ALL PASS\n", fail);
    return fail;
}
