/* llm_dsv4_packed_test.c — dot packed vs dequant, sui blocchi reali del GGUF DS4. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_packed.h"
#include "llm_gguf.h"
static int check(LlmEngineMoeGguf* g, const char* name) {
    int i = llm_gguf_find_tensor(g, name);
    if (i < 0) { printf("  %-40s ASSENTE\n", name); return 1; }
    const LlmCodecGgufTensor* t = &g->ti[i];
    int type = t->type; int64_t n = t->dims[0];
    if (n > 256) n = 256;  /* un blocco / prima riga */
    const uint8_t* p = llm_gguf_tensor_data(g, i, NULL);
    float* y = (float*)malloc(sizeof(float) * (size_t)n);
    float* dq = (float*)malloc(sizeof(float) * (size_t)n);
    for (int64_t k = 0; k < n; k++) y[k] = 0.3f*sinf((float)k*0.7f) - 0.1f;
    if (llmcodec_ggml_dequant_row(type, p, dq, n) != 0) { printf("  %-40s dequant non supportato\n", name); free(y); free(dq); return 1; }
    float dot = 0; bool ok = llm_dsv4_dot_tensor(type, p, n, y, &dot);
    double ref = 0; for (int64_t k = 0; k < n; k++) ref += (double)dq[k] * y[k];
    float rel = fabsf(dot - (float)ref) / (fabsf((float)ref) + 1e-6f);
    printf("  %-46s type=%-3d n=%-4lld dot=%.6f ref=%.6f rel=%.2e %s\n", name, type, (long long)n,
           dot, (float)ref, rel, (ok && rel < 1e-4f) ? "OK" : "FAIL");
    free(y); free(dq);
    return (ok && rel < 1e-4f) ? 0 : 1;
}
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    static LlmEngineMoeGguf g; void* m = NULL; size_t ml = 0;
    if (!llm_gguf_open_file(path, &g, &m, &ml)) { char why[160]; fprintf(stderr, "open fallita: %s\n", llm_gguf_errmsg(&g, why, sizeof why)); return 2; }
    int bad = 0;
    bad += check(&g, "blk.0.attn_q_a.weight");        /* Q8_0 */
    bad += check(&g, "blk.0.ffn_down_exps.weight");   /* Q2_K (per-expert: dims [2048,4096,256]) */
    bad += check(&g, "blk.0.ffn_gate_exps.weight");   /* IQ2_XXS */
    bad += check(&g, "blk.0.attn_norm.weight");       /* F32 o F16 */
    bad += check(&g, "token_embd.weight");            /* F16 */
    llm_gguf_close_file(&g, m, ml);
    printf(bad ? "FAILURES: %d\n" : "ALL PASS\n", bad);
    return bad ? 1 : 0;
}
