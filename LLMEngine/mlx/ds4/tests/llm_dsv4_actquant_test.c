/* llm_dsv4_actquant_test.c — dot interi vs dequant, su blocchi reali del GGUF DS4. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_actquant.h"
#include "llm_gguf.h"
static int check(LlmEngineMoeGguf* g, const char* name) {
    int i = llm_gguf_find_tensor(g, name); if (i < 0) { printf("  %s ASSENTE\n", name); return 1; }
    const LlmCodecGgufTensor* t = &g->ti[i]; int type = t->type;
    const uint8_t* p = llm_gguf_tensor_data(g, i, NULL);
    float x[256]; for (int k = 0; k < 256; k++) x[k] = 0.4f*sinf((float)k*0.7f) - 0.1f;
    LlmDsv4Q8K y; llm_dsv4_q8_K_quantize_row(x, 256, &y);
    float dq = llm_dsv4_dot_w_q8_K(type, p, 256, &y);
    /* riferimento: y.d * sum value(k)*qs[k] con value da dequant */
    float w[256]; llmcodec_ggml_dequant_row(type, p, w, 256);
    double ref = 0; for (int k = 0; k < 256; k++) ref += (double)w[k] * (double)y.qs[k];
    ref *= y.d;
    float rel = fabsf(dq - (float)ref) / (fabsf((float)ref) + 1e-6f);
    printf("  %-46s type=%-3d dq=%.6f ref=%.6f rel=%.2e %s\n", name, type, dq, (float)ref, rel, rel<1e-4f?"OK":"FAIL");
    return rel < 1e-4f ? 0 : 1;
}
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    static LlmEngineMoeGguf g; void* m = NULL; size_t ml = 0;
    if (!llm_gguf_open_file(path, &g, &m, &ml)) { char why[160]; fprintf(stderr, "open fallita: %s\n", llm_gguf_errmsg(&g, why, sizeof why)); return 2; }
    int bad = 0;
    bad += check(&g, "blk.0.ffn_down_exps.weight");   /* Q2_K */
    bad += check(&g, "blk.0.ffn_gate_exps.weight");   /* IQ2_XXS */
    bad += check(&g, "blk.37.ffn_gate_exps.weight");  /* Q4_K (ultimi 6 layer) */
    bad += check(&g, "blk.37.ffn_down_exps.weight");  /* Q4_K */
    /* Q8_0 prequant vs dequant */
    { int i = llm_gguf_find_tensor(&g, "blk.0.attn_q_a.weight");
      const uint8_t* p = llm_gguf_tensor_data(&g, i, NULL);
      float x[256]; int8_t xq[256]; float xs[8];
      for (int k = 0; k < 256; k++) x[k] = 0.3f*sinf((float)k*0.5f);
      llm_dsv4_q8_0_quantize_row(x, 256, xq, xs);
      float dot = llm_dsv4_dot_q8_0_prequant(p, 256, xq, xs);
      float w[256]; llmcodec_ggml_dequant_row(g.ti[i].type, p, w, 256);
      double ref = 0; for (int b = 0; b < 8; b++) { int dsum = 0; for (int j = 0; j < 32; j++) dsum += (int)xq[b*32+j]*0; for (int j = 0; j < 32; j++) dsum += (int)(int8_t)0; (void)dsum; }
      /* confronto con i pesi dequantizzati e l'attivazione dequantizzata Q8_0 */
      double s2 = 0; for (int b = 0; b < 8; b++) for (int j = 0; j < 32; j++) s2 += (double)w[b*32+j] * (double)((float)xq[b*32+j]*xs[b]);
      float rel = fabsf(dot - (float)s2) / (fabsf((float)s2) + 1e-6f);
      (void)ref;
      printf("  %-46s rel=%.2e %s\n", "Q8_0 prequant dot", rel, rel<1e-4f?"OK":"FAIL");
      if (rel >= 1e-4f) bad++; }
    llm_gguf_close_file(&g, m, ml);
    printf(bad ? "FAILURES: %d\n" : "ALL PASS\n", bad);
    return bad ? 1 : 0;
}
