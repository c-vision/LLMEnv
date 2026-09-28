/* llm_dsv4_moe_packed_test.c — MoE packed sul modello reale (layer 0): routing hash + esperti
 * IQ2_XXS/Q2_K via dot packed. Verifica finitezza/determinismo + coerenza con un esperto f32. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_moe_fwd.h"
#include "llm_gguf.h"
#include "llm_dsv4_weights.h"
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    static LlmEngineMoeGguf g; void* m = NULL; size_t ml = 0;
    if (!llm_gguf_open_file(path, &g, &m, &ml)) { char why[160]; fprintf(stderr, "open fallita: %s\n", llm_gguf_errmsg(&g, why, sizeof why)); return 2; }
    LlmDsv4Config c; llm_dsv4_config_from_gguf(&g, &c);
    LlmDsv4LayerWeights L; memset(&L, 0, sizeof L);
    bool ok = true;
    L.ffn_gate_inp = llm_dsv4_lw(&g, 0, "ffn_gate_inp", true, &ok);
    { char b[200]; snprintf(b,sizeof b,"blk.0.exp_probs_b.bias"); L.exp_probs_b = llm_gguf_load_f32(&g,b,NULL); }
    L.ffn_gate_tid2eid = llm_dsv4_lw(&g, 0, "ffn_gate_tid2eid", true, &ok);
    L.ffn_gate_shexp = llm_dsv4_lw(&g, 0, "ffn_gate_shexp", true, &ok);
    L.ffn_up_shexp = llm_dsv4_lw(&g, 0, "ffn_up_shexp", true, &ok);
    L.ffn_down_shexp = llm_dsv4_lw(&g, 0, "ffn_down_shexp", true, &ok);
    LlmDsv4PackedExperts pe; memset(&pe, 0, sizeof pe);
    pe.gate = llm_dsv4_bind_packed(&g, 0, "ffn_gate_exps", &pe.gate_type, &ok);
    pe.up = llm_dsv4_bind_packed(&g, 0, "ffn_up_exps", &pe.up_type, &ok);
    pe.down = llm_dsv4_bind_packed(&g, 0, "ffn_down_exps", &pe.down_type, &ok);
    pe.shexp_gate = llm_dsv4_bind_packed(&g, 0, "ffn_gate_shexp", &pe.shexp_gate_type, &ok);
    pe.shexp_up = llm_dsv4_bind_packed(&g, 0, "ffn_up_shexp", &pe.shexp_up_type, &ok);
    pe.shexp_down = llm_dsv4_bind_packed(&g, 0, "ffn_down_shexp", &pe.shexp_down_type, &ok);
    pe.in_dim = c.n_embd; pe.ff_dim = c.n_ff_exp;
    if (!ok) { fprintf(stderr, "bind fallito\n"); return 1; }
    printf("layer0: experts gate_type=%d up_type=%d down_type=%d (hash=%d)\n",
           pe.gate_type, pe.up_type, pe.down_type, c.n_hash_layer);

    float* x = (float*)malloc(sizeof(float) * (size_t)c.n_embd);
    for (int i = 0; i < c.n_embd; i++) x[i] = 0.05f*sinf((float)i*0.11f) - 0.02f;
    float* out = (float*)malloc(sizeof(float) * (size_t)c.n_embd);
    float* out2 = (float*)malloc(sizeof(float) * (size_t)c.n_embd);
    llm_dsv4_moe_forward_packed(&c, &L, &pe, x, 0, 1234, out);
    llm_dsv4_moe_forward_packed(&c, &L, &pe, x, 0, 1234, out2);
    int bad = 0; double nrm = 0;
    for (int i = 0; i < c.n_embd; i++) { if (!isfinite(out[i])) bad++; if (out[i] != out2[i]) bad++; nrm += (double)out[i]*out[i]; }
    printf("%s moe packed reale: finite/deterministico (|out|=%.4f, hash token 1234)\n", bad?"FAIL":"PASS", sqrt(nrm));

    /* coerenza: gate di un esperto scelto via dot packed vs dequant della prima riga */
    int sel[LLM_DSV4_MAX_EXPERT_USED]; float wt[LLM_DSV4_MAX_EXPERT_USED];
    llm_dsv4_moe_route(&c, &L, x, 0, 1234, sel, wt);
    int e = sel[0];
    int64_t nb_in = llm_ggml_nbytes(pe.gate_type, pe.in_dim);
    float d0; llm_dsv4_dot_tensor(pe.gate_type, pe.gate + (size_t)e*pe.ff_dim*nb_in, pe.in_dim, x, &d0);
    float* row = (float*)malloc(sizeof(float) * (size_t)(pe.ff_dim > 256 ? 256 : pe.ff_dim));
    llmcodec_ggml_dequant_row(pe.gate_type, pe.gate + (size_t)e*pe.ff_dim*nb_in, row, 256);
    double ref = 0; for (int k = 0; k < 256; k++) ref += (double)row[k] * x[k];  /* prime 256 dim */
    float d0p; llm_dsv4_dot_tensor(pe.gate_type, pe.gate + (size_t)e*pe.ff_dim*nb_in, 256, x, &d0p);
    printf("%s esperto %d gate row0: dot(n=256)=%.6f ref=%.6f\n", (fabsf(d0p-(float)ref)<1e-4f)?"PASS":"FAIL", e, d0p, (float)ref);
    (void)d0;
    free(row); free(out2); free(out); free(x);
    free(L.ffn_gate_inp); free(L.exp_probs_b); free(L.ffn_gate_tid2eid);
    free(L.ffn_gate_shexp); free(L.ffn_up_shexp); free(L.ffn_down_shexp);
    llm_gguf_close_file(&g, m, ml);
    printf(bad ? "FAILURES\n" : "ALL PASS\n");
    return bad ? 1 : 0;
}
