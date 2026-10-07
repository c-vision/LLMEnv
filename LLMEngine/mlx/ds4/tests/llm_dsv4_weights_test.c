/* llm_dsv4_weights_test.c — verifica loader pesi V4 Flash (binding nomi) su GGUF sintetico. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "llm_dsv4_weights.h"
static unsigned char B[8192]; static size_t BP;
static void w_u32(unsigned v){ for(int i=0;i<4;i++) B[BP++]=(unsigned char)(v>>(8*i)); }
static void w_u64(uint64_t v){ for(int i=0;i<8;i++) B[BP++]=(unsigned char)(v>>(8*i)); }
static void w_str(const char* s){ size_t n=strlen(s); w_u64(n); memcpy(B+BP,s,n); BP+=n; }
static void w_kv_u32(const char* k, unsigned v){ w_str(k); w_u32(4); w_u32(v); }
static void w_ti1(const char* nm){ w_str(nm); w_u32(1); w_u64(1); w_u32(0); w_u64(0); }
static const char* L0[] = {
    "hc_attn_fn","hc_attn_scale","hc_attn_base","attn_norm","attn_q_a","attn_q_a_norm","attn_q_b",
    "attn_kv","attn_kv_a_norm","attn_sinks","attn_output_a","attn_output_b",
    "hc_ffn_fn","hc_ffn_scale","hc_ffn_base","ffn_norm","ffn_gate_inp","ffn_gate_exps","ffn_up_exps",
    "ffn_down_exps","ffn_gate_shexp","ffn_up_shexp","ffn_down_shexp","ffn_gate_tid2eid" };
static const char* GLB[] = { "token_embd.weight","output_norm.weight","output.weight",
    "output_hc_base.weight","output_hc_fn.weight","output_hc_scale.weight" };
int main(void) {
    BP=0; memcpy(B,"GGUF",4); BP=4;
    w_u32(3); w_u64(6 + 24 + 1); w_u64(2);   /* n_tensors, n_kv */
    w_kv_u32("deepseek4.block_count", 1);
    w_kv_u32("deepseek4.hash_layer_count", 1);
    for (int i=0;i<6;i++) w_ti1(GLB[i]);
    for (int i=0;i<24;i++){ char b[200]; snprintf(b,sizeof b,"blk.0.%s.weight",L0[i]); w_ti1(b); }
    { char b[200]; snprintf(b,sizeof b,"blk.0.exp_probs_b.bias"); w_ti1(b); }
    while (BP%32) B[BP++]=0;
    { float f=1.25f; memcpy(B+BP,&f,4); for(int i=1;i<64;i++) B[BP+i*4]=0; BP += 256; }
    LlmEngineMoeGguf g; int fail=0;
    if(!llm_gguf_parse(&g,B,BP)){ printf("FAIL parse\n"); return 1; }
    int ok = (g.nti == 31);
    printf("%s %-40s\n", ok?"PASS":"FAIL", "gguf sintetico 31 tensori"); if(!ok) fail++;
    LlmDsv4Weights w;
    if(!llm_dsv4_load(&w,&g)){ printf("FAIL load\n"); return 1; }
    ok = (w.cfg.n_layer==1 && w.token_embd && w.output && w.layers &&
          w.layers[0].hc_attn_fn && w.layers[0].attn_q_a && w.layers[0].ffn_down_exps &&
          w.layers[0].ffn_gate_tid2eid);
    printf("%s %-40s\n", ok?"PASS":"FAIL", "load: campi richiesti presenti"); if(!ok) fail++;
    ok = (w.layers[0].comp_kv == NULL) && (w.layers[0].idx_q_b == NULL);
    printf("%s %-40s\n", ok?"PASS":"FAIL", "load: ratio0 -> niente compressor/indexer"); if(!ok) fail++;
    ok = (w.token_embd[0] == 1.25f);
    printf("%s %-40s\n", ok?"PASS":"FAIL", "load: valore dequant F32"); if(!ok) fail++;
    llm_dsv4_weights_free(&w);
    printf(fail?"FAILURES: %d\n":"ALL PASS\n", fail);
    return fail;
}
