/* llm_dsv4_layer_real_test.c — layer N INTERO sul modello reale (attn/comp/indexer f32 + esperti
 * packed): bind, cache, forward cached su alcuni token. Verifica finitezza + determinismo.
 * Uso: <model.gguf> [layer_index] */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_layer.h"
#include "llm_dsv4_weights.h"
#include "llm_gguf.h"
int main(int argc, char** argv){
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int IL = argc > 2 ? atoi(argv[2]) : 0;
    static LlmEngineMoeGguf g; void* m=NULL; size_t ml=0;
    if(!llm_gguf_open_file(path,&g,&m,&ml)){ char why[160]; fprintf(stderr,"open fallita: %s\n", llm_gguf_errmsg(&g, why, sizeof why)); return 2; }
    LlmDsv4Config c; llm_dsv4_config_from_gguf(&g,&c);
    int ratio = llm_dsv4_layer_compress_ratio(&c, IL);
    LlmDsv4LayerWeights L; memset(&L,0,sizeof L); bool ok=true;
#define LW(s) llm_dsv4_lw(&g, IL, s, true, &ok)
    L.hc_attn_fn=LW("hc_attn_fn"); L.hc_attn_scale=LW("hc_attn_scale"); L.hc_attn_base=LW("hc_attn_base");
    L.attn_norm=LW("attn_norm"); L.attn_q_a=LW("attn_q_a"); L.attn_q_a_norm=LW("attn_q_a_norm");
    L.attn_q_b=LW("attn_q_b"); L.attn_kv=LW("attn_kv"); L.attn_kv_a_norm=LW("attn_kv_a_norm");
    L.attn_sinks=LW("attn_sinks"); L.attn_output_a=LW("attn_output_a"); L.attn_output_b=LW("attn_output_b");
    if(ratio!=0){ L.comp_ape=LW("attn_compressor_ape"); L.comp_kv=LW("attn_compressor_kv");
        L.comp_gate=LW("attn_compressor_gate"); L.comp_norm=LW("attn_compressor_norm"); }
    if(ratio==4){ L.idx_q_b=LW("indexer.attn_q_b"); L.idx_proj=LW("indexer.proj");
        L.idx_c_ape=LW("indexer_compressor_ape"); L.idx_c_kv=LW("indexer_compressor_kv");
        L.idx_c_gate=LW("indexer_compressor_gate"); L.idx_c_norm=LW("indexer_compressor_norm"); }
    L.hc_ffn_fn=LW("hc_ffn_fn"); L.hc_ffn_scale=LW("hc_ffn_scale"); L.hc_ffn_base=LW("hc_ffn_base");
    L.ffn_norm=LW("ffn_norm"); L.ffn_gate_inp=LW("ffn_gate_inp");
    { char b[200]; snprintf(b,sizeof b,"blk.%d.exp_probs_b.bias",IL); L.exp_probs_b=llm_gguf_load_f32(&g,b,NULL); }
    L.ffn_gate_shexp=LW("ffn_gate_shexp"); L.ffn_up_shexp=LW("ffn_up_shexp"); L.ffn_down_shexp=LW("ffn_down_shexp");
    if(IL < c.n_hash_layer) L.ffn_gate_tid2eid=LW("ffn_gate_tid2eid");
    LlmDsv4PackedExperts pe; memset(&pe,0,sizeof pe);
    pe.gate=llm_dsv4_bind_packed(&g,IL,"ffn_gate_exps",&pe.gate_type,&ok);
    pe.up=llm_dsv4_bind_packed(&g,IL,"ffn_up_exps",&pe.up_type,&ok);
    pe.down=llm_dsv4_bind_packed(&g,IL,"ffn_down_exps",&pe.down_type,&ok);
    pe.in_dim=c.n_embd; pe.ff_dim=c.n_ff_exp;
    if(!ok){ fprintf(stderr,"bind fallito\n"); return 1; }
    printf("layer %d reale (ratio=%d): embd=%d head=%d hd=%d hc=%d ff=%d exps=%d/%d\n",
           IL,ratio,c.n_embd,c.n_head,c.n_head_dim,c.n_hc,c.n_ff_exp,c.n_expert,c.n_expert_used);
    LlmDsv4LayerCache k; llm_dsv4_cache_init(&k,&c,IL,128,8);
    LlmDsv4LayerCache k2; llm_dsv4_cache_init(&k2,&c,IL,128,8);
    float* hc=(float*)malloc(sizeof(float)*c.n_hc*c.n_embd);
    for(int h=0;h<c.n_hc;h++) for(int i=0;i<c.n_embd;i++) hc[h*c.n_embd+i]=0.03f*sinf((float)(i+h*7)*0.13f)-0.01f;
    float* out=(float*)malloc(sizeof(float)*c.n_hc*c.n_embd);
    float* out2=(float*)malloc(sizeof(float)*c.n_hc*c.n_embd);
    int bad=0; double nrm=0; const int NP=9;
    for(int pos=0; pos<NP; pos++){
        llm_dsv4_layer_forward_cached_pe(&c,&L,&pe,&k,hc,500+pos,pos,IL,out);
        llm_dsv4_layer_forward_cached_pe(&c,&L,&pe,&k2,hc,500+pos,pos,IL,out2);
        for(int i=0;i<c.n_hc*c.n_embd;i++){ if(!isfinite(out[i])) bad++; if(out[i]!=out2[i]) bad++; nrm+=(double)out[i]*out[i]; }
    }
    printf("%s layer %d reale: finite+deterministico=%s |out|=%.4f n_comp=%d n_index=%d\n",
           bad?"FAIL":"PASS", IL, bad?"no":"si", sqrt(nrm), k.n_comp, k.n_index_comp);
    llm_dsv4_cache_free(&k2); llm_dsv4_cache_free(&k);
    free(out2); free(out); free(hc);
    llm_gguf_close_file(&g,m,ml);
    return bad?1:0;
}
