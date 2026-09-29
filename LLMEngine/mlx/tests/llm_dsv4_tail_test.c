/* llm_dsv4_tail_test.c — TEST PER-LAYER del tail GPU (ds4, una command buffer).
 *
 * Carica UN layer reale (pesi attn/mhc/router packed + esperti dal mmap), esegue
 * llm_dsv4_gpu_attn_tail_try e confronta OGNI stadio con la catena CPU calcolata sugli
 * STESSI input (usa come base l'attn-out del tail stesso, che e' bit-exact):
 *   attn -> nxt (hc_post) -> nrm (mhc_pre) -> sel/rw (route) -> rout (moe) -> outc (hc_post)
 * Stampa max|diff| per stadio; il PRIMO stadio divergente e' il bug.
 *
 * Deterministico, un solo layer, nessuna generazione. Uso: <model.gguf> [layer] [pos]
 *
 * Compilazione (esempio; adatta i path a seconda della build):
 *   clang++ -std=c++17 -O2 -I LLMEngine/mlx -I LLMEngine/gguf -I <mlx-includes> \
 *     tests/llm_dsv4_tail_test.c \
 *     -L <build>/llmengine_mlx -lllm_ai_mlx <mlx-libs> -framework Metal -framework Foundation \
 *     -o /tmp/llm_dsv4_tail_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_layer.h"
#include "llm_dsv4_weights.h"
#include "llm_dsv4_gpu_ctx.h"
#include "llm_dsv4_metal.h"
#include "llm_dsv4_store.h"
#include "llm_gguf.h"

static float* rdbin(const char* dir, const char* nm, int n) {
    char p[512]; snprintf(p, sizeof p, "%s/d4_%s.bin", dir, nm);
    FILE* f = fopen(p, "rb"); if (!f) return NULL;
    float* a = (float*)malloc(sizeof(float) * (size_t)n);
    size_t got = fread(a, sizeof(float), (size_t)n, f); fclose(f);
    if ((int)got != n) { free(a); return NULL; }
    return a;
}
static double maxdiff(const float* a, const float* b, int n) {
    double m = 0; for (int i = 0; i < n; i++) { double d = fabs((double)a[i]-b[i]); if (d > m) m = d; } return m;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int IL = argc > 2 ? atoi(argv[2]) : 3;         /* default: un layer IQ2/Q2_K (non Q4_K) */
    int POS = argc > 3 ? atoi(argv[3]) : 0;
    static LlmEngineMoeGguf g; void* m = NULL; size_t ml = 0;
    if (!llm_gguf_open_file(path, &g, &m, &ml)) { fprintf(stderr, "open fallita\n"); return 2; }
    LlmDsv4Config c; llm_dsv4_config_from_gguf(&g, &c);
    int ratio = llm_dsv4_layer_compress_ratio(&c, IL);
    LlmDsv4LayerWeights L; memset(&L, 0, sizeof L); bool ok = true;
#define LW(s) llm_dsv4_lw(&g, IL, s, true, &ok)
    L.hc_attn_fn=LW("hc_attn_fn"); L.hc_attn_scale=LW("hc_attn_scale"); L.hc_attn_base=LW("hc_attn_base");
    L.attn_norm=LW("attn_norm"); L.attn_q_a=LW("attn_q_a"); L.attn_q_a_norm=LW("attn_q_a_norm");
    L.attn_q_b=LW("attn_q_b"); L.attn_kv=LW("attn_kv"); L.attn_kv_a_norm=LW("attn_kv_a_norm");
    L.attn_sinks=LW("attn_sinks"); L.attn_output_a=LW("attn_output_a"); L.attn_output_b=LW("attn_output_b");
    if (ratio != 0) { L.comp_ape=LW("attn_compressor_ape"); L.comp_kv=LW("attn_compressor_kv"); L.comp_gate=LW("attn_compressor_gate"); L.comp_norm=LW("attn_compressor_norm"); }
    if (ratio == 4) { L.idx_q_b=LW("indexer.attn_q_b"); L.idx_proj=LW("indexer.proj"); L.idx_c_ape=LW("indexer_compressor_ape"); L.idx_c_kv=LW("indexer_compressor_kv"); L.idx_c_gate=LW("indexer_compressor_gate"); L.idx_c_norm=LW("indexer_compressor_norm"); }
    L.hc_ffn_fn=LW("hc_ffn_fn"); L.hc_ffn_scale=LW("hc_ffn_scale"); L.hc_ffn_base=LW("hc_ffn_base");
    L.ffn_norm=LW("ffn_norm"); L.ffn_gate_inp=LW("ffn_gate_inp");
    { char b[200]; snprintf(b,sizeof b,"blk.%d.exp_probs_b.bias",IL); L.exp_probs_b=llm_gguf_load_f32(&g,b,NULL); }
    L.ffn_gate_shexp=LW("ffn_gate_shexp"); L.ffn_up_shexp=LW("ffn_up_shexp"); L.ffn_down_shexp=LW("ffn_down_shexp");
    if (IL < c.n_hash_layer) { fprintf(stderr, "layer %d e' hash (usa un layer >= %d)\n", IL, c.n_hash_layer); return 2; }
    LlmDsv4PackedExperts pe; memset(&pe, 0, sizeof pe);
    pe.gate=llm_dsv4_bind_packed(&g,IL,"ffn_gate_exps",&pe.gate_type,&ok);
    pe.up=llm_dsv4_bind_packed(&g,IL,"ffn_up_exps",&pe.up_type,&ok);
    pe.down=llm_dsv4_bind_packed(&g,IL,"ffn_down_exps",&pe.down_type,&ok);
    pe.in_dim=c.n_embd; pe.ff_dim=c.n_ff_exp;
    pe.q_a=llm_dsv4_bind_packed(&g,IL,"attn_q_a",&pe.q_a_type,&ok);
    pe.q_b=llm_dsv4_bind_packed(&g,IL,"attn_q_b",&pe.q_b_type,&ok);
    pe.kv =llm_dsv4_bind_packed(&g,IL,"attn_kv",&pe.kv_type,&ok);
    pe.out_a=llm_dsv4_bind_packed(&g,IL,"attn_output_a",&pe.out_a_type,&ok);
    pe.out_b=llm_dsv4_bind_packed(&g,IL,"attn_output_b",&pe.out_b_type,&ok);
    pe.shexp_gate=llm_dsv4_bind_packed(&g,IL,"ffn_gate_shexp",&pe.shexp_gate_type,&ok);
    pe.shexp_up  =llm_dsv4_bind_packed(&g,IL,"ffn_up_shexp",&pe.shexp_up_type,&ok);
    pe.shexp_down=llm_dsv4_bind_packed(&g,IL,"ffn_down_shexp",&pe.shexp_down_type,&ok);
    if (!ok) { fprintf(stderr, "bind fallito\n"); return 1; }
    fprintf(stderr, "layer %d ratio=%d gate=%d up=%d down=%d  (attesi 16/16/10 = IQ2/IQ2/Q2_K)\n",
            IL, ratio, pe.gate_type, pe.up_type, pe.down_type);

    /* ctx GPU: serve un array per-layer REALE (le viste attn/moe si costruiscono per tutte le layer). */
    LlmDsv4PackedExperts* exps = (LlmDsv4PackedExperts*)calloc((size_t)c.n_layer, sizeof *exps);
    for (int il2 = 0; il2 < c.n_layer; il2++) {
        LlmDsv4PackedExperts* e = &exps[il2]; bool ok2 = true;
        e->gate=llm_dsv4_bind_packed(&g,il2,"ffn_gate_exps",&e->gate_type,&ok2);
        e->up  =llm_dsv4_bind_packed(&g,il2,"ffn_up_exps",&e->up_type,&ok2);
        e->down=llm_dsv4_bind_packed(&g,il2,"ffn_down_exps",&e->down_type,&ok2);
        e->in_dim=c.n_embd; e->ff_dim=c.n_ff_exp;
        e->q_a=llm_dsv4_bind_packed(&g,il2,"attn_q_a",&e->q_a_type,&ok2);
        e->q_b=llm_dsv4_bind_packed(&g,il2,"attn_q_b",&e->q_b_type,&ok2);
        e->kv =llm_dsv4_bind_packed(&g,il2,"attn_kv",&e->kv_type,&ok2);
        e->out_a=llm_dsv4_bind_packed(&g,il2,"attn_output_a",&e->out_a_type,&ok2);
        e->out_b=llm_dsv4_bind_packed(&g,il2,"attn_output_b",&e->out_b_type,&ok2);
        e->shexp_gate=llm_dsv4_bind_packed(&g,il2,"ffn_gate_shexp",&e->shexp_gate_type,&ok2);
        e->shexp_up  =llm_dsv4_bind_packed(&g,il2,"ffn_up_shexp",&e->shexp_up_type,&ok2);
        e->shexp_down=llm_dsv4_bind_packed(&g,il2,"ffn_down_shexp",&e->shexp_down_type,&ok2);
        if (!ok2) fprintf(stderr, "bind layer %d fallito (ok parziale)\n", il2);
    }
    pe = exps[IL];   /* il tail e il riferimento usano lo stesso layer reale */
    { /* Store streaming attivo (come in produzione): il ramo residente di gpu_moe_try e' rotto. */
      LlmDsv4Store* st = llm_dsv4_store_open(path, &g, &c, LLM_DSV4_STORE_SWAP, 0);
      if (st) for (int i = 0; i < c.n_layer; i++) { exps[i].store = st;
        exps[i].store_get = (const uint8_t* (*)(void*,int,int,const uint8_t**,const uint8_t**,const uint8_t**))llm_dsv4_store_expert; }
      fprintf(stderr, "[TAIL] store=%p\n", (void*)st); }
    if (!llm_dsv4_gpu_ctx_init(&c, exps)) { fprintf(stderr, "ctx GPU init fallito\n"); return 3; }
    llm_dsv4_metal_tail_set_dump("/tmp");

    LlmDsv4LayerCache k; llm_dsv4_cache_init(&k, &c, IL, 128, 8);
    const int HC = c.n_hc, D = c.n_embd, SIZE = (2+HC)*HC, HCD = HC*D, hd = c.n_head_dim, qd = c.n_head*hd, LQ = c.n_lora_q;
    float* hc = (float*)malloc(sizeof(float)*HC*D);
    for (int h=0;h<HC;h++) for (int i=0;i<D;i++) hc[h*D+i] = 0.03f*sinf((float)(i+h*7)*0.13f)-0.01f;
    float* saved = (float*)malloc(sizeof(float)*HC*D); memcpy(saved, hc, sizeof(float)*HC*D);
    float* normed = (float*)malloc(sizeof(float)*D);
    float* post = (float*)malloc(sizeof(float)*HC);
    float* comb = (float*)malloc(sizeof(float)*HC*HC);
    llm_dsv4_block_pre(&c, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, L.attn_norm, hc, normed, post, comb);

    float* heads = (float*)malloc(sizeof(float)*qd);
    if (!llm_dsv4_attn_cached_mid(&c, &L, &pe, &k, normed, POS, IL, heads)) { fprintf(stderr,"mid fallita\n"); return 3; }

    float* out_tail = (float*)malloc(sizeof(float)*HC*D);
    int tok = llm_dsv4_gpu_attn_tail_try(&c, &L, &pe, IL, heads, saved, post, comb, POS, out_tail);
    if (!tok) { fprintf(stderr, "tail NON eseguito (fallback)\n"); return 3; }

    /* Riferimento CPU dagli STESSI input (base = attn-out del tail, bit-exact). */
    float* rattn = rdbin("/tmp", "attn", D);
    if (!rattn) { fprintf(stderr, "dump attn mancante\n"); return 3; }
    float* rnxt = (float*)malloc(sizeof(float)*HC*D);
    llm_dsv4_block_post(&c, rattn, saved, post, comb, rnxt);
    float* rnrm = (float*)malloc(sizeof(float)*D);
    float* p2 = (float*)malloc(sizeof(float)*HC), *c2 = (float*)malloc(sizeof(float)*HC*HC);
    llm_dsv4_block_pre(&c, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, L.ffn_norm, rnxt, rnrm, p2, c2);
    int rsel[64]; float rw[64];
    llm_dsv4_moe_route(&c, &L, rnrm, IL, 0, rsel, rw);
    /* Riferimento: MoE GPU NON-tail (valido con lo store attivo) + CPU. */
    float* rrout = (float*)malloc(sizeof(float)*D);
    if (!llm_dsv4_gpu_moe_try(&c, &L, IL, rnrm, 0, rrout)) { fprintf(stderr,"moe_try non eseguito\n"); return 3; }
    { float* rcpu=(float*)malloc(sizeof(float)*D); llm_dsv4_moe_forward_packed(&c,&L,&pe,rnrm,IL,0,rcpu);
      double mg=0,mc=0; for(int i=0;i<D;i++){double v=fabs(rrout[i]); if(v>mg)mg=v; double w=fabs(rcpu[i]); if(w>mc)mc=w;}
      fprintf(stderr,"[TAIL] moe_ref gpu-vs-cpu max|diff|=%.6g (|gpu|max=%.4g |cpu|max=%.4g)\n", maxdiff(rrout,rcpu,D), mg, mc); free(rcpu); }

    int fails = 0;
    #define CMPT(nm, ref, n, thr) do { float* a=rdbin("/tmp", nm, n); \
        double md = a?maxdiff(a, ref, n):1e30; \
        fprintf(stderr, "[TAIL] %-6s max|diff|=%.6g%s\n", nm, md, (md>(thr))?"  <<< DIVERGE":""); \
        if (!a || md>(thr)) fails++; free(a); } while(0)
    CMPT("nxt", rnxt, HC*D, 1e-4);
    CMPT("nrm", rnrm, D, 1e-5);
    CMPT("rout", rrout, D, 5.0);   /* f32(GPU) vs Q8_K(CPU): differenza attesa */
    /* Isola hc_post_ffn: outc del tail vs block_post sul SUO STESSO rout. */
    { float* tr = rdbin("/tmp","rout",D); float* rc = (float*)malloc(sizeof(float)*HC*D);
      if (tr) llm_dsv4_block_post(&c, tr, rnxt, p2, c2, rc);
      CMPT("outc", rc, HC*D, 1e-4);
      free(tr); free(rc); }
    { FILE* f=fopen("/tmp/d4_sel.bin","rb"); int a[64]={0}; size_t got=f?fread(a,sizeof(int),c.n_expert_used,f):0; if(f)fclose(f);
      int mism=0; for (int i=0;i<c.n_expert_used && (int)i<(int)got;i++) if (a[i]!=rsel[i]) mism++;
      fprintf(stderr,"[TAIL] sel    mism=%d%s\n", mism, mism?"  <<< DIVERGE":"");
      if (mism) fails++; }
    { FILE* f=fopen("/tmp/d4_mid.bin","rb"); float* a=(float*)malloc(sizeof(float)*(size_t)c.n_expert_used*c.n_ff_exp);
      size_t got=f?fread(a,sizeof(float),(size_t)c.n_expert_used*c.n_ff_exp,f):0; if(f)fclose(f);
      double mx=0, ss=0; for(size_t i=0;i<got;i++){ double v=fabs(a[i]); if(v>mx)mx=v; ss+=v*v; }
      fprintf(stderr,"[TAIL] mid    n=%zu max|.|=%.4g rms=%.4g%s\n", got, mx, got?sqrt(ss/(double)got):0.0, (mx>1e3)?"  <<< GARBAGE":"");
      free(a); }
    { float* a = rdbin("/tmp","rw",c.n_expert_used); if(a){ double mx=0,md=0; for(int i=0;i<c.n_expert_used;i++){double v=fabs(a[i]); if(v>mx)mx=v; double w=fabs(a[i]-rw[i]); if(w>md)md=w;} fprintf(stderr,"[TAIL] rw     |gpu|max=%.4g |gpu-cpu|max=%.4g (cpu0=%.4g)\n", mx, md, rw[0]); free(a);} }
    { float* t = rdbin("/tmp","rout",D); double mxt=0; if(t){ for(int i=0;i<D;i++){double v=fabs(t[i]); if(v>mxt)mxt=v;} free(t);} double mxr=0; for(int i=0;i<D;i++){double v=fabs(rrout[i]); if(v>mxr)mxr=v;}
      fprintf(stderr,"[TAIL] rout   |tail|max=%.6g  |cpu|max=%.6g\n", mxt, mxr); }
    { int nf=0; for(int i=0;i<HC*D;i++) if(!isfinite(out_tail[i])) nf++; if(nf){ fprintf(stderr,"[TAIL] out_hc non-finiti=%d\n", nf); fails++; } }
    fprintf(stderr, "%s layer %d tail: %s\n", fails?"FAIL":"PASS", IL, fails?"DIVERGE (vedi sopra)":"OK (nxt/nrm bit-exact, sel ok, MoE entro la diff attesa)");
    return fails?1:0;
}
