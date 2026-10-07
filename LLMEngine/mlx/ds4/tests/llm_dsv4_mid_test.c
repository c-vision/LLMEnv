/* llm_dsv4_mid_test.c — TEST d'integrazione del MIDDLE attenzione su GPU.
 *
 * Su UN layer reale, per una sequenza di posizioni, confronta `heads` del path CPU
 * (`llm_dsv4_attn_cached_mid`) con `llm_dsv4_metal_attn_mid` (compressor+indexer+attn su device,
 * stato KV su device). Proiezione via CPU (`llm_dsv4_attn_project`) per entrambi.
 *
 * Uso: <model.gguf> [layer] [n_pos]. Deterministico.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_cache.h"
#include "llm_dsv4_attn_fwd.h"
#include "llm_dsv4_gpu_ctx.h"
#include "llm_dsv4_metal.h"
#include "llm_dsv4_weights.h"
#include "llm_gguf.h"

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int IL = argc > 2 ? atoi(argv[2]) : 3;
    int NP = argc > 3 ? atoi(argv[3]) : 12;
    if (!llm_dsv4_metal_available()) { printf("SKIP: nessun device Metal\n"); return 0; }
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
    if (!ok) { fprintf(stderr, "load pesi fallito\n"); return 1; }
    fprintf(stderr, "layer %d ratio=%d embd=%d hd=%d ihd=%d nh_idx=%d top_k=%d\n",
            IL, ratio, c.n_embd, c.n_head_dim, c.n_indexer_head_dim, c.n_indexer_head, c.n_indexer_top_k);
    if (ratio == 0) { fprintf(stderr, "layer senza compressor: niente middle GPU (skip)\n"); return 0; }

    const int embd=c.n_embd, hd=c.n_head_dim, ihd=c.n_indexer_head_dim, co=(ratio==4)?2:1;
    const int width=co*hd, iwidth=co*ihd, qdim=c.n_indexer_head*c.n_indexer_head_dim;
    const int raw_cap=128, comp_cap=64;
    LlmDsv4Metal* bm = llm_dsv4_metal_create();
    if (!bm) { fprintf(stderr, "metal create fallito\n"); return 3; }

    /* pesi f32 del compressor/indexer su device (upload una volta) */
    void* dw_ckv  = llm_dsv4_metal_buf_alloc(bm, (size_t)width*embd*4);   llm_dsv4_metal_buf_write(dw_ckv,0,L.comp_kv,(size_t)width*embd*4);
    void* dw_cgate= llm_dsv4_metal_buf_alloc(bm, (size_t)width*embd*4);   llm_dsv4_metal_buf_write(dw_cgate,0,L.comp_gate,(size_t)width*embd*4);
    void* dw_cape = llm_dsv4_metal_buf_alloc(bm, (size_t)ratio*width*4);  llm_dsv4_metal_buf_write(dw_cape,0,L.comp_ape,(size_t)ratio*width*4);
    void* dw_cnorm= llm_dsv4_metal_buf_alloc(bm, (size_t)hd*4);           llm_dsv4_metal_buf_write(dw_cnorm,0,L.comp_norm,(size_t)hd*4);
    void* dw_sinks= llm_dsv4_metal_buf_alloc(bm, (size_t)c.n_head*4);     llm_dsv4_metal_buf_write(dw_sinks,0,L.attn_sinks,(size_t)c.n_head*4);
    void* dw_ickv=NULL,*dw_icgate=NULL,*dw_icape=NULL,*dw_icnorm=NULL,*dw_iqb=NULL,*dw_iproj=NULL;
    if (ratio==4) {
        dw_ickv  = llm_dsv4_metal_buf_alloc(bm,(size_t)iwidth*embd*4); llm_dsv4_metal_buf_write(dw_ickv,0,L.idx_c_kv,(size_t)iwidth*embd*4);
        dw_icgate= llm_dsv4_metal_buf_alloc(bm,(size_t)iwidth*embd*4); llm_dsv4_metal_buf_write(dw_icgate,0,L.idx_c_gate,(size_t)iwidth*embd*4);
        dw_icape = llm_dsv4_metal_buf_alloc(bm,(size_t)ratio*iwidth*4); llm_dsv4_metal_buf_write(dw_icape,0,L.idx_c_ape,(size_t)ratio*iwidth*4);
        dw_icnorm= llm_dsv4_metal_buf_alloc(bm,(size_t)ihd*4); llm_dsv4_metal_buf_write(dw_icnorm,0,L.idx_c_norm,(size_t)ihd*4);
        dw_iqb   = llm_dsv4_metal_buf_alloc(bm,(size_t)qdim*c.n_lora_q*4); llm_dsv4_metal_buf_write(dw_iqb,0,L.idx_q_b,(size_t)qdim*c.n_lora_q*4);
        dw_iproj = llm_dsv4_metal_buf_alloc(bm,(size_t)c.n_indexer_head*embd*4); llm_dsv4_metal_buf_write(dw_iproj,0,L.idx_proj,(size_t)c.n_indexer_head*embd*4);
    }
    /* stato + KV su device */
    void* d_akv = llm_dsv4_metal_buf_alloc(bm,(size_t)co*ratio*width*4);
    void* d_asc = llm_dsv4_metal_buf_alloc(bm,(size_t)co*ratio*width*4);
    void* d_ikv = llm_dsv4_metal_buf_alloc(bm,(size_t)(ratio==4?co*ratio*iwidth:1)*4);
    void* d_isc = llm_dsv4_metal_buf_alloc(bm,(size_t)(ratio==4?co*ratio*iwidth:1)*4);
    void* d_raw = llm_dsv4_metal_buf_alloc(bm,(size_t)raw_cap*hd*4);
    void* d_acomp = llm_dsv4_metal_buf_alloc(bm,(size_t)comp_cap*hd*4);
    void* d_icomp = llm_dsv4_metal_buf_alloc(bm,(size_t)comp_cap*ihd*4);
    void* d_allowed = llm_dsv4_metal_buf_alloc(bm,(size_t)comp_cap*sizeof(int));
    if (!dw_ckv||!dw_cgate||!dw_cape||!dw_cnorm||!dw_sinks||!d_akv||!d_asc||!d_raw||!d_acomp) { fprintf(stderr,"alloc fallita\n"); return 3; }
    /* stato iniziale = come calloc della cache: kv=0, score=-1e30 (i buffer Metal NON sono zero) */
    { int n=co*ratio*width; float* z=(float*)calloc((size_t)n,sizeof(float)); llm_dsv4_metal_buf_write(d_akv,0,z,(size_t)n*4); free(z);
      float* s=(float*)malloc(sizeof(float)*(size_t)n); for(int i=0;i<n;i++) s[i]=-1.0e30f; llm_dsv4_metal_buf_write(d_asc,0,s,(size_t)n*4); free(s); }
    if (ratio==4) { int n=co*ratio*iwidth; float* z=(float*)calloc((size_t)n,sizeof(float)); llm_dsv4_metal_buf_write(d_ikv,0,z,(size_t)n*4); free(z);
      float* s=(float*)malloc(sizeof(float)*(size_t)n); for(int i=0;i<n;i++) s[i]=-1.0e30f; llm_dsv4_metal_buf_write(d_isc,0,s,(size_t)n*4); free(s); }

    /* rope cfg (come il tail) */
    int compressed = ratio != 0;
    float freq_base = (compressed && c.compress_rope_freq_base > 0.0f) ? c.compress_rope_freq_base : c.rope_freq_base;
    float freq_scale = (!compressed || c.rope_scale_factor <= 0.0f) ? 1.0f : 1.0f / c.rope_scale_factor;
    float ext_factor = (compressed && c.rope_scale_factor > 1.0f) ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);

    LlmDsv4LayerCache k; llm_dsv4_cache_init(&k, &c, IL, raw_cap, comp_cap);
    float* xn  = (float*)malloc(sizeof(float)*embd);
    for (int i=0;i<embd;i++) xn[i] = 0.05f*sinf((float)i*0.017f) - 0.02f*cosf((float)i*0.031f);
    float* q   = (float*)malloc(sizeof(float)*c.n_head*hd);
    float* kv  = (float*)malloc(sizeof(float)*hd);
    float* qrn = (float*)malloc(sizeof(float)*c.n_lora_q);
    float* hcpu= (float*)malloc(sizeof(float)*c.n_head*hd);
    float* hgpu= (float*)malloc(sizeof(float)*c.n_head*hd);

    int n_raw=0, n_acomp=0, n_icomp=0, bad=0; double worst=0.0;
    for (int pos=0; pos<NP; pos++) {
        if (!llm_dsv4_attn_cached_mid(&c, &L, NULL, &k, xn, pos, IL, hcpu)) { fprintf(stderr,"cpu mid fallita\n"); return 4; }
        llm_dsv4_attn_project(&c, &L, NULL, xn, pos, IL, q, kv, qrn);
        int due = ((pos+1)%ratio)==0, pos_mod = pos%ratio;
        int state_row = (ratio==4)? (ratio+pos_mod) : pos_mod;
        int raw_full = (n_raw >= raw_cap);
        int raw_off = raw_full ? (raw_cap-1)*hd : n_raw*hd;
        int na = (due && n_acomp<comp_cap) ? n_acomp+1 : n_acomp;
        int ni = (due && n_icomp<comp_cap) ? n_icomp+1 : n_icomp;
        int nr = raw_full ? raw_cap : n_raw+1;
        LlmDsv4MidParams p; memset(&p,0,sizeof p);
        p.embd=embd; p.hd=hd; p.ihd=ihd; p.ratio=ratio; p.co=co;
        p.n_head=c.n_head; p.n_rot=c.n_rot; p.rms_eps=c.rms_eps;
        p.idx_nh=c.n_indexer_head; p.idx_hd=ihd; p.idx_top_k=c.n_indexer_top_k; p.lq=c.n_lora_q; p.qdim=qdim;
        p.pos=pos; p.due=due; p.state_row=state_row; p.ape_off=pos_mod; p.has_idx=(ratio==4);
        p.push_acomp=(due && n_acomp<comp_cap); p.push_icomp=(due && n_icomp<comp_cap);
        p.rope_cfg[0]=freq_base; p.rope_cfg[1]=freq_scale; p.rope_cfg[2]=ext_factor; p.rope_cfg[3]=attn_factor;
        p.rope_cfg[4]=c.rope_yarn_beta_fast; p.rope_cfg[5]=c.rope_yarn_beta_slow; p.rope_cfg[6]=(float)(compressed?c.rope_orig_ctx:0);
        p.w_ckv=dw_ckv; p.w_cgate=dw_cgate; p.w_cape=dw_cape; p.w_cnorm=dw_cnorm;
        p.w_ickv=dw_ickv; p.w_icgate=dw_icgate; p.w_icape=dw_icape; p.w_icnorm=dw_icnorm;
        p.w_iqb=dw_iqb; p.w_iproj=dw_iproj; p.sinks=dw_sinks;
        p.st_akv=d_akv; p.st_asc=d_asc; p.st_ikv=d_ikv; p.st_isc=d_isc;
        p.raw=d_raw; p.raw_cap=raw_cap; p.raw_off=raw_off; p.raw_full=raw_full;
        p.acomp=d_acomp; p.acomp_off=n_acomp*hd; p.icomp=d_icomp; p.icomp_off=n_icomp*ihd;
        p.allowed=d_allowed;
        p.xn=xn; p.qrn=qrn; p.q=q; p.kv_cur=kv;
        p.n_raw=nr; p.n_acomp=na; p.n_icomp=ni;
        p.heads=hgpu;
        int rc = llm_dsv4_metal_attn_mid(bm, &p);
        double md=0.0; for (int i=0;i<c.n_head*hd;i++){ double d=fabs((double)hcpu[i]-hgpu[i]); if(d>md)md=d; }
        if (md>worst) worst=md;
        int okp = (rc==0 && md<1e-3);
        if (due && na-1 < comp_cap) {   /* verifica extra: riga compressa device == CPU */
            float* cg=(float*)malloc(sizeof(float)*hd);
            llm_dsv4_metal_buf_read(d_acomp,(size_t)(na-1)*hd*4,cg,(size_t)hd*4);
            double cm=0.0; for(int i=0;i<hd;i++){ double d=fabs((double)k.attn_comp[(size_t)(na-1)*hd+i]-cg[i]); if(d>cm)cm=d; }
            if (cm>1e-3) { printf("        comp[%d] max|diff|=%.3e BAD\n", na-1, cm); okp=0; }
            free(cg);
        }
        if (!okp) bad++;
        printf("[MID] pos=%2d due=%d raw=%d comp=%d icomp=%d rc=%d max|diff|=%.3e %s\n",
               pos, due, nr, na, ni, rc, md, okp?"ok":"BAD");
        n_raw=nr; n_acomp=na; n_icomp=ni;
    }
    if (bad) printf("FAIL mid: %d/%d posizioni fuori tolleranza (worst=%.3e)\n", bad, NP, worst);
    else     printf("PASS mid: heads GPU == CPU (worst=%.3e) su %d posizioni\n", worst, NP);
    return bad ? 1 : 0;
}
