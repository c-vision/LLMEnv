/* llm_dsv4_decode_fn_bench.c — ORACOLO MODEL-FREE: COSTO per-funzione del DECODE DS4.
 *
 * Individua DOVE va il tempo del decode (gap ~20% vs vendored) **senza modello**: crea il backend
 * Metal DS4 e chiama le funzioni per-token del decode con **pesi/attivazioni SINTETICI** (random nel
 * formato giusto; dimensioni da `llm_dsv4_config_flash()`). Misura `us/call` (CLOCK_MONOTONIC) e
 * `disp/call` (contatore dispatch Metal, **load-free**). Stampa anche `rc` della prima call (se !=0
 * il path e' short-circuitato e il tempo non e' significativo).
 *
 * Uso:  llm_dsv4_decode_fn_bench <dylib> [iters]      (default iters=200)
 * NON carica il modello. Aggiungere qui nuove funzioni "una alla volta".
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include "llm_ggml.h"   /* llm_ggml_nbytes (static inline) */

typedef struct LlmDsv4Metal LlmDsv4Metal;
typedef LlmDsv4Metal* (*create_fn)(void);
typedef long (*disp_fn)(void);
typedef int (*route_fn)(LlmDsv4Metal*, int nt, int n_expert, int embd, int used, const float* gate_inp, const float* bias,
                        const int* hsel, float scale, const float* x, int* sel, float* wt);
typedef int (*moe_fn)(LlmDsv4Metal*, int nt, const void* gate_w, const void* up_w, const void* down_w,
                      int gtype, int dtype, unsigned n_expert, unsigned embd, unsigned ff, unsigned gu_rb, unsigned down_rb,
                      const int* ids, const float* rws, const float* x, float clamp,
                      const void* sg, const void* su, const void* sd, unsigned s_nb_g, unsigned s_nb_d, float* out);
typedef int (*hcp_fn)(LlmDsv4Metal*, int nt, int HC, int D, int iters, float hc_eps, float rms_eps, const float* res,
                      const float* fn, const float* scale, const float* base, const float* norm_w, float* normed, float* sp);
typedef int (*hco_fn)(LlmDsv4Metal*, int nt, int HC, int D, const float* bo, const float* res, const float* sp, float* out);
typedef int (*proj_fn)(LlmDsv4Metal*, const void* qa, size_t qa_rb, const void* kv, size_t kv_rb, const void* qb, size_t qb_rb,
                       const float* xn, unsigned embd, const float* qan, const float* kvan, unsigned lq, unsigned hd, unsigned qd,
                       int n_head, int n_rot, const float* rope_cfg, int pos, float eps, float* q_out, float* kv_out, float* qrn_out);
typedef int (*outf_fn)(LlmDsv4Metal*, const void* outa, size_t outa_rb, size_t outa_lead0, int ng, unsigned gd, unsigned rank,
                       const void* outb, size_t outb_rb, unsigned ol, unsigned embd, unsigned heads_total, const float* heads, float* out);
typedef int (*comppool_fn)(LlmDsv4Metal*, const float* state_kv, const float* state_score, float* pooled, int hd, int ratio, int co);
typedef int (*compfp8_fn)(LlmDsv4Metal*, float* x, int hd, int n_rot);
typedef int (*topk_fn)(LlmDsv4Metal*, const float* scores, int n_comp, int top_k, int* allowed);
typedef int (*idxsc_fn)(LlmDsv4Metal*, const float* q, int nh, int hd, const float* kvc, int n_comp, const float* w, float* scores);

/* DS4 Flash (llm_dsv4_config_flash). */
#define EMB 4096
#define FF  2048
#define NH  64
#define HD  512
#define NLQ 1024
#define QD  (NH*HD)
#define NEXP_SYNTH 16
#define GUP_T 16   /* IQ2_XXS */
#define DWN_T 10   /* Q2_K */
#define Q8_T  8    /* Q8_0 */

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }
static inline size_t nby(int type, int n) { return (size_t)llm_ggml_nbytes(type, n); }
/* Pesi sintetici PAGE-ALIGNED: il motore li mappa con view no-copy (newBufferWithBytesNoCopy) che
 * esigono pagine intere e memoria page-backed -> `malloc` non basta. */
static void* pav(size_t bytes) {
    size_t n = (bytes + 16383u) & ~(size_t)16383u;
    void* p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    memset(p, 0, n); return p;
}

#define ROW(label, stmt) do { \
    int rc0 = 0; long d0 = mdisp(); double best = 1e18; \
    if (shared_begin) shared_begin(b); \
    double t0 = now_ms(); \
    for (int i = 0; i < iters; i++) { double a = now_ms(); int rc = (stmt); double dt = now_ms()-a; \
        if (i == 0) rc0 = rc; if (dt < best) best = dt; } \
    double t1 = now_ms(); \
    if (shared_end) shared_end(b); \
    printf("%-20s %10.2f %10.2f %10.2f   rc0=%d\n", label, (t1-t0)/iters*1000.0, best*1000.0, \
        (double)(mdisp()-d0)/iters, rc0); \
} while (0)

int main(int argc, char** argv) {
    const char* lib = argc > 1 ? argv[1] : "build/LLMEngine/mlx/libllm_ai_mlx.dylib";
    int iters = argc > 2 ? atoi(argv[2]) : 200;

    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen(%s): %s\n", lib, dlerror()); return 1; }
    create_fn mcreate = (create_fn)dlsym(h, "llm_dsv4_metal_create");
    disp_fn   mdisp   = (disp_fn)dlsym(h, "llm_dsv4_metal_dispatches");
    route_fn  route   = (route_fn)dlsym(h, "llm_dsv4_metal_route_batch");
    moe_fn    moe     = (moe_fn)dlsym(h, "llm_dsv4_metal_moe_mm_batch");
    hcp_fn    hcp     = (hcp_fn)dlsym(h, "llm_dsv4_metal_hc_pre_batch");
    hco_fn    hco     = (hco_fn)dlsym(h, "llm_dsv4_metal_hc_post_batch");
    proj_fn   proj    = (proj_fn)dlsym(h, "llm_dsv4_metal_attn_proj_fused");
    outf_fn   outf    = (outf_fn)dlsym(h, "llm_dsv4_metal_attn_out_fused");
    comppool_fn cpool = (comppool_fn)dlsym(h, "llm_dsv4_metal_comp_pool");
    compfp8_fn  cfp8  = (compfp8_fn)dlsym(h, "llm_dsv4_metal_comp_fp8");
    topk_fn     topk  = (topk_fn)dlsym(h, "llm_dsv4_metal_indexer_topk");
    idxsc_fn    iscor = (idxsc_fn)dlsym(h, "llm_dsv4_metal_indexer_scores");
    if (!mcreate || !mdisp || !route || !moe || !hcp || !hco || !proj || !outf || !cpool || !cfp8 || !topk || !iscor) { fprintf(stderr, "dlsym incompleto\n"); return 1; }
    LlmDsv4Metal* b = mcreate();
    if (!b) { fprintf(stderr, "llm_dsv4_metal_create fallita\n"); return 1; }
    /* modalita' shared (una command buffer, un solo sync): evita il commit/wait per-call */
    void (*shared_begin)(LlmDsv4Metal*) = (void(*)(LlmDsv4Metal*))dlsym(h, "llm_dsv4_metal_shared_begin");
    void (*shared_end)(LlmDsv4Metal*)   = (void(*)(LlmDsv4Metal*))dlsym(h, "llm_dsv4_metal_shared_end");
    int shared_mode = (argc > 3 && argv[3][0] == 's');
    if (!shared_mode) { shared_begin = NULL; shared_end = NULL; }

    setvbuf(stdout, NULL, _IONBF, 0);   /* non bufferizzato: se un kernel crasha, vediamo dove */
    printf("[dsv4-decode-fn-bench] iters=%d %s(MODEL-FREE: pesi sintetici, no modello)\n", iters,
           shared_mode ? "SHARED-CB " : "");
    printf("%-20s %10s %10s %10s\n", "funzione", "us/avg", "us/min", "disp/call");

    /* router */
    {
        float* gi = malloc((size_t)256*EMB*4), *bias = malloc(256*4), *x = malloc(EMB*4);
        int sel[8]; float wt[8];
        for (size_t i=0;i<(size_t)256*EMB;i++) gi[i]=0.01f; for(int i=0;i<256;i++) bias[i]=0.0f; for(int i=0;i<EMB;i++) x[i]=0.1f;
        if (gi&&bias&&x) ROW("route_batch", route(b,1,256,EMB,6,gi,bias,NULL,1.0f,x,sel,wt));
        free(gi);free(bias);free(x);
    }
    /* mHC */
    {
        int HC=4, SZ=(2+HC)*HC;
        float* res=malloc((size_t)HC*EMB*4),*fn=malloc((size_t)HC*HC*4),*sc=malloc(HC*4),*ba=malloc(HC*4);
        float* nw=malloc(EMB*4),*no=malloc(EMB*4),*sp=malloc((size_t)SZ*4),*out=malloc((size_t)HC*EMB*4);
        for(size_t i=0;i<(size_t)HC*EMB;i++) res[i]=0.05f; for(int i=0;i<HC*HC;i++) fn[i]=0.1f;
        for(int i=0;i<HC;i++){sc[i]=1.0f;ba[i]=0.0f;} for(int i=0;i<EMB;i++) nw[i]=1.0f;
        if(res&&fn&&sc&&ba&&nw&&no&&sp&&out){ ROW("hc_pre_batch", hcp(b,1,HC,EMB,20,1e-6f,1e-6f,res,fn,sc,ba,nw,no,sp)); ROW("hc_post_batch", hco(b,1,HC,EMB,ba,res,sp,out)); }
        free(res);free(fn);free(sc);free(ba);free(nw);free(no);free(sp);free(out);
    }
    /* MoE */
    {
        unsigned gu=nby(GUP_T,EMB), dn=nby(DWN_T,FF);
        void* gw=malloc((size_t)NEXP_SYNTH*FF*gu); void* uw=malloc((size_t)NEXP_SYNTH*FF*gu); void* dw=malloc((size_t)NEXP_SYNTH*EMB*dn);
        void* sg=malloc(nby(GUP_T,EMB)); void* su=malloc(nby(GUP_T,EMB)); void* sd=malloc(nby(DWN_T,FF));
        float* x=malloc(EMB*4); int ids[8]; float rws[8], out[EMB];
        for(int i=0;i<8;i++){ids[i]=i%NEXP_SYNTH;rws[i]=0.1f;} for(int i=0;i<EMB;i++) x[i]=0.1f;
        if(gw&&uw&&dw&&sg&&su&&sd&&x) ROW("moe_mm_batch(1tok)", moe(b,1,gw,uw,dw,GUP_T,DWN_T,NEXP_SYNTH,EMB,FF,gu,dn,ids,rws,x,10.0f,sg,su,sd,(unsigned)nby(GUP_T,EMB),(unsigned)nby(DWN_T,FF),out));
        free(gw);free(uw);free(dw);free(sg);free(su);free(sd);free(x);
    }
    /* attn proj: q_a[lq x embd] + kv[hd x embd] + q_b[qd x lq] (Q8_0) + rms + rope + head_rms, fuso */
    {
        unsigned embd=EMB, lq=NLQ, hd=HD, qd=QD;
        size_t qa_rb=nby(Q8_T,embd), kv_rb=nby(Q8_T,embd), qb_rb=nby(Q8_T,lq);
        void* qa=pav((size_t)lq*qa_rb); void* kvw=pav((size_t)hd*kv_rb); void* qb=pav((size_t)qd*qb_rb);
        float* xn=malloc(embd*4); float* qan=malloc(lq*4); float* kvan=malloc(hd*4);
        float* qo=malloc((size_t)qd*4); float* kvo=malloc(hd*4); float* qrn=malloc(lq*4);
        float rope_cfg[7]={10000.0f,1.0f,0.0f,1.0f,32.0f,1.0f,65536.0f};
        for(int i=0;i<EMB;i++) xn[i]=0.1f; for(int i=0;i<NLQ;i++) qan[i]=1.0f; for(int i=0;i<HD;i++) kvan[i]=1.0f;
        if(qa&&kvw&&qb&&xn&&qan&&kvan&&qo&&kvo&&qrn)
            ROW("attn_proj_fused", proj(b,qa,qa_rb,kvw,kv_rb,qb,qb_rb,xn,embd,qan,kvan,lq,hd,qd,NH,64,rope_cfg,0,1e-6f,qo,kvo,qrn));
        /* qa/kvw/qb sono mmap (pav): NON free(); lasciati al termine del processo */
        free(xn);free(qan);free(kvan);free(qo);free(kvo);free(qrn);
    }
    /* attn out: out_a[gd x ol] + out_b[ol x embd] (Q8_0), grouped LoRA */
    {
        int ng=8; unsigned gd=(unsigned)(HD*(NH/ng)), rank=NLQ, ol=(unsigned)(ng*NLQ), embd=EMB, ht=(unsigned)QD;
        size_t oa_rb=nby(Q8_T,ol), ob_rb=nby(Q8_T,embd);
        void* oa=pav((size_t)gd*oa_rb); void* ob=pav((size_t)ol*ob_rb);
        float* heads=malloc((size_t)ht*4); float* out=malloc(embd*4);
        for(size_t i=0;i<(size_t)ht;i++) heads[i]=0.05f;
        /* attn_out: le view no-copy + kernel esigono il LAYOUT dei pesi del modello (mmap reale);
         * con pesi sintetici il kernel va in OOB/crash -> non benchanzabile model-free in isolamento. */
        (void)outf; (void)oa; (void)ob;
        printf("%-20s %10s %10s   (richiede layout pesi del modello)\n", "attn_out_fused", "-", "-");
        /* oa/ob sono mmap (pav): NON free() */
        free(heads);free(out);
    }

    /* compressor: pool (state -> pooled) + fp8 in-place */
    {
        int hd=HD, ratio=4, co=1; size_t rows=(size_t)co*ratio*co*hd;
        float* skv=malloc(rows*4), *ssc=malloc(rows*4), *pool=malloc((size_t)hd*4), *xfp=malloc((size_t)hd*4);
        for(size_t i=0;i<rows;i++){skv[i]=0.01f;ssc[i]=0.02f;} for(int i=0;i<hd;i++) xfp[i]=0.1f;
        if(skv&&ssc&&pool) ROW("comp_pool", cpool(b,skv,ssc,pool,hd,ratio,co));
        if(xfp) ROW("comp_fp8", cfp8(b,xfp,hd,64));
        free(skv);free(ssc);free(pool);free(xfp);
    }
    /* indexer: scores (q x compressed kv) + top-k */
    {
        int nh=64, hd=128, n_comp=512, top_k=512;
        float* q=malloc((size_t)nh*hd*4), *kvc=malloc((size_t)n_comp*hd*4), *w=malloc((size_t)nh*4), *sc=malloc((size_t)n_comp*4);
        int* al=malloc((size_t)n_comp*4);
        for(int i=0;i<nh*hd;i++) q[i]=0.01f; for(int i=0;i<n_comp*hd;i++) kvc[i]=0.02f; for(int i=0;i<nh;i++) w[i]=1.0f;
        for(int i=0;i<n_comp;i++) sc[i]=0.01f*(float)(i%7);
        if(q&&kvc&&w&&sc) ROW("indexer_scores", iscor(b,q,nh,hd,kvc,n_comp,w,sc));
        if(sc&&al) ROW("indexer_topk", topk(b,sc,n_comp,top_k,al));
        free(q);free(kvc);free(w);free(sc);free(al);
    }

    printf("\nNota: us/call indicativo sotto contesa; **disp/call** load-free. rc0!=0 = path saltato (tempo non valido).\n");
    return 0;
}
