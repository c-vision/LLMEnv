/* Plugin CODEC "turboquant": VQ a codebook. weight U32 (indici 4-bit, 8/word), codebook F16[16],
 * scales F16 [rows, cols/group]. aux = codebook(32B) ++ scales. p1 = group.
 * ATTENZIONE: il valore decodificato e' PRE-rotazione (la rotazione Hadamard del checkpoint e'
 * una PRIMITIVA separata — non ancora agganciata). Decoder fornito come ADAPTER (canonico). */
#include "llmcodec.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
static float t_f16(const uint8_t* p){ uint16_t h=*(const uint16_t*)p; uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int t_cbn(int fmt){ return (fmt==1)?4:((fmt==2)?256:16); }   /* codebook entries */
static int t_perw(int fmt){ return (fmt==1)?16:((fmt==2)?4:8); }    /* indici per parola U32 */
static int t_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){ (void)p0; int g=p1?p1:64; *ob=r*c/2; *ab=t_cbn(fmt)*2 + r*((c+g-1)/g)*2 + c*2; return 0; }  /* + signs [cols] f16 */
static int t_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)p0;(void)p1; *r=d0; *c=d1*(size_t)t_perw(fmt); return 0; }
static int t_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)p0; int g=p1?p1:64; if(w==0){*d0=r;*d1=c/(size_t)t_perw(fmt);}else{*d0=r;*d1=(c+g-1)/g;} return 0; }
static const char* t_dt(int which,int fmt){ (void)fmt; return which==0?"U32":"F16"; }
/* Inverse randomized-Hadamard (WHT ortonormale * signs) lungo l'ultima dim (potenza di 2). */
static void t_wht(float* x, int n){ int h=1; while(h<n){ for(int i=0;i<n;i+=(h<<1)) for(int j=0;j<h;j++){
    float a=x[i+j], b=x[i+j+h]; x[i+j]=a+b; x[i+j+h]=a-b; } h<<=1; }
    float inv=1.0f/sqrtf((float)n); for(int i=0;i<n;i++) x[i]*=inv; }
static uint32_t t_rs(uint32_t* st){ uint32_t x=*st; x^=x<<13; x^=x>>17; x^=x<<5; *st=x; return x; }
static void t_unrotate(float* row, int cols, int seed, float* tmp){
    if(seed==0 || cols<2) return;
    if((cols & (cols-1))!=0) return;   /* non potenza di 2: nessuna rotazione (documentato) */
    uint32_t st=(uint32_t)seed;
    for(int c=0;c<cols;c++){ uint32_t r=t_rs(&st); float sgn=(r&1)?-1.0f:1.0f; tmp[c]=row[c]*sgn; }
    t_wht(tmp, cols);
    for(int c=0;c<cols;c++){ uint32_t r=t_rs(&st); float sgn=(r&1)?-1.0f:1.0f; row[c]=tmp[c]*sgn; }
}

static int t_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)fmt;(void)p0;
    int g=p1?p1:64; size_t scn=(cols+g-1)/g; int cbn=t_cbn(fmt), perw=t_perw(fmt); int ibit=(perw==8)?4:((perw==4)?8:2);
    if(aux==NULL || al < (size_t)cbn*2 + rows*scn*2) return -1;
    const uint32_t* w=(const uint32_t*)in; const uint8_t* cb=(const uint8_t*)aux; const uint8_t* sc=(const uint8_t*)((const uint8_t*)aux + cbn*2);
    float codebook[256]; for(int k=0;k<cbn;k++) codebook[k]=t_f16(cb+k*2);
    for(size_t r=0;r<rows;r++) for(size_t c=0;c<cols;c++){
        uint32_t word=w[r*(cols/(size_t)perw)+(c/(size_t)perw)]; int idx=(word>>(ibit*(c%(size_t)perw)))&(cbn-1);
        float s=t_f16(sc+(r*scn + c/g)*2);
        out[r*cols+c]=codebook[idx]*s;
    }
    /* signs REALI in coda all'aux (modello tq4a): out[o][c] *= signs[c] */
    { size_t scbase=(size_t)cbn*2 + rows*scn*2;
      if(al >= scbase + cols*2){ const uint8_t* sg=(const uint8_t*)aux + scbase;
          for(size_t r=0;r<rows;r++) for(size_t c=0;c<cols;c++){
              uint16_t hh; memcpy(&hh,sg+c*2,2); float sgn=t_f16((const uint8_t*)&hh);
              if(sgn<0) out[r*cols+c]=-out[r*cols+c]; } }
      else { float* tmp=(float*)malloc(cols*sizeof(float)); if(tmp){ for(size_t r=0;r<rows;r++) t_unrotate(out+r*cols,(int)cols,p0,tmp); free(tmp); } } }
    return 0; }
static int t_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "turboquant", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, t_dt, t_shape, t_ssh, t_sizes, t_decode, t_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
