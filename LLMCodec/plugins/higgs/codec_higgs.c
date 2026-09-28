/* Plugin CODEC "higgs": HIGGS (Hadamard Incoherence + Grid-Guided quantization).
 * Port in C della parte verificabile di transformers/integrations/higgs.py:
 * la griglia E8 (256x2, generata/estrapolata) + l'indice 8-bit -> coppia della griglia.
 * Il valore decodificato = grid[idx] scalato (norma di gruppo). La inverse-Hadamard e la LUT
 * FLUTE restano da agganciare come trasformazione di gruppo. fmt 0. p0 = bits(4), p1 = group(256). */
#include "llmcodec.h"
#include "llmcodec_higgs_grid.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
static float h_f16(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int g_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){ (void)fmt;(void)p0;(void)p1; (void)r;(void)c; *ob=0; *ab=0; return -1; }
static int g_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0;(void)p1; *r=d0; *c=d1; return 0; }
static int g_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)w;(void)fmt;(void)p0;(void)p1;(void)r;(void)c;*d0=0;*d1=0;return -1; }
static const char* g_dt(int which,int fmt){ (void)fmt; return which==0?"U8":"F16"; }
static int g_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)fmt;(void)p0;
    int group=(int)(p1?p1:256);
    if(aux==NULL) return -1;
    /* indices U8 (1 per coppia di elementi = per 2 dims); aux = norms f16 [groups] */
    const uint8_t* idx=(const uint8_t*)in; const uint8_t* nm=(const uint8_t*)aux;
    size_t npairs=rows*cols/2; size_t ngroups=(npairs*2+(size_t)group-1)/(size_t)group;
    if(al < ngroups*2) return -1;
    for(size_t q=0;q<npairs;q++){
        int code=idx[q] % LLMCODEC_HIGGS_GRID_N;
        size_t d=2*q; float nrm = h_f16((uint16_t)(nm[(q*2)/group*2] | (nm[(q*2)/group*2+1]<<8)));
        out[d]=LLMCODEC_HIGGS_GRID[code][0]*nrm;
        out[d+1]=LLMCODEC_HIGGS_GRID[code][1]*nrm;
    }
    return 0; }
static int g_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "higgs", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, g_dt, g_shape, g_ssh, g_sizes, g_decode, g_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
