/* Plugin CODEC "fp4": E2M1. fmt 0 = I8 (2 nibble/byte, scala e8m0), fmt 1 = MLX mxfp4 (U32, 8/word, e8m0 u8). */
#include "llmcodec.h"
#include "llmcodec_fp8_fp4.h"
#include <stdint.h>
static int f_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){ (void)p0; int g=p1?p1:32;
    if(c%2) return -1; *ob=r*c/2;
    if(fmt==1) *ab=r*((c+g-1)/g); else { int br=p0?p0:1; *ab=((r+br-1)/br)*((c+g-1)/g); } return 0; }
static int f_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)p0;(void)p1; *r=d0; *c=d1*((fmt==1)?8:2); return 0; }
static int f_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)p0; int g=p1?p1:32;
    if(w==0){*d0=r;*d1=(fmt==1)?c/8:c/2;}else{*d0=r;*d1=(c+g-1)/g;} return 0; }
static const char* f_dt(int which,int fmt){ (void)fmt; return which==0?((fmt==1)?"U32":"I8"):"F8_E8M0"; }
static int f_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il; if(aux==NULL) return -1; int g=p1?p1:32;
    if(fmt==1) llmcodec_dsv41_dequant_fp4_block_sz((const uint8_t*)in,(const uint8_t*)aux,(int)rows,(int)cols,p0?p0:1,g,out);
    else       llmcodec_dsv41_dequant_fp4_block_sz((const uint8_t*)in,(const uint8_t*)aux,(int)rows,(int)cols,p0?p0:1,g,out);
    return 0; }
static int f_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "fp4", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, f_dt, f_shape, f_ssh, f_sizes, f_decode, f_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
