/* Plugin CODEC "nvfp": NVFP4 (E2M1 + scala E4M3 u8, group 16, no bias) + global f32 opzionale (OCP).
 * fmt 0 = MLX storage (U32 8 nibble/word). aux = scala u8 [rows,cols/g] (+ global f32 in coda). */
#include "llmcodec.h"
#include "llmcodec_fp8_fp4.h"
#include <stdint.h>
#include <string.h>
static int n_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){ (void)fmt;(void)p0; int g=p1?p1:16; *ob=r*c/2; *ab=r*((c+g-1)/g); return 0; }
static int n_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0;(void)p1; *r=d0; *c=d1*8; return 0; }
static int n_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)fmt;(void)p0; int g=p1?p1:16; if(w==0){*d0=r;*d1=c/8;}else{*d0=r;*d1=(c+g-1)/g;} return 0; }
static const char* n_dt(int which,int fmt){ (void)fmt; return which==0?"U32":"U8"; }
static int n_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)fmt;(void)p0;
    if(aux==NULL) return -1; int g=p1?p1:16; size_t base=rows*((cols+g-1)/g);
    float glob=1.0f; if(al>=base+4) memcpy(&glob,(const uint8_t*)aux+base,4);
    llmcodec_dsv41_dequant_nvfp4_g((const uint8_t*)in,(const uint8_t*)aux,(int)rows,(int)cols,g,glob,out);
    return 0; }
static int n_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "nvfp", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, n_dt, n_shape, n_ssh, n_sizes, n_decode, n_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
