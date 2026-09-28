/* Plugin CODEC "bnb": bitsandbytes. fmt 0 = NF4 (codebook 16 normale-float), fmt 1 = FP4 (e2m1),
 * fmt 2 = INT8 (block absmax, outlier opz.). Pesi: packed U8 (2 nibble/byte per NF4/FP4),
 * scala = absmax f32 per blocco (block_size, default 64). aux = absmax f32 [rows, cols/block]. */
#include "llmcodec.h"
#include "llmcodec_fp8_fp4.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
static const float NF4[16] = {
 -1.0f,-0.6961928009986877f,-0.5250730514526367f,-0.39491748809814453f,-0.28444138169288635f,
 -0.18477343022823334f,-0.09105003625154495f,0.0f,0.07958029955625534f,0.16093020141124725f,
 0.24611230194568634f,0.33791524171829224f,0.44070982933044434f,0.5626170039176941f,
 0.7229568362236023f,1.0f };
static int b_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){ (void)p0;(void)p1;
    if(fmt==2){ *ob=r*c; *ab=r*((c+63)/64)*4; return 0; }
    if(fmt==3){ if(c%2) return -1; *ob=r*c/2; *ab=16*4 + r*((c+63)/64) + 266*4 + 256*4; return 0; }
    if(c%2) return -1; *ob=r*c/2; *ab=r*((c+63)/64)*4; return 0; }
static int b_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0;(void)p1; *r=d0; *c=d1*2; return 0; }
static int b_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)fmt;(void)p0;(void)p1; if(w==0){*d0=r;*d1=c/2;}else{*d0=r;*d1=(c+63)/64;} return 0; }
static const char* b_dt(int which,int fmt){ (void)fmt; return which==0?"U8":"F32"; }
static int b_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)p0;(void)p1;
    if(aux==NULL) return -1; size_t blocks=(cols+63)/64;
    const uint8_t* w=(const uint8_t*)in;
    if(fmt==3){
        const float* cmap=(const float*)aux; const uint8_t* aq=((const uint8_t*)aux)+64;
        size_t abl=rows*blocks; const float* nabs=(const float*)(aq+abl); const float* nmap=(const float*)(aq+abl+266*4);
        for(size_t r=0;r<rows;r++) for(size_t c=0;c<cols;c++){
            int idx=(c&1)?(w[r*(cols/2)+(c/2)]>>4):(w[r*(cols/2)+(c/2)]&0xF);
            float ar=nabs[0]*nmap[aq[r*blocks + c/64]];
            out[r*cols+c]=cmap[idx]*ar;
        }
        return 0; }
    const float* ab=(const float*)aux;
    for(size_t r=0;r<rows;r++) for(size_t c=0;c<cols;c++){
        float s=ab[r*blocks + c/64];
        float v;
        if(fmt==2){ v=(float)(int8_t)w[r*cols+c]; }
        else { uint8_t byte=w[r*(cols/2)+(c/2)]; int idx=(c&1)?(byte>>4):(byte&0xF); v=(fmt==0)?NF4[idx]:(llmcodec_fp4_e2m1_level(idx&7)*((idx&8)?-1.0f:1.0f)); }
        out[r*cols+c]=v*s;
    }
    return 0; }
static int b_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "bnb", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, b_dt, b_shape, b_ssh, b_sizes, b_decode, b_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
