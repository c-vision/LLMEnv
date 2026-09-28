/* Plugin CODEC "bitnet": BitNet b1.58 (ternario {-1,0,+1}). Packing 2-bit (4 per byte),
 * valori: 00=0, 01=+1, 10=-1 (o 11=-1). Scala f16 per riga (aux). p0 = group(riga=cols). */
#include "llmcodec.h"
#include "llmcodec_fp16.h"
#include <stdint.h>
#include <string.h>
static float b_f16(const uint8_t* p){ uint16_t h=*(const uint16_t*)p; uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int b_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){ (void)fmt;(void)p0;(void)p1; if(c%4) return -1; *ob=r*c/4; *ab=r*2; return 0; }
static int b_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0;(void)p1; *r=d0; *c=d1*4; return 0; }
static int b_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)fmt;(void)p0;(void)p1; if(w==0){*d0=r;*d1=c/4;}else{*d0=r;*d1=1;} return 0; }
static const char* b_dt(int which,int fmt){ (void)fmt; return which==0?"U8":"F16"; }
static int b_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)fmt;(void)p0;(void)p1;
    if(aux==NULL || al < rows*2) return -1;
    const uint8_t* w=(const uint8_t*)in; const uint8_t* sc=(const uint8_t*)aux;
    static const float t[4]={0.0f,1.0f,-1.0f,-1.0f};
    for(size_t r=0;r<rows;r++){ float s=b_f16(sc+r*2);
        for(size_t c=0;c<cols;c++){ uint8_t byte=w[r*(cols/4)+(c/4)]; int code=(byte>>(2*(c%4)))&0x3;
            out[r*cols+c]=t[code]*s; } }
    return 0; }
static int b_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){
    (void)fmt;(void)p0;(void)p1; if(oc<rows*cols/4 || ac<rows*2) return -1;
    uint8_t* w=(uint8_t*)out; uint8_t* sc=(uint8_t*)aux;
    for(size_t r=0;r<rows;r++){ float amax=0; for(size_t c=0;c<cols;c++){ float a=in[r*cols+c]; if(a<0)a=-a; if(a>amax)amax=a; }
        float s=amax>0?amax:1.0f; uint16_t h=llmcodec_f16_from_f32(s);   /* atomo CODEC */
        memcpy(sc+r*2,&h,2);
        for(size_t c=0;c<cols;c++){ float v=in[r*cols+c]; int code=(v>s*0.5f)?1:((v<-s*0.5f)?2:0);
            uint8_t* bp=&w[r*(cols/4)+(c/4)]; if((c%4)==0)*bp=0; *bp=(uint8_t)(*bp | (code<<(2*(c%4)))); }
    }
    *ol=rows*cols/4; *al=rows*2; return 0; }
static const LlmCodec C = { "bitnet", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, b_dt, b_shape, b_ssh, b_sizes, b_decode, b_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
