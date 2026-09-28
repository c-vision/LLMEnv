/* Plugin CODEC "plain": tensori non quantizzati F32/F16/BF16. fmt: 0=F32, 1=F16, 2=BF16. */
#include "llmcodec.h"
#include "llmcodec_fp16.h"   /* f32->f16 = atomo del CODEC */
#include <stdint.h>
#include <string.h>
static int esz(int fmt){ return fmt==0?4:2; }
static int p_sizes(int fmt,int p0,int p1,size_t rows,size_t cols,size_t* ob,size_t* ab){ (void)p0;(void)p1; *ob=rows*cols*(size_t)esz(fmt); *ab=0; return 0; }
static int p_shp(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0;(void)p1; *r=d0; *c=d1; return 0; }
static const char* p_dt(int which,int fmt){ if(which!=0) return NULL; return fmt==1?"F16":fmt==2?"BF16":"F32"; }
static float f16_to_f32(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u; if(e==0){ if(m==0)u=s<<31; else{ e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} } else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13); float f; memcpy(&f,&u,4); return f; }
static int p_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){ (void)il;(void)p0;(void)p1;(void)aux;(void)al; size_t n=rows*cols;
    if(fmt==0) memcpy(out,in,n*4);
    else if(fmt==1){ for(size_t i=0;i<n;i++) out[i]=f16_to_f32(((const uint16_t*)in)[i]); }
    else { for(size_t i=0;i<n;i++){ uint32_t u=(uint32_t)((const uint16_t*)in)[i]<<16; float f; memcpy(&f,&u,4); out[i]=f; } }
    return 0; }
static int p_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)p0;(void)p1;(void)aux;(void)ac; size_t n=rows*cols;
    if(fmt==0){ if(oc<n*4) return -1; memcpy(out,in,n*4); *ol=n*4; }
    else if(fmt==1){ if(oc<n*2) return -1; uint16_t* o=(uint16_t*)out; for(size_t i=0;i<n;i++) o[i]=llmcodec_f16_from_f32(in[i]); *ol=n*2; }
    else { if(oc<n*2) return -1; uint16_t* o=(uint16_t*)out; for(size_t i=0;i<n;i++){ uint32_t u; memcpy(&u,&in[i],4); o[i]=(uint16_t)(u>>16); } *ol=n*2; }
    *al=0; return 0; }
static int p_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){(void)w;(void)fmt;(void)p0;(void)p1;*d0=r;*d1=c;return 0;}
static const LlmCodec C = { "plain", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_NONE, p_dt, p_shp, p_ssh, p_sizes, p_decode, p_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
