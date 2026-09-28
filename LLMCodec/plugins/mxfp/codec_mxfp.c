/* Plugin CODEC "mxfp": OCP MX FP4/FP8. Scala E8M0 (u8) per gruppo.
 * fmt 0 = MXFP8 (E4M3 8-bit, packed 4/u32), fmt 1 = MXFP4 (E2M1 4-bit, 8/u32).
 * (Lo storage MLX separa la scala dal peso; qui il peso = U32 + aux = scala u8.) */
#include "llmcodec.h"
#include "llmcodec_fp8_fp4.h"
#include <stdint.h>
#include <string.h>
static int m_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){
    (void)p0; int g=p1?p1:32;
    if(fmt==1){ if(c%2) return -1; *ob=r*c/2; } else *ob=r*c;
    *ab=r*((c+g-1)/g); return 0; }
static int m_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){
    (void)p0;(void)p1; *r=d0; *c=(fmt==1)?d1*8:d1*4; return 0; }   /* U32 packed */
static int m_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){
    (void)p0; int g=p1?p1:32;
    if(w==0){ *d0=r; *d1=(fmt==1)?c/8:c/4; } else { *d0=r; *d1=(c+g-1)/g; }
    return 0; }
static const char* m_dt(int which,int fmt){ (void)fmt; return which==0?"U32":"U8"; }
static int m_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)p0;(void)al;
    if(aux==NULL) return -1;
    int g=p1?p1:32; const uint8_t* w=(const uint8_t*)in; const uint8_t* sc=(const uint8_t*)aux;
    int scn=(int)((cols+g-1)/g);
    for(size_t r=0;r<rows;r++) for(size_t c=0;c<cols;c++){
        float s=llmcodec_e8m0_to_f32(sc[r*scn + (int)(c/g)]);
        float v;
        if(fmt==1){ uint8_t b=w[r*(cols/2)+(c/2)]; int code=(c&1)?(b>>4):(b&0xF);
                    v=llmcodec_fp4_e2m1_level(code&7); if(code&8) v=-v; }
        else v=llmcodec_fp8_e4m3_to_f32(w[r*cols+c]);
        out[r*cols+c]=v*s;
    }
    return 0; }
static int m_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){
    (void)p0; int g=p1?p1:32;
    if(aux==NULL) return -1;
    uint8_t* w=(uint8_t*)out; uint8_t* sc=(uint8_t*)aux;
    size_t ob=(fmt==1)?rows*cols/2:rows*cols; if(oc<ob || ac<rows*((cols+g-1)/g)) return -1;
    int scn=(int)((cols+g-1)/g);
    for(size_t r=0;r<rows;r++) for(size_t gi=0;gi<(size_t)scn;gi++){
        float amax=0.0f; size_t c0=(size_t)gi*g, c1=c0+g; if(c1>cols)c1=cols;
        for(size_t c=c0;c<c1;c++){ float a=in[r*cols+c]; if(a<0)a=-a; if(a>amax)amax=a; }
        float limit=(fmt==1)?6.0f:448.0f;
        float ex=0.0f; if(amax>0) ex=__builtin_ceilf(__builtin_log2f(amax/limit)); if(ex<-126.0f)ex=-126.0f;
        sc[r*scn+gi]=(uint8_t)((int)ex+127); float s=llmcodec_e8m0_to_f32(sc[r*scn+gi]);
        for(size_t c=c0;c<c1;c++){ float v=in[r*cols+c]/s;
            if(fmt==1){ int code=llmcodec_fp4_e2m1_level(0); (void)code;
                        float a=v<0?-v:v; int ci=0; { float t[8]={0,.5f,1,1.5f,2,3,4,6}; for(int k=7;k>=0;k--) if(a>=t[k]){ci=k;break;} }
                        if(v<0) ci|=8; uint8_t* bp=&w[r*(cols/2)+(c/2)]; if((c&1)==0)*bp=(uint8_t)((*bp&0xF0)|(ci&0xF)); else *bp=(uint8_t)((*bp&0x0F)|((ci&0xF)<<4)); }
            else { float a=v<0?-v:v; if(a>448)a=448; w[r*cols+c]=llmcodec_fp8_to_e4m3_byte(v); }
        }
    }
    *ol=ob; *al=rows*((cols+g-1)/g); return 0; }
static const LlmCodec C = { "mxfp", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, m_dt, m_shape, m_ssh, m_sizes, m_decode, m_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
