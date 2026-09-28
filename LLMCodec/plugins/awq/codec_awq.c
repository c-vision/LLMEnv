/* Plugin CODEC "awq": AWQ/HF safetensors. qweight int32 [in, out/8] (nibble lungo OUT),
 * scales f16 [in/group, out], qzeros int32 [in/group, out/8]. aux = scales ++ qzeros.
 * p0 = group. decode: w[o,i] = (q - z) * s. */
#include "llmcodec.h"
#include <stdint.h>
#include <math.h>
#include <string.h>
static float a_f16(const uint8_t* p){ uint16_t h=*(const uint16_t*)p; uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int a_sizes(int fmt,int p0,int p1,size_t rows,size_t cols,size_t* ob,size_t* ab){ (void)fmt;(void)rows;(void)cols;(void)p0;(void)p1; return -1; }
static int a_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0;(void)p1; *r=d1*8; *c=d0; return 0; } /* stored [in,out/8] -> [out,in] */
static int a_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)w;(void)fmt;(void)p0;(void)p1;(void)r;(void)c;*d0=0;*d1=0;return -1; }
static const char* a_dt(int which,int fmt){ (void)which;(void)fmt; return "I32"; }
static int a_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)fmt;
    int group=p0?p0:128; int nin=(int)cols, outn=(int)rows, g=nin/group; if(g<1)g=1;
    const uint32_t* qweight=(const uint32_t*)in;
    size_t sc_n=(size_t)g*outn; size_t sc_b=sc_n*2;
    if(aux==NULL || al < sc_b + ((size_t)g*(outn/8))*4) return -1;
    const uint8_t* scales=(const uint8_t*)aux; const uint32_t* qzeros=(const uint32_t*)((const uint8_t*)aux+sc_b);
    for(int o=0;o<outn;o++) for(int i=0;i<nin;i++){
        uint32_t qwi = qweight[(size_t)i*(outn/8) + (o/8)];
        int q = (qwi >> (4*(o%8))) & 0xF;
        uint32_t zrow = qzeros[(size_t)(i/group)*(outn/8) + (o/8)];
        int z = (zrow >> (4*(o%8))) & 0xF;
        float s = a_f16(scales + ((size_t)(i/group)*outn + o)*2);
        out[(size_t)o*nin + i] = (float)(q - z) * s;
    }
    return 0; }
static int a_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "awq", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, a_dt, a_shape, a_ssh, a_sizes, a_decode, a_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
