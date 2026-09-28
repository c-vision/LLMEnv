/* Plugin CODEC "gptq": GPTQ/HF safetensors. qweight int32 [in/8, out] (nibble lungo IN),
 * scales f16 [in/group, out], qzeros int32 [in/group, out/8], g_idx int32 [in] (opz.).
 * aux = scales ++ qzeros. p0 = group. decode: w[o,i] = (q - z) * s. */
#include "llmcodec.h"
#include <stdint.h>
#include <math.h>
#include <string.h>
static float f16(const uint8_t* p){ uint16_t h=*(const uint16_t*)p; uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int g_sizes(int fmt,int p0,int p1,size_t rows,size_t cols,size_t* ob,size_t* ab){
    (void)fmt;(void)rows;(void)cols; (void)p0;(void)p1; return -1; }   /* GPTQ: main/aux gestiti dal converter (multi-tensore) */
static int g_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){
    (void)fmt;(void)p0;(void)p1; *r=d1; *c=d0*8; return 0; }           /* stored [in/8,out] -> logico [out,in] */
static int g_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)w;(void)fmt;(void)p0;(void)p1;(void)r;(void)c;*d0=0;*d1=0;return -1; }
static const char* g_dt(int which,int fmt){ (void)which;(void)fmt; return "I32"; }
static int g_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)fmt;
    int group=p0?p0:128; int nin=(int)cols, outn=(int)rows, g=nin/group; if(g<1)g=1;
    const uint32_t* qweight=(const uint32_t*)in;
    size_t sc_n=(size_t)g*outn; size_t sc_b=sc_n*2;   /* scales f16 */
    if(aux==NULL || al < sc_b + ((size_t)g*(outn/8))*4) return -1;
    const uint8_t* scales=(const uint8_t*)aux; const uint32_t* qzeros=(const uint32_t*)((const uint8_t*)aux+sc_b);
    const uint32_t* gidx = (const uint32_t*)((const uint8_t*)aux+sc_b+((size_t)g*(outn/8))*4);
    int has_g = (al >= sc_b + ((size_t)g*(outn/8))*4 + (size_t)nin*4);
    for(int o=0;o<outn;o++) for(int i=0;i<nin;i++){
        uint32_t qwi = qweight[(size_t)(i/8)*outn + o];
        int q = (qwi >> (4*(i%8))) & 0xF;
        int gi = has_g ? (int)gidx[i] : (i/group);
        uint32_t zrow = qzeros[(size_t)gi*(outn/8) + (o/8)];
        int z = (zrow >> (4*(o%8))) & 0xF;
        float s = f16(scales + ((size_t)gi*outn + o)*2);
        out[(size_t)o*nin + i] = (float)(q - z) * s;
    }
    return 0; }
static int g_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "gptq", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, g_dt, g_shape, g_ssh, g_sizes, g_decode, g_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
