/* Plugin CODEC "hqq": HQQ / Quanto / torchao weight-only (famiglia affine-con-zero).
 * Pesi: int (packed o byte). aux = scales f16 ++ zeros f16 (per gruppo). p0 = group, p1 = bits.
 * decode: w[i] = (q_i - zero_g) * scale_g. Copre HQQ (scale+zero), Quanto (scale, zero=0),
 * torchao int4/int8 (scale+zero per gruppo). */
#include "llmcodec.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
static float h_f16(const uint8_t* p){ uint16_t h=*(const uint16_t*)p; uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int h_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){ (void)fmt; int bits=p1?p1:4; if(bits!=4 && bits!=8) return -1;
    *ob=(bits==4)?r*c/2:r*c; *ab=2*r*((c+(p0?p0:64)-1)/(p0?p0:64))*2; return 0; }
static int h_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0; int bits=p1?p1:4; *r=d0; *c=(bits==4)?d1*2:d1; return 0; }
static int h_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)fmt; int g=p0?p0:64,bits=p1?p1:4; if(w==0){*d0=r;*d1=(bits==4)?c/2:c;}else{*d0=r;*d1=(c+g-1)/g;} return 0; }
static const char* h_dt(int which,int fmt){ (void)which;(void)fmt; return "F16"; }
static int h_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)fmt; int g=p0?p0:64, bits=p1?p1:4; size_t scn=(cols+g-1)/g, sb=rows*scn*2;
    if(aux==NULL || al<2*sb) return -1;
    const uint8_t* w=(const uint8_t*)in; const uint8_t* scales=(const uint8_t*)aux; const uint8_t* zeros=(const uint8_t*)aux+sb;
    for(size_t r=0;r<rows;r++) for(size_t c=0;c<cols;c++){
        int q;
        if(bits==4){ uint8_t b=w[r*(cols/2)+(c/2)]; q=(c&1)?(b>>4):(b&0xF); }
        else q=w[r*cols+c];
        float s=h_f16(scales+(r*scn+c/g)*2), z=h_f16(zeros+(r*scn+c/g)*2);
        out[r*cols+c]=((float)q - z)*s;
    }
    return 0; }
static int h_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "hqq", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, h_dt, h_shape, h_ssh, h_sizes, h_decode, h_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
