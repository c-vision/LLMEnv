/* Plugin CODEC "spqr": SpQR (Sparse-Quantized Representation). Port del layout del kernel
 * di riferimento (Vahe1994/SpQR, inference_kernels/common.cuh, Apache-2.0) in C portabile.
 *   dense_weights int64: tile = s(3) | z(3) | q_0(3) | q_1(3) | ... (3-bit, LSB->MSB)
 *   col_vals int32 = { short c (colonna) ; half v (valore outlier) }   (sparse, CSR)
 *   row_offsets int32                                                  (CSR)
 *   2o ordine half2 ss{s_scale,s_zero}, zz{z_scale,z_zero} per tile
 * w = (q - z_val) * s_val  con  s_val = ss.x*s + ss.y,  z_val = zz.x*z + zz.y.
 * fmt 0 = dense-only.  p0 = beta1 (valori/tile), p1 = beta2 (righe/tile). */
#include "llmcodec.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
static float s_f16(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int q_sizes(int fmt,int p0,int p1,size_t r,size_t c,size_t* ob,size_t* ab){ (void)fmt;(void)r;(void)c;(void)p0;
    int b2=p1?p1:16; (void)b2; *ob=0; *ab=0; return -1; }   /* layout multi-tensore: dal converter */
static int q_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0;(void)p1; *r=d0; *c=d1; return 0; }
static int q_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)w;(void)fmt;(void)p0;(void)p1;(void)r;(void)c;*d0=0;*d1=0;return -1; }
static const char* q_dt(int which,int fmt){ (void)fmt; return which==0?"I64":"F16"; }
static int q_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)fmt;(void)il;
    int beta1=p0?p0:16, beta2=p1?p1:16;
    /* aux = 2o ordine (half2 ss,zz per tile) ++ sparse (col_vals int32) ; vedi converter */
    if(aux==NULL || al<8) return -1;
    const uint64_t* dense=(const uint64_t*)in;
    const uint8_t* so=(const uint8_t*)aux;           /* per-tile: ss(2 half), zz(2 half) = 8 byte */
    size_t tiles_x=(cols+(size_t)beta1-1)/(size_t)beta1, tiles_y=(rows+(size_t)beta2-1)/(size_t)beta2;
    size_t n_so = tiles_x*tiles_y; if(al < n_so*8) return -1;
    for(size_t ty=0;ty<tiles_y;ty++) for(size_t tx=0;tx<tiles_x;tx++){
        size_t t=ty*tiles_x+tx; const uint8_t* s8=so+t*8;
        uint16_t ssx; memcpy(&ssx,s8,2); uint16_t ssy; memcpy(&ssy,s8+2,2);
        uint16_t zzx; memcpy(&zzx,s8+4,2); uint16_t zzy; memcpy(&zzy,s8+6,2);
        uint64_t word=dense[t];
        int scode=(int)(word & 0x7), zcode=(int)((word>>3)&0x7);
        float s_val=s_f16(ssx)*scode + s_f16(ssy);
        float z_val=s_f16(zzx)*zcode + s_f16(zzy);
        for(size_t k=0;k<(size_t)beta1;k++){
            size_t ci=tx*(size_t)beta1+k; if(ci>=cols) break;
            int q=(int)((word>>(6+3*k))&0x7);
            float w=(q - z_val)*s_val;
            for(size_t j=0;j<(size_t)beta2;j++){ size_t ri=ty*(size_t)beta2+j; if(ri>=rows) break; out[ri*cols+ci]=w; }
        }
    }
    return 0; }
static int q_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "spqr", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, q_dt, q_shape, q_ssh, q_sizes, q_decode, q_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
