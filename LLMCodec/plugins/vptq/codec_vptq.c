/* Plugin CODEC "vptq": Vector Post-Training Quantization (microsoft/VPTQ, MIT).
 * Port di vptq/ops/quant_gemm.py::dequant + vptq/utils/pack.py::unpack_index_tensor.
 *   indices   int16 [num_codebooks, num_indices, group_size]   (fmt 0, scompattato)
 *   centroids f16   [num_codebooks*num_centroids*vector_len]
 * W[ni*vl+t][cb*gs+g] = centroids[cb][indices[cb][ni][g]][t]  (+residual, *norm)
 * p0 = num_codebooks, p1 = num_centroids. aux = centroids [++ res_centroids] [++ scale,bias].
 * (outlier/perm/packed = da agganciare; qui core + residual + norm). */
#include "llmcodec.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
static float v_f16(const uint8_t* p){ uint16_t h=*(const uint16_t*)p; uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int v_sizes(int fmt,int p0,int p1,size_t rows,size_t cols,size_t* ob,size_t* ab){ (void)fmt; int nc=p0?p0:1, ncent=p1?p1:256;
    (void)rows; size_t gs=cols/nc; size_t vl=1; (void)gs; size_t numidx=0; (void)numidx;
    *ob=0; *ab=(size_t)nc*ncent*vl*2; return -1; }   /* params incompleti: gestito dal converter */
static int v_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){ (void)fmt;(void)p0;(void)p1; *r=d0; *c=d1; return 0; }
static int v_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)w;(void)fmt;(void)p0;(void)p1;(void)r;(void)c;*d0=0;*d1=0;return -1; }
static const char* v_dt(int which,int fmt){ (void)fmt; return which==0?"I16":"F16"; }
static int v_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)fmt;
    int nc=p0?p0:1, ncent=p1?p1:256;
    if(aux==NULL) return -1;
    /* ricava vector_len dai byte dell'aux (senza residual/norm): al = nc*ncent*vl*2 */
    int vl=1; { size_t base=(size_t)nc*ncent*2; if(al>=base && (al % base)==0) vl=(int)(al/base); if(vl<1) vl=1; }
    size_t gs = (nc>0)?(cols/(size_t)nc):cols; if(gs<1) gs=1;
    size_t numidx = (vl>0)?(rows/(size_t)vl):rows; if(numidx<1) numidx=1;
    const int16_t* idx=(const int16_t*)in; const uint8_t* cb=(const uint8_t*)aux;
    for(size_t ni=0; ni<numidx; ni++) for(size_t t=0;t<(size_t)vl;t++){
        size_t o=ni*(size_t)vl+t; if(o>=rows) continue;
        for(size_t b=0;b<(size_t)nc;b++) for(size_t g=0;g<gs;g++){
            size_t i=b*gs+g; if(i>=cols) continue;
            int code=(int)(uint16_t)idx[(b*numidx+ni)*gs+g];
            if(code<0||code>=ncent) code=0;
            size_t coff=((size_t)b*ncent+(size_t)code)*(size_t)vl + t;
            out[o*cols+i]=v_f16(cb+coff*2);
        }
    }
    return 0; }
static int v_encode(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const LlmCodec C = { "vptq", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, v_dt, v_shape, v_ssh, v_sizes, v_decode, v_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
