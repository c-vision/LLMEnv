/* Plugin CODEC di test (dylib) ABI v2: decode/encode "F32 grezzo" (identita'). */
#include "llmcodec.h"
#include <string.h>
static int s(int fmt,int p0,int p1,size_t rows,size_t cols,size_t* ob,size_t* ab){
  (void)fmt;(void)p0;(void)p1; *ob=rows*cols*4; *ab=0; return 0; }
static int d(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
  (void)il;(void)fmt;(void)p0;(void)p1;(void)aux;(void)al; memcpy(out,in,rows*cols*4); return 0; }
static int e(const float* in,size_t rows,size_t cols,int fmt,int p0,int p1,void* out,size_t oc,size_t* ol,void* aux,size_t ac,size_t* al){
  (void)fmt;(void)p0;(void)p1;(void)aux;(void)ac; if(oc<rows*cols*4) return -1; memcpy(out,in,rows*cols*4); *ol=rows*cols*4; *al=0; return 0; }
static const char* t_dt(int which,int fmt){ (void)fmt; return which==0 ? "F32" : NULL; }
static int t_shp(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){(void)fmt;(void)p0;(void)p1;*r=d0;*c=d1;return 0;}
static int t_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* a,size_t* b){(void)w;(void)fmt;(void)p0;(void)p1;*a=r;*b=c;return 0;}
static const LlmCodec C = { "test_id", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_NONE, t_dt, t_shp, t_ssh, s, d, e };
const LlmCodec* llmcodec_describe(void){ return &C; }
