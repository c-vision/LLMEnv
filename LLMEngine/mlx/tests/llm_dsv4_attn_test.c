/* llm_dsv4_attn_test.c — verifica riuso dsv41 compressor_pool/index_score per DS4 (ratio4/128). */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv41_math.h"
int main(void){
    int fail=0;
    /* compressor pooling: softmax per-dimensione sui ratio valori, somma pesata */
    for (int ratio = 4; ratio <= 128; ratio = (ratio==4?128:129)) {
        const int H=8, m=2; float kv[m*ratio*H], gate[m*ratio*H], out[m*H];
        for(int i=0;i<m*ratio*H;i++){ kv[i]=0.1f*(float)((i*3)%11)-0.5f; gate[i]=0.2f*(float)((i*5)%7)-0.6f; }
        llm_dsv41_compressor_pool(kv, gate, ratio, H, m, out);
        int bad=0;
        for(int i=0;i<m;i++) for(int d=0;d<H;d++){
            float mx=gate[(i*ratio)*H+d]; for(int k=1;k<ratio;k++){float g=gate[(i*ratio+k)*H+d]; if(g>mx)mx=g;}
            float s=0; for(int k=0;k<ratio;k++) s+=expf(gate[(i*ratio+k)*H+d]-mx);
            float acc=0; for(int k=0;k<ratio;k++) acc+=kv[(i*ratio+k)*H+d]*expf(gate[(i*ratio+k)*H+d]-mx)/s;
            if(fabsf(acc-out[i*H+d])>1e-5f) bad++;
        }
        printf("%s compressor_pool ratio=%d\n", bad?"FAIL":"PASS", ratio); if(bad) fail++;
    }
    /* index score: sum_h relu(dot(q_h,k_h))*w_h */
    { const int NH=4, IH=8; float q[NH*IH], k[NH*IH], w[NH];
      for(int i=0;i<NH*IH;i++){ q[i]=0.1f*(float)((i*7)%9)-0.4f; k[i]=0.1f*(float)((i*5)%11)-0.5f; }
      for(int i=0;i<NH;i++) w[i]=0.3f+0.1f*(float)i;
      float s=llm_dsv41_index_score(q,k,w,NH,IH);
      float acc=0; for(int h=0;h<NH;h++){ float dot=0; for(int d=0;d<IH;d++) dot+=q[h*IH+d]*k[h*IH+d]; if(dot>0) acc+=dot*w[h]; }
      int ok=fabsf(acc-s)<1e-5f; printf("%s %-30s\n", ok?"PASS":"FAIL", "index_score"); if(!ok) fail++; }
    printf(fail?"FAILURES: %d\n":"ALL PASS\n", fail);
    return fail;
}
