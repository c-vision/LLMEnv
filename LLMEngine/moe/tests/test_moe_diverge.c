/* Self-test: routing compaction (sort/group) — correttezza + stabilita'. */
#include <stdio.h>
#include <string.h>
extern int llmengine_moe_diverge_plan(const int*, int, int, int*, int*);
int main(void){
    const int a[10] = {2,0,2,1,0,2,1,1,0,2}; int es[6]; int sorted[10];
    if (llmengine_moe_diverge_plan(a,10,5,es,sorted)!=0){ printf("FAIL rc\n"); return 1; }
    int exp[6]; memset(exp,0,sizeof exp); for(int i=0;i<10;i++) exp[a[i]+1]++; for(int e=0;e<5;e++) exp[e+1]+=exp[e];
    for(int e=0;e<=5;e++) if(es[e]!=exp[e]){ printf("FAIL experts e=%d got=%d exp=%d\n",e,es[e],exp[e]); return 1; }
    /* sorted: tutti gli appartenenti all'esperto e in ordine stabile */
    int occ[10]; memset(occ,0,sizeof occ);
    for(int e=0;e<5;e++) for(int k=es[e];k<es[e+1];k++){ int idx=sorted[k]; if(a[idx]!=e){ printf("FAIL group e=%d\n",e); return 1; } occ[idx]++; }
    for(int i=0;i<10;i++) if(occ[i]!=1){ printf("FAIL permutation i=%d\n",i); return 1; }
    /* stabilita': per ogni esperto, gli indici crescenti */
    for(int e=0;e<5;e++) for(int k=es[e]+1;k<es[e+1];k++) if(sorted[k]<sorted[k-1]){ printf("FAIL stable e=%d\n",e); return 1; }
    if(llmengine_moe_diverge_plan(a,10,5,es,NULL)!=0){ printf("FAIL rc2\n"); return 1; }
    if(llmengine_moe_diverge_plan((int[]){3},1,2,es,NULL)!=-1){ printf("FAIL range\n"); return 1; }
    printf("PASS llmengine_moe_diverge: compaction ok (stabile, range-check)\n"); return 0;
}
