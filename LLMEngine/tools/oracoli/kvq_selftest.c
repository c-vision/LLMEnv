#include <stdio.h>
#include <stddef.h>
extern int llm_ai_mlx_kvq_splash_selftest(char* out, size_t cap);
extern int llm_ai_mlx_kvq_page_tier_selftest(char* out, size_t cap);
extern int llm_ai_mlx_kvq_spill_policy_selftest(char* out, size_t cap);
int main(void){
  char b[8192];
  int r1=llm_ai_mlx_kvq_splash_selftest(b,sizeof b); printf("[splash] rc=%d %s\n",r1,b);
  int r2=llm_ai_mlx_kvq_page_tier_selftest(b,sizeof b); printf("[page_tier] rc=%d %s\n",r2,b);
  int r3=llm_ai_mlx_kvq_spill_policy_selftest(b,sizeof b); printf("[spill] rc=%d %s\n",r3,b);
  return (r1||r2||r3)?1:0;
}
