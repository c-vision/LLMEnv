/* llm_perop_profile.c — profilo GPU PER PRIMITIVA MLX di N forward da M token su una sessione calda (default: decode/verify).
 *
 * Usa il profiler gia' presente nel sorgente MLX locale (libs/mlx-deps/mlx-src/mlx/backend/metal/eval.cpp, GMB_MLX_PEROP=1):
 * ogni primitiva e' chiusa in un PROPRIO command buffer e cronometrata con GPUEndTime-GPUStartTime. Qui si aggiungono le due
 * funzioni esportate gmb_perop_reset() / gmb_perop_dump_now() (stesso file) per azzerare dopo prefill+warmup e stampare solo i
 * forward misurati. Valori in stderr = ms TOTALI su `reps` forward (dividere per reps).
 *
 * AVVERTENZA (fedelta'): l'esecuzione e' SERIALIZZATA (1 primitiva = 1 command buffer): niente sovrapposizione fra dispatch
 * indipendenti (gate/up del MLP) e ogni op paga ~25-50 us di overhead di command buffer => la somma (113 ms/forward a M=8 sul
 * 27B-4bit) supera il tempo reale (70 ms). Le QUOTE relative sono utili, i valori assoluti dei kernel piccoli no.
 *
 * Build:  cc -O2 -o llm_perop_profile LLMEngine/tools/llm_perop_profile.c
 * Uso:    GMB_MLX_PEROP=1 llm_perop_profile <libllm_ai_mlx.dylib> <modello> <M> <reps> [opzione=valore ...] 2>&1 | grep -a -A40 "=== PEROP"
 * Nota:   disattiva di default dflash_model (=/nonexistent), spec_ngram e temperatura. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char**argv){
  void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if(!h){fprintf(stderr,"%s\n",dlerror());return 1;}
  void*(*load)(const char*)=dlsym(h,"llm_ai_mlx_load_model"); void*(*create)(void*)=dlsym(h,"llm_ai_mlx_create_session");
  int(*append)(void*,const int32_t*,int)=dlsym(h,"llm_ai_mlx_session_append"); int(*optset)(const char*,const char*)=dlsym(h,"llm_ai_mlx_opt_set");
  void(*reset)(void)=dlsym(h,"gmb_perop_reset"); void(*dump)(void)=dlsym(h,"gmb_perop_dump_now");
  optset("dflash_model","/nonexistent"); optset("spec_ngram","0"); optset("temp","0");
  for(int i=5;i<argc;i++){char*eq=strchr(argv[i],'='); if(!eq)continue; *eq=0; optset(argv[i],eq+1);}
  void*m=load(argv[2]); void*s=create(m); int M=atoi(argv[3]), reps=atoi(argv[4]);
  int32_t ids[512]; for(int i=0;i<512;i++) ids[i]=100+(i%2000);
  append(s,ids,256); append(s,ids,M); append(s,ids,M);
  if(reset) reset();
  for(int r=0;r<reps;r++) append(s,ids+(r*7)%300,M);
  fprintf(stderr,"=== PEROP M=%d reps=%d (valori = ms TOTALI su %d forward) ===\n",M,reps,reps);
  if(dump) dump(); return 0;}
