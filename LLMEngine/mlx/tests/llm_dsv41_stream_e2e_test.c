/* llm_dsv41_stream_e2e_test.c — E2E DS4.1: motore materializzato (safetensors) vs STREAMING (GGUF).
 * Build: tools/dsv41_stream_e2e_test.sh (tiny: gen_dsv41_st_ref.py --no-probes + llmmodelio_model_quantize). */
#include <stdio.h>
#include <string.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_engine.h"

static int gen(LlmDsv41Engine* e, mlx_stream s, const int32_t* prompt, int np, int32_t* out) {
    LlmDsv41Cache* c = NULL; int off = 0, pend = 0; bool hp = false;
    LlmDsv41Sampling sm; memset(&sm, 0, sizeof sm); sm.active = false;
    if (llm_dsv41_engine_append(e, (void**)&c, prompt, np, &off, &pend, &hp, &sm, s) != 0) return -1;
    int n = llm_dsv41_engine_generate(e, (void**)&c, &off, pend, hp, 6, -1, out, &sm, s);
    llm_dsv41_cache_free(c); free(c);
    return n;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stderr,"[1] gpu stream...\n");
    mlx_stream s = mlx_default_gpu_stream_new();
    fprintf(stderr,"[2] stream ok\n");
    const char* dir  = "/tmp/dsv41_st_np/sharded";
    const char* cont = "/tmp/dsv41_st_np/tinyGGUF";
    int32_t prompt[4] = { 3, 7, 5, 2 };

    fprintf(stderr,"[3] load materializzato...\n");
    LlmDsv41Engine* A = llm_dsv41_engine_load(dir, s);
    fprintf(stderr,"[4] mat load %p\n",(void*)A);
    if (!A) { printf("FAIL load materializzato\n"); return 1; }
    fprintf(stderr,"[5] load streaming...\n");
    LlmDsv41Engine* B = llm_dsv41_engine_load_container_stream(dir, cont, s);
    fprintf(stderr,"[6] stream load %p\n",(void*)B);
    if (!B) { printf("FAIL load streaming\n"); return 1; }

    int32_t ga[6] = {0}, gb[6] = {0};
    fprintf(stderr,"[7] gen A...\n");
    int na = gen(A, s, prompt, 4, ga);
    fprintf(stderr,"[8] gen A=%d\n",na);
    fprintf(stderr,"[9] gen B...\n");
    int nb = gen(B, s, prompt, 4, gb);
    fprintf(stderr,"[10] gen B=%d\n",nb);
    printf("materializzato: n=%d [%d %d %d %d %d %d]\n", na, ga[0],ga[1],ga[2],ga[3],ga[4],ga[5]);
    printf("streaming:      n=%d [%d %d %d %d %d %d]\n", nb, gb[0],gb[1],gb[2],gb[3],gb[4],gb[5]);
    int ok = (na == nb && na > 0) && memcmp(ga, gb, (size_t)na * sizeof(int32_t)) == 0;
    printf(ok ? "E2E PASS: streaming == materializzato\n" : "E2E FAIL\n");
    llm_dsv41_engine_free(A); llm_dsv41_engine_free(B);
    return ok ? 0 : 1;
}
