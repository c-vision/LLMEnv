/* moe_ep_run.c — ESECUZIONE DISTRIBUITA EP multi-processo (§4.6). Fork di WORLD rank; ogni rank
 * calcola SOLO il contributo dei suoi esperti (expert-parallel) in una regione mmap condivisa;
 * il padre fa l'ALL-REDUCE (somma) -> verifica vs la reference single-process.
 * Uso: moe_ep_run [world]   (default 3) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
extern int llmengine_moe_diverge_plan(const int*, int, int, int*, int*);
extern int llmengine_moe_diverge_forward(const float*, int, int, int, const float*, int, const int*, const float*, int, float*);

int main(int argc, char** argv) {
    int world = argc > 1 ? atoi(argv[1]) : 3; if (world < 1) world = 1;
    int T = 8, H = 6, O = 5, E = 6, K = 2, n = T * K;
    float* x = malloc(sizeof(float)*T*H); for (int i = 0; i < T*H; i++) x[i] = sinf(i*0.3f);
    float* W = malloc(sizeof(float)*(size_t)E*H*O); for (int i = 0; i < E*H*O; i++) W[i] = cosf(i*0.17f);
    int* aex = malloc(sizeof(int)*n); float* aw = malloc(sizeof(float)*n); int sd = 12345;
    for (int a = 0; a < n; a++) { sd = (sd*1103515245+12345)&0x7fffffff; aex[a] = sd % E; aw[a] = 0.25f + (float)((sd>>8)%4)/4.0f; }
    int* start = malloc(sizeof(int)*(E+1)); int* sorted = malloc(sizeof(int)*n);
    if (llmengine_moe_diverge_plan(aex, n, E, start, sorted) != 0) { printf("FAIL plan\n"); return 1; }
    float* ref = calloc((size_t)T*O, sizeof(float));
    llmengine_moe_diverge_forward(x, T, H, O, W, E, aex, aw, K, ref);

    /* regione condivisa [world][T][O] (rank r scrive la sua parte) */
    size_t sz = (size_t)world*T*O*sizeof(float);
    float* sh = (float*)mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
    if (sh == MAP_FAILED) { printf("FAIL mmap\n"); return 1; }
    memset(sh, 0, sz);

    printf("# EP distribuito: world=%d experts=%d\n", world, E);
    for (int r = 0; r < world; r++) { printf("  rank %d: esperti", r); for (int e = r; e < E; e += world) printf(" %d", e); printf("\n"); }
    for (int r = 0; r < world; r++) {
        pid_t pid = fork();
        if (pid == 0) {
            float* o = sh + (size_t)r*T*O;
            for (int e = r; e < E; e += world)
                for (int k = start[e]; k < start[e+1]; k++) {
                    int a = sorted[k], ti = a/K; float w = aw[a];
                    for (int oo = 0; oo < O; oo++) { float acc = 0; for (int h = 0; h < H; h++) acc += x[ti*H+h]*W[((size_t)e*H+h)*O+oo]; o[ti*O+oo] += w*acc; }
                }
            _exit(0);
        }
        if (pid < 0) { printf("FAIL fork\n"); return 1; }
    }
    for (int r = 0; r < world; r++) wait(NULL);

    float* out = calloc((size_t)T*O, sizeof(float));   /* ALL-REDUCE */
    for (int r = 0; r < world; r++) for (int i = 0; i < T*O; i++) out[i] += sh[(size_t)r*T*O + i];
    float md = 0; for (int i = 0; i < T*O; i++) { float d = fabsf(out[i]-ref[i]); if (d > md) md = d; }
    if (md > 1e-4f) { printf("FAIL EP distribuito vs single-process: max|diff|=%.6f\n", md); return 1; }
    printf("PASS moe_ep_run: EP multi-processo (%d rank) == single-process (max|diff|=%.2e)\n", world, md);
    return 0;
}
