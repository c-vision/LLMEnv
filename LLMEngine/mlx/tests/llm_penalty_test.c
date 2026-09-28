/* llm_penalty_test.c — §I: frequency/presence penalty separate (E2E su modello reale).
 * Uso: llm_penalty_test <model_dir>
 * Con sampling (temp>0) e seed fisso: A senza penalty, B con frequency=presence=2.0.
 * Verifica: l'output CAMBIA e le ripetizioni (token adiacenti uguali) NON aumentano. */
#include "llmengine_mlx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NP 48
#define NG 32

static int adjacent_repeats(const int32_t* ids, int n) {
    int r = 0;
    for (int i = 1; i < n; i++) if (ids[i] == ids[i - 1] || (i >= 2 && ids[i] == ids[i - 2])) r++;
    return r;
}

static int run(llm_ai_mlx_model_handle mh, const int32_t* p, int np, float fp, float pp, int32_t* out) {
    llm_ai_mlx_session_handle sh = llm_ai_mlx_create_session(mh);
    if (llm_ai_mlx_session_append(sh, p, np) != 0) return -1;
    llm_ai_mlx_session_set_sampling(sh, 0.8f, 0.95f, 0, 42);
    if (fp > 0.0f) llm_ai_mlx_session_set_frequency_penalty(sh, fp);
    if (pp > 0.0f) llm_ai_mlx_session_set_presence_penalty(sh, pp);
    int n = llm_ai_mlx_session_generate(sh, NG, -1, out);
    llm_ai_mlx_destroy_session(sh);
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <model_dir>\n", argv[0]); return 2; }
    int32_t prompt[NP], a[NG], b[NG];
    for (int i = 0; i < NP; i++) prompt[i] = 1 + (i % 1000);

    llm_ai_mlx_model_handle mh = llm_ai_mlx_load_model(argv[1]);
    if (!mh) { fprintf(stderr, "load_model fallita\n"); return 3; }

    int na = run(mh, prompt, NP, 0.0f, 0.0f, a);
    int nb = run(mh, prompt, NP, 2.0f, 2.0f, b);
    if (na != NG || nb != NG) { fprintf(stderr, "generate fallita (%d,%d)\n", na, nb); return 4; }

    printf("A (no penalty) :"); for (int i = 0; i < NG; i++) printf(" %d", a[i]); printf("\n");
    printf("B (fp=pp=2.0)  :"); for (int i = 0; i < NG; i++) printf(" %d", b[i]); printf("\n");

    int diff = (memcmp(a, b, sizeof(a)) != 0);
    int ra = adjacent_repeats(a, NG), rb = adjacent_repeats(b, NG);
    printf("output diverso=%d  ripetizioni adiacenti: A=%d B=%d\n", diff, ra, rb);
    int ok = diff && (rb < ra);
    printf("penalty: %s\n", ok ? "OK (cambia l'output e non aumenta le ripetizioni)" : "FAIL");
    return ok ? 0 : 1;
}
