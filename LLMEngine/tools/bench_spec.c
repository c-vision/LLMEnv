/* bench_spec (2026-10-31) — confronto AR contro speculazione n-gram su PROMPT SET FISSI.
 *
 * Perche' esiste: il tok/s da solo non dice se la speculazione conviene (dipende dal contenuto) e
 * non dice DOVE si perde. Qui si stampa, per dominio e per prompt:
 *   AR   = llm_ai_mlx_session_generate (greedy, nessuna speculazione)
 *   SPEC = llm_ai_mlx_session_build_ngram + generate_ngram(K=3)
 * con **wall-clock** (la verita'), AL = token confermati per verifica, accept_h{i}.
 *
 * Regole: prompt CONGELATI in LLMEngine/tools/prompts/spec/*.txt (separatore `---PROMPT---`),
 * stesso numero di token generati per AR e SPEC, stessa macchina, nessuna env (registro opzioni).
 *
 * Uso: ./bench_spec <model_dir> [max_new=192] [K=3]
 */
#include "llmengine.h"
#include "llmengine_mlx.h"
#include "llmtoken.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

#define MAX_PROMPTS 8
#define MAX_IDS     4096

typedef struct { char text[8192]; } Prompt;

static int load_prompts(const char* path, Prompt* out, int cap) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    char* buf = malloc(1 << 20);
    if (buf == NULL) { fclose(f); return -1; }
    size_t n = fread(buf, 1, (1 << 20) - 1, f); buf[n] = 0; fclose(f);
    int np = 0;
    char* p = buf;
    while (np < cap) {
        char* s = strstr(p, "---PROMPT---");
        if (s == NULL) break;
        s += strlen("---PROMPT---");
        char* e = strstr(s, "---PROMPT---");
        size_t len = e ? (size_t)(e - s) : strlen(s);
        if (len >= sizeof out[np].text) len = sizeof out[np].text - 1;
        memcpy(out[np].text, s, len); out[np].text[len] = 0;
        np++; p = e ? e : s + len;
        if (e == NULL) break;
    }
    free(buf);
    return np;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <model_dir> [max_new=192] [K=3]\n", argv[0]); return 2; }
    const char* dir = argv[1];
    int max_new = (argc > 2) ? atoi(argv[2]) : 192;
    int K = (argc > 3) ? atoi(argv[3]) : 3;

    llm_ai_mlx_opt_set("context", "4096");
    llm_ai_mlx_opt_set("spec_ngram", "2");

    char tp[1200]; snprintf(tp, sizeof tp, "%s/tokenizer.json", dir);
    llmtoken_handle tk = llmtoken_load(tp);
    if (tk == NULL) { fprintf(stderr, "FAIL: tokenizer.json in %s\n", dir); return 1; }

    const LlmEngineLlmEngine* e = llmengine_llm_engine_mlx();
    if (e == NULL || e->spec_stats == NULL) { fprintf(stderr, "FAIL: engine/ABI v7\n"); return 1; }
    LlmEngineLlmModel m = e->load_model(dir);
    if (m == NULL) { fprintf(stderr, "FAIL: load\n"); return 1; }

    static const char* doms[] = { "prose", "code", "math", "repeat" };
    printf("modello: %s · max_new=%d · K=%d · ABI %u\n\n", dir, max_new, K, (unsigned)llmengine_abi_version());
    printf("%-7s %3s %8s %8s %7s %6s %6s  accept_h\n", "dominio", "#", "AR tok/s", "SP tok/s", "speedup", "AL", "rate");

    double sum_ar = 0, sum_sp = 0; int nrun = 0;
    for (size_t d = 0; d < sizeof doms / sizeof doms[0]; ++d) {
        char path[512]; snprintf(path, sizeof path, "LLMEngine/tools/prompts/spec/%s.txt", doms[d]);
        Prompt pr[MAX_PROMPTS];
        int np = load_prompts(path, pr, MAX_PROMPTS);
        if (np <= 0) { fprintf(stderr, "[%s] prompt set non trovato: %s\n", doms[d], path); continue; }
        for (int i = 0; i < np; ++i) {
            int32_t ids[MAX_IDS];
            int n_ids = llmtoken_encode(tk, pr[i].text, ids, MAX_IDS);
            if (n_ids <= 0) { fprintf(stderr, "[%s #%d] encode fallito\n", doms[d], i); continue; }

            double t_ar = 0, t_sp = 0; int n_ar = 0, n_sp = 0;
            LlmEngineLlmSpecStats st; memset(&st, 0, sizeof st);

            /* AR */
            {   LlmEngineLlmSession s = e->create_session(m);
                if (s == NULL) continue;
                LlmEngineLlmSampling sp; memset(&sp, 0, sizeof sp);
                sp.temperature = 0.0f; sp.top_p = 1.0f; sp.seed = 42;
                if (e->set_sampling) (void)e->set_sampling(s, &sp);
                if (e->append_ids(s, ids, n_ids) == 0) {
                    int32_t out[512];
                    double t0 = now_s();
                    n_ar = llm_ai_mlx_session_generate((llm_ai_mlx_session_handle)s, max_new, -1, out);
                    t_ar = now_s() - t0;
                }
                e->destroy_session(s);
            }
            /* SPEC */
            {   LlmEngineLlmSession s = e->create_session(m);
                if (s == NULL) continue;
                LlmEngineLlmSampling sp; memset(&sp, 0, sizeof sp);
                sp.temperature = 0.0f; sp.top_p = 1.0f; sp.seed = 42;
                if (e->set_sampling) (void)e->set_sampling(s, &sp);
                if (e->append_ids(s, ids, n_ids) == 0) {
                    int32_t out[512];
                    (void)llm_ai_mlx_session_build_ngram((llm_ai_mlx_session_handle)s, 2);
                    double t0 = now_s();
                    n_sp = llm_ai_mlx_session_generate_ngram((llm_ai_mlx_session_handle)s, max_new, -1, out, K);
                    t_sp = now_s() - t0;
                    st.size = (int32_t)sizeof st;
                    (void)e->spec_stats(s, &st);
                }
                e->destroy_session(s);
            }
            double ar_tps = (t_ar > 0 && n_ar > 0) ? n_ar / t_ar : 0.0;
            double sp_tps = (t_sp > 0 && n_sp > 0) ? n_sp / t_sp : 0.0;
            /* Se il percorso spec si ferma prima (EOS o ngram non costruito) il rapporto non e'
             * confrontabile: n_sp basso con tempo basso darebbe tok/s gonfiati. */
            const int comparable = (n_ar >= max_new - 1) && (n_sp >= max_new - 1);
            double sp_up  = (ar_tps > 0) ? sp_tps / ar_tps : 0.0;
            printf("%-7s %3d %8.1f %8.1f %6.2fx %6.2f %6.2f %s ", doms[d], i, ar_tps, sp_tps, sp_up, st.al, st.rate,
                   comparable ? "  " : "(n.c.)");
            for (int k = 0; k < st.npos; ++k)
                printf("h%d=%.2f ", k + 1, st.proposed[k] ? (double)st.accepted_pos[k] / st.proposed[k] : 0.0);
            printf("\n");
            fflush(stdout);
            if (comparable && ar_tps > 0 && sp_tps > 0) { sum_ar += ar_tps; sum_sp += sp_tps; nrun++; }
        }
    }
    if (nrun > 0)
        printf("\nmedia: AR %.1f tok/s · SPEC %.1f tok/s · speedup %.2fx (%d run)\n",
               sum_ar / nrun, sum_sp / nrun, sum_sp / sum_ar, nrun);
    e->free_model(m);
    llmtoken_free(tk);
    return 0;
}
