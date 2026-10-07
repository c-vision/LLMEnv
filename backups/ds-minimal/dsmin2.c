/* dsmin2.c — driver UNICO engine-only che usa la STESSA ABI (llmengine.h) per ENTRAMBI i motori:
 *   llmengine_llm_engine_ds4       (vendored, libds4engine.a)
 *   llmengine_llm_engine_ds4slow   (ds-slow, libllm_ai_mlx)
 * Chiama SOLO load / create_session / append_ids / generate_ids (+ kv_set_max / set_expert_swap) =
 * esattamente la superficie che ds-slow espone. Niente fastllm, niente CLI, niente ds-server.
 *
 * Uso: dsmin2 <vendored|slow> <model.gguf> <max_new> <id,id,...>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "llmengine.h"

extern const LlmEngineLlmEngine* llmengine_llm_engine_ds4(void);
extern const LlmEngineLlmEngine* llmengine_llm_engine_ds4slow(void);
extern double llm_dsv4_metal_gpu_total_ms(void);          /* slow: GPU reale (ms) */
extern long llm_dsv4_metal_command_buffers(void);         /* slow (profiler, altro AI) */
extern long llm_dsv4_metal_dispatches(void);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "uso: %s <vendored|slow> <model.gguf> <max_new> <id,id,...>\n", argv[0]); return 2; }
    const char* which = argv[1]; const char* model = argv[2]; int maxn = atoi(argv[3]);
    int slow = strcmp(which, "slow") == 0;
    const LlmEngineLlmEngine* E = slow ? llmengine_llm_engine_ds4slow() : llmengine_llm_engine_ds4();
    if (!E) { fprintf(stderr, "engine nullo\n"); return 1; }
    /* opzioni del registro (5° arg in poi, "k=v"). `eos=N` e' SPECIALE: seleziona l'eos di generate_ids
     * (default 1 = eos reale del DS4; eos=-1 forza N token fissi -> i motori fanno lo STESSO lavoro
     * forzato). Stampa SEMPRE l'eos usato + conteggio + id, cosi' l'A/B puo' verificarli. */
    int eos = 1;
    { extern int llm_ai_mlx_opt_set(const char*, const char*);
      for (int i = 5; i < argc; i++) { char* eq = strchr(argv[i], '='); if (!eq) continue; *eq = 0;
        if (!strcmp(argv[i], "eos")) { eos = atoi(eq + 1); } else { llm_ai_mlx_opt_set(argv[i], eq + 1); } } }
    fprintf(stderr, "[dsmin] eos=%d\n", eos);
    if (E->set_expert_swap) E->set_expert_swap(model);
    LlmEngineLlmModel m = E->load_model(model);
    if (!m) { fprintf(stderr, "load fallita\n"); return 1; }
    LlmEngineLlmSession s = E->create_session(m);
    if (!s) { fprintf(stderr, "session fallita\n"); return 1; }
    if (E->kv_set_max) E->kv_set_max(m, 4096);

    int32_t ids[4096]; int n = 0;
    { char* b = strdup(argv[4]); char* p = strtok(b, ",");
      while (p && n < 4096) { ids[n++] = atoi(p); p = strtok(NULL, ","); } free(b); }
    if (E->append_ids(s, ids, n) != 0) { fprintf(stderr, "append fallita\n"); return 1; }

    int32_t out[256];
    long cb0 = slow ? llm_dsv4_metal_command_buffers() : 0L;
    long dp0 = slow ? llm_dsv4_metal_dispatches() : 0;
    double gp0 = slow ? llm_dsv4_metal_gpu_total_ms() : 0.0;
    double a = now();
    int got = E->generate_ids(s, maxn, eos, out);
    double b = now(); double ms = (b - a) * 1e3;
    long cb1 = slow ? llm_dsv4_metal_command_buffers() : 0L;
    long dp1 = slow ? llm_dsv4_metal_dispatches() : 0;
    double gp1 = slow ? llm_dsv4_metal_gpu_total_ms() : 0.0;
    fprintf(stderr, "[dsmin-cb-%s] command_buffers=%ld (%.1f/tok) dispatch=%ld | GPU-reale=%.1f ms (%.1f/tok) | wall=%.1f ms (%.1f/tok)\n",
            which, cb1 - cb0, got ? (double)(cb1 - cb0) / got : 0, dp1 - dp0,
            gp1 - gp0, got ? (gp1 - gp0) / got : 0, ms, got ? ms / got : 0);
    printf("[dsmin-%s] %d tok, %.1f ms/tok (%.2f tok/s)\n", which, got, got ? ms / got : 0, ms ? (double)got * 1000 / ms : 0);
    printf("[ids-%s] ", which); for (int i = 0; i < got; i++) printf("%d%s", out[i], i + 1 < got ? "," : ""); printf("\n");
    return 0;
}
