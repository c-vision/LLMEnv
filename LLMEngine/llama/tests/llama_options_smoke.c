/* Integration smoke (2026-10-30): il percorso REALE che usa FastLLM per i GGUF —
 * l'engine ABI `llamacpp_ids` — deve leggere i parametri dal REGISTRO OPZIONI
 * (mai env): n_cpu_moe, llama_ngl, llama_n_batch/ubatch.
 * Uso: ./llama_options_smoke <model.gguf>
 */
#include "llmengine.h"
#include "llm_engine_opts.h"
#include <stdio.h>
/* teardown ordinato: libera modelli/sessioni PRIMA di ggml-metal (assert rsets) */
extern void llm_ai_shutdown(void);
#include <stdlib.h>
#include <string.h>

/* Simbolo interno del modulo llama: FastLLM lo risolve via dlsym con questo nome. */
extern const LlmEngineLlmEngine* llmengine_llm_engine_llamacpp_ids(void);

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <model.gguf>\n", argv[0]); return 2; }

    /* 1. opzioni PRIMA del load, come fa `fastllm --opt` */
    if (llm_ai_mlx_opt_set("n_cpu_moe", "-1") != 0)   { fprintf(stderr, "FAIL opt n_cpu_moe\n"); return 1; }
    if (llm_ai_mlx_opt_set("llama_ngl", "-1") != 0)   { fprintf(stderr, "FAIL opt llama_ngl\n"); return 1; }
    if (llm_ai_mlx_opt_set("llama_n_batch", "512") != 0)  { fprintf(stderr, "FAIL opt llama_n_batch\n"); return 1; }
    if (llm_ai_mlx_opt_set("llama_n_ubatch", "128") != 0) { fprintf(stderr, "FAIL opt llama_n_ubatch\n"); return 1; }
    if (llm_ai_mlx_opt_is_set("n_cpu_moe") != 1) { fprintf(stderr, "FAIL is_set\n"); return 1; }

    /* 2. engine ABI + load */
    const LlmEngineLlmEngine* e = llmengine_llm_engine_llamacpp_ids();
    if (e == NULL || e->load_model == NULL) { fprintf(stderr, "FAIL engine\n"); return 1; }
    LlmEngineLlmModel m = e->load_model(argv[1]);
    if (m == NULL) { fprintf(stderr, "FAIL load via ABI\n"); return 1; }

    /* 3. sessione + generazione */
    LlmEngineLlmSession s = e->create_session(m);
    if (s == NULL) { fprintf(stderr, "FAIL session\n"); return 1; }
    int32_t ids[4] = { 1, 2, 3, 4 };
    if (e->append_ids(s, ids, 4) != 0) { fprintf(stderr, "FAIL append_ids\n"); return 1; }
    int32_t out[8] = {0};
    int n = e->generate_ids(s, 8, 0, out);
    if (n <= 0) { fprintf(stderr, "FAIL generate_ids (%d)\n", n); return 1; }

    int used = 0, cap = 0;
    if (e->kv_stats) (void)e->kv_stats(s, &used, &cap);
    printf("ABI ok: %d token generati, kv %d/%d\n", n, used, cap);

    e->destroy_session(s);
    e->free_model(m);
    llm_ai_shutdown();
    printf("OK\n");
    return 0;
}
