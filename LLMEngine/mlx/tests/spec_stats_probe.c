/* spec_stats_probe (2026-10-30): verifica end-to-end delle statistiche di speculazione
 * (ABI v7): carica un modello MLX, genera N token con spec_ngram attivo e stampa
 * drafts/accepted/rate/AL + accept_h{i}. Nessuna modifica al motore: usa solo l'ABI.
 * Uso: ./spec_stats_probe <model_dir> [max_new] [spec_ngram]
 */
#include "llmengine.h"
#include "llmengine_mlx.h"
#include "llmtoken.h"
#include "llm_engine_opts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_n_ids = 0;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <model_dir> [max_new] [spec_ngram]\n", argv[0]); return 2; }
    int max_new = (argc > 2) ? atoi(argv[2]) : 48;
    const char* sn = (argc > 3) ? argv[3] : "2";

    llm_ai_mlx_opt_set("spec_ngram", sn);
    llm_ai_mlx_opt_set("context", "4096");
    const LlmEngineLlmEngine* e = llmengine_llm_engine_mlx();
    if (e == NULL || e->spec_stats == NULL) { fprintf(stderr, "FAIL: engine senza spec_stats (ABI %u)\n", (unsigned)llmengine_abi_version()); return 1; }
    LlmEngineLlmModel m = e->load_model(argv[1]);
    if (m == NULL) { fprintf(stderr, "FAIL: load\n"); return 1; }
    LlmEngineLlmSession s = e->create_session(m);
    if (s == NULL) { fprintf(stderr, "FAIL: session\n"); return 1; }

    LlmEngineLlmSampling sp; memset(&sp, 0, sizeof sp);
    sp.temperature = 0.0f; sp.top_p = 1.0f; sp.top_k = 0; sp.seed = 42;
    if (e->set_sampling) (void)e->set_sampling(s, &sp);

    /* prompt di 64 id qualsiasi: basta a far girare router + speculazione */
    int32_t ids[64];
    /* Prompt reale: con testo ripetuto l'n-gram trova match (senza ripetizioni non propone
     * nulla e i contatori restano a zero — corretto, non un difetto). */
    { char tp[1200]; snprintf(tp, sizeof tp, "%s/tokenizer.json", argv[1]);
      llmtoken_handle tk = llmtoken_load(tp);
      if (tk == NULL) { fprintf(stderr, "FAIL: tokenizer\n"); return 1; }
      const char* txt = "Rome is the capital of Italy. Rome is the capital of Italy. "
                        "Rome is the capital of Italy. Rome is the capital of Italy.";
      int nn = llmtoken_encode(tk, txt, ids, 64);
      if (nn <= 0) { fprintf(stderr, "FAIL: encode\n"); return 1; }
      for (int i = 0; i < nn; ++i) ids[i] = ids[i];
      memset(ids + nn, 0, 0);
      llmtoken_free(tk);
      g_n_ids = nn; }
    if (e->append_ids(s, ids, g_n_ids) != 0) { fprintf(stderr, "FAIL: append\n"); return 1; }
    /* `generate_ngram` e' il percorso self-spec (K=draft_len): e' lui che aggiorna i contatori
     * F20/F22. `generate_ids` dell'ABI non usa la speculazione, quindi non serve qui. */
    int32_t out[512];
    int n = llm_ai_mlx_session_generate_ngram((llm_ai_mlx_session_handle)s, max_new, -1, out, 3);
    if (n <= 0) { fprintf(stderr, "FAIL: generate_ngram (%d)\n", n); return 1; }

    LlmEngineLlmSpecStats st; memset(&st, 0, sizeof st); st.size = (int32_t)sizeof st;
    if (e->spec_stats(s, &st) != 0) { fprintf(stderr, "FAIL: spec_stats\n"); return 1; }

    printf("ABI %u · generati %d token · spec_ngram=%s\n", (unsigned)llmengine_abi_version(), n, sn);
    printf("drafts=%lld accepted=%lld rate=%.3f rounds=%lld AL=%.2f\n",
           (long long)st.drafts, (long long)st.accepted, st.rate, (long long)st.rounds, st.al);
    printf("accept_h:");
    for (int i = 0; i < st.npos; ++i) {
        double h = st.proposed[i] > 0 ? (double)st.accepted_pos[i] / (double)st.proposed[i] : 0.0;
        printf(" h%d=%.2f (%d/%d)", i + 1, h, st.accepted_pos[i], st.proposed[i]);
    }
    printf("\n");

    e->destroy_session(s);
    e->free_model(m);
    return 0;
}
