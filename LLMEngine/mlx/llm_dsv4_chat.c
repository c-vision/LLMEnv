/* llm_dsv4_chat.c — driver DS4 end-to-end con testo, 100% dal GGUF: tokenizer DeepSeek letto
 * dal GGUF (llm_dsv4_tokenizer) + engine DS4 (packed) + decode. Uso: <gguf> "<prompt>" [n_new] [ctx] */
#include <stdio.h>
#include "llm_engine_opts.h"   /* opt_env: registro */
#include <stdlib.h>
#include <string.h>
#include "llm_dsv4_engine.h"
#include "llm_dsv4_tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "uso: %s <model.gguf> \"<prompt>\" [n_new] [ctx]\n", argv[0]); return 2; }
    const char* gguf = argv[1], *prompt = argv[2];
    int n_new = argc > 3 ? atoi(argv[3]) : 4;
    int ctx = argc > 4 ? atoi(argv[4]) : 256;

    const char* sw = llm_ai_mlx_opt_env("LLM_DSV4_SWAP");
    int swap_cap = sw ? atoi(sw) : 0;
    LlmDsv4Engine* e = swap_cap > 0 ? llm_dsv4_engine_load_swap(gguf, ctx, swap_cap)
                                    : llm_dsv4_engine_load(gguf, ctx);
    if (!e) { fprintf(stderr, "engine load fallito: %s\n", gguf); return 1; }
    if (swap_cap > 0 && e->store)
        printf("store: SWAP cap=%d bundle\n", swap_cap);
    llmtoken_handle tok = llm_dsv4_tokenizer_from_gguf(&e->gguf);   /* vocab+merges dal GGUF */
    if (!tok) { fprintf(stderr, "tokenizer dal GGUF fallito\n"); return 1; }
    printf("modello: layer=%d embd=%d vocab=%d experts=%d/%d (packed, tokenizer da GGUF)\n",
           e->w.cfg.n_layer, e->w.cfg.n_embd, e->w.cfg.n_vocab, e->w.cfg.n_expert, e->w.cfg.n_expert_used);

    int32_t ids[512];
    int np = llmtoken_encode(tok, prompt, ids, 512);
    if (np <= 0) { fprintf(stderr, "encode fallito\n"); return 1; }
    printf("prompt (%d token):", np); for (int i = 0; i < np; i++) printf(" %d", ids[i]); printf("\n");

    int total = np + n_new;
    llm_dsv4_engine_generate(e, ids, np, n_new);
    printf("output (%d token):", total); for (int i = np; i < total; i++) printf(" %d", ids[i]); printf("\n");
    char text[4096];
    int nbytes = llmtoken_decode(tok, ids, total, text, (int)sizeof text - 1);
    if (nbytes >= 0) { text[nbytes] = '\0'; printf("testo: %s\n", text); }

    if (e->store)
        printf("swap metriche: hit=%llu miss=%llu evict=%llu bytes_read=%.1f MB\n",
               (unsigned long long)e->store->hits, (unsigned long long)e->store->misses,
               (unsigned long long)e->store->evictions, (double)e->store->bytes_read/1e6);
    llmtoken_free(tok);
    llm_dsv4_engine_free(e);
    return 0;
}
