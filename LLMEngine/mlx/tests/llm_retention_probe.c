/* llm_retention_probe.c — §M: attribuzione della ritenzione di memoria nel decode.
 * Uso: [LLM_KVQUANT=1 ...] llm_retention_probe <model_dir> <ctx> <gen>
 * Stampa, a stadi (load / post-prefill / ogni blocco di decode), il BILANCIO array
 * (mlx_array_new*−free; net costante = nessun leak NOSTRO) + active/peak/cache MLX. */
#include "llmengine_mlx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void probe(const char* tag) {
    long n = 0, f = 0, net = 0;
    llm_ai_mlx_debug_array_balance(tag, &n, &f, &net);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "uso: %s <model_dir> <ctx> <gen>\n", argv[0]); return 2; }
    int ctx = atoi(argv[2]), gen = atoi(argv[3]);
    if (ctx < 1 || gen < 1) return 2;

    llm_ai_mlx_model_handle mh = llm_ai_mlx_load_model(argv[1]);
    if (!mh) { fprintf(stderr, "load_model fallita\n"); return 3; }
    probe("post-load");

    llm_ai_mlx_session_handle sh = llm_ai_mlx_create_session(mh);
    int32_t* prompt = (int32_t*)malloc((size_t)ctx * sizeof(int32_t));
    for (int i = 0; i < ctx; i++) prompt[i] = 1 + (i % 1000);
    if (llm_ai_mlx_session_append(sh, prompt, ctx) != 0) { fprintf(stderr, "append fallito\n"); return 4; }
    free(prompt);
    probe("post-prefill");

    int32_t out[64];
    for (int b = 0; b < gen; b += 16) {
        int n = (gen - b) < 16 ? (gen - b) : 16;
        if (llm_ai_mlx_session_generate(sh, n, -1, out) != n) { fprintf(stderr, "generate fallito\n"); return 5; }
        char tag[64]; snprintf(tag, sizeof(tag), "post-decode-%d", b + n);
        probe(tag);
    }
    llm_ai_mlx_destroy_session(sh);
    probe("post-destroy-session");
    return 0;
}
