/* gguf_convert.c — CLI: <mlx_dir> <out.gguf> <llmarch> [ggml_type] [group] [bits]
 * Chiama llmcodec_conv_mlx_to_gguf_arch_ex (rename canonico via .llmarch + stacking esperti).
 * Il tokenizer (tokenizer.ggml.*) viene dal PROGETTO tokenizer: unica sorgente, il codec
 * NON ri-parsa tokenizer.json. Slice opzionale dei layer via env LLMCODEC_GGUF_LAYERS=l0:l1. */
#include "llmcodec_gguf_mlx.h"
#include "llmtoken.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "uso: %s <mlx_dir> <out.gguf> <llmarch> [type=12] [group=32] [bits=4]\n", argv[0]); return 2; }
    int t = argc > 4 ? atoi(argv[4]) : 12;
    int g = argc > 5 ? atoi(argv[5]) : 32;
    int b = argc > 6 ? atoi(argv[6]) : 4;
    LlmCodecTokenizerMeta tm; memset(&tm, 0, sizeof tm);
    char tp[1300]; snprintf(tp, sizeof tp, "%s/tokenizer.json", argv[1]);
    llmtoken_handle th = llmtoken_load(tp);
    if (th) {
        size_t nt = 0, nm = 0;
        tm.tokens = llmtoken_export_tokens(th, &nt); tm.n_tokens = (int)nt;
        tm.merges = llmtoken_export_merges(th, &nm); tm.n_merges = (int)nm;
        tm.model  = llmtoken_gguf_model(th);
    }
    int rc = llmcodec_conv_mlx_to_gguf_arch_ex(argv[1], argv[2], argv[3], t, (unsigned)g, (unsigned)b, th ? &tm : NULL);
    if (th) llmtoken_free(th);   /* gli array tm sono "borrowed": vivono quanto th */
    fprintf(stderr, "[gguf_convert] rc=%d -> %s\n", rc, argv[2]);
    return rc == 0 ? 0 : 1;
}
