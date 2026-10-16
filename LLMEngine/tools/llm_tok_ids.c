/* llm_tok_ids.c — tokenizza TESTO NATURALE in id di token con LLMToken (BPE byte-level nativo,
 * nessun Python), per misurare la speculazione su prompt REALI invece che sintetici.
 *
 * Perche' esiste: decode_profile genera un prompt SINTETICO (id 100,101,...) che un drafter
 * addestrato su testo non puo' predire -> l'acceptance risulta bassa per un ARTEFATTO del prompt
 * (caso reale: "16%" su DFlash2 Qwen3.8-27B mentre su contesto naturale era 93-100%). Questo
 * harness produce gli id VERI di un testo, da passare a dflash_parity (che prende una lista di id).
 *
 * Uso:
 *   llm_tok_ids <tokenizer.json | model_dir> "testo da tokenizzare"      # stampa: id,id,id,...
 *   llm_tok_ids <...> --decode 1,2,3                                      # inverso (id -> testo), verifica
 *   echo "testo" | llm_tok_ids <...> -                                    # testo da stdin
 *
 * Build (linkando libllmtoken):
 *   cc -O2 -I LLMToken -o /tmp/llm_tok_ids LLMEngine/tools/llm_tok_ids.c \
 *      -L build/LLMToken -lllmtoken -Wl,-rpath,$(pwd)/build/LLMToken
 */
#include "llmtoken.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char* read_all(FILE* f) {
    size_t cap = 4096, n = 0;
    char* b = (char*)malloc(cap);
    if (!b) return NULL;
    size_t r;
    while ((r = fread(b + n, 1, cap - n - 1, f)) > 0) {
        n += r;
        if (n + 1 >= cap) { cap *= 2; char* nb = (char*)realloc(b, cap); if (!nb) { free(b); return NULL; } b = nb; }
    }
    b[n] = '\0';
    return b;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "uso: %s <tokenizer.json|model_dir> \"testo\" | %s <...> --decode id,id,...\n", argv[0], argv[0]);
        return 2;
    }
    /* tokenizer.json: se e' una directory usa <dir>/tokenizer.json, altrimenti il file dato */
    char tpath[4096];
    struct stat st;
    if (stat(argv[1], &st) == 0 && S_ISDIR(st.st_mode))
        snprintf(tpath, sizeof tpath, "%s/tokenizer.json", argv[1]);
    else
        snprintf(tpath, sizeof tpath, "%s", argv[1]);
    llmtoken_handle tk = llmtoken_load(tpath);
    if (!tk) { fprintf(stderr, "llmtoken_load FALLITO: %s\n", tpath); return 3; }

    if (strcmp(argv[2], "--decode") == 0 && argc >= 4) {
        int cap = 0; for (const char* p = argv[3]; *p; p++) if (*p == ',') cap++;
        int32_t* ids = (int32_t*)malloc((size_t)(cap + 2) * sizeof(int32_t));
        int n = 0; char* e = NULL; const char* p = argv[3];
        while (*p && n < cap + 2) { long v = strtol(p, &e, 10); if (e == p) break; ids[n++] = (int32_t)v; p = e; if (*p == ',') p++; }
        int need = llmtoken_decode(tk, ids, n, NULL, 0);
        if (need < 0) { fprintf(stderr, "decode errore\n"); free(ids); llmtoken_free(tk); return 4; }
        char* out = (char*)malloc((size_t)need + 1);
        llmtoken_decode(tk, ids, n, out, need + 1);
        printf("%s\n", out);
        free(out); free(ids); llmtoken_free(tk);
        return 0;
    }

    char* text = NULL;
    if (strcmp(argv[2], "-") == 0) text = read_all(stdin);
    else { text = strdup(argv[2]); }
    if (!text) { fprintf(stderr, "testo mancante\n"); llmtoken_free(tk); return 5; }

    int need = llmtoken_encode(tk, text, NULL, 0);
    if (need < 0) { fprintf(stderr, "encode errore\n"); free(text); llmtoken_free(tk); return 6; }
    int32_t* ids = (int32_t*)malloc((size_t)need * sizeof(int32_t));
    int n = llmtoken_encode(tk, text, ids, need);
    for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? "," : "");
    printf("\n");

    free(ids); free(text); llmtoken_free(tk);
    return 0;
}
