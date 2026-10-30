/* llmdesc_gen — crea i descrittori `.gmbarch` mancanti per una o piu' cartelle modello HF.
 * Uso: llmdesc_gen [--abi DIR] <cartella_modello>...   (senza --abi: la cartella ABI di LLMEnv, vedi llmdesc_abi_dir)
 * Per partire da un URL Hugging Face usare `fastllm gmbarch <url|org/nome>` (scarica solo config e indice dei tensori). */
#include "llmdesc.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* abi = NULL;
    int rc = 0, n = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--abi") && i + 1 < argc) { abi = argv[++i]; continue; }
        char path[4096] = "", err[256] = "";
        int r = llmdesc_ensure_for_dir(argv[i], abi, path, sizeof path, err, sizeof err);
        n++;
        if (r < 0) { printf("%s: %s\n", argv[i], err); rc = 1; }
        else printf("%s: %s %s\n", argv[i], r ? "creato" : "gia' presente", path);
    }
    if (!n) { fprintf(stderr, "uso: %s [--abi DIR] <cartella_modello>...\n", argv[0]); return 2; }
    return rc;
}
