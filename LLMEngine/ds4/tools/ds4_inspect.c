/* Verifica che il motore ds4 (vendor patchato) accetti un GGUF: apre e stampa il vocabolario. */
#include "ds4_bridge.h"
#include <stdio.h>
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <model.gguf>\n", argv[0]); return 2; }
    ds4_bridge* b = ds4_bridge_open_ex(argv[1], 0, 0, 0, 1);
    if (!b) { printf("ds4: open FAIL\n"); return 1; }
    printf("ds4: open OK, vocab=%d\n", ds4_bridge_vocab(b));
    ds4_bridge_close(b);
    return 0;
}
