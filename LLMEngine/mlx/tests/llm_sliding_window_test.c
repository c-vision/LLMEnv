/* llm_sliding_window_test.c — P0-quinquies: blocco che ATTRAVERSA la finestra == incrementale.
 * Uso: llm_sliding_window_test <pre> <Lq> <win> <D> <mode>   (mode 0=mlx_slice, 1=senza)
 * Nessun modello: millisecondi. 0 = coerente. */
#include "llmengine_mlx.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    if (argc < 6) { printf("uso: %s <pre> <Lq> <win> <D> <mode>\n", argv[0]); return 2; }
    char buf[192];
    int rc = llm_ai_mlx_sliding_selftest(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
                                         atoi(argv[4]), atoi(argv[5]), buf, (int)sizeof buf);
    printf("rc=%d %s\n", rc, buf);
    return rc == 0 ? 0 : 1;
}
