/* llm_encode_cost_test.c — P0-ter: costo BUILD vs ENCODING (submit) per nodo del grafo.
 * Uso: llm_encode_cost_test <n_ops> <reps>   (nessun modello: millisecondi).
 * Serve a decidere se la leva del decode CPU-bound e' FONDERE i dispatch. */
#include "llmengine_mlx.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    if (argc < 3) { printf("uso: %s <n_ops> <reps>\n", argv[0]); return 2; }
    char buf[1024];
    int rc = llm_ai_mlx_encode_cost_selftest(atoi(argv[1]), atoi(argv[2]), buf, (int)sizeof buf);
    printf("rc=%d %s\n", rc, buf);
    return rc == 0 ? 0 : 1;
}
