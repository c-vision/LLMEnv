/* dsmin.c — ds-MINIMAL: driver ENGINE-ONLY (niente fastllm, niente CLI, niente ds-server).
 * Usa SOLO le stesse funzioni che ds-slow espone via la sua ABI: load / session / append / generate /
 * argmax. Serve a misurare il MOTORE PURO (vendored) e confrontarlo col ds-slow.
 *
 * Uso: dsmin <model.gguf> <max_new> <id,id,...>   (gli id del prompt, uguali per entrambi i motori)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ds4.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "uso: %s <model.gguf> <max_new> <id,id,...>\n", argv[0]); return 2; }
    const char* model = argv[1];
    int maxn = atoi(argv[2]);

    ds4_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = model;
    opt.ssd_streaming = true;                 /* come fastllm --offload-experts */

    ds4_engine* e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine_open fallito\n"); return 1; }
    ds4_session* s = NULL;
    if (ds4_session_create(&s, e, 4096) != 0) { fprintf(stderr, "session_create fallita\n"); return 1; }

    ds4_tokens t; memset(&t, 0, sizeof t);
    { char* buf = strdup(argv[3]); char* p = strtok(buf, ",");
      while (p) { ds4_tokens_push(&t, atoi(p)); p = strtok(NULL, ","); } free(buf); }
    char err[256]; err[0] = 0;
    if (ds4_session_sync(s, &t, err, sizeof err) != 0) { fprintf(stderr, "session_sync fallita: %s\n", err); return 1; }

    int n = 0; double a = now();
    for (int i = 0; i < maxn; i++) {
        int id = ds4_session_argmax(s);
        if (id < 0) break;
        ds4_tokens_push(&t, id);
        if (ds4_session_sync(s, &t, err, sizeof err) != 0) break;
        n++;
    }
    double b = now(); double ms = (b - a) * 1e3;
    printf("[dsmin-vendored] %d tok, %.1f ms/tok (%.2f tok/s)\n", n, n ? ms / n : 0, ms ? (double)n * 1000 / ms : 0);

    ds4_session_free(s);
    ds4_engine_close(e);
    return 0;
}
