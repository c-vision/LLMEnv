/* ds4_shim_test.c — driver di prova del motore ds4 vendorizzato.
 * Uso: ds4_shim_test <model.gguf> [prompt] [max_tokens] [ctx] [stream 0|1] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_vendor_load(const char* path, int stream_experts, int ctx_size);
int ds4_vendor_generate(const char* prompt, int max_tokens, int ctx_size,
                        void (*cb)(void*, const char*, int), void* ud);
void ds4_vendor_close(void);

static void cb(void* ud, const char* text, int len) { (void)ud; fwrite(text, 1, (size_t)len, stdout); fflush(stdout); }

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    const char* prompt = argc > 2 ? argv[2] : "Ciao";
    int max = argc > 3 ? atoi(argv[3]) : 16;
    int ctx = argc > 4 ? atoi(argv[4]) : 4096;
    int stream = argc > 5 ? atoi(argv[5]) : 1;
    fprintf(stderr, "ds4_vendor: load %s (stream=%d ctx=%d)\n", path, stream, ctx);
    if (ds4_vendor_load(path, stream, ctx) != 0) { fprintf(stderr, "load FALLITA\n"); return 1; }
    fprintf(stderr, "ds4_vendor: generate (max=%d)\n", max);
    ds4_vendor_generate(prompt, max, ctx, cb, NULL);
    printf("\n");
    ds4_vendor_close();
    return 0;
}
