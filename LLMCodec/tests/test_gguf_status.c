/* test_gguf_status.c — TASSONOMIA D'ERRORE del parser GGUF (asset assorbito da lane_gguf).
 *
 * Porta del test delle corruzioni di FastLLM (`tests/gguf/test_gguf.c`): i file MALATI si
 * costruiscono byte per byte, e si asserisce che ognuno abbia il SUO codice — mai lo stesso
 * `-1` per "troncato", "tipo ignoto", "sovrapposto", "versione non supportata".
 */
#include "llmcodec_gguf_read.h"
#include "llmcodec_gguf_status.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails, checks;
#define CHECK(cond, ...) do {                                     \
    checks++;                                                     \
    if (!(cond)) { fails++;                                       \
        printf("  FAIL %s:%d  ", __FILE__, __LINE__);             \
        printf(__VA_ARGS__); printf("\n");                        \
    }                                                             \
} while (0)

typedef struct { unsigned char *b; size_t n, cap; } Buf;
static void buf_init(Buf *x) { x->b = NULL; x->n = x->cap = 0; }
static void buf_free(Buf *x) { free(x->b); buf_init(x); }
static void buf_put(Buf *x, const void *p, size_t n) {
    if (x->n + n > x->cap) {
        size_t nc = x->cap ? x->cap * 2 : 256;
        while (nc < x->n + n) nc *= 2;
        x->b = (unsigned char *)realloc(x->b, nc);
        if (!x->b) { fprintf(stderr, "OOM\n"); exit(2); }
        x->cap = nc;
    }
    memcpy(x->b + x->n, p, n); x->n += n;
}
static void buf_u(Buf *x, uint64_t v, int bytes) {
    unsigned char t[8]; int i;
    for (i = 0; i < bytes; i++) t[i] = (unsigned char)(v >> (8 * i));
    buf_put(x, t, (size_t)bytes);
}
static void buf_str(Buf *x, const char *s) { size_t n = strlen(s); buf_u(x, n, 8); buf_put(x, s, n); }
static void buf_tensor(Buf *x, const char *name, uint64_t dim0, uint32_t type, uint64_t offset) {
    buf_str(x, name); buf_u(x, 2, 4); buf_u(x, dim0, 8); buf_u(x, 1, 8); buf_u(x, type, 4); buf_u(x, offset, 8);
}
static void pad_and_data(Buf *x, uint64_t align, uint64_t n) {
    while (x->n % align) buf_u(x, 0, 1);
    for (uint64_t i = 0; i < n; i++) buf_u(x, 0, 1);
}

static void expect(const unsigned char *b, size_t n, LlmCodecGgufStatus want, const char *what) {
    LlmCodecGgufFileInfo info;
    LlmCodecGgufStatus got = llmcodec_gguf_parse_mem_status(b, n, NULL, NULL, &info);
    CHECK(got == want, "%s: atteso %d (%s), got %d (%s)", what, (int)want,
          llmcodec_gguf_status_str(want), (int)got, llmcodec_gguf_status_str(got));
    llmcodec_gguf_free_info(&info);
}

static void test_codes(void) {
    Buf x;

    buf_init(&x); buf_u(&x, 0x11223344u, 4); buf_u(&x, 3, 4); buf_u(&x, 0, 8); buf_u(&x, 0, 8);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_NOT_GGUF, "magic diversa"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 1, 4); buf_u(&x, 0, 8); buf_u(&x, 0, 8);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_VERSION, "versione 1"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 1, 8); buf_u(&x, 0, 8);
    buf_str(&x, "w"); buf_u(&x, 2, 4); buf_u(&x, 32, 8);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_TRUNCATED, "tensor-info troncato"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 1, 8); buf_u(&x, 0, 8);
    buf_str(&x, "w"); buf_u(&x, 9, 4);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_NDIMS, "n_dims = 9"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 1, 8); buf_u(&x, 0, 8);
    buf_str(&x, "w"); buf_u(&x, 2, 4); buf_u(&x, 0, 8); buf_u(&x, 1, 8);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_NEG_DIM, "dim[0] = 0"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 1, 8); buf_u(&x, 0, 8);
    buf_tensor(&x, "w", 256, 999, 0);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_TYPE_UNKNOWN, "ggml_type 999"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 1, 8); buf_u(&x, 0, 8);
    buf_tensor(&x, "w", 256, 4, 0);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_TYPE_REMOVED, "Q4_2 ritirato"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 0, 8); buf_u(&x, 1, 8);
    buf_str(&x, "general.alignment"); buf_u(&x, 4, 4); buf_u(&x, 3, 4);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_BAD_ALIGN, "alignment = 3"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 2, 8); buf_u(&x, 0, 8);
    buf_tensor(&x, "a", 32, 0, 0); buf_tensor(&x, "b", 32, 0, 64);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_OVERLAP, "tensori sovrapposti"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 1, 8); buf_u(&x, 0, 8);
    buf_tensor(&x, "a", 32, 0, 1000000);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_DATA_SHORT, "offset oltre la fine"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 2, 8); buf_u(&x, 0, 8);
    buf_tensor(&x, "dup", 32, 0, 0); buf_tensor(&x, "dup", 32, 0, 128);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_NAME_DUP, "nome duplicato"); buf_free(&x);

    buf_init(&x); buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 4000000000ull, 8); buf_u(&x, 0, 8);
    expect(x.b, x.n, LLMCODEC_GGUF_STATUS_COUNT, "n_tensors impossibile"); buf_free(&x);

    printf("  ok   12 corruzioni diverse => 12 codici diversi\n");
}

static void test_valid_ok(void) {
    Buf x;
    LlmCodecGgufFileInfo info;
    buf_init(&x);
    buf_u(&x, 0x46554747u, 4); buf_u(&x, 3, 4); buf_u(&x, 1, 8); buf_u(&x, 0, 8);
    buf_tensor(&x, "w", 32, 0, 0);       /* F32 32 = 128 byte */
    pad_and_data(&x, 32, 128);
    LlmCodecGgufStatus st = llmcodec_gguf_parse_mem_status(x.b, x.n, NULL, NULL, &info);
    CHECK(st == LLMCODEC_GGUF_STATUS_OK, "file valido rifiutato: %s", llmcodec_gguf_status_str(st));
    CHECK(info.version == 3 && info.n_tensors == 1 && info.n_kv == 0, "conteggi %u/%llu/%llu",
          info.version, (unsigned long long)info.n_tensors, (unsigned long long)info.n_kv);
    CHECK(info.alignment == 32, "alignment %llu", (unsigned long long)info.alignment);
    CHECK(info.tensors && strcmp(info.tensors[0].name, "w") == 0, "nome tensore");
    llmcodec_gguf_free_info(&info);
    buf_free(&x);
    printf("  ok   file valido => OK\n");
}

static void test_status_str(void) {
    int i;
    for (i = 0; i <= 25; i++) CHECK(llmcodec_gguf_status_str((LlmCodecGgufStatus)i) != NULL,
                                    "status_str(%d) NULL", i);
    CHECK(strcmp(llmcodec_gguf_status_str(LLMCODEC_GGUF_STATUS_OK), "ok") == 0, "OK != ok");
    printf("  ok   status_str non-NULL per ogni codice\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("llmcodec/gguf-status:\n");
    test_status_str();
    test_codes();
    test_valid_ok();
    printf("\n%d verifiche, %d fallite\n", checks, fails);
    return fails ? 1 : 0;
}
