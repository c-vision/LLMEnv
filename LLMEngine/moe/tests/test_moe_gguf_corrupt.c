/* test_moe_gguf_corrupt.c — K7: il loader GGUF del runtime MoE rifiuta ESPLICITAMENTE i file
 * corrotti (mai output silenzioso). La tassonomia fine dei codici e' gia' coperta dal CODEC
 * (`LLMCodec/tests/test_gguf_status.c`); qui si verifica la CATENA del loader usato dal motore:
 * buffer corrotto → status del codec atteso → `llmengine_moe_gguf_read` = ERR_FORMAT.
 * Offline, deterministico, senza modello. */
#include "llmengine_moe_gguf.h"
#include "llmcodec_gguf_read.h"
#include "llmcodec_gguf_status.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(cond, ...) do {                                         \
    if (!(cond)) { fails++;                                           \
        printf("  FAIL %s:%d  ", __FILE__, __LINE__);                 \
        printf(__VA_ARGS__); printf("\n");                            \
    }                                                                 \
} while (0)

typedef struct { unsigned char* b; size_t n, cap; } Buf;
static void buf_init(Buf* x) { x->b = NULL; x->n = x->cap = 0; }
static void buf_free(Buf* x) { free(x->b); buf_init(x); }
static void buf_put(Buf* x, const void* p, size_t n) {
    if (x->n + n > x->cap) {
        size_t nc = x->cap ? x->cap * 2 : 256;
        while (nc < x->n + n) nc *= 2;
        x->b = (unsigned char*)realloc(x->b, nc);
        if (!x->b) { fprintf(stderr, "OOM\n"); exit(2); }
        x->cap = nc;
    }
    memcpy(x->b + x->n, p, n); x->n += n;
}
static void buf_u(Buf* x, uint64_t v, int bytes) {
    unsigned char t[8]; int i;
    for (i = 0; i < bytes; i++) t[i] = (unsigned char)(v >> (8 * i));
    buf_put(x, t, (size_t)bytes);
}
static void buf_str(Buf* x, const char* s) { size_t n = strlen(s); buf_u(x, n, 8); buf_put(x, s, n); }
static void buf_tensor(Buf* x, const char* name, uint64_t dim0, uint32_t type, uint64_t offset) {
    buf_str(x, name); buf_u(x, 1, 4); buf_u(x, dim0, 8); buf_u(x, type, 4); buf_u(x, offset, 8);
}
static void pad_data(Buf* x, uint64_t align, uint64_t n) {
    while (x->n % align) buf_u(x, 0, 1);
    for (uint64_t i = 0; i < n; i++) buf_u(x, 0, 1);
}
static void hdr(Buf* x, uint64_t n_tensors) {
    buf_u(x, 0x46554747u, 4); buf_u(x, 3, 4); buf_u(x, n_tensors, 8); buf_u(x, 0, 8);
}

/* Scrive il buffer in un file temp e chiama il loader del motore: deve fallire con ERR_FORMAT. */
static void expect_reject(const Buf* x, LlmCodecGgufStatus want, const char* what) {
    LlmCodecGgufFileInfo info;
    LlmCodecGgufStatus got = llmcodec_gguf_parse_mem_status(x->b, x->n, NULL, NULL, &info);
    CHECK(got == want, "%s: codec atteso %d (%s), got %d (%s)", what, (int)want,
          llmcodec_gguf_status_str(want), (int)got, llmcodec_gguf_status_str(got));
    llmcodec_gguf_free_info(&info);

    char path[] = "/tmp/moe-gguf-corrupt-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "%s: mkstemp", what);
    if (fd >= 0) {
        ssize_t w = write(fd, x->b, x->n);
        close(fd);
        CHECK(w == (ssize_t)x->n, "%s: write", what);
        LlmEngineMoeGguf g;
        LlmEngineMoeStatus ls = llmengine_moe_gguf_read(path, &g);
        CHECK(ls == LLMENGINE_MOE_ERR_FORMAT, "%s: loader atteso ERR_FORMAT, got %d", what, (int)ls);
        if (ls == LLMENGINE_MOE_OK) llmengine_moe_gguf_free(&g);
        unlink(path);
    }
}

int main(void) {
    /* 1. controllo: un GGUF minimo VALIDO viene accettato (il test non e' "tutto rosso"). */
    {
        Buf x; buf_init(&x);
        hdr(&x, 1);
        buf_tensor(&x, "w", 4, 0 /*F32*/, 0);
        pad_data(&x, 32, 16);
        LlmCodecGgufFileInfo info;
        CHECK(llmcodec_gguf_parse_mem_status(x.b, x.n, NULL, NULL, &info) == LLMCODEC_GGUF_STATUS_OK,
              "valid: il codec deve accettare");
        llmcodec_gguf_free_info(&info);
        char path[] = "/tmp/moe-gguf-valid-XXXXXX";
        int fd = mkstemp(path);
        if (fd >= 0) {
            ssize_t w = write(fd, x.b, x.n); close(fd);
            CHECK(w == (ssize_t)x.n, "valid: write");
            LlmEngineMoeGguf g;
            CHECK(llmengine_moe_gguf_read(path, &g) == LLMENGINE_MOE_OK, "valid: il loader deve accettare");
            llmengine_moe_gguf_free(&g);
            unlink(path);
        }
        buf_free(&x);
    }

    /* 2. troncato: header senza tensor-info. */
    {
        Buf x; buf_init(&x); hdr(&x, 1);
        expect_reject(&x, LLMCODEC_GGUF_STATUS_TRUNCATED, "troncato");
        buf_free(&x);
    }
    /* 3. n_tensors impossibile per la size. */
    {
        Buf x; buf_init(&x); hdr(&x, 1000000000ull);
        expect_reject(&x, LLMCODEC_GGUF_STATUS_COUNT, "count");
        buf_free(&x);
    }
    /* 4. dati dichiarati oltre la fine del file. */
    {
        Buf x; buf_init(&x);
        hdr(&x, 1);
        buf_tensor(&x, "w", 4, 0, 1ull << 40);
        pad_data(&x, 32, 16);
        expect_reject(&x, LLMCODEC_GGUF_STATUS_DATA_SHORT, "data short");
        buf_free(&x);
    }
    /* 5. ggml_type fuori tabella. */
    {
        Buf x; buf_init(&x);
        hdr(&x, 1);
        buf_tensor(&x, "w", 4, 99, 0);
        pad_data(&x, 32, 16);
        expect_reject(&x, LLMCODEC_GGUF_STATUS_TYPE_UNKNOWN, "tipo ignoto");
        buf_free(&x);
    }
    /* 6. nome duplicato. */
    {
        Buf x; buf_init(&x);
        hdr(&x, 2);
        buf_tensor(&x, "w", 4, 0, 0);
        buf_tensor(&x, "w", 4, 0, 16);
        pad_data(&x, 32, 32);
        expect_reject(&x, LLMCODEC_GGUF_STATUS_NAME_DUP, "nome duplicato");
        buf_free(&x);
    }
    /* 7. tensori sovrapposti. */
    {
        Buf x; buf_init(&x);
        hdr(&x, 2);
        buf_tensor(&x, "a", 4, 0, 0);
        buf_tensor(&x, "b", 4, 0, 8);   /* 8 < 16 = fine di "a" */
        pad_data(&x, 32, 32);
        expect_reject(&x, LLMCODEC_GGUF_STATUS_OVERLAP, "overlap");
        buf_free(&x);
    }

    if (fails == 0) { printf("test_moe_gguf_corrupt: OK\n"); return 0; }
    printf("test_moe_gguf_corrupt: %d FAIL\n", fails);
    return 1;
}
