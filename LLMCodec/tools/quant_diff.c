/* quant_diff.c — verifica DIFFERENZIALE del dequant di LLMCodec contro implementazioni
 * indipendenti, sui tensori REALI dei GGUF di questa macchina. Nessun download, nessun
 * modello in memoria: si dequantizza una finestra per tipo e si confrontano i bit.
 *
 * Porta del gate di lane_gguf (piano LLMEnv §5). I testimoni:
 *   C) llama.cpp/ggml (tools/quant_diff_ggml.c) — AUTORITÀ: è il codice che i GGUF li SCRIVE,
 *      quindi fissa anche i NUMERI di enum e le dimensioni di blocco;
 *   D) gguflib (tools/quant_diff_gguflib.c) — il lettore che alcuni motori esterni usano
 *      DAVVERO per i quantizzati che non sanno impacchettare; copre 6 tipi
 *      (`Q4_0 Q4_1 Q8_0 Q2_K Q4_K Q6_K`) su 7 GGUF reali.
 * (Rispetto all'originale si omettono i due testimoni che vivevano in un ALTRO repo, per non
 *  far citare al codec sorgenti di consumatori. Restano l'autorità e il testimone esterno.)
 *
 * Build (i path dei testimoni sono fuori repo di proposito):
 *   U=<...>/gguflib-src           # gguflib.c + fp16.c + gguflib.h
 *   L=<...>/llamacpp              # ggml/include + build/bin/libggml-base.dylib
 *   cc -Wall -Wextra -Werror -std=gnu17 \
 *      -I include -I "$U" -I "$L/ggml/src" -I "$L/ggml/include" \
 *      tools/quant_diff.c tools/quant_diff_ggml.c tools/quant_diff_gguflib.c \
 *      "$U/gguflib.c" "$U/fp16.c" \
 *      -L "$L/build/bin" -lggml-base -Wl,-rpath,"$L/build/bin" \
 *      -L <build-llmcodec> -lllmcodec -lllmcodec_ggml -Wl,-rpath,<build-llmcodec> -o quant_diff
 * Uso: quant_diff <file.gguf> [...]   oppure   quant_diff --synthetic
 *      exit 0 = nessun ns. bit divergente, 1 = un ns. bit diverge, 3 = un testimone esterno
 *      diverge da ggml (bug da aprire dove quel codice vive), 2 = un file non si è aperto.
 */
#include "llmcodec_gguf_read.h"
#include "llmcodec_ggml_quants.h"
#include "llmcodec_fp16.h"

int ggc_block_vals(int type);
int ggc_dequant_row(int type, const uint8_t *bytes, int64_t n, float *dst);
uint64_t ggc_row_bytes(int type, int64_t n);
int ggc_type_count(void);
uint64_t ggc_row_bytes_any(int type, int64_t n);
uint32_t ggc_f16_bits(uint16_t h);

int ggu_covers(int type);
int ggu_dequant_row(int type, const uint8_t *bytes, int64_t n, float *dst);
uint64_t ggu_row_bytes(int type, int64_t n);
const char *ggu_type_name(int type);

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SAMPLE_ELEMS 8192u

static float mine[SAMPLE_ELEMS], refC[SAMPLE_ELEMS], refD[SAMPLE_ELEMS];

static int cmp_bufs(const float *x, const float *y, uint64_t n, const char *what,
                    const char *tensor, uint64_t *first_diff, float *maxdiff) {
    uint64_t i;
    for (i = 0; i < n; i++) {
        if (memcmp(&x[i], &y[i], sizeof(float)) != 0) {
            double a = (double)x[i], b = (double)y[i];
            double d = a > b ? a - b : b - a;
            if (*maxdiff < (float)d) *maxdiff = (float)d;
            if (!*first_diff) {
                *first_diff = i + 1;
                printf("    DIVERGENZA %s su \"%s\"[%llu]: %.9g vs %.9g\n", what, tensor,
                       (unsigned long long)i, a, b);
            }
        }
    }
    return memcmp(x, y, (size_t)n * sizeof(float)) == 0;
}

struct witness { const char *tag; int ok; float *buf; };
struct verdict { uint64_t ours; uint64_t ext; uint64_t calls; };

static void compare(int type, const uint8_t *bytes, uint64_t elems, const char *label,
                    int verbose, struct verdict *v) {
    struct witness w[2];
    int eq[2];
    int i, n_ok = 0;

    v->calls++;
    if (llmcodec_ggml_dequant_row(type, bytes, mine, (int64_t)elems) != 0) {
        printf("    %s tipo %d: IL NS. DEQUANT RIFIUTA (%" PRIu64 " elem)\n", label, type, elems);
        v->ours++;
        return;
    }
    w[0].tag = "C (ggml)";
    w[0].ok = ggc_dequant_row(type, bytes, (int64_t)elems, refC);
    w[0].buf = refC;
    w[1].tag = "D (gguflib)";
    w[1].ok = ggu_dequant_row(type, bytes, (int64_t)elems, refD);
    w[1].buf = refD;

    for (i = 0; i < 2; i++) n_ok += w[i].ok != 0;
    if (!n_ok) {
        printf("    %s tipo %d: nessun testimone lo copre (gap dichiarato, non confrontabile)\n",
               label, type);
        return;
    }
    for (i = 0; i < 2; i++) {
        uint64_t first = 0;
        float maxdiff = 0;
        eq[i] = -1;
        if (!w[i].ok) continue;
        eq[i] = cmp_bufs(mine, w[i].buf, elems, w[i].tag, label, &first, &maxdiff) ? 1 : 0;
        if (verbose || !eq[i]) {
            printf("    %s tipo %d vs %s: %s", label, type, w[i].tag, eq[i] ? "bit-exact" : "DIVERSO");
            if (!eq[i]) printf("  max|diff|=%.9g", (double)maxdiff);
            printf("\n");
        }
    }
    /* C (indice 0) è l'autorità. Se i ns. bit coincidono con C e un secondario dice altro,
     * l'outlier è LUI. Se C non copre, la divergenza resta nostra. */
    if (eq[1] == 0) { if (eq[0] == 1) v->ext++; else v->ours++; }
    if (eq[0] == 0) v->ours++;
}

static void one_tensor(const LlmCodecGgufFileInfo *f, const uint8_t *map, uint64_t idx,
                       int type, uint64_t elems, struct verdict *v) {
    char label[160];
    uint64_t need = 0;
    if (llmcodec_ggml_type_bytes(type, elems, &need) != 0) return;
    if (f->data_start + f->tensors[idx].offset + need > f->file_size) {
        printf("    (finestra da %" PRIu64 " byte non disponibile su \"%s\")\n", need,
               f->tensors[idx].name);
        return;
    }
    snprintf(label, sizeof label, "\"%s\"", f->tensors[idx].name);
    compare(type, map + f->data_start + f->tensors[idx].offset, elems, label, 1, v);
}

/* 0 = nulla di nostro; 2 = il file non si è aperto (un path sbagliato NON è un verdetto sui bit). */
static int one(const char *path, struct verdict *v) {
    LlmCodecGgufFileInfo f;
    int fd;
    struct stat st;
    void *map;
    uint64_t i, before = v->ours;
    int seen[64];

    memset(seen, 0, sizeof seen);
    if (llmcodec_gguf_read_info(path, &f) != 0) {
        printf("%s: testata rifiutata\n", path);
        return 2;
    }
    fd = open(path, O_RDONLY);
    if (fd < 0 || fstat(fd, &st) != 0) { if (fd >= 0) close(fd); llmcodec_gguf_free_info(&f); return 2; }
    map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { llmcodec_gguf_free_info(&f); return 2; }

    printf("%s: %" PRIu64 " tensori\n", path, f.n_tensors);
    for (i = 0; i < f.n_tensors; i++) {
        int type = f.tensors[i].ggml_type;
        int64_t bs;
        uint64_t elems, numel = 1;
        int d;
        if (type < 0 || type >= 64 || seen[type]) continue;
        bs = llmcodec_ggml_block_vals(type);
        if (bs <= 0) continue;
        elems = SAMPLE_ELEMS - (SAMPLE_ELEMS % (uint64_t)bs);
        if (elems == 0) continue;
        for (d = 0; d < f.tensors[i].ndim; d++) numel *= f.tensors[i].dims[d];
        if (numel < elems) continue;
        seen[type] = 1;
        one_tensor(&f, (const uint8_t *)map, i, type, elems, v);
    }
    munmap(map, (size_t)st.st_size);
    llmcodec_gguf_free_info(&f);
    return v->ours > before ? 1 : 0;
}

static void synthetic(struct verdict *v) {
    unsigned char bytes[SAMPLE_ELEMS * 4];
    uint64_t seed = 20260922u;
    size_t i;
    int t;
    printf("synthetic (blocchi pseudo-casuali, seed %" PRIu32 ")\n", (uint32_t)seed);
    for (i = 0; i < sizeof bytes; i++) {
        seed = seed * 1103515245u + 12345u;
        bytes[i] = (unsigned char)(seed >> 16);
    }
    for (t = 0; t <= 42; t++) {
        int64_t bs = llmcodec_ggml_block_vals(t);
        uint64_t elems;
        if (bs <= 0) continue;
        elems = SAMPLE_ELEMS - (SAMPLE_ELEMS % (uint64_t)bs);
        if (elems == 0) continue;
        if (llmcodec_ggml_dequant_row(t, bytes, mine, (int64_t)elems) != 0) continue;
        compare(t, bytes, elems, "synthetic", 1, v);
    }
}

/* Audit della FORMA: per OGNI valore di enum, gli elementi per blocco e i byte per riga del
 * codec devono coincidere con quelli di ggml. È l'asse che ha beccato i tipi spostati di uno. */
static int coverage_audit(struct verdict *v) {
    int t, bad = 0, forms = 0, gaps = 0;
    printf("audit di copertura (forma + byte) contro ggml\n");
    for (t = 0; t < ggc_type_count(); t++) {
        int64_t bv = llmcodec_ggml_block_vals(t);
        int cbv = ggc_block_vals(t);
        uint64_t n = 2048, need = 0, gg = 0;
        forms++;
        if (bv > 0 && cbv > 0 && (uint64_t)cbv != (uint64_t)bv) {
            printf("    tipo %d: elementi per blocco %lld, ggml dice %d\n", t, (long long)bv, cbv);
            bad++;
        }
        if (bv > 0) {
            n -= n % (uint64_t)bv;
            if (n) (void)llmcodec_ggml_type_bytes(t, n, &need);
            if (n && need) gg = ggc_row_bytes_any(t, (int64_t)n);
        }
        if (n && gg && need && gg != need) {
            printf("    tipo %d: %" PRIu64 " byte per %" PRIu64 " elementi, ggml ne dice %" PRIu64
                   " — i byte sbagliati spostano la data section di TUTTI i tensori\n", t, need, n, gg);
            bad++;
        }
    }
    for (t = 0; t <= 42; t++) {
        uint64_t n, need;
        int64_t bv = llmcodec_ggml_block_vals(t);
        static const unsigned char zeros[sizeof mine];
        if (bv <= 0) continue;
        n = 2048u - (2048u % (uint64_t)bv);
        if (n == 0 || llmcodec_ggml_type_bytes(t, n, &need) != 0 || need > sizeof(mine)) continue;
        if (llmcodec_ggml_dequant_row(t, zeros, mine, (int64_t)n) == 0) {
            int cok = ggc_dequant_row(t, zeros, (int64_t)n, refC);
            int dok = ggu_dequant_row(t, zeros, (int64_t)n, refD);
            if (!cok && !dok) {
                printf("    tipo %d: NOI asseriamo di coprirlo e NESSUN testimone lo fa\n", t);
                bad++;
            }
        } else {
            gaps++;
        }
    }
    printf("    %d forme confrontate con ggml, %d incoerenze, %d gap\n", forms, bad, gaps);
    v->ours += (uint64_t)bad;
    return bad ? 1 : 0;
}

/* Il convertitore f16->f32 è usato da OGNI dequant: si girano tutti i 65536 pattern e si
 * confrontano i BIT con ggml (un convertitore che diverge sugli NaN non si vedrebbe altrove). */
static void f16_exhaustive(struct verdict *v) {
    uint32_t h, bad = 0;
    int shown = 0;
    for (h = 0; h < 65536u; h++) {
        uint16_t s = (uint16_t)h;
        uint32_t mine_bits, gg_bits;
        float f = llmcodec_f16_to_f32(s);
        memcpy(&mine_bits, &f, 4);
        gg_bits = ggc_f16_bits(s);
        if (mine_bits != gg_bits) {
            bad++;
            if (shown < 6) {
                printf("    f16 0x%04x: nostri 0x%08x, ggml 0x%08x\n", s, mine_bits, gg_bits);
                shown++;
            }
        }
    }
    printf("f16->f32 esaustivo: 65536 pattern, %u bit-diversi%s\n", bad, bad ? " (primi 6)" : "");
    v->ours += bad;
}

int main(int argc, char **argv) {
    struct verdict v;
    int i, io = 0;
    memset(&v, 0, sizeof v);
    if (argc < 2) {
        fprintf(stderr, "uso: %s <--synthetic | <file.gguf> [...]>\n", argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "--synthetic")) {
        if (argc != 2) { fprintf(stderr, "--synthetic si usa da solo\n"); return 2; }
        argc = 1;
    }
    for (i = 1; i < argc; i++) if (one(argv[i], &v) == 2) io = 1;
    synthetic(&v);
    f16_exhaustive(&v);
    coverage_audit(&v);
    if (io) {
        printf("ESITO: almeno un file NON è stato aperto — nessun verdetto sui bit\n");
        return 2;
    }
    if (v.ours) {
        printf("ESITO: %" PRIu64 " DIVERGENZE NOSTRE su %" PRIu64 " confronti\n", v.ours, v.calls);
        return 1;
    }
    if (v.ext) {
        printf("ESITO: nessun ns. bit divergente su %" PRIu64 " confronti, ma %" PRIu64
               " disaccordi di un testimone esterno contro ggml (bug da aprire là). VERDE PER NOI\n",
               v.calls, v.ext);
        return 3;
    }
    printf("ESITO: nessun bit divergente (%" PRIu64 " confronti)\n", v.calls);
    return 0;
}
