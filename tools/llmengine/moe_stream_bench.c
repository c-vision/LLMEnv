/* tools/llmengine/moe_stream_bench.c — P0 "truth harness" dello streaming esperti MoE.
 *
 * Scopo (piano Piani/(open) - Piano-Metal-Streaming-Esperti-Banda.md): misurare in modo
 * DETERMINISTICO, senza il motore, come si leggono i byte degli esperti MoE da disco:
 *   - `mmap` + page-fault (come fa oggi `llmengine_moe_safetensors_ingest.h`),
 *   - `pread` posizionale (via proposta dal piano),
 *   - `read` sequenziale (baseline),
 * con variante WARM (page cache) e F_NOCACHE (cold-ish, bypass page cache).
 *
 * Individua da solo i tensori esperti (`...switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`)
 * leggendo gli header safetensors (8 byte di lunghezza + JSON), e stima i byte/token da
 * `config.json` (num_experts_per_tok × num_hidden_layers × bundle esperto).
 *
 * Nessuna dipendenza: solo POSIX + libc. Non e' codice di motore, e' un misuratore.
 *
 * Uso:
 *   cc -O2 tools/llmengine/moe_stream_bench.c -o tools/llmengine/moe_stream_bench
 *   tools/llmengine/moe_stream_bench <dir_modello> [--max-mb N] [--iters K] [--chunk-mb M]
 *
 * Esce 0 se ha misurato almeno una metrica; 1 su errore d'uso/IO.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

#define MAX_TENSORS 4096
#define CHUNK_DEFAULT (64u << 20)   /* 64 MiB */
#define MAXBYTES_DEFAULT (4ull << 30) /* 4 GiB per passata (runtime bounded) */

typedef struct { uint64_t off, len; } Range;

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* --- minimal JSON scan (header safetensors) ---------------------------------
 * L'header e' `{"nome":{"dtype":"..","shape":[..],"data_offsets":[a,b]}, ...}`.
 * Si scandiscono le chiavi a livello 1; per ogni oggetto-valore si estraggono
 * `shape[0]` e `data_offsets`. Sufficiente per il formato safetensors (regolare). */
static const char* skip_ws(const char* p) { while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') ++p; return p; }

static int parse_u64_array(const char* obj, const char* key, uint64_t* out, int max, int* n) {
    const char* p = strstr(obj, key);
    if (!p) return -1;
    p = strchr(p, '['); if (!p) return -1;
    ++p; *n = 0;
    while (*p && *p != ']' && *n < max) {
        p = skip_ws(p);
        char* e = NULL;
        unsigned long long v = strtoull(p, &e, 10);
        if (e == p) break;
        out[(*n)++] = (uint64_t)v; p = e;
        p = skip_ws(p);
        if (*p == ',') ++p;
    }
    return *n > 0 ? 0 : -1;
}

/* Scandisce l'header e invoca on_tensor per ogni tensore (nome, offset, lunghezza). */
static int scan_header(const uint8_t* hdr, size_t hn,
                       void (*on_tensor)(const char* name, uint64_t off, uint64_t len, void* ud),
                       void* ud) {
    const char* j = (const char*)hdr;
    const char* end = j + hn;
    const char* p = j;
    int depth = 0;
    while (p < end) {
        char c = *p;
        if (c == '"') {
            const char* s = ++p;
            while (p < end && *p != '"') ++p;
            size_t n = (size_t)(p - s);
            const char* q = skip_ws(p + 1);
            if (depth == 1 && q < end && *q == ':') {
                const char* v = skip_ws(q + 1);
                if (*v == '{') {
                    /* oggetto del tensore: estrai shape + data_offsets */
                    const char* obj = v;
                    uint64_t shp[8]; int ns = 0;
                    uint64_t doff[2]; int nd = 0;
                    parse_u64_array(obj, "\"shape\"", shp, 8, &ns);
                    parse_u64_array(obj, "\"data_offsets\"", doff, 2, &nd);
                    if (nd == 2) {
                        char name[512];
                        size_t cp = n < sizeof name - 1 ? n : sizeof name - 1;
                        memcpy(name, s, cp); name[cp] = 0;
                        on_tensor(name, doff[0], doff[1] - doff[0], ud);
                    }
                }
            }
            if (p < end) ++p;
        } else if (c == '{') { depth++; ++p; }
        else if (c == '}') { depth--; ++p; }
        else ++p;
        if (depth < 0) break;
    }
    return 0;
}

typedef struct {
    Range r[MAX_TENSORS]; int n;
    uint64_t total;
} Collect;

static int is_expert(const char* name) {
    return strstr(name, "switch_mlp") != NULL && strstr(name, "_proj.") != NULL;
}

static void on_tensor(const char* name, uint64_t off, uint64_t len, void* ud) {
    Collect* c = (Collect*)ud;
    if (!is_expert(name)) return;
    if (c->n < MAX_TENSORS) { c->r[c->n].off = off; c->r[c->n].len = len; c->n++; }
    c->total += len;
}

/* --- lettura dell'header di un singolo .safetensors --- */
static int load_file_ranges(const char* path, Collect* c) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    uint64_t hn = 0;
    if (read(fd, &hn, 8) != 8 || hn == 0 || hn > (64u<<20)) { close(fd); return -1; }
    uint8_t* hdr = malloc(hn + 1);
    if (!hdr) { close(fd); return -1; }
    if (read(fd, hdr, hn) != (ssize_t)hn) { free(hdr); close(fd); return -1; }
    hdr[hn] = 0;
    close(fd);
    scan_header(hdr, hn, on_tensor, c);
    free(hdr);
    return 0;
}

static int cmp_range(const void* a, const void* b) {
    const Range* x = a; const Range* y = b;
    return x->off < y->off ? -1 : (x->off > y->off ? 1 : 0);
}

/* --- bench ---------------------------------------------------------------- */
static double bench_mmap(const char* path, const Collect* c, uint64_t maxbytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return -1; }
    uint8_t* base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return -1; }
    volatile uint64_t sink = 0;
    uint64_t done = 0;
    double t0 = now_s();
    for (int i = 0; i < c->n && done < maxbytes; ++i) {
        const uint8_t* p = base + c->r[i].off;
        uint64_t len = c->r[i].len;
        for (uint64_t k = 0; k < len; k += 4096) sink += p[k];
        done += len;
    }
    double dt = now_s() - t0;
    munmap(base, (size_t)st.st_size);
    close(fd);
    (void)sink;
    return dt > 0 ? (double)done / dt / 1e9 : -1;
}

static double bench_read(const char* path, const Collect* c, uint64_t maxbytes, int nocache) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    if (nocache) fcntl(fd, F_NOCACHE, 1);
    size_t bufsz = 64u<<20;
    uint8_t* buf = malloc(bufsz);
    if (!buf) { close(fd); return -1; }
    uint64_t done = 0;
    double t0 = now_s();
    for (int i = 0; i < c->n && done < maxbytes; ++i) {
        uint64_t off = c->r[i].off, left = c->r[i].len;
        while (left > 0) {
            size_t want = left < bufsz ? (size_t)left : bufsz;
            ssize_t r = pread(fd, buf, want, (off_t)(off + (c->r[i].len - left)));
            if (r <= 0) { left = 0; break; }
            left -= (uint64_t)r; done += (uint64_t)r;
        }
    }
    double dt = now_s() - t0;
    free(buf); close(fd);
    return dt > 0 ? (double)done / dt / 1e9 : -1;
}

/* legge config.json (best-effort) per top_k / layers / experts.
 * Un config HF puo' contenere piu' occorrenze di una chiave (es. vision vs text_config):
 * si prende il MASSIMO, che nel caso MoE e' il valore del modello di testo. */
static long max_int_for_keys(const char* j, const char** keys) {
    long best = 0;
    for (int i = 0; keys[i]; i++) {
        const char* q = j;
        while ((q = strstr(q, keys[i])) != NULL) {
            const char* c = strchr(q, ':');
            if (c) { long v = strtol(c + 1, NULL, 10); if (v > best) best = v; }
            q += strlen(keys[i]);
        }
    }
    return best;
}

static void read_config(const char* dir, long* topk, long* layers, long* experts) {
    char p[1400]; snprintf(p, sizeof p, "%s/config.json", dir);
    FILE* f = fopen(p, "rb"); if (!f) return;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > (16<<20)) { fclose(f); return; }
    char* j = malloc((size_t)n + 1); if (!j) { fclose(f); return; }
    if (fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return; }
    fclose(f); j[n] = 0;
    const char* kt[] = {"\"num_experts_per_tok\"", "\"num_experts_per_token\"", NULL};
    const char* kl[] = {"\"num_hidden_layers\"", NULL};
    const char* ke[] = {"\"num_experts\"", "\"num_local_experts\"", NULL};
    *topk = max_int_for_keys(j, kt);
    *layers = max_int_for_keys(j, kl);
    *experts = max_int_for_keys(j, ke);
    free(j);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <dir_modello> [--max-mb N] [--iters K]\n", argv[0]); return 1; }
    const char* dir = argv[1];
    uint64_t maxbytes = MAXBYTES_DEFAULT; int iters = 1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--max-mb") && i+1 < argc) maxbytes = strtoull(argv[++i], NULL, 10) << 20;
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) iters = atoi(argv[++i]);
    }

    /* elenco .safetensors */
    char files[256][1400]; int nf = 0;
    DIR* d = opendir(dir);
    if (!d) { fprintf(stderr, "opendir %s: %s\n", dir, strerror(errno)); return 1; }
    struct dirent* e;
    while ((e = readdir(d)) && nf < 256) {
        if (e->d_name[0] == '.') continue;
        size_t ln = strlen(e->d_name);
        if (ln > 12 && !strcmp(e->d_name + ln - 12, ".safetensors")) {
            snprintf(files[nf], sizeof files[nf], "%s/%s", dir, e->d_name); nf++;
        }
    }
    closedir(d);
    if (nf == 0) { fprintf(stderr, "nessun .safetensors in %s\n", dir); return 1; }

    Collect c; memset(&c, 0, sizeof c);
    for (int i = 0; i < nf; i++) load_file_ranges(files[i], &c);
    if (c.n == 0) { fprintf(stderr, "nessun tensore esperto (switch_mlp.*_proj.*) trovato\n"); return 1; }
    qsort(c.r, c.n, sizeof(Range), cmp_range);

    long topk = 0, layers = 0, experts = 0;
    read_config(dir, &topk, &layers, &experts);

    printf("== moe_stream_bench ==\n");
    printf("model        : %s\n", dir);
    printf("shards       : %d\n", nf);
    printf("expert tens. : %d\n", c.n);
    printf("expert bytes : %" PRIu64 " (%.2f GiB)\n", c.total, (double)c.total/1073741824.0);
    printf("config       : layers=%ld experts=%ld top_k=%ld\n", layers, experts, topk);
    if (topk > 0 && layers > 0 && experts > 0) {
        double per_expert_layer = (double)c.total / ((double)experts * (double)layers);
        double bpt = per_expert_layer * (double)topk * (double)layers; /* = top_k * total/experts */
        printf("per-expert/l : %.3f MiB (bundle w+s+b di 1 esperto in 1 layer)\n", per_expert_layer/1048576.0);
        printf("bytes/token  : %.1f MiB  (top_k * layers * per-expert/l)\n", bpt/1048576.0);
        printf("             : ipotesi 80%% hit -> %.1f MiB letti/token\n", bpt*0.2/1048576.0);
    }
    printf("\n%-18s %10s %8s\n", "metodo", "GB/s", "secs");

    /* primo file per il bench (la maggior parte degli esperti e' nei primi shard) */
    const char* f0 = files[0];
    for (int it = 0; it < iters; it++) {
        double s;
        s = bench_mmap(f0, &c, maxbytes);        if (s >= 0) printf("%-18s %10.2f (mmap warm)\n", "mmap", s); else printf("%-18s %10s\n", "mmap", "n/a");
        s = bench_read(f0, &c, maxbytes, 0);     if (s >= 0) printf("%-18s %10.2f (pread warm)\n", "pread", s);
        s = bench_read(f0, &c, maxbytes, 1);     if (s >= 0) printf("%-18s %10.2f (pread F_NOCACHE)\n", "pread-nocache", s);
    }
    return 0;
}
