/* moe_iobw.c — K8: bench LOAD-FREE dello storage per il path esperti MoE.
 *
 * Misura il throughput di lettura sequenziale (pread) di un file con due modalità:
 *   - buffered (page cache del SO);
 *   - cache-bypass: macOS `F_NOCACHE` / Linux `O_DIRECT` (buffer allineato).
 * Serve a rispondere alla domanda "il nostro reader e' al massimo del disco?" (Kimi:
 * O_DIRECT 3.2 GB/s > buffered 2.3 GB/s; su macOS l'analogo e' F_NOCACHE) SENZA lanciare
 * un modello: solo byte letti e tempo. Metodo: min su `--reps` (macchina quieta), chunk
 * fisso, nessuna allocazione per iterazione oltre al buffer.
 *
 * Uso:
 *   moe_iobw <file> [--mb N] [--reps K] [--chunk-mb M] [--nocache] [--json]
 * Default: legge tutto il file, 5 reps, chunk 8 MB, buffered (senza --nocache).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#if defined(__APPLE__)
#include <sys/fcntl.h>
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "uso: %s <file> [--mb N] [--reps K] [--chunk-mb M] [--nocache] [--json]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    uint64_t max_bytes = 0;
    int reps = 5, json = 0, nocache = 0;
    size_t chunk = 8u << 20;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--mb") && i + 1 < argc) max_bytes = strtoull(argv[++i], NULL, 10) << 20;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk-mb") && i + 1 < argc) chunk = (size_t)strtoull(argv[++i], NULL, 10) << 20;
        else if (!strcmp(argv[i], "--nocache")) nocache = 1;
        else if (!strcmp(argv[i], "--json")) json = 1;
    }
    if (reps < 1) reps = 1;
    if (chunk == 0) chunk = 8u << 20;

    int flags = O_RDONLY;
#if defined(__linux__)
    if (nocache) flags |= O_DIRECT;
#endif
    int fd = open(path, flags);
    if (fd < 0) { perror("open"); return 1; }
#if defined(__APPLE__)
    if (nocache) { if (fcntl(fd, F_NOCACHE, 1) != 0) { perror("F_NOCACHE"); close(fd); return 1; } }
#endif
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { fprintf(stderr, "file non valido\n"); close(fd); return 1; }
    uint64_t size = (uint64_t)st.st_size;
    if (max_bytes > 0 && max_bytes < size) size = max_bytes;

    /* buffer allineato a 4K (O_DIRECT su Linux lo richiede; innocuo altrove) */
    void* buf = NULL;
    if (posix_memalign(&buf, 4096, chunk) != 0 || buf == NULL) { fprintf(stderr, "OOM\n"); close(fd); return 1; }

    double* mbs = (double*)calloc((size_t)reps, sizeof(double));
    if (mbs == NULL) { free(buf); close(fd); return 1; }
    uint64_t total = 0;
    for (int r = 0; r < reps; ++r) {
        if (lseek(fd, 0, SEEK_SET) != 0) { perror("lseek"); break; }
        uint64_t done = 0;
        double t0 = now_s();
        while (done < size) {
            size_t want = (size - done) < chunk ? (size_t)(size - done) : chunk;
            ssize_t got = pread(fd, buf, want, (off_t)done);
            if (got <= 0) { perror("pread"); break; }
            done += (uint64_t)got;
        }
        double dt = now_s() - t0;
        mbs[r] = dt > 0 ? ((double)done / (1024.0 * 1024.0)) / dt : 0.0;
        if (r == 0) total = done;
    }
    qsort(mbs, (size_t)reps, sizeof(double), cmp_double);
    double best = mbs[0], med = mbs[reps / 2];
    if (json) {
        printf("{\"file\":\"%s\",\"bytes\":%llu,\"chunk_mb\":%zu,\"reps\":%d,\"nocache\":%s,"
               "\"best_mb_s\":%.1f,\"median_mb_s\":%.1f}\n",
               path, (unsigned long long)total, chunk >> 20, reps, nocache ? "true" : "false", best, med);
    } else {
        printf("moe_iobw: %s  bytes=%llu  chunk=%zuMB  reps=%d  nocache=%s\n",
               path, (unsigned long long)total, chunk >> 20, reps, nocache ? "on" : "off");
        printf("  best=%.1f MB/s  median=%.1f MB/s  (min su %d reps; macchina quieta richiesta)\n",
               best, med, reps);
    }
    free(mbs); free(buf); close(fd);
    return 0;
}
