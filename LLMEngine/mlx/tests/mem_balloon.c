/* mem_balloon (2026-10-30) — pallone di memoria per MISURARE il comportamento
 * sotto pressione: alloca e tocca N MiB (pagine anonime) e dorme. Serve a simulare
 * una macchina più piccola (es. 24 GB) su una macchina grande, per verificare che
 * un modello con pesi file-backed (mmap) venga paginato dal SO invece di
 * sbattere contro la RAM.
 *
 * Uso: ./mem_balloon <MiB> [secondi]
 * Stampa il PID e resta in attesa: terminarlo (SIGTERM/kill) libera subito la RAM.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int s) { (void)s; g_stop = 1; }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <MiB> [secondi]\n", argv[0]); return 2; }
    size_t mib = (size_t)strtoull(argv[1], NULL, 10);
    int secs = (argc > 2) ? atoi(argv[2]) : 0;
    if (mib == 0) { fprintf(stderr, "MiB deve essere > 0\n"); return 2; }

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);

    size_t bytes = mib * 1024u * 1024u;
    char* p = (char*)malloc(bytes);
    if (p == NULL) { fprintf(stderr, "allocazione di %zu MiB fallita\n", mib); return 1; }

    /* Touch a pagine intere: senza scrittura la RAM non viene realmente impegnata.
     * `volatile` + checksum finale: altrimenti il compilatore può eliminare gli
     * store verso memoria solo allocata e poi liberata (misurato: RSS 1 MB). */
    volatile char* vp = (volatile char*)p;
    const size_t page = 4096;
    for (size_t i = 0; i < bytes; i += page) vp[i] = (char)(i >> 12);
    unsigned sum = 0;
    for (size_t i = 0; i < bytes; i += page) sum += (unsigned char)vp[i];

    printf("pid=%d balloon=%zuMiB checksum=%u\n", (int)getpid(), mib, sum);
    fflush(stdout);

    if (secs > 0) { for (int i = 0; i < secs * 10 && !g_stop; ++i) usleep(100 * 1000); }
    else { while (!g_stop) usleep(100 * 1000); }

    free(p);
    printf("balloon rilasciato\n");
    return 0;
}
