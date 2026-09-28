/* llm_mlx_kvdisk_test.c — test del tier SSD della KV cache (llm_mlx_kvdisk.h).
 * Round-trip, LRU+cap, riuso range, corruzione (checksum), statistiche. Nessuna dipendenza MLX. */
#include "llm_mlx_kvdisk.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

int main(void) {
    const size_t BS = 1024;                 /* blocco KV di prova */
    uint8_t a[1024], b[1024], c[1024], d_[1024], e_[1024], out[1024];
    for (size_t i = 0; i < BS; i++) { a[i] = (uint8_t)(i * 3 + 1); b[i] = (uint8_t)(i * 5); c[i] = (uint8_t)(i ^ 0x55); d_[i] = (uint8_t)(255 - i); e_[i] = (uint8_t)(i + 7); }
    uint64_t hA = llm_kv_disk_hash(a, BS), hB = llm_kv_disk_hash(b, BS), hC = llm_kv_disk_hash(c, BS);
    uint64_t hD = llm_kv_disk_hash(d_, BS), hE = llm_kv_disk_hash(e_, BS);

    char path[] = "/tmp/llm_kvdisk_test.XXXXXX.scratch";
    KvDisk kd = llm_kv_disk_open(path, 1024, 3 * BS);   /* budget: 3 blocchi */
    CHECK(kd.f != NULL && kd.e != NULL);

    /* round-trip */
    CHECK(llm_kv_disk_put(&kd, hA, a, BS) == 0);
    CHECK(llm_kv_disk_has(&kd, hA, NULL) == 1);
    CHECK(llm_kv_disk_get(&kd, hA, out, BS, NULL) == 0);
    CHECK(memcmp(out, a, BS) == 0);
    uint64_t offA = 0; for (int i = 0; i < kd.n_entries; i++) if (kd.e[i].used && kd.e[i].hash == hA) offA = kd.e[i].off;

    /* put idempotente (stesso hash): non cresce live_bytes */
    uint64_t live1 = kd.live_bytes;
    CHECK(llm_kv_disk_put(&kd, hA, a, BS) == 0);
    CHECK(kd.live_bytes == live1);

    /* riempi (B,C) -> 3 blocchi pieni */
    CHECK(llm_kv_disk_put(&kd, hB, b, BS) == 0);
    CHECK(llm_kv_disk_put(&kd, hC, c, BS) == 0);
    CHECK(kd.live_bytes == 3 * BS);
    uint64_t offB = 0; for (int i = 0; i < kd.n_entries; i++) if (kd.e[i].used && kd.e[i].hash == hB) offB = kd.e[i].off;

    /* D -> evict LRU (A): A manca, D ok, D RIUSA il range di A, live resta <= cap */
    CHECK(llm_kv_disk_put(&kd, hD, d_, BS) == 0);
    CHECK(llm_kv_disk_has(&kd, hA, NULL) == 0);
    CHECK(llm_kv_disk_has(&kd, hD, NULL) == 1);
    CHECK(kd.live_bytes <= 3 * BS);
    CHECK(kd.evictions >= 1);
    uint64_t offD = 0; for (int i = 0; i < kd.n_entries; i++) if (kd.e[i].used && kd.e[i].hash == hD) offD = kd.e[i].off;
    CHECK(offD == offA);   /* D occupa il range liberato da A (primo-fit) */

    /* E -> evict LRU (B) e riusa il range di B: il file NON cresce */
    uint64_t fb = kd.file_bytes;
    CHECK(llm_kv_disk_put(&kd, hE, e_, BS) == 0);
    CHECK(kd.file_bytes == fb);
    uint64_t offE = 0; for (int i = 0; i < kd.n_entries; i++) if (kd.e[i].used && kd.e[i].hash == hE) offE = kd.e[i].off;
    CHECK(offE == offB);   /* E occupa il range liberato da B */

    /* corruzione: altero un byte del blocco D (VIVO) -> get fallisce (checksum), entry invalidata */
    CHECK(fseeko(kd.f, (off_t)offD, SEEK_SET) == 0);
    CHECK(fputc(0x00, kd.f) != EOF);
    CHECK(llm_kv_disk_get(&kd, hD, out, BS, NULL) == -1);
    CHECK(llm_kv_disk_has(&kd, hD, NULL) == 0);

    /* reopen: l'indice e' di durata-processo (scratch temporaneo, §G-ter) -> il file e' ricreato */
    llm_kv_disk_close(&kd);
    KvDisk k2 = llm_kv_disk_open(path, 16, 2 * BS);
    CHECK(k2.f != NULL);
    CHECK(llm_kv_disk_has(&k2, hD, NULL) == 0);   /* nuovo scratch: nessun contenuto */
    llm_kv_disk_close(&k2);

    printf("kvdisk: puts=%ld gets=%ld hits=%ld misses=%ld evictions=%ld\n",
           kd.puts, kd.gets, kd.hits, kd.misses, kd.evictions);
    if (fails == 0) printf("llm_mlx_kvdisk_test: OK\n");
    else fprintf(stderr, "llm_mlx_kvdisk_test: %d FAIL\n", fails);
    return fails ? 1 : 0;
}
