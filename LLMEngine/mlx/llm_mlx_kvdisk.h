/* llm_mlx_kvdisk.h — TIER SSD della KV cache (offload su disco, §G).
 *
 * Complementa `llm_mlx_kvpage.h` (pool RAM content-addressed, LRU, pin): qui il tier **cold**.
 * I blocchi KV (serializzati dal motore) sono salvati in UN file scratch con record packed +
 * free-list a range + indice in RAM (chiave = content_hash, coerente col page pool). La cache
 * disk e' TEMPORANEA e di durata del modello: alla chiusura l'indice e' scartato e il file
 * rimosso (come LM Studio / §G-ter). LRU eviction rispetta un budget di byte.
 *
 * Adatto a **rewind agentico** (LM Studio mlx-engine): salva i blocchi del prefisso, ripristina
 * il prefisso piu' lungo disponibile, ricalcola solo il suffisso mancante.
 *
 * Convenzione (come kvpage/ops): funzioni `static inline`, PURE, nessun MLX, testabile in isolamento.
 */
#ifndef LLM_MLX_KVDISK_H
#define LLM_MLX_KVDISK_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    bool     used;
    uint64_t hash;
    uint64_t off;
    uint64_t size;
    uint64_t ck;          /* checksum FNV-1a del contenuto */
    uint64_t last_used;
} KvDiskEntry;

typedef struct { uint64_t off; uint64_t size; } KvDiskRange;

typedef struct {
    FILE*        f;
    char         path[1200];
    KvDiskEntry* e;
    int          n_entries;     /* capienza tabella */
    int          n_live;        /* entry usate */
    KvDiskRange* free_list;
    int          n_free, cap_free;
    uint64_t     cap_bytes;     /* budget massimo di byte VIVI */
    uint64_t     live_bytes;
    uint64_t     file_bytes;    /* dimensione corrente del file */
    uint64_t     clock;
    /* statistiche */
    long puts, gets, hits, misses, evictions;
    bool persistent;            /* D15: persistente tra processi (no truncate, no remove, indice su sidecar) */
} KvDisk;

static inline uint64_t llm_kv_disk_hash(const void* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

/* D15: INDICE del kvdisk su sidecar `<path>.idx` (per la modalita' persistente). Formato:
 * magic "KVDSKI1" + u32 ver=1 + u32 n, poi n x {u64 hash,off,size,ck,last_used}. */
#define LLM_KVDSK_IDX_MAGIC "KVDSKI1"
static inline void llm_kv_disk_save_index(KvDisk* d) {
    if (!d || !d->persistent) return;
    char ip[1300]; snprintf(ip, sizeof ip, "%s.idx", d->path);
    char tp[1320]; snprintf(tp, sizeof tp, "%s.tmp", ip);
    FILE* f = fopen(tp, "wb"); if (!f) return;
    uint32_t ver = 1, n = 0; for (int i = 0; i < d->n_entries; i++) if (d->e[i].used) n++;
    fwrite(LLM_KVDSK_IDX_MAGIC, 1, 7, f); fwrite(&ver, 4, 1, f); fwrite(&n, 4, 1, f);
    for (int i = 0; i < d->n_entries; i++) if (d->e[i].used) {
        uint64_t v[5]; v[0]=d->e[i].hash; v[1]=d->e[i].off; v[2]=d->e[i].size; v[3]=d->e[i].ck; v[4]=d->e[i].last_used;
        fwrite(v, sizeof(uint64_t), 5, f);
    }
    fclose(f); rename(tp, ip);
}
static inline void llm_kv_disk_load_index(KvDisk* d) {
    if (!d || !d->persistent) return;
    char ip[1300]; snprintf(ip, sizeof ip, "%s.idx", d->path);
    FILE* f = fopen(ip, "rb"); if (!f) return;
    char magic[7]; uint32_t ver=0, n=0;
    if (fread(magic,1,7,f)==7 && !memcmp(magic,LLM_KVDSK_IDX_MAGIC,7) && fread(&ver,4,1,f)==1 && ver==1 && fread(&n,4,1,f)==1) {
        for (uint32_t k = 0; k < n && k < (uint32_t)d->n_entries; k++) {
            uint64_t v[5]; if (fread(v,sizeof(uint64_t),5,f)!=5) break;
            int slot = -1; for (int i=0;i<d->n_entries;i++) if (!d->e[i].used) { slot=i; break; }
            if (slot < 0) break;
            d->e[slot].used=true; d->e[slot].hash=v[0]; d->e[slot].off=v[1]; d->e[slot].size=v[2];
            d->e[slot].ck=v[3]; d->e[slot].last_used=v[4];
            d->n_live++; d->live_bytes += v[2]; if (v[4] > d->clock) d->clock = v[4];
        }
    }
    fclose(f);
}
static inline KvDisk llm_kv_disk_open_mode(const char* path, int max_entries, uint64_t cap_bytes, int persistent) {
    KvDisk d; memset(&d, 0, sizeof(d));
    snprintf(d.path, sizeof(d.path), "%s", path ? path : "/tmp/llm_kvdisk.scratch");
    d.persistent = persistent ? true : false;
    if (d.persistent) { d.f = fopen(d.path, "r+b"); if (!d.f) d.f = fopen(d.path, "w+b"); }
    else { d.f = fopen(d.path, "w+b"); if (!d.f) d.f = fopen(d.path, "w+b"); }
    d.n_entries = max_entries > 0 ? max_entries : 65536;
    d.e = (KvDiskEntry*)calloc((size_t)d.n_entries, sizeof(KvDiskEntry));
    d.cap_free = 256;
    d.free_list = (KvDiskRange*)calloc((size_t)d.cap_free, sizeof(KvDiskRange));
    d.cap_bytes = cap_bytes;
    if (d.persistent && d.f) { fseeko(d.f, 0, SEEK_END); d.file_bytes = (uint64_t)ftello(d.f); llm_kv_disk_load_index(&d); }
    return d;
}
static inline KvDisk llm_kv_disk_open(const char* path, int max_entries, uint64_t cap_bytes) {
    return llm_kv_disk_open_mode(path, max_entries, cap_bytes, 0);
}

static inline void llm_kv_disk_close(KvDisk* d) {
    if (!d) return;
    if (d->f) { if (d->persistent) llm_kv_disk_save_index(d); fclose(d->f); d->f = NULL; if (!d->persistent) remove(d->path); }
    free(d->e); d->e = NULL;
    free(d->free_list); d->free_list = NULL;
    d->n_live = d->n_free = 0;
}

static inline int llm_kv_disk_find(const KvDisk* d, uint64_t hash) {
    for (int i = 0; i < d->n_entries; i++) if (d->e[i].used && d->e[i].hash == hash) return i;
    return -1;
}
static inline int llm_kv_disk_free_slot(const KvDisk* d) {
    for (int i = 0; i < d->n_entries; i++) if (!d->e[i].used) return i;
    return -1;
}
static inline void llm_kv_disk_push_range(KvDisk* d, uint64_t off, uint64_t size) {
    if (d->n_free == d->cap_free) {
        d->cap_free *= 2;
        d->free_list = (KvDiskRange*)realloc(d->free_list, (size_t)d->cap_free * sizeof(KvDiskRange));
    }
    d->free_list[d->n_free].off = off;
    d->free_list[d->n_free].size = size;
    d->n_free++;
}
/* primo-fit; ritorna l'offset, o -1 (nessun range adatto -> append). */
static inline int llm_kv_disk_take_range(KvDisk* d, uint64_t size, uint64_t* off_out) {
    for (int i = 0; i < d->n_free; i++) {
        if (d->free_list[i].size >= size) {
            *off_out = d->free_list[i].off;
            if (d->free_list[i].size == size) {
                d->free_list[i] = d->free_list[--d->n_free];
            } else {
                d->free_list[i].off += size;
                d->free_list[i].size -= size;
            }
            return 0;
        }
    }
    return -1;
}
/* evict LRU finche' live+need <= cap. Ritorna 0 se c'e' spazio, -1 se impossibile. */
static inline int llm_kv_disk_ensure(KvDisk* d, uint64_t need) {
    while (d->cap_bytes > 0 && d->live_bytes + need > d->cap_bytes) {
        int victim = -1; uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < d->n_entries; i++)
            if (d->e[i].used && d->e[i].last_used < oldest) { oldest = d->e[i].last_used; victim = i; }
        if (victim < 0) return -1;
        llm_kv_disk_push_range(d, d->e[victim].off, d->e[victim].size);
        d->live_bytes -= d->e[victim].size;
        d->e[victim].used = false;
        d->evictions++;
    }
    return 0;
}

/* Salva un blocco: se esiste (stesso hash) touch; altrimenti occupa un range. Ritorna 0/-1. */
static inline int llm_kv_disk_put(KvDisk* d, uint64_t hash, const void* data, size_t n) {
    if (!d || !d->f || !data || n == 0) return -1;
    d->puts++;
    int ex = llm_kv_disk_find(d, hash);
    if (ex >= 0) { d->e[ex].last_used = ++d->clock; return 0; }
    if (llm_kv_disk_ensure(d, n) != 0) return -1;
    int slot = llm_kv_disk_free_slot(d);
    if (slot < 0) return -1;
    uint64_t off = 0;
    if (llm_kv_disk_take_range(d, n, &off) != 0) { off = d->file_bytes; d->file_bytes += n; }
    if (fseeko(d->f, (off_t)off, SEEK_SET) != 0) return -1;
    if (fwrite(data, 1, n, d->f) != n) return -1;
    d->e[slot].used = true;
    d->e[slot].hash = hash;
    d->e[slot].off = off;
    d->e[slot].size = n;
    d->e[slot].ck = llm_kv_disk_hash(data, n);
    d->e[slot].last_used = ++d->clock;
    d->n_live++;
    d->live_bytes += n;
    if (d->persistent) llm_kv_disk_save_index(d);
    return 0;
}

/* Legge un blocco per hash. Ritorna 0 e scrive n_out su successo; -1 se manca/corrotto. */
static inline int llm_kv_disk_get(KvDisk* d, uint64_t hash, void* out, size_t cap, size_t* n_out) {
    if (!d || !d->f || !out) return -1;
    d->gets++;
    int i = llm_kv_disk_find(d, hash);
    if (i < 0) { d->misses++; return -1; }
    if (d->e[i].size > cap) { d->misses++; return -1; }
    if (fseeko(d->f, (off_t)d->e[i].off, SEEK_SET) != 0) { d->misses++; return -1; }
    size_t got = fread(out, 1, (size_t)d->e[i].size, d->f);
    if (got != d->e[i].size) { d->misses++; return -1; }
    if (llm_kv_disk_hash(out, got) != d->e[i].ck) {   /* corrotto: invalida l'entry */
        llm_kv_disk_push_range(d, d->e[i].off, d->e[i].size);
        d->live_bytes -= d->e[i].size;
        d->e[i].used = false;
        d->misses++;
        return -1;
    }
    d->e[i].last_used = ++d->clock;
    d->hits++;
    if (n_out) *n_out = got;
    return 0;
}

static inline int llm_kv_disk_has(const KvDisk* d, uint64_t hash, size_t* n_out) {
    int i = llm_kv_disk_find(d, hash);
    if (i < 0) return 0;
    if (n_out) *n_out = (size_t)d->e[i].size;
    return 1;
}

/* Elimina un blocco: libera il range (riusabile) e invalida l'entry. 0 se esisteva, -1 altrimenti. */
static inline int llm_kv_disk_del(KvDisk* d, uint64_t hash) {
    if (!d) return -1;
    int i = llm_kv_disk_find(d, hash);
    if (i < 0) return -1;
    llm_kv_disk_push_range(d, d->e[i].off, d->e[i].size);
    d->live_bytes -= d->e[i].size;
    d->e[i].used = false;
    d->n_live--;
    return 0;
}

#endif /* LLM_MLX_KVDISK_H */
