/* llmengine_moe_safetensors_ingest.h — INGEST esperti per MoE in formato SAFETENSORS (MLX).
 *
 * Ruolo nel disegno "no duplicazione": il runtime MoE generico (`llmengine_moe`) conosce solo
 * esperti = blob `(layer, expert)` (vedi `llmengine_moe_expert.h`, "nessuna semantica"); l'unico pezzo
 * per-formato finora era l'adapter GGUF. Questo ingest copre il caso SAFETENSORS e produce, per
 * `(layer, expert)`, le SLICE dei tensori coinvolti.
 *
 * Perche' slice multiple (e non UN blob come GGUF): in safetensors gli esperti sono STACKED per
 * proiezione e weight/scales/biases stanno in regioni SEPARATE -> per un esperto sono 3..9 slice
 * NON contigue (offset diversi). Struttura verificata (Qwen3.8-Flash-Next):
 *   switch_mlp.{gate,up,down}_proj.weight  U32  [E,d1,d2]   (es. gate [512,640,320])
 *   switch_mlp.{gate,up,down}_proj.scales  BF16 [E,d1,d2]   (es. gate [512,640,80])
 *   switch_mlp.{gate,up,down}_proj.biases  BF16 [E,d1,d2]
 * Self-contained (POSIX open/mmap), solo libc. NON contiene forward: i consumer (MLX 4-bit Metal,
 * o float) leggono le slice e calcolano a modo loro -> nessun forward duplicato. */
#ifndef LLMENGINE_MOE_SAFETENSORS_INGEST_H
#define LLMENGINE_MOE_SAFETENSORS_INGEST_H
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

#define LLMENGINE_MOE_TS_MAXLAYER 512
#define LLMENGINE_MOE_TS_NPROJ 3 /* 0=gate 1=up 2=down */
#define LLMENGINE_MOE_TS_NKIND 3 /* 0=weight 1=scales 2=biases */
#define LLMENGINE_MOE_TS_MAXMAP 64

typedef struct { int fd; const uint8_t* base; size_t len; } LlmEngineMoeTsMap;

/* Descrittore di UNA slice di tensore per-esperto. */
typedef struct {
    int map;          /* indice in .maps, -1 = assente */
    uint64_t off;     /* offset del tensore nello shard (base dell'esperto 0) */
    uint32_t stride;  /* byte per esperto */
    uint32_t bytes;   /* byte della slice dell'esperto selezionato */
    int n;            /* n esperti */
    int d1, d2;       /* shape per-esperto [d1,d2] */
    int dtype;        /* 0=U32 1=BF16 (convenzione del formato) */
} LlmEngineMoeTsSlice;

/* Tensore AUSILIARIO TurboQuant (`.signs` / `.codebook`): 1-D e CONDIVISO per proiezione
 * (un solo vettore per (layer,proj), non uno per-esperto). Vive in un ramo SEPARATO dal
 * modello per-esperto (kind 0/1/2) e NON lo altera. */
typedef struct {
    int map;        /* indice in .maps (valido solo se n>0) */
    uint64_t off;   /* offset ASSOLUTO del tensore (data_start + data_offsets[0]) */
    uint32_t bytes; /* byte totali del tensore */
    int n;          /* elementi F16: signs = `in`, codebook = voci (3 o 8); 0 = assente */
} LlmEngineMoeTsAux;

typedef struct {
    LlmEngineMoeTsMap maps[LLMENGINE_MOE_TS_MAXMAP];
    int nmaps;
    LlmEngineMoeTsSlice slice[LLMENGINE_MOE_TS_MAXLAYER][LLMENGINE_MOE_TS_NPROJ][LLMENGINE_MOE_TS_NKIND];
    LlmEngineMoeTsAux tq_signs[LLMENGINE_MOE_TS_MAXLAYER][LLMENGINE_MOE_TS_NPROJ];
    LlmEngineMoeTsAux tq_codebook[LLMENGINE_MOE_TS_MAXLAYER][LLMENGINE_MOE_TS_NPROJ];
    int present[LLMENGINE_MOE_TS_MAXLAYER];
} LlmEngineMoeTsIngest;

static inline int llmengine_moe_ts_map(LlmEngineMoeTsIngest* s, const char* path) {
    if (s->nmaps >= LLMENGINE_MOE_TS_MAXMAP) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return -1; }
    void* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { close(fd); return -1; }
    /* MADV_RANDOM: gli esperti sono letti in ordine SPARSO (per-token, per-layer), non sequenziale:
     * disattiva il read-ahead sequenziale, che tirerebbe in cache gli esperti ADIACENTI mai usati
     * (spreco di I/O su modelli > RAM). Hint puro, nessun cambio di correttezza. */
    madvise(m, (size_t)st.st_size, MADV_RANDOM);
    s->maps[s->nmaps].fd = fd; s->maps[s->nmaps].base = (const uint8_t*)m; s->maps[s->nmaps].len = (size_t)st.st_size;
    return s->nmaps++;
}

static inline void llmengine_moe_ts_put(LlmEngineMoeTsIngest* s, int map, const char* name, const char* obj, uint64_t data_start) {
    const char* k = strstr(name, ".layers."); if (!k) return;
    int layer = atoi(k + 8);
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER) return;
    int proj = strstr(name, "gate_proj") ? 0 : strstr(name, "up_proj") ? 1 : strstr(name, "down_proj") ? 2 : -1;
    if (proj < 0 || !strstr(name, "switch_mlp")) return;
    int kind = strstr(name, ".scales") ? 1 : strstr(name, ".biases") ? 2 : 0;
    const char* sh = strstr(obj, "\"shape\":["); const char* of = strstr(obj, "\"data_offsets\":[");
    long long e = -1, o0 = -1, o1 = -1, d1 = 0, d2 = 0;
    if (sh) sscanf(sh, "\"shape\":[%lld,%lld,%lld", &e, &d1, &d2);
    if (of) sscanf(of, "\"data_offsets\":[%lld,%lld", &o0, &o1);
    if (e <= 0 || o0 < 0 || o1 <= o0) return;
    LlmEngineMoeTsSlice* t = &s->slice[layer][proj][kind];
    /* "data_offsets" nel formato safetensors sono relativi all'INIZIO DELLA SEZIONE DATI (dopo
     * gli 8 byte di lunghezza header + l'header JSON), NON all'inizio del file — bug
     * "MLX-Streaming-Esperti-SWAP-Regressione" §Bug 3: senza +data_start ogni lettura finiva
     * `data_start` byte troppo PRESTO nel file (dentro l'header JSON o un tensore precedente),
     * producendo dati apparentemente plausibili ma con bit pattern NaN occasionali (output
     * degenere, es. token costante). Port del fix GMBasic (gmb_moe_safetensors_ingest.h). */
    t->map = map; t->off = (uint64_t)o0 + data_start; t->stride = (uint32_t)((o1 - o0) / e);
    t->bytes = t->stride; t->n = (int)e; t->d1 = (int)d1; t->d2 = (int)d2;
    t->dtype = (kind == 0) ? 0 : 1;
    s->present[layer] = 1;
}

/* Ramo SEPARATO per i tensori ausiliari TurboQuant (`.signs` / `.codebook`). Ritorna 1 se il nome
 * e' un ausiliario tq (GESTITO: il chiamante NON deve passarlo a llmengine_moe_ts_put), 0 altrimenti.
 *
 * CRITICO: un nome `.signs`/`.codebook` non contiene ne' `.scales` ne' `.biases` -> in
 * llmengine_moe_ts_put cadrebbe in kind 0 (weight) e SOVRASCRIVEREBBE lo slice weight vero con una
 * shape 1-D (d1=d2=0), rompendo il path affine per-esperto ("sorted MoE branch failed"). Per questo
 * i due tensori non passano MAI dal descrittore per-esperto. */
static inline int llmengine_moe_ts_aux(LlmEngineMoeTsIngest* s, int map, const char* name, const char* obj, uint64_t data_start) {
    if (!strstr(name, "switch_mlp")) return 0;
    int is_cb = 0;
    const char* suf = strstr(name, ".signs");
    if (!suf) { suf = strstr(name, ".codebook"); is_cb = 1; }
    if (!suf) return 0;
    const char* k = strstr(name, ".layers."); if (!k) return 0;
    int layer = atoi(k + 8);
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER) return 0;
    int proj = strstr(name, "gate_proj") ? 0 : strstr(name, "up_proj") ? 1 : strstr(name, "down_proj") ? 2 : -1;
    if (proj < 0) return 0;
    const char* sh = strstr(obj, "\"shape\":["); const char* of = strstr(obj, "\"data_offsets\":[");
    long long n = -1, o0 = -1, o1 = -1;
    if (sh) sscanf(sh, "\"shape\":[%lld", &n);
    if (of) sscanf(of, "\"data_offsets\":[%lld,%lld", &o0, &o1);
    if (n <= 0 || o0 < 0 || o1 <= o0) return 1; /* ausiliario riconosciuto ma malformato: non inoltrarlo */
    LlmEngineMoeTsAux* a = is_cb ? &s->tq_codebook[layer][proj] : &s->tq_signs[layer][proj];
    a->map = map; a->off = (uint64_t)o0 + data_start; a->bytes = (uint32_t)(o1 - o0); a->n = (int)n;
    return 1;
}

/* Indicizza tutti i *.safetensors della dir. Ritorna n file mappati, o -1. */
static inline int llmengine_moe_ts_open(LlmEngineMoeTsIngest* s, const char* dir) {
    memset(s, 0, sizeof *s);
    DIR* d = opendir(dir); if (!d) return -1;
    struct dirent* de;
    while ((de = readdir(d)) != NULL) {
        size_t l = strlen(de->d_name);
        if (l < 12 || strcmp(de->d_name + l - 12, ".safetensors") != 0) continue;
        char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, de->d_name);
        int map = llmengine_moe_ts_map(s, path);
        if (map < 0) { closedir(d); return -1; }
        const uint8_t* b = s->maps[map].base;
        uint64_t hn = 0; memcpy(&hn, b, 8);
        const uint64_t data_start = 8 + hn;   /* inizio sezione dati: 8 byte lunghezza + header JSON */
        char* h = (char*)malloc((size_t)hn + 1); if (!h) { closedir(d); return -1; }
        memcpy(h, b + 8, (size_t)hn); h[hn] = 0;
        char* p = h;
        while ((p = strstr(p, "\"")) != NULL) {
            char* ne = strchr(p + 1, '"'); if (!ne) break;
            size_t nl = (size_t)(ne - p - 1);
            if (nl < 500) { char name[512]; memcpy(name, p + 1, nl); name[nl] = 0;
                /* Ausiliari tq (signs/codebook): ramo separato, mai nel descrittore per-esperto. */
                if (!llmengine_moe_ts_aux(s, map, name, ne + 1, data_start))
                    llmengine_moe_ts_put(s, map, name, ne + 1, data_start); }
            p = (ne > p) ? ne + 1 : p + 1;
        }
        free(h);
    }
    closedir(d);
    return s->nmaps;
}

/* Slice dell'esperto `e` per (layer, proj, kind): puntatore mmap + bytes, o NULL. */
static inline const uint8_t* llmengine_moe_ts_get(const LlmEngineMoeTsIngest* s, int layer, int proj, int kind, int e, uint32_t* bytes) {
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER || proj < 0 || proj >= LLMENGINE_MOE_TS_NPROJ || kind < 0 || kind >= LLMENGINE_MOE_TS_NKIND) return NULL;
    const LlmEngineMoeTsSlice* t = &s->slice[layer][proj][kind];
    if (t->map < 0 || e < 0 || e >= t->n) return NULL;
    if (bytes) *bytes = t->bytes;
    return s->maps[t->map].base + t->off + (uint64_t)e * t->stride;
}

/* Descrittore della slice (senza il puntatore): per i consumer che vogliono offset/shape/dtype. */
static inline const LlmEngineMoeTsSlice* llmengine_moe_ts_slice(const LlmEngineMoeTsIngest* s, int layer, int proj, int kind) {
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER || proj < 0 || proj >= LLMENGINE_MOE_TS_NPROJ || kind < 0 || kind >= LLMENGINE_MOE_TS_NKIND) return NULL;
    const LlmEngineMoeTsSlice* t = &s->slice[layer][proj][kind];
    return t->map < 0 ? NULL : t;
}

/* Hint di page-cache su un range del mapping (prefetch/evict/diradamento). `madvise` vuole indirizzi
 * allineati a pagina: si arrotonda il range verso il basso/alto. Hint PURO: nessun cambio di valori. */
static inline void llmengine_moe_ts_advise_range(const LlmEngineMoeTsIngest* s, int map,
                                                 uint64_t off, uint32_t bytes, int advise) {
    if (s == NULL || map < 0 || map >= s->nmaps || bytes == 0) return;
    long pg = sysconf(_SC_PAGESIZE); if (pg <= 0) pg = 4096;
    const uintptr_t mask = (uintptr_t)pg - 1;
    uintptr_t start = (uintptr_t)s->maps[map].base + (uintptr_t)off;
    uintptr_t a0 = start & ~mask;
    uintptr_t a1 = (start + (uintptr_t)bytes + mask) & ~mask;
    madvise((void*)a0, (size_t)(a1 - a0), advise);
}

/* Hint su TUTTE le slice (weight/scales/biases) dell'esperto `e` per (layer,proj):
 * MADV_WILLNEED = prefetch; MADV_DONTNEED = rilascia le pagine (ri-letture dal file). */
static inline void llmengine_moe_ts_advise_expert(const LlmEngineMoeTsIngest* s, int layer, int proj,
                                                  int e, int advise) {
    if (s == NULL || layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER ||
        proj < 0 || proj >= LLMENGINE_MOE_TS_NPROJ) return;
    for (int kind = 0; kind < LLMENGINE_MOE_TS_NKIND; kind++) {
        const LlmEngineMoeTsSlice* t = &s->slice[layer][proj][kind];
        if (t->map < 0 || e < 0 || e >= t->n) continue;
        llmengine_moe_ts_advise_range(s, t->map, t->off + (uint64_t)e * t->stride, t->bytes, advise);
    }
}

/* Ausiliario tq per (layer,proj): puntatore mmap + byte/numero elementi, o NULL. */
static inline const uint8_t* llmengine_moe_ts_aux_get(const LlmEngineMoeTsIngest* s, int layer, int proj,
                                                      int is_codebook, uint32_t* bytes, int* n) {
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER || proj < 0 || proj >= LLMENGINE_MOE_TS_NPROJ) return NULL;
    const LlmEngineMoeTsAux* a = is_codebook ? &s->tq_codebook[layer][proj] : &s->tq_signs[layer][proj];
    if (a->n <= 0) return NULL;
    if (bytes) *bytes = a->bytes;
    if (n) *n = a->n;
    return s->maps[a->map].base + a->off;
}

/* true se (layer,proj) ha ENTRAMBI signs e codebook (proiezione TurboQuant). */
static inline int llmengine_moe_ts_is_tq(const LlmEngineMoeTsIngest* s, int layer, int proj) {
    if (layer < 0 || layer >= LLMENGINE_MOE_TS_MAXLAYER || proj < 0 || proj >= LLMENGINE_MOE_TS_NPROJ) return 0;
    return s->tq_signs[layer][proj].n > 0 && s->tq_codebook[layer][proj].n > 0;
}

static inline void llmengine_moe_ts_close(LlmEngineMoeTsIngest* s) {
    for (int i = 0; i < s->nmaps; i++) { munmap((void*)s->maps[i].base, s->maps[i].len); close(s->maps[i].fd); }
    s->nmaps = 0;
}

#endif /* LLMENGINE_MOE_SAFETENSORS_INGEST_H */
