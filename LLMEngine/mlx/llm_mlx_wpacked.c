/* llm_mlx_wpacked.c — implementazione M1 (mmap + risoluzione weight/scales/biases).
 * Nei checkpoint MLX multi-shard i tre componenti di un tensore (<base>.weight/.scales/.biases)
 * possono stare in shard DIVERSI: ognuno viene localizzato (prima nel file dato, poi via
 * model.safetensors.index.json nella stessa directory) e mappato per conto proprio. */
#include "llm_mlx_wpacked.h"
#include "llmcodec_safetensors.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const uint8_t* ptr;      /* inizio dei dati del tensore dentro la mappa */
    uint64_t len;
    int ndim; uint64_t shape[8];
    int bf16;
    void* map; size_t map_bytes;
    int owns_map;
} Loc;

static int find_info(const LlmCodecStTensorInfo* ti, int n, const char* nm) {
    for (int i = 0; i < n; i++) if (strcmp(ti[i].name, nm) == 0) return i;
    return -1;
}

/* Cerca `name` nell'header di `path`; se presente mappa il file e riempie `L`. 0 ok, 1 assente, -1 errore. */
static int locate_in_file(const char* path, const char* name, Loc* L) {
    LlmCodecStTensorInfo* ti = NULL; int n = 0; uint64_t base_off = 0;
    if (llmcodec_safetensors_read(path, &ti, &n, &base_off) != 0) return -1;
    int i = find_info(ti, n, name);
    if (i < 0) { llmcodec_safetensors_info_free(ti); return 1; }
    memset(L, 0, sizeof *L);
    L->len = ti[i].end - ti[i].begin; L->ndim = ti[i].ndim;
    for (int d = 0; d < ti[i].ndim && d < 8; d++) L->shape[d] = ti[i].shape[d];
    L->bf16 = strcmp(ti[i].dtype, "BF16") == 0;
    uint64_t off = base_off + ti[i].begin;
    llmcodec_safetensors_info_free(ti);
    int fd = open(path, O_RDONLY); if (fd < 0) return -1;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return -1; }
    void* map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return -1;
    L->map = map; L->map_bytes = (size_t)st.st_size; L->owns_map = 1;
    L->ptr = (const uint8_t*)map + off;
    return 0;
}

/* Come locate_in_file, ma se `name` non e' in `hint_path` risolve lo shard dall'index della stessa directory. */
static int locate(const char* hint_path, const char* name, Loc* L) {
    int rc = locate_in_file(hint_path, name, L);
    if (rc <= 0) return rc;
    char dir[1600]; snprintf(dir, sizeof dir, "%s", hint_path);
    char* sl = strrchr(dir, '/'); if (!sl) return -1; *sl = 0;
    char ip[1700]; snprintf(ip, sizeof ip, "%s/model.safetensors.index.json", dir);
    FILE* f = fopen(ip, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (sz > 0) ? (char*)malloc((size_t)sz + 1) : NULL;
    if (!j || fread(j, 1, (size_t)sz, f) != (size_t)sz) { free(j); fclose(f); return -1; }
    fclose(f); j[sz] = 0;
    char key[600]; snprintf(key, sizeof key, "\"%s\"", name);
    char* p = strstr(j, key);
    char shard[300] = {0};
    if (p) {
        char* c = strchr(p + strlen(key), '"');   /* apertura del valore */
        if (c) { char* e = strchr(c + 1, '"'); if (e && (size_t)(e - c - 1) < sizeof shard) { memcpy(shard, c + 1, (size_t)(e - c - 1)); } }
    }
    free(j);
    if (!shard[0]) return -1;
    char path[2000]; snprintf(path, sizeof path, "%s/%s", dir, shard);
    rc = locate_in_file(path, name, L);
    return rc == 0 ? 0 : -1;
}

static void loc_release(Loc* L) { if (L && L->owns_map && L->map) { munmap(L->map, L->map_bytes); L->map = NULL; } }

int llm_mlx_packed_open(const char* path, const char* base, int bits, int group, LlmMlxPackedW* out) {
    if (!path || !base || !out || bits <= 0) return -1;   /* group<=0 = derivalo dal file */
    memset(out, 0, sizeof *out);
    char nm[1024];
    Loc W, S, B; memset(&W, 0, sizeof W); memset(&S, 0, sizeof S); memset(&B, 0, sizeof B);
    snprintf(nm, sizeof nm, "%s.weight", base);
    if (locate_in_file(path, nm, &W) != 0) return -1;
    snprintf(nm, sizeof nm, "%s.scales", base);
    if (locate(path, nm, &S) != 0) { loc_release(&W); return -1; }
    snprintf(nm, sizeof nm, "%s.biases", base);
    int has_b = (locate(path, nm, &B) == 0);
    /* dimensioni dall'header: weight U32 [rows][cols*bits/32] (cols = dim1*32/bits), scales [rows][cols/group] */
    uint64_t rows = W.ndim >= 2 ? W.shape[0] : 0, wcols = W.ndim >= 2 ? W.shape[1] : 0;
    uint64_t scols = S.ndim >= 2 ? S.shape[1] : 0;
    uint64_t cols = wcols * 32ull / (uint64_t)bits;
    if (rows == 0 || wcols == 0 || scols == 0 || cols % scols != 0) { loc_release(&W); loc_release(&S); if (has_b) loc_release(&B); return -1; }
    group = (int)(cols / scols);   /* il gruppo varia per tensore: derivato dal file */
    out->w = (const uint32_t*)W.ptr;
    out->scales = (const uint16_t*)S.ptr;
    out->biases = has_b ? (const uint16_t*)B.ptr : NULL;
    out->rows = (uint32_t)rows; out->cols = (uint32_t)cols; out->bits = bits; out->group = group;
    out->scale_bf16 = S.bf16;
    out->w_bytes = (size_t)W.len; out->s_bytes = (size_t)S.len; out->b_bytes = has_b ? (size_t)B.len : 0;
    out->map = W.map; out->map_bytes = W.map_bytes;
    out->map_s = S.map; out->map_s_bytes = S.map_bytes;
    out->map_b = has_b ? B.map : NULL; out->map_b_bytes = has_b ? B.map_bytes : 0;
    return 0;
}

void llm_mlx_packed_close(LlmMlxPackedW* w) {
    if (!w || !w->map) return;
    munmap(w->map, w->map_bytes);
    if (w->map_s) munmap(w->map_s, w->map_s_bytes);
    if (w->map_b) munmap(w->map_b, w->map_b_bytes);
    memset(w, 0, sizeof *w);
}

/* ---- vista 3D (esperti): [E, rows, cols] ---- */
int llm_mlx_packed_open3d(const char* path, const char* base, int bits, int group, LlmMlxPackedW3* out) {
    if (!path || !base || !out) return -1;
    memset(out, 0, sizeof *out);
    char nm[1024];
    Loc W, S, B; memset(&W, 0, sizeof W); memset(&S, 0, sizeof S); memset(&B, 0, sizeof B);
    snprintf(nm, sizeof nm, "%s.weight", base);
    if (locate_in_file(path, nm, &W) != 0) return -1;
    snprintf(nm, sizeof nm, "%s.scales", base);
    if (locate(path, nm, &S) != 0) { loc_release(&W); return -1; }
    snprintf(nm, sizeof nm, "%s.biases", base);
    int has_b = (locate(path, nm, &B) == 0);
    uint64_t E = W.ndim >= 3 ? W.shape[0] : 0, rows = W.ndim >= 3 ? W.shape[1] : 0, wcols = W.ndim >= 3 ? W.shape[2] : 0;
    uint64_t scols = S.ndim >= 3 ? S.shape[2] : 0;
    uint64_t cols = wcols * 32ull / (uint64_t)bits;
    if (E == 0 || rows == 0 || wcols == 0 || scols == 0 || cols % scols != 0) {
        fprintf(stderr, "[wpacked3d] %s: forma non valida (E=%llu rows=%llu wcols=%llu scols=%llu group=%d)\n", base,
                (unsigned long long)E, (unsigned long long)rows, (unsigned long long)wcols, (unsigned long long)scols, group);
        loc_release(&W); loc_release(&S); if (has_b) loc_release(&B);
        return -1;
    }
    out->w = (const uint32_t*)W.ptr;
    out->scales = (const uint16_t*)S.ptr;
    out->biases = has_b ? (const uint16_t*)B.ptr : NULL;
    out->experts = (uint32_t)E; out->rows = (uint32_t)rows; out->cols = (uint32_t)cols;
    out->bits = bits; out->group = (int)(cols / scols);   /* gruppo per-tensore dal file */
    out->scale_bf16 = S.bf16;
    out->w_stride = (size_t)rows * wcols * 4u; out->s_stride = (size_t)rows * scols * 2u;
    out->map = W.map; out->map_bytes = W.map_bytes;
    out->map_s = S.map; out->map_s_bytes = S.map_bytes;
    out->map_b = has_b ? B.map : NULL; out->map_b_bytes = has_b ? B.map_bytes : 0;
    return 0;
}

void llm_mlx_packed_close3d(LlmMlxPackedW3* w) {
    if (!w || !w->map) return;
    munmap(w->map, w->map_bytes);
    if (w->map_s) munmap(w->map_s, w->map_s_bytes);
    if (w->map_b) munmap(w->map_b, w->map_b_bytes);
    memset(w, 0, sizeof *w);
}
