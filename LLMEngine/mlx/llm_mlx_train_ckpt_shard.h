/* llm_mlx_train_ckpt_shard.h — §D R8: checkpoint SHARDED (adapter su piu' file + manifest).
 *
 * Estende llm_mlx_train_ckpt.h: il payload (A,B + stato AdamW mA/vA/mB/vB) e' spezzato in
 * `n_shards` file `.partN` di byte contigui; un MANIFEST testuale tiene i metadati (fingerprint,
 * sizes, rng, t/steps) e la lista dei file con la dimensione di ciascuno. `read` riassembla e
 * verifica magic + sizes. Scrittura atomica per shard (.tmp+rename) e per il manifest.
 * PURO. */
#ifndef LLM_MLX_TRAIN_CKPT_SHARD_H
#define LLM_MLX_TRAIN_CKPT_SHARD_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm_mlx_adamw.h"

#define LLM_SHK_MAGIC 0x324B4853u  /* "SHK2" */

static inline int llm_ckpt_shard_write(const char* manifest, const char* prefix, int n_shards,
                                       const char* fp, const float* A, int An, const float* B, int Bn,
                                       const LlmAdamW* oA, const LlmAdamW* oB, uint64_t rng_state) {
    if (!manifest || !prefix || !A || !B || !oA || !oB || n_shards < 1 || An <= 0 || Bn <= 0) return -1;
    if (oA->n != An || oB->n != Bn) return -1;
    /* payload = [A][B][mA][vA][mB][vB] */
    size_t total = (size_t)(An + Bn + An + An + Bn + Bn) * sizeof(float);
    const float* seg[6]; size_t segn[6];
    seg[0]=A; segn[0]=An; seg[1]=B; segn[1]=Bn; seg[2]=oA->m; segn[2]=An;
    seg[3]=oA->v; segn[3]=An; seg[4]=oB->m; segn[4]=Bn; seg[5]=oB->v; segn[5]=Bn;
    size_t per = (total + (size_t)n_shards - 1) / (size_t)n_shards;
    /* scrivi i file .partN (ognuno = header + chunk) in modo atomico */
    size_t written = 0; int si = 0;
    char names[256][512]; size_t sizes[256];
    while (written < total && si < n_shards) {
        size_t chunk = (total - written) < per ? (total - written) : per;
        char path[600], tmp[640]; snprintf(path, sizeof(path), "%s.part%d", prefix, si);
        snprintf(tmp, sizeof(tmp), "%s.tmp", path);
        FILE* f = fopen(tmp, "wb"); if (!f) return -1;
        uint32_t hdr[2] = { LLM_SHK_MAGIC, (uint32_t)chunk };
        int ok = fwrite(hdr, 4, 2, f) == 2;
        /* copia il chunk dai segmenti (con offset) */
        size_t off = written, n = chunk;
        while (ok && n > 0) {
            size_t so = 0; int k = 0; size_t cum = 0;
            for (k = 0; k < 6; k++) { if (off < cum + segn[k] * 4) break; cum += segn[k] * 4; }
            if (k >= 6) break;
            so = off - cum; size_t avail = segn[k] * 4 - so; size_t take = n < avail ? n : avail;
            ok = fwrite((const char*)seg[k] + so, 1, take, f) == take;
            off += take; n -= take;
        }
        fclose(f);
        if (!ok) { remove(tmp); return -1; }
        if (rename(tmp, path) != 0) { remove(tmp); return -1; }
        snprintf(names[si], sizeof(names[si]), "%s.part%d", prefix, si);
        sizes[si] = chunk;
        written += chunk; si++;
    }
    /* manifest atomico */
    char mt[640]; snprintf(mt, sizeof(mt), "%s.tmp", manifest);
    FILE* m = fopen(mt, "w"); if (!m) return -1;
    fprintf(m, "LLM_SHARD %u %d %d %d %llu %ld %ld %ld %ld %zu %d\n",
            LLM_SHK_MAGIC, An, Bn, si, (unsigned long long)rng_state, oA->t, oA->steps, oB->t, oB->steps, total,
            (fp ? (int)strlen(fp) : 0));
    if (fp) fprintf(m, "FP %s\n", fp);
    for (int k = 0; k < si; k++) fprintf(m, "PART %s %zu\n", names[k], sizes[k]);
    fclose(m);
    return (rename(mt, manifest) == 0) ? 0 : -1;
}

static inline int llm_ckpt_shard_read(const char* manifest, const char* expect_fp,
                                      float* A, int An, float* B, int Bn,
                                      LlmAdamW* oA, LlmAdamW* oB, uint64_t* rng_state) {
    if (!manifest || !A || !B || !oA || !oB) return -1;
    FILE* m = fopen(manifest, "r"); if (!m) return -1;
    char line[1024];
    if (!fgets(line, sizeof(line), m)) { fclose(m); return -1; }
    unsigned magic; int a_n, b_n, nsh; unsigned long long rng; long tA, sA, tB, sB; size_t total; int fplen;
    if (sscanf(line, "LLM_SHARD %u %d %d %d %llu %ld %ld %ld %ld %zu %d",
               &magic, &a_n, &b_n, &nsh, &rng, &tA, &sA, &tB, &sB, &total, &fplen) != 11) { fclose(m); return -1; }
    if (magic != LLM_SHK_MAGIC || a_n != An || b_n != Bn) { fclose(m); return -1; }
    char fp[256] = ""; if (fplen > 0) { if (!fgets(line, sizeof(line), m)) { fclose(m); return -1; }
        if (strncmp(line, "FP ", 3) == 0) { snprintf(fp, sizeof(fp), "%s", line + 3); size_t l = strlen(fp); while (l && (fp[l-1]=='\n'||fp[l-1]=='\r')) fp[--l]=0; } }
    if (expect_fp && strcmp(fp, expect_fp) != 0) { fclose(m); return -1; }
    /* assembla in un buffer contiguo */
    uint8_t* buf = (uint8_t*)malloc(total); if (!buf) { fclose(m); return -1; }
    size_t off = 0;
    for (int k = 0; k < nsh; k++) {
        char part[600]; size_t sz = 0;
        if (!fgets(line, sizeof(line), m)) { free(buf); fclose(m); return -1; }
        if (sscanf(line, "PART %599s %zu", part, &sz) != 2) { free(buf); fclose(m); return -1; }
        FILE* p = fopen(part, "rb"); if (!p) { free(buf); fclose(m); return -1; }
        uint32_t hdr[2]; if (fread(hdr, 4, 2, p) != 2 || hdr[0] != LLM_SHK_MAGIC || hdr[1] != sz) { fclose(p); free(buf); fclose(m); return -1; }
        if (off + sz > total || fread(buf + off, 1, sz, p) != sz) { fclose(p); free(buf); fclose(m); return -1; }
        fclose(p); off += sz;
    }
    fclose(m);
    if (off != total) { free(buf); return -1; }
    memcpy(A, buf, (size_t)An * 4);
    memcpy(B, buf + (size_t)An * 4, (size_t)Bn * 4);
    memcpy(oA->m, buf + (size_t)(An + Bn) * 4, (size_t)An * 4);
    memcpy(oA->v, buf + (size_t)(2 * An + Bn) * 4, (size_t)An * 4);
    memcpy(oB->m, buf + (size_t)(3 * An + Bn) * 4, (size_t)Bn * 4);
    memcpy(oB->v, buf + (size_t)(3 * An + 2 * Bn) * 4, (size_t)Bn * 4);
    free(buf);
    oA->t = tA; oA->steps = sA; oB->t = tB; oB->steps = sB;
    if (rng_state) *rng_state = rng;
    return 0;
}

#endif /* LLM_MLX_TRAIN_CKPT_SHARD_H */
