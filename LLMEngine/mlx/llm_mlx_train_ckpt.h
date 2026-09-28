/* llm_mlx_train_ckpt.h — §D R4.5: checkpoint COMPLETO ATOMICO e RIPRENDIBILE del training.
 *
 * Salva/ripristina: adapter LoRA (A,B), stato AdamW (m,v,t,steps) dei due optimizer, stato RNG,
 * e un FINGERPRINT (es. model_dir+config) per rifiutare un resume su un modello diverso.
 * Scrittura ATOMICA: scrive su `<path>.tmp`, fsync, poi rename() -> nessun file parziale.
 * Formato binario versionato + checksum FNV-1a del payload.
 * PURO (nessuna dipendenza MLX, oltre a llm_mlx_adamw.h). */
#ifndef LLM_MLX_TRAIN_CKPT_H
#define LLM_MLX_TRAIN_CKPT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm_mlx_adamw.h"

#define LLM_CKPT_MAGIC 0x3154504B43544D47ULL /* "GMTCKPT1" LE (8 byte) */
#define LLM_CKPT_VER   1u
#define LLM_CKPT_FPMAX 128

static inline uint64_t llm_ckpt_fnv1a(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

/* Salva il checkpoint in modo ATOMICO. `fp` = fingerprint (model_dir+config). 0 ok / -1. */
static inline int llm_train_ckpt_save(const char* path, const char* fp,
                                      const float* A, int An, const float* B, int Bn,
                                      const LlmAdamW* optA, const LlmAdamW* optB, uint64_t rng_state) {
    if (!path || !A || !B || !optA || !optB || An <= 0 || Bn <= 0) return -1;
    if (optA->n != An || optB->n != Bn) return -1;
    char tmp[1200]; snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return -1;
    uint8_t fpz[LLM_CKPT_FPMAX]; memset(fpz, 0, sizeof(fpz));
    if (fp) snprintf((char*)fpz, sizeof(fpz), "%s", fp);
    struct { uint64_t magic; uint32_t ver; int32_t An, Bn; uint64_t rng; int64_t tA, sA, tB, sB; } hdr = {
        LLM_CKPT_MAGIC, LLM_CKPT_VER, An, Bn, rng_state, optA->t, optA->steps, optB->t, optB->steps };
    int ok = fwrite(&hdr, sizeof(hdr), 1, f) == 1 && fwrite(fpz, 1, sizeof(fpz), f) == sizeof(fpz);
    if (ok) ok = fwrite(A, sizeof(float), (size_t)An, f) == (size_t)An;
    if (ok) ok = fwrite(B, sizeof(float), (size_t)Bn, f) == (size_t)Bn;
    if (ok) ok = fwrite(optA->m, sizeof(float), (size_t)An, f) == (size_t)An &&
                 fwrite(optA->v, sizeof(float), (size_t)An, f) == (size_t)An;
    if (ok) ok = fwrite(optB->m, sizeof(float), (size_t)Bn, f) == (size_t)Bn &&
                 fwrite(optB->v, sizeof(float), (size_t)Bn, f) == (size_t)Bn;
    if (ok) fflush(f);
    if (ok) { FILE* g = fopen(tmp, "rb"); /* checksum del payload (header escluso) */
        long off = (long)sizeof(hdr) + LLM_CKPT_FPMAX;
        fseek(g, off, SEEK_SET);
        uint8_t buf[65536]; size_t r; uint64_t h = 1469598103934665603ULL; size_t tot = 0;
        while ((r = fread(buf, 1, sizeof(buf), g)) > 0) { for (size_t i = 0; i < r; i++) { h ^= buf[i]; h *= 1099511628211ULL; } tot += r; }
        fclose(g);
        (void)tot; ok = fwrite(&h, sizeof(h), 1, f) == 1;
    }
    fclose(f);
    if (!ok) { remove(tmp); return -1; }
    if (rename(tmp, path) != 0) { remove(tmp); return -1; }   /* atomico */
    return 0;
}

/* Carica il checkpoint; verifica magic/versione/sizes/fingerprint/checksum. 0 ok / -1. */
static inline int llm_train_ckpt_load(const char* path, const char* fp,
                                      float* A, int An, float* B, int Bn,
                                      LlmAdamW* optA, LlmAdamW* optB, uint64_t* rng_state) {
    if (!path || !A || !B || !optA || !optB) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    struct { uint64_t magic; uint32_t ver; int32_t An, Bn; uint64_t rng; int64_t tA, sA, tB, sB; } hdr;
    uint8_t fpz[LLM_CKPT_FPMAX];
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || fread(fpz, 1, sizeof(fpz), f) != sizeof(fpz)) { fclose(f); return -1; }
    if (hdr.magic != LLM_CKPT_MAGIC || hdr.ver != LLM_CKPT_VER) { fclose(f); return -1; }
    if (hdr.An != An || hdr.Bn != Bn) { fclose(f); return -1; }                       /* shape mismatch */
    if (fp) { char fpz2[LLM_CKPT_FPMAX]; memset(fpz2, 0, sizeof(fpz2)); snprintf(fpz2, sizeof(fpz2), "%s", fp);
              if (memcmp(fpz, fpz2, sizeof(fpz)) != 0) { fclose(f); return -1; } }     /* modello diverso */
    long dstart = (long)(sizeof(hdr) + LLM_CKPT_FPMAX);
    int ok = fread(A, sizeof(float), (size_t)An, f) == (size_t)An;
    if (ok) ok = fread(B, sizeof(float), (size_t)Bn, f) == (size_t)Bn;
    if (ok) ok = fread(optA->m, sizeof(float), (size_t)An, f) == (size_t)An && fread(optA->v, sizeof(float), (size_t)An, f) == (size_t)An;
    if (ok) ok = fread(optB->m, sizeof(float), (size_t)Bn, f) == (size_t)Bn && fread(optB->v, sizeof(float), (size_t)Bn, f) == (size_t)Bn;
    uint64_t sum = 0;
    if (ok) ok = fread(&sum, sizeof(sum), 1, f) == 1;
    fclose(f);
    if (!ok) return -1;
    /* checksum del payload */
    f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, dstart, SEEK_SET);
    uint8_t buf[65536]; size_t r; uint64_t h = 1469598103934665603ULL;
    size_t pay = (size_t)(3 * An + 3 * Bn) * sizeof(float);   /* A,B,mA,vA,mB,vB (checksum escluso) */
    size_t left = pay;
    while (left > 0 && (r = fread(buf, 1, left < sizeof(buf) ? left : sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < r; i++) { h ^= buf[i]; h *= 1099511628211ULL; }
        left -= r;
    }
    fclose(f);
    if (h != sum) return -1;
    optA->t = hdr.tA; optA->steps = hdr.sA;
    optB->t = hdr.tB; optB->steps = hdr.sB;
    if (rng_state) *rng_state = hdr.rng;
    return 0;
}

#endif /* LLM_MLX_TRAIN_CKPT_H */
