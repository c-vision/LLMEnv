/* llm_mlx_bonsai_hadamard.h — contratto Hadamard del pack PrismML (Ternary-Bonsai-2-27B).
 *
 * PERCHE'. Il pack `Ternary-Bonsai-2-27B-mlx-2bit` NON e' un checkpoint affine Qwen ordinario:
 * i pesi elencati in `hadamard.json` sono **folded** in una basis ruotata, e vanno de-ruotati
 * applicando la trasformata alle ATTIVAZIONI in ingresso (de-ruotare i pesi richiederebbe
 * l'unpack a f16: ~54 GB, non praticabile). Il README del pack lo dice esplicitamente:
 * "Do not load as an ordinary affine Qwen checkpoint: that would omit activation and
 * inverse-embedding transforms."
 *
 * CONTRATTO (letto da `hadamard.json` del pack, verificato 2026-09-23):
 *   prism.hadamard.block_size   = 1024
 *   prism.hadamard.transform    = normalized-sylvester-walsh-hadamard  (== llm_mlx_hadamard_rotate)
 *   prism.hadamard.axis         = input-last-dimension
 *   prism.hadamard.sign_mode    = explicit
 *   prism.hadamard.weight_names = 401 nomi (lm_head + proiezioni lineari) -> trasformata ATTIVA
 *   prism.hadamard.inverse_weight_names = ["...embed_tokens.weight"]   -> trasformata INVERSA
 *   prism.hadamard.sign_widths  = [5120, 6144, 17408]  -> un vettore di ±1 per larghezza
 *   prism.hadamard.sign_values  = 28672 valori ±1 (= somma delle width, in quell'ordine)
 *   prism.hadamard.gdn_v_grouped = true  (conferma: il layout GDN e' "grouped", non permutare)
 *
 * ORIENTAMENTO DEI SEGNI (NON ancora stabilito a tavolino): R puo' essere H·diag(s) (segni sulle
 * colonne = dim di input, `signs_right=true`) oppure diag(s)·H (segni sulle righe). Entrambe sono
 * ortogonali, quindi il test di ortogonalita' NON discrimina: la scelta va fatta **empiricamente
 * col modello** (produzione coerente + NLL plausibile). Qui si costruiscono entrambe.
 * Nota utile: H e' SIMMETRICA, quindi R^T = (H·diag(s))^T = diag(s)·H -> la trasposta equivale a
 * scambiare l'orientamento. Per questo `invert` e' implementato come XOR dell'orientamento.
 *
 * IMPLEMENTAZIONE: funzioni `static` in header (convenzione del motore, come llm_mlx_ops.h), per
 * non toccare CMake. Richiede `<cJSON.h>` gia' incluso dal chiamante (il motore lo include).
 */
#ifndef LLM_MLX_BONSAI_HADAMARD_H
#define LLM_MLX_BONSAI_HADAMARD_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mlx/c/mlx.h>

#define LLM_BONSAI_MAX_WIDTHS 8
#define LLM_BONSAI_MAX_NAMES  512
#define LLM_BONSAI_CACHE      8   /* (width, orientamento) */

typedef struct LlmBonsaiHadamard {
    int   block;
    int   n_widths;
    int   widths[LLM_BONSAI_MAX_WIDTHS];
    float* signs[LLM_BONSAI_MAX_WIDTHS];
    int   gdn_v_grouped;
    char* names[LLM_BONSAI_MAX_NAMES];
    int   n_names;
    char* inv_names[LLM_BONSAI_MAX_NAMES];
    int   n_inv;
} LlmBonsaiHadamard;

static void llm_bonsai_hadamard_free(LlmBonsaiHadamard* h) {
    if (h == NULL) return;
    for (int i = 0; i < h->n_widths; i++) { free(h->signs[i]); h->signs[i] = NULL; }
    for (int i = 0; i < h->n_names; i++) { free(h->names[i]); h->names[i] = NULL; }
    for (int i = 0; i < h->n_inv; i++) { free(h->inv_names[i]); h->inv_names[i] = NULL; }
    memset(h, 0, sizeof(*h));
}

/* Legge <model_dir>/hadamard.json. Ritorna 0 su successo, -1 su errore. */
static int llm_bonsai_hadamard_load(const char* model_dir, LlmBonsaiHadamard* h) {
    if (model_dir == NULL || h == NULL) return -1;
    memset(h, 0, sizeof(*h));
    char path[1200];
    snprintf(path, sizeof(path), "%s/hadamard.json", model_dir);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return -1; }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (buf == NULL) { fclose(f); return -1; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;
    cJSON* j = cJSON_Parse(buf);
    free(buf);
    if (j == NULL) return -1;

    cJSON* jb = cJSON_GetObjectItemCaseSensitive(j, "prism.hadamard.block_size");
    cJSON* jw = cJSON_GetObjectItemCaseSensitive(j, "prism.hadamard.sign_widths");
    cJSON* jv = cJSON_GetObjectItemCaseSensitive(j, "prism.hadamard.sign_values");
    cJSON* jn = cJSON_GetObjectItemCaseSensitive(j, "prism.hadamard.weight_names");
    cJSON* ji = cJSON_GetObjectItemCaseSensitive(j, "prism.hadamard.inverse_weight_names");
    cJSON* jg = cJSON_GetObjectItemCaseSensitive(j, "prism.hadamard.gdn_v_grouped");
    if (!cJSON_IsNumber(jb) || !cJSON_IsArray(jw) || !cJSON_IsArray(jv)) { cJSON_Delete(j); return -1; }
    h->block = (int)jb->valuedouble;
    h->gdn_v_grouped = cJSON_IsTrue(jg) ? 1 : 0;

    /* sign_widths + sign_values (flat, concatenati nell'ordine delle width) */
    int off = 0;
    int nw = cJSON_GetArraySize(jw);
    if (nw > LLM_BONSAI_MAX_WIDTHS) nw = LLM_BONSAI_MAX_WIDTHS;
    for (int i = 0; i < nw; i++) {
        cJSON* e = cJSON_GetArrayItem(jw, i);
        if (!cJSON_IsNumber(e)) { llm_bonsai_hadamard_free(h); cJSON_Delete(j); return -1; }
        int w = (int)e->valuedouble;
        h->widths[i] = w;
        h->signs[i] = (float*)malloc((size_t)w * sizeof(float));
        if (h->signs[i] == NULL) { llm_bonsai_hadamard_free(h); cJSON_Delete(j); return -1; }
        for (int k = 0; k < w; k++) {
            cJSON* s = cJSON_GetArrayItem(jv, off + k);
            if (!cJSON_IsNumber(s)) { llm_bonsai_hadamard_free(h); cJSON_Delete(j); return -1; }
            h->signs[i][k] = (float)s->valuedouble;
        }
        off += w;
    }
    h->n_widths = nw;

    /* weight_names / inverse_weight_names */
    int nn = cJSON_GetArraySize(jn);
    for (int i = 0; i < nn && h->n_names < LLM_BONSAI_MAX_NAMES; i++) {
        cJSON* e = cJSON_GetArrayItem(jn, i);
        if (!cJSON_IsString(e) || e->valuestring == NULL) continue;
        h->names[h->n_names] = strdup(e->valuestring);
        if (h->names[h->n_names] != NULL) h->n_names++;
    }
    if (cJSON_IsArray(ji)) {
        int ni = cJSON_GetArraySize(ji);
        for (int i = 0; i < ni && h->n_inv < LLM_BONSAI_MAX_NAMES; i++) {
            cJSON* e = cJSON_GetArrayItem(ji, i);
            if (!cJSON_IsString(e) || e->valuestring == NULL) continue;
            h->inv_names[h->n_inv] = strdup(e->valuestring);
            if (h->inv_names[h->n_inv] != NULL) h->n_inv++;
        }
    }
    cJSON_Delete(j);
    return 0;
}

/* 0 = non trasformato, 1 = trasformata attiva, -1 = inversa (embed). */
static int llm_bonsai_hadamard_kind(const LlmBonsaiHadamard* h, const char* name) {
    if (h == NULL || name == NULL) return 0;
    for (int i = 0; i < h->n_inv; i++) if (strcmp(h->inv_names[i], name) == 0) return -1;
    for (int i = 0; i < h->n_names; i++) if (strcmp(h->names[i], name) == 0) return 1;
    return 0;
}

static const float* llm_bonsai_hadamard_signs_for(const LlmBonsaiHadamard* h, int width) {
    if (h == NULL) return NULL;
    for (int i = 0; i < h->n_widths; i++) if (h->widths[i] == width) return h->signs[i];
    return NULL;
}

/* Matrice R [block,block] = H_block (Sylvester normalizzata) con i sign sulle colonne
 * (signs_right) o sulle righe. Cache per (width, orientamento): la matrice e' la stessa per
 * tutte le width uguali e dipende solo dai sign. */
static mlx_array llm_bonsai_R_cache[LLM_BONSAI_CACHE];
static int llm_bonsai_R_key[LLM_BONSAI_CACHE];
static int llm_bonsai_R_n = 0;

static mlx_array llm_bonsai_R_get(int width, int block, const float* signs, bool signs_right) {
    int key = width * 2 + (signs_right ? 1 : 0);
    for (int i = 0; i < llm_bonsai_R_n; i++)
        if (llm_bonsai_R_key[i] == key) return llm_bonsai_R_cache[i];
    if (llm_bonsai_R_n >= LLM_BONSAI_CACHE) return mlx_array_new();   /* pieno: no-op */
    int B = block;
    if (B <= 0 || (B & (B - 1)) != 0 || width % B != 0) return mlx_array_new();
    float* m = (float*)malloc((size_t)B * (size_t)B * sizeof(float));
    if (m == NULL) return mlx_array_new();
    const float inv = 1.0f / sqrtf((float)B);
    /* Ordinamento: il contratto dice "sylvester-WALSH": il termine Walsh puo' indicare l'ordine
     * per SEQUENCY (righe permutate) invece del natural. Variante da LLM_BONSAI_HAD_ORDER. */
    static int s_order = -1;
    if (s_order < 0) {
        const char* e = getenv("LLM_BONSAI_HAD_ORDER");
        s_order = (e == NULL || strcmp(e, "natural") == 0) ? 0
                 : (strcmp(e, "bitrev") == 0 ? 1 : (strcmp(e, "gray") == 0 ? 2 : 3));
    }
    int bits = 0; while ((1 << bits) < B) bits++;
    for (int i = 0; i < B; i++) {
        int ri = i;
        if (s_order == 1) { int r = 0; for (int b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b); ri = r; }
        else if (s_order == 2) { ri = i ^ (i >> 1); }
        else if (s_order == 3) { int r = 0; for (int b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b); ri = i ^ (r >> 1); }
        for (int j = 0; j < B; j++) {
            int v = ri & j, pc = 0;
            while (v) { pc ^= (v & 1); v >>= 1; }
            float hh = (pc ? -1.0f : 1.0f) * inv;
            /* NIENTE segni qui: i sign del pack variano PER BLOCCO; applicarli dentro una sola
             * matrice [B,B] li replicherebbe dal primo blocco (bug: h somma 94 vs reference 52).
             * I segni vanno applicati ELEMENTWISE sull'intero vettore (vedi rotate). */
            m[(size_t)i * B + j] = hh;
        }
    }
    int shp[2] = {B, B};
    mlx_array R = mlx_array_new_data(m, shp, 2, MLX_FLOAT32);
    free(m);
    if (R.ctx == NULL) return mlx_array_new();
    llm_bonsai_R_cache[llm_bonsai_R_n] = R;
    llm_bonsai_R_key[llm_bonsai_R_n] = key;
    llm_bonsai_R_n++;
    return R;
}

/* Ruota x [..., width] con la R (block-diagonale a blocchi `block`). `invert` usa R^T.
 * Ritorna false (no-op, out invariato) su input non validi. */
static bool llm_bonsai_hadamard_rotate(mlx_array* out, mlx_array x, int width, int block,
                                       const float* signs, bool signs_right, bool invert,
                                       mlx_stream stream) {
    if (out == NULL || signs == NULL) return false;
    if (width <= 0 || block <= 0 || width % block != 0) return false;
    if (mlx_array_ndim(x) < 1) return false;
    if (mlx_array_shape(x)[mlx_array_ndim(x) - 1] != width) return false;
    /* I sign variano per blocco: si applicano ELEMENTWISE sull'intero vettore (come il reference
     * `x = x * signs` prima della H per forward, `out = out * signs` dopo per inverse). La
     * matrice R e' la sola H normalizzata block-diagonale. */
    mlx_array R = llm_bonsai_R_get(width, block, signs, signs_right);
    if (R.ctx == NULL) return false;
    int sshape[1] = {width};
    mlx_array sg = mlx_array_new_data(signs, sshape, 1, MLX_FLOAT32);
    mlx_array xuse = mlx_array_new();
    bool ok = true;
    if (!invert) { ok = mlx_multiply(&xuse, x, sg, stream) == 0; }      /* forward: sign PRIMA */
    else { ok = mlx_array_set(&xuse, x) == 0; }
    int nblk = width / block;
    int xnd = mlx_array_ndim(x);
    const int* xs = mlx_array_shape(x);
    size_t total = mlx_array_size(x);
    int outer = (int)(total / (size_t)width);
    int rs[3] = {outer, nblk, block};
    mlx_array xr = mlx_array_new();
    if (ok) ok = mlx_reshape(&xr, xuse, rs, 3, stream) == 0;
    mlx_array y = mlx_array_new();
    if (ok) ok = mlx_matmul(&y, xr, R, stream) == 0;
    mlx_array yr = mlx_array_new();
    if (ok) {
        int back[8];
        if (xnd > 8) ok = false;
        else {
            for (int i = 0; i < xnd; i++) back[i] = xs[i];
            ok = mlx_reshape(&yr, y, back, xnd, stream) == 0;
        }
    }
    if (ok && invert) { ok = mlx_multiply(out, yr, sg, stream) == 0; }  /* inverse: sign DOPO */
    else if (ok) { ok = mlx_array_set(out, yr) == 0; }
    mlx_array_free(sg); mlx_array_free(xuse);
    mlx_array_free(xr);
    mlx_array_free(y);
    mlx_array_free(yr);
    return ok;
}

#endif /* LLM_MLX_BONSAI_HADAMARD_H */
