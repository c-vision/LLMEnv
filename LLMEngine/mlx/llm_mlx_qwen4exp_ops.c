/* ============================================================================
 * llm_mlx_qwen4exp_ops.c — dominio Qwen4-Exp operativo: hyper-connection residual,
 * tabella n-gram (open/shard/read), PLE inject e dispatch attn/ffn. Estratto VERBATIM
 * da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_qwen4_hc_packed.h"
#include <cJSON.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>

/* Come llm_mlx_gated_residual (llm_mlx_qwen4exp.h) ma con down/up/inject caricati via
 * quantized_linear_cfg invece di un matmul plain -- VERIFICATO sul checkpoint reale
 * (2026-09-06, ~/ai/Models/Qwen3.8-Flash-Next-MLX-Serve-4bit/model.safetensors.index.json):
 * `attn_hyper_connection.input_mix_weight_down/up` HANNO `.scales`/`.biases` (quantizzati,
 * la versione plain con multishard_get semplice avrebbe letto il tensore U32 impacchettato
 * come se fosse gia' float, numeri sbagliati SENZA errore); `block_inject_weight` invece ha
 * SOLO `.weight` (non quantizzato). multishard_get_triple gia' tollera scales/biases
 * assenti (li lascia ctx==NULL) e quantized_linear_cfg instrada automaticamente a
 * plain_linear in quel caso -- quindi qui si usa lo STESSO schema per tutti e tre invece di
 * doverli distinguere caso per caso; funziona per costruzione sia se un peso e' quantizzato
 * sia se non lo e'. Duplica la formula di llm_mlx_gated_residual (down/up sono
 * sequenziali -- l'input di "up" e' l'output di "down" dopo silu, quindi non si possono
 * precalcolare entrambi fuori e passarli come tensori densi come fatto per q/k/v in
 * qwen4_full_qsa_block) invece di refactorizzare l'header, per non rischiare la versione
 * gia' verificata contro numpy e usata dai test standalone. */
bool qwen4_exp_gated_residual_q(LlmMlxModel* m, mlx_array hyper_input, mlx_array hc_norm_w,
                                        mlx_array down_w, mlx_array down_s, mlx_array down_b,
                                        mlx_array up_w, mlx_array up_s, mlx_array up_b,
                                        mlx_array inject_w, mlx_array inject_s, mlx_array inject_b,
                                        mlx_array* out_mixed, mlx_array* out_residual, mlx_array* out_injection) {
    mlx_stream s = m->stream;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int hc_count = m->cfg.hc_count, hidden = m->cfg.hidden_size;
    const int* hshape = mlx_array_shape(hyper_input);
    int hndim = mlx_array_ndim(hyper_input);
    if (hshape == NULL || hndim <= 0 || hshape[hndim - 1] != hc_count * hidden) {
        /* Check silenzioso (bug 2026-09-17, AiWiki "MLX-Qwen4Exp-Attn-HyperConnection-Mix-Failed"):
         * senza questo log la diagnosi richiedeva rileggere il codice per ogni divergenza di
         * shape -- l'ultima causa (path LLM_ASYNC_DECODE senza il tile hc_count) e' stata trovata
         * cosi'. */
        fprintf(stderr, "qwen4_exp_gated_residual_q: mismatch shape hyper_input (hndim=%d, ultimo asse=%d, "
                        "atteso hc_count*hidden=%d*%d=%d)\n",
                hndim, (hshape != NULL && hndim > 0) ? hshape[hndim - 1] : -1, hc_count, hidden, hc_count * hidden);
        return false;
    }

    mlx_array normed = mlx_array_new();
    bool ok = llm_mlx_zero_centered_rmsnorm(&normed, hyper_input, hc_norm_w, hidden, m->cfg.rms_norm_eps, s);
    mlx_array down_out = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&down_out, normed, down_w, down_s, down_b, gsz, bts, m->cfg.quant_mode, s);
    mlx_array hcf = mlx_array_new_float32((float)hc_count);
    mlx_array down_scaled = mlx_array_new();
    if (ok) ok = mlx_divide(&down_scaled, down_out, hcf, s) == 0;
    mlx_array down_silu = mlx_array_new();
    if (ok) ok = llm_mlx_qwen4exp_silu(&down_silu, down_scaled, s);
    mlx_array up_out = mlx_array_new();
    if (ok) ok = quantized_linear_cfg(&up_out, down_silu, up_w, up_s, up_b, gsz, bts, m->cfg.quant_mode, s);
    mlx_array mix = mlx_array_new();
    if (ok) ok = mlx_sigmoid(&mix, up_out, s) == 0;

    int* mshape = NULL;
    mlx_array mix_r = mlx_array_new(), normed_r = mlx_array_new();
    if (ok) { mshape = (int*)malloc((size_t)(hndim + 1) * sizeof(int)); if (mshape == NULL) ok = false; }
    if (ok) {
        for (int i = 0; i < hndim - 1; i++) mshape[i] = hshape[i];
        mshape[hndim - 1] = hc_count; mshape[hndim] = hidden;
        ok = mlx_reshape(&mix_r, mix, mshape, hndim + 1, s) == 0 &&
             mlx_reshape(&normed_r, normed, mshape, hndim + 1, s) == 0;
    }
    mlx_array prod = mlx_array_new();
    if (ok) ok = mlx_multiply(&prod, mix_r, normed_r, s) == 0;
    if (ok) ok = mlx_mean_axis(out_mixed, prod, -2, /*keepdims=*/false, s) == 0;

    if (ok && inject_w.ctx != NULL && out_residual != NULL && out_injection != NULL) {
        mlx_array inj_out = mlx_array_new();
        ok = quantized_linear_cfg(&inj_out, normed, inject_w, inject_s, inject_b, gsz, bts, m->cfg.quant_mode, s);
        mlx_array inj_scaled = mlx_array_new();
        if (ok) ok = mlx_divide(&inj_scaled, inj_out, hcf, s) == 0;
        mlx_array inj_sig = mlx_array_new();
        if (ok) ok = mlx_sigmoid(&inj_sig, inj_scaled, s) == 0;
        mlx_array two = mlx_array_new_float32(2.0f);
        if (ok) ok = mlx_multiply(out_injection, inj_sig, two, s) == 0;
        if (ok) ok = mlx_array_set(out_residual, hyper_input) == 0;
        mlx_array_free(inj_out); mlx_array_free(inj_scaled); mlx_array_free(inj_sig); mlx_array_free(two);
    }

    if (mshape) free(mshape);
    mlx_array_free(normed); mlx_array_free(down_out); mlx_array_free(hcf);
    mlx_array_free(down_scaled); mlx_array_free(down_silu); mlx_array_free(up_out);
    mlx_array_free(mix); mlx_array_free(mix_r); mlx_array_free(normed_r); mlx_array_free(prod);
    return ok;
}

/* Mappa una stringa dtype safetensors (es. "U32", "BF16") sul dtype/itemsize mlx-c
 * corrispondente. Copre solo i tipi che compaiono realisticamente in un file pesi
 * (interi per dati quantizzati, float per scale/bias) -- non l'intero standard
 * safetensors (niente BOOL/COMPLEX qui, non hanno senso per questo uso). */
static bool st_dtype_info(const char* dt, mlx_dtype* out_dtype, int* out_itemsize) {
    if (strcmp(dt, "U8") == 0)   { *out_dtype = MLX_UINT8;   *out_itemsize = 1; return true; }
    if (strcmp(dt, "I8") == 0)   { *out_dtype = MLX_INT8;    *out_itemsize = 1; return true; }
    if (strcmp(dt, "U16") == 0)  { *out_dtype = MLX_UINT16;  *out_itemsize = 2; return true; }
    if (strcmp(dt, "I16") == 0)  { *out_dtype = MLX_INT16;   *out_itemsize = 2; return true; }
    if (strcmp(dt, "F16") == 0)  { *out_dtype = MLX_FLOAT16; *out_itemsize = 2; return true; }
    if (strcmp(dt, "BF16") == 0) { *out_dtype = MLX_BFLOAT16;*out_itemsize = 2; return true; }
    if (strcmp(dt, "U32") == 0)  { *out_dtype = MLX_UINT32;  *out_itemsize = 4; return true; }
    if (strcmp(dt, "I32") == 0)  { *out_dtype = MLX_INT32;   *out_itemsize = 4; return true; }
    if (strcmp(dt, "F32") == 0)  { *out_dtype = MLX_FLOAT32; *out_itemsize = 4; return true; }
    if (strcmp(dt, "U64") == 0)  { *out_dtype = MLX_UINT64;  *out_itemsize = 8; return true; }
    if (strcmp(dt, "I64") == 0)  { *out_dtype = MLX_INT64;   *out_itemsize = 8; return true; }
    if (strcmp(dt, "F64") == 0)  { *out_dtype = MLX_FLOAT64; *out_itemsize = 8; return true; }
    return false;
}

/* Apre `ngram_table.bin` SENZA mai caricarlo con mlx_load_safetensors: parsa a mano il
 * piccolo header JSON safetensors (8 byte di lunghezza LE + JSON, tipicamente poche
 * centinaia di byte anche per un file da 30GB -- l'header non contiene i dati) per
 * ricavare offset/shape/dtype di weight/scales/biases, poi tiene il file aperto per
 * tutta la vita del modello: le righe si leggono on-demand con pread() in
 * qwen4_exp_ngram_read_rows, mai l'array intero. Ritorna false (fd chiuso, ngram_fd=-1)
 * su QUALUNQUE anomalia -- il chiamante tratta "non caricato" come oggi: i layer PLE si
 * rifiutano esplicitamente invece di usare dati mancanti. */
bool qwen4_exp_ngram_table_open(const char* path, LlmMlxModel* m) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;

    uint64_t header_len = 0;
    if (pread(fd, &header_len, sizeof(header_len), 0) != (ssize_t)sizeof(header_len)) { close(fd); return false; }
    if (header_len == 0 || header_len > (64u * 1024u * 1024u)) { close(fd); return false; } /* sanity: header vero e' minuscolo */

    char* hbuf = (char*)malloc((size_t)header_len + 1);
    if (hbuf == NULL) { close(fd); return false; }
    ssize_t hread = pread(fd, hbuf, (size_t)header_len, (off_t)sizeof(header_len));
    if (hread != (ssize_t)header_len) { free(hbuf); close(fd); return false; }
    hbuf[header_len] = '\0';

    cJSON* j = cJSON_ParseWithLength(hbuf, header_len);
    free(hbuf);
    if (j == NULL) { close(fd); return false; }

    bool ok = true;
    int64_t data_start = (int64_t)sizeof(header_len) + (int64_t)header_len;

#define ST_TENSOR(key, off_field, cols_field, itemsize_field, dtype_field, required) do { \
        cJSON* t = cJSON_GetObjectItemCaseSensitive(j, key); \
        if (t == NULL) { if (required) ok = false; } \
        else { \
            cJSON* dtype_j = cJSON_GetObjectItemCaseSensitive(t, "dtype"); \
            cJSON* shape_j = cJSON_GetObjectItemCaseSensitive(t, "shape"); \
            cJSON* offs_j = cJSON_GetObjectItemCaseSensitive(t, "data_offsets"); \
            if (!cJSON_IsString(dtype_j) || !cJSON_IsArray(shape_j) || cJSON_GetArraySize(shape_j) != 2 || \
                !cJSON_IsArray(offs_j) || cJSON_GetArraySize(offs_j) < 1) { ok = false; } \
            else { \
                mlx_dtype dt; int isz; \
                if (!st_dtype_info(dtype_j->valuestring, &dt, &isz)) { ok = false; } \
                else { \
                    m->dtype_field = dt; m->itemsize_field = isz; \
                    m->cols_field = cJSON_GetArrayItem(shape_j, 1)->valueint; \
                    m->off_field = (int64_t)cJSON_GetArrayItem(offs_j, 0)->valuedouble; \
                } \
            } \
        } \
    } while (0)

    ST_TENSOR("weight", ngram_w_off, ngram_w_cols, ngram_w_itemsize, ngram_w_dtype, true);
    ST_TENSOR("scales", ngram_s_off, ngram_s_cols, ngram_s_itemsize, ngram_s_dtype, true);
    ST_TENSOR("biases", ngram_b_off, ngram_b_cols, ngram_b_itemsize, ngram_b_dtype, false);
#undef ST_TENSOR
    m->ngram_has_biases = ok && cJSON_GetObjectItemCaseSensitive(j, "biases") != NULL;

    const char* bits_s = NULL; const char* gsz_s = NULL;
    cJSON* meta = cJSON_GetObjectItemCaseSensitive(j, "__metadata__");
    if (meta != NULL) {
        cJSON* bits_j = cJSON_GetObjectItemCaseSensitive(meta, "bits");
        cJSON* gsz_j = cJSON_GetObjectItemCaseSensitive(meta, "group_size");
        if (cJSON_IsString(bits_j)) bits_s = bits_j->valuestring;
        if (cJSON_IsString(gsz_j)) gsz_s = gsz_j->valuestring;
    }
    m->ngram_table_bits = bits_s ? atoi(bits_s) : 4;
    m->ngram_table_group_size = gsz_s ? atoi(gsz_s) : 64;

    cJSON_Delete(j);
    if (!ok) { close(fd); return false; }

    m->ngram_fd = fd;
    m->ngram_data_start = data_start;
    return true;
}

/* Apre la tabella N-gram/PLE dal checkpoint UFFICIALE mlx-community, dove NON esiste un
 * `ngram_table.bin` monolitico: la stessa tabella e' inline nei safetensors principali,
 * divisa row-wise in 128 tensori `...ngram_embedding.shards.N.{weight,scales,biases}`
 * (shard = row // rows_per_shard, come la ShardedEmbedding di riferimento). Qui si legge
 * SOLO l'index e gli header safetensors (piccoli) per costruire una mappa file/offset per
 * shard; i byte delle righe restano su disco e vengono letti via pread in
 * qwen4_exp_ngram_read_rows -- MAI l'array intero (32GB) in RAM. Ritorna false senza side
 * effects se gli shard mancano o non sono coerenti (shape/dtype misti): il chiamante
 * tratta "non caricato" come oggi. */
/* qparams della tabella PLE dal config PER-MODULO (es. shard a 4-bit mentre il top e' 2-bit). */
static void ngram_qp_from_config(const char* dir, const char* prefix, int* bits, int* group) {
    char pth[1200]; snprintf(pth, sizeof pth, "%s/config.json", dir);
    FILE* f = fopen(pth, "rb"); if (!f) return;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (1<<24)) { fclose(f); return; }
    char* js = (char*)malloc((size_t)sz + 1); if (!js) { fclose(f); return; }
    size_t rd = fread(js, 1, (size_t)sz, f); js[rd] = 0; fclose(f);
    const char* q = strstr(js, "\"quantization\"");
    if (!q) { free(js); return; }
    const char* v = strchr(q, ':'); if (!v) { free(js); return; }
    ++v; while (*v==' '||*v=='\t'||*v=='\n'||*v=='\r') ++v;
    if (*v != '{') { free(js); return; }
    const char* p = v + 1;
    while (*p && *p != '}') {
        while (*p && (*p==' '||*p==','||*p=='\t'||*p=='\n'||*p=='\r')) ++p;
        if (*p != '"') { p++; continue; }
        const char* ke = strchr(p+1, '"'); if (!ke) break;
        size_t kl = (size_t)(ke - p - 1);
        const char* c = strchr(ke, ':'); if (!c) break;
        const char* o = c + 1; while (*o==' '||*o=='\t'||*o=='\n'||*o=='\r') ++o;
        if (*o != '{') { const char* np=o; if(*np=='"'){const char* q2=strchr(np+1,'"');np=q2?q2+1:np+1;} p=np; continue; }
        const char* oe = o; int dep = 0;
        for (; *oe; ++oe) { if (*oe=='{') dep++; else if (*oe=='}') { dep--; if (!dep) break; } }
        if (kl > strlen(prefix) && strncmp(p+1, prefix, strlen(prefix)) == 0) {
            const char* bx = strstr(o, "\"bits\""); const char* gx = strstr(o, "\"group_size\"");
            if (bx && bx < oe) { const char* cc = strchr(bx, ':'); if (cc) *bits = atoi(cc+1); }
            if (gx && gx < oe) { const char* cc = strchr(gx, ':'); if (cc) *group = atoi(cc+1); }
            free(js); return;
        }
        p = oe + 1;
    }
    free(js);
}

bool qwen4_exp_ngram_table_open_shards(const char* model_dir, LlmMlxModel* m) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", model_dir);
    FILE* fp = fopen(path, "rb");
    if (fp == NULL) return false;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return false; }
    long fsz = ftell(fp);
    if (fsz <= 0) { fclose(fp); return false; }
    rewind(fp);
    char* buf = (char*)malloc((size_t)fsz + 1);
    if (buf == NULL) { fclose(fp); return false; }
    size_t nr = fread(buf, 1, (size_t)fsz, fp);
    fclose(fp);
    buf[nr] = '\0';
    cJSON* root = cJSON_ParseWithLength(buf, nr);
    free(buf);
    if (root == NULL) return false;
    cJSON* wmap = cJSON_GetObjectItemCaseSensitive(root, "weight_map");
    if (!cJSON_IsObject(wmap)) { cJSON_Delete(root); return false; }

    const char* needle = ".ngram_embedding.shards.";
    char prefix[384];
    prefix[0] = '\0';
    int nshards = 0;
    cJSON* it = NULL;
    cJSON_ArrayForEach(it, wmap) {
        if (!cJSON_IsString(it) || it->string == NULL) continue;
        const char* k = it->string;
        size_t kl = strlen(k);
        if (kl <= 7 || strcmp(k + kl - 7, ".weight") != 0) continue;
        const char* np = strstr(k, needle);
        if (np == NULL) continue;
        size_t plen = (size_t)(np - k) + strlen(needle);
        if (plen >= sizeof(prefix)) continue;
        if (prefix[0] == '\0') { memcpy(prefix, k, plen); prefix[plen] = '\0'; }
        else if (strncmp(prefix, k, plen) != 0) continue;
        int idx = atoi(k + plen);
        if (idx + 1 > nshards) nshards = idx + 1;
    }
    if (prefix[0] == '\0' || nshards <= 0 || nshards > 256) { cJSON_Delete(root); return false; }

    cJSON* fhdr[16];
    for (int q = 0; q < 16; q++) fhdr[q] = NULL;
    bool ok = true;
    int64_t rows_expected = -1;
    int w_cols = 0, s_cols = 0, b_cols = 0, w_isz = 0, s_isz = 0, b_isz = 0;
    mlx_dtype w_dt = MLX_UINT32, s_dt = MLX_BFLOAT16, b_dt = MLX_BFLOAT16;
    const char* leaves[3] = {"weight", "scales", "biases"};
    for (int s = 0; ok && s < nshards; s++) {
        for (int li = 0; li < 3; li++) {
            char key[512];
            int kn = snprintf(key, sizeof(key), "%s%d.%s", prefix, s, leaves[li]);
            if (kn <= 0 || kn >= (int)sizeof(key)) { ok = false; break; }
            cJSON* fn = cJSON_GetObjectItemCaseSensitive(wmap, key);
            if (!cJSON_IsString(fn)) { ok = false; break; }
            int fi = -1;
            char fpath[1200];
            snprintf(fpath, sizeof(fpath), "%s/%s", model_dir, fn->valuestring);
            for (int q = 0; q < m->ngram_nfiles; q++)
                if (strcmp(m->ngram_file[q].path, fpath) == 0) { fi = q; break; }
            if (fi < 0) {
                if (m->ngram_nfiles >= 16) { ok = false; break; }
                int fd = open(fpath, O_RDONLY);
                if (fd < 0) { ok = false; break; }
                uint64_t hlen = 0;
                if (pread(fd, &hlen, sizeof(hlen), 0) != (ssize_t)sizeof(hlen) ||
                    hlen == 0 || hlen > (64u * 1024u * 1024u)) { close(fd); ok = false; break; }
                char* hb = (char*)malloc((size_t)hlen + 1);
                if (hb == NULL) { close(fd); ok = false; break; }
                if (pread(fd, hb, (size_t)hlen, (off_t)sizeof(hlen)) != (ssize_t)hlen) { free(hb); close(fd); ok = false; break; }
                hb[hlen] = '\0';
                cJSON* hj = cJSON_ParseWithLength(hb, hlen);
                free(hb);
                if (hj == NULL) { close(fd); ok = false; break; }
                fi = m->ngram_nfiles++;
                snprintf(m->ngram_file[fi].path, sizeof(m->ngram_file[fi].path), "%s", fpath);
                m->ngram_file[fi].fd = fd;
                m->ngram_file[fi].data_start = (int64_t)sizeof(hlen) + (int64_t)hlen;
                fhdr[fi] = hj;
            }
            cJSON* t = cJSON_GetObjectItemCaseSensitive(fhdr[fi], key);
            cJSON* dtj = t ? cJSON_GetObjectItemCaseSensitive(t, "dtype") : NULL;
            cJSON* shj = t ? cJSON_GetObjectItemCaseSensitive(t, "shape") : NULL;
            cJSON* ofj = t ? cJSON_GetObjectItemCaseSensitive(t, "data_offsets") : NULL;
            if (!cJSON_IsString(dtj) || !cJSON_IsArray(shj) || cJSON_GetArraySize(shj) != 2 ||
                !cJSON_IsArray(ofj) || cJSON_GetArraySize(ofj) < 1) { ok = false; break; }
            mlx_dtype dt; int isz;
            if (!st_dtype_info(dtj->valuestring, &dt, &isz)) { ok = false; break; }
            int64_t r = (int64_t)cJSON_GetArrayItem(shj, 0)->valuedouble;
            int cols = cJSON_GetArrayItem(shj, 1)->valueint;
            int64_t off = (int64_t)cJSON_GetArrayItem(ofj, 0)->valuedouble;
            if (rows_expected < 0) rows_expected = r;
            else if (r != rows_expected) { ok = false; break; }
            if (li == 0) {
                if (s == 0) { w_dt = dt; w_isz = isz; w_cols = cols; }
                else if (dt != w_dt || isz != w_isz || cols != w_cols) { ok = false; break; }
                m->ngram_shard[s].w_file = fi; m->ngram_shard[s].w_off = off;
            } else if (li == 1) {
                if (s == 0) { s_dt = dt; s_isz = isz; s_cols = cols; }
                else if (dt != s_dt || isz != s_isz || cols != s_cols) { ok = false; break; }
                m->ngram_shard[s].s_file = fi; m->ngram_shard[s].s_off = off;
            } else {
                if (s == 0) { b_dt = dt; b_isz = isz; b_cols = cols; }
                else if (dt != b_dt || isz != b_isz || cols != b_cols) { ok = false; break; }
                m->ngram_shard[s].b_file = fi; m->ngram_shard[s].b_off = off;
            }
        }
    }

    for (int q = 0; q < m->ngram_nfiles; q++)
        if (fhdr[q] != NULL) cJSON_Delete(fhdr[q]);
    cJSON_Delete(root);

    if (!ok) {
        for (int q = 0; q < m->ngram_nfiles; q++)
            if (m->ngram_file[q].fd >= 0) { close(m->ngram_file[q].fd); m->ngram_file[q].fd = -1; }
        m->ngram_nfiles = 0;
        return false;
    }

    m->ngram_w_dtype = w_dt; m->ngram_w_itemsize = w_isz; m->ngram_w_cols = w_cols;
    m->ngram_s_dtype = s_dt; m->ngram_s_itemsize = s_isz; m->ngram_s_cols = s_cols;
    m->ngram_b_dtype = b_dt; m->ngram_b_itemsize = b_isz; m->ngram_b_cols = b_cols;
    m->ngram_has_biases = true;
    m->ngram_sharded = 1;
    m->ngram_nshards = nshards;
    m->ngram_rows_per_shard = rows_expected;
    m->ngram_table_bits = (m->cfg.quant_bits > 0) ? m->cfg.quant_bits : 4;
    m->ngram_table_group_size = (m->cfg.quant_group_size > 0) ? m->cfg.quant_group_size : 32;
    /* override per-modulo degli shard PLE (possono avere bits/group diversi dal top-level) */
    ngram_qp_from_config(model_dir, prefix, &m->ngram_table_bits, &m->ngram_table_group_size);
    m->ngram_table_loaded = true;
    return true;
}

/* Apre (lazy) il fd di un file shard, tenuto in cache per la vita del modello. */
static int q4e_ngram_shard_fd(LlmMlxModel* m, int fi) {
    if (fi < 0 || fi >= m->ngram_nfiles) return -1;
    if (m->ngram_file[fi].fd < 0)
        m->ngram_file[fi].fd = open(m->ngram_file[fi].path, O_RDONLY);
    return m->ngram_file[fi].fd;
}

/* Legge SOLO le righe in row_ids (n_rows di esse, id in [0, num_righe_tabella)) dal file
 * grezzo aperto da qwen4_exp_ngram_table_open, via pread() ad offset calcolato --
 * MAI l'array intero. out_w/out_s ricevono un mlx_array 2D [n_rows, cols]; out_b idem
 * MA SOLO se m->ngram_has_biases (altrimenti *out_b non e' scritto, il chiamante deve
 * controllare has_biases prima di usarlo -- stesso schema di have_b nel resto del
 * modulo N-gram/PLE). Nessuna eval MLX qui: i buffer letti sono gia' host-side, l'array
 * risultante e' costruito direttamente da quei byte (mlx_array_new_data copia). */
static bool qwen4_exp_ngram_read_rows(LlmMlxModel* m, const uint32_t* row_ids, size_t n_rows,
                                       mlx_array* out_w, mlx_array* out_s, mlx_array* out_b) {
    if (!m->ngram_table_loaded || row_ids == NULL || n_rows == 0) return false;
    if (!m->ngram_sharded && m->ngram_fd < 0) return false;

    const size_t w_row_bytes = (size_t)m->ngram_w_cols * (size_t)m->ngram_w_itemsize;
    const size_t s_row_bytes = (size_t)m->ngram_s_cols * (size_t)m->ngram_s_itemsize;
    const size_t b_row_bytes = m->ngram_has_biases ? (size_t)m->ngram_b_cols * (size_t)m->ngram_b_itemsize : 0;

    unsigned char* wbuf = (unsigned char*)malloc(n_rows * w_row_bytes);
    unsigned char* sbuf = (unsigned char*)malloc(n_rows * s_row_bytes);
    unsigned char* bbuf = m->ngram_has_biases ? (unsigned char*)malloc(n_rows * b_row_bytes) : NULL;
    if (wbuf == NULL || sbuf == NULL || (m->ngram_has_biases && bbuf == NULL)) {
        free(wbuf); free(sbuf); free(bbuf);
        return false;
    }

    bool ok = true;
    for (size_t i = 0; ok && i < n_rows; i++) {
        const int64_t row = (int64_t)row_ids[i];
        int wfd, sfd, bfd;
        int64_t wbase, sbase, bbase;
        if (m->ngram_sharded) {
            /* Shard = row // rows_per_shard (stesso partizionamento row-wise della
             * ShardedEmbedding di riferimento). Offset assoluto = data_start del file +
             * offset del tensore + riga locale * byte-per-riga. */
            int64_t shard = row / m->ngram_rows_per_shard;
            int64_t local = row - shard * m->ngram_rows_per_shard;
            if (shard < 0 || shard >= m->ngram_nshards) { ok = false; break; }
            int sfi = (int)shard;
            int wf = m->ngram_shard[sfi].w_file, sf = m->ngram_shard[sfi].s_file, bf = m->ngram_shard[sfi].b_file;
            wfd = q4e_ngram_shard_fd(m, wf);
            sfd = q4e_ngram_shard_fd(m, sf);
            bfd = q4e_ngram_shard_fd(m, bf);
            if (wfd < 0 || sfd < 0 || (m->ngram_has_biases && bfd < 0)) { ok = false; break; }
            wbase = m->ngram_file[wf].data_start + m->ngram_shard[sfi].w_off + local * (int64_t)w_row_bytes;
            sbase = m->ngram_file[sf].data_start + m->ngram_shard[sfi].s_off + local * (int64_t)s_row_bytes;
            bbase = m->ngram_file[bf].data_start + m->ngram_shard[sfi].b_off + local * (int64_t)b_row_bytes;
        } else {
            wfd = sfd = bfd = m->ngram_fd;
            wbase = m->ngram_data_start + m->ngram_w_off + row * (int64_t)w_row_bytes;
            sbase = m->ngram_data_start + m->ngram_s_off + row * (int64_t)s_row_bytes;
            bbase = m->ngram_data_start + m->ngram_b_off + row * (int64_t)b_row_bytes;
        }
        ok = pread(wfd, wbuf + i * w_row_bytes, w_row_bytes, (off_t)wbase) == (ssize_t)w_row_bytes &&
             pread(sfd, sbuf + i * s_row_bytes, s_row_bytes, (off_t)sbase) == (ssize_t)s_row_bytes;
        if (ok && m->ngram_has_biases)
            ok = pread(bfd, bbuf + i * b_row_bytes, b_row_bytes, (off_t)bbase) == (ssize_t)b_row_bytes;
    }

    if (ok) {
        const int wshape[2] = {(int)n_rows, m->ngram_w_cols};
        const int sshape[2] = {(int)n_rows, m->ngram_s_cols};
        *out_w = mlx_array_new_data(wbuf, wshape, 2, m->ngram_w_dtype);
        *out_s = mlx_array_new_data(sbuf, sshape, 2, m->ngram_s_dtype);
        if (m->ngram_has_biases) {
            const int bshape[2] = {(int)n_rows, m->ngram_b_cols};
            *out_b = mlx_array_new_data(bbuf, bshape, 2, m->ngram_b_dtype);
        }
    }
    free(wbuf); free(sbuf); free(bbuf);
    return ok;
}

/* Qwen4-Exp N-gram/PLE (qwen4_exp.py righe 933-1172) -- iniezione additiva su hidden_states
 * PRIMA dell'hyper-connection, sui soli layer in ple_layer_ids. Ritorna false SENZA
 * scrivere *out se una qualunque precondizione manca (tabella non caricata, side-channel
 * token non impostato dal chiamante corrente, eos_token_id assente dal config) -- il
 * chiamante (qwen4_exp_attention_dispatch) tratta questo come "non disponibile" e si
 * rifiuta esplicitamente, MAI un fallback silenzioso.
 * layer_multipliers/ngram_heads_offsets/ngram_heads_vocab_sizes sono LETTI dal
 * checkpoint (tensori gia' precalcolati dall'esportatore, verificato sul checkpoint
 * reale), non ricalcolati da un seed -- piu' robusto, stesso schema "non facciamo gli
 * inventori" seguito per tutto il resto di questa architettura. */
static bool qwen4_exp_ple_inject(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens,
                                  mlx_array hidden_states, mlx_array* out) {
    if (!m->ngram_table_loaded) return false;
    if (m->q4e_ctx_ids == NULL || m->q4e_ctx_len < cache_offset + n_tokens) return false;
    if (m->cfg.eos_token_id < 0) return false;

    bool ok = true;
    char namebuf[256];
    mlx_stream s = m->stream;
    const int gsz = m->cfg.quant_group_size, bts = m->cfg.quant_bits;
    const int ngram_size = m->cfg.ngram_size, heads_per_ngram = m->cfg.heads_per_ngram;
    const int n_heads = (ngram_size - 1) * heads_per_ngram;
    const int embed_dim = m->cfg.ple_embed_dim;

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen4_exp_ple[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)
    /* DIAGNOSTICA F2 (todo Qwen38-FlashNext-60tok): la PLE non cambia l'output -> dove si annulla? */
#define PLEDBG_F(tag, arr) do { if (llm_env_isset("LLM_QWEN4EXP_PLE_DEBUG") && (arr).ctx != NULL) { \
        mlx_array ab_ = mlx_array_new(), mx_ = mlx_array_new(); \
        if (mlx_abs(&ab_, (arr), s) == 0 && mlx_max(&mx_, ab_, false, s) == 0 && mlx_array_eval(mx_) == 0) { \
            const float* pp_ = mlx_array_data_float32(mx_); \
            fprintf(stderr, "[ple-debug][layer=%d] %-16s shape[0]=%d absmax=%.6g\n", layer_idx, tag, \
                    mlx_array_ndim(arr) > 0 ? (int)mlx_array_shape(arr)[0] : -1, pp_ ? (double)pp_[0] : -1.0); } \
        mlx_array_free(ab_); mlx_array_free(mx_); } } while (0)

    PLEDBG_F("hidden_in", hidden_states);
    mlx_array mult_arr = mlx_array_new(), voff = mlx_array_new(), vsz = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "ple.ple_embedding.layer_multipliers", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &mult_arr);
    tname(m->cfg.layer_prefix, layer_idx, "ple.ple_embedding.ngram_heads_offsets", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &voff);
    tname(m->cfg.layer_prefix, layer_idx, "ple.ple_embedding.ngram_heads_vocab_sizes", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &vsz);
    CHK(ok, "ple_embedding metadata lookup");

    mlx_array mult_i64 = mlx_array_new();
    if (ok) { ok = mlx_astype(&mult_i64, mult_arr, MLX_INT64, s) == 0 && mlx_array_eval(mult_i64) == 0; CHK(ok, "layer_multipliers readback"); }
    const int64_t* mult_host = ok ? mlx_array_data_int64(mult_i64) : NULL;
    if (ok && (mult_host == NULL || (int)mlx_array_size(mult_i64) != ngram_size)) { CHK(false, "layer_multipliers shape"); }

    mlx_array history = mlx_array_new();
    if (ok) {
        int64_t* h64 = (int64_t*)malloc((size_t)m->q4e_ctx_len * sizeof(int64_t));
        if (h64 == NULL) { CHK(false, "history alloc"); }
        else {
            for (int i = 0; i < m->q4e_ctx_len; i++) h64[i] = (int64_t)m->q4e_ctx_ids[i];
            const int hshape[2] = {1, m->q4e_ctx_len};
            history = mlx_array_new_data(h64, hshape, 2, MLX_INT64);
            free(h64);
        }
    }

    mlx_array ids = mlx_array_new();
    if (ok) { ok = llm_mlx_qwen4exp_ngram_compute_ids(&ids, history, n_tokens, ngram_size, heads_per_ngram,
                                                       mult_host, vsz, voff, m->cfg.eos_token_id, s);
              CHK(ok, "ngram_compute_ids"); }

    mlx_array embeddings = mlx_array_new();
    mlx_array ids_u32 = mlx_array_new();
    mlx_array w_rows_flat = mlx_array_new(), s_rows_flat = mlx_array_new(), b_rows_flat = mlx_array_new();
    mlx_array w_rows = mlx_array_new(), s_rows = mlx_array_new(), b_rows = mlx_array_new();
    mlx_array gathered = mlx_array_new();
    const bool have_b = m->ngram_has_biases;
    if (ok) {
        /* Lettura mirata a righe via pread() sul file grezzo (qwen4_exp_ngram_read_rows,
         * sopra) -- MAI l'array intero in RAM (30GB sul checkpoint reale). mlx_take_axis su
         * un array lazy-loaded da mlx_load_safetensors forzava mlx/backend/common/load.cpp
         * a materializzare l'intera tabella per estrarre poche righe: root cause verificata
         * dello swap thrashing/157s-per-token. Vedi Piano-Correzioni-Modelli-MLX.md §C17
         * "PIANO -- memoria" (2026-09-07). */
        ok = mlx_astype(&ids_u32, ids, MLX_UINT32, s) == 0;
        if (ok) ok = mlx_array_eval(ids_u32) == 0;
        CHK(ok, "ngram ids astype/eval");
        if (ok && llm_env_isset("LLM_QWEN4EXP_PLE_DEBUG")) {
            const uint32_t* rid = mlx_array_data_uint32(ids_u32);
            int d0 = mlx_array_dim(ids_u32, 0), d1 = mlx_array_dim(ids_u32, 1), d2 = mlx_array_dim(ids_u32, 2);
            fprintf(stderr, "[ple-ids][layer=%d] shape=[%d,%d,%d] last:", layer_idx, d0, d1, d2);
            for (int i = 0; i < d2; i++) fprintf(stderr, " %u", rid[(size_t)(d1 - 1) * (size_t)d2 + (size_t)i]);
            fprintf(stderr, "\n");
        }
        const size_t n_rows = ok ? mlx_array_size(ids_u32) : 0;
        const uint32_t* row_ids = ok ? mlx_array_data_uint32(ids_u32) : NULL;
        if (ok) ok = qwen4_exp_ngram_read_rows(m, row_ids, n_rows, &w_rows_flat, &s_rows_flat, have_b ? &b_rows_flat : NULL);
        CHK(ok, "ngram table targeted row read");
        if (ok) {
            const int d0 = mlx_array_dim(ids_u32, 0), d1 = mlx_array_dim(ids_u32, 1), d2 = mlx_array_dim(ids_u32, 2);
            const int ws[4] = {d0, d1, d2, m->ngram_w_cols};
            const int ss[4] = {d0, d1, d2, m->ngram_s_cols};
            ok = mlx_reshape(&w_rows, w_rows_flat, ws, 4, s) == 0 &&
                 mlx_reshape(&s_rows, s_rows_flat, ss, 4, s) == 0;
            if (ok && have_b) {
                const int bs[4] = {d0, d1, d2, m->ngram_b_cols};
                ok = mlx_reshape(&b_rows, b_rows_flat, bs, 4, s) == 0;
            }
            CHK(ok, "ngram rows reshape");
        }
        if (ok) {
            mlx_optional_int qgsz = {m->ngram_table_group_size, true};
            mlx_optional_int qbts = {m->ngram_table_bits, true};
            mlx_optional_dtype qdt = {MLX_FLOAT32, true};
            mlx_array no_gscale = mlx_array_new();
            ok = mlx_dequantize(&gathered, w_rows, s_rows, have_b ? b_rows : mlx_array_new(),
                                 qgsz, qbts, "affine", no_gscale, qdt, s) == 0;
            mlx_array_free(no_gscale);
        }
        CHK(ok, "ngram table gather/dequantize");
    }
    if (ok) { const int es[3] = {1, n_tokens, embed_dim}; ok = mlx_reshape(&embeddings, gathered, es, 3, s) == 0; CHK(ok, "embeddings reshape"); }
    PLEDBG_F("embeddings", embeddings);
    if (ok && llm_env_isset("LLM_QWEN4EXP_PLE_DEBUG")) {
        mlx_array ef = mlx_array_new();
        if (mlx_astype(&ef, embeddings, MLX_FLOAT32, s) == 0 && mlx_array_eval(ef) == 0) {
            const float* ed = mlx_array_data_float32(ef);
            int d0 = mlx_array_dim(ef, 0), d1 = mlx_array_dim(ef, 1), d2 = mlx_array_dim(ef, 2);
            size_t off = (size_t)(d1 - 1) * (size_t)d2;
            fprintf(stderr, "[ple-emb][layer=%d] shape=[%d,%d,%d] last_first8:", layer_idx, d0, d1, d2);
            for (int i = 0; i < 8 && i < d2; i++) fprintf(stderr, " %.6g", ed[off + (size_t)i]);
            fprintf(stderr, "\n");
        }
        mlx_array_free(ef);
    }

    mlx_array key_w = mlx_array_new(), key_sc = mlx_array_new(), key_b = mlx_array_new();
    mlx_array val_w = mlx_array_new(), val_sc = mlx_array_new(), val_b = mlx_array_new();
    mlx_array nk_w = mlx_array_new(), nq_w = mlx_array_new(), nc_w = mlx_array_new(), cv_w = mlx_array_new();
    if (ok) {
        tname(m->cfg.layer_prefix, layer_idx, "ple.key_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &key_w, &key_sc, &key_b);
        tname(m->cfg.layer_prefix, layer_idx, "ple.value_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &val_w, &val_sc, &val_b);
        tname(m->cfg.layer_prefix, layer_idx, "ple.norm_key.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &nk_w);
        tname(m->cfg.layer_prefix, layer_idx, "ple.norm_query.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &nq_w);
        tname(m->cfg.layer_prefix, layer_idx, "ple.norm_conv.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &nc_w);
        tname(m->cfg.layer_prefix, layer_idx, "ple.conv1d.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &cv_w);
        CHK(ok, "ple weights lookup");
    }

    mlx_array key_lin = mlx_array_new(), values = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&key_lin, embeddings, key_w, key_sc, key_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "key_proj matmul"); }
    if (ok) { ok = quantized_linear_cfg(&values, embeddings, val_w, val_sc, val_b, gsz, bts, m->cfg.quant_mode, s); CHK(ok, "value_proj matmul"); }
    PLEDBG_F("key_lin", key_lin); PLEDBG_F("values", values);

    bool has_conv_state = false;
    mlx_array* conv_cache = llm_mlx_qwen4exp_ple_conv_cache(m, layer_idx, &has_conv_state);
    CHK(conv_cache != NULL, "ple conv cache alloc");
    mlx_array conv_state_in = mlx_array_new();
    if (ok && has_conv_state) ok = mlx_array_set(&conv_state_in, *conv_cache) == 0;
    mlx_array conv_state_out = mlx_array_new();
    if (ok) { ok = llm_mlx_qwen4exp_ple_layer(out, &conv_state_out, hidden_states, key_lin, values,
                                               nk_w, nq_w, nc_w, cv_w, conv_state_in,
                                               m->cfg.hc_count, m->cfg.hidden_size,
                                               m->cfg.ple_conv_kernel_size, ngram_size, m->cfg.rms_norm_eps, s);
              CHK(ok, "ple_layer"); }
    PLEDBG_F("ple_out", *out);
    if (ok && conv_cache != NULL) { mlx_array_free(*conv_cache); *conv_cache = mlx_array_new(); mlx_array_set(conv_cache, conv_state_out); }

    mlx_array_free(mult_arr); mlx_array_free(voff); mlx_array_free(vsz); mlx_array_free(mult_i64);
    mlx_array_free(history); mlx_array_free(ids); mlx_array_free(ids_u32);
    mlx_array_free(w_rows); mlx_array_free(s_rows); mlx_array_free(b_rows); mlx_array_free(gathered);
    mlx_array_free(w_rows_flat); mlx_array_free(s_rows_flat); mlx_array_free(b_rows_flat);
    mlx_array_free(embeddings);
    mlx_array_free(key_w); mlx_array_free(key_sc); mlx_array_free(key_b);
    mlx_array_free(val_w); mlx_array_free(val_sc); mlx_array_free(val_b);
    mlx_array_free(nk_w); mlx_array_free(nq_w); mlx_array_free(nc_w); mlx_array_free(cv_w);
    mlx_array_free(key_lin); mlx_array_free(values);
    mlx_array_free(conv_state_in); mlx_array_free(conv_state_out);
    return ok;
#undef CHK
}

/* DecoderLayer di Qwen4-Exp (qwen4_exp.py righe 1175-1226) -- wiring completo
 * dell'hyper-connection attorno al sotto-layer di attenzione (GDN se is_linear_layer,
 * QSA altrimenti). x/out_h: [1,L,hc_width] (hc_width = hc_count*hidden_size) -- l'INTERO
 * modello lavora su questa larghezza dall'embedding (embed_token/embed_tokens_batch,
 * tile hc_count) fino al mixer finale (llm_mlx_final_norm): ARCH_TABLE.attn per
 * qwen4_exp riceve e ritorna SEMPRE hc_width, mai hidden_size nudo -- a differenza di
 * ogni altra architettura, dove attn/ffn lavorano sempre a hidden_size fisso.
 * PLE (righe 1145-1172, additivo su hidden_states PRIMA dell'hyper-connection, sui soli
 * layer in ple_layer_ids) e' WIRED (qwen4_exp_ple_inject sopra) quando tabella + config +
 * side-channel token sono disponibili; altrimenti rifiuto esplicito invariato (mai un
 * risultato silenziosamente diverso dal reference). */
bool qwen4_exp_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                          mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h) {
    bool ok = true;
    char namebuf[256];
    mlx_stream s = m->stream;

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen4_exp_attn_dispatch[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array x_ple = mlx_array_new();
    ok = mlx_array_set(&x_ple, x) == 0;
    for (int i = 0; i < m->cfg.ple_layer_ids_n && ok; i++) {
        if (m->cfg.ple_layer_ids[i] == layer_idx + 1) {
            /* KILL-SWITCH VERO per l'A/B (2026-09-18): `LLM_QWEN4EXP_SKIP_PLE` NON salta
             * l'iniezione (sopprime solo l'errore quando fallisce) -> un A/B con quello era
             * invalido. Questo invece salta DAVVERO l'iniezione (x_ple = x). */
            if (llm_env_isset("LLM_QWEN4EXP_PLE_OFF")) continue;
            mlx_array ple_out = mlx_array_new();
            bool ple_ok = qwen4_exp_ple_inject(m, layer_idx, cache_offset, n_tokens, x, &ple_out);
            if (ple_ok) {
                mlx_array summed = mlx_array_new();
                ple_ok = mlx_add(&summed, x, ple_out, s) == 0;
                if (ple_ok) { mlx_array_free(x_ple); x_ple = summed; } else mlx_array_free(summed);
            }
            mlx_array_free(ple_out);
            if (!ple_ok) {
                if (!llm_env_isset("LLM_QWEN4EXP_SKIP_PLE")) {
                    fprintf(stderr, "qwen4_exp[layer=%d]: PLE richiesto da config (ple_layer_ids) ma non "
                                    "disponibile (tabella/side-channel/config mancante) -- rifiuto esplicito\n", layer_idx);
                    ok = false;
                } else {
                    fprintf(stderr, "qwen4_exp[layer=%d]: LLM_QWEN4EXP_SKIP_PLE attivo -- iniezione PLE SALTATA "
                                    "(diagnostica, output non fedele al reference su questo layer)\n", layer_idx);
                }
            }
        }
    }
    if (!ok) { mlx_array_free(x_ple); return false; }
    PLEDBG_F("x_ple_after", x_ple);

    mlx_array hc_norm_w = mlx_array_new();
    mlx_array down_w = mlx_array_new(), down_sc = mlx_array_new(), down_b = mlx_array_new();
    mlx_array up_w = mlx_array_new(), up_sc = mlx_array_new(), up_b = mlx_array_new();
    mlx_array inj_w = mlx_array_new(), inj_sc = mlx_array_new(), inj_b = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "attn_hyper_connection.hc_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &hc_norm_w);
    mlx_array mixed = mlx_array_new(), residual = mlx_array_new(), injection = mlx_array_new();
    if (ok &&
        qwen4_hc_packed_try(m, layer_idx, "attn_hyper_connection", x_ple, hc_norm_w, &mixed, &residual, &injection) == 0) {
        /* packed: pesi letti dai descrittori, nessun array caricato */
    } else if (ok) {
        tname(m->cfg.layer_prefix, layer_idx, "attn_hyper_connection.input_mix_weight_down", namebuf, sizeof(namebuf));
        ok = multishard_get_triple(&m->weights, namebuf, &down_w, &down_sc, &down_b);
        tname(m->cfg.layer_prefix, layer_idx, "attn_hyper_connection.input_mix_weight_up", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &up_w, &up_sc, &up_b);
        tname(m->cfg.layer_prefix, layer_idx, "attn_hyper_connection.block_inject_weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &inj_w, &inj_sc, &inj_b);
        CHK(ok, "attn_hyper_connection weights lookup");
        if (ok) { ok = qwen4_exp_gated_residual_q(m, x_ple, hc_norm_w, down_w, down_sc, down_b, up_w, up_sc, up_b,
                                                  inj_w, inj_sc, inj_b, &mixed, &residual, &injection);
                  CHK(ok, "attn_hyper_connection mix"); }
    }

    mlx_array sub_out = mlx_array_new();
    if (ok) {
        bool is_linear = m->cfg.is_linear_layer != NULL && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx];
        if (is_linear)
            ok = qwen4_linear_gdn_block(m, layer_idx, cache_offset, n_tokens, mixed, k_cache, v_cache, has_cache, &sub_out);
        else
            ok = qwen4_full_qsa_block(m, layer_idx, cache_offset, n_tokens, mixed, k_cache, v_cache, has_cache, &sub_out);
        CHK(ok, "sub-layer (GDN/QSA)");
    }

    if (ok) { ok = (llm_mlx_hc_combine_packed(out_h, sub_out, residual, injection, m) == 0) || llm_mlx_hc_combine(out_h, sub_out, residual, injection, m->cfg.hc_count, m->cfg.hidden_size, s);
              CHK(ok, "attn_hyper_connection combine"); }

    /* F3: dump dell'output ATTENTION (fine sub-blocco, prima del ffn) per-layer. */
    if (ok && llm_env_isset("LLM_QWEN4EXP_LAYER_DUMP")) {
        const char* e = getenv("LLM_QWEN4EXP_LAYER_DUMP");
        int lim = (e && *e) ? atoi(e) : 48;
        if (layer_idx < lim) {
            mlx_array dbga = mlx_array_new();
            if (mlx_astype(&dbga, *out_h, MLX_FLOAT32, s) == 0 && mlx_array_eval(dbga) == 0) {
                const float* d = mlx_array_data_float32(dbga);
                int nd = mlx_array_ndim(dbga);
                if (d && nd >= 2) {
                    int W = (int)mlx_array_shape(dbga)[nd - 1];
                    int S = (int)mlx_array_shape(dbga)[nd - 2];
                    if (W > 0 && S > 0) {
                        const float* last = d + (size_t)(S - 1) * (size_t)W;
                        float am = 0.0f; double sum = 0.0;
                        for (int i = 0; i < W; i++) { float a = fabsf(last[i]); if (a > am) am = a; sum += (double)last[i]; }
                        const char* lty = (m->cfg.is_linear_layer && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx]) ? "GDN" : "QSA";
                        fprintf(stderr, "[Q4_ATTN] layer=%d type=%s absmax=%.6f mean=%.6f sum=%.6f first4=%.6f,%.6f,%.6f,%.6f W=%d\n",
                                layer_idx, lty, am, sum / (double)W, sum, last[0], last[1], last[2], last[3], W);
                    }
                }
            }
            mlx_array_free(dbga);
        }
    }

    mlx_array_free(hc_norm_w);
    mlx_array_free(down_w); mlx_array_free(down_sc); mlx_array_free(down_b);
    mlx_array_free(up_w); mlx_array_free(up_sc); mlx_array_free(up_b);
    mlx_array_free(inj_w); mlx_array_free(inj_sc); mlx_array_free(inj_b);
    mlx_array_free(mixed); mlx_array_free(residual); mlx_array_free(injection); mlx_array_free(sub_out);
    mlx_array_free(x_ple);
    return ok;
#undef CHK
}

/* Specchio di qwen4_exp_attention_dispatch per il sotto-layer MLP (mlp_hyper_connection
 * invece di attn_hyper_connection, SparseMoeBlock invece di GDN/QSA -- qwen4_exp_moe_block
 * sopra). h/out: [1,L,hc_width], stesso principio. */
bool qwen4_exp_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    bool ok = true;
    char namebuf[256];
    mlx_stream s = m->stream;

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "qwen4_exp_ffn_dispatch[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array hc_norm_w = mlx_array_new();
    mlx_array down_w = mlx_array_new(), down_sc = mlx_array_new(), down_b = mlx_array_new();
    mlx_array up_w = mlx_array_new(), up_sc = mlx_array_new(), up_b = mlx_array_new();
    mlx_array inj_w = mlx_array_new(), inj_sc = mlx_array_new(), inj_b = mlx_array_new();
    tname(m->cfg.layer_prefix, layer_idx, "mlp_hyper_connection.hc_norm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &hc_norm_w);
    mlx_array mixed = mlx_array_new(), residual = mlx_array_new(), injection = mlx_array_new();
    if (ok &&
        qwen4_hc_packed_try(m, layer_idx, "mlp_hyper_connection", h, hc_norm_w, &mixed, &residual, &injection) == 0) {
        /* packed */
    } else if (ok) {
        tname(m->cfg.layer_prefix, layer_idx, "mlp_hyper_connection.input_mix_weight_down", namebuf, sizeof(namebuf));
        ok = multishard_get_triple(&m->weights, namebuf, &down_w, &down_sc, &down_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp_hyper_connection.input_mix_weight_up", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &up_w, &up_sc, &up_b);
        tname(m->cfg.layer_prefix, layer_idx, "mlp_hyper_connection.block_inject_weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &inj_w, &inj_sc, &inj_b);
        CHK(ok, "mlp_hyper_connection weights lookup");
        if (ok) { ok = qwen4_exp_gated_residual_q(m, h, hc_norm_w, down_w, down_sc, down_b, up_w, up_sc, up_b,
                                                  inj_w, inj_sc, inj_b, &mixed, &residual, &injection);
                  CHK(ok, "mlp_hyper_connection mix"); }
    }

    mlx_array moe_out = mlx_array_new();
    if (ok) { ok = qwen4_exp_moe_block(m, layer_idx, n_tokens, mixed, &moe_out); CHK(ok, "moe block"); }

    if (ok) { ok = (llm_mlx_hc_combine_packed(out, moe_out, residual, injection, m) == 0) || llm_mlx_hc_combine(out, moe_out, residual, injection, m->cfg.hc_count, m->cfg.hidden_size, s);
              CHK(ok, "mlp_hyper_connection combine"); }

    /* F3 (parità qwen4_exp): dump per-layer dell'hidden di output del layer (ultimo token).
     * LLM_QWEN4EXP_LAYER_DUMP=<N> → stampa absmax/mean sui primi N layer. Confrontato col
     * reference lazy (tools/q4_ref_lazy.py). No-op se non impostato. */
    if (ok && llm_env_isset("LLM_QWEN4EXP_LAYER_DUMP")) {
        const char* e = getenv("LLM_QWEN4EXP_LAYER_DUMP");
        int lim = (e && *e) ? atoi(e) : 48;
        if (layer_idx < lim) {
            mlx_array dbg = mlx_array_new();
            if (mlx_astype(&dbg, *out, MLX_FLOAT32, s) == 0 && mlx_array_eval(dbg) == 0) {
                const float* d = mlx_array_data_float32(dbg);
                int nd = mlx_array_ndim(dbg);
                if (d && nd >= 2) {
                    int W = (int)mlx_array_shape(dbg)[nd - 1];
                    int S = (int)mlx_array_shape(dbg)[nd - 2];
                    if (W > 0 && S > 0) {
                        const float* last = d + (size_t)(S - 1) * (size_t)W;
                        float am = 0.0f; double sum = 0.0;
                        for (int i = 0; i < W; i++) { float a = fabsf(last[i]); if (a > am) am = a; sum += (double)last[i]; }
                        const char* lty = (m->cfg.is_linear_layer && layer_idx < m->cfg.n_layers && m->cfg.is_linear_layer[layer_idx]) ? "GDN" : "QSA";
                        fprintf(stderr, "[Q4_DUMP] layer=%d type=%s absmax=%.6f mean=%.6f sum=%.6f first4=%.6f,%.6f,%.6f,%.6f W=%d\n",
                                layer_idx, lty, am, sum / (double)W, sum, last[0], last[1], last[2], last[3], W);
                        /* F3: dump vettoriale (ultimo token, f32) per il confronto di correlazione. */
                        const char* vp = llm_ai_mlx_opt_env("LLM_QWEN4EXP_LAYER_VEC");
                        if (vp && *vp && n_tokens > 1) {
                            char vpath[1024];
                            snprintf(vpath, sizeof vpath, "%s.%d.bin", vp, layer_idx);
                            FILE* vf = fopen(vpath, "wb");
                            if (vf) { fwrite(last, sizeof(float), (size_t)W, vf); fclose(vf); }
                        }
                    }
                }
            }
            mlx_array_free(dbg);
        }
    }

    mlx_array_free(hc_norm_w);
    mlx_array_free(down_w); mlx_array_free(down_sc); mlx_array_free(down_b);
    mlx_array_free(up_w); mlx_array_free(up_sc); mlx_array_free(up_b);
    mlx_array_free(inj_w); mlx_array_free(inj_sc); mlx_array_free(inj_b);
    mlx_array_free(mixed); mlx_array_free(residual); mlx_array_free(injection); mlx_array_free(moe_out);
    return ok;
#undef CHK
}
