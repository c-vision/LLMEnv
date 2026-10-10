/* llmcodec_gguf_compare.c — vedi llmcodec_gguf_compare.h per semantica e metriche. */
#include "llmcodec_gguf_compare.h"
#include "llmcodec_gguf_read.h"
#include "llmcodec_ggml_quants.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ glob ---
 * Adattato da antirez/gguf-tools (MIT). Opera su (ptr,len) per evitare copie; la
 * funzione pubblica lo invoca con le lunghezze delle stringhe C. */
static int strmatch(const char* pattern, int patternLen,
                    const char* string, int stringLen, int nocase)
{
    while (patternLen && stringLen) {
        switch (pattern[0]) {
        case '*':
            while (patternLen && pattern[1] == '*') { pattern++; patternLen--; }
            if (patternLen == 1) return 1;
            while (stringLen) {
                if (strmatch(pattern + 1, patternLen - 1, string, stringLen, nocase)) return 1;
                string++; stringLen--;
            }
            return 0;
        case '?':
            string++; stringLen--; break;
        case '[': {
            int not, match;
            pattern++; patternLen--;
            not = pattern[0] == '^';
            if (not) { pattern++; patternLen--; }
            match = 0;
            while (1) {
                if (pattern[0] == '\\' && patternLen >= 2) {
                    pattern++; patternLen--;
                    if (pattern[0] == string[0]) match = 1;
                } else if (pattern[0] == ']') {
                    break;
                } else if (patternLen == 0) {
                    pattern--; patternLen++; break;
                } else if (patternLen >= 3 && pattern[1] == '-') {
                    int start = pattern[0], end = pattern[2], c = string[0];
                    if (start > end) { int t = start; start = end; end = t; }
                    if (nocase) { start = tolower(start); end = tolower(end); c = tolower(c); }
                    pattern += 2; patternLen -= 2;
                    if (c >= start && c <= end) match = 1;
                } else {
                    if (!nocase) { if (pattern[0] == string[0]) match = 1; }
                    else { if (tolower((int)pattern[0]) == tolower((int)string[0])) match = 1; }
                }
                pattern++; patternLen--;
            }
            if (not) match = !match;
            if (!match) return 0;
            string++; stringLen--; break;
        }
        case '\\':
            if (patternLen >= 2) { pattern++; patternLen--; }
            /* fallthrough */
        default:
            if (!nocase) { if (pattern[0] != string[0]) return 0; }
            else { if (tolower((int)pattern[0]) != tolower((int)string[0])) return 0; }
            string++; stringLen--; break;
        }
        pattern++; patternLen--;
        if (stringLen == 0) { while (*pattern == '*') { pattern++; patternLen--; } break; }
    }
    if (patternLen == 0 && stringLen == 0) return 1;
    return 0;
}

int llmcodec_globs_match(const char* pattern, const char* string, int nocase)
{
    if (pattern == NULL || string == NULL) return 0;
    return strmatch(pattern, (int)strlen(pattern), string, (int)strlen(string), nocase);
}

/* ------------------------------------------------------------ compare ------ */
/* Chunk di dequant: 2^16 pesi per volta. Multiplo di OGNI block-size supportato (1/32/64/128/256,
 * tutti divisori di 65536), quindi sia il loop sia il resto finale restano allineati al blocco. */
#define CMP_CHUNK 65536u

static const LlmCodecGgufTensorInfo* find_tensor(const LlmCodecGgufFileInfo* fi, const char* name)
{
    for (uint64_t i = 0; i < fi->n_tensors; i++)
        if (strcmp(fi->tensors[i].name, name) == 0) return &fi->tensors[i];
    return NULL;
}

static int dims_equal(const LlmCodecGgufTensorInfo* a, const LlmCodecGgufTensorInfo* b)
{
    if (a->ndim != b->ndim) return 0;
    for (int d = 0; d < a->ndim; d++) if (a->dims[d] != b->dims[d]) return 0;
    return 1;
}

int llmcodec_gguf_compare(const char* path_a, const char* path_b, const char* glob,
                          llmcodec_gguf_compare_cb cb, void* ud,
                          LlmCodecGgufCompareSummary* out)
{
    if (path_a == NULL || path_b == NULL || out == NULL) return -1;
    memset(out, 0, sizeof *out);

    LlmCodecGgufFileInfo fa, fb;
    if (llmcodec_gguf_read_info(path_a, &fa) != 0) return -1;
    if (llmcodec_gguf_read_info(path_b, &fb) != 0) { llmcodec_gguf_free_info(&fa); return -1; }
    out->n_a = fa.n_tensors; out->n_b = fb.n_tensors;

    FILE* fpa = fopen(path_a, "rb");
    FILE* fpb = fopen(path_b, "rb");
    if (fpa == NULL || fpb == NULL) {
        if (fpa) fclose(fpa); if (fpb) fclose(fpb);
        llmcodec_gguf_free_info(&fa); llmcodec_gguf_free_info(&fb);
        return -1;
    }

    float* bufa = (float*)malloc((size_t)CMP_CHUNK * sizeof(float));
    float* bufb = (float*)malloc((size_t)CMP_CHUNK * sizeof(float));
    if (bufa == NULL || bufb == NULL) {
        free(bufa); free(bufb); fclose(fpa); fclose(fpb);
        llmcodec_gguf_free_info(&fa); llmcodec_gguf_free_info(&fb);
        return -1;
    }

    double agg_err = 0.0, agg_ref = 0.0;
    double sum_diff_pct = 0.0;

    for (uint64_t i = 0; i < fa.n_tensors; i++) {
        const LlmCodecGgufTensorInfo* ta = &fa.tensors[i];
        if (glob && *glob && !llmcodec_globs_match(glob, ta->name, 0)) continue;
        const LlmCodecGgufTensorInfo* tb = find_tensor(&fb, ta->name);
        if (tb == NULL) { out->only_a++; continue; }
        if (!dims_equal(ta, tb)) { out->mismatch++; continue; }

        uint64_t numel = 1;
        for (int d = 0; d < ta->ndim; d++) numel *= ta->dims[d];
        if (numel == 0) { out->skipped++; continue; }

        /* buffer-byte per il chunk peggiore dei due tipi */
        uint64_t cbytes_a = 0, cbytes_b = 0;
        llmcodec_ggml_type_bytes(ta->ggml_type, CMP_CHUNK, &cbytes_a);
        llmcodec_ggml_type_bytes(tb->ggml_type, CMP_CHUNK, &cbytes_b);
        unsigned char* rawa = (unsigned char*)malloc((size_t)(cbytes_a ? cbytes_a : 1));
        unsigned char* rawb = (unsigned char*)malloc((size_t)(cbytes_b ? cbytes_b : 1));
        if (rawa == NULL || rawb == NULL) { free(rawa); free(rawb); out->skipped++; continue; }

        double sum_err = 0.0, sum_ref = 0.0, sum_abs_a = 0.0, sum_abs_b = 0.0;
        double sum_abs_delta = 0.0, max_abs = 0.0;
        int bad = 0, first = 1;

        for (uint64_t off = 0; off < numel; off += CMP_CHUNK) {
            uint64_t n = numel - off; if (n > CMP_CHUNK) n = CMP_CHUNK;
            uint64_t ba = 0, bb = 0;
            llmcodec_ggml_type_bytes(ta->ggml_type, n, &ba);
            llmcodec_ggml_type_bytes(tb->ggml_type, n, &bb);
            uint64_t oa = 0, ob = 0;
            llmcodec_ggml_type_bytes(ta->ggml_type, off, &oa);
            llmcodec_ggml_type_bytes(tb->ggml_type, off, &ob);
            if (ba == 0 || bb == 0) { bad = 1; break; }
            if (fseek(fpa, (long)(fa.data_start + ta->offset + oa), SEEK_SET) != 0 ||
                fread(rawa, 1, (size_t)ba, fpa) != (size_t)ba ||
                fseek(fpb, (long)(fb.data_start + tb->offset + ob), SEEK_SET) != 0 ||
                fread(rawb, 1, (size_t)bb, fpb) != (size_t)bb) { bad = 1; break; }
            if (llmcodec_ggml_dequant_row(ta->ggml_type, rawa, bufa, (int64_t)n) != 0 ||
                llmcodec_ggml_dequant_row(tb->ggml_type, rawb, bufb, (int64_t)n) != 0) { bad = 1; break; }
            first = 0;
            for (uint64_t k = 0; k < n; k++) {
                double va = (double)bufa[k], vb = (double)bufb[k];
                double d = vb - va;
                sum_err += d * d; sum_ref += va * va;
                sum_abs_a += fabs(va); sum_abs_b += fabs(vb); sum_abs_delta += fabs(d);
                if (fabs(d) > max_abs) max_abs = fabs(d);
            }
        }
        free(rawa); free(rawb);
        if (bad || first) { out->skipped++; continue; }

        double avg_mag = (sum_abs_a + sum_abs_b) / (2.0 * (double)numel);
        LlmCodecGgufCompareStat st;
        st.rel_l2 = sum_ref > 0.0 ? sqrt(sum_err / sum_ref) : 0.0;
        st.avg_mag = avg_mag;
        st.avg_diff_pct = avg_mag > 0.0 ? (sum_abs_delta / (double)numel) / avg_mag * 75.0 : 0.0;
        st.max_abs = max_abs;
        st.numel = numel;
        st.ggml_type_a = ta->ggml_type;
        st.ggml_type_b = tb->ggml_type;

        out->matched++;
        agg_err += sum_err; agg_ref += sum_ref;
        sum_diff_pct += st.avg_diff_pct;
        if (st.avg_diff_pct > out->max_diff_pct) {
            out->max_diff_pct = st.avg_diff_pct;
            snprintf(out->max_diff_name, sizeof out->max_diff_name, "%s", ta->name);
        }
        if (cb && cb(ta->name, &st, ud) != 0) break;
    }

    /* only_b: tensori di B (col filtro) non presenti in A */
    for (uint64_t j = 0; j < fb.n_tensors; j++) {
        const LlmCodecGgufTensorInfo* tb = &fb.tensors[j];
        if (glob && *glob && !llmcodec_globs_match(glob, tb->name, 0)) continue;
        if (find_tensor(&fa, tb->name) == NULL) out->only_b++;
    }

    out->rel_l2_agg = agg_ref > 0.0 ? sqrt(agg_err / agg_ref) : 0.0;
    out->avg_diff_pct = out->matched ? sum_diff_pct / (double)out->matched : 0.0;

    free(bufa); free(bufb);
    fclose(fpa); fclose(fpb);
    llmcodec_gguf_free_info(&fa); llmcodec_gguf_free_info(&fb);
    return 0;
}
