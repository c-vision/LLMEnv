/* llmmodelio_model.c — MODEL COMPILER (namespace LLM): operazioni sui file modello.
 *
 * Primo comportamento reale: `llmmodelio_model_save(in, out)` — round-trip GGUF (legge metadata
 * KV + tensor-info dall'input e RISCRIVE un GGUF nuovo con gli stessi metadata e i byte dei
 * tensori). È il mattone di `LLM MODEL SAVE` (metodo antirez: template = metadata dall'input,
 * byte rigenerati). §0bis: comportamento reale, niente stub.
 */
#include "llmmodelio_model.h"
#include "llmcodec_ggml_quants.h"      /* llmcodec_ggml_type_of_fmt, LLMCODEC_GGML_T_* */
#include "llmcodec_fp16.h"
#include "llmcodec_ggml_quants.h"   /* helper K-quant ggml (CORE codec, §0bis) */
#include "llmcodec_gguf_mlx.h"   /* llmcodec_conv_mlx_to_gguf_arch_ex / llmcodec_conv_gguf_to_mlx_arch (progetto codec) */
#include "llmtoken.h"   /* unica sorgente tokenizer.json (export per i metadata GGUF) */
#include "llmcodec_gguf_writer.h"
#include "llmcodec_safetensors.h"   /* adapter LoRA (.safetensors) */
#include "llmcodec_iq2_xxs.h"
#include "llmmodelio_imatrix.h"
#include "llmmodelio_infer.h"
#include "llmengine_moe_quant_alloc.h"



/* AUTO (§E3): tabella formato per-tensore decisa dall'allocatore, consultata dal quantize. */
static const int* g_auto_type = (const int*)0;
double llmmodelio_roundtrip_rel(int ty, const float* x, uint64_t numel, uint32_t cols, const float* imp);
double llmmodelio_roundtrip_rel_cov(int ty, const float* x, uint64_t numel, uint32_t cols,
                             const float* samp, int64_t n_samp, int64_t rows_per_expert);

#include <stdio.h>

/* Output delle funzioni di ispezione (ARCH/POLICY): di norma stdout; se `g_llm_out` e' impostato
 * (fmemopen) la scrittura va nel buffer — cosi' i backend gestiti (IL/PCodeVM via P/Invoke), che
 * non catturano lo stdout nativo del processo C, ricevono il testo e lo stampano col proprio
 * console. Stesso comportamento dei backend C (VM/C, C99*), che stampano diretto. */
static FILE* g_llm_out = NULL;
#define LOUT (g_llm_out ? g_llm_out : stdout)
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dlfcn.h>   /* LLM QUALITY: engine MLX via dlopen (dipendenza debole) */   /* mkdir (checkpoint store) */   /* getcwd (arch-resolution da file .gguf) */

static uint64_t llm_numel(const uint64_t* dims, int ndim) {
    uint64_t numel = 1; for (int d = 0; d < ndim; d++) numel *= dims[d]; return numel;
}
/* 0 = "non calcolabile" (tipo ignoto o numel non multiplo del blocco): i chiamatori qui sotto
 * già trattano 0 come errore (`b == 0 -> rc = -1`), e la distinzione fra le due cause non serve
 * a loro — serve a chi deve DARE un messaggio, che chiama llmcodec_ggml_type_bytes direttamente. */
static uint64_t llm_ts_bytes(int ggml_type, const uint64_t* dims, int ndim) {
    uint64_t numel = 1, b = 0;
    for (int d = 0; d < ndim; d++) numel *= dims[d];
    return llmcodec_ggml_type_bytes(ggml_type, numel, &b) == 0 ? b : 0;
}

int llmmodelio_model_save(const char* in_path, const char* out_path) {
    if (in_path == NULL || out_path == NULL) return -1;

    LlmCodecGgufFileInfo fi;
    LlmCodecGgufMetaKVOwned* kv = NULL;
    int nkv = 0;
    if (llmcodec_gguf_read_meta(in_path, &kv, &nkv, &fi) != 0) return -1;
    if (fi.n_tensors == 0) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }

    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)fi.n_tensors, sizeof(*ts));
    uint64_t* dims = (uint64_t*)calloc((size_t)fi.n_tensors * 8, sizeof(uint64_t));
    LlmCodecGgufMetaKV* wkv = (LlmCodecGgufMetaKV*)calloc((size_t)(nkv ? nkv : 1), sizeof(*wkv));
    if (ts == NULL || dims == NULL || wkv == NULL) {
        free(ts); free(dims); free(wkv); llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1;
    }
    FILE* src = fopen(in_path, "rb");
    if (src == NULL) { free(ts); free(dims); free(wkv); llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }

    int rc = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        LlmCodecGgufTensorInfo* ti = &fi.tensors[i];
        ts[i].name = ti->name;
        ts[i].ggml_type = ti->ggml_type;
        ts[i].ndim = ti->ndim;
        ts[i].dims = dims + i * 8;
        for (int d = 0; d < ti->ndim; d++) ((uint64_t*)ts[i].dims)[d] = ti->dims[d];
        uint64_t b = llm_ts_bytes(ti->ggml_type, ti->dims, ti->ndim);
        if (getenv("LLMMODELIO_DEBUG")) fprintf(stderr, "[save] i=%llu name=%s type=%d ndim=%d b=%llu\n", (unsigned long long)i, ti->name, ti->ggml_type, ti->ndim, (unsigned long long)b);
        if (b == 0 || b > (1ull << 34)) { rc = -1; break; }   /* tipo non supportato / >16GB */
        void* buf = malloc((size_t)b);
        if (buf == NULL) { rc = -1; break; }
        if (fseek(src, (long)(fi.data_start + ti->offset), SEEK_SET) != 0 ||
            fread(buf, 1, (size_t)b, src) != (size_t)b) { free(buf); rc = -1; break; }
        ((LlmCodecGgufTensor*)ts)[i].data = buf;   /* ownership: liberato sotto */
    }
    if (rc == 0) for (int i = 0; i < nkv; i++) llmcodec_gguf_meta_to_writer(&kv[i], &wkv[i]);
    if (rc == 0) rc = llmcodec_gguf_write(out_path, wkv, nkv, ts, (int)fi.n_tensors, 32);
    if (getenv("LLMMODELIO_DEBUG")) fprintf(stderr, "[save] write rc=%d\n", rc);

    for (uint64_t i = 0; i < fi.n_tensors; i++) free((void*)ts[i].data);
    fclose(src);
    free(ts); free(dims); free(wkv);
    llmcodec_gguf_meta_free(kv, nkv);
    llmcodec_gguf_free_info(&fi);
    return rc == 0 ? 0 : -1;
}

/* ---- QUANTIZE: F32 -> Q8_0 (block_q8_0: F16 scale + 32 int8 = 34 byte / 32 valori) ---- */
#include <math.h>

/* h2f: delega all'atomo header-only di llmcodec_fp16.h (§0bis: una sola sorgente). */
static float llmmodelio_h2f(uint16_t h) { return llmcodec_f16_to_f32(h); }
uint16_t llmmodelio_fp32_to_fp16(float f);

uint16_t llmmodelio_fp32_to_fp16(float f) { return llmcodec_f16_from_f32(f); }

/* q8_0: delega al codec GGML (§0bis). round-to-nearest di llama.cpp, non lrintf. */
static void llmmodelio_q8_0(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_Q8_0, x, out, n, NULL);
}

/* Round-trip con QUANTIZE: i tensori F32 con ultima dim multipla di 32 -> Q8_0 (scritti in
 * formato GGUF Q8_0), gli altri copiati invariati. 0 ok / -1. */
static const char* llmmodelio_role_of(const char* name);   /* definita sotto (PLAN) */
/* MXFP4 (OCP microscaling, ggml tipo 39): blocco = {e:E8M0, qs[16]} = 17B/32.
 * Port da quantize_row_mxfp4_ref: e = floor(log2(amax))-2+127; d = 2^(e-128) (kvalues sono DOUBLED);
 * q = argmin |kvalues[i]*d - x|; packing a META' (low=j, high=j+16). */
/* NVFP4 (ggml tipo 40): blocco = {d[4]:UE4M3, qs[32]} = 36B/64; 4 sotto-blocchi di 16; scala UE4M3
 * (E4M3 unsigned) con amax/6; stesso lookup FP4 di MXFP4. Port di quantize_row_nvfp4_ref + le UE4M3
 * da ggml-impl.h (verbatim). */

static uint8_t llmmodelio_f32_to_ue4m3(float x) { return llmcodec_ue4m3_from_f32(x); }
/* nvfp4: delega al codec GGML (§0bis). */
static void llmmodelio_nvfp4(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_NVFP4, x, out, n, NULL);
}
/* mxfp4: delega al codec GGML (§0bis). */
static void llmmodelio_mxfp4(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_MXFP4, x, out, n, NULL);
}
/* bf16: delega all'atomo header-only di llmcodec_fp16.h (§0bis: una sola sorgente). */
static void llmmodelio_bf16(const float* x, int64_t n, uint8_t* out) {
    uint16_t* o = (uint16_t*)out;
    for (int64_t i = 0; i < n; i++) o[i] = llmcodec_bf16_from_f32(x[i]);
}
static void llmmodelio_q4_0(const float* x, int64_t n, uint8_t* out);   /* definita sotto */
static void llm_q4_K(const float* x, int64_t n, uint8_t* out);   /* definita sotto */
static void llm_q2_K(const float* x, int64_t n, uint8_t* out);   /* definita sotto */
static void llm_q6_K(const float* x, int64_t n, uint8_t* out);
static void llm_q5_K(const float* x, int64_t n, uint8_t* out);
static void llm_q3_K(const float* x, int64_t n, uint8_t* out);
static int llmmodelio_K_imatrix(const float* x, int64_t n, uint8_t* out, const float* qw,
                         int64_t row_len, int64_t weight_sets, int bpb, int is_q4);
/* ABLATE (model surgery: MODIFICARE) — azzera i tensori il cui nome contiene `pattern`
 * (per-ruolo/esperto, es. "wo_b", "layers.17.mlp"). Byte-zero: F32/F16 -> 0.0 esatto;
 * quantizzati -> scale 0 -> dequant ~0. Scrive un GGUF nuovo (metadata preservati).
 * Se nessun tensore combacia -> -1 (mai un risultato silenzioso). */
int llmmodelio_model_ablate(const char* in_path, const char* out_path, const char* pattern) {
    if (in_path == NULL || out_path == NULL || pattern == NULL || !*pattern) return -1;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(in_path, &kv, &nkv, &fi) != 0) return -1;
    if (fi.n_tensors == 0) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }
    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)fi.n_tensors, sizeof(*ts));
    uint64_t* dims = (uint64_t*)calloc((size_t)fi.n_tensors * 8, sizeof(uint64_t));
    LlmCodecGgufMetaKV* wkv = (LlmCodecGgufMetaKV*)calloc((size_t)(nkv ? nkv : 1), sizeof(*wkv));
    FILE* src = fopen(in_path, "rb");
    if (ts == NULL || dims == NULL || wkv == NULL || src == NULL) {
        free(ts); free(dims); free(wkv); if (src) fclose(src);
        llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1;
    }
    int rc = 0; uint64_t hit = 0;
    for (uint64_t i = 0; i < fi.n_tensors && rc == 0; i++) {
        LlmCodecGgufTensorInfo* ti = &fi.tensors[i];
        uint64_t numel = llm_numel(ti->dims, ti->ndim);
        ts[i].name = ti->name; ts[i].ggml_type = ti->ggml_type; ts[i].ndim = ti->ndim; ts[i].dims = dims + i*8;
        for (int d = 0; d < ti->ndim; d++) ((uint64_t*)ts[i].dims)[d] = ti->dims[d];
        uint64_t b = 0;
        if (llmcodec_ggml_type_bytes(ti->ggml_type, numel, &b) != 0 || b == 0) { rc = -1; break; }
        void* buf = malloc((size_t)b);
        if (buf == NULL) { rc = -1; break; }
        if (fseek(src, (long)(fi.data_start + ti->offset), SEEK_SET) != 0 || fread(buf, 1, (size_t)b, src) != (size_t)b) { free(buf); rc = -1; break; }
        if (strstr(ti->name, pattern) != NULL) { memset(buf, 0, (size_t)b); hit++; }
        ts[i].data = buf;
    }
    if (rc == 0 && hit == 0) rc = -1;   /* pattern inesistente -> errore esplicito */
    if (rc == 0) for (int i = 0; i < nkv; i++) llmcodec_gguf_meta_to_writer(&kv[i], &wkv[i]);
    if (rc == 0) rc = llmcodec_gguf_write(out_path, wkv, nkv, ts, (int)fi.n_tensors, 32);
    for (uint64_t i = 0; i < fi.n_tensors; i++) free((void*)ts[i].data);
    fclose(src); free(ts); free(dims); free(wkv);
    llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
    return rc == 0 ? 0 : -1;
}
/* ── AGGIUNGERE adapter (LoRA .safetensors): W += (alpha/r) * B@A ── */
static int llm_st_esize(const char* dt) {
    if (!strcmp(dt, "F32")) return 4; if (!strcmp(dt, "F16") || !strcmp(dt, "BF16")) return 2;
    if (!strcmp(dt, "F64") || !strcmp(dt, "I64")) return 8; if (!strcmp(dt, "I32")) return 4;
    if (!strcmp(dt, "I16")) return 2; if (!strcmp(dt, "I8") || !strcmp(dt, "U8") || !strcmp(dt, "BOOL")) return 1;
    return 0;
}
static float llm_st_elem_f32(const unsigned char* p, const char* dt) {
    if (!strcmp(dt, "F32")) { float f; memcpy(&f, p, 4); return f; }
    if (!strcmp(dt, "F16")) { uint16_t h; memcpy(&h, p, 2); return llmcodec_f16_to_f32(h); }
    if (!strcmp(dt, "BF16")) { uint16_t h; memcpy(&h, p, 2); return llmcodec_bf16_to_f32(h); }
    if (!strcmp(dt, "F64")) { double d; memcpy(&d, p, 8); return (float)d; }
    return 0.0f;
}
static float* llm_st_load_f32(FILE* fp, uint64_t data_base, const LlmCodecStTensorInfo* t, uint64_t* out_numel) {
    uint64_t numel = 1; for (int d = 0; d < t->ndim; d++) numel *= t->shape[d];
    int es = llm_st_esize(t->dtype); if (es <= 0) return NULL;
    uint64_t nbytes = numel * (uint64_t)es;
    if (t->end < t->begin || t->end - t->begin < nbytes) return NULL;
    unsigned char* buf = (unsigned char*)malloc((size_t)nbytes); if (buf == NULL) return NULL;
    if (fseek(fp, (long)(data_base + t->begin), SEEK_SET) != 0 || fread(buf, 1, (size_t)nbytes, fp) != (size_t)nbytes) { free(buf); return NULL; }
    float* f = (float*)malloc((size_t)(numel * sizeof(float))); if (f == NULL) { free(buf); return NULL; }
    for (uint64_t i = 0; i < numel; i++) f[i] = llm_st_elem_f32(buf + (size_t)i * es, t->dtype);
    free(buf); if (out_numel) *out_numel = numel; return f;
}
static int llm_st_find(const LlmCodecStTensorInfo* st, int n, const char* name) {
    for (int i = 0; i < n; i++) if (strcmp(st[i].name, name) == 0) return i;
    return -1;
}
static uint64_t llm_ggml_esize(int t) { return (t == LLMCODEC_GGML_T_F32) ? 4u : ((t == LLMCODEC_GGML_T_F16 || t == LLMCODEC_GGML_T_BF16) ? 2u : 0u); }
/* per una base tensor 2D, ritorna A/B nello store (o -1). */
static int llm_lora_pair(const LlmCodecStTensorInfo* st, int n, const char* base_name, int* ia, int* ib) {
    const char* suf = ".weight"; size_t bl = strlen(base_name), sl = strlen(suf);
    char prefix[256];
    if (bl > sl && strcmp(base_name + bl - sl, suf) == 0) { size_t pl = bl - sl; if (pl > 255) pl = 255; memcpy(prefix, base_name, pl); prefix[pl] = 0; }
    else { if (bl > 255) return -1; strcpy(prefix, base_name); }
    char a[600], b[600], a2[600], b2[600];
    snprintf(a, sizeof a, "%s.lora_A.weight", prefix); snprintf(b, sizeof b, "%s.lora_B.weight", prefix);
    snprintf(a2, sizeof a2, "base_model.model.%s.lora_A.weight", prefix); snprintf(b2, sizeof b2, "base_model.model.%s.lora_B.weight", prefix);
    int x = llm_st_find(st, n, a), y = llm_st_find(st, n, b);
    if (x < 0 || y < 0) { x = llm_st_find(st, n, a2); y = llm_st_find(st, n, b2); }
    if (x < 0 || y < 0) return -1;
    *ia = x; *ib = y; return 0;
}
int llmmodelio_model_adapter(const char* base_path, const char* adapter_path, const char* out_path, const char* alpha_str) {
    if (base_path == NULL || adapter_path == NULL || out_path == NULL) return -1;
    double alpha = alpha_str ? atof(alpha_str) : 1.0;
    LlmCodecGgufFileInfo fb; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(base_path, &kv, &nkv, &fb) != 0) return -1;
    LlmCodecStTensorInfo* st = NULL; int nst = 0; uint64_t dbase = 0;
    if (llmcodec_safetensors_read(adapter_path, &st, &nst, &dbase) != 0) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fb); return -1; }
    uint64_t cap = fb.n_tensors;
    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)cap, sizeof(*ts));
    uint64_t* dims = (uint64_t*)calloc((size_t)cap * 8, sizeof(uint64_t));
    LlmCodecGgufMetaKV* wkv = (LlmCodecGgufMetaKV*)calloc((size_t)(nkv ? nkv : 1), sizeof(*wkv));
    FILE* bfp = fopen(base_path, "rb"); FILE* afp = fopen(adapter_path, "rb");
    if (ts == NULL || dims == NULL || wkv == NULL || bfp == NULL || afp == NULL) {
        free(ts); free(dims); free(wkv); if (bfp) fclose(bfp); if (afp) fclose(afp);
        llmcodec_safetensors_info_free(st); llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fb); return -1;
    }
    int rc = 0; uint64_t nout = 0, merged = 0;
    for (uint64_t i = 0; i < fb.n_tensors && rc == 0; i++) {
        LlmCodecGgufTensorInfo* tb = &fb.tensors[i];
        uint64_t numel = llm_numel(tb->dims, tb->ndim), b = 0;
        if (llmcodec_ggml_type_bytes(tb->ggml_type, numel, &b) != 0 || b == 0) { rc = -1; break; }
        void* buf = malloc((size_t)b); if (buf == NULL) { rc = -1; break; }
        int did_merge = 0;
        int ia = -1, ib = -1;
        if (tb->ndim == 2 && llm_ggml_esize(tb->ggml_type) != 0 && llm_lora_pair(st, nst, tb->name, &ia, &ib) == 0) {
            uint64_t in = tb->dims[0], out = tb->dims[1];
            uint64_t An = 0, Bn = 0;
            float* A = llm_st_load_f32(afp, dbase, &st[ia], &An);
            float* B = llm_st_load_f32(afp, dbase, &st[ib], &Bn);
            if (A == NULL || B == NULL || st[ia].ndim != 2 || st[ib].ndim != 2
                || st[ia].shape[1] != in || st[ib].shape[0] != out || st[ia].shape[0] != st[ib].shape[1]
                || An != st[ia].shape[0] * in || Bn != out * st[ib].shape[1]) {
                free(A); free(B); rc = -1; free(buf); break;   /* coppia presente ma forma incoerente -> errore esplicito */
            }
            uint64_t r = st[ia].shape[0];
            if (fseek(bfp, (long)(fb.data_start + tb->offset), SEEK_SET) != 0 || fread(buf, 1, (size_t)b, bfp) != (size_t)b) { free(A); free(B); rc = -1; free(buf); break; }
            unsigned char* bb = (unsigned char*)buf; int et = tb->ggml_type;
            double scale = alpha / (double)(r ? r : 1);
            for (uint64_t o = 0; o < out; o++)
                for (uint64_t ii = 0; ii < in; ii++) {
                    double acc = 0.0;
                    for (uint64_t k = 0; k < r; k++) acc += (double)B[o * r + k] * (double)A[k * in + ii];
                    float cur;
                    if (et == LLMCODEC_GGML_T_F32) { float t; memcpy(&t, bb + (size_t)(o * in + ii) * 4, 4); cur = t; }
                    else if (et == LLMCODEC_GGML_T_F16) { uint16_t h; memcpy(&h, bb + (size_t)(o * in + ii) * 2, 2); cur = llmcodec_f16_to_f32(h); }
                    else { uint16_t h; memcpy(&h, bb + (size_t)(o * in + ii) * 2, 2); cur = llmcodec_bf16_to_f32(h); }
                    float nv = (float)((double)cur + scale * acc);
                    if (et == LLMCODEC_GGML_T_F32) memcpy(bb + (size_t)(o * in + ii) * 4, &nv, 4);
                    else if (et == LLMCODEC_GGML_T_F16) { uint16_t h = llmcodec_f16_from_f32(nv); memcpy(bb + (size_t)(o * in + ii) * 2, &h, 2); }
                    else { uint16_t h = llmcodec_bf16_from_f32(nv); memcpy(bb + (size_t)(o * in + ii) * 2, &h, 2); }
                }
            free(A); free(B); did_merge = 1; merged++;
        }
        if (!did_merge) {
            if (fseek(bfp, (long)(fb.data_start + tb->offset), SEEK_SET) != 0 || fread(buf, 1, (size_t)b, bfp) != (size_t)b) { free(buf); rc = -1; break; }
        }
        ts[nout].name = tb->name; ts[nout].ggml_type = tb->ggml_type; ts[nout].ndim = tb->ndim; ts[nout].dims = dims + nout * 8; ts[nout].data = buf;
        for (int d = 0; d < tb->ndim; d++) ((uint64_t*)ts[nout].dims)[d] = tb->dims[d];
        nout++;
    }
    if (rc == 0) for (int i = 0; i < nkv; i++) llmcodec_gguf_meta_to_writer(&kv[i], &wkv[i]);
    if (rc == 0) rc = llmcodec_gguf_write(out_path, wkv, nkv, ts, (int)nout, 32);
    for (uint64_t i = 0; i < nout; i++) free((void*)ts[i].data);
    fclose(bfp); fclose(afp); free(ts); free(dims); free(wkv);
    llmcodec_safetensors_info_free(st); llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fb);
    (void)merged;
    return rc == 0 ? 0 : -1;
}
/* AGGIUNGERE (model surgery): aggiunge/sovrascrive in <base> i tensori di <module> (per nome),
 * scrivendo un GGUF nuovo (<out>). Metadata da <base>. Building block di adapter/moduli.
 * Base/modulo non GGUF o illeggibili -> -1 (mai silenzioso). */
static LlmCodecGgufTensorInfo* llm_find_tensor(LlmCodecGgufFileInfo* fi, const char* name) {
    for (uint64_t i = 0; i < fi->n_tensors; i++) if (strcmp(fi->tensors[i].name, name) == 0) return &fi->tensors[i];
    return NULL;
}
int llmmodelio_model_add(const char* base_path, const char* module_path, const char* out_path) {
    if (base_path == NULL || module_path == NULL || out_path == NULL) return -1;
    LlmCodecGgufFileInfo fb, fm;
    LlmCodecGgufMetaKVOwned *kvb = NULL, *kvm = NULL; int nkvb = 0, nkvm = 0;
    if (llmcodec_gguf_read_meta(base_path, &kvb, &nkvb, &fb) != 0) return -1;
    if (llmcodec_gguf_read_meta(module_path, &kvm, &nkvm, &fm) != 0) {
        llmcodec_gguf_meta_free(kvb, nkvb); llmcodec_gguf_free_info(&fb); return -1;
    }
    uint64_t cap = fb.n_tensors + fm.n_tensors;
    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)cap, sizeof(*ts));
    uint64_t* dims = (uint64_t*)calloc((size_t)cap * 8, sizeof(uint64_t));
    LlmCodecGgufMetaKV* wkv = (LlmCodecGgufMetaKV*)calloc((size_t)(nkvb ? nkvb : 1), sizeof(*wkv));
    FILE* bfp = fopen(base_path, "rb"); FILE* mfp = fopen(module_path, "rb");
    if (ts == NULL || dims == NULL || wkv == NULL || bfp == NULL || mfp == NULL) {
        free(ts); free(dims); free(wkv); if (bfp) fclose(bfp); if (mfp) fclose(mfp);
        llmcodec_gguf_meta_free(kvb, nkvb); llmcodec_gguf_meta_free(kvm, nkvm);
        llmcodec_gguf_free_info(&fb); llmcodec_gguf_free_info(&fm); return -1;
    }
    int rc = 0; uint64_t nout = 0; uint64_t replaced = 0, added = 0;
    /* 1) tensori della base NON presenti nel modulo -> copiati dalla base. */
    for (uint64_t i = 0; i < fb.n_tensors && rc == 0; i++) {
        LlmCodecGgufTensorInfo* tb = &fb.tensors[i];
        if (llm_find_tensor(&fm, tb->name) != NULL) continue;   /* sara' sovrascritto dal modulo */
        uint64_t numel = llm_numel(tb->dims, tb->ndim), b = 0;
        if (llmcodec_ggml_type_bytes(tb->ggml_type, numel, &b) != 0 || b == 0) { rc = -1; break; }
        void* buf = malloc((size_t)b); if (buf == NULL) { rc = -1; break; }
        if (fseek(bfp, (long)(fb.data_start + tb->offset), SEEK_SET) != 0 || fread(buf, 1, (size_t)b, bfp) != (size_t)b) { free(buf); rc = -1; break; }
        ts[nout].name = tb->name; ts[nout].ggml_type = tb->ggml_type; ts[nout].ndim = tb->ndim; ts[nout].dims = dims + nout * 8; ts[nout].data = buf;
        for (int d = 0; d < tb->ndim; d++) ((uint64_t*)ts[nout].dims)[d] = tb->dims[d];
        nout++;
    }
    /* 2) tensori del modulo -> aggiunti/sovrascritti. */
    for (uint64_t j = 0; j < fm.n_tensors && rc == 0; j++) {
        LlmCodecGgufTensorInfo* tm = &fm.tensors[j];
        uint64_t numel = llm_numel(tm->dims, tm->ndim), b = 0;
        if (llmcodec_ggml_type_bytes(tm->ggml_type, numel, &b) != 0 || b == 0) { rc = -1; break; }
        void* buf = malloc((size_t)b); if (buf == NULL) { rc = -1; break; }
        if (fseek(mfp, (long)(fm.data_start + tm->offset), SEEK_SET) != 0 || fread(buf, 1, (size_t)b, mfp) != (size_t)b) { free(buf); rc = -1; break; }
        if (llm_find_tensor(&fb, tm->name) != NULL) replaced++; else added++;
        ts[nout].name = tm->name; ts[nout].ggml_type = tm->ggml_type; ts[nout].ndim = tm->ndim; ts[nout].dims = dims + nout * 8; ts[nout].data = buf;
        for (int d = 0; d < tm->ndim; d++) ((uint64_t*)ts[nout].dims)[d] = tm->dims[d];
        nout++;
    }
    if (rc == 0) for (int i = 0; i < nkvb; i++) llmcodec_gguf_meta_to_writer(&kvb[i], &wkv[i]);
    if (rc == 0) rc = llmcodec_gguf_write(out_path, wkv, nkvb, ts, (int)nout, 32);
    for (uint64_t i = 0; i < nout; i++) free((void*)ts[i].data);
    fclose(bfp); fclose(mfp); free(ts); free(dims); free(wkv);
    llmcodec_gguf_meta_free(kvb, nkvb); llmcodec_gguf_meta_free(kvm, nkvm);
    llmcodec_gguf_free_info(&fb); llmcodec_gguf_free_info(&fm);
    (void)replaced; (void)added;
    return rc == 0 ? 0 : -1;
}
/* ── Checkpoint/rollback: copia file + hash di integrita' (FNV-1a 64) ── */
static int llm_copy_file(const char* src, const char* dst) {
    FILE* a = fopen(src, "rb"); if (a == NULL) return -1;
    FILE* b = fopen(dst, "wb"); if (b == NULL) { fclose(a); return -1; }
    const size_t BUF = (size_t)1 << 20; unsigned char* buf = (unsigned char*)malloc(BUF);
    if (buf == NULL) { fclose(a); fclose(b); return -1; }
    size_t n; int rc = 0;
    while ((n = fread(buf, 1, BUF, a)) > 0) if (fwrite(buf, 1, n, b) != n) { rc = -1; break; }
    if (ferror(a)) rc = -1;
    free(buf); fclose(a); if (fclose(b) != 0) rc = -1;
    return rc;
}
static int llm_fnv64_file(const char* path, unsigned long long* out_size, unsigned long long* out_hash) {
    FILE* f = fopen(path, "rb"); if (f == NULL) return -1;
    const size_t BUF = (size_t)1 << 20; unsigned char* buf = (unsigned char*)malloc(BUF);
    if (buf == NULL) { fclose(f); return -1; }
    unsigned long long h = 1469598103934665603ULL, sz = 0; size_t n;
    while ((n = fread(buf, 1, BUF, f)) > 0) { for (size_t i = 0; i < n; i++) { h ^= buf[i]; h *= 1099511628211ULL; } sz += n; }
    int err = ferror(f); free(buf); fclose(f);
    if (err) return -1;
    if (out_size) *out_size = sz; if (out_hash) *out_hash = h;
    return 0;
}
/* nome-checkpoint sicuro (nessun path traversal). */
static int llm_name_safe(const char* n) {
    if (n == NULL || !*n) return 0;
    if (strcmp(n, ".") == 0 || strcmp(n, "..") == 0) return 0;
    for (const char* p = n; *p; p++) if (*p == '/' || *p == '\\') return 0;
    return 1;
}
/* CHECKPOINT: snapshot di <src> nel checkpoint-store <store_dir> con nome <name>
 * (+ <name>.meta con size/fnv64 = integrita' per il rollback). */
int llmmodelio_model_checkpoint(const char* src, const char* store_dir, const char* name) {
    if (src == NULL || store_dir == NULL || name == NULL || !llm_name_safe(name)) return -1;
    unsigned long long sz = 0, h = 0;
    if (llm_fnv64_file(src, &sz, &h) != 0) return -1;   /* src inesistente/illeggibile -> errore */
    mkdir(store_dir, 0777);   /* -1 se esiste: ok (single-level store) */
    char ckpt[4096], meta[4096];
    snprintf(ckpt, sizeof ckpt, "%s/%s.ckpt", store_dir, name);
    snprintf(meta, sizeof meta, "%s/%s.meta", store_dir, name);
    if (llm_copy_file(src, ckpt) != 0) return -1;
    FILE* m = fopen(meta, "w"); if (m == NULL) return -1;
    fprintf(m, "name=%s\nsize=%llu\nfnv64=%016llx\nsrc=%s\n", name, sz, h, src);
    fclose(m);
    return 0;
}
/* ROLLBACK: ripristina <store_dir>/<name>.ckpt su <dst>, verificando l'integrita' col meta. */
int llmmodelio_model_rollback(const char* store_dir, const char* name, const char* dst) {
    if (store_dir == NULL || name == NULL || dst == NULL || !llm_name_safe(name)) return -1;
    char ckpt[4096], meta[4096];
    snprintf(ckpt, sizeof ckpt, "%s/%s.ckpt", store_dir, name);
    snprintf(meta, sizeof meta, "%s/%s.meta", store_dir, name);
    unsigned long long sz = 0, h = 0;
    if (llm_fnv64_file(ckpt, &sz, &h) != 0) return -1;   /* checkpoint assente */
    FILE* m = fopen(meta, "r");
    if (m != NULL) {
        char line[512]; unsigned long long msz = 0, mh = 0;
        while (fgets(line, sizeof line, m)) {
            unsigned long long v = 0;
            if (sscanf(line, "size=%llu", &v) == 1) msz = v;
            if (sscanf(line, "fnv64=%llx", &v) == 1) mh = v;
        }
        fclose(m);
        if (msz != 0 && (msz != sz || mh != h)) return -1;   /* checkpoint corrotto -> errore */
    }
    return llm_copy_file(ckpt, dst);
}
/* DIFF (model surgery: checkpoint/rollback + quality gate) — confronta due artefatti GGUF:
 * tensori presenti solo in A/B, e per i comuni differenze di tipo/forma/contenuto (byte).
 * Scrive un report testuale in out_path. A/B non GGUF o illeggibili -> -1 (mai silenzioso). */
/* VALIDATE (model surgery: integrita' strutturale di un artefatto) — per ogni tensore: forma/
 * tipo calcolabili e dati dentro il file; + conteggio kv. Report in out_path; issues>0 -> -1. */
int llmmodelio_model_validate(const char* path, const char* out_path) {
    if (path == NULL || out_path == NULL) return -1;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    if (fi.n_tensors == 0) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }
    long fsz = 0; { FILE* f = fopen(path, "rb"); if (f) { fseek(f, 0, SEEK_END); fsz = ftell(f); fclose(f); } }
    FILE* o = fopen(out_path, "w");
    if (o == NULL) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }
    fprintf(o, "# LLM MODEL VALIDATE %s\n", path);
    fprintf(o, "tensors=%llu kv=%d data_start=%llu file_size=%ld\n",
            (unsigned long long)fi.n_tensors, nkv, (unsigned long long)fi.data_start, fsz);
    int issues = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        LlmCodecGgufTensorInfo* t = &fi.tensors[i];
        uint64_t numel = 1; for (int d = 0; d < t->ndim; d++) numel *= t->dims[d];
        uint64_t b = 0; const char* why = NULL;
        if (t->ndim <= 0) why = "ndim<=0";
        else if (numel == 0) why = "numel=0";
        else if (llmcodec_ggml_type_bytes(t->ggml_type, numel, &b) != 0 || b == 0) why = "tipo/byte non calcolabile";
        else if (fsz > 0 && (long)(fi.data_start + t->offset + b) > fsz) why = "offset+dati oltre la fine del file";
        if (why) { issues++; fprintf(o, "ISSUE %s: %s\n", t->name, why); }
    }
    fprintf(o, "SUMMARY tensors=%llu issues=%d -> %s\n", (unsigned long long)fi.n_tensors, issues, issues ? "INVALID" : "VALID");
    fclose(o);
    llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
    return issues ? -1 : 0;
}
int llmmodelio_model_diff(const char* a_path, const char* b_path, const char* out_path) {
    if (a_path == NULL || b_path == NULL || out_path == NULL) return -1;
    LlmCodecGgufFileInfo fa, fb;
    LlmCodecGgufMetaKVOwned* kva = NULL; LlmCodecGgufMetaKVOwned* kvb = NULL; int nkva = 0, nkvb = 0;
    if (llmcodec_gguf_read_meta(a_path, &kva, &nkva, &fa) != 0) return -1;
    if (llmcodec_gguf_read_meta(b_path, &kvb, &nkvb, &fb) != 0) {
        llmcodec_gguf_meta_free(kva, nkva); llmcodec_gguf_free_info(&fa); return -1;
    }
    FILE* os = fopen(out_path, "w");
    FILE* afp = fopen(a_path, "rb");
    FILE* bfp = fopen(b_path, "rb");
    if (os == NULL || afp == NULL || bfp == NULL) {
        if (os) fclose(os); if (afp) fclose(afp); if (bfp) fclose(bfp);
        llmcodec_gguf_meta_free(kva, nkva); llmcodec_gguf_meta_free(kvb, nkvb);
        llmcodec_gguf_free_info(&fa); llmcodec_gguf_free_info(&fb); return -1;
    }
    uint64_t eq = 0, differ = 0, only_a = 0, only_b = 0, bytes_differ = 0;
    for (uint64_t i = 0; i < fa.n_tensors; i++) {
        LlmCodecGgufTensorInfo* ta = &fa.tensors[i];
        LlmCodecGgufTensorInfo* tb = NULL;
        for (uint64_t j = 0; j < fb.n_tensors; j++)
            if (strcmp(fa.tensors[i].name, fb.tensors[j].name) == 0) { tb = &fb.tensors[j]; break; }
        if (tb == NULL) { only_a++; fprintf(os, "ONLY_A %s\n", ta->name); continue; }
        if (ta->ggml_type != tb->ggml_type || ta->ndim != tb->ndim) {
            differ++; fprintf(os, "META_DIFF %s (type %d/%d ndim %d/%d)\n", ta->name, ta->ggml_type, tb->ggml_type, ta->ndim, tb->ndim); continue;
        }
        int dims_ok = 1;
        for (int d = 0; d < ta->ndim; d++) if (ta->dims[d] != tb->dims[d]) dims_ok = 0;
        if (!dims_ok) { differ++; fprintf(os, "SHAPE_DIFF %s\n", ta->name); continue; }
        uint64_t b = 0, numel = llm_numel(ta->dims, ta->ndim);
        if (llmcodec_ggml_type_bytes(ta->ggml_type, numel, &b) != 0 || b == 0) { differ++; fprintf(os, "UNKNOWN_TYPE %s\n", ta->name); continue; }
        unsigned char* buf = (unsigned char*)malloc((size_t)b);
        unsigned char* buf2 = (unsigned char*)malloc((size_t)b);
        if (buf == NULL || buf2 == NULL) { free(buf); free(buf2); differ++; continue; }
        int ok = (fseek(afp, (long)(fa.data_start + ta->offset), SEEK_SET) == 0) && (fread(buf, 1, (size_t)b, afp) == (size_t)b)
              && (fseek(bfp, (long)(fb.data_start + tb->offset), SEEK_SET) == 0) && (fread(buf2, 1, (size_t)b, bfp) == (size_t)b);
        if (!ok) { free(buf); free(buf2); differ++; fprintf(os, "READ_ERR %s\n", ta->name); continue; }
        if (memcmp(buf, buf2, (size_t)b) == 0) eq++;
        else { differ++; bytes_differ += b; fprintf(os, "DIFF %s (%llu bytes)\n", ta->name, (unsigned long long)b); }
        free(buf); free(buf2);
    }
    for (uint64_t j = 0; j < fb.n_tensors; j++) {
        int found = 0;
        for (uint64_t i = 0; i < fa.n_tensors; i++) if (strcmp(fa.tensors[i].name, fb.tensors[j].name) == 0) { found = 1; break; }
        if (!found) { only_b++; fprintf(os, "ONLY_B %s\n", fb.tensors[j].name); }
    }
    fprintf(os, "SUMMARY tensors_a=%llu tensors_b=%llu equal=%llu differ=%llu only_a=%llu only_b=%llu bytes_differ=%llu\n",
            (unsigned long long)fa.n_tensors, (unsigned long long)fb.n_tensors, (unsigned long long)eq,
            (unsigned long long)differ, (unsigned long long)only_a, (unsigned long long)only_b, (unsigned long long)bytes_differ);
    fclose(afp); fclose(bfp); fclose(os);
    llmcodec_gguf_meta_free(kva, nkva); llmcodec_gguf_meta_free(kvb, nkvb);
    llmcodec_gguf_free_info(&fa); llmcodec_gguf_free_info(&fb);
    return 0;
}
int llmmodelio_model_quantize(const char* in_path, const char* out_path) {
    if (in_path == NULL || out_path == NULL) return -1;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(in_path, &kv, &nkv, &fi) != 0) return -1;
    if (fi.n_tensors == 0) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }

    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)fi.n_tensors, sizeof(*ts));
    uint64_t* dims = (uint64_t*)calloc((size_t)fi.n_tensors * 8, sizeof(uint64_t));
    LlmCodecGgufMetaKV* wkv = (LlmCodecGgufMetaKV*)calloc((size_t)(nkv ? nkv : 1), sizeof(*wkv));
    FILE* src = fopen(in_path, "rb");
    if (ts == NULL || dims == NULL || wkv == NULL || src == NULL) {
        free(ts); free(dims); free(wkv); if (src) fclose(src);
        llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1;
    }
    int rc = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        LlmCodecGgufTensorInfo* ti = &fi.tensors[i];
        int auto_ty = (g_auto_type && g_auto_type[i] >= 0) ? g_auto_type[i] : -1;   /* AUTO: formato deciso */
        ts[i].name = ti->name; ts[i].ndim = ti->ndim; ts[i].dims = dims + i * 8;
        for (int d = 0; d < ti->ndim; d++) ((uint64_t*)ts[i].dims)[d] = ti->dims[d];
        uint64_t last = ti->ndim > 0 ? ti->dims[0] : 0;   /* asse blocco = ne[0] (GGML) */
        const char* role = llmmodelio_role_of(ti->name);
        int isf32 = (ti->ggml_type == LLMCODEC_GGML_T_F32), isf16 = (ti->ggml_type == LLMCODEC_GGML_T_F16);
        int canq = (isf32 || isf16) && (last > 0) && (last % 32 == 0);
        static const char* efmt_env = NULL; static int efmt_init = 0;
        if (!efmt_init) { efmt_env = getenv("LLMMODELIO_EXPERT_FMT"); efmt_init = 1; }
        const char* pol = llmmodelio_model_policy_get("ROUTED_EXPERT");
        const char* efmt = pol ? pol : (efmt_env ? efmt_env : "q4k");
        int is_exp = canq && strcmp(role, "ROUTED_EXPERT") == 0;
        /* S1 (todo auto-migliorare): formato esperti PER-FASCIA — gli ultimi layer piu' sensibili.
         * LLMMODELIO_EXPERT_TAIL="<N>|<fmt>": gli ultimi N layer (blk.M., M>=n_layer-N) usano <fmt>,
         * gli altri il default efmt. Bassa dimensionalita' -> robusto al rumore. */
        { static int tail_n = -2; static char tail_fmt[16] = "q4k";
          if (tail_n == -2) { const char* t = getenv("LLMMODELIO_EXPERT_TAIL"); tail_n = 0;
              if (t) { const char* bar = strchr(t, '|'); if (bar) { tail_n = atoi(t); snprintf(tail_fmt, sizeof tail_fmt, "%s", bar+1); } } }
          if (tail_n > 0 && is_exp) { int lay = -1;
              if (strncmp(ti->name, "blk.", 4) == 0) lay = atoi(ti->name+4);
              if (lay >= 0) { int nlay = 16;
                  for (int k=0;k<nkv;k++) if (strstr(kv[k].key,"block_count") && kv[k].type==LLMCODEC_GGUF_MT_U32) nlay=(int)kv[k].u32;
                  if (lay >= nlay - tail_n) efmt = tail_fmt; } } }
        /* Ricetta-baseline (todo auto-migliorare): il down_exps e' piu' sensibile degli altri esperti
         * (la sua uscita entra diretta nel residuo). La baseline Q4_K_M usa Q6_K per il down e Q4_K per
         * gate/up. LLMMODELIO_DOWN_FMT (es. q6k) applica un formato dedicato ai soli *_down_exps. */
        { static const char* down_fmt = (const char*)-1;
          if (down_fmt == (const char*)-1) { down_fmt = getenv("LLMMODELIO_DOWN_FMT"); }
          if (down_fmt != NULL && is_exp && strstr(ti->name, "_down") != NULL) efmt = down_fmt; }
        int e4k = is_exp && (last % 256 == 0) && strcmp(efmt, "q4k") == 0;
        int e2k = is_exp && (last % 256 == 0) && strcmp(efmt, "q2k") == 0;
        int e6k = is_exp && (last % 256 == 0) && strcmp(efmt, "q6k") == 0;
        int e5k = is_exp && (last % 256 == 0) && strcmp(efmt, "q5k") == 0;
        int e3k = is_exp && (last % 256 == 0) && strcmp(efmt, "q3k") == 0;
        int eix = is_exp && (last % 256 == 0) && strcmp(efmt, "iq2xxs") == 0;
        int emx = is_exp && (last % 32 == 0) && strcmp(efmt, "mxfp4") == 0;
        int env = is_exp && (last % 64 == 0) && strcmp(efmt, "nvfp4") == 0;
        int q4 = canq && strcmp(role, "FFN") == 0 && (last % 32 == 0);                  /* FFN: Q4_0 */
        int q8 = canq && (strcmp(role, "ATTENTION") == 0 || strcmp(role, "SHARED_EXPERT") == 0
                          || strcmp(role, "OUTPUT") == 0);                         /* sensibili: Q8 */
        int emb16 = (isf32 || (isf16 && 0)) && strcmp(role, "EMBEDDING") == 0;     /* embedding -> F16 */
        if (auto_ty >= 0) {   /* AUTO (§E3): formato deciso per-tensore dall'allocatore a sensibilità */
            q4 = (auto_ty == 2); q8 = (auto_ty == 8); emb16 = 0;
            e2k = (auto_ty == 10); e3k = (auto_ty == 11); e4k = (auto_ty == 12);
            e5k = (auto_ty == 13); e6k = (auto_ty == 14); eix = (auto_ty == 16);
        }
        /* Ricetta baseline (todo auto-migliorare): attn e embedding con formato esplicito.
         * LLMMODELIO_ATTN_FMT="q6k" -> attenzione a Q6_K (la baseline non usa Q8_0); LLMMODELIO_EMBD_FMT="q5k"
         * -> embedding QUANTIZZATO (oggi escluso: resta F16, +206MB). Riusa i flag e4k/e5k/e6k. */
        { static const char* a_fmt=(const char*)-1, *e_fmt=(const char*)-1;
          if (a_fmt==(const char*)-1) { a_fmt=getenv("LLMMODELIO_ATTN_FMT"); e_fmt=getenv("LLMMODELIO_EMBD_FMT"); }
          const char* rf = NULL;
          if (strcmp(role,"ATTENTION")==0) rf=a_fmt; else if (strcmp(role,"EMBEDDING")==0) rf=e_fmt;
          if (rf!=NULL) { q8=0; emb16=0;
              if (last%256==0) { if(strcmp(rf,"q6k")==0) e6k=1; else if(strcmp(rf,"q4k")==0) e4k=1;
                                 else if(strcmp(rf,"q5k")==0) e5k=1; else if(strcmp(rf,"q3k")==0) e3k=1; }
              else if (strcmp(rf,"q8_0")==0) q8=1; } }
        /* RICETTA ESATTA Q4_K_M (dal dump della baseline, non per ruolo): pesi Q4_K; **attn_v** e
         * **ffn_down_exps** a Q6_K per la **META' FINALE** dei layer (il "medium" di llama.cpp); il
         * **output.weight** a Q6_K; token_embd Q4_K; norm/router F32. Evita il +9% del per-ruolo.
         * LLMMODELIO_RECIPE=q4km. */
        /* RICETTE llama.cpp (regole da llama-quant.cpp:llama_tensor_get_type_impl, verificate col dump):
         * base X_K; attn_v e ffn_down con boost per-fascia; output Q6_K; norm/router F32.
         * LLMMODELIO_RECIPE = q2k|q3km|q4ks|q4km|q5ks|q5km|q6k. */
        { static int s_rcp=-1; static char s_ft[8]="";
          if (s_rcp<0){ const char* r=getenv("LLMMODELIO_RECIPE"); s_rcp=(r&&strlen(r)>0)?1:0; if(s_rcp) snprintf(s_ft,sizeof s_ft,"%s",r); }
          if (s_rcp && canq && last%256==0 && ti->ndim >= 2 && strstr(ti->name,"gate_inp")==NULL) {
              int lay=-1; if (strncmp(ti->name,"blk.",4)==0) lay=atoi(ti->name+4);
              int nlay=16; for(int k=0;k<nkv;k++) if(strstr(kv[k].key,"block_count")&&kv[k].type==LLMCODEC_GGUF_MT_U32) nlay=(int)kv[k].u32;
              int more = (lay>=0) && (lay < nlay/8 || lay >= 7*nlay/8 || ((lay-nlay/8)%3==2));
              int is_v = strstr(ti->name,"attn_v.")!=NULL, is_down = strstr(ti->name,"_down_exps")!=NULL;
              int is_out = strcmp(ti->name,"output.weight")==0;
              int t = 0;   /* ggml type scelto */
              if (strcmp(s_ft,"q6k")==0) t = 14;
              else if (strcmp(s_ft,"q5ks")==0) t = is_out ? 14 : 13;
              else if (strcmp(s_ft,"q5km")==0) t = is_out ? 14 : (more && (is_v||is_down) ? 14 : 13);
              else if (strcmp(s_ft,"q4km")==0) t = is_out ? 14 : (more && (is_v||is_down) ? 14 : 12);
              else if (strcmp(s_ft,"q4ks")==0) t = is_out ? 14 : ((is_v && lay>=0 && lay<4) || (is_down && lay>=0 && lay<nlay/8) ? 13 : 12);
              else if (strcmp(s_ft,"q3km")==0) t = is_out ? 14 : (is_v ? (lay>=0 && lay<2 ? 13 : 12) : (is_down ? (lay>=0 && lay<1 ? 13 : 12) : 11));
              else if (strcmp(s_ft,"q2k")==0)  t = is_out ? 14 : ((is_v||is_down) ? 11 : 10);
              else t = -1;
              if (t > 0) { q4=0;q8=0;emb16=0;e2k=e3k=e5k=e6k=e4k=eix=0;
                  if(t==10)e2k=1; else if(t==11)e3k=1; else if(t==12)e4k=1; else if(t==13)e5k=1; else if(t==14)e6k=1; }
          } }
        /* LLMMODELIO_ROUTER_FMT=bf16: il router (ffn_gate_inp) a BF16 (antirez DS4 usa il router BF16 piatto). */
        { static int s_bft=-1; if(s_bft<0){ const char* r=getenv("LLMMODELIO_ROUTER_FMT"); s_bft=(r&&strcmp(r,"bf16")==0)?1:0; }
          if (s_bft && strstr(ti->name,"gate_inp")!=NULL && (isf32||isf16)) {
              uint64_t numel=1; for(int d=0;d<ti->ndim;d++) numel*=ti->dims[d];
              float* xf=(float*)malloc((size_t)numel*4);
              uint8_t* sraw=malloc((size_t)llm_ts_bytes(ti->ggml_type,ti->dims,ti->ndim));
              if(xf&&sraw){ FILE* fs=fopen(in_path,"rb");
                  if(fs){ fseek(fs,(long)(fi.data_start+ti->offset),SEEK_SET);
                      uint64_t sb=llm_ts_bytes(ti->ggml_type,ti->dims,ti->ndim);
                      if(fread(sraw,1,(size_t)sb,fs)==(size_t)sb){
                          if(isf32) memcpy(xf,sraw,(size_t)numel*4);
                          else { const uint16_t* hb=(const uint16_t*)sraw; for(uint64_t k=0;k<numel;k++) xf[k]=llmmodelio_h2f(hb[k]); }
                          uint8_t* q=(uint8_t*)malloc((size_t)numel*2); llmmodelio_bf16(xf,(int64_t)numel,q);
                          ts[i].name=ti->name; ts[i].ggml_type=30; ts[i].ndim=ti->ndim; ts[i].dims=dims+i*8;
                          for(int d=0;d<ti->ndim;d++)((uint64_t*)ts[i].dims)[d]=ti->dims[d];
                          ts[i].data=q; free(sraw); free(xf); fclose(fs); continue;
                      } fclose(fs); }
              } free(sraw); free(xf); }
        }
        int quant = q4 || e4k || e2k || e6k || e5k || e3k || eix || emx || env || q8 || emb16;
        uint64_t srcb = 0;
        { uint64_t numel = 1; for (int d = 0; d < ti->ndim; d++) numel *= ti->dims[d];
          if (llmcodec_ggml_type_bytes(ti->ggml_type, numel, &srcb) != 0) {
              /* Il messaggio è il punto: prima questo stesso caso ritornava 0 e finiva nel
               * `srcb == 0` qui sotto, identico a un tensore vuoto, senza dire QUALE tipo. */
              fprintf(stderr, "LLM MODEL: %s: tipo GGML %d con %llu elementi: dimensione non calcolabile\n",
                      ti->name, ti->ggml_type, (unsigned long long)numel);
              srcb = 0;
          } }
        if (srcb == 0 || srcb > (1ull << 34)) { rc = -1; break; }
        void* sraw = malloc((size_t)srcb);
        if (sraw == NULL) { rc = -1; break; }
        if (fseek(src, (long)(fi.data_start + ti->offset), SEEK_SET) != 0 ||
            fread(sraw, 1, (size_t)srcb, src) != (size_t)srcb) { free(sraw); rc = -1; break; }
        if (quant) {
            uint64_t numel = 1; for (int d = 0; d < ti->ndim; d++) numel *= ti->dims[d];
            /* sorgente -> float[] */
            float* xf = (float*)malloc((size_t)numel * sizeof(float));
            if (xf == NULL) { free(sraw); rc = -1; break; }
            if (isf32) memcpy(xf, sraw, (size_t)numel * 4);
            else { const uint16_t* hb = (const uint16_t*)sraw; for (uint64_t k = 0; k < numel; k++) xf[k] = llmmodelio_h2f(hb[k]); }
            if (emb16) {                                  /* embedding -> F16 (2x) */
                uint16_t* hh = (uint16_t*)malloc((size_t)numel * 2);
                if (hh == NULL) { free(xf); free(sraw); rc = -1; break; }
                for (uint64_t k = 0; k < numel; k++) hh[k] = llmcodec_f16_from_f32(xf[k]);
                ts[i].ggml_type = LLMCODEC_GGML_T_F16; ts[i].data = hh;
                free(xf); free(sraw);
            } else if (env) {   /* esperti -> NVFP4 (36B/64) */
                uint64_t qb = ((numel + 63) / 64) * 36;
                uint8_t* q = (uint8_t*)malloc((size_t)qb);
                if (q == NULL) { free(xf); free(sraw); rc = -1; break; }
                llmmodelio_nvfp4(xf, (int64_t)numel, q);
                ts[i].ggml_type = 40; ts[i].data = q;
                free(xf); free(sraw);
            } else if (emx) {   /* esperti -> MXFP4 (17B/32) */
                uint64_t qb = ((numel + 31) / 32) * 17;
                uint8_t* q = (uint8_t*)malloc((size_t)qb);
                if (q == NULL) { free(xf); free(sraw); rc = -1; break; }
                llmmodelio_mxfp4(xf, (int64_t)numel, q);
                ts[i].ggml_type = 39; ts[i].data = q;
                free(xf); free(sraw);
            } else if (e2k || e4k || e6k || e5k || e3k || eix) {   /* esperti -> formato selezionato */
                int bpb = eix?66:e2k?84:e4k?144:e6k?210:e5k?176:110;
                int ty  = eix?16:e2k?10:e4k?12:e6k?14:e5k?13:11;
                uint64_t qb = ((numel + 255) / 256) * (uint64_t)bpb;
                uint8_t* q = (uint8_t*)malloc((size_t)qb);
                if (q == NULL) { free(xf); free(sraw); rc = -1; break; }
                const char* imx = llmmodelio_model_policy_get("IMATRIX"); if (imx == NULL) imx = getenv("LLMMODELIO_IMATRIX");
                int64_t rl = 0; float* wts = imx ? llmmodelio_imatrix_weights(imx, ts[i].name, &rl) : NULL;
                int64_t weight_sets = ti->ndim > 2 ? (int64_t)ti->dims[2] : 1;
                int imx_ok = (wts != NULL && last % 256 == 0 && weight_sets > 0 &&
                              rl == (int64_t)last*weight_sets && numel % last == 0);
                if (eix) llmcodec_iq2_xxs_quantize(xf, (int64_t)numel, q,
                                              imx_ok ? wts : NULL, imx_ok ? rl : 0, (int64_t)last);
                else if (e2k) llm_q2_K(xf, (int64_t)numel, q);
                else if (e4k) { if (imx_ok) llmmodelio_K_imatrix(xf, (int64_t)numel, q, wts, (int64_t)last, weight_sets, 144, 1); else llm_q4_K(xf, (int64_t)numel, q); }
                else if (e6k) llm_q6_K(xf, (int64_t)numel, q);
                else if (e5k) { if (imx_ok) llmmodelio_K_imatrix(xf, (int64_t)numel, q, wts, (int64_t)last, weight_sets, 176, 0); else llm_q5_K(xf, (int64_t)numel, q); }
                else llm_q3_K(xf, (int64_t)numel, q);
                free(wts);
                ts[i].ggml_type = ty; ts[i].data = q;
                free(xf); free(sraw);
            } else {
                uint64_t qb = ((numel + 31) / 32) * (q4 ? 18 : 34);
                uint8_t* q = (uint8_t*)malloc((size_t)qb);
                if (q == NULL) { free(xf); free(sraw); rc = -1; break; }
                if (q4) llmmodelio_q4_0(xf, (int64_t)numel, q);
                else    llmmodelio_q8_0(xf, (int64_t)numel, q);
                ts[i].ggml_type = q4 ? LLMCODEC_GGML_T_Q4_0 : LLMCODEC_GGML_T_Q8_0;
                ts[i].data = q;                        /* ownership */
                free(xf); free(sraw);
            }
        } else {
            ts[i].ggml_type = ti->ggml_type;
            ts[i].data = sraw;
        }
    }
    if (rc == 0) for (int i = 0; i < nkv; i++) llmcodec_gguf_meta_to_writer(&kv[i], &wkv[i]);
    if (rc == 0) rc = llmcodec_gguf_write(out_path, wkv, nkv, ts, (int)fi.n_tensors, 32);
    for (uint64_t i = 0; i < fi.n_tensors; i++) free((void*)ts[i].data);
    fclose(src); free(ts); free(dims); free(wkv);
    llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
    return rc == 0 ? 0 : -1;
}

/* ---- PLAN (model compiler): classifica i tensori per ruolo + formato proposto ----
 * Regole per NOME (pattern, ispirate alla ricetta DS4/antirez): gli esperti routed (la
 * maggioranza dei parametri) si comprimono aggressivamente; attention/shared/output restano
 * ad alta precisione; norm/router/stato esatti. Scrive un report per-tensore in out_path. */
static const char* llmmodelio_role_of(const char* name) {
    if (name == NULL) return "OTHER";
    if (strstr(name, "shared_expert") || strstr(name, "shexp")) return "SHARED_EXPERT";
    if (strstr(name, "_exps") || strstr(name, "exps.") || strstr(name, ".experts")) return "ROUTED_EXPERT";
    if (strstr(name, "router") || strstr(name, ".gate.") || strstr(name, "gate_inp")) return "ROUTER";
    /* NORM PRIMA di ATTENTION: 'attn_q_norm'/'output_norm' contengono 'norm' e vanno tenuti F32
     * (una norm quantizzata = modello rotto), anche se il nome contiene 'attn'. */
    if (strstr(name, "token_embd") || strstr(name, "patch_embd") || strstr(name, "position_embd")
        || strstr(name, "pos_embd") || strstr(name, "word_embeddings") || strstr(name, "embed_tokens")
        || strstr(name, "wte") || strstr(name, "wpe") || strstr(name, "embd")) return "EMBEDDING";
    if (strstr(name, "norm") || strstr(name, ".ln1") || strstr(name, ".ln2") || strstr(name, ".ln_")
        || strstr(name, "layer_norm") || strstr(name, "layernorm")) return "NORM";
    if (strstr(name, "attn") || strstr(name, "attention")) return "ATTENTION";
    if (strstr(name, "lm_head") || strstr(name, "output")) return "OUTPUT";
    if (strstr(name, "ssm") || strstr(name, "conv") || strstr(name, "dt_")) return "STATE";
    /* FFN/MLP denso (non-esperti): il 'knowledge' bulk di un modello denso -> quantizzabile. */
    if (strstr(name, "ffn_") || strstr(name, "mlp") || strstr(name, "fc1") || strstr(name, "fc2")
        || strstr(name, "feed_forward") || strstr(name, ".w1") || strstr(name, ".w2") || strstr(name, ".w3"))
        return "FFN";
    return "OTHER";
}
static const char* llmmodelio_proposed_of(const char* role) {
    if (strcmp(role, "ROUTED_EXPERT") == 0) return "IQ2_XXS|Q2_K";   /* bulk: aggressivo */
    if (strcmp(role, "EMBEDDING") == 0)      return "F16";            /* antirez: embedding F16 */
    if (strcmp(role, "SHARED_EXPERT") == 0) return "Q8_0";
    if (strcmp(role, "FFN") == 0)            return "Q8_0";
    if (strcmp(role, "ATTENTION") == 0)      return "Q8_0";
    if (strcmp(role, "OUTPUT") == 0)         return "Q8_0";
    if (strcmp(role, "ROUTER") == 0)         return "F32";
    if (strcmp(role, "NORM") == 0)           return "F32";
    if (strcmp(role, "STATE") == 0)          return "F32";
    return "F16";
}
/* Scrive il piano (testo per-tensore) in out_path. 0 ok / -1. */
int llmmodelio_model_plan(const char* in_path, const char* out_path) {
    if (in_path == NULL || out_path == NULL) return -1;
    LlmCodecGgufFileInfo fi;
    if (llmcodec_gguf_read_info(in_path, &fi) != 0) return -1;
    FILE* f = fopen(out_path, "w");
    if (f == NULL) { llmcodec_gguf_free_info(&fi); return -1; }
    fprintf(f, "# LLM MODEL PLAN (role/format proposto) — %llu tensori\n", fi.n_tensors);
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        LlmCodecGgufTensorInfo* ti = &fi.tensors[i];
        const char* role = llmmodelio_role_of(ti->name);
        fprintf(f, "%s\trole=%s\ttype=%d\tproposed=%s\tndim=%d\n",
                ti->name, role, ti->ggml_type, llmmodelio_proposed_of(role), ti->ndim);
    }
    fclose(f);
    llmcodec_gguf_free_info(&fi);
    return 0;
}

/* ---- Q4_0: delega al codec GGML (§0bis). ---- */
static void llmmodelio_q4_0(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_Q4_0, x, out, n, NULL);
}

/* ---- POLICY (role -> formato): l'utente guida la scelta di formato per ruolo ---- */
#define LLMMODELIO_POLICY_MAX 16
static char s_pol_role[LLMMODELIO_POLICY_MAX][24];
static char s_pol_fmt[LLMMODELIO_POLICY_MAX][512];   /* i valori includono PATH (imatrix), non solo formati brevi */
static int  s_pol_n = 0;
void llmmodelio_model_policy_set(const char* role, const char* fmt) {
    if (role == NULL || fmt == NULL) return;
    for (int i = 0; i < s_pol_n; i++) if (strcmp(s_pol_role[i], role) == 0) { snprintf(s_pol_fmt[i], 512, "%s", fmt); return; }
    if (s_pol_n >= LLMMODELIO_POLICY_MAX) return;
    snprintf(s_pol_role[s_pol_n], 24, "%s", role);
    snprintf(s_pol_fmt[s_pol_n], 512, "%s", fmt);
    s_pol_n++;
}
void llmmodelio_model_policy_reset(void) { s_pol_n = 0; }
const char* llmmodelio_model_policy_get(const char* role) {
    if (role == NULL) return NULL;
    for (int i = 0; i < s_pol_n; i++) if (strcmp(s_pol_role[i], role) == 0) return s_pol_fmt[i];
    return NULL;
}
void llmmodelio_model_policy_show(void) {
    fprintf(LOUT, "LLM POLICY (role -> formato):\n");
    if (s_pol_n == 0) fprintf(LOUT, "  (nessuna regola esplicita: si usano i default)\n");
    for (int i = 0; i < s_pol_n; i++) fprintf(LOUT, "  %-16s -> %s\n", s_pol_role[i], s_pol_fmt[i]);
    fprintf(LOUT, "  default: ROUTED_EXPERT->q4k  FFN->q4_0  EMBEDDING->f16  ATTENTION->q8_0  NORM->f32\n");
}

/* get_scale_min_k4: decodifica scale/min di un sotto-blocco Q4_K (usata anche fuori dai kernel). */
static void llmmodelio_get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

/* ---- Q4_K: delega al codec GGML (§0bis). ---- */
static void llm_q4_K(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_Q4_K, x, out, n, NULL);
}

/* ---- Q2_K: delega al codec GGML (§0bis). ---- */
static void llm_q2_K(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_Q2_K, x, out, n, NULL);
}

/* ---- make_qx_quants (Q6_K) / make_q3_quants (Q3_K) — port fedele ggml ---- */
#define LLMMODELIO_GROUP_MAX_EPS 1e-15f



/* ---- Q6_K: delega al codec GGML (§0bis). ---- */
static void llm_q6_K(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_Q6_K, x, out, n, NULL);
}
/* ---- Q5_K: delega al codec GGML (§0bis). ---- */
static void llm_q5_K(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_Q5_K, x, out, n, NULL);
}
/* ---- Q3_K: delega al codec GGML (§0bis). ---- */
static void llm_q3_K(const float* x, int64_t n, uint8_t* out) {
    llmcodec_ggml_quant_row(LLMCODEC_GGML_T_Q3_K, x, out, n, NULL);
}

/* ---- Q4_K/Q5_K con imatrix: una riga per volta e una matrice pesi per esperto, come ggml. ---- */
static int llmmodelio_K_imatrix(const float* x, int64_t n, uint8_t* out, const float* qw,
                         int64_t row_len, int64_t weight_sets, int bpb, int is_q4) {
    if (!x || !out || row_len <= 0 || row_len % 256 != 0 || n % row_len != 0 ||
        weight_sets <= 0) return -1;
    int64_t nbl = row_len/256, nrows = n/row_len;
    if (nrows % weight_sets != 0) return -1;
    const int64_t rows_per_set = nrows/weight_sets;
    for (int64_t row = 0; row < nrows; row++)
        for (int64_t b = 0; b < nbl; b++) {
            const float* xb = x + row*row_len + b*256;
            uint8_t* ob = out + (row*nbl + b)*bpb;
            const float* wb = qw ? qw + (row/rows_per_set)*row_len + b*256 : NULL;
            llmcodec_ggml_quant_row(is_q4 ? LLMCODEC_GGML_T_Q4_K : LLMCODEC_GGML_T_Q5_K, xb, ob, 256, wb);
        }
    return 0;
}

int llmmodelio_quantize_q4_k_weighted(const float* x, int64_t nrows, int64_t n_per_row,
                                    uint8_t* out, const float* quant_weights, int64_t weight_sets) {
    if (nrows <= 0 || n_per_row <= 0 || nrows > INT64_MAX/n_per_row) return -1;
    return llmmodelio_K_imatrix(x, nrows*n_per_row, out, quant_weights, n_per_row, weight_sets, 144, 1);
}

/* ---- EXTRACT: estrae l'esperto `expert` del layer `layer` come FFN denso (fragment GGUF).
 * I tensori exps hanno dims [in,out,n_experts] (expert = ultima dim, blocco contiguo finale). ---- */
int llmmodelio_model_extract(const char* in_path, const char* out_path, int64_t layer, int64_t expert) {
    if (in_path == NULL || out_path == NULL) return -1;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(in_path, &kv, &nkv, &fi) != 0) return -1;
    char prefix[64]; snprintf(prefix, sizeof prefix, "blk.%lld.", (long long)layer);
    static const char* suf[4] = { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_inp.weight" };
    int idx[4] = { -1, -1, -1, -1 };
    for (uint64_t i = 0; i < fi.n_tensors; i++)
        for (int s = 0; s < 4; s++) { char want[256]; snprintf(want, sizeof want, "%s%s", prefix, suf[s]); if (strcmp(fi.tensors[i].name, want) == 0) idx[s] = (int)i; }
    if (idx[0] < 0 || idx[1] < 0 || idx[2] < 0) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }
    FILE* src = fopen(in_path, "rb");
    if (src == NULL) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }
    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc(4, sizeof(*ts));
    uint64_t* dims = (uint64_t*)calloc(4 * 8, sizeof(uint64_t));
    static char names[4][256];
    LlmCodecGgufMetaKV* wkv = (LlmCodecGgufMetaKV*)calloc((size_t)(nkv ? nkv : 1), sizeof(*wkv));
    int nout = 0, rc = 0;
    for (int s = 0; s < 4 && rc == 0; s++) {
        if (idx[s] < 0) continue;
        LlmCodecGgufTensorInfo* ti = &fi.tensors[idx[s]];
        uint64_t numel = 1; for (int d = 0; d < ti->ndim; d++) numel *= ti->dims[d];
        uint64_t E = ti->ndim > 0 ? ti->dims[ti->ndim-1] : 0;
        if (E == 0 || expert < 0 || (uint64_t)expert >= E) continue;
        /* Prima: `tot` 0 (tipo ignoto) → `per` 0 → `fread` di 0 byte RIESCE e produce un tensore
         * VUOTO in silenzio. Lo slicing dell'esperto è il punto in cui lo 0 diventava un file
         * scritto, non un errore: qui si ferma. NON è stata aggiunta la pretesa che `tot % E == 0`
         * (che prima non c'era): sarebbe severità nuova su un path non testabile qui. */
        uint64_t tot = 0, per = 0;
        if (llmcodec_ggml_type_bytes(ti->ggml_type, numel, &tot) != 0) {
            fprintf(stderr, "LLM MODEL EXPERT: %s: tipo GGML %d con %llu elementi: dimensione non calcolabile\n",
                    ti->name, ti->ggml_type, (unsigned long long)numel);
            rc = -1; break;
        }
        per = tot / E;
        if (per == 0) { rc = -1; break; }
        void* buf = malloc((size_t)per);
        if (buf == NULL) { rc = -1; break; }
        if (fseek(src, (long)(fi.data_start + ti->offset + (uint64_t)expert * per), SEEK_SET) != 0 || fread(buf, 1, (size_t)per, src) != (size_t)per) { free(buf); rc = -1; break; }
        char nm[256]; snprintf(nm, sizeof nm, "%s", ti->name);
        char* p = strstr(nm, "_exps"); if (p) memmove(p, p + 5, strlen(p + 5) + 1);
        snprintf(names[nout], sizeof names[nout], "%s", nm);
        ts[nout].name = names[nout]; ts[nout].ggml_type = ti->ggml_type;
        ts[nout].ndim = ti->ndim - 1; ts[nout].dims = dims + nout * 8;
        for (int d = 0; d < ti->ndim - 1; d++) ((uint64_t*)ts[nout].dims)[d] = ti->dims[d];
        ts[nout].data = buf;
        nout++;
    }
    if (rc == 0 && nout > 0) { for (int i = 0; i < nkv; i++) llmcodec_gguf_meta_to_writer(&kv[i], &wkv[i]); rc = llmcodec_gguf_write(out_path, wkv, nkv, ts, nout, 32); }
    for (int i = 0; i < nout; i++) free((void*)ts[i].data);
    fclose(src); free(ts); free(dims); free(wkv);
    llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
    return rc == 0 && nout > 0 ? 0 : -1;
}

/* ---- MERGE: media dei tensori (F32/F16) presenti in entrambi i modelli (model soup). ---- */
int llmmodelio_model_merge(const char* a_path, const char* b_path, const char* out_path) {
    if (a_path == NULL || b_path == NULL || out_path == NULL) return -1;
    LlmCodecGgufFileInfo fa, fb; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(a_path, &kv, &nkv, &fa) != 0) return -1;
    if (llmcodec_gguf_read_info(b_path, &fb) != 0) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fa); return -1; }
    FILE* sa = fopen(a_path, "rb"); FILE* sb = fopen(b_path, "rb");
    if (sa == NULL || sb == NULL) { if (sa) fclose(sa); if (sb) fclose(sb); llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fa); llmcodec_gguf_free_info(&fb); return -1; }
    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)fa.n_tensors, sizeof(*ts));
    uint64_t* dims = (uint64_t*)calloc((size_t)fa.n_tensors * 8, sizeof(uint64_t));
    LlmCodecGgufMetaKV* wkv = (LlmCodecGgufMetaKV*)calloc((size_t)(nkv ? nkv : 1), sizeof(*wkv));
    int rc = 0, merged = 0;
    for (uint64_t i = 0; i < fa.n_tensors && rc == 0; i++) {
        LlmCodecGgufTensorInfo* ti = &fa.tensors[i];
        ts[i].name = ti->name; ts[i].ggml_type = ti->ggml_type; ts[i].ndim = ti->ndim; ts[i].dims = dims + i * 8;
        for (int d = 0; d < ti->ndim; d++) ((uint64_t*)ts[i].dims)[d] = ti->dims[d];
        uint64_t b = 0;
        if (llmcodec_ggml_type_bytes(ti->ggml_type, llm_numel(ti->dims, ti->ndim), &b) != 0) { rc = -1; break; }
        if (b == 0) { rc = -1; break; }
        void* buf = malloc((size_t)b);
        if (buf == NULL) { rc = -1; break; }
        if (fseek(sa, (long)(fa.data_start + ti->offset), SEEK_SET) != 0 || fread(buf, 1, (size_t)b, sa) != (size_t)b) { free(buf); rc = -1; break; }
        /* match in B con stesso nome/tipo/dims -> media (solo F32=0 / F16=1) */
        int mb = -1;
        for (uint64_t j = 0; j < fb.n_tensors; j++) {
            LlmCodecGgufTensorInfo* tj = &fb.tensors[j];
            if (strcmp(tj->name, ti->name) != 0 || tj->ggml_type != ti->ggml_type || tj->ndim != ti->ndim) continue;
            int same = 1; for (int d = 0; d < ti->ndim; d++) if (tj->dims[d] != ti->dims[d]) same = 0;
            if (same) { mb = (int)j; break; }
        }
        if (mb >= 0 && (ti->ggml_type == 0 || ti->ggml_type == 1)) {
            LlmCodecGgufTensorInfo* tj = &fb.tensors[mb];
            uint64_t nb2 = 0;
            if (llmcodec_ggml_type_bytes(tj->ggml_type, llm_numel(tj->dims, tj->ndim), &nb2) != 0) nb2 = 0;
            void* buf2 = malloc((size_t)nb2);
            /* `nb2 > 0` è parte della guardia: un `malloc(0)` che ritorna non-NULL con un fread
             * di 0 byte "riesce", e la media sotto leggerebbe una zona vuota come se fosse il
             * tensore. (La media scatta solo per F32/F16, che sono sempre in tabella: qui è la
             * forma corretta della guardia, non un caso atteso.) */
            if (buf2 && nb2 > 0) {
                if (fseek(sb, (long)(fb.data_start + tj->offset), SEEK_SET) == 0 && fread(buf2, 1, (size_t)nb2, sb) == (size_t)nb2) {
                    uint64_t n = llm_numel(ti->dims, ti->ndim);
                    float* o = (float*)buf;
                    for (uint64_t k = 0; k < n; k++) {
                        float va = (ti->ggml_type == 0) ? o[k] : llmmodelio_h2f(((uint16_t*)buf)[k]);
                        float vb = (tj->ggml_type == 0) ? ((float*)buf2)[k] : llmmodelio_h2f(((uint16_t*)buf2)[k]);
                        float m = 0.5f*(va + vb);
                        if (ti->ggml_type == 0) o[k] = m; else ((uint16_t*)buf)[k] = llmcodec_f16_from_f32(m);
                    }
                    merged++;
                }
                free(buf2);
            }
        }
        ts[i].data = buf;
    }
    if (rc == 0) { for (int i = 0; i < nkv; i++) llmcodec_gguf_meta_to_writer(&kv[i], &wkv[i]); rc = llmcodec_gguf_write(out_path, wkv, nkv, ts, (int)fa.n_tensors, 32); }
    for (uint64_t i = 0; i < fa.n_tensors; i++) free((void*)ts[i].data);
    fclose(sa); fclose(sb); free(ts); free(dims); free(wkv);
    llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fa); llmcodec_gguf_free_info(&fb);
    if (getenv("LLMMODELIO_DEBUG")) fprintf(stderr, "[merge] tensori mediati: %d/%llu\n", merged, (unsigned long long)fa.n_tensors);
    return rc == 0 ? 0 : -1;
}

/* ---- PRUNE: riduce gli esperti a `keep` (i primi), con router COERENTE (stessa fetta) e
 * aggiornamento di `*.expert_count` / `*.expert_used_count`. I tensori esperti hanno l'esperto
 * nell'ultima dim: la fetta e' il prefisso contiguo. ---- */
/* Ritorna una copia (malloc) di `json` con il valore intero della chiave `"key"` -> `val`.
 * Usato dal PRUNE per aggiornare il config.json VERBATIM dopo la riduzione esperti. */
static char* llm_json_set_int(const char* json, const char* key, long val) {
    char pat[96]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* p = strstr(json, pat); if (!p) return NULL;
    p += strlen(pat);
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') ++p;
    if (*p != ':') return NULL; ++p;
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') ++p;
    const char* d0 = p; while (*p=='-'||(*p>='0'&&*p<='9')) ++p; const char* d1 = p;
    if (d0 == d1) return NULL;
    size_t pre = (size_t)(d0 - json), post = strlen(d1);
    char buf[32]; int bl = snprintf(buf, sizeof buf, "%ld", val);
    char* o = (char*)malloc(pre + (size_t)bl + post + 1); if (!o) return NULL;
    memcpy(o, json, pre); memcpy(o + pre, buf, (size_t)bl); memcpy(o + pre + bl, d1, post + 1);
    return o;
}
static int llm_is_expert_tensor(const char* name, int ndim, uint64_t last) {
    if (ndim < 2 || last == 0) return 0;
    return (strstr(name, "_exps") != NULL || strstr(name, "experts") != NULL
            || strstr(name, "gate_inp") != NULL || strstr(name, "expert") != NULL
            /* naming MLX/HF (modelli importati): esperti switch_mlp.*_proj, router mlp.gate. */
            || strstr(name, "switch_mlp") != NULL || strstr(name, ".mlp.gate.") != NULL);
}
/* PRUNE per IMPORTANZA: per ogni layer sceglie i `keep` esperti con la norma di colonna del
 * router piu' alta (proxy di uso), e fa una GATHER coerente (stessi indici) su esperti + router.
 * LLMMODELIO_PRUNE=first -> invece i primi N. Esps [in,out,E]: esperto = blocco contiguo finale. */
int llmmodelio_model_prune(const char* in_path, const char* out_path, int64_t keep) {
    if (in_path == NULL || out_path == NULL || keep <= 0) return -1;
    /* Criterio di selezione: POLICY("PRUNE_EXPERT") > LLMMODELIO_PRUNE > default "first".
 * Cosi' si sceglie da BASIC:  LLM POLICY SET "PRUNE_EXPERT", "norm_asc"  */
    int mode = 0; { const char* m = llmmodelio_model_policy_get("PRUNE_EXPERT");
                    if (m == NULL) m = getenv("LLMMODELIO_PRUNE"); if (m) { if (strcmp(m,"top")==0) mode=1; else if (strcmp(m,"bottom")==0) mode=2; else if (strcmp(m,"rand")==0) mode=3; else if (strcmp(m,"ctop")==0) mode=4; else if (strcmp(m,"cbot")==0) mode=5; else if (strcmp(m,"usage")==0) mode=6; } }
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(in_path, &kv, &nkv, &fi) != 0) return -1;
    uint64_t E = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        const LlmCodecGgufTensorInfo* ti = &fi.tensors[i];
        uint64_t last = ti->ndim > 0 ? ti->dims[ti->ndim-1] : 0;
        if (last && (strstr(ti->name, "_exps") || strstr(ti->name, "gate_inp") || strstr(ti->name, "switch_mlp"))) { E = last; break; }
    }
    if (E == 0 || (uint64_t)keep > E) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }
    FILE* src = fopen(in_path, "rb");
    if (src == NULL) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }
    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)fi.n_tensors, sizeof(*ts));
    uint64_t* dims = (uint64_t*)calloc((size_t)fi.n_tensors * 8, sizeof(uint64_t));
    LlmCodecGgufMetaKV* wkv = (LlmCodecGgufMetaKV*)calloc((size_t)(nkv ? nkv : 1), sizeof(*wkv));
    int rc = 0, pruned = 0;
    char cur_layer[64] = ""; int64_t sel[512]; int nsel = 0;
    for (uint64_t i = 0; i < fi.n_tensors && rc == 0; i++) {
        LlmCodecGgufTensorInfo* ti = &fi.tensors[i];
        uint64_t numel = llm_numel(ti->dims, ti->ndim);
        uint64_t last = ti->ndim > 0 ? ti->dims[ti->ndim-1] : 0;
        int eax = (ti->ndim > 0 && ti->dims[ti->ndim-1] == E) ? ti->ndim-1 : 0;
        int is_exp = llm_is_expert_tensor(ti->name, ti->ndim, last) && (last == E || ti->dims[0] == E);
        uint64_t b = 0;
        if (llmcodec_ggml_type_bytes(ti->ggml_type, numel, &b) != 0) {
            fprintf(stderr, "LLM MODEL: %s: tipo GGML %d con %llu elementi: dimensione non calcolabile\n",
                    ti->name, ti->ggml_type, (unsigned long long)numel);
            rc = -1; break;
        }
        if (b == 0) { rc = -1; break; }
        void* buf = malloc((size_t)b);
        if (buf == NULL) { rc = -1; break; }
        if (fseek(src, (long)(fi.data_start + ti->offset), SEEK_SET) != 0 || fread(buf, 1, (size_t)b, src) != (size_t)b) { free(buf); rc = -1; break; }
        ts[i].name = ti->name; ts[i].ggml_type = ti->ggml_type; ts[i].dims = dims + i*8; ts[i].ndim = ti->ndim;
        for (int d = 0; d < ti->ndim; d++) ((uint64_t*)ts[i].dims)[d] = ti->dims[d];
        if (is_exp) {
            /* layer corrente dal nome (blk.N.) */
            char lay[64] = ""; { const char* bp = strstr(ti->name, "blk."); if (bp) bp += 4; else { bp = strstr(ti->name, ".layers."); if (bp) bp += 8; }
                if (bp) { int k = 0; while (bp[k] && bp[k] != '.' && k < 62) { lay[k] = bp[k]; k++; } lay[k] = 0; } }
            if (strcmp(lay, cur_layer) != 0) {
                snprintf(cur_layer, sizeof cur_layer, "%s", lay); nsel = 0;
                char rn[256]; snprintf(rn, sizeof rn, "blk.%s.ffn_gate_inp.weight", lay);
                int ri = -1; for (uint64_t j = 0; j < fi.n_tensors; j++) if (strcmp(fi.tensors[j].name, rn) == 0) { ri = (int)j; break; }
                if (mode == 0 || ri < 0) { for (int64_t e = 0; e < keep; e++) sel[e] = e; nsel = (int)keep; }
                else {
                    if (mode == 6) {   /* USO REALE: counts del routing dall'imatrix per questo layer */
                        const char* imx = llmmodelio_model_policy_get("IMATRIX"); if (!imx) imx = getenv("LLMMODELIO_IMATRIX");
                        char xt[256]; snprintf(xt, sizeof xt, "blk.%s.ffn_gate_exps.weight", lay);
                        int64_t nu = 0; float* use = imx ? llmmodelio_imatrix_expert_usage(imx, xt, &nu) : NULL;
                        if (use && nu == (int64_t)E) {
                            for (int64_t a = 0; a < keep; a++) { int64_t best = -1; float bv = -1; for (int64_t e = 0; e < (int64_t)E; e++) { int u = 0; for (int q = 0; q < a; q++) if (sel[q] == e) { u = 1; break; } if (!u && use[e] > bv) { bv = use[e]; best = e; } } sel[a] = best < 0 ? a : best; }
                            nsel = (int)keep;
                            if (getenv("LLMMODELIO_DEBUG")) { fprintf(stderr, "[prune] layer %s USO:", lay); for (int q = 0; q < nsel; q++) fprintf(stderr, " %lld(%g)", (long long)sel[q], use[sel[q]]); fprintf(stderr, "\n"); }
                        }
                        free(use);
                        if (nsel == 0) { for (int64_t e = 0; e < keep; e++) sel[e] = e; nsel = (int)keep; }
                    } else {
                    LlmCodecGgufTensorInfo* rt = &fi.tensors[ri];
                    uint64_t rh = rt->ndim > 0 ? rt->dims[0] : 0;
                    float* rw = (float*)malloc((size_t)(rh * E) * 4);
                    if (rw) {
                        int r16 = (rt->ggml_type == 1), r32 = (rt->ggml_type == 0);
                        if (r16 || r32) {
                            uint64_t rb = rh*E*(r16?2:4); uint8_t* rb8 = malloc(rb);
                            if (rb8 && fseek(src,(long)(fi.data_start+rt->offset),SEEK_SET)==0 && fread(rb8,1,rb,src)==rb) {
                                if (mode == 4 || mode == 5) {   /* norma CENTRATA: ||col_e - media_col|| (toglie la componente comune) */
                                    double* mean = (double*)calloc(rh, sizeof(double));
                                    for (uint64_t hh = 0; hh < rh; hh++) { double s = 0; for (uint64_t e = 0; e < E; e++) s += r16? llmmodelio_h2f(((uint16_t*)rb8)[e*rh+hh]) : ((float*)rb8)[e*rh+hh]; mean[hh] = s/(double)E; }
                                    for (uint64_t e = 0; e < E; e++) { double s = 0; for (uint64_t hh = 0; hh < rh; hh++) { double v = (r16? llmmodelio_h2f(((uint16_t*)rb8)[e*rh+hh]) : ((float*)rb8)[e*rh+hh]) - mean[hh]; s += v*v; } rw[e] = (float)sqrt(s); }
                                    free(mean);
                                } else { for (uint64_t e = 0; e < E; e++) { double s = 0; for (uint64_t hh = 0; hh < rh; hh++) { float v = r16? llmmodelio_h2f(((uint16_t*)rb8)[e*rh+hh]) : ((float*)rb8)[e*rh+hh]; s += (double)v*v; } rw[e] = (float)sqrt(s); } }
                                if (mode == 4 || mode == 5) { int asc = (mode == 5); for (int64_t a = 0; a < keep; a++) { int64_t best = -1; float bv = asc ? 3.4e38f : -1.0f; for (uint64_t e = 0; e < E; e++) { int used = 0; for (int q = 0; q < a; q++) if (sel[q] == (int64_t)e) { used = 1; break; } if (used) continue; if (asc ? (rw[e] < bv) : (rw[e] > bv)) { bv = rw[e]; best = (int64_t)e; } } sel[a] = best < 0 ? a : best; } }
                                    else if (mode == 3) { unsigned st = 12345u; for (int64_t a = 0; a < keep; a++) { st = st*1103515245u + 12345u; uint64_t cand = (st >> 8) % E; int used = 0; for (int q = 0; q < a; q++) if (sel[q] == (int64_t)cand) { used = 1; break; } if (used) { a--; continue; } sel[a] = (int64_t)cand; } }
                                    else for (int64_t a = 0; a < keep; a++) { int64_t best = -1; float bv = (mode == 2) ? 3.4e38f : -1.0f; for (uint64_t e = 0; e < E; e++) { int used = 0; for (int q = 0; q < a; q++) if (sel[q] == (int64_t)e) { used = 1; break; } if (used) continue; if (mode == 2 ? (rw[e] < bv) : (rw[e] > bv)) { bv = rw[e]; best = (int64_t)e; } } sel[a] = best < 0 ? a : best; }
                                nsel = (int)keep;
                            }
                            free(rb8);
                        }
                        free(rw);
                    }
                    fseek(src, (long)(fi.data_start + ti->offset), SEEK_SET);   /* ripristina */
                    }
                }
                if (nsel == 0) { for (int64_t e = 0; e < keep; e++) sel[e] = e; nsel = (int)keep; }
                if (getenv("LLMMODELIO_DEBUG")) { fprintf(stderr, "[prune] layer %s sel:", lay); for (int q = 0; q < nsel; q++) fprintf(stderr, " %lld", (long long)sel[q]); fprintf(stderr, "\n"); }
            }
            /* gather: esps [d0,d1,E] con esperto = ultima dim (blocco contiguo) */
            uint64_t per = numel / E;
            uint64_t esz = (b / E);
            uint8_t* nb = (uint8_t*)malloc((size_t)(esz * (uint64_t)keep));
            if (nb == NULL) { free(buf); rc = -1; break; }
            for (int q = 0; q < nsel; q++) memcpy(nb + (size_t)q*esz, (uint8_t*)buf + (size_t)sel[q]*esz, (size_t)esz);
            (void)per;
            free(buf);
            ts[i].data = nb;
            ((uint64_t*)ts[i].dims)[eax] = (uint64_t)keep;
            pruned++;
        } else {
            ts[i].data = buf;
        }
    }
    if (rc == 0) {
        for (int i = 0; i < nkv; i++) {
            if (strstr(kv[i].key, ".expert_count") || strstr(kv[i].key, ".expert_used_count")) {
                if (kv[i].type == LLMCODEC_GGUF_MT_U32 && kv[i].u32 > (uint32_t)keep) kv[i].u32 = (uint32_t)keep;
                else if (kv[i].type == LLMCODEC_GGUF_MT_I32 && kv[i].i32 > (int32_t)keep) kv[i].i32 = (int32_t)keep;
            }
            if (strcmp(kv[i].key, "model.gmb.config") == 0 && kv[i].type == LLMCODEC_GGUF_MT_STRING && kv[i].str) {
                char* patched = llm_json_set_int(kv[i].str, "n_routed_experts", (long)keep);
                if (patched == NULL) patched = llm_json_set_int(kv[i].str, "num_experts", (long)keep);
                if (patched) kv[i].str = patched;
            }
            llmcodec_gguf_meta_to_writer(&kv[i], &wkv[i]);
        }
        rc = llmcodec_gguf_write(out_path, wkv, nkv, ts, (int)fi.n_tensors, 32);
    }
    for (uint64_t i = 0; i < fi.n_tensors; i++) free((void*)ts[i].data);
    fclose(src); free(ts); free(dims); free(wkv);
    llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
    if (getenv("LLMMODELIO_DEBUG")) fprintf(stderr, "[prune] E=%llu keep=%lld modo=%s pruned=%d rc=%d\n", (unsigned long long)E, (long long)keep, (mode==0)?"first":(mode==1)?"top":(mode==2)?"bottom":(mode==3)?"rand":(mode==4)?"ctop":(mode==5)?"cbot":"usage", pruned, rc);
    return rc == 0 ? 0 : -1;
}


/* ---- PROFILE CHE CALCOLA: forward sul corpus (token ids) -> conteggi di routing.
 * Chiude il cerchio BASIC: nessun imatrix esterno. ids_path: interi separati da spazi/virgole. ---- */
int llmmodelio_model_profile_compute(const char* model_path, const char* ids_path, const char* out_path, int64_t chunk) {
    if (model_path == NULL || ids_path == NULL || out_path == NULL) return -1;
    FILE* fi = fopen(ids_path, "r");
    if (fi == NULL) return -1;
    int cap = 1 << 16, n = 0; int32_t* toks = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    int v;
    while (fscanf(fi, "%d", &v) == 1) { if (n >= cap) { cap *= 2; toks = (int32_t*)realloc(toks, (size_t)cap*sizeof(int32_t)); } toks[n++] = v; }
    /* salta i separatori: fscanf("%d") ignora [,] e spazi */
    fclose(fi);
    if (n <= 0) { free(toks); return -1; }
    if (chunk <= 0) chunk = 512;
    int nl = 0, ne = 0; double* acc = NULL;
    for (int off = 0; off < n; off += (int)chunk) {
        int m = (n - off) < chunk ? (n - off) : (int)chunk;
        int a = 0, b = 0; double* c = llmmodelio_routing_counts(model_path, toks + off, m, &a, &b);
        if (c == NULL) { free(acc); free(toks); return -1; }
        if (acc == NULL) { acc = (double*)calloc((size_t)a * (size_t)b, sizeof(double)); nl = a; ne = b; }
        for (int k = 0; k < a * b; k++) acc[k] += c[k];
        free(c);
    }
    free(toks);
    if (acc == NULL) return -1;
    FILE* f = fopen(out_path, "w");
    if (f == NULL) { free(acc); return -1; }
    fprintf(f, "# LLM ROUTING PROFILE (CALCOLATO dal forward sui token di %s)\n", ids_path);
    double grand = 0;
    for (int l = 0; l < nl; l++) {
        double tot = 0; int zeros = 0;
        for (int e = 0; e < ne; e++) { tot += acc[(size_t)l*ne+e]; if (acc[(size_t)l*ne+e] <= 0.0) zeros++; }
        grand += tot;
        fprintf(f, "blk.%d  esperto: %d, attivazioni: %.0f, mai usati: %d\n", l, ne, tot, zeros);
        if (zeros) { fprintf(f, "  mai usati:"); for (int e = 0; e < ne; e++) if (acc[(size_t)l*ne+e] <= 0.0) fprintf(f, " %d", e); fprintf(f, "\n"); }
        fprintf(f, "  top-8:"); { double* cp = (double*)malloc((size_t)ne*sizeof(double)); memcpy(cp, acc+(size_t)l*ne, (size_t)ne*sizeof(double));
            for (int a2 = 0; a2 < 8 && a2 < ne; a2++) { int bi = -1; double bv = -1; for (int e = 0; e < ne; e++) if (cp[e] > bv) { bv = cp[e]; bi = e; } if (bi < 0) break; fprintf(f, " %d(%.0f)", bi, bv); cp[bi] = -1.0; } free(cp); }
        fprintf(f, "\n");
    }
    fprintf(f, "# layer: %d, attivazioni totali: %.0f\n", nl, grand);
    fclose(f);
    /* Scrivo anche il profilo come GGUF (i counts per layer) COSI' il prune lo legge via IMATRIX:
     * chiude il cerchio BASIC:  LLM PROFILE <model>,<ids>,<profile.gguf>  ->  IMATRIX=profile.gguf */
    {
        LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)nl, sizeof(*ts));
        uint64_t* dstore = (uint64_t*)calloc((size_t)nl * 2, sizeof(uint64_t));
        char (*names)[64] = (char(*)[64])calloc((size_t)nl, 64);
        float** datas = (float**)calloc((size_t)nl, sizeof(float*));
        LlmCodecGgufMetaKV mkv[2]; memset(mkv, 0, sizeof mkv);
        mkv[0].key = "general.architecture"; mkv[0].type = LLMCODEC_GGUF_MT_STRING; mkv[0].str = "routing-profile";
        mkv[1].key = "profile.n_layer"; mkv[1].type = LLMCODEC_GGUF_MT_U32; mkv[1].u32 = (uint32_t)nl;
        if (ts && dstore && names && datas) {
            for (int l = 0; l < nl; l++) {
                datas[l] = (float*)malloc((size_t)ne * sizeof(float));
                for (int e = 0; e < ne; e++) datas[l][e] = (float)acc[(size_t)l*ne+e];
                snprintf(names[l], 64, "blk.%d.ffn_gate_exps.weight.counts", l);
                dstore[l*2] = (uint64_t)ne; dstore[l*2+1] = 1;
                ts[l].name = names[l]; ts[l].ggml_type = 0; ts[l].ndim = 2; ts[l].dims = dstore + l*2; ts[l].data = datas[l];
            }
            if (llmcodec_gguf_write(out_path, mkv, 2, ts, nl, 32) == 0)
                fprintf(stderr, "[profile] scritto profilo routing GGUF: %s\n", out_path);
            for (int l = 0; l < nl; l++) free(datas[l]);
        }
        free(ts); free(dstore); free(names); free(datas);
    }
    free(acc);
    return 0;
}

/* ---- PROFILE: report del routing reale (uso per esperto dai counts dell'imatrix). Scrive un
 * file di testo: per ogni layer, esperti mai usati, top usati, totale. Rende ispezionabile da
 * BASIC la calibrazione che guida il prune. ---- */
int llmmodelio_model_profile(const char* model_path, const char* out_path) {
    if (model_path == NULL || out_path == NULL) return -1;
    const char* imx = llmmodelio_model_policy_get("IMATRIX"); if (imx == NULL) imx = getenv("LLMMODELIO_IMATRIX");
    if (imx == NULL) return -1;
    LlmCodecGgufFileInfo fi;
    if (llmcodec_gguf_read_info(model_path, &fi) != 0) return -1;
    FILE* f = fopen(out_path, "w");
    if (f == NULL) { llmcodec_gguf_free_info(&fi); return -1; }
    fprintf(f, "# LLM ROUTING PROFILE — uso reale per esperto (counts del routing, imatrix)\n");
    fprintf(f, "# imatrix: %s\n", imx);
    int nlay = 0; double grand = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        const char* nm = fi.tensors[i].name;
        const char* p = strstr(nm, "ffn_gate_exps");
        if (p == NULL || fi.tensors[i].ndim < 2) continue;
        int64_t E = (int64_t)fi.tensors[i].dims[fi.tensors[i].ndim-1];
        int64_t nu = 0; float* use = llmmodelio_imatrix_expert_usage(imx, nm, &nu);
        if (use == NULL || nu != E) { free(use); continue; }
        double tot = 0; int zeros = 0;
        for (int64_t e = 0; e < E; e++) { tot += use[e]; if (use[e] <= 0.0f) zeros++; }
        grand += tot; nlay++;
        fprintf(f, "%-32s esperto: %lld, attivazioni: %.0f, mai usati: %d\n", nm, (long long)E, tot, zeros);
        if (zeros > 0) { fprintf(f, "  mai usati:"); for (int64_t e = 0; e < E; e++) if (use[e] <= 0.0f) fprintf(f, " %lld", (long long)e); fprintf(f, "\n"); }
        /* top-8 per uso */
        fprintf(f, "  top-8:");
        for (int a = 0; a < 8 && a < E; a++) { int64_t best = -1; float bv = -1; for (int64_t e = 0; e < E; e++) { int u = 0; for (int q = 0; q < a; q++) { /* evita ripetizioni marcando con NaN sotto */ } (void)u; if (use[e] > bv) { bv = use[e]; best = e; } } if (best < 0) break; fprintf(f, " %lld(%.0f)", (long long)best, use[best]); use[best] = -1.0f; }
        fprintf(f, "\n");
        free(use);
    }
    fprintf(f, "# layer analizzati: %d, attivazioni totali: %.0f\n", nlay, grand);
    fclose(f); llmcodec_gguf_free_info(&fi);
    return nlay > 0 ? 0 : -1;
}


/* ---- AUTO (§E3): allocazione bit per-tensore guidata dalla SENSIBILITA' sotto un budget bpw.
 * Pre-pass: misura l'errore relativo di ricostruzione (proxy affine group=32, pesato dall'imatrix)
 * per ogni tensore eleggibile ai candidati {2,3,4,5,6,8} -> allocatore greedy -> formato per-tensore
 * (K-quant per la misura; Q8_0 per il non-eleggibile). Poi richiama il quantize col formato deciso. */
static int llmmodelio_bits_to_ggml(int b) { (void)b; return -1; }  /* ora cand[] sono gia' ggml type */

int llmmodelio_model_quantize_auto(const char* in_path, const char* out_path, double target_bpw) {
    if (in_path == NULL || out_path == NULL || target_bpw <= 0.0) return -1;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(in_path, &kv, &nkv, &fi) != 0) return -1;
    int n = (int)fi.n_tensors;
    /* 6 candidati (§ 2026-09-22, riabilitati ora che la misura usa la forma quadratica ΔWᵀΣΔW su
     * campioni reali invece del diagonale: il 5-cand {Q2K,Q4K,Q5K,Q6K,Q8_0} era stato escluso sotto
     * il VECCHIO misuratore diagonale — dava PPL peggiore e un Q3_K anomalo (rel>Q2_K). Col misuratore
     * a campioni, attn_v/ffn_down_exps hanno rel[Q4_K] alto (~0.05-0.08): con SOLO {Q2K,Q4K,Q8_0}
     * l'allocatore doveva scegliere fra "troppo poco" (Q4_K) e "troppo caro" (Q8_0, +4bpw), lasciando
     * la maggior parte a Q4_K sotto-precisa -> PPL 2x peggio del baseline. Q5_K/Q6_K colmano il salto
     * (la baseline Q4_K_M usa proprio Q6_K per attn_v/output/down_exps). */
    /* BUDGET IN bpw REALI x1000 (non i bit nominali): Q2_K=84/256=2.625, Q3_K=110/256=3.4375,
     * Q4_K=144/256=4.5, Q5_K=176/256=5.5, Q6_K=210/256=6.5625, Q8_0=34/32=8.5. */
    const int cand[6] = { 2625, 3438, 4500, 5500, 6563, 8500 };
    static const int cand_ty[6] = { 10, 11, 12, 13, 14, 8 };   /* -> Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 */
    const int NCAND = 6;
    int* ftab = (int*)malloc((size_t)n * sizeof(int));
    int* elig = (int*)malloc((size_t)n * sizeof(int)); int nel = 0;
    for (int i = 0; i < n; i++) ftab[i] = -1;
    const char* imx = llmmodelio_model_policy_get("IMATRIX"); if (imx == NULL) imx = getenv("LLMMODELIO_IMATRIX");
    FILE* f = fopen(in_path, "rb");
    if (f == NULL) { free(ftab); free(elig); llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return -1; }
    /* elenco eleggibili: pesi 2D/3D quantizzabili, blocco ne[0] % 256 (K-quant), non norm/embedding */
    for (int i = 0; i < n; i++) {
        LlmCodecGgufTensorInfo* ti = &fi.tensors[i];
        uint64_t last = ti->ndim > 0 ? ti->dims[0] : 0;
        if (!(ti->ggml_type == 0 || ti->ggml_type == 1) || ti->ndim < 2 || last == 0 || last % 256 != 0) continue;
        const char* role = llmmodelio_role_of(ti->name);
        if (strcmp(role, "EMBEDDING") == 0 || strcmp(role, "NORM") == 0) continue;
        /* il ROUTER (ffn_gate_inp) NON si quantizza: decide l'instradamento (la baseline Q4_K_M lo
         * tiene F32). Quantizzarlo degrada la selezione degli esperti -> era un pezzo del gap. */
        if (strstr(ti->name, "gate_inp") != NULL || strcmp(role, "ROUTER") == 0) continue;
        /* OUTPUT (lm_head): l'imatrix NON lo copre (8 tensori/layer, non l'output) -> la metrica
         * ricade sul diagonale NON pesato e lo sottostima -> Q4_K sul lm_head = logits corrotti
         * (verificato: PPL "negative std dev"). Senza dati di calibrazione NON si rialloca: resta
         * il default per-ruolo (Q8_0). Principio: se manca la misura, non decidere. */
        if (strcmp(role, "OUTPUT") == 0) continue;
        elig[nel++] = i;
    }
    uint64_t* npar = (uint64_t*)calloc((size_t)(nel ? nel : 1), 8);
    double* rel = (double*)calloc((size_t)(nel ? nel : 1) * (size_t)NCAND, 8);
    int rc = 0;
    for (int q = 0; q < nel && rc == 0; q++) {
        LlmCodecGgufTensorInfo* ti = &fi.tensors[elig[q]];
        uint64_t numel = llm_numel(ti->dims, ti->ndim), last = ti->dims[0];
        uint64_t b = 0;
        if (llmcodec_ggml_type_bytes(ti->ggml_type, numel, &b) != 0 || b == 0 || b > (1ull << 34)) { rc = -1; break; }
        uint8_t* raw = (uint8_t*)malloc((size_t)b);
        float* xf = (float*)malloc((size_t)numel * 4);
        if (raw == NULL || xf == NULL) { free(raw); free(xf); rc = -1; break; }
        if (fseek(f, (long)(fi.data_start + ti->offset), SEEK_SET) != 0 || fread(raw, 1, (size_t)b, f) != (size_t)b) { free(raw); free(xf); rc = -1; break; }
        if (ti->ggml_type == 0) memcpy(xf, raw, (size_t)numel * 4);
        else { const uint16_t* hb = (const uint16_t*)raw; for (uint64_t k = 0; k < numel; k++) xf[k] = llmmodelio_h2f(hb[k]); }
        free(raw);
        uint32_t cols = (uint32_t)last, rows = (uint32_t)(numel / last);
        int64_t nl = 0; float* imp = imx ? llmmodelio_imatrix_weights(imx, ti->name, &nl) : NULL;
        const float* impu = (imp && nl == (int64_t)cols) ? imp : NULL;
        /* Campioni Monte Carlo (§ diagnosi 2026-09-22): tensori per-esperto (ndim==3, ffn_*_exps) usano
         * la forma quadratica ΔWᵀΣΔW stimata su LLMMODELIO_SAMP_K_EXPERT campioni/esperto; i condivisi
         * (attn_q/k/v/output) su LLMMODELIO_SAMP_K_SHARED. Mismatch di forma o imatrix vecchio (senza
         * ".samples") -> ricade sul diagonale invariato (llmmodelio_roundtrip_rel via llmmodelio_roundtrip_rel_cov). */
        int64_t rows_per_expert = (ti->ndim == 3) ? (int64_t)ti->dims[1] : 0;
        int64_t expect_k = rows_per_expert > 0 ? (int64_t)LLMMODELIO_SAMP_K_EXPERT : (int64_t)LLMMODELIO_SAMP_K_SHARED;
        int64_t n_expert_s = rows_per_expert > 0 ? (int64_t)ti->dims[2] : 1;
        int64_t samp_kt = 0, samp_cols = 0;
        float* samp = imx ? llmmodelio_imatrix_samples(imx, ti->name, &samp_kt, &samp_cols) : NULL;
        int samp_ok = samp && samp_cols == (int64_t)cols && samp_kt == expect_k * n_expert_s;
        /* PAVIMENTO DIAGONALE (suggerimento diagnosi 2026-09-22): la metrica a campioni soffre di
         * cancellazione eccessiva (Q4_K..Q8_0 appaiono quasi uguali). Si prende il MAX fra la stima
         * a campioni e la diagonale scontata: la diagonale (che distingue 10-50x) fa da pavimento.
         * LLM_AUTO_FLOOR = sconto (default 0.5; 0 = solo campioni, 1 = diagonale piena). */
        static double s_floor = -1.0;
        if (s_floor < 0.0) { const char* e = getenv("LLM_AUTO_FLOOR"); s_floor = e ? atof(e) : 0.5; }
        for (int c = 0; c < NCAND; c++) {
            double rd = llmmodelio_roundtrip_rel(cand_ty[c], xf, numel, cols, impu);
            static int s_nosamp = -1; if (s_nosamp < 0) s_nosamp = getenv("LLM_AUTO_NOSAMP") != NULL;
            double r = (samp_ok && !s_nosamp) ? llmmodelio_roundtrip_rel_cov(cand_ty[c], xf, numel, cols, samp, expect_k, rows_per_expert) : rd;
            if (s_floor > 0.0 && rd > 0.0 && rd * s_floor > r) r = rd * s_floor;
            if (r < 0) r = 1e9; rel[(size_t)q*NCAND+c] = r;
        }
        for (int c = 1; c < NCAND; c++) if (rel[(size_t)q*NCAND+c] > rel[(size_t)q*NCAND+c-1]) rel[(size_t)q*NCAND+c] = rel[(size_t)q*NCAND+c-1];
        npar[q] = numel;
        if (getenv("LLMMODELIO_DEBUG_TENSORS"))
            fprintf(stderr, "[auto-t] %-40s samp_ok=%d rel=[%.5f %.5f %.5f %.5f %.5f %.5f]\n", ti->name, samp_ok,
                    rel[(size_t)q*NCAND+0], rel[(size_t)q*NCAND+1], rel[(size_t)q*NCAND+2], rel[(size_t)q*NCAND+3], rel[(size_t)q*NCAND+4], rel[(size_t)q*NCAND+5]);
        free(imp); free(samp); free(xf);
    }
    fclose(f);
    int* oc = (int*)calloc((size_t)(nel ? nel : 1), 4);
    if (rc == 0 && nel > 0) rc = llmengine_moe_quant_alloc_bits(npar, rel, nel, NCAND, cand, target_bpw * 1000.0, oc);
    /* PAVIMENTO PER RUOLO (diagnosi 2026-09-22): la metrica per-tensore SOTTOSTIMA l'attenzione ->
     * Q4_K/Q6_K dove la baseline Q4_K_M usa Q8_0/Q6_K = errore ~2x. Si impone un MINIMO per ruolo
     * preso dalla baseline: l'allocatore resta libero SOLO sopra il pavimento. Disattivabile con
     * LLM_AUTO_NOFLOOR=1 (per esperimenti). Indici candidati: 0=Q2_K 1=Q3_K 2=Q4_K 3=Q5_K 4=Q6_K 5=Q8_0 */
    static int s_nofloor = -1; if (s_nofloor < 0) s_nofloor = getenv("LLM_AUTO_NOFLOOR") != NULL;
    if (rc == 0 && !s_nofloor) for (int q = 0; q < nel; q++) {
        const char* nm = fi.tensors[elig[q]].name;
        const char* ro = llmmodelio_role_of(nm);
        int fl = 0;
        if (strcmp(ro, "ATTENTION") == 0)      fl = 4;   /* Q6_K */
        else if (strcmp(ro, "OUTPUT") == 0)    fl = 4;
        else if (strcmp(ro, "SHARED_EXPERT") == 0) fl = 4;
        else if (strcmp(ro, "ROUTED_EXPERT") == 0) fl = (strstr(nm, "_down") != NULL) ? 4 : 2; /* down Q6_K, gate/up Q4_K */
        if (oc[q] < fl) oc[q] = fl;
    }
    if (rc == 0) for (int q = 0; q < nel; q++) {
        ftab[elig[q]] = cand_ty[oc[q]];
        if (getenv("LLMMODELIO_DEBUG_TENSORS"))
            fprintf(stderr, "[auto-f] %-40s -> ty=%d (oc=%d)\n", fi.tensors[elig[q]].name, cand_ty[oc[q]], oc[q]);
    }
    if (getenv("LLMMODELIO_DEBUG")) {
        double bits = 0, tot = 0;   /* bits = somma nparams*cand (in bpw*1000) */ int cnt[6] = {0,0,0,0,0,0};
        for (int q = 0; q < nel; q++) { bits += (double)npar[q]*cand[oc[q]]; tot += (double)npar[q]; cnt[oc[q]]++; }
        fprintf(stderr, "[auto] eleggibili=%d bpw_medio=%.3f (target %.3f) Q2K=%d Q3K=%d Q4K=%d Q5K=%d Q6K=%d Q8_0=%d\n",
                nel, tot>0?bits/tot/1000.0:0, target_bpw, cnt[0],cnt[1],cnt[2],cnt[3],cnt[4],cnt[5]);
    }
    g_auto_type = ftab;
    int qrc = (rc == 0) ? llmmodelio_model_quantize(in_path, out_path) : -1;
    g_auto_type = NULL;
    free(ftab); free(elig); free(npar); free(rel); free(oc);
    llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
    return qrc;
}

/* ---- Errore di ricostruzione REALE per formato (quantize -> dequantize), per l'allocatore §E3.
 * Dequant portati da llama.cpp ggml-quants.c (Q2_K/Q4_K/Q8_0; gli altri formati sono i candidati).
 * Fattorizzato in llmmodelio_quant_dequant (condiviso fra llmmodelio_roundtrip_rel e la variante _cov sotto) per
 * non duplicare il porting bit-per-bit. ---- */
static int llmmodelio_quant_dequant(int ty, const float* x, uint64_t numel, float* dq) {
    uint64_t bb = 0; uint32_t bv = 0;
    switch (ty) { case 10: bb = 84; bv = 256; break;   /* Q2_K */
                  case 11: bb = 110; bv = 256; break;  /* Q3_K */
                  case 12: bb = 144; bv = 256; break;  /* Q4_K */
                  case 13: bb = 176; bv = 256; break;  /* Q5_K */
                  case 14: bb = 210; bv = 256; break;  /* Q6_K */
                  case 8:  bb = 34; bv = 32; break;    /* Q8_0 */
                  default: return -1; }
    if (bv == 0 || numel % bv != 0) return -1;
    uint64_t nb = numel / bv;
    uint8_t* q = (uint8_t*)malloc((size_t)nb * bb);
    if (!q) return -1;
    if (ty == 10) llm_q2_K(x, (int64_t)numel, q);
    else if (ty == 11) llm_q3_K(x, (int64_t)numel, q);
    else if (ty == 12) llm_q4_K(x, (int64_t)numel, q);
    else if (ty == 13) llm_q5_K(x, (int64_t)numel, q);
    else if (ty == 14) llm_q6_K(x, (int64_t)numel, q);
    else llmmodelio_q8_0(x, (int64_t)numel, q);
    if (ty == 8) {                                    /* Q8_0: {d:F16, qs[32]:i8} */
        for (uint64_t b = 0; b < nb; b++) {
            uint16_t dh; memcpy(&dh, q + b*34, 2); float d = llmmodelio_h2f(dh);
            const int8_t* qs = (const int8_t*)(q + b*34 + 2);
            for (int j = 0; j < 32; j++) dq[b*32 + j] = qs[j] * d;
        }
    } else if (ty == 10) {                            /* Q2_K: {scales[16], qs[64], d, dmin} @0,16,80,82 */
        for (uint64_t b = 0; b < nb; b++) {
            const uint8_t* blk = q + b*84;
            uint16_t h1, h2; memcpy(&h1, blk+80, 2); memcpy(&h2, blk+82, 2);
            float d = llmmodelio_h2f(h1), mn = llmmodelio_h2f(h2);
            const uint8_t* sc = blk; const uint8_t* qq = blk + 16; float* y = dq + b*256;
            int is = 0;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    for (int pass = 0; pass < 2; pass++) {
                        uint8_t s = sc[is++]; float dl = d*(s & 0xF), ml = mn*(s >> 4);
                        for (int l = 0; l < 16; l++) *y++ = dl * ((qq[l] >> shift) & 3) - ml;
                    }
                    shift += 2;
                }
                qq += 32;
            }
        }
    } else if (ty == 12) {                            /* Q4_K: {d, dmin, scales[12], qs[128]} @0,2,4,16 */
        for (uint64_t b = 0; b < nb; b++) {
            const uint8_t* blk = q + b*144;
            uint16_t h1, h2; memcpy(&h1, blk, 2); memcpy(&h2, blk+2, 2);
            float d = llmmodelio_h2f(h1), mn = llmmodelio_h2f(h2);
            const uint8_t* sc12 = blk + 4; const uint8_t* qq = blk + 16; float* y = dq + b*256;
            int is = 0;
            for (int n = 0; n < 256; n += 64) {
                uint8_t sc, m;
                llmmodelio_get_scale_min_k4(is + 0, sc12, &sc, &m); float d1 = d*sc, m1 = mn*m;
                llmmodelio_get_scale_min_k4(is + 1, sc12, &sc, &m); float d2 = d*sc, m2 = mn*m;
                for (int l = 0; l < 32; l++) y[n + l] = d1 * (qq[l] & 0xF) - m1;
                for (int l = 0; l < 32; l++) y[n + 32 + l] = d2 * (qq[l] >> 4) - m2;
                qq += 32; is += 2;
            }
        }
    }
    else if (ty == 11) {                              /* Q3_K: {hmask[32], qs[64], scales[12], d} @0,32,96,108 */
        for (uint64_t b = 0; b < nb; b++) {
            const uint8_t* blk = q + b*110;
            uint16_t h; memcpy(&h, blk+108, 2); float dall = llmmodelio_h2f(h);
            const uint8_t* hm = blk; const uint8_t* qq = blk+32;   /* hmask@0, qs@32 (v. commento riga sopra) — invertiti: bug */
            uint32_t aux[4]; memcpy(aux, blk+96, 12); uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
            aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
            const int8_t* scales3 = (const int8_t*)aux;
            float* y = dq + b*256; int is = 0; uint8_t m = 1;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    float dl = dall*(scales3[is++]-32);
                    for (int l = 0; l < 16; l++) y[n + j*32 + l] = dl * ((int8_t)((qq[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                    dl = dall*(scales3[is++]-32);
                    for (int l = 0; l < 16; l++) y[n + j*32 + 16 + l] = dl * ((int8_t)((qq[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                    shift += 2; m <<= 1;
                }
                qq += 32;
            }
        }
    } else if (ty == 13) {                            /* Q5_K: {d,dmin,scales[12],qh[32],qs[128]} @0,2,4,16,48 */
        for (uint64_t b = 0; b < nb; b++) {
            const uint8_t* blk = q + b*176;
            uint16_t h1,h2; memcpy(&h1,blk,2); memcpy(&h2,blk+2,2);
            float d = llmmodelio_h2f(h1), mn = llmmodelio_h2f(h2);
            const uint8_t* sc12 = blk+4; const uint8_t* qh = blk+16; const uint8_t* ql = blk+48;
            float* y = dq + b*256; int is = 0; uint8_t u1 = 1, u2 = 2;
            for (int n = 0; n < 256; n += 64) {
                uint8_t sc, m;
                llmmodelio_get_scale_min_k4(is+0, sc12, &sc, &m); float d1 = d*sc, m1 = mn*m;
                llmmodelio_get_scale_min_k4(is+1, sc12, &sc, &m); float d2 = d*sc, m2 = mn*m;
                for (int l = 0; l < 32; l++) y[n+l] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                for (int l = 0; l < 32; l++) y[n+32+l] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
        }
    } else if (ty == 14) {                            /* Q6_K: {ql[128], qh[64], scales[16]:i8, d} @0,128,192,208 */
        for (uint64_t b = 0; b < nb; b++) {
            const uint8_t* blk = q + b*210;
            const uint8_t* ql = blk; const uint8_t* qh = blk+128; const int8_t* sc = (const int8_t*)(blk+192);
            uint16_t h; memcpy(&h, blk+208, 2); float d = llmmodelio_h2f(h);
            float* y = dq + b*256;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; l++) {
                    int is2 = l/16;
                    int8_t q1 = (int8_t)((ql[l+0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l+0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d*sc[is2+0]*q1; y[l+32] = d*sc[is2+2]*q2; y[l+64] = d*sc[is2+4]*q3; y[l+96] = d*sc[is2+6]*q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
    }
    free(q);
    return 0;
}

double llmmodelio_roundtrip_rel(int ty, const float* x, uint64_t numel, uint32_t cols, const float* imp) {
    float* dq = (float*)malloc((size_t)numel * 4);
    if (!dq) return -1.0;
    if (llmmodelio_quant_dequant(ty, x, numel, dq) != 0) { free(dq); return -1.0; }
    double err = 0, sig = 0;
    for (uint64_t i = 0; i < numel; i++) { double w = imp ? (double)imp[i % cols] : 1.0; double d = (double)x[i] - dq[i]; err += w*d*d; sig += w*(double)x[i]*x[i]; }
    free(dq);
    return sig > 0 ? sqrt(err / sig) : 0.0;
}

/* ---- Variante "covarianza" di llmmodelio_roundtrip_rel (§ diagnosi 2026-09-22): invece di pesare l'errore
 * per-canale con la sola varianza (approssimazione diagonale di E[(ΔW·x)²]), stima la forma quadratica
 * COMPLETA ΔWᵀΣΔW via Monte Carlo su campioni REALI di attivazione catturati durante il forward pass
 * (llmmodelio_infer.c, "<tensor>.samples" nell'imatrix) — cattura le correlazioni fra canali che la
 * diagonale ignora, senza mai materializzare la matrice cols×cols (intrattabile per i tensori esperti:
 * vedi AiWiki). samp: [n_samp*cols] (tensori condivisi) o [n_expert*n_samp*cols] (tensori per-esperto,
 * rows_per_expert>0 -> riga o appartiene all'esperto o/rows_per_expert). Nessun campione -> ricade sul
 * diagonale (llmmodelio_roundtrip_rel), esattamente come prima di questa estensione. */
double llmmodelio_roundtrip_rel_cov(int ty, const float* x, uint64_t numel, uint32_t cols,
                             const float* samp, int64_t n_samp, int64_t rows_per_expert) {
    if (!samp || n_samp <= 0 || cols == 0 || numel % cols != 0) return llmmodelio_roundtrip_rel(ty, x, numel, cols, NULL);
    float* dq = (float*)malloc((size_t)numel * 4);
    if (!dq) return -1.0;
    if (llmmodelio_quant_dequant(ty, x, numel, dq) != 0) { free(dq); return -1.0; }
    uint64_t rows = numel / cols;
    double err = 0, sig = 0;
    for (uint64_t o = 0; o < rows; o++) {
        const float* xr = x + (size_t)o * cols;
        const float* dr = dq + (size_t)o * cols;
        const float* S = samp;
        if (rows_per_expert > 0) {
            uint64_t e = o / (uint64_t)rows_per_expert;
            S = samp + (size_t)e * (size_t)n_samp * cols;
        }
        for (int64_t s = 0; s < n_samp; s++) {
            const float* xs = S + (size_t)s * cols;
            double de = 0, ds = 0;
            for (uint32_t j = 0; j < cols; j++) {
                double diff = (double)xr[j] - (double)dr[j];
                de += diff * (double)xs[j];
                ds += (double)xr[j] * (double)xs[j];
            }
            err += de * de; sig += ds * ds;
        }
    }
    free(dq);
    return sig > 0 ? sqrt(err / sig) : 0.0;
}

/* ---- ARCH: superficie runtime del .llmarch (descrizione d'architettura data-driven) ----
 * Risolve la descrizione del modello (dal config) e l'espone: ruolo di un tensore, formato per ruolo,
 * show/list/check. E' quello che il converter e le funzioni BASIC (LLM.ARCH/LLM.ROLE, LLM ARCH ...)
 * usano — lo script BASIC non vede mai il file. */
#include "llmdesc.h"
#include "llmengine_plugin.h"
#include <dirent.h>
#include <stdlib.h>
static LlmDesc g_arch; static int g_arch_ok = 0; static char g_arch_path[4096] = "";

int llmmodelio_arch_resolve(const char* hf_dir) {
    if (llmdesc_resolve(hf_dir, &g_arch, g_arch_path, sizeof g_arch_path) != 0) { g_arch_ok = 0; return -1; }
    g_arch_ok = 1; return 0;
}
const char* llmmodelio_arch_name(void) { return g_arch_ok ? g_arch.arch : ""; }
const char* llmmodelio_arch_path(void) { return g_arch_path; }
/* Applica le regole: se g_arch non e' risolta, ritorna "OTHER". */
const char* llmmodelio_arch_role_of(const char* tensor) { return g_arch_ok ? llmdesc_role(&g_arch, tensor) : "OTHER"; }
const char* llmmodelio_arch_fmt_of(const char* role) { return g_arch_ok ? llmdesc_fmt(&g_arch, role) : NULL; }

void llmmodelio_arch_show(void) {
    if (!g_arch_ok) { fprintf(LOUT, "LLM ARCH: nessuna descrizione risolta (usa LLM ARCH RESOLVE <dir>)\n"); return; }
    fprintf(LOUT, "LLM ARCH '%s' (%s)\n", g_arch.arch, g_arch_path);
    fprintf(LOUT, "  ruoli (%d):\n", g_arch.n_roles);
    for (int i = 0; i < g_arch.n_roles; i++)
        fprintf(LOUT, "    %-22s -> %s%s\n", g_arch.roles[i].match, g_arch.roles[i].role,
               g_arch.roles[i].submodel[0] ? " (submodel)" : "");
    fprintf(LOUT, "  quant (%d):", g_arch.n_quant);
    for (int i = 0; i < g_arch.n_quant; i++) fprintf(LOUT, " %s=%s", g_arch.quant[i].role, g_arch.quant[i].fmt);
    fprintf(LOUT, "\n  recipe=%s boost_fmt=%s scale_suffix=%s\n", g_arch.recipe, g_arch.boost_fmt, g_arch.scale_suffix);
    fprintf(LOUT, "  modality=%s source.format=%s mmproj=%s\n", llmdesc_modality(&g_arch),
            g_arch.source_format[0] ? g_arch.source_format : "-", llmdesc_source_mmproj(&g_arch));
}
/* Valida un modello: ogni tensore dell'index ha un ruolo noto (non OTHER)? Ritorna il n. di OTHER.
 * DETERMINISTICO sull'argomento: `CHECK <hf_dir>` risolve SEMPRE la descrizione di QUELLA dir
 * (non riusa uno stato risolto da un'altra chiamata). Senza, in un processo che esegue più
 * comandi (IL/PCodeVM nel runner) il nome/ruoli stampati dipendevano dal `g_arch` precedente
 * (divergenza vs VM/C, processo fresco: `ARCH CHECK tiny_arch` stampava 'qwen3_moe'). */
int llmmodelio_arch_check(const char* hf_dir) {
    if (hf_dir && hf_dir[0]) {
        if (llmmodelio_arch_resolve(hf_dir) != 0) { fprintf(LOUT, "LLM ARCH CHECK: descrizione non risolta\n"); return -1; }
    } else if (!g_arch_ok) { fprintf(LOUT, "LLM ARCH CHECK: nessuna descrizione risolta (usa LLM ARCH RESOLVE <dir>)\n"); return -1; }
    char idx[2048]; snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", hf_dir);
    FILE* f = fopen(idx, "rb");
    int other = 0, tot = 0;
    if (f) {
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        char* j=(char*)malloc((size_t)n+1); if (j && fread(j,1,(size_t)n,f)==(size_t)n) { j[n]=0;
            char* p = j; const char* KEYS[2] = { "\"layers.", "\"model." };
            for (int ki = 0; ki < 2; ki++) { p = j;
            while ((p = strstr(p, KEYS[ki])) != NULL) {
                char* q = strchr(p+1,'"'); if (!q) break;
                char nm[256]; size_t L=(size_t)(q-p-1); if(L>255)L=255; memcpy(nm,p+1,L); nm[L]=0;
                tot++; if (strcmp(llmdesc_role(&g_arch, nm), "OTHER") == 0) other++;
                p = q+1;
            } }
            free(j);
        }
        fclose(f);
    }
    fprintf(LOUT, "LLM ARCH CHECK '%s': %d tensori, %d senza ruolo (OTHER)\n", g_arch.arch, tot, other);
    return other;
}

/* Elenca le descrizioni .llmarch disponibili (LLMDESC_ARCH_DIR o ./arch). */
static void arch_list_dir(const char* d) {
    fprintf(LOUT, "LLM ARCH: descrizioni in '%s':\n", d);
    DIR* dp = opendir(d);
    if (!dp) { fprintf(LOUT, "  (directory non trovata)\n"); return; }
    struct dirent* e; int cnt = 0;
    while ((e = readdir(dp)) != NULL) {
        const char* n = e->d_name; size_t L = strlen(n);
        if (L > 8 && strcmp(n + L - 8, ".llmarch") == 0) {
            char nm[256]; snprintf(nm, sizeof nm, "%.*s", (int)(L - 8), n);
            fprintf(LOUT, "  %s\n", nm); cnt++;
        }
    }
    closedir(dp);
    if (!cnt) fprintf(LOUT, "  (nessuna)\n");
}

void llmmodelio_arch_list(void) {
    const char* dir = getenv("LLMDESC_ARCH_DIR"); const char* d = (dir && *dir) ? dir : "arch";
    arch_list_dir(d);
    /* ABI/ canonica del repo (risalendo dal cwd): dove vivono TUTTE le descrizioni locali. */
    char abi[4096];
    if (llmdesc_abi_dir(abi, sizeof abi) && strcmp(abi, d) != 0) arch_list_dir(abi);
}

/* Esegue un comando LLM ARCH a runtime: sub = RESOLVE|SHOW|LIST|CHECK, path = dir modello (puo' essere "").
 * Unico punto d'ingresso per TUTTI i backend (C99Native, C99Backend, VM/C): cosi' nessuno
 * reimplementa il dispatch del sotto-verbo. Ritorna 0 ok, -1 errore. */
int llmmodelio_arch_cmd(const char* sub, const char* a, const char* b)
{
    const char* s = sub ? sub : "";
    const char* p = a ? a : "";
    const char* o = b ? b : "";
    if (strcasecmp(s, "RESOLVE") == 0) { if (!*p) return -1; return llmmodelio_arch_resolve(p); }
    if (strcasecmp(s, "SHOW") == 0) { llmmodelio_arch_show(); return 0; }
    if (strcasecmp(s, "LIST") == 0) { llmmodelio_arch_list(); return 0; }
    if (strcasecmp(s, "CHECK") == 0) { if (!*p) return -1; return llmmodelio_arch_check(p) < 0 ? -1 : 0; }
    if (strcasecmp(s, "PLAN") == 0) { if (!*p || !*o) return -1; return llmmodelio_arch_plan(p, o); }
    fprintf(stderr, "LLM ARCH: sotto-verbo '%s' non supportato (RESOLVE|SHOW|LIST|CHECK|PLAN)\n", s);
    return -1;
}

/* Esegue un comando LLM POLICY a runtime: sub = SET|SHOW|RESET|CLEAR, a=ruolo, b=formato.
 * Unico punto d'ingresso per tutti i backend (come llmmodelio_arch_cmd). */
int llmmodelio_policy_cmd(const char* sub, const char* a, const char* b)
{
    const char* s = sub ? sub : "";
    if (strcasecmp(s, "SET") == 0) {
        if (!a || !b || !*a || !*b) return -1;
        llmmodelio_model_policy_set(a, b); return 0;
    }
    if (strcasecmp(s, "RESET") == 0 || strcasecmp(s, "CLEAR") == 0) { llmmodelio_model_policy_reset(); return 0; }
    if (strcasecmp(s, "SHOW") == 0 || !*s) { llmmodelio_model_policy_show(); return 0; }
    fprintf(stderr, "LLM POLICY: sotto-verbo '%s' non supportato (SET|SHOW|RESET)\n", s);
    return -1;
}

/* ---- CONSUMER del .llmarch: piano guidato dalla DESCRIZIONE --------------------------------
 * Per ogni tensore dell'index: ruolo + formato proposto (dal `quant` della descrizione) +
 * accoppiamento con la sua `.scale`. NON legge i pesi (solo i nomi): è il consumer verificabile
 * senza I/O, ed è la base per la mappatura verso il container (data-driven, non per-arch). */
/* Sezione del container di destinazione per RUOLO: il modello testo, il vision tower, l'aligner.
 * (Il .llmarch può raffinare via `submodel`; qui la mappa role->sezione è la base del container.) */
static const char* llmmodelio_arch_dest(const char* role)
{
    if (role == NULL) return "text";
    if (strcmp(role, "VISION") == 0) return "vision";
    if (strcmp(role, "VISION_ALIGNER") == 0) return "aligner";
    if (strncmp(role, "AUDIO", 5) == 0) return "audio";
    if (strcmp(role, "MTP") == 0) return "mtp";
    if (strcmp(role, "ENGRAM") == 0) return "engram";
    return "text";
}

int llmmodelio_arch_plan(const char* hf_dir, const char* out_path)
{
    if (!hf_dir || !out_path) return -1;
    if (llmmodelio_arch_resolve(hf_dir) != 0) { fprintf(LOUT, "LLM ARCH PLAN: descrizione non risolta\n"); return -1; }
    char idx[2048]; snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", hf_dir);
    FILE* f = fopen(idx, "rb"); if (!f) { fprintf(LOUT, "LLM ARCH PLAN: index non trovato\n"); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)n + 1);
    if (!j || fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; }
    j[n] = 0; fclose(f);
    char* wm = strstr(j, "\"weight_map\"");
    if (!wm) { free(j); return -1; }
    FILE* o = fopen(out_path, "w");
    if (!o) { free(j); return -1; }
    fprintf(o, "# LLM ARCH PLAN — arch=%s (ruolo/formato/scale/destinazione per tensore)\n# name\trole\tfmt\tscale\tdst\n", g_arch.arch);
    struct { const char* role; int num; } cnt[96]; int nc = 0; int total = 0, scales = 0;
    const char* ss = g_arch.scale_suffix[0] ? g_arch.scale_suffix : ".scale";
    size_t ssl = strlen(ss);
    char* colon = strchr(wm, ':');           /* parti DOPO "weight_map": la chiave stessa non è un tensore */
    char* p = colon ? colon + 1 : wm;
    while ((p = strchr(p, '"')) != NULL) {
        char* q = strchr(p + 1, '"');
        if (!q) break;
        if (q[1] != ':') { p = q + 1; continue; }          /* è un valore (nome file), non una chiave */
        size_t L = (size_t)(q - p - 1);
        char nm[512]; if (L > 511) L = 511; memcpy(nm, p + 1, L); nm[L] = 0;
        int is_scale = 0;
        if (ssl && strlen(nm) > ssl && strcmp(nm + strlen(nm) - ssl, ss) == 0) { is_scale = 1; nm[strlen(nm) - ssl] = 0; }
        const char* role = llmdesc_role(&g_arch, nm);
        const char* fmt = llmdesc_fmt(&g_arch, role);
        fprintf(o, "%s\t%s\t%s\t%s\t%s\n", nm, role, fmt ? fmt : "-", is_scale ? "scale" : "", llmmodelio_arch_dest(role));
        total++; if (is_scale) scales++;
        int found = 0;
        for (int i = 0; i < nc; i++) if (strcmp(cnt[i].role, role) == 0) { cnt[i].num++; found = 1; break; }
        if (!found && nc < 96) { cnt[nc].role = role; cnt[nc].num = 1; nc++; }
        p = q + 1;
    }
    fclose(o); free(j);
    fprintf(LOUT, "LLM ARCH PLAN '%s': %d tensori (%d scale) -> %s\n", g_arch.arch, total, scales, out_path);
    for (int i = 0; i < nc; i++) fprintf(LOUT, "  %-18s %d\n", cnt[i].role, cnt[i].num);
    /* destinazioni nel container (sezioni) */
    { struct { const char* d; int n; } dc[4]; int nd = 0; (void)dc;
      fprintf(LOUT, "  destinazioni:");
      const char* ds[3] = { "text", "vision", "aligner" };
      for (int k = 0; k < 3; k++) { int n = 0; for (int i = 0; i < nc; i++) if (strcmp(llmmodelio_arch_dest(cnt[i].role), ds[k]) == 0) n += cnt[i].num; if (n) fprintf(LOUT, " %s=%d", ds[k], n); }
      fprintf(LOUT, "\n"); (void)nd; }
    return 0;
}


/* ---- varianti "_to": eseguono il comando e CATTURANO il testo in `out` (per IL/PCodeVM) ---- */
int llmmodelio_arch_cmd_to(const char* sub, const char* a, const char* b, char* out, size_t cap)
{
    if (!out || cap == 0) return -1;
    out[0] = 0;
    FILE* f = fmemopen(out, cap - 1, "w");
    if (!f) return llmmodelio_arch_cmd(sub, a, b);
    g_llm_out = f;
    int rc = llmmodelio_arch_cmd(sub, a, b);
    fflush(f); fclose(f); g_llm_out = NULL;
    out[cap - 1] = 0;
    return rc;
}
int llmmodelio_policy_cmd_to(const char* sub, const char* a, const char* b, char* out, size_t cap)
{
    if (!out || cap == 0) return -1;
    out[0] = 0;
    FILE* f = fmemopen(out, cap - 1, "w");
    if (!f) return llmmodelio_policy_cmd(sub, a, b);
    g_llm_out = f;
    int rc = llmmodelio_policy_cmd(sub, a, b);
    fflush(f); fclose(f); g_llm_out = NULL;
    out[cap - 1] = 0;
    return rc;
}

/* Verifica quali STRUTTURE dichiarate dalla .llmarch il CONVERTER sa mappare verso il container.
 * Scrive in `out` un report (supported / NOT supported) e ritorna il numero di NON supportate.
 * Onesto per costruzione (§0bis): le strutture non ancora gestite (compression/indexer/engram/
 * vision) sono dichiarate esplicitamente, mai ignorate in silenzio. */
int llmmodelio_arch_support(const char* hf_dir, char* out, size_t cap)
{
    if (llmmodelio_arch_resolve(hf_dir) != 0) { snprintf(out, cap, "LLM ARCH SUPPORT: descrizione non risolta\n"); return -1; }
    /* strutture che il converter gestisce oggi (mappatura tensori -> container);
     * le altre sono dichiarate NON supportate (gap visibile). */
    /* Stesse strutture del gate `llmdesc_unsupported` (llmdesc.c): il converter (ruoli .llmarch +
     * ricetta) mappa i TENSORI di tutte queste; il compute resta dell'engine. */
    static const struct { const char* key; const char* type; int ok; } known[] = {
        { "attention",   "mla",                 1 },
        { "attention",   "gqa",                 1 },
        { "moe",         NULL,                  1 },
        { "norm",        NULL,                  1 },
        { "compression", "csa",                 1 },
        { "compression", "csa_hca",             1 },
        { "indexer",     "indexer",             1 },
        { "indexer",     "topk_selector",       1 },
        { "engram",      "engram",              1 },
        { "engram",      "deterministic_router",1 },
        { "vision",      "vit",                 1 },
    };
    int nbad = 0; size_t o = 0;
    o += (size_t)snprintf(out+o, cap-o, "LLM ARCH SUPPORT '%s':\n", g_arch.arch);
    for (int i = 0; i < g_arch.n_struct && o < cap; i++) {
        const char* key = g_arch.structures[i].key;
        const char* type = g_arch.structures[i].type;
        int supported = 0;
        for (size_t k = 0; k < sizeof(known)/sizeof(known[0]); k++)
            if (strcmp(known[k].key, key) == 0 && (known[k].type == NULL || strcmp(known[k].type, type) == 0))
                { supported = known[k].ok; break; }
        if (!supported) nbad++;
        o += (size_t)snprintf(out+o, cap-o, "  %-12s %-22s %s\n", key, type, supported ? "supportata" : "NON supportata");
    }
    if (o < cap) snprintf(out+o, cap-o, "  -> %d struttura/e non supportata/e\n", nbad);
    return nbad;
}


/* Risolve il CODEC di un formato: PRIMA un PLUGIN (dichiarato dal .llmarch in `codecs`, oppure per
 * convenzione `libllmcodec_<fmt>`), POI il built-in (i formati nativi del runtime/GGUF).
 * Ritorna 0 se un PLUGIN fornisce `fmt` (e scrive il suo LlmCodec in `out` + il percorso in
 * `plugin_path`); -1 se il chiamante deve usare il BUILT-IN (nessun plugin). Cosi' un formato
 * nuovo = una dylib, senza ricompilare il core. */
int llmmodelio_codec_resolve(const char* hf_dir, const char* fmt, LlmCodec* out, char* plugin_path, size_t cap)
{
    if (fmt == NULL || *fmt == 0 || out == NULL) return -1;
    if (plugin_path && cap) plugin_path[0] = 0;
    char pname[64] = "";
    LlmDesc arch;
    if (hf_dir && llmdesc_resolve(hf_dir, &arch, NULL, 0) == 0) {
        const char* pl = llmdesc_codec_plugin(&arch, fmt);
        if (pl && *pl) snprintf(pname, sizeof pname, "%s", pl);
    }
    const char* cands[2] = { pname[0] ? pname : NULL, fmt };
    for (int i = 0; i < 2; i++) {
        if (cands[i] == NULL) continue;
        const LlmCodec* c = llmengine_plugin_codec_get(cands[i]);
        if (c != NULL) {
            *out = *c;
            if (plugin_path && cap) { const char* pp = llmengine_plugin_path("codec", cands[i]); snprintf(plugin_path, cap, "%s", pp ? pp : ""); }
            return 0;
        }
    }
    return -1;   /* built-in */
}


/* Risolve la PRIMITIVA dichiarata per una struttura (kind) nel .llmarch: il campo `prim` della
 * struttura -> il plugin (llmengine_plugin_prim_get). Ritorna 0 e il descrittore in `out` (+ `plugin_path`);
 * -1 se non dichiarata o plugin assente (il chiamante usa la primitiva built-in, se ne ha una). */
int llmmodelio_prim_resolve(const char* hf_dir, const char* kind, LlmEnginePluginPrim* out, char* plugin_path, size_t cap)
{
    if (hf_dir == NULL || kind == NULL || out == NULL) return -1;
    if (plugin_path && cap) plugin_path[0] = 0;
    LlmDesc arch;
    if (llmdesc_resolve(hf_dir, &arch, NULL, 0) != 0) return -1;
    const char* pname = llmdesc_struct_prim(&arch, kind);
    if (pname == NULL || *pname == 0) return -1;
    const LlmEnginePluginPrim* pr = llmengine_plugin_prim_get(pname);
    if (pr == NULL) return -1;
    *out = *pr;
    if (plugin_path && cap) { const char* pp = llmengine_plugin_path("prim", pname); snprintf(plugin_path, cap, "%s", pp ? pp : ""); }
    return 0;
}


/* Risolve la PIPELINE (composizione, livello 3) di una struttura: per ogni step carica la primitiva
 * dichiarata (plugin MLX -> plugin arithmetica) e scrive il report in `out`. Ritorna il n. di step
 * risolti (-1 se la struttura non ha steps). Così una struttura nuova = una LISTA di primitive. */
int llmmodelio_arch_pipeline(const char* hf_dir, const char* kind, char* out, size_t cap)
{
    if (hf_dir == NULL || kind == NULL || out == NULL) return -1;
    LlmDesc arch;
    if (llmdesc_resolve(hf_dir, &arch, NULL, 0) != 0) return -1;
    const LlmDescStruct* S = llmdesc_struct(&arch, kind);
    if (S == NULL || S->n_steps == 0) return -1;
    size_t o = 0; int found = 0;
    o += (size_t)snprintf(out + o, cap - o, "pipeline '%s' (%s):", kind, S->type[0] ? S->type : "-");
    for (int i = 0; i < S->n_steps && o < cap; i++) {
        const char* pr = S->steps[i].prim;
        const char* src = "-";
        if (pr && *pr) {
            if (llmengine_plugin_prim_mlx_get(pr) != NULL) src = "mlx";
            else if (llmengine_plugin_prim_get(pr) != NULL) src = "prim";
            else if (llmengine_plugin_codec_get(pr) != NULL) src = "codec";
            else src = "ASSENTE";
            if (strcmp(src, "ASSENTE") != 0) found++;
        }
        o += (size_t)snprintf(out + o, cap - o, " %s->%s(%s)", S->steps[i].op, pr && *pr ? pr : "-", src);
    }
    if (o < cap) snprintf(out + o, cap - o, " [%d/%d risolti]", found, S->n_steps);
    return found;
}

/* ---- MODEL COMPILER LLM: verbi TENSOR/KERNEL/MODULE/KV (superficie BASIC) -------------------
 * Stessa forma `llmmodelio_<x>_cmd(sub, a, b)` di ARCH/POLICY: UN solo punto d'ingresso per tutti i
 * backend (IL/PCodeVM/C99Backend/VM-C/C99Native), nessuna reimplementazione per backend (§0bis).
 * Il report va su LOUT; le varianti `_to` lo catturano in un buffer (i backend gestiti via P/Invoke
 * non hanno lo stdout nativo del processo C). Descrizione d'architettura data-driven (.llmarch):
 * lo script BASIC non vede mai il file. */

/* TENSOR: sub=ROLE a=<dir> b=<tensor> -> ruolo+formato del tensore; sub=LIST a=<dir> -> tutti i
 * ruoli (match->ruolo + formato proposto). */
/* ── MODEL IR: tabella per-tensore del MODELLO (ruolo/geometria/tipo/byte) ──
 * GGUF: legge i tensor-info + ruolo (dall'arch del gguf). Dir MLX/HF: legge weight_map
 * (index.json) + ruolo/formato dall'arch risolta. Formato-agnostico (§4.2 Model IR). */
static int llmmodelio_arch_resolve_gguf(const char* gguf_path);

static int llmmodelio_model_ir_print(const char* path) {
    size_t ln = strlen(path);
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) {
        if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
        LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
        if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
        fprintf(LOUT, "# LLM TENSOR IR '%s' GGUF arch=%s tensori=%llu\n# name\trole\ttype\tndim\tdims\tbytes\n",
                path, g_arch_ok ? g_arch.arch : "-", (unsigned long long)fi.n_tensors);
        for (uint64_t i = 0; i < fi.n_tensors; i++) {
            LlmCodecGgufTensorInfo* t = &fi.tensors[i];
            const char* role = g_arch_ok ? llmdesc_role(&g_arch, t->name) : "OTHER";
            char tn[24]; if (llmcodec_ggml_type_name(t->ggml_type, tn, sizeof tn) != 0) snprintf(tn, sizeof tn, "t%d", t->ggml_type);
            char ds[96]; size_t o = 0; ds[0] = 0;
            for (int d = 0; d < t->ndim; d++) o += (size_t)snprintf(ds + o, sizeof ds - o, "%s%llu", d ? "x" : "", (unsigned long long)t->dims[d]);
            uint64_t numel = 1; for (int d = 0; d < t->ndim; d++) numel *= t->dims[d];
            uint64_t b = 0; llmcodec_ggml_type_bytes(t->ggml_type, numel, &b);
            fprintf(LOUT, "%s\t%s\t%s\t%d\t%s\t%llu\n", t->name, role, tn, t->ndim, ds, (unsigned long long)b);
        }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
        return 0;
    }
    if (llmmodelio_arch_resolve(path) != 0) return -1;
    char idx[2048]; snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", path);
    FILE* f = fopen(idx, "rb"); if (f == NULL) { fprintf(LOUT, "# LLM TENSOR IR '%s' MLX arch=%s (index assente)\n", path, g_arch.arch); return 0; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)n + 1); if (j == NULL || fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; }
    j[n] = 0; fclose(f);
    char* wm = strstr(j, "\"weight_map\""); if (wm == NULL) { free(j); return -1; }
    fprintf(LOUT, "# LLM TENSOR IR '%s' MLX arch=%s\n# name\trole\tfmt\n", path, g_arch.arch);
    char* colon = strchr(wm, ':'); char* p = colon ? colon + 1 : wm; int nt = 0;
    while ((p = strchr(p, '"')) != NULL) {
        char* q = strchr(p + 1, '"'); if (!q) break;
        if (q[1] != ':') { p = q + 1; continue; }
        size_t L = (size_t)(q - p - 1); char nm[512]; if (L > 511) L = 511; memcpy(nm, p + 1, L); nm[L] = 0;
        const char* role = llmdesc_role(&g_arch, nm); const char* fmt = llmdesc_fmt(&g_arch, role);
        fprintf(LOUT, "%s\t%s\t%s\n", nm, role, fmt ? fmt : "-"); nt++; p = q + 1;
    }
    free(j);
    return 0;
}
static int model_layer_of(const char* name) {
    const char* p = strstr(name, "blk.");    if (p) return atoi(p + 4);
    p = strstr(name, "layers.");             if (p) return atoi(p + 7);
    p = strstr(name, "mtp.");                if (p) return atoi(p + 4);
    return -1;
}
/* LAYERS: indici di layer presenti + n. tensori per layer (GGUF o dir MLX). */
static int llmmodelio_model_ir_layers(const char* path) {
    int cnt[1024]; memset(cnt, 0, sizeof cnt); int maxl = -1, glob = 0;
    size_t ln = strlen(path);
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) {
        if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
        LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
        if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
        for (uint64_t i = 0; i < fi.n_tensors; i++) { int L = model_layer_of(fi.tensors[i].name); if (L < 0) { glob++; continue; } if (L < 1024) cnt[L]++; if (L > maxl) maxl = L; }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    } else {
        if (llmmodelio_arch_resolve(path) != 0) return -1;
        char idx[2048]; snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", path);
        FILE* f = fopen(idx, "rb"); if (f == NULL) return -1;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char* j = (char*)malloc((size_t)n + 1); if (!j || fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; } j[n] = 0; fclose(f);
        char* wm = strstr(j, "\"weight_map\""); char* p = wm ? strchr(wm, ':') : NULL;
        while (p && (p = strchr(p, '"')) != NULL) { char* q = strchr(p + 1, '"'); if (!q) break; if (q[1] != ':') { p = q + 1; continue; }
            size_t L = (size_t)(q - p - 1); char nm[512]; if (L > 511) L = 511; memcpy(nm, p + 1, L); nm[L] = 0;
            int lyr = model_layer_of(nm); if (lyr < 0) glob++; else { if (lyr < 1024) cnt[lyr]++; if (lyr > maxl) maxl = lyr; } p = q + 1; }
        free(j);
    }
    fprintf(LOUT, "# LLM TENSOR LAYERS '%s' arch=%s layers=0..%d globals=%d\n", path, g_arch_ok ? g_arch.arch : "-", maxl, glob);
    for (int i = 0; i <= maxl && i < 1024; i++) if (cnt[i] > 0) fprintf(LOUT, "layer %d: %d tensori\n", i, cnt[i]);
    return 0;
}
/* CHECK (spec-vs-modello): ogni tensore ha un ruolo noto? Ritorna -1 se l'arch non e' risolta o
 * se restano tensori OTHER (errore esplicito). */
static int llmmodelio_model_ir_check(const char* path) {
    size_t ln = strlen(path); int tot = 0, other = 0;
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) {
        if (llmmodelio_arch_resolve_gguf(path) != 0) return -1;
        LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
        if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
        for (uint64_t i = 0; i < fi.n_tensors; i++) { tot++; const char* r = llmdesc_role(&g_arch, fi.tensors[i].name); if (strcmp(r, "OTHER") == 0) { other++; if (other <= 8) fprintf(LOUT, "  OTHER: %s\n", fi.tensors[i].name); } }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    } else {
        if (llmmodelio_arch_resolve(path) != 0) { fprintf(LOUT, "LLM TENSOR CHECK: arch non risolta per '%s'\n", path); return -1; }
        char idx[2048]; snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", path);
        FILE* f = fopen(idx, "rb"); if (f == NULL) return -1;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char* j = (char*)malloc((size_t)n + 1); if (!j || fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; } j[n] = 0; fclose(f);
        char* wm = strstr(j, "\"weight_map\""); char* p = wm ? strchr(wm, ':') : NULL;
        while (p && (p = strchr(p, '"')) != NULL) { char* q = strchr(p + 1, '"'); if (!q) break; if (q[1] != ':') { p = q + 1; continue; }
            size_t L = (size_t)(q - p - 1); char nm[512]; if (L > 511) L = 511; memcpy(nm, p + 1, L); nm[L] = 0;
            tot++; const char* r = llmdesc_role(&g_arch, nm); if (strcmp(r, "OTHER") == 0) { other++; if (other <= 8) fprintf(LOUT, "  OTHER: %s\n", nm); } p = q + 1; }
        free(j);
    }
    fprintf(LOUT, "# LLM TENSOR CHECK '%s' arch=%s tensori=%d OTHER=%d -> %s\n", path, g_arch_ok ? g_arch.arch : "-", tot, other, other ? "INVALID" : "OK");
    return other ? -1 : 0;
}
/* INFO: riepilogo del modello (arch, tensori, layer, kv, byte, modality) — GGUF o dir MLX. */
static int llmmodelio_model_ir_info(const char* path) {
    size_t ln = strlen(path);
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) {
        if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
        LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
        if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
        int maxl = -1; unsigned long long totb = 0;
        for (uint64_t i = 0; i < fi.n_tensors; i++) {
            int L = model_layer_of(fi.tensors[i].name); if (L > maxl) maxl = L;
            uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d];
            uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) totb += b;
        }
        fprintf(LOUT, "LLM TENSOR INFO '%s': format=GGUF arch=%s tensori=%llu kv=%d layers=%d modality=%s tensor_bytes=%llu\n",
                path, g_arch_ok ? g_arch.arch : "-", (unsigned long long)fi.n_tensors, nkv, maxl + 1,
                g_arch_ok ? llmdesc_modality(&g_arch) : "-", totb);
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
        return 0;
    }
    if (llmmodelio_arch_resolve(path) != 0) return -1;
    char idx[2048]; snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", path);
    FILE* f = fopen(idx, "rb"); if (f == NULL) { fprintf(LOUT, "LLM TENSOR INFO '%s': format=MLX arch=%s (index assente) modality=%s\n", path, g_arch.arch, llmdesc_modality(&g_arch)); return 0; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)n + 1); if (!j || fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; } j[n] = 0; fclose(f);
    char* wm = strstr(j, "\"weight_map\""); char* p = wm ? strchr(wm, ':') : NULL; int nt = 0, maxl = -1;
    while (p && (p = strchr(p, '"')) != NULL) { char* q = strchr(p + 1, '"'); if (!q) break; if (q[1] != ':') { p = q + 1; continue; }
        size_t L = (size_t)(q - p - 1); char nm[512]; if (L > 511) L = 511; memcpy(nm, p + 1, L); nm[L] = 0;
        nt++; int lyr = model_layer_of(nm); if (lyr > maxl) maxl = lyr; p = q + 1; }
    free(j);
    fprintf(LOUT, "LLM TENSOR INFO '%s': format=MLX arch=%s tensori=%d layers=%d modality=%s\n", path, g_arch.arch, nt, maxl + 1, llmdesc_modality(&g_arch));
    return 0;
}
/* GRAPH (I1, report-only): struttura del modello per layer (attention/moe/ffn/norm) dall arch. */
static int llmmodelio_model_ir_graph(const char* path) {
    int F[1024]; memset(F, 0, sizeof F); int maxl = -1;
    /* bit: 1=attn 2=moe 4=ffn_dense 8=norm 16=embed/out */
    size_t ln = strlen(path);
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) {
        if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
        LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
        if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
        for (uint64_t i = 0; i < fi.n_tensors; i++) {
            const char* r = g_arch_ok ? llmdesc_role(&g_arch, fi.tensors[i].name) : "OTHER";
            int L = model_layer_of(fi.tensors[i].name); if (L < 0) L = 0; if (L > maxl) maxl = L; if (L > 1023) continue;
            if (strncmp(r, "ATTN", 4) == 0) F[L] |= 1;
            else if (strcmp(r,"ROUTED_EXPERT")==0||strcmp(r,"SHARED_EXPERT")==0||strcmp(r,"ROUTER")==0||strstr(fi.tensors[i].name,"exps")) F[L] |= 2;
            else if (strcmp(r,"FFN")==0||strcmp(r,"FFN_GATE")==0||strcmp(r,"FFN_UP")==0||strcmp(r,"FFN_DOWN")==0) F[L] |= 4;
            else if (strstr(r,"NORM")) F[L] |= 8;
            else if (strcmp(r,"EMBEDDING")==0||strcmp(r,"OUTPUT")==0) F[L] |= 16;
        }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    } else {
        if (llmmodelio_arch_resolve(path) != 0) return -1;
        char idx[2048]; snprintf(idx, sizeof idx, "%s/model.safetensors.index.json", path);
        FILE* f = fopen(idx, "rb"); if (f == NULL) return -1;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char* j = (char*)malloc((size_t)n + 1); if (!j || fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; } j[n] = 0; fclose(f);
        char* wm = strstr(j, "\"weight_map\""); char* p = wm ? strchr(wm, ':') : NULL;
        while (p && (p = strchr(p, '"')) != NULL) { char* q = strchr(p + 1, '"'); if (!q) break; if (q[1] != ':') { p = q + 1; continue; }
            size_t L2 = (size_t)(q - p - 1); char nm[512]; if (L2 > 511) L2 = 511; memcpy(nm, p + 1, L2); nm[L2] = 0;
            const char* r = llmdesc_role(&g_arch, nm); int L = model_layer_of(nm); if (L < 0) L = 0; if (L > maxl) maxl = L; if (L > 1023) { p = q + 1; continue; }
            if (strncmp(r, "ATTN", 4) == 0) F[L] |= 1;
            else if (strcmp(r,"ROUTED_EXPERT")==0||strcmp(r,"SHARED_EXPERT")==0||strcmp(r,"ROUTER")==0||strstr(nm,"exps")) F[L] |= 2;
            else if (strcmp(r,"FFN")==0||strcmp(r,"FFN_GATE")==0||strcmp(r,"FFN_UP")==0||strcmp(r,"FFN_DOWN")==0) F[L] |= 4;
            else if (strstr(r,"NORM")) F[L] |= 8; else if (strcmp(r,"EMBEDDING")==0||strcmp(r,"OUTPUT")==0) F[L] |= 16;
            p = q + 1; }
        free(j);
    }
    const char* atn = g_arch_ok ? llmdesc_struct_type(&g_arch,"attention") : NULL;
    const char* moe = g_arch_ok ? llmdesc_struct_type(&g_arch,"moe") : NULL;
    fprintf(LOUT, "# LLM TENSOR GRAPH '%s' arch=%s attention=%s moe=%s layers=0..%d\n", path, g_arch_ok?g_arch.arch:"-", atn?atn:"-", moe?moe:"-", maxl);
    for (int L = 0; L <= maxl && L < 1024; L++) {
        if (!F[L]) continue;
        fprintf(LOUT, "layer %d:", L);
        if (F[L]&1) fprintf(LOUT, " attention%s", atn?atn:"");
        if (F[L]&2) fprintf(LOUT, " moe%s", moe?moe:"");
        if (F[L]&4) fprintf(LOUT, " ffn");
        if (F[L]&8) fprintf(LOUT, " norm");
        if (F[L]&16) fprintf(LOUT, " embed/out");
        fprintf(LOUT, "\n");
    }
    return 0;
}
/* MEMORY IR (I2, report-only): byte per categoria (embed/attn/experts/router/ffn/norm/output/other). */
static int llmmodelio_model_ir_memory(const char* path) {
    unsigned long long cat[8] = {0}; /* 0 embed 1 attn 2 experts 3 router 4 ffn 5 norm 6 output 7 other */
    const char* cn[8] = {"embedding","attention","experts","router","ffn_dense","norm","output","other"};
    size_t ln = strlen(path); int n = 0;
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) {
        if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
        LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
        if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
        for (uint64_t i = 0; i < fi.n_tensors; i++) {
            const char* r = g_arch_ok ? llmdesc_role(&g_arch, fi.tensors[i].name) : "OTHER";
            uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d];
            uint64_t b = 0; llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b);
            int c = 7;
            if (strcmp(r,"EMBEDDING")==0) c=0; else if (strncmp(r,"ATTN",4)==0) c=1;
            else if (strcmp(r,"ROUTED_EXPERT")==0||strcmp(r,"SHARED_EXPERT")==0) c=2; else if (strcmp(r,"ROUTER")==0) c=3;
            else if (strncmp(r,"FFN",3)==0) c=4; else if (strstr(r,"NORM")) c=5; else if (strcmp(r,"OUTPUT")==0) c=6;
            cat[c] += b; n++;
        }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    } else { fprintf(LOUT, "LLM TENSOR MEMORY: solo GGUF (per ora) — '%s'\n", path); return -1; }
    unsigned long long tot = 0; for (int c = 0; c < 8; c++) tot += cat[c];
    fprintf(LOUT, "# LLM TENSOR MEMORY '%s' tensori=%d total_bytes=%llu (%.2f GiB)\n", path, n, tot, (double)tot/(1024.0*1024.0*1024.0));
    for (int c = 0; c < 8; c++) if (cat[c]) fprintf(LOUT, "  %-10s %llu bytes (%.1f%%)\n", cn[c], cat[c], tot ? 100.0*(double)cat[c]/(double)tot : 0.0);
    return 0;
}
/* COST MODEL (I3, report-only): pesi + KV byte/token (da metadati GGUF) + totale per contesto. */
static int llmmodelio_model_ir_cost(const char* path, int ctx) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(LOUT, "LLM TENSOR COST: solo GGUF (per ora)\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    unsigned long long wb = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) { uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d]; uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) wb += b; }
    long bc = -1, hkv = -1, hd = -1, kl = -1, vl = -1, hl = -1, en = -1;
    for (int i = 0; i < nkv; i++) {
        const char* k = kv[i].key; long v = (long)kv[i].u32;
        if (strstr(k, "attention.head_count_kv")) hkv = v; else if (strstr(k, "attention.head_count")) hd = v;
        else if (strstr(k, "attention.key_length")) kl = v; else if (strstr(k, "attention.value_length")) vl = v;
        else if (strstr(k, "block_count")) bc = v; else if (strstr(k, "embedding_length")) en = v;
    }
    if (ctx <= 0) ctx = 2048;
    double kvpt = -1.0;
    if (bc > 0 && hkv > 0) { long kd = (kl > 0) ? kl : ((en > 0 && hd > 0) ? en / hd : -1); long vd = (vl > 0) ? vl : kd;
        if (kd > 0 && vd > 0) kvpt = (double)bc * (double)hkv * (double)(kd + vd) * 2.0; }  /* 2 = F16 KV */
    fprintf(LOUT, "# LLM TENSOR COST '%s' arch=%s ctx=%d\n  weights: %llu bytes (%.2f GiB)\n",
            path, g_arch_ok?g_arch.arch:"-", ctx, wb, (double)wb/(1024.0*1024.0*1024.0));
    if (kvpt >= 0) fprintf(LOUT, "  kv/token: %.0f bytes (%.2f MiB) -> kv@ctx: %.2f GiB\n  total@ctx: %.2f GiB\n",
                           kvpt, kvpt/(1024.0*1024.0), kvpt*(double)ctx/(1024.0*1024.0*1024.0), ((double)wb + kvpt*(double)ctx)/(1024.0*1024.0*1024.0));
    else fprintf(LOUT, "  kv/token: n/d (metadati KV assenti: block_count/head_count_kv/key_length)\n");
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    return 0;
}
/* SPECULATION IR (I9, report-only): modalita' dichiarata dall ABI (mtp/dspark) + ruoli MTP. */
static int llmmodelio_model_ir_spec(const char* path) {
    size_t ln = strlen(path);
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) { if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0; }
    else { if (llmmodelio_arch_resolve(path) != 0) return -1; }
    if (!g_arch_ok) { fprintf(LOUT, "LLM TENSOR SPEC: arch non risolta per '%s'\n", path); return -1; }
    int has_mtp = g_arch.mm_mtp || 0;
    for (int i = 0; i < g_arch.n_roles && !has_mtp; i++) if (strcmp(g_arch.roles[i].role, "MTP") == 0) has_mtp = 1;
    const char* mode = has_mtp ? (g_arch.mm_dspark ? "DSPARK(MTP)" : "MTP") : "NONE";
    fprintf(LOUT, "# LLM TENSOR SPEC '%s' arch=%s speculation=%s mtp=%d dspark=%d n_mtp_roles=%d\n",
            path, g_arch.arch, mode, g_arch.mm_mtp, g_arch.mm_dspark, (int)0);
    return 0;
}
/* KV IR (report-only, §4.6): formato/dtype KV dichiarati dall ABI (runtime.kv.dtype/state). */
static int llmmodelio_model_ir_kv(const char* path) {
    size_t ln = strlen(path);
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) { if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0; }
    else { if (llmmodelio_arch_resolve(path) != 0) return -1; }
    if (!g_arch_ok) { fprintf(LOUT, "LLM TENSOR KV: arch non risolta per '%s'\n", path); return -1; }
    fprintf(LOUT, "# LLM TENSOR KV '%s' arch=%s kv.dtype=%s state=%s activation.dtype=%s\n", path, g_arch.arch,
            g_arch.runtime_kv_dtype[0] ? g_arch.runtime_kv_dtype : "-", g_arch.runtime_state[0] ? g_arch.runtime_state : "-",
            g_arch.runtime_activation_dtype[0] ? g_arch.runtime_activation_dtype : "-");
    return 0;
}
/* EXECPLAN (GMB_AI_PLAN serializzabile, §4.5): piano esecutivo del modello in JSON (report-only). */
static int llmmodelio_model_ir_execplan(const char* path, const char* out_path) {
    if (!out_path || !*out_path) { fprintf(stderr, "LLM TENSOR EXECPLAN: atteso <path>, <out>\\n"); return -1; }
    size_t ln = strlen(path); int F[1024]; memset(F, 0, sizeof F); int maxl = -1;
    unsigned long long wb = 0; int nten = 0; long bc = -1, hkv = -1, hd = -1, kl = -1, vl = -1, en = -1;
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) {
        if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
        LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
        if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
        for (uint64_t i = 0; i < fi.n_tensors; i++) {
            const char* r = g_arch_ok ? llmdesc_role(&g_arch, fi.tensors[i].name) : "OTHER";
            uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d];
            uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) wb += b;
            int L = model_layer_of(fi.tensors[i].name); if (L < 0) L = 0; if (L > maxl) maxl = L; if (L > 1023) { nten++; continue; }
            if (strncmp(r,"ATTN",4)==0) F[L]|=1; else if (strcmp(r,"ROUTED_EXPERT")==0||strcmp(r,"SHARED_EXPERT")==0||strcmp(r,"ROUTER")==0||strstr(fi.tensors[i].name,"exps")) F[L]|=2;
            else if (strncmp(r,"FFN",3)==0) F[L]|=4; else if (strstr(r,"NORM")) F[L]|=8; nten++;
        }
        for (int i = 0; i < nkv; i++) { const char* k = kv[i].key; long v = (long)kv[i].u32;
            if (strstr(k,"attention.head_count_kv")) hkv=v; else if (strstr(k,"attention.head_count")) hd=v;
            else if (strstr(k,"attention.key_length")) kl=v; else if (strstr(k,"attention.value_length")) vl=v;
            else if (strstr(k,"block_count")) bc=v; else if (strstr(k,"embedding_length")) en=v; }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    } else { fprintf(stderr, "LLM TENSOR EXECPLAN: solo GGUF (per ora)\\n"); return -1; }
    double kvpt = -1.0;
    if (bc > 0 && hkv > 0) { long kd = kl>0?kl:((en>0&&hd>0)?en/hd:-1); long vd = vl>0?vl:kd; if (kd>0&&vd>0) kvpt = (double)bc*(double)hkv*(double)(kd+vd)*2.0; }
    const char* atn = g_arch_ok ? llmdesc_struct_type(&g_arch,"attention") : NULL;
    const char* moe = g_arch_ok ? llmdesc_struct_type(&g_arch,"moe") : NULL;
    FILE* o = fopen(out_path, "w"); if (o == NULL) return -1;
    fprintf(o, "{\n  \"format\": \"GGUF\",\n  \"arch\": \"%s\",\n  \"tensors\": %d,\n  \"layers\": %d,\n", g_arch_ok?g_arch.arch:"-", nten, maxl+1);
    fprintf(o, "  \"attention\": \"%s\",\n  \"moe\": \"%s\",\n  \"modality\": \"%s\",\n", atn?atn:"-", moe?moe:"-", g_arch_ok?llmdesc_modality(&g_arch):"-");
    int mtp = g_arch_ok ? (g_arch.mm_mtp || 0) : 0;
    fprintf(o, "  \"speculation\": \"%s\",\n", mtp ? (g_arch.mm_dspark?"DSPARK(MTP)":"MTP") : "NONE");
    fprintf(o, "  \"memory\": { \"weights_bytes\": %llu, \"kv_per_token_bytes\": ", wb);
    if (kvpt >= 0) fprintf(o, "%.0f", kvpt); else fprintf(o, "null");
    fprintf(o, " },\n");
    fprintf(o, "  \"layer_ops\": [\n");
    int first = 1;
    for (int L = 0; L <= maxl && L < 1024; L++) { if (!F[L]) continue;
        fprintf(o, "%s    { \"layer\": %d, \"ops\": [", first?"":",\n", L);
        int f1 = 1; if (F[L]&1){fprintf(o,"%s\"attention\"",f1?"":" ,");f1=0;} if (F[L]&4){fprintf(o,"%s\"ffn\"",f1?"":" ,");f1=0;} if (F[L]&2){fprintf(o,"%s\"moe\"",f1?"":" ,");f1=0;} if (F[L]&8){fprintf(o,"%s\"norm\"",f1?"":" ,");f1=0;}
        fprintf(o, "] }"); first = 0; }
    fprintf(o, "\n  ]\n}\n");
    fclose(o);
    fprintf(LOUT, "LLM TENSOR EXECPLAN '%s' -> '%s' (arch=%s layers=%d weights=%llu, kv/token=%.0f)\n", path, out_path, g_arch_ok?g_arch.arch:"-", maxl+1, wb, kvpt);
    return 0;
}
/* COMPILE (§4.9, ZGC): artefatto SPECIALIZZATO `.gmbllm` (JSON) = graph + memory plan +
 * exact capacity + kernel/precision per categoria. Report-only, serializzabile. */
static int llmmodelio_model_ir_compile(const char* path, const char* out_path) {
    if (!out_path || !*out_path) { fprintf(stderr, "LLM TENSOR COMPILE: atteso <path>, <out.gmbllm>\\n"); return -1; }
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR COMPILE: solo GGUF (per ora)\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    unsigned long long wb = 0; int F[1024]; memset(F, 0, sizeof F); int maxl = -1;
    /* kernel/precision per categoria (dall arch quant, fallback al tipo del tensore) */
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d];
        uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) wb += b;
        const char* r = g_arch_ok ? llmdesc_role(&g_arch, fi.tensors[i].name) : "OTHER";
        int L = model_layer_of(fi.tensors[i].name); if (L < 0) L = 0; if (L > maxl) maxl = L; if (L > 1023) continue;
        if (strncmp(r,"ATTN",4)==0) F[L]|=1; else if (strcmp(r,"ROUTED_EXPERT")==0||strcmp(r,"SHARED_EXPERT")==0||strcmp(r,"ROUTER")==0||strstr(fi.tensors[i].name,"exps")) F[L]|=2;
        else if (strncmp(r,"FFN",3)==0) F[L]|=4; else if (strstr(r,"NORM")) F[L]|=8;
    }
    long bc=-1,hkv=-1,hd=-1,kl=-1,vl=-1,en=-1;
    for (int i=0;i<nkv;i++){ const char* k=kv[i].key; long v=(long)kv[i].u32;
        if(strstr(k,"attention.head_count_kv"))hkv=v; else if(strstr(k,"attention.head_count"))hd=v;
        else if(strstr(k,"attention.key_length"))kl=v; else if(strstr(k,"attention.value_length"))vl=v;
        else if(strstr(k,"block_count"))bc=v; else if(strstr(k,"embedding_length"))en=v; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    double kvpt=-1.0; if(bc>0&&hkv>0){ long kd=kl>0?kl:((en>0&&hd>0)?en/hd:-1); long vd=vl>0?vl:kd; if(kd>0&&vd>0) kvpt=(double)bc*(double)hkv*(double)(kd+vd)*2.0; }
    const double BUDGET = 96.0*1024.0*1024.0*1024.0, RESERVE = 2048.0*1024.0*1024.0; /* 96GB cap, 2GB riserva */
    long cap = (kvpt>0) ? (long)((BUDGET - RESERVE - (double)wb) / kvpt) : -1;
    const char* atn = g_arch_ok ? llmdesc_struct_type(&g_arch,"attention") : NULL;
    const char* moe = g_arch_ok ? llmdesc_struct_type(&g_arch,"moe") : NULL;
    FILE* o = fopen(out_path, "w"); if (!o) return -1;
    fprintf(o, "{\n  \"artifact\": \"gmbllm\",\n  \"version\": 1,\n  \"format\": \"GGUF\",\n  \"arch\": \"%s\",\n  \"layers\": %d,\n", g_arch_ok?g_arch.arch:"-", maxl+1);
    fprintf(o, "  \"attention\": \"%s\",\n  \"moe\": \"%s\",\n  \"modality\": \"%s\",\n  \"speculation\": \"%s\",\n",
            atn?atn:"-", moe?moe:"-", g_arch_ok?llmdesc_modality(&g_arch):"-", (g_arch_ok&&g_arch.mm_mtp)?(g_arch.mm_dspark?"DSPARK(MTP)":"MTP"):"NONE");
    fprintf(o, "  \"memory_plan\": { \"weights_bytes\": %llu, \"kv_per_token_bytes\": ", wb); if(kvpt>=0)fprintf(o,"%.0f",kvpt); else fprintf(o,"null"); fprintf(o, " },\n");
    fprintf(o, "  \"exact_capacity\": { \"rope\": 0, \"kv_bound\": %ld },\n", cap);
    fprintf(o, "  \"kernels\": { \"attention\": \"%s\", \"moe\": \"%s\", \"norm\": \"rmsnorm\", \"matmul\": \"affine_qmm\" },\n", atn?atn:"gqa", moe?moe:"moe_sorted_gather");
    fprintf(o, "  \"layer_ops\": [\n"); int first=1;
    for(int L=0;L<=maxl&&L<1024;L++){ if(!F[L])continue; fprintf(o,"%s    { \"layer\": %d, \"ops\": [",first?"":",\n",L);
        int f1=1; if(F[L]&1){fprintf(o,"%s\"attention\"",f1?"":" ,");f1=0;} if(F[L]&4){fprintf(o,"%s\"ffn\"",f1?"":" ,");f1=0;} if(F[L]&2){fprintf(o,"%s\"moe\"",f1?"":" ,");f1=0;} if(F[L]&8){fprintf(o,"%s\"norm\"",f1?"":" ,");f1=0;}
        fprintf(o,"] }"); first=0; }
    fprintf(o, "\n  ]\n}\n"); fclose(o);
    fprintf(LOUT, "LLM TENSOR COMPILE '%s' -> '%s' (arch=%s layers=%d capacity=%ld)\n", path, out_path, g_arch_ok?g_arch.arch:"-", maxl+1, cap);
    return 0;
}
/* GPU (§4.7 SIMT, report-only): divergenza/uniformita' del routing MoE + kernel specialization +
 * GPUCostModel. Analisi statica dal modello/arch (nessun GPU backend qui). */
static int llmmodelio_model_ir_gpu(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR GPU: solo GGUF (per ora)\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    unsigned long long wb = 0; long nexp = -1, nact = -1, nl = -1;
    for (uint64_t i = 0; i < fi.n_tensors; i++) { uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d]; uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) wb += b; }
    for (int i = 0; i < nkv; i++) { const char* k = kv[i].key; long v = (long)kv[i].u32;
        if (strstr(k, "expert_count")) nexp = v; else if (strstr(k, "expert_used_count")) nact = v; else if (strstr(k, "block_count")) nl = v; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    const char* atn = g_arch_ok ? llmdesc_struct_type(&g_arch,"attention") : NULL;
    double div = (nexp > 0 && nact > 0) ? (double)nact / (double)nexp : -1.0;  /* activated/routed = "spread" */
    fprintf(LOUT, "# LLM TENSOR GPU '%s' arch=%s\n", path, g_arch_ok?g_arch.arch:"-");
    fprintf(LOUT, "  kernel_specialization: attention=%s moe=%s matmul=affine_qmm norm=rmsnorm\n",
            atn?atn:"gqa", (nexp>0)?"moe_sorted_gather":"-");
    if (div >= 0) fprintf(LOUT, "  routing: experts=%ld activated=%ld spread=%.3f -> %s\n", nexp, nact, div,
                          div < 0.1 ? "ALTA divergenza (batch MoE: sort/group->compact->GEMM->scatter)" : "bassa divergenza");
    else fprintf(LOUT, "  routing: n/d (expert_count/used_count assenti)\n");
    fprintf(LOUT, "  gpu_cost_model: layers=%ld weights=%llu bytes (%.2f GiB)\n", nl, wb, (double)wb/(1024.0*1024.0*1024.0));
    return 0;
}
/* PROFILE (I4) + RESIDENCY (I12) + STATE (I7) — report-only. */
static int llmmodelio_model_ir_profile(const char* path, int budget_gb) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR PROFILE: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    unsigned long long cat[8] = {0}; unsigned long long wb = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        const char* r = g_arch_ok ? llmdesc_role(&g_arch, fi.tensors[i].name) : "OTHER";
        uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d];
        uint64_t b = 0; llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b); wb += b;
        int c = (strcmp(r,"EMBEDDING")==0)?0:(strncmp(r,"ATTN",4)==0)?1:((strcmp(r,"ROUTED_EXPERT")==0||strcmp(r,"SHARED_EXPERT")==0)?2:(strcmp(r,"ROUTER")==0)?3:(strncmp(r,"FFN",3)==0)?4:(strstr(r,"NORM")!=NULL)?5:(strcmp(r,"OUTPUT")==0)?6:7);
        cat[c] += b;
    }
    long bc=-1,hkv=-1,hd=-1,kl=-1,vl=-1,en=-1;
    for (int i=0;i<nkv;i++){ const char* k=kv[i].key; long v=(long)kv[i].u32;
        if(strstr(k,"attention.head_count_kv"))hkv=v; else if(strstr(k,"attention.head_count"))hd=v;
        else if(strstr(k,"attention.key_length"))kl=v; else if(strstr(k,"attention.value_length"))vl=v;
        else if(strstr(k,"block_count"))bc=v; else if(strstr(k,"embedding_length"))en=v; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    double kvpt=-1.0; if(bc>0&&hkv>0){ long kd=kl>0?kl:((en>0&&hd>0)?en/hd:-1); long vd=vl>0?vl:kd; if(kd>0&&vd>0) kvpt=(double)bc*(double)hkv*(double)(kd+vd)*2.0; }
    double budget = (budget_gb > 0) ? (double)budget_gb*1024.0*1024.0*1024.0 : 96.0*1024.0*1024.0*1024.0;
    const char* cn[8]={"embedding","attention","experts","router","ffn_dense","norm","output","other"};
    fprintf(LOUT, "# LLM TENSOR PROFILE '%s' arch=%s weights=%llu (%.2f GiB)\n", path, g_arch_ok?g_arch.arch:"-", wb, (double)wb/(1024.0*1024.0*1024.0));
    for (int c=0;c<8;c++) if (cat[c]) fprintf(LOUT, "  %-10s %.2f GiB (%.1f%%)\n", cn[c], (double)cat[c]/(1024.0*1024.0*1024.0), 100.0*(double)cat[c]/(double)(wb?wb:1));
    if (kvpt >= 0) fprintf(LOUT, "  kv/token=%.0f bytes\n", kvpt);
    int fits = ((double)wb <= budget);
    fprintf(LOUT, "  RESIDENCY: budget=%.0f GiB -> %s%s\n", budget/(1024.0*1024.0*1024.0), fits ? "RESIDENT (tutto in RAM)" : "SWAP (streaming out-of-core)",
            (!fits && cat[2]) ? " — esperti candidate al demote/streaming" : "");
    return 0;
}
/* DIST (§4.6, tensor-parallel planner) + CED (§4.8, computation elimination) — report-only. */
static int llmmodelio_model_ir_dist(const char* path, int world) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR DIST: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    if (world <= 0) world = 1;
    unsigned long long expb = 0, attnb = 0, otherb = 0; long nexp = -1, nh = -1, nkvh = -1;
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        const char* r = g_arch_ok ? llmdesc_role(&g_arch, fi.tensors[i].name) : "OTHER";
        uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d];
        uint64_t b = 0; llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b);
        if (strcmp(r,"ROUTED_EXPERT")==0) expb += b; else if (strncmp(r,"ATTN",4)==0) attnb += b; else otherb += b;
    }
    for (int i=0;i<nkv;i++){ const char* k=kv[i].key; long v=(long)kv[i].u32;
        if(strstr(k,"expert_count"))nexp=v; else if(strstr(k,"attention.head_count_kv"))nkvh=v; else if(strstr(k,"attention.head_count"))nh=v; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    fprintf(LOUT, "# LLM TENSOR DIST '%s' world_size=%d\n", path, world);
    fprintf(LOUT, "  expert-parallel (EP): experts=%ld -> %ld/rank (byte/rank=%.2f GiB)\n", nexp>0?nexp:0, nexp>0?nexp/world:0, (double)expb/(double)world/(1024.0*1024.0*1024.0));
    fprintf(LOUT, "  tensor-parallel (TP): attn heads=%ld kv=%ld -> %ld/%ld head/rank (byte/rank=%.2f GiB)\n", nh, nkvh, (nh>0?nh/world:0), (nkvh>0?nkvh/world:0), (double)attnb/(double)world/(1024.0*1024.0*1024.0));
    fprintf(LOUT, "  replicated (other): %.2f GiB\n  nota: EP=esperti, TP=teste/KV; all-reduce su residual (TP).\n", (double)otherb/(1024.0*1024.0*1024.0));
    return 0;
}
static int llmmodelio_model_ir_ced(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR CED: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    /* eliminazione model-aware (report-only): tensori ALL-ZERO (candidati elimina) — lettura campionata */
    FILE* f = fopen(path, "rb");
    int allzero = 0, checked = 0;
    for (uint64_t i = 0; i < fi.n_tensors && f; i++) {
        uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d];
        uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) != 0 || b == 0) continue;
        if (b > (1u<<20)) continue;   /* campiona solo tensori <1MB (piccoli): niente scan dei giganti */
        unsigned char* buf = (unsigned char*)malloc((size_t)b);
        if (!buf) continue;
        if (fseek(f, (long)(fi.data_start + fi.tensors[i].offset), SEEK_SET) == 0 && fread(buf, 1, (size_t)b, f) == (size_t)b) {
            checked++; int z = 1; for (uint64_t k = 0; k < b; k++) if (buf[k]) { z = 0; break; }
            if (z) { allzero++; if (allzero <= 8) fprintf(LOUT, "  ALL-ZERO (candidato ELIMINA): %s\n", fi.tensors[i].name); }
        }
        free(buf);
    }
    if (f) fclose(f);
    fprintf(LOUT, "# LLM TENSOR CED '%s' arch=%s: tensori campionati(piccoli)=%d all-zero=%d\n", path, g_arch_ok?g_arch.arch:"-", checked, allzero);
    fprintf(LOUT, "  GATE/kill-switch: %s (soglia all-zero>0 = eliminazione proposta; OFF di default = report-only)\n",
            allzero > 0 ? "ELIMINABILE (candidati presenti) — richiede A/B + quality gate per abilitare" : "NESSUN candidato (niente da eliminare)");
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    return 0;
}
/* SCHEDULE (§4.6, continuous batching plan) — report-only: max batch da budget KV. */
static int llmmodelio_model_ir_schedule(const char* path, int seqlen) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR SCHEDULE: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    unsigned long long wb = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) { uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d]; uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) wb += b; }
    long bc=-1,hkv=-1,hd=-1,kl=-1,vl=-1,en=-1;
    for (int i=0;i<nkv;i++){ const char* k=kv[i].key; long v=(long)kv[i].u32;
        if(strstr(k,"attention.head_count_kv"))hkv=v; else if(strstr(k,"attention.head_count"))hd=v;
        else if(strstr(k,"attention.key_length"))kl=v; else if(strstr(k,"attention.value_length"))vl=v;
        else if(strstr(k,"block_count"))bc=v; else if(strstr(k,"embedding_length"))en=v; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    double kvpt=-1.0; if(bc>0&&hkv>0){ long kd=kl>0?kl:((en>0&&hd>0)?en/hd:-1); long vd=vl>0?vl:kd; if(kd>0&&vd>0) kvpt=(double)bc*(double)hkv*(double)(kd+vd)*2.0; }
    if (seqlen <= 0) seqlen = 2048;
    double budget = 96.0*1024.0*1024.0*1024.0 - 2048.0*1024.0*1024.0 - (double)wb;
    long maxbatch = (kvpt>0 && budget>0) ? (long)(budget / (kvpt*(double)seqlen)) : -1;
    fprintf(LOUT, "# LLM TENSOR SCHEDULE '%s' seqlen=%d weights=%.2f GiB\n", path, seqlen, (double)wb/(1024.0*1024.0*1024.0));
    if (kvpt>0) fprintf(LOUT, "  kv/seq=%d bytes (%.2f MiB)  max_batch(kv_bound)=%ld  -> CONTINUOUS BATCHING (paged KV, prefill/decode mixed)\n",
                        (int)(kvpt*seqlen), kvpt*(double)seqlen/(1024.0*1024.0), maxbatch);
    else fprintf(LOUT, "  max_batch: n/d (metadati KV assenti)\n");
    return 0;
}
/* LOADPLAN (§4.9): legge e VALIDA un artefatto .gmbllm (JSON) — artifact=gmbllm, version, arch,
 * layers. 0 ok, -1 invalido/assente. */
static const char* lj_v(const char* j, const char* key);
static void lj_s(const char* p, char* out, size_t cap);

static int llmmodelio_model_ir_loadplan(const char* path) {
    FILE* f = fopen(path, "rb"); if (f == NULL) { fprintf(stderr, "LLM TENSOR LOADPLAN: file assente '%s'\\n", path); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > (1 << 20)) { fclose(f); return -1; }
    char* j = (char*)malloc((size_t)n + 1); if (!j || fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; } j[n] = 0; fclose(f);
    char art[32] = "", arch[96] = ""; int ver = -1;
    lj_s(lj_v(j, "artifact"), art, sizeof art); lj_s(lj_v(j, "arch"), arch, sizeof arch);
    { const char* v = lj_v(j, "version"); if (v) ver = atoi(v); }
    free(j);
    if (strcmp(art, "gmbllm") != 0) { fprintf(stderr, "LLM TENSOR LOADPLAN: non e' un artefatto gmbllm\\n"); return -1; }
    if (ver != 1) { fprintf(stderr, "LLM TENSOR LOADPLAN: versione %d non supportata\\n", ver); return -1; }
    fprintf(LOUT, "LLM TENSOR LOADPLAN '%s': artifact=gmbllm version=%d arch=%s -> VALIDO\n", path, ver, arch);
    return 0;
}
/* STATE (I7) + GRAPH EXPORT (§4.9) — report-only. */
static int llmmodelio_model_ir_state(const char* path) {
    size_t ln = strlen(path);
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) { if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0; }
    else { if (llmmodelio_arch_resolve(path) != 0) return -1; }
    if (!g_arch_ok) { fprintf(LOUT, "LLM TENSOR STATE: arch non risolta\\n"); return -1; }
    int mtp = g_arch.mm_mtp || 0;
    fprintf(LOUT, "# LLM TENSOR STATE '%s' arch=%s\n  phase: prefill+decode (I7)\n  state=%s modality=%s speculation=%s kv.dtype=%s\n",
            path, g_arch.arch, g_arch.runtime_state[0]?g_arch.runtime_state:"(none)", llmdesc_modality(&g_arch),
            mtp?(g_arch.mm_dspark?"DSPARK(MTP)":"MTP"):"NONE", g_arch.runtime_kv_dtype[0]?g_arch.runtime_kv_dtype:"-");
    return 0;
}
static int llmmodelio_model_ir_gexport(const char* path, const char* out) {
    if (!out || !*out) { fprintf(stderr, "LLM TENSOR GEXPORT: atteso <path>, <out.dot>\\n"); return -1; }
    int F[1024]; memset(F, 0, sizeof F); int maxl = -1; size_t ln = strlen(path);
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) {
        if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
        LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
        if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
        for (uint64_t i = 0; i < fi.n_tensors; i++) { const char* r = g_arch_ok ? llmdesc_role(&g_arch, fi.tensors[i].name) : "OTHER";
            int L = model_layer_of(fi.tensors[i].name); if (L < 0) L = 0; if (L > maxl) maxl = L; if (L > 1023) continue;
            if (strncmp(r,"ATTN",4)==0) F[L]|=1; else if (strstr(r,"EXPERT")||strcmp(r,"ROUTER")==0||strstr(fi.tensors[i].name,"exps")) F[L]|=2; else if (strncmp(r,"FFN",3)==0) F[L]|=4; else if (strstr(r,"NORM")) F[L]|=8; }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    } else { if (llmmodelio_arch_resolve(path) != 0) return -1; }
    FILE* o = fopen(out, "w"); if (!o) return -1;
    fprintf(o, "digraph model { rankdir=LR; node [shape=box];\n");
    for (int L = 0; L <= maxl && L < 1024; L++) { if (!F[L]) continue;
        fprintf(o, "  L%d[shape=record,label=\"layer %d\"];\n", L, L);
        if (F[L]&1) fprintf(o, "  L%d -> A%d[label=\"attn\"]; A%d[label=\"attention\"];\n", L, L, L);
        if (F[L]&4) fprintf(o, "  L%d -> F%d[label=\"ffn\"]; F%d[label=\"ffn\"];\n", L, L, L);
        if (F[L]&2) fprintf(o, "  L%d -> M%d[label=\"moe\"]; M%d[label=\"moe\"];\n", L, L, L);
        if (L+1 <= maxl && F[L+1]) fprintf(o, "  L%d -> L%d;\n", L, L+1);
    }
    fprintf(o, "}\n"); fclose(o);
    fprintf(LOUT, "LLM TENSOR GEXPORT '%s' -> '%s' (DOT, layers=%d)\n", path, out, maxl+1);
    return 0;
}
/* MOBJ (M10, MEMORY OBJECT IR): oggetti di memoria per regione (weights/kv/activation) con
 * device, lifetime, transfer_cost. Report-only. */
static int llmmodelio_model_ir_mobj(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR MOBJ: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    unsigned long long wb = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) { uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d]; uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) wb += b; }
    long hkv = -1, kd = -1, vd = -1, en = -1, hd = -1, nl = -1;
    for (int i = 0; i < nkv; i++) { const char* k = kv[i].key; long v = (long)kv[i].u32;
        if (strstr(k,"head_count_kv")) hkv=v; else if (strstr(k,"key_length")) kd=v; else if (strstr(k,"value_length")) vd=v;
        else if (strstr(k,"embedding_length")) en=v; else if (strstr(k,"head_count")) hd=v; else if (strstr(k,"block_count")) nl=v; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    double kvpt=-1; if(nl>0&&hkv>0){ long kd2=kd>0?kd:((en>0&&hd>0)?en/hd:-1); long vd2=vd>0?vd:kd2; if(kd2>0&&vd2>0) kvpt=(double)nl*hkv*(kd2+vd2)*2.0; }
    fprintf(LOUT, "# LLM TENSOR MOBJ '%s' (M10 memory objects)\n", path);
    fprintf(LOUT, "  {region=weights,  bytes=%llu, device=gpu, lifetime=model, transfer_cost=one-time(mmap)}\n", wb);
    if (kvpt>=0) fprintf(LOUT, "  {region=kv,        bytes_per_token=%.0f, device=gpu, lifetime=session, transfer_cost=0}\n", kvpt);
    fprintf(LOUT, "  {region=activation, device=gpu, lifetime=step, transfer_cost=0}\n");
    fprintf(LOUT, "  {region=expert_workset, policy=residency, lifecycle=demote/evict LRU}\n");
    return 0;
}
/* EPLAN (§4.6): assegnazione esperti->rank (round-robin) + conteggio per rank. Report-only. */
static int llmmodelio_model_ir_eplan(const char* path, int world) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR EPLAN: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    long nexp = -1;
    for (int i = 0; i < nkv; i++) if (strstr(kv[i].key, "expert_count")) nexp = (long)kv[i].u32;
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    if (world <= 0) world = 1;
    if (nexp <= 0) { fprintf(LOUT, "# LLM TENSOR EPLAN '%s': n/d experts\\n", path); return 0; }
    fprintf(LOUT, "# LLM TENSOR EPLAN '%s' experts=%ld world=%d\n", path, nexp, world);
    int* cnt = (int*)calloc((size_t)world, sizeof(int));
    for (long e = 0; e < nexp; e++) cnt[e % world]++;
    for (int r = 0; r < world; r++) fprintf(LOUT, "  rank %d: %d esperti\n", r, cnt[r]);
    free(cnt);
    return 0;
}
/* CONTEXT (§4.10): budget di contesto (max token da KV) + trim/GC. Report-only. */
static int llmmodelio_model_ir_context(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR CONTEXT: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    unsigned long long wb = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++) { uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d]; uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) wb += b; }
    long bc=-1,hkv=-1,hd=-1,kl=-1,vl=-1,en=-1;
    for (int i=0;i<nkv;i++){ const char* k=kv[i].key; long v=(long)kv[i].u32;
        if(strstr(k,"head_count_kv"))hkv=v; else if(strstr(k,"head_count"))hd=v; else if(strstr(k,"key_length"))kl=v;
        else if(strstr(k,"value_length"))vl=v; else if(strstr(k,"block_count"))bc=v; else if(strstr(k,"embedding_length"))en=v; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    double kvpt=-1; if(bc>0&&hkv>0){ long kd=kl>0?kl:((en>0&&hd>0)?en/hd:-1); long vd=vl>0?vl:kd; if(kd>0&&vd>0) kvpt=(double)bc*hkv*(kd+vd)*2.0; }
    long maxtok = (kvpt>0) ? (long)((96.0*1024.0*1024.0*1024.0 - 2048.0*1024.0*1024.0 - (double)wb) / kvpt) : -1;
    fprintf(LOUT, "# LLM TENSOR CONTEXT '%s' arch=%s\n  kv/token=%.0f bytes  max_tokens(kv_bound)=%ld\n", path, g_arch_ok?g_arch.arch:"-", kvpt, maxtok);
    fprintf(LOUT, "  policy: TRIM (drop oldest prefix) | GC (evict prefix-cache) | PAGED (kv pages)\n");
    return 0;
}
/* QUALITY (I6, gate NLL/perplexity): carica il modello col motore MLX (dlopen, dipendenza
 * debole), tokenizza (llmtoken), punteggia token-per-token con logprob_of -> NLL/perplexity.
 * Nessun link statico MLX (una sola istanza; i backend C restano snelli). */
static int llm_quality_nll(const char* model_dir, const char* prompt, double* nll_mean, double* ppl, int* tokens);
static int llm_quality_nll(const char* model_dir, const char* prompt, double* out_nll_mean, double* out_ppl, int* out_tokens) {
    void* h = dlopen("libgmb_ai_mlx.dylib", RTLD_NOW | RTLD_LOCAL);
    if (h == NULL) h = dlopen("libllm_ai_mlx.dylib", RTLD_NOW | RTLD_LOCAL);
    if (h == NULL) { fprintf(stderr, "LLM QUALITY: engine MLX non caricabile (libgmb_ai_mlx)\n"); return -1; }
    typedef void* (*load_fn)(const char*); typedef void* (*sess_fn)(void*);
    typedef int (*app_fn)(void*, const int32_t*, int); typedef int (*lp_fn)(void*, int32_t, float*);
    typedef void (*free_fn)(void*); typedef void (*dsess_fn)(void*);
    load_fn L = (load_fn)dlsym(h, "llm_ai_mlx_load_model");
    sess_fn C = (sess_fn)dlsym(h, "llm_ai_mlx_create_session");
    app_fn  A = (app_fn)dlsym(h, "llm_ai_mlx_session_append");
    lp_fn   LP = (lp_fn)dlsym(h, "llm_ai_mlx_session_logprob_of");
    free_fn FM = (free_fn)dlsym(h, "llm_ai_mlx_free_model");
    dsess_fn DS = (dsess_fn)dlsym(h, "llm_ai_mlx_destroy_session");
    if (L == NULL || C == NULL || A == NULL || LP == NULL) { fprintf(stderr, "LLM QUALITY: simboli engine assenti\n"); return -1; }
    char tp[1300]; snprintf(tp, sizeof tp, "%s/tokenizer.json", model_dir);
    llmtoken_handle th = llmtoken_load(tp); if (th == NULL) { fprintf(stderr, "LLM QUALITY: tokenizer.json assente\n"); return -1; }
    int32_t ids[2048]; int n = llmtoken_encode(th, prompt, ids, 2048);
    if (n <= 1) { llmtoken_free(th); fprintf(stderr, "LLM QUALITY: prompt troppo corto\n"); return -1; }
    void* m = L(model_dir); if (m == NULL) { llmtoken_free(th); fprintf(stderr, "LLM QUALITY: load modello fallito\n"); return -1; }
    void* s = C(m); if (s == NULL) { if (FM) FM(m); llmtoken_free(th); return -1; }
    double nll = 0.0; int cnt = 0;
    for (int k = 0; k + 1 < n; k++) {
        if (A(s, &ids[k], 1) != 0) break;
        float lp = 0.0f; if (LP(s, ids[k + 1], &lp) != 0) break;
        nll += -(double)lp; cnt++;
    }
    double nll_mean = cnt > 0 ? nll / (double)cnt : -1.0;
    double ppl = nll_mean >= 0 ? exp(nll_mean) : -1.0;
    if (DS) DS(s); if (FM) FM(m); llmtoken_free(th);
    if (out_nll_mean) *out_nll_mean = nll_mean; if (out_ppl) *out_ppl = ppl; if (out_tokens) *out_tokens = n;
    return cnt > 0 ? 0 : -1;
}
/* Wrapper: NLL del modello; se ref_model dato -> GATE (REJECT-FLAG se ppl peggiore > 2%). */
int llmmodelio_model_quality(const char* model_dir, const char* prompt, const char* out_path, const char* ref_model) {
    if (model_dir == NULL || prompt == NULL || out_path == NULL) return -1;
    double nll = -1, ppl = -1; int tok = 0;
    if (llm_quality_nll(model_dir, prompt, &nll, &ppl, &tok) != 0) return -1;
    FILE* o = fopen(out_path, "w");   /* non fatale: il valore NLL resta valido anche senza report */
    if (ref_model != NULL && *ref_model != '\0') {
        double rn = -1, rp = -1; int rt = 0;
        if (llm_quality_nll(ref_model, prompt, &rn, &rp, &rt) != 0) { fclose(o); return -1; }
        const char* verdict = (rp <= 0) ? "ACCEPT" : (ppl <= rp * 1.02 ? "ACCEPT" : "REJECT-FLAG");
        if (o != NULL) fprintf(o, "model=%s tokens=%d nll_mean=%.6f ppl=%.6f\nref=%s ppl=%.6f\nGATE: %s (ppl %s rif..)\n",
                model_dir, tok, nll, ppl, ref_model, rp, verdict, ppl <= rp ? "<=" : ">");
        fprintf(LOUT, "LLM QUALITY '%s' ppl=%.4f vs ref '%s' ppl=%.4f -> GATE %s\n", model_dir, ppl, ref_model, rp, verdict);
    } else {
        if (o != NULL) fprintf(o, "model=%s tokens=%d nll_mean=%.6f ppl=%.6f\n", model_dir, tok, nll, ppl);
        fprintf(LOUT, "LLM QUALITY '%s': tokens=%d nll_mean=%.4f ppl=%.4f\n", model_dir, tok, nll, ppl);
    }
    if (o != NULL) fclose(o);
    return 0;
}
int llmmodelio_model_compile(const char* path, const char* out_path) { return llmmodelio_model_ir_compile(path, out_path); }

/* QUANTIR (I11) + MEMPLAN + KVPLAN + VIEW — report-only (§4.4/§4.5/§4.6/§4.9). */
static int llmmodelio_model_ir_quantir(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR QUANTIR: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    fprintf(LOUT, "# LLM TENSOR QUANTIR '%s' arch=%s (I11: precisione per OP)\n# role\tweights\tactivation\taccum\toutput\n", path, g_arch_ok?g_arch.arch:"-");
    for (int i = 0; i < g_arch.n_quant; i++) {
        const char* f = g_arch.quant[i].fmt;
        fprintf(LOUT, "%s\tweights=%s\tactivation=f16\taccum=f32\toutput=%s\n", g_arch.quant[i].role, f, f);
    }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    return 0;
}
/* QUALITY BUDGET O0..O4 (§4.5 I6): mappa il livello a un budget bit/peso e verifica il piano per ruolo. */
static double llmo_bits(const char* f) {
    if (!f) return 0;
    if (strstr(f, "f32")) return 32; if (strstr(f, "f16") || strstr(f, "bf16")) return 16;
    if (strstr(f, "q8")) return 8.5; if (strstr(f, "q6")) return 6.6; if (strstr(f, "q5")) return 5.6;
    if (strstr(f, "mxfp4") || strstr(f, "q4") || strstr(f, "iq4")) return 4.5;
    if (strstr(f, "q3")) return 3.4; if (strstr(f, "q2")) return 2.6; if (strstr(f, "iq2")) return 2.1;
    return 4.5;
}
/* Riconosce i ruoli da nomi GGUF (blk.N.attn_q, ffn_gate_exps...) E HF/MLX (model.layers.N.self_attn.q_proj...). */
static const char* llmo_role_of(const char* nm) {
    if (!nm) return "other";
    /* HF/MLX (piu' specifici per primi) */
    if (strstr(nm, "switch_mlp")) {
        if (strstr(nm, "gate_proj")) return "experts.gate";
        if (strstr(nm, "up_proj"))   return "experts.up";
        if (strstr(nm, "down_proj")) return "experts.down";
        return "experts";
    }
    if (strstr(nm, "shared_experts")) {
        if (strstr(nm, "gate_proj")) return "shared.gate";
        if (strstr(nm, "up_proj"))   return "shared.up";
        if (strstr(nm, "down_proj")) return "shared.down";
        return "shared";
    }
    if (strstr(nm, "self_attn.q_proj") || strstr(nm, "attn_q")) return "attn.q";
    if (strstr(nm, "self_attn.k_proj") || strstr(nm, "attn_k")) return "attn.k";
    if (strstr(nm, "self_attn.v_proj") || strstr(nm, "attn_v")) return "attn.v";
    if (strstr(nm, "kv_a_proj") || strstr(nm, "kv_b_proj")) return "attn.kv";
    if (strstr(nm, "self_attn.o_proj") || strstr(nm, "attn_output") || strstr(nm, "attn_out")) return "attn.o";
    if (strstr(nm, "mlp.gate.weight") || strstr(nm, "ffn_gate_inp")) return "ffn.router";
    if (strstr(nm, "ffn_gate") || strstr(nm, "gate_proj")) return "ffn.gate";
    if (strstr(nm, "ffn_up") || strstr(nm, "up_proj")) return "ffn.up";
    if (strstr(nm, "ffn_down") || strstr(nm, "down_proj")) return "ffn.down";
    if (strstr(nm, "token_embd") || strstr(nm, "embed_tokens")) return "embed";
    if (strstr(nm, "lm_head") || strstr(nm, "output")) return "output";
    if (strstr(nm, "norm") || strstr(nm, "layernorm")) return "norm";
    return "other";
}
static int llmmodelio_model_ir_obudget(const char* path, const char* level) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR OBUDGET: solo GGUF\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    int lv = 2; if (level && (level[0] == 'O' || level[0] == 'o')) lv = atoi(level + 1);
    if (lv < 0) lv = 0; if (lv > 4) lv = 4;
    static const double tgt[5] = {16.0, 8.5, 4.5, 2.6, 2.1};
    static const char* nm[5] = {"O0 (f16, qualita' massima)", "O1 (q8)", "O2 (q4/mxfp4)", "O3 (q3/q2)", "O4 (iq2, massima compressione)"};
    double budget = tgt[lv];
    fprintf(LOUT, "# LLM TENSOR OBUDGET '%s' arch=%s qualita'=%s -> budget=%.1f bit/peso\n# role\tfmt\tbits\tverdetto\n",
            path, g_arch_ok?g_arch.arch:"-", nm[lv], budget);
    int over = 0, roles = 0, fallback = (g_arch.n_quant == 0);
    if (!fallback) {
        for (int i = 0; i < g_arch.n_quant; i++) {
            const char* f = g_arch.quant[i].fmt; double b = llmo_bits(f);
            const char* v = (b <= budget) ? "OK" : "OVER"; if (b > budget) over++;
            fprintf(LOUT, "%s\t%s\t%.1f\t%s\n", g_arch.quant[i].role, f, b, v); roles++;
        }
    } else {
        /* fallback: deriva i ruoli dai NOMI tensore + bits/peso dal tipo ggml reale. */
        const char* order[16] = {"embed","attn.q","attn.k","attn.v","attn.kv","attn.o","ffn.router","ffn.gate","ffn.up","ffn.down","experts.gate","experts.up","experts.down","shared.down","output","other"};
        double rb[16] = {0}; unsigned long long rbytes[16] = {0}, rnum[16] = {0}; int rt[16]; for (int i=0;i<16;i++) rt[i]=-1;
        for (uint64_t i = 0; i < fi.n_tensors; i++) {
            const char* r = llmo_role_of(fi.tensors[i].name);
            int idx = 15; for (int j=0;j<15;j++) if (strcmp(order[j], r)==0) { idx=j; break; }
            uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d];
            uint64_t bb = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &bb) == 0) { rbytes[idx]+=bb; rnum[idx]+=numel; }
            if (rt[idx] < 0) rt[idx] = fi.tensors[i].ggml_type;
        }
        for (int j = 0; j < 15; j++) {
            if (rnum[j] == 0) continue;
            rb[j] = (double)(rbytes[j]*8)/(double)rnum[j];
            char tn[64]; if (llmcodec_ggml_type_name(rt[j], tn, sizeof tn) != 0) snprintf(tn, sizeof tn, "type%d", rt[j]);
            const char* v = (rb[j] <= budget) ? "OK" : "OVER"; if (rb[j] > budget) over++;
            fprintf(LOUT, "%s\t%s\t%.2f\t%s\n", order[j], tn, rb[j], v); roles++;
        }
    }
    fprintf(LOUT, "# TOTALE ruoli=%d over_budget=%d fonte=%s -> %s\n", roles, over, fallback ? "tensori(fallback)" : "arch",
            over == 0 ? "ACCEPT (piano entro il budget)" : "REJECT-FLAG (ruoli sopra il budget)");
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    return 0;
}
static int llmmodelio_model_ir_memplan(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR MEMPLAN: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    unsigned long long wb = 0; for (uint64_t i = 0; i < fi.n_tensors; i++) { uint64_t numel = 1; for (int d = 0; d < fi.tensors[i].ndim; d++) numel *= fi.tensors[i].dims[d]; uint64_t b = 0; if (llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, numel, &b) == 0) wb += b; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    fprintf(LOUT, "# LLM TENSOR MEMPLAN '%s' (STATIC MEMORY PLAN / arena)\n", path);
    fprintf(LOUT, "  arena[0] weights  size=%llu B (region=model, lifetime=process, align=page)\n", wb);
    fprintf(LOUT, "  arena[1] kv       size=kv/token*ctx (region=session, lifetime=session)\n");
    fprintf(LOUT, "  arena[2] scratch  size=O(max_act) (region=step, lifetime=step, riuso)\n");
    return 0;
}
static int llmmodelio_model_ir_kvplan(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR KVPLAN: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    fprintf(LOUT, "# LLM TENSOR KVPLAN '%s' (§4.6 KV IR)\n  PAGE (block fisso) | MAP (block<->asse) | UNMAP (evict) | PREFETCH (router->kv) | EVICT (LRU) | SHARE (prefix-cache)\n", path);
    fprintf(LOUT, "  mode=FULL|PAGED|ROTATING|DISK_BACKED  kv.dtype=%s\n", (g_arch_ok && g_arch.runtime_kv_dtype[0]) ? g_arch.runtime_kv_dtype : "-");
    return 0;
}
static int llmmodelio_model_ir_view(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR VIEW: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    fprintf(LOUT, "# LLM TENSOR VIEW '%s' (VIEW/LAYOUT IR) — primi 8 tensori\n", path);
    for (uint64_t i = 0; i < fi.n_tensors && i < 8; i++) { LlmCodecGgufTensorInfo* t = &fi.tensors[i]; char ds[96]; size_t o=0; ds[0]=0; for (int d=0; d<t->ndim; d++) o+=(size_t)snprintf(ds+o,sizeof ds-o,"%s%llu",d?"x":"",(unsigned long long)t->dims[d]); fprintf(LOUT, "  %s: layout=contiguous dims=%s view=zero-copy(mmap)\n", t->name, ds); }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    return 0;
}
/* POLICYIR (§4.3 policy definibile) + IRLEVELS (§4.2 L0-L5) — report-only. */
static int llmmodelio_model_ir_policyir(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR POLICYIR: solo GGUF\\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    if (!g_arch_ok) { fprintf(stderr, "LLM TENSOR POLICYIR: arch non risolta\\n"); return -1; }
    fprintf(LOUT, "# LLM TENSOR POLICYIR '%s' arch=%s (Policy definibile — regole dichiarative)\n", path, g_arch.arch);
    fprintf(LOUT, "  WHEN role=%s THEN fmt=%s (quant); recipe=%s boost_fmt=%s\n",
            "ROUTED_EXPERT", llmdesc_fmt(&g_arch, "ROUTED_EXPERT") ? llmdesc_fmt(&g_arch, "ROUTED_EXPERT") : "-", g_arch.recipe, g_arch.boost_fmt);
    for (int i = 0; i < g_arch.n_quant; i++)
        fprintf(LOUT, "  WHEN role=%s THEN fmt=%s\n", g_arch.quant[i].role, g_arch.quant[i].fmt);
    fprintf(LOUT, "  DEFAULT fmt=(dal modello)  PRIORITY=roles-first-match\n");
    return 0;
}
/* I7 — fasi PREFILL | PREFILL_KV | DECODE: geometria, KV, classe di op, roofline (§4.5). */
static int llmmodelio_model_ir_phase(const char* path, long seq) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR PHASE: solo GGUF\n"); return -1; }
    if (llmmodelio_arch_resolve_gguf(path) != 0) g_arch_ok = 0;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(path, &kv, &nkv, &fi) != 0) return -1;
    long bc = -1, hkv = -1, hd = -1, kl = -1, vl = -1, en = -1;
    for (int i = 0; i < nkv; i++) { const char* k = kv[i].key; long v = (long)kv[i].u32;
        if (strstr(k,"attention.head_count_kv")) hkv=v; else if (strstr(k,"attention.head_count")) hd=v;
        else if (strstr(k,"attention.key_length")) kl=v; else if (strstr(k,"attention.value_length")) vl=v;
        else if (strstr(k,"block_count")) bc=v; else if (strstr(k,"embedding_length")) en=v; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    if (seq <= 0) seq = 512;
    long kd = kl>0?kl:((en>0&&hd>0)?en/hd:-1); long vd = vl>0?vl:kd;
    double kvpt = (bc>0&&hkv>0&&kd>0&&vd>0) ? (double)bc*(double)hkv*(double)(kd+vd)*2.0 : -1.0;
    fprintf(LOUT, "# LLM TENSOR PHASE '%s' arch=%s seq=%ld (I7: PREFILL | PREFILL_KV | DECODE)\n",
            path, g_arch_ok?g_arch.arch:"-", seq);
    fprintf(LOUT, "# fase\ttokens\tbatch\tkv_step(B)\tkv_total(B)\top\troofline\n");
    double kv_pre = kvpt>0 ? kvpt*(double)seq : -1.0;
    fprintf(LOUT, "PREFILL\t%ld\t%ld\t%.0f\t%.0f\t%s\tcompute-bound (GEMM, AI alto)\n",
            seq, seq, kvpt, kv_pre, "matmul(batched)");
    fprintf(LOUT, "PREFILL_KV\t%ld\t%ld\t%.0f\t%.0f\t%s\tKV scritta durante prefill, riusata in decode\n",
            0, 0, kvpt, kv_pre, "kv_write");
    fprintf(LOUT, "DECODE\t1\t1\t%.0f\t%.0f\t%s\tmemory-bound (GEMV, AI basso; KV riuso, no recompute)\n",
            kvpt, kvpt, "matvec+attn");
    fprintf(LOUT, "# note: prefill domina il compute, decode domina la banda (KV). kv/token=%.1f B, kv@seq=%.0f B\n", kvpt, kv_pre);
    return 0;
}
static int llmmodelio_model_ir_irlevels(const char* path) {
    size_t ln = strlen(path);
    if (!(ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0)) { fprintf(stderr, "LLM TENSOR IRLEVELS: solo GGUF\\n"); return -1; }
    fprintf(LOUT, "# LLM TENSOR IRLEVELS '%s' (gerarchia IR L0-L5)\n", path);
    fprintf(LOUT, "  L0 source (HF/MLX/safetensors) | L1 container (GGUF) <-- questo | L2 Model IR (ruoli) | L3 primitive (attn/moe) | L4 execution | L5 lowering(MLX/Metal/C)\n");
    fprintf(LOUT, "  fasi: PREFILL | PREFILL_KV | DECODE\n");
    return 0;
}
int llmmodelio_tensor_cmd(const char* sub, const char* a, const char* b)
{
    const char* s = sub ? sub : "";
    const char* dir = a ? a : "";
    const char* tn = b ? b : "";
    if (strcasecmp(s, "ROLE") == 0) {
        if (!*dir || !*tn) { fprintf(stderr, "LLM TENSOR ROLE: attesi <dir>, <tensor>\n"); return -1; }
        if (!g_arch_ok && llmmodelio_arch_resolve(dir) != 0) return -1;
        const char* role = llmdesc_role(&g_arch, tn);
        const char* fmt = llmdesc_fmt(&g_arch, role);
        fprintf(LOUT, "LLM TENSOR '%s': %s -> ruolo=%s formato=%s\n", tn, g_arch.arch, role, fmt ? fmt : "-");
        return 0;
    }
    if (strcasecmp(s, "SCHEDULE") == 0 || strcasecmp(s, "BATCH") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR SCHEDULE: atteso <path>[, <seqlen>]\n"); return -1; }
        return llmmodelio_model_ir_schedule(dir, *tn ? atoi(tn) : 0);
    }
    if (strcasecmp(s, "DIST") == 0 || strcasecmp(s, "DISTRIBUTION") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR DIST: atteso <path>[, <world>]\n"); return -1; }
        return llmmodelio_model_ir_dist(dir, *tn ? atoi(tn) : 1);
    }
    if (strcasecmp(s, "CED") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR CED: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_ced(dir);
    }
    if (strcasecmp(s, "PROFILE") == 0 || strcasecmp(s, "RESIDENCY") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR PROFILE: atteso <path>[, <budget_gb>]\n"); return -1; }
        return llmmodelio_model_ir_profile(dir, *tn ? atoi(tn) : 0);
    }
    if (strcasecmp(s, "GPU") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR GPU: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_gpu(dir);
    }
    if (strcasecmp(s, "POLICYIR") == 0) { if(!*dir){fprintf(stderr,"LLM TENSOR POLICYIR: atteso <path>\n");return -1;} return llmmodelio_model_ir_policyir(dir); }
    if (strcasecmp(s, "PHASE") == 0 || strcasecmp(s, "PREFILL") == 0 || strcasecmp(s, "PREFILL_KV") == 0 || strcasecmp(s, "DECODE") == 0) { if(!*dir){fprintf(stderr,"LLM TENSOR PHASE: atteso <path>[, <seqlen>]\n");return -1;} return llmmodelio_model_ir_phase(dir, *tn ? atol(tn) : 0); }
    if (strcasecmp(s, "IRLEVELS") == 0) { if(!*dir){fprintf(stderr,"LLM TENSOR IRLEVELS: atteso <path>\n");return -1;} return llmmodelio_model_ir_irlevels(dir); }
    if (strcasecmp(s, "QUANTIR") == 0 || strcasecmp(s, "QUANT_IR") == 0) { if(!*dir){fprintf(stderr,"LLM TENSOR QUANTIR: atteso <path>\n");return -1;} return llmmodelio_model_ir_quantir(dir); }
    if (strcasecmp(s, "OBUDGET") == 0 || strcasecmp(s, "QUALITYLEVEL") == 0 || strcasecmp(s, "OBUDGETIR") == 0) { if(!*dir){fprintf(stderr,"LLM TENSOR OBUDGET: atteso <path>[, <O0..O4>]\n");return -1;} return llmmodelio_model_ir_obudget(dir, tn); }
    if (strcasecmp(s, "MEMPLAN") == 0) { if(!*dir){fprintf(stderr,"LLM TENSOR MEMPLAN: atteso <path>\n");return -1;} return llmmodelio_model_ir_memplan(dir); }
    if (strcasecmp(s, "KVPLAN") == 0) { if(!*dir){fprintf(stderr,"LLM TENSOR KVPLAN: atteso <path>\n");return -1;} return llmmodelio_model_ir_kvplan(dir); }
    if (strcasecmp(s, "VIEW") == 0) { if(!*dir){fprintf(stderr,"LLM TENSOR VIEW: atteso <path>\n");return -1;} return llmmodelio_model_ir_view(dir); }
    if (strcasecmp(s, "CONTEXT") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR CONTEXT: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_context(dir);
    }
    if (strcasecmp(s, "EPLAN") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR EPLAN: atteso <path>[, <world>]\n"); return -1; }
        return llmmodelio_model_ir_eplan(dir, *tn ? atoi(tn) : 1);
    }
    if (strcasecmp(s, "MOBJ") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR MOBJ: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_mobj(dir);
    }
    if (strcasecmp(s, "STATE") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR STATE: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_state(dir);
    }
    if (strcasecmp(s, "GEXPORT") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR GEXPORT: atteso <path>, <out.dot>\n"); return -1; }
        return llmmodelio_model_ir_gexport(dir, tn);
    }
    if (strcasecmp(s, "LOADPLAN") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR LOADPLAN: atteso <file.gmbllm>\n"); return -1; }
        return llmmodelio_model_ir_loadplan(dir);
    }
    if (strcasecmp(s, "COMPILE") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR COMPILE: atteso <path>, <out.gmbllm>\n"); return -1; }
        return llmmodelio_model_ir_compile(dir, tn);
    }
    if (strcasecmp(s, "EXECPLAN") == 0 || strcasecmp(s, "PLAN") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR EXECPLAN: atteso <path>, <out>\n"); return -1; }
        return llmmodelio_model_ir_execplan(dir, tn);
    }
    if (strcasecmp(s, "KV") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR KV: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_kv(dir);
    }
    if (strcasecmp(s, "SPEC") == 0 || strcasecmp(s, "SPECULATION") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR SPEC: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_spec(dir);
    }
    if (strcasecmp(s, "COST") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR COST: atteso <path>[, <ctx>]\n"); return -1; }
        return llmmodelio_model_ir_cost(dir, *tn ? atoi(tn) : 0);
    }
    if (strcasecmp(s, "MEMORY") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR MEMORY: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_memory(dir);
    }
    if (strcasecmp(s, "GRAPH") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR GRAPH: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_graph(dir);
    }
    if (strcasecmp(s, "INFO") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR INFO: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_info(dir);
    }
    if (strcasecmp(s, "LAYERS") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR LAYERS: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_layers(dir);
    }
    if (strcasecmp(s, "CHECK") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR CHECK: atteso <path>\n"); return -1; }
        return llmmodelio_model_ir_check(dir);
    }
    if (strcasecmp(s, "IR") == 0 || strcasecmp(s, "MAP") == 0) {
        if (!*dir) { fprintf(stderr, "LLM TENSOR IR: atteso <path mod?>\n"); return -1; }
        return llmmodelio_model_ir_print(dir);
    }
    if (strcasecmp(s, "LIST") == 0 || *s == 0) {
        if (!g_arch_ok && (!*dir || llmmodelio_arch_resolve(dir) != 0)) { fprintf(LOUT, "LLM TENSOR: nessuna descrizione risolta\n"); return -1; }
        fprintf(LOUT, "LLM TENSOR '%s': %d ruoli\n", g_arch.arch, g_arch.n_roles);
        for (int i = 0; i < g_arch.n_roles; i++) {
            const char* f = llmdesc_fmt(&g_arch, g_arch.roles[i].role);
            fprintf(LOUT, "  %-24s -> %-16s %s\n", g_arch.roles[i].match, g_arch.roles[i].role, f ? f : "-");
        }
        return 0;
    }
    fprintf(stderr, "LLM TENSOR: sotto-verbo '%s' non supportato (ROLE|LIST)\n", s);
    return -1;
}

/* KERNEL: sub=SHOW a=<dir> b=<kind> -> pipeline della struttura (primitive/step);
 * sub=CODEC a=<dir> b=<fmt> -> plugin che implementa il formato; sub=PRIM a=<dir> b=<kind> ->
 * plugin che implementa la primitiva. Nessun plugin = il runtime usa il built-in (mai silenzioso). */
int llmmodelio_kernel_cmd(const char* sub, const char* a, const char* b)
{
    const char* s = sub ? sub : "";
    const char* dir = a ? a : "";
    const char* k = b ? b : "";
    if (strcasecmp(s, "CODEC") == 0) {
        if (!*dir || !*k) { fprintf(stderr, "LLM KERNEL CODEC: attesi <dir>, <fmt>\n"); return -1; }
        LlmCodec c; char pp[1024];
        if (llmmodelio_codec_resolve(dir, k, &c, pp, sizeof pp) == 0)
            fprintf(LOUT, "LLM KERNEL CODEC '%s': plugin '%s'%s%s%s\n", k, c.name ? c.name : "?",
                    pp[0] ? " (" : "", pp[0] ? pp : "", pp[0] ? ")" : "");
        else
            fprintf(LOUT, "LLM KERNEL CODEC '%s': nessun plugin dichiarato (il runtime usa il built-in)\n", k);
        return 0;
    }
    if (strcasecmp(s, "PRIM") == 0) {
        if (!*dir || !*k) { fprintf(stderr, "LLM KERNEL PRIM: attesi <dir>, <kind>\n"); return -1; }
        LlmEnginePluginPrim p; char pp[1024];
        if (llmmodelio_prim_resolve(dir, k, &p, pp, sizeof pp) == 0)
            fprintf(LOUT, "LLM KERNEL PRIM '%s': plugin '%s'%s%s%s\n", k, p.name ? p.name : "?",
                    pp[0] ? " (" : "", pp[0] ? pp : "", pp[0] ? ")" : "");
        else
            fprintf(LOUT, "LLM KERNEL PRIM '%s': nessun plugin dichiarato (il runtime usa il built-in)\n", k);
        return 0;
    }
    /* SHOW / PIPELINE / default: se `b` e' una struttura nota, mostra la sua pipeline. */
    if (!g_arch_ok && (!*dir || llmmodelio_arch_resolve(dir) != 0)) { fprintf(LOUT, "LLM KERNEL: nessuna descrizione risolta\n"); return -1; }
    if (*k) {
        char buf[2048]; buf[0] = 0;
        if (llmmodelio_arch_pipeline(*dir ? dir : llmmodelio_arch_path(), k, buf, sizeof buf) >= 0) {
            fprintf(LOUT, "LLM KERNEL '%s': %s\n", k, buf);
            return 0;
        }
        fprintf(LOUT, "LLM KERNEL '%s': struttura senza pipeline dichiarata\n", k);
        return 0;
    }
    fprintf(LOUT, "LLM KERNEL '%s': %d strutture\n", g_arch.arch, g_arch.n_struct);
    for (int i = 0; i < g_arch.n_struct; i++) {
        const LlmDescStruct* S = &g_arch.structures[i];
        fprintf(LOUT, "  %-12s type=%-16s prim=%s", S->key, S->type[0] ? S->type : "-", S->prim[0] ? S->prim : "-");
        for (int j = 0; j < S->n_steps; j++) fprintf(LOUT, " %s->%s", S->steps[j].op, S->steps[j].prim);
        fprintf(LOUT, "\n");
    }
    return 0;
}

/* MODULE: sub=SHOW/SUPPORT a=<dir> -> strutture (moduli) dichiarate e se il converter le sa
 * mappare (llmmodelio_arch_support); sub=REGISTER -> riservato (richiede permesso esplicito: le
 * librerie/moduli sono infrastruttura). Mai no-op silenzioso. */
int llmmodelio_module_cmd(const char* sub, const char* a, const char* b)
{
    const char* s = sub ? sub : "";
    const char* dir = a ? a : "";
    (void)b;
    if (strcasecmp(s, "REGISTER") == 0 || strcasecmp(s, "PRIORITY") == 0) {
        fprintf(stderr, "LLM MODULE %s: riservato (registrazione moduli/librerie richiede permesso esplicito)\n", s);
        return -1;
    }
    if (strcasecmp(s, "MODALITY") == 0) {
        if (!*dir) { fprintf(stderr, "LLM MODULE MODALITY: atteso <path>\n"); return -1; }
        char mod[32], mm[512], ar[96];
        if (llmmodelio_model_modality_of(dir, mod, sizeof mod, mm, sizeof mm, ar, sizeof ar) != 0) return -1;
        fprintf(LOUT, "LLM MODULE MODALITY '%s': arch=%s modality=%s mmproj=%s\n", dir, ar, mod, mm[0] ? mm : "-");
        return 0;
    }
    if (strcasecmp(s, "SHOW") == 0 || strcasecmp(s, "SUPPORT") == 0 || *s == 0) {
        if (!*dir) { fprintf(stderr, "LLM MODULE: atteso <dir>\n"); return -1; }
        char buf[4096];
        int nbad = llmmodelio_arch_support(dir, buf, sizeof buf);
        if (nbad < 0) return -1;
        fprintf(LOUT, "%s", buf);
        return 0;
    }
    fprintf(stderr, "LLM MODULE: sotto-verbo '%s' non supportato (SHOW|SUPPORT)\n", s);
    return -1;
}

/* KV: sub=SHOW a=<dir> -> metadati RUNTIME (engine) dichiarati dalla .llmarch: state, dtype del
 * KV e dell'attivazione. NON sono codec di checkpoint (Knowledge/Encoder-Decoder-Modelli.md §19).
 * sub=SET -> non supportato qui (la KV è configurazione del motore, non del compilatore): errore
 * esplicito, mai silenzioso. */
int llmmodelio_kv_cmd(const char* sub, const char* a, const char* b)
{
    const char* s = sub ? sub : "";
    const char* dir = a ? a : "";
    (void)b;
    if (strcasecmp(s, "SET") == 0) {
        fprintf(stderr, "LLM KV SET: la KV è configurazione del MOTORE (non del compilatore): non supportato qui\n");
        return -1;
    }
    if (strcasecmp(s, "SHOW") == 0 || *s == 0) {
        if (!g_arch_ok && (!*dir || llmmodelio_arch_resolve(dir) != 0)) { fprintf(LOUT, "LLM KV: nessuna descrizione risolta\n"); return -1; }
        fprintf(LOUT, "LLM KV '%s': state=%s kv.dtype=%s activation.dtype=%s\n", g_arch.arch,
                g_arch.runtime_state[0] ? g_arch.runtime_state : "-",
                g_arch.runtime_kv_dtype[0] ? g_arch.runtime_kv_dtype : "-",
                g_arch.runtime_activation_dtype[0] ? g_arch.runtime_activation_dtype : "-");
        return 0;
    }
    fprintf(stderr, "LLM KV: sotto-verbo '%s' non supportato (SHOW)\n", s);
    return -1;
}

/* Wrapper `_to`: come llmmodelio_arch_cmd_to — cattura il testo in `out` per i backend gestiti. */
#define LLMMODELIO_CMD_TO(name, fn)                                                       \
    int name(const char* sub, const char* a, const char* b, char* out, size_t cap)     \
    {                                                                                  \
        if (!out || cap == 0) return fn(sub, a, b);                                    \
        out[0] = 0;                                                                    \
        FILE* f = fmemopen(out, cap - 1, "w");                                          \
        if (!f) return fn(sub, a, b);                                                  \
        g_llm_out = f;                                                                 \
        int rc = fn(sub, a, b);                                                        \
        fflush(f); fclose(f); g_llm_out = NULL;                                        \
        out[cap - 1] = 0;                                                              \
        return rc;                                                                     \
    }
LLMMODELIO_CMD_TO(llmmodelio_tensor_cmd_to, llmmodelio_tensor_cmd)
LLMMODELIO_CMD_TO(llmmodelio_kernel_cmd_to, llmmodelio_kernel_cmd)
LLMMODELIO_CMD_TO(llmmodelio_module_cmd_to, llmmodelio_module_cmd)
LLMMODELIO_CMD_TO(llmmodelio_kv_cmd_to,     llmmodelio_kv_cmd)

/* LLM MODEL CONVERT <src>, <dst>[, <fmt>] — conversione di formato del modello.
 * fmt: GGUF | MLX (il formato GGUF NON esiste piu').
 *   - GGUF  : MLX/HF (safetensors) -> GGUF, guidato dalla .llmarch risolta dal sorgente.
 *   - MLX   : GGUF -> MLX (safetensors), idem.
 * I converter sono quelli del progetto CODEC (llmcodec_conv_*_arch): una sola sorgente per tutti i
 * backend (IL/PCodeVM/C99/VM-C/C99Native), nessuna ricopiatura. */
/* Sorgente .gguf: llmmodelio_arch_resolve si aspetta una dir HF (config.json). Per un file
 * GGUF si ricava `general.architecture` e si risolve <slug>.llmarch cercando in ABI/ (risalendo
 * dal cwd) o <cwd>/arch/ — stessa logica di llmdesc_resolve, senza config.json. */
static int llmmodelio_arch_resolve_gguf(const char* gguf_path) {
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(gguf_path, &kv, &nkv, &fi) != 0) return -1;
    char arch[96] = "";
    for (int i = 0; i < nkv; i++)
        if (strcmp(kv[i].key, "general.architecture") == 0 && kv[i].str) { snprintf(arch, sizeof arch, "%s", kv[i].str); break; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    if (!arch[0]) return -1;
    char slug[96]; { size_t k = 0; for (const char* p = arch; *p && k < sizeof slug - 1; ++p) { char c = *p; if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a'); slug[k++] = c; } slug[k] = 0; }
    char cwd[2048]; if (getcwd(cwd, sizeof cwd) == NULL) return -1;
    char path[4096];
    for (int up = 0; up < 10; up++) {
        snprintf(path, sizeof path, "%s/ABI/%s.llmarch", cwd, slug);
        if (llmdesc_load(path, &g_arch) == 0) { snprintf(g_arch_path, sizeof g_arch_path, "%s", path); g_arch_ok = 1; return 0; }
        snprintf(path, sizeof path, "%s/ABI/%s.gmbarch", cwd, slug);
        if (llmdesc_load(path, &g_arch) == 0) { snprintf(g_arch_path, sizeof g_arch_path, "%s", path); g_arch_ok = 1; return 0; }
        snprintf(path, sizeof path, "%s/arch/%s.llmarch", cwd, slug);
        if (llmdesc_load(path, &g_arch) == 0) { snprintf(g_arch_path, sizeof g_arch_path, "%s", path); g_arch_ok = 1; return 0; }
        char* sl = strrchr(cwd, '/'); if (sl == NULL || sl == cwd) break; *sl = 0;
    }
    g_arch_ok = 0; return -1;
}
/* ── MODALITY: emit automatico del .gmmod da modello+mmproj reali ──
 * Risolve l'arch DALL'ABI confrontando il basename del mmproj con `source.mmproj` dichiarato in
 * ogni .llmarch/.gmbarch (fallback: arch del base/ dell mmproj). Deriva la modality dichiarata,
 * valida che il mmproj porti tensori di quella famiglia, e scrive il descriptor `out_path`. */
static int lj_mod_role(const LlmDesc* a, const char* name, const char* modality);
int llmmodelio_model_modality(const char* base_path, const char* mmproj_path, const char* out_path) {
    if (base_path == NULL || mmproj_path == NULL || out_path == NULL) return -1;
    const char* sl = strrchr(mmproj_path, '/'); const char* mmb = sl ? sl + 1 : mmproj_path;
    int found = 0; char arch[96] = "";
    char abi[2048];
    if (llmdesc_abi_dir(abi, sizeof abi)) {
        DIR* dp = opendir(abi);
        if (dp != NULL) {
            struct dirent* de;
            while ((de = readdir(dp)) != NULL && !found) {
                size_t ln = strlen(de->d_name);
                int isarch = (ln > 8 && (strcmp(de->d_name + ln - 8, ".llmarch") == 0 || strcmp(de->d_name + ln - 8, ".gmbarch") == 0));
                if (!isarch) continue;
                char path[4096]; snprintf(path, sizeof path, "%s/%s", abi, de->d_name);
                LlmDesc d;
                if (llmdesc_load(path, &d) != 0) continue;
                if (d.source_mmproj[0]) {
                    const char* sb = strrchr(d.source_mmproj, '/'); const char* sbn = sb ? sb + 1 : d.source_mmproj;
                    if (strcmp(sbn, mmb) == 0) { g_arch = d; snprintf(g_arch_path, sizeof g_arch_path, "%s", path); g_arch_ok = 1; found = 1; snprintf(arch, sizeof arch, "%s", d.arch); }
                }
            }
            closedir(dp);
        }
    }
    if (!found) {
        if (llmmodelio_arch_resolve_gguf(base_path) != 0 && llmmodelio_arch_resolve_gguf(mmproj_path) != 0) return -1;
        if (g_arch_ok) snprintf(arch, sizeof arch, "%s", g_arch.arch);
    }
    if (!g_arch_ok) return -1;
    const char* mod = llmdesc_modality(&g_arch);
    if (strcmp(mod, "text") == 0) return -1;                  /* nessuna modality dichiarata -> errore */
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(mmproj_path, &kv, &nkv, &fi) != 0) return -1;
    int carry = 0;
    for (uint64_t i = 0; i < fi.n_tensors; i++)
        if (lj_mod_role(&g_arch, fi.tensors[i].name, mod) || lj_mod_role(&g_arch, fi.tensors[i].name, "vision")
            || lj_mod_role(&g_arch, fi.tensors[i].name, "audio") || lj_mod_role(&g_arch, fi.tensors[i].name, "mtp")) { carry = 1; break; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    if (!carry) return -1;                                    /* mmproj incoerente con la modality */
    FILE* o = fopen(out_path, "w"); if (o == NULL) return -1;
    fprintf(o, "{\n  \"module\": \"%s\",\n  \"arch\": \"%s\",\n  \"modality\": \"%s\",\n  \"base\": \"%s\",\n  \"mmproj\": \"%s\",\n  \"source\": { \"format\": \"gguf\", \"mmproj\": \"%s\" }\n}\n",
            mmb, arch, mod, base_path, mmproj_path, mmb);
    fclose(o);
    return 0;
}
/* ── AGGIUNGERE modulo dichiarato (.gmmod): dichiarata nell'ABI + risolta + eseguita ── */
static char* lj_slurp(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char* b=(char*)malloc((size_t)n+1); if(!b){fclose(f);return NULL;}
    if (fread(b,1,(size_t)n,f)!=(size_t)n){free(b);fclose(f);return NULL;} b[n]=0; fclose(f); return b;
}
static const char* lj_v(const char* j, const char* key) {
    if (!j) return NULL;
    char pat[96]; snprintf(pat,sizeof pat,"\"%s\"",key);
    const char* p=j;
    while ((p=strstr(p,pat))) { const char* c=p+strlen(pat); while(*c==' '||*c=='\t')c++; if(*c==':'){c++;while(*c==' '||*c=='\t')c++;return c;} p++; }
    return NULL;
}
static void lj_s(const char* p, char* out, size_t cap) {
    if (cap) out[0]=0;
    if (!p) return;
    while (*p==' '||*p=='\t') p++;
    if (*p!='"') return; p++;
    size_t i=0; while (*p && *p!='"') { if (i<cap-1) out[i++]=*p; p++; } out[i]=0;
}
/* il tensore `name` appartiene alla famiglia della modalita' dichiarata? */
/* L'arch DICHIARA la modalita' `mod`? (per-famiglia: MTP/VISION/AUDIO/OCR coesistono). */
static int llm_desc_has_modality(const LlmDesc* a, const char* mod) {
    if (a == NULL || mod == NULL || !*mod) return 0;
    if (strcasecmp(mod, "text") == 0 || strcasecmp(mod, "adapter") == 0 || strcasecmp(mod, "module") == 0) return 1;
    if (strcasecmp(mod, "mtp") == 0) {
        if (a->mm_mtp) return 1;
        for (int i = 0; i < a->n_roles; i++) if (strcmp(a->roles[i].role, "MTP") == 0) return 1;
        return 0;
    }
    if (strcasecmp(mod, "audio") == 0) {
        for (int i = 0; i < a->n_roles; i++) if (strncmp(a->roles[i].role, "AUDIO", 5) == 0) return 1;
        return llmdesc_struct(a, "audio") != NULL;
    }
    if (strcasecmp(mod, "vision") == 0 || strcasecmp(mod, "ocr") == 0) {
        for (int i = 0; i < a->n_roles; i++) { const char* r = a->roles[i].role; if (strcmp(r, "VISION") == 0 || strcmp(r, "VISION_ALIGNER") == 0) return 1; }
        return llmdesc_struct(a, "vision") != NULL;
    }
    if (strcasecmp(mod, "omni") == 0) return llm_desc_has_modality(a, "vision") && llm_desc_has_modality(a, "audio");
    return 1;
}
static int lj_mod_role(const LlmDesc* a, const char* name, const char* modality) {
    const char* role = llmdesc_role(a, name);
    int vis = (strcmp(role,"VISION")==0 || strcmp(role,"VISION_ALIGNER")==0 || strstr(name,"vision") || strstr(name,"mmproj") || strstr(name,"image_"));
    int aud = (strncmp(role,"AUDIO",5)==0 || strstr(name,"audio") || strstr(name,"whisper"));
    int mtp = (strcmp(role,"MTP")==0 || strncmp(name,"mtp.",4)==0);
    if (strcasecmp(modality,"vision")==0 || strcasecmp(modality,"ocr")==0) return vis;
    if (strcasecmp(modality,"audio")==0) return aud;
    if (strcasecmp(modality,"mtp")==0) return mtp;
    if (strcasecmp(modality,"omni")==0) return vis || aud;
    return 1;   /* adapter|text|module: nessun vincolo di ruolo */
}
int llmmodelio_model_module(const char* base_path, const char* gmmod_path, const char* module_path, const char* out_path) {
    if (base_path==NULL || gmmod_path==NULL || module_path==NULL || out_path==NULL) return -1;
    char* j = lj_slurp(gmmod_path); if (j == NULL) return -1;
    char modality[32]; lj_s(lj_v(j,"modality"), modality, sizeof modality);
    char arch[96]; lj_s(lj_v(j,"arch"), arch, sizeof arch);
    char mmproj[256]; lj_s(lj_v(j,"mmproj"), mmproj, sizeof mmproj);
    free(j);
    if (!modality[0]) return -1;                 /* la modalita' DEVE essere dichiarata nel .gmmod */
    /* 1) RISOLVI l'ABI dell'arch (dal .gmmod o dal modulo gguf). */
    int ok = 0;
    if (arch[0]) {
        char abi[2048];
        if (llmdesc_abi_dir(abi, sizeof abi)) {
            char path[4096];
            snprintf(path,sizeof path,"%s/%s.llmarch", abi, arch);
            if (llmdesc_load(path,&g_arch)==0){ snprintf(g_arch_path,sizeof g_arch_path,"%s",path); g_arch_ok=1; ok=1; }
            if (!ok){ snprintf(path,sizeof path,"%s/%s.gmbarch", abi, arch); if (llmdesc_load(path,&g_arch)==0){ snprintf(g_arch_path,sizeof g_arch_path,"%s",path); g_arch_ok=1; ok=1; } }
        }
    }
    if (!ok) { if (llmmodelio_arch_resolve_gguf(module_path)!=0) return -1; }
    if (!g_arch_ok) return -1;
    /* 2) VALIDA: l'ABI dichiara la modalita'? (dichiarata nell'ABI) */
    if (!llm_desc_has_modality(&g_arch, modality)) return -1;   /* la modality dev'essere DICHIARATA dall'ABI */
    (void)llmdesc_modality;
    /* 3) VALIDA: il modulo porta almeno un tensore della famiglia dichiarata (risolta via ruoli ABI). */
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv=NULL; int nkv=0;
    if (llmcodec_gguf_read_meta(module_path, &kv, &nkv, &fi) != 0) return -1;
    int carry = 0;
    for (uint64_t i=0;i<fi.n_tensors;i++) if (lj_mod_role(&g_arch, fi.tensors[i].name, modality)) { carry=1; break; }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, nkv);
    if (!carry) return -1;
    (void)mmproj;
    /* 4) ESEGUI: fondi il modulo nella base. */
    return llmmodelio_model_add(base_path, module_path, out_path);
}
/* ── MODALITY OF: query della modality dichiarata (per consumatori: loader/scanner) ──
 * .gmmod -> legge {modality,arch,mmproj}; dir HF -> config.json + .llmarch; .gguf -> arch resolve. */
int llmmodelio_model_modality_of(const char* path, char* modality, size_t mcap, char* mmproj, size_t mmcap, char* arch, size_t acap) {
    if (path == NULL) return -1;
    if (modality && mcap) modality[0] = 0;
    if (mmproj && mmcap) mmproj[0] = 0;
    if (arch && acap) arch[0] = 0;
    size_t ln = strlen(path);
    if (ln > 6 && strcasecmp(path + ln - 6, ".gmmod") == 0) {
        char* j = lj_slurp(path); if (j == NULL) return -1;
        lj_s(lj_v(j,"modality"), modality, mcap);
        lj_s(lj_v(j,"arch"), arch, acap);
        lj_s(lj_v(j,"mmproj"), mmproj, mmcap);
        free(j);
        return (modality && modality[0]) ? 0 : -1;
    }
    int ok = 0;
    if (ln > 5 && strcasecmp(path + ln - 5, ".gguf") == 0) { if (llmmodelio_arch_resolve_gguf(path) == 0) ok = 1; }
    else { if (llmmodelio_arch_resolve(path) == 0) ok = 1; }
    if (!ok || !g_arch_ok) return -1;
    if (modality && mcap) snprintf(modality, mcap, "%s", llmdesc_modality(&g_arch));
    if (mmproj && mmcap) snprintf(mmproj, mmcap, "%s", llmdesc_source_mmproj(&g_arch));
    if (arch && acap) snprintf(arch, acap, "%s", g_arch.arch);
    return 0;
}
/* Ritorna la modality dichiarata come stringa STATICA (buffer interno) — comodo per P/Invoke.
 * "" se non risolta. Non liberare: e' un buffer statico interno. */
const char* llmmodelio_model_modality_str(const char* path) {
    static char buf[32];
    char m[32];
    if (llmmodelio_model_modality_of(path, m, sizeof m, NULL, 0, NULL, 0) != 0) return "";
    snprintf(buf, sizeof buf, "%s", m);
    return buf;
}
int llmmodelio_model_convert(const char* src, const char* dst, const char* fmt)
{
    const char* s = src ? src : "";
    const char* d = dst ? dst : "";
    const char* f = fmt ? fmt : "";
    if (!*s || !*d) return -1;
    /* Conversione: SOLO GGUF/MLX via codec (llmcodec_conv_*_arch). */
    /* Nessun diagnostico su stderr: il messaggio d'errore viaggia col CODICE DI RITORNO (il
     * chiamante solleva/`llm_raise_error`). Un fprintf qui romperebbe la parità cross-backend
     * (i runner C fanno `2>&1`, i backend gestiti no) — verificato 2026-09-24. */
    { size_t l = strlen(s); int is_gguf = (l > 5 && strcasecmp(s + l - 5, ".gguf") == 0);
      if (is_gguf) { if (llmmodelio_arch_resolve_gguf(s) != 0) return -1; }
      else if (llmmodelio_arch_resolve(s) != 0) return -1; }
    const char* ap = llmmodelio_arch_path();
    if (strcasecmp(f, "GGUF") == 0) {
        /* Tokenizer: unica sorgente = progetto tokenizer (llmtoken). Il codec
         * riceve i dati digeriti (LlmCodecTokenizerMeta) e NON ri-parsa tokenizer.json. */
        LlmCodecTokenizerMeta tm; memset(&tm, 0, sizeof tm);
        char tp[1300]; snprintf(tp, sizeof tp, "%s/tokenizer.json", s);
        llmtoken_handle th = llmtoken_load(tp);
        if (th) {
            size_t ntok = 0, nmg = 0;
            tm.tokens = llmtoken_export_tokens(th, &ntok); tm.n_tokens = (int)ntok;
            tm.merges = llmtoken_export_merges(th, &nmg); tm.n_merges = (int)nmg;
            tm.model  = llmtoken_gguf_model(th);
        }
        int rc = llmcodec_conv_mlx_to_gguf_arch_ex(s, d, ap, LLMCODEC_GGML_T_MXFP4 /*default: affine compat*/, 32, 4, th ? &tm : NULL);
        if (th) llmtoken_free(th);   /* vive fino a qui: gli array sono "borrowed" */
        return rc;
    }
    if (strcasecmp(f, "MLX") == 0)  return llmcodec_conv_gguf_to_mlx_arch(s, d, ap, 32, 4);
    return -1;
}
