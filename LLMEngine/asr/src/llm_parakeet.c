/* llm_parakeet.c — Parakeet-TDT (FastConformer + transducer con durate) da GGUF. Vedi llm_parakeet.h.
 *
 * Pipeline (stessa del riferimento NeMo, in inferenza):
 *   1. mel: pre-enfasi 0.97, STFT (n_fft 512, finestra 400 dal GGUF, hop 160, centrata con riflessione), potenza,
 *      banco di filtri dal GGUF, log(x + guard), normalizzazione per-feature sul tempo;
 *   2. sottocampionamento convoluzionale x8 (conv 3x3 s2 + 2 coppie depthwise/pointwise) e proiezione a d_model;
 *   3. 24 blocchi Conformer: 0.5*FFN, attenzione a posizione relativa (Transformer-XL), convoluzione (GLU, depthwise k=9,
 *      batch-norm, SiLU), 0.5*FFN, layer-norm finale;
 *   4. decodifica greedy TDT: rete di predizione LSTM a 2 strati + giunzione; il passo sceglie simbolo e durata. */
#include "llm_parakeet.h"
#include "llmcodec_gguf_read.h"
#include "llmcodec_ggml_quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#endif

#define PK_MAXDUR 8
#define PK_NFFT 512
#define PK_NBIN (PK_NFFT / 2 + 1)
#define PK_SEG_SAMPLES (240 * 16000)

typedef struct {
    const float *ff1_ln_w, *ff1_ln_b, *ff1_l1, *ff1_l2;
    const float *att_ln_w, *att_ln_b, *pos_u, *pos_v, *wq, *wk, *wv, *wo, *wpos;
    const float *cv_ln_w, *cv_ln_b, *pw1, *dw, *dw_b, *bn_w, *bn_b, *bn_m, *bn_v, *pw2;
    const float *ff2_ln_w, *ff2_ln_b, *ff2_l1, *ff2_l2;
    const float *out_ln_w, *out_ln_b;
} PkLayer;

struct LlmParakeet {
    uint8_t* map;
    size_t map_len;
    LlmCodecGgufFileInfo fi;
    LlmCodecGgufMetaKVOwned* kv;
    int nkv;
    float** owned;
    int n_owned;
    int d, n_layers, n_heads, ff, conv_k, feat_in, sub_ch, pred_h, pred_layers, joint_h, vocab, blank, n_dur, max_symbols;
    int durs[PK_MAXDUR];
    float preemph, log_guard;
    int win, hop, n_mels;
    const float *window, *fb;
    const float *sub_conv_w[7], *sub_conv_b[7];
    const float *sub_out_w, *sub_out_b;
    PkLayer* layers;
    const float *emb, *lstm_wih[2], *lstm_whh[2], *lstm_bih[2], *lstm_bhh[2];
    const float *jpred_w, *jpred_b, *jenc_w, *jenc_b, *jout_w, *jout_b;
    char** pieces;
    int n_pieces;
    float cs[PK_NFFT / 2], sn[PK_NFFT / 2];   /* twiddle della FFT */
};

static void seterr(char* err, size_t cap, const char* msg) {
    if (err && cap) snprintf(err, cap, "%s", msg);
}

/* ---------------------------------------------------------------------------------------------------------- */
/* matrici */

/* C[M,N] = A[M,K] * B[N,K]^T (+ beta*C) */
static void gemm_nt(float* C, const float* A, const float* B, int M, int N, int K, int lda, int ldb, int ldc, float beta) {
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, lda, B, ldb, beta, C, ldc);
#else
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[(size_t)i * lda + k] * B[(size_t)j * ldb + k];
            C[(size_t)i * ldc + j] = s + (beta != 0.f ? beta * C[(size_t)i * ldc + j] : 0.f);
        }
#endif
}

/* C[M,N] = A[M,K] * B[K,N] */
static void gemm_nn(float* C, const float* A, const float* B, int M, int N, int K, int lda, int ldb, int ldc) {
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
#else
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[(size_t)i * lda + k] * B[(size_t)k * ldb + j];
            C[(size_t)i * ldc + j] = s;
        }
#endif
}

/* y[T,out] = x[T,in] * W[out,in]^T (+ bias[out]) */
static void linear(float* y, const float* x, const float* W, const float* bias, int T, int in, int out) {
    gemm_nt(y, x, W, T, out, in, in, in, out, 0.f);
    if (bias)
        for (int t = 0; t < T; t++)
            for (int o = 0; o < out; o++) y[(size_t)t * out + o] += bias[o];
}

static void layer_norm(float* y, const float* x, const float* w, const float* b, int T, int d) {
    for (int t = 0; t < T; t++) {
        const float* xr = x + (size_t)t * d;
        float* yr = y + (size_t)t * d;
        double mu = 0, var = 0;
        for (int i = 0; i < d; i++) mu += xr[i];
        mu /= d;
        for (int i = 0; i < d; i++) { double c = xr[i] - mu; var += c * c; }
        var /= d;
        float inv = (float)(1.0 / sqrt(var + 1e-5));
        for (int i = 0; i < d; i++) yr[i] = ((float)(xr[i] - mu)) * inv * w[i] + b[i];
    }
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }
static inline float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

/* ---------------------------------------------------------------------------------------------------------- */
/* caricamento */

static const LlmCodecGgufTensorInfo* find_tensor(const LlmParakeet* m, const char* name) {
    for (uint64_t i = 0; i < m->fi.n_tensors; i++)
        if (strcmp(m->fi.tensors[i].name, name) == 0) return &m->fi.tensors[i];
    return NULL;
}

/* Puntatore f32 al tensore: f32 resta nel mmap, gli altri tipi si dequantizzano in un buffer posseduto. */
static const float* tget(LlmParakeet* m, const char* name, int required, char* err, size_t errcap) {
    const LlmCodecGgufTensorInfo* t = find_tensor(m, name);
    if (!t) {
        if (required) { char b[300]; snprintf(b, sizeof b, "missing tensor %s", name); seterr(err, errcap, b); }
        return NULL;
    }
    const uint8_t* src = m->map + m->fi.data_start + t->offset;
    if (t->ggml_type == 0) return (const float*)src;
    int64_t numel = 1;
    for (int i = 0; i < t->ndim; i++) numel *= (int64_t)t->dims[i];
    float* dst = (float*)malloc((size_t)numel * sizeof(float));
    if (!dst) { seterr(err, errcap, "out of memory"); return NULL; }
    if (llmcodec_ggml_dequant_row(t->ggml_type, src, dst, numel) != 0) {
        free(dst);
        char b[300]; snprintf(b, sizeof b, "cannot dequantize %s (type %d)", name, t->ggml_type);
        seterr(err, errcap, b);
        return NULL;
    }
    m->owned = (float**)realloc(m->owned, (size_t)(m->n_owned + 1) * sizeof(float*));
    m->owned[m->n_owned++] = dst;
    return dst;
}

static int kv_u32(const LlmParakeet* m, const char* key, int def) {
    for (int i = 0; i < m->nkv; i++)
        if (strcmp(m->kv[i].key, key) == 0) return (int)m->kv[i].u32;
    return def;
}
static float kv_f32(const LlmParakeet* m, const char* key, float def) {
    for (int i = 0; i < m->nkv; i++)
        if (strcmp(m->kv[i].key, key) == 0) return m->kv[i].f32;
    return def;
}

int llm_parakeet_is_gguf(const char* path) {
    LlmCodecGgufMetaKVOwned* kv = NULL;
    int n = 0, ok = 0;
    LlmCodecGgufFileInfo fi;
    memset(&fi, 0, sizeof fi);
    if (!path || llmcodec_gguf_read_meta(path, &kv, &n, &fi) != 0) return 0;
    for (int i = 0; i < n; i++)
        if (strcmp(kv[i].key, "general.architecture") == 0 && kv[i].str && strcmp(kv[i].str, "parakeet") == 0) ok = 1;
    llmcodec_gguf_meta_free(kv, n);
    llmcodec_gguf_free_info(&fi);
    return ok;
}

void llm_parakeet_free(LlmParakeet* m) {
    if (!m) return;
    for (int i = 0; i < m->n_owned; i++) free(m->owned[i]);
    free(m->owned);
    free(m->layers);
    if (m->pieces) { for (int i = 0; i < m->n_pieces; i++) free(m->pieces[i]); free(m->pieces); }
    if (m->kv) llmcodec_gguf_meta_free(m->kv, m->nkv);
    if (m->fi.tensors) llmcodec_gguf_free_info(&m->fi);
    if (m->map) munmap(m->map, m->map_len);
    free(m);
}

LlmParakeet* llm_parakeet_load(const char* path, char* err, size_t errcap) {
    LlmParakeet* m = (LlmParakeet*)calloc(1, sizeof *m);
    if (!m) { seterr(err, errcap, "out of memory"); return NULL; }
    if (llmcodec_gguf_read_meta(path, &m->kv, &m->nkv, &m->fi) != 0) {
        seterr(err, errcap, "cannot read the GGUF file");
        free(m);
        return NULL;
    }
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) { if (fd >= 0) close(fd); seterr(err, errcap, "cannot open the file"); llm_parakeet_free(m); return NULL; }
    m->map_len = (size_t)st.st_size;
    m->map = (uint8_t*)mmap(NULL, m->map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->map == MAP_FAILED) { m->map = NULL; seterr(err, errcap, "cannot map the file"); llm_parakeet_free(m); return NULL; }

    int is_pk = 0;
    for (int i = 0; i < m->nkv; i++) {
        if (strcmp(m->kv[i].key, "general.architecture") == 0 && m->kv[i].str && !strcmp(m->kv[i].str, "parakeet")) is_pk = 1;
        if (strcmp(m->kv[i].key, "parakeet.tokenizer.pieces") == 0 && m->kv[i].arr) {
            m->n_pieces = m->kv[i].arr_n;
            m->pieces = (char**)calloc((size_t)m->n_pieces, sizeof(char*));
            for (int k = 0; k < m->n_pieces; k++) m->pieces[k] = strdup(m->kv[i].arr[k] ? m->kv[i].arr[k] : "");
        }
        if (strcmp(m->kv[i].key, "parakeet.tdt.durations") == 0 && m->kv[i].arr_raw) {
            m->n_dur = (int)m->kv[i].arr_count > PK_MAXDUR ? PK_MAXDUR : (int)m->kv[i].arr_count;
            for (int k = 0; k < m->n_dur; k++) m->durs[k] = ((const int32_t*)m->kv[i].arr_raw)[k];
        }
    }
    if (!is_pk) { seterr(err, errcap, "not a parakeet GGUF (general.architecture)"); llm_parakeet_free(m); return NULL; }
    if (!m->pieces || m->n_dur == 0) { seterr(err, errcap, "the GGUF lacks the vocabulary or the TDT durations"); llm_parakeet_free(m); return NULL; }

    m->d = kv_u32(m, "parakeet.encoder.d_model", 1024);
    m->n_layers = kv_u32(m, "parakeet.encoder.n_layers", 24);
    m->n_heads = kv_u32(m, "parakeet.encoder.n_heads", 8);
    m->ff = kv_u32(m, "parakeet.encoder.ff_dim", 4096);
    m->conv_k = kv_u32(m, "parakeet.encoder.conv_kernel", 9);
    m->feat_in = kv_u32(m, "parakeet.encoder.feat_in", 128);
    m->sub_ch = kv_u32(m, "parakeet.encoder.subsampling_conv_channels", 256);
    m->pred_h = kv_u32(m, "parakeet.decoder.pred_hidden", 640);
    m->pred_layers = kv_u32(m, "parakeet.decoder.pred_rnn_layers", 2);
    m->joint_h = kv_u32(m, "parakeet.joint.joint_hidden", 640);
    m->vocab = kv_u32(m, "parakeet.vocab_size", 8192);
    m->blank = kv_u32(m, "parakeet.blank_id", m->vocab);
    m->max_symbols = kv_u32(m, "parakeet.decoding.max_symbols", 10);
    m->win = kv_u32(m, "parakeet.preprocessor.win_length", 400);
    m->hop = kv_u32(m, "parakeet.preprocessor.hop_length", 160);
    m->n_mels = kv_u32(m, "parakeet.preprocessor.n_mels", 128);
    m->preemph = kv_f32(m, "parakeet.preprocessor.preemph", 0.97f);
    m->log_guard = kv_f32(m, "parakeet.preprocessor.log_zero_guard", 5.96046e-8f);
    if (kv_u32(m, "parakeet.preprocessor.n_fft", PK_NFFT) != PK_NFFT || m->pred_layers != 2 || m->d % m->n_heads != 0 ||
        m->n_layers <= 0 || m->n_layers > 64 || m->sub_ch != 256) {
        seterr(err, errcap, "unsupported parakeet configuration");
        llm_parakeet_free(m);
        return NULL;
    }

    for (int i = 0; i < PK_NFFT / 2; i++) {
        double a = -2.0 * M_PI * i / PK_NFFT;
        m->cs[i] = (float)cos(a);
        m->sn[i] = (float)sin(a);
    }

#define NEED(dst, name) do { (dst) = tget(m, (name), 1, err, errcap); if (!(dst)) { llm_parakeet_free(m); return NULL; } } while (0)
    char nm[200];
    NEED(m->window, "preprocessor.featurizer.window");
    NEED(m->fb, "preprocessor.featurizer.fb");
    static const int conv_ids[5] = { 0, 2, 3, 5, 6 };
    for (int i = 0; i < 5; i++) {
        snprintf(nm, sizeof nm, "encoder.pre_encode.conv.%d.weight", conv_ids[i]);
        NEED(m->sub_conv_w[i], nm);
        snprintf(nm, sizeof nm, "encoder.pre_encode.conv.%d.bias", conv_ids[i]);
        NEED(m->sub_conv_b[i], nm);
    }
    NEED(m->sub_out_w, "encoder.pre_encode.out.weight");
    NEED(m->sub_out_b, "encoder.pre_encode.out.bias");
    m->layers = (PkLayer*)calloc((size_t)m->n_layers, sizeof(PkLayer));
    for (int l = 0; l < m->n_layers; l++) {
        PkLayer* L = &m->layers[l];
#define LN(dst, suffix) do { snprintf(nm, sizeof nm, "encoder.layers.%d.%s", l, suffix); NEED(dst, nm); } while (0)
        LN(L->ff1_ln_w, "norm_feed_forward1.weight"); LN(L->ff1_ln_b, "norm_feed_forward1.bias");
        LN(L->ff1_l1, "feed_forward1.linear1.weight"); LN(L->ff1_l2, "feed_forward1.linear2.weight");
        LN(L->att_ln_w, "norm_self_att.weight"); LN(L->att_ln_b, "norm_self_att.bias");
        LN(L->pos_u, "self_attn.pos_bias_u"); LN(L->pos_v, "self_attn.pos_bias_v");
        LN(L->wq, "self_attn.linear_q.weight"); LN(L->wk, "self_attn.linear_k.weight");
        LN(L->wv, "self_attn.linear_v.weight"); LN(L->wo, "self_attn.linear_out.weight");
        LN(L->wpos, "self_attn.linear_pos.weight");
        LN(L->cv_ln_w, "norm_conv.weight"); LN(L->cv_ln_b, "norm_conv.bias");
        LN(L->pw1, "conv.pointwise_conv1.weight"); LN(L->dw, "conv.depthwise_conv.weight");
        LN(L->bn_w, "conv.batch_norm.weight"); LN(L->bn_b, "conv.batch_norm.bias");
        LN(L->bn_m, "conv.batch_norm.running_mean"); LN(L->bn_v, "conv.batch_norm.running_var");
        LN(L->pw2, "conv.pointwise_conv2.weight");
        LN(L->ff2_ln_w, "norm_feed_forward2.weight"); LN(L->ff2_ln_b, "norm_feed_forward2.bias");
        LN(L->ff2_l1, "feed_forward2.linear1.weight"); LN(L->ff2_l2, "feed_forward2.linear2.weight");
        LN(L->out_ln_w, "norm_out.weight"); LN(L->out_ln_b, "norm_out.bias");
        snprintf(nm, sizeof nm, "encoder.layers.%d.conv.depthwise_conv.bias", l);
        L->dw_b = tget(m, nm, 0, NULL, 0);   /* assente nei GGUF senza bias */
#undef LN
    }
    NEED(m->emb, "decoder.prediction.embed.weight");
    for (int l = 0; l < 2; l++) {
        snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.weight_ih_l%d", l); NEED(m->lstm_wih[l], nm);
        snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.weight_hh_l%d", l); NEED(m->lstm_whh[l], nm);
        snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.bias_ih_l%d", l); NEED(m->lstm_bih[l], nm);
        snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.bias_hh_l%d", l); NEED(m->lstm_bhh[l], nm);
    }
    NEED(m->jpred_w, "joint.pred.weight"); NEED(m->jpred_b, "joint.pred.bias");
    NEED(m->jenc_w, "joint.enc.weight"); NEED(m->jenc_b, "joint.enc.bias");
    NEED(m->jout_w, "joint.joint_net.2.weight"); NEED(m->jout_b, "joint.joint_net.2.bias");
#undef NEED
    return m;
}

/* ---------------------------------------------------------------------------------------------------------- */
/* 1. mel */

/* FFT radix-2 complessa in-place (n = 512). */
static void fft512(const LlmParakeet* m, float* re, float* im) {
    const int n = PK_NFFT;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < half; k++) {
                float wr = m->cs[k * step], wi = m->sn[k * step];
                float xr = re[i + k + half] * wr - im[i + k + half] * wi;
                float xi = re[i + k + half] * wi + im[i + k + half] * wr;
                re[i + k + half] = re[i + k] - xr; im[i + k + half] = im[i + k] - xi;
                re[i + k] += xr; im[i + k] += xi;
            }
    }
}

/* Ritorna [T][n_mels] normalizzato (malloc); *out_T = n/hop + 1. */
static float* mel_features(const LlmParakeet* m, const float* x, int n, int* out_T) {
    const int hop = m->hop, pad = PK_NFFT / 2, nm = m->n_mels;
    int T = n / hop + 1;
    float* pe = (float*)malloc(sizeof(float) * (size_t)n);
    float* padded = (float*)calloc((size_t)n + 2 * pad, sizeof(float));
    float* pw = (float*)malloc(sizeof(float) * (size_t)T * PK_NBIN);
    float* mel = (float*)malloc(sizeof(float) * (size_t)T * nm);
    if (!pe || !padded || !pw || !mel) { free(pe); free(padded); free(pw); free(mel); return NULL; }
    pe[0] = x[0];
    for (int i = 1; i < n; i++) pe[i] = x[i] - m->preemph * x[i - 1];
    for (int i = 0; i < n; i++) padded[pad + i] = pe[i];
    for (int i = 0; i < pad; i++) {                      /* riflessione (senza ripetere il bordo) */
        padded[pad - 1 - i] = pe[(i + 1) < n ? i + 1 : n - 1];
        padded[pad + n + i] = pe[(n - 2 - i) >= 0 ? n - 2 - i : 0];
    }
    const int wl = (PK_NFFT - m->win) / 2;
    float re[PK_NFFT], im[PK_NFFT];
    for (int t = 0; t < T; t++) {
        const float* fr = padded + (size_t)t * hop;
        memset(re, 0, sizeof re); memset(im, 0, sizeof im);
        for (int i = 0; i < m->win; i++) re[wl + i] = fr[wl + i] * m->window[i];
        fft512(m, re, im);
        for (int k = 0; k < PK_NBIN; k++) pw[(size_t)t * PK_NBIN + k] = re[k] * re[k] + im[k] * im[k];
    }
    /* mel[t][j] = sum_k fb[j][k] * power[t][k] */
    gemm_nt(mel, pw, m->fb, T, nm, PK_NBIN, PK_NBIN, PK_NBIN, nm, 0.f);
    for (size_t i = 0; i < (size_t)T * nm; i++) mel[i] = logf(mel[i] + m->log_guard);
    for (int j = 0; j < nm; j++) {                       /* per-feature: media e dev. std (n-1) sul tempo */
        double mu = 0, var = 0;
        for (int t = 0; t < T; t++) mu += mel[(size_t)t * nm + j];
        mu /= T;
        for (int t = 0; t < T; t++) { double c = mel[(size_t)t * nm + j] - mu; var += c * c; }
        double sd = T > 1 ? sqrt(var / (T - 1)) : 0.0;
        for (int t = 0; t < T; t++) mel[(size_t)t * nm + j] = (float)((mel[(size_t)t * nm + j] - mu) / (sd + 1e-5));
    }
    free(pe); free(padded); free(pw);
    *out_T = T;
    return mel;
}

/* ---------------------------------------------------------------------------------------------------------- */
/* 2. sottocampionamento */

static int conv_out(int L) { return (L + 2 - 3) / 2 + 1; }

/* conv 3x3 stride 2 pad 1 su [H][W]; w[3][3] */
static void conv_s2(float* out, const float* in, const float* w, float b, int H, int W, int Ho, int Wo) {
    for (int ho = 0; ho < Ho; ho++)
        for (int wo = 0; wo < Wo; wo++) {
            float s = b;
            for (int kh = 0; kh < 3; kh++) {
                int h = ho * 2 + kh - 1;
                if (h < 0 || h >= H) continue;
                for (int kw = 0; kw < 3; kw++) {
                    int ww = wo * 2 + kw - 1;
                    if (ww < 0 || ww >= W) continue;
                    s += in[(size_t)h * W + ww] * w[kh * 3 + kw];
                }
            }
            out[(size_t)ho * Wo + wo] = s;
        }
}

/* feats [T][128] -> [T3][d] (malloc), *T3 */
static float* pre_encode(const LlmParakeet* m, const float* feats, int T, int* out_T3) {
    const int C = m->sub_ch;
    int H0 = T, W0 = m->feat_in;
    int H1 = conv_out(H0), W1 = conv_out(W0), H2 = conv_out(H1), W2 = conv_out(W1), H3 = conv_out(H2), W3 = conv_out(W2);
    float* a = (float*)malloc(sizeof(float) * (size_t)C * H1 * W1);
    float* b = (float*)malloc(sizeof(float) * (size_t)C * H2 * W2);
    float* c = (float*)malloc(sizeof(float) * (size_t)C * H2 * W2);
    float* d = (float*)malloc(sizeof(float) * (size_t)C * H3 * W3);
    float* e = (float*)malloc(sizeof(float) * (size_t)C * H3 * W3);
    float* flat = (float*)malloc(sizeof(float) * (size_t)H3 * C * W3);
    float* out = (float*)malloc(sizeof(float) * (size_t)H3 * m->d);
    if (!a || !b || !c || !d || !e || !flat || !out) { free(a); free(b); free(c); free(d); free(e); free(flat); free(out); return NULL; }
    /* conv.0: 1 -> C, ReLU */
    for (int co = 0; co < C; co++) {
        float* o = a + (size_t)co * H1 * W1;
        conv_s2(o, feats, m->sub_conv_w[0] + (size_t)co * 9, m->sub_conv_b[0][co], H0, W0, H1, W1);
        for (int i = 0; i < H1 * W1; i++) if (o[i] < 0) o[i] = 0;
    }
    /* conv.2 depthwise + conv.3 pointwise + ReLU */
    for (int ch = 0; ch < C; ch++)
        conv_s2(b + (size_t)ch * H2 * W2, a + (size_t)ch * H1 * W1, m->sub_conv_w[1] + (size_t)ch * 9, m->sub_conv_b[1][ch], H1, W1, H2, W2);
    gemm_nn(c, m->sub_conv_w[2], b, C, H2 * W2, C, C, H2 * W2, H2 * W2);
    for (int co = 0; co < C; co++)
        for (int i = 0; i < H2 * W2; i++) {
            float v = c[(size_t)co * H2 * W2 + i] + m->sub_conv_b[2][co];
            c[(size_t)co * H2 * W2 + i] = v < 0 ? 0 : v;
        }
    /* conv.5 depthwise + conv.6 pointwise + ReLU */
    for (int ch = 0; ch < C; ch++)
        conv_s2(d + (size_t)ch * H3 * W3, c + (size_t)ch * H2 * W2, m->sub_conv_w[3] + (size_t)ch * 9, m->sub_conv_b[3][ch], H2, W2, H3, W3);
    gemm_nn(e, m->sub_conv_w[4], d, C, H3 * W3, C, C, H3 * W3, H3 * W3);
    for (int co = 0; co < C; co++)
        for (int i = 0; i < H3 * W3; i++) {
            float v = e[(size_t)co * H3 * W3 + i] + m->sub_conv_b[4][co];
            e[(size_t)co * H3 * W3 + i] = v < 0 ? 0 : v;
        }
    /* [C][H3][W3] -> [H3][C*W3] (indice c*W3+f) */
    for (int t = 0; t < H3; t++)
        for (int ch = 0; ch < C; ch++)
            for (int f = 0; f < W3; f++) flat[(size_t)t * C * W3 + (size_t)ch * W3 + f] = e[((size_t)ch * H3 + t) * W3 + f];
    linear(out, flat, m->sub_out_w, m->sub_out_b, H3, C * W3, m->d);
    free(a); free(b); free(c); free(d); free(e); free(flat);
    *out_T3 = H3;
    return out;
}

/* ---------------------------------------------------------------------------------------------------------- */
/* 3. blocchi Conformer */

static int conformer_layer(const LlmParakeet* m, const PkLayer* L, float* x, int T, const float* pos_emb) {
    const int d = m->d, H = m->n_heads, dk = d / H, ffd = m->ff, P = 2 * T - 1;
    float* ln = (float*)malloc(sizeof(float) * (size_t)T * d);
    float* hid = (float*)malloc(sizeof(float) * (size_t)T * (size_t)(ffd > 2 * d ? ffd : 2 * d));
    float* tmp = (float*)malloc(sizeof(float) * (size_t)T * d);
    if (!ln || !hid || !tmp) { free(ln); free(hid); free(tmp); return -1; }

    /* FFN 1 */
    layer_norm(ln, x, L->ff1_ln_w, L->ff1_ln_b, T, d);
    linear(hid, ln, L->ff1_l1, NULL, T, d, ffd);
    for (size_t i = 0; i < (size_t)T * ffd; i++) hid[i] = silu(hid[i]);
    linear(tmp, hid, L->ff1_l2, NULL, T, ffd, d);
    for (size_t i = 0; i < (size_t)T * d; i++) x[i] += 0.5f * tmp[i];

    /* attenzione a posizione relativa */
    {
        float* q = (float*)malloc(sizeof(float) * (size_t)T * d);
        float* k = (float*)malloc(sizeof(float) * (size_t)T * d);
        float* v = (float*)malloc(sizeof(float) * (size_t)T * d);
        float* ctx = (float*)malloc(sizeof(float) * (size_t)T * d);
        float* p = (float*)malloc(sizeof(float) * (size_t)P * d);
        float* qu = (float*)malloc(sizeof(float) * (size_t)T * dk);
        float* qv = (float*)malloc(sizeof(float) * (size_t)T * dk);
        float* ac = (float*)malloc(sizeof(float) * (size_t)T * T);
        float* bd = (float*)malloc(sizeof(float) * (size_t)T * P);
        if (!q || !k || !v || !ctx || !p || !qu || !qv || !ac || !bd) {
            free(q); free(k); free(v); free(ctx); free(p); free(qu); free(qv); free(ac); free(bd); free(ln); free(hid); free(tmp);
            return -1;
        }
        layer_norm(ln, x, L->att_ln_w, L->att_ln_b, T, d);
        linear(q, ln, L->wq, NULL, T, d, d);
        linear(k, ln, L->wk, NULL, T, d, d);
        linear(v, ln, L->wv, NULL, T, d, d);
        linear(p, pos_emb, L->wpos, NULL, P, d, d);
        const float scale = 1.0f / sqrtf((float)dk);
        for (int h = 0; h < H; h++) {
            for (int t = 0; t < T; t++)
                for (int c = 0; c < dk; c++) {
                    float qq = q[(size_t)t * d + (size_t)h * dk + c];
                    qu[(size_t)t * dk + c] = qq + L->pos_u[h * dk + c];
                    qv[(size_t)t * dk + c] = qq + L->pos_v[h * dk + c];
                }
            gemm_nt(ac, qu, k + (size_t)h * dk, T, T, dk, dk, d, T, 0.f);
            gemm_nt(bd, qv, p + (size_t)h * dk, T, P, dk, dk, d, P, 0.f);
            for (int i = 0; i < T; i++) {
                float* row = ac + (size_t)i * T;
                const float* brow = bd + (size_t)i * P;
                float mx = -1e30f;
                for (int j = 0; j < T; j++) {
                    row[j] = (row[j] + brow[T - 1 - i + j]) * scale;
                    if (row[j] > mx) mx = row[j];
                }
                float sum = 0;
                for (int j = 0; j < T; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
                float inv = 1.0f / sum;
                for (int j = 0; j < T; j++) row[j] *= inv;
            }
            gemm_nn(ctx + (size_t)h * dk, ac, v + (size_t)h * dk, T, dk, T, T, d, d);
        }
        linear(tmp, ctx, L->wo, NULL, T, d, d);
        for (size_t i = 0; i < (size_t)T * d; i++) x[i] += tmp[i];
        free(q); free(k); free(v); free(ctx); free(p); free(qu); free(qv); free(ac); free(bd);
    }

    /* modulo di convoluzione */
    layer_norm(ln, x, L->cv_ln_w, L->cv_ln_b, T, d);
    linear(hid, ln, L->pw1, NULL, T, d, 2 * d);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < d; c++) hid[(size_t)t * 2 * d + c] = hid[(size_t)t * 2 * d + c] * sigm(hid[(size_t)t * 2 * d + d + c]);   /* GLU */
    {
        const int K = m->conv_k, pad = K / 2;
        for (int c = 0; c < d; c++) {
            const float* w = L->dw + (size_t)c * K;
            float scale = L->bn_w[c] / sqrtf(L->bn_v[c] + 1e-5f);
            float shift = L->bn_b[c] - L->bn_m[c] * scale;
            for (int t = 0; t < T; t++) {
                float s = L->dw_b ? L->dw_b[c] : 0.f;
                for (int kk = 0; kk < K; kk++) {
                    int tt = t + kk - pad;
                    if (tt >= 0 && tt < T) s += hid[(size_t)tt * 2 * d + c] * w[kk];
                }
                ln[(size_t)t * d + c] = silu(s * scale + shift);   /* depthwise -> batch-norm (inferenza) -> SiLU */
            }
        }
    }
    linear(tmp, ln, L->pw2, NULL, T, d, d);
    for (size_t i = 0; i < (size_t)T * d; i++) x[i] += tmp[i];

    /* FFN 2 */
    layer_norm(ln, x, L->ff2_ln_w, L->ff2_ln_b, T, d);
    linear(hid, ln, L->ff2_l1, NULL, T, d, ffd);
    for (size_t i = 0; i < (size_t)T * ffd; i++) hid[i] = silu(hid[i]);
    linear(tmp, hid, L->ff2_l2, NULL, T, ffd, d);
    for (size_t i = 0; i < (size_t)T * d; i++) x[i] += 0.5f * tmp[i];

    layer_norm(tmp, x, L->out_ln_w, L->out_ln_b, T, d);
    memcpy(x, tmp, sizeof(float) * (size_t)T * d);
    free(ln); free(hid); free(tmp);
    return 0;
}

/* codifica posizionale relativa: riga r <-> posizione (T-1) - r */
static float* rel_pos_emb(int T, int d) {
    int P = 2 * T - 1;
    float* pe = (float*)malloc(sizeof(float) * (size_t)P * d);
    if (!pe) return NULL;
    for (int r = 0; r < P; r++) {
        double pos = (double)(T - 1 - r);
        for (int i = 0; i < d / 2; i++) {
            double div = exp(-(2.0 * i) * log(10000.0) / d);
            pe[(size_t)r * d + 2 * i] = (float)sin(pos * div);
            pe[(size_t)r * d + 2 * i + 1] = (float)cos(pos * div);
        }
    }
    return pe;
}

/* ---------------------------------------------------------------------------------------------------------- */
/* 4. decodifica TDT */

typedef struct { float h[2][640], c[2][640]; } PkState;

static void lstm_step(const LlmParakeet* m, PkState* st, const float* in, float* out) {
    const int H = m->pred_h;
    float gates[4 * 640], x[640];
    memcpy(x, in, sizeof(float) * (size_t)H);
    for (int l = 0; l < 2; l++) {
        linear(gates, x, m->lstm_wih[l], m->lstm_bih[l], 1, H, 4 * H);
        float g2[4 * 640];
        linear(g2, st->h[l], m->lstm_whh[l], m->lstm_bhh[l], 1, H, 4 * H);
        for (int i = 0; i < 4 * H; i++) gates[i] += g2[i];
        for (int i = 0; i < H; i++) {
            float ig = sigm(gates[i]), fg = sigm(gates[H + i]), gg = tanhf(gates[2 * H + i]), og = sigm(gates[3 * H + i]);
            st->c[l][i] = fg * st->c[l][i] + ig * gg;
            st->h[l][i] = og * tanhf(st->c[l][i]);
        }
        memcpy(x, st->h[l], sizeof(float) * (size_t)H);
    }
    memcpy(out, x, sizeof(float) * (size_t)H);
}

static int append_piece(const LlmParakeet* m, int id, char** buf, size_t* len, size_t* cap) {
    if (id < 0 || id >= m->n_pieces) return 0;
    const char* p = m->pieces[id];
    if (!strncmp(p, "<|", 2) || !strcmp(p, "<unk>") || !strcmp(p, "<pad>")) return 0;
    size_t pl = strlen(p);
    if (*len + pl + 2 > *cap) { *cap = (*len + pl + 2) * 2; *buf = (char*)realloc(*buf, *cap); if (!*buf) return -1; }
    for (size_t i = 0; i < pl;) {
        if ((unsigned char)p[i] == 0xE2 && i + 2 < pl && (unsigned char)p[i + 1] == 0x96 && (unsigned char)p[i + 2] == 0x81) {
            (*buf)[(*len)++] = ' ';   /* U+2581 "▁" */
            i += 3;
        } else {
            (*buf)[(*len)++] = p[i++];
        }
    }
    (*buf)[*len] = '\0';
    return 0;
}

/* Greedy TDT (come NeMo GreedyTDTInfer): a ogni passo il giunto sceglie simbolo e durata; un blank avanza di almeno un
 * fotogramma, un simbolo avanza della durata scelta (anche 0: si resta sul fotogramma, fino a max_symbols). La rete di
 * predizione avanza solo quando si emette un simbolo. */
static char* decode_segment(const LlmParakeet* m, const float* enc, int Te) {
    const int J = m->joint_h, V1 = m->blank + 1, NO = V1 + m->n_dur, H = m->pred_h;
    float* encp = (float*)malloc(sizeof(float) * (size_t)Te * J);
    float* logits = (float*)malloc(sizeof(float) * (size_t)NO);
    size_t len = 0, cap = 256;
    char* text = (char*)malloc(cap);
    if (!encp || !logits || !text) { free(encp); free(logits); free(text); return NULL; }
    text[0] = '\0';
    linear(encp, enc, m->jenc_w, m->jenc_b, Te, m->d, J);

    PkState state, next;
    memset(&state, 0, sizeof state);
    float g[640], predp[640], hj[640], zero[640];
    memset(zero, 0, sizeof zero);
    int last = m->blank, have_pred = 0, t = 0;
    while (t < Te) {
        int symbols = 0, skip = 0;
        while (symbols < m->max_symbols) {
            if (!have_pred) {
                next = state;
                lstm_step(m, &next, last == m->blank ? zero : m->emb + (size_t)last * H, g);
                linear(predp, g, m->jpred_w, m->jpred_b, 1, H, J);
                have_pred = 1;
            }
            for (int i = 0; i < J; i++) { float s = encp[(size_t)t * J + i] + predp[i]; hj[i] = s > 0 ? s : 0; }
            linear(logits, hj, m->jout_w, m->jout_b, 1, J, NO);
            int tok = 0, di = 0;
            for (int i = 1; i < V1; i++) if (logits[i] > logits[tok]) tok = i;
            for (int i = 1; i < m->n_dur; i++) if (logits[V1 + i] > logits[V1 + di]) di = i;
            skip = m->durs[di];
            symbols++;
            if (tok == m->blank) {
                if (skip == 0) skip = 1;
                break;
            }
            if (append_piece(m, tok, &text, &len, &cap) != 0) { free(encp); free(logits); return NULL; }
            state = next;
            last = tok;
            have_pred = 0;
            if (skip > 0) break;
        }
        t += skip > 0 ? skip : 1;   /* skip == 0 dopo max_symbols: si forza l'avanzamento */
    }
    free(encp); free(logits);
    return text;
}

/* ---------------------------------------------------------------------------------------------------------- */

static char* transcribe_one(const LlmParakeet* m, const float* x, int n, char* err, size_t errcap) {
    if (n < 400) return strdup("");
    int T = 0, T3 = 0;
    float* mel = mel_features(m, x, n, &T);
    if (!mel) { seterr(err, errcap, "out of memory"); return NULL; }
    float* h = pre_encode(m, mel, T, &T3);
    free(mel);
    if (!h || T3 <= 0) { free(h); seterr(err, errcap, "audio too short or out of memory"); return NULL; }
    float* pe = rel_pos_emb(T3, m->d);
    if (!pe) { free(h); seterr(err, errcap, "out of memory"); return NULL; }
    for (int l = 0; l < m->n_layers; l++)
        if (conformer_layer(m, &m->layers[l], h, T3, pe) != 0) { free(h); free(pe); seterr(err, errcap, "out of memory"); return NULL; }
    free(pe);
    char* text = decode_segment(m, h, T3);
    free(h);
    if (!text) seterr(err, errcap, "decoding failed");
    return text;
}

/* punto di minima energia (finestra 100 ms) nell'ultimo tratto [end-10s, end] */
static int find_cut(const float* x, int start, int end) {
    int lo = end - 10 * 16000, win = 1600, best = end;
    double be = 1e30;
    if (lo < start + win) lo = start + win;
    for (int p = lo; p + win < end; p += 400) {
        double e = 0;
        for (int i = 0; i < win; i += 4) e += (double)x[p + i] * x[p + i];
        if (e < be) { be = e; best = p + win / 2; }
    }
    return best;
}

char* llm_parakeet_transcribe(LlmParakeet* m, const float* samples, int n, char* err, size_t errcap) {
    if (!m || !samples || n <= 0) { seterr(err, errcap, "no audio"); return NULL; }
    char* out = (char*)malloc(1);
    size_t len = 0;
    if (!out) return NULL;
    out[0] = '\0';
    int pos = 0;
    while (pos < n) {
        int end = (n - pos > PK_SEG_SAMPLES) ? find_cut(samples, pos, pos + PK_SEG_SAMPLES) : n;
        char* seg = transcribe_one(m, samples + pos, end - pos, err, errcap);
        if (!seg) { free(out); return NULL; }
        size_t sl = strlen(seg);
        out = (char*)realloc(out, len + sl + 2);
        if (len > 0 && sl > 0 && seg[0] != ' ') out[len++] = ' ';
        memcpy(out + len, seg, sl + 1);
        len += sl;
        free(seg);
        pos = end;
    }
    /* senza lo spazio iniziale del primo pezzo */
    size_t lead = 0;
    while (out[lead] == ' ') lead++;
    if (lead) memmove(out, out + lead, strlen(out + lead) + 1);
    return out;
}
