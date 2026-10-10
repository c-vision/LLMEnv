/* gmb_asr.h — front-end audio + kernel host per Qwen3-ASR (porting da antirez/qwen-asr, MIT).
 *
 * Layer HOST (CPU, senza dipendenze GPU): WAV/resample/log-mel, kernel numerici (matmul, conv2d,
 * norm, attivazioni, attenzione bidirezionale/causale-GQA, PE/rope), quantizzazione Q8 runtime,
 * segmentazione a confini di silenzio e bookkeeping dello streaming. Il FORWARD del modello
 * (encoder Qwen3-ASR + decoder Qwen3) si aggancia sopra questi kernel (slice successiva).
 *
 * Riferimento di parità: i valori del mel (16 kHz, 128 bin, hop 160, win 400, clamp dinamico,
 * normalizzazione (val+4)/4) sono portati VERBATIM da qwen-asr per fedeltà bit-per-bit.
 */
#ifndef GMB_ASR_H
#define GMB_ASR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Audio front-end ===== */
#define GMB_ASR_SR      16000
#define GMB_ASR_N_MEL   128
#define GMB_ASR_HOP     160
#define GMB_ASR_WIN     400
#define GMB_ASR_N_FFT   400
#define GMB_ASR_N_FREQ  (GMB_ASR_N_FFT / 2 + 1)

/* WAV (16-bit PCM, mono/stereo->mono) -> F32 [-1,1] @16kHz. `data` = buffer WAV.
 * Ritorna buffer malloc (free col chiamante) o NULL; *out_n = campioni. */
float* gmb_asr_wav_decode(const uint8_t* data, size_t size, int* out_n);
/* Da file. */
float* gmb_asr_wav_load(const char* path, int* out_n);
/* Resample windowed-sinc (Kaiser beta=6, sinc_half=16) da sr_in a 16kHz. */
float* gmb_asr_resample(const float* in, int n_in, int sr_in, int* out_n);
/* Log-mel [N_MEL, frames] (`frames` = out). `samples` mono F32 @16k. */
float* gmb_asr_mel(const float* samples, int n, int* out_frames);

/* ===== Kernel numerici (row-major) ===== */
void gmb_asr_matmul_t(float* C, const float* A, const float* B, int M, int K, int N);       /* C=A·Bᵀ */
void gmb_asr_linear(float* y, const float* x, const float* W, const float* b,
                    int seq, int in, int out);                                              /* y=x·Wᵀ(+b), f32 */
void gmb_asr_add_inplace(float* a, const float* b, int n);
void gmb_asr_linear_bf16(float* y, const float* x, const uint16_t* W, const float* b,
                         int seq, int in, int out);                                          /* y=x·Wᵀ(+b) */
void gmb_asr_conv2d(float* out, const float* in, const float* w, const float* b,
                    int cin, int cout, int hin, int win, int kh, int kw, int stride, int pad);
void gmb_asr_layer_norm(float* out, const float* x, const float* w, const float* b,
                        int seq, int hid, float eps);
void gmb_asr_rms_norm(float* out, const float* x, const float* w, int seq, int hid, float eps);
void gmb_asr_rms_norm_per_head(float* x, const float* w, int seq, int heads, int hd, float eps);
void gmb_asr_silu(float* x, int n);
void gmb_asr_gelu(float* x, int n);
void gmb_asr_softmax(float* x, int rows, int cols);
void gmb_asr_swiglu(float* out, const float* gate_up, int seq, int inter);
/* Attn bidirezionale a finestre (encoder). window_starts[n+1], sentinella=seq. */
void gmb_asr_attn_bidirectional(float* out, const float* Q, const float* K, const float* V,
                                int seq, int heads, int hd, float scale,
                                const int* window_starts, int n_windows);
/* Attn causale GQA (decoder). q_offset = posizione globale della prima query. */
void gmb_asr_attn_causal_gqa(float* out, const float* Q, const float* K, const float* V,
                             int sq, int sk, int heads, int kv_heads, int hd, float scale, int q_offset);
/* PE sinusoidale (encoder): metà sin, metà cos. pe[n_pos, d_model]. */
void gmb_asr_sinusoidal_pe(float* pe, int n_pos, int d_model);
/* NeoX RoPE: cos/sin [seq,hd] (duplicati metà) + apply in-place su x[seq, heads*hd]. */
void gmb_asr_rope_neox_prep(float* cos_out, float* sin_out, const int* pos, int seq, int hd, float theta);
void gmb_asr_rope_neox_apply(float* x, const float* cos_v, const float* sin_v, int seq, int heads, int hd);
/* argmax(W_bf16 · x) senza materializzare i logits. */
int gmb_asr_argmax_bf16(const float* x, const uint16_t* W, int in, int out);

/* ===== Q8 runtime (point 5) ===== */
#define GMB_ASR_Q8_BLK 32
/* Quantizza n valori in int8 a blocchi di 32 con UNA scala FP32/blocco. n multiplo di 32.
 * scales: n/32 valori. q[i] in [-127,127]. */
void gmb_asr_q8_quantize(const float* x, int8_t* q, float* scales, int n);
/* Dequantizza. x = q[i] * scales[i/32]. */
void gmb_asr_q8_dequantize(const int8_t* q, const float* scales, float* x, int n);

/* ===== Segmentazione a confini di silenzio (point 3) ===== */
/* Trova fino a `max_cuts` punti di taglio (~ogni `seg_samples`) cercando il minimo di energia
 * (RMS su finestra ~20ms) entro ±`search_samples`. Scrive le posizioni in `cuts` (offset in
 * campioni, crescenti), ritorna il numero di tagli (>=0). `cuts[0..n)` delimita n+1 segmenti. */
int gmb_asr_segment_split(const float* samples, int n, int seg_samples, int search_samples,
                          int* cuts, int max_cuts);

/* ===== Streaming bookkeeping (point 2) ===== */
/* Cache delle finestre encoder + eviction sliding + capping del prefisso decoder + rollback.
 * Pura logica (nessun peso): testabile headless. */
typedef struct {
    int window_samples;      /* ampiezza finestra encoder */
    int max_windows;         /* quante finestre tenere (eviction oltre) */
    int uncommitted_tokens;  /* token non ancora "fissati" (finestra di rollback) */
    int max_prefix_tokens;   /* tetto sul prefisso decoder */
    int n_windows;           /* finestre encoder correnti */
    int first_window;        /* indice della finestra più vecchia (eviction) */
    int n_prefix;            /* token di prefisso decoder correnti */
    int64_t evicted_windows; /* contatore diagnostico */
} GmbAsrStream;

void gmb_asr_stream_init(GmbAsrStream* s, int window_samples, int max_windows,
                         int uncommitted_tokens, int max_prefix_tokens);
/* Aggiunge una finestra encoder completata; oltre max_windows evince la più vecchia. */
void gmb_asr_stream_add_window(GmbAsrStream* s);
/* Aggiorna il prefisso decoder dopo aver generato `new_tokens` e fissato `fixed`
 * (rollback): tiene solo gli ultimi max_prefix_tokens. */
void gmb_asr_stream_advance_prefix(GmbAsrStream* s, int fixed_tokens, int new_tokens, int total_tokens);
/* Il prefisso da dare in input al decoder (min(total, max_prefix)). */
int gmb_asr_stream_prefix_len(const GmbAsrStream* s, int total_tokens);

/* ===== Self-check (parità generic vs NEON, round-trip Q8, mel shape, segmentazione) ===== */
/* Ritorna 0 se tutti i check passano, 1 altrimenti. Stampa un report leggibile su `out`. */
int gmb_asr_selfcheck(void* out_stream);

/* ===== Encoder Qwen3-ASR (point 1: Conv2D stem + transformer bidirezionale) ===== */
typedef struct {
    int d_model;        /* 896 / 1024 */
    int layers;         /* 18 / 24 */
    int heads;          /* 14 / 16 */
    int head_dim;       /* 64 */
    int ffn_dim;        /* 3584 / 4096 */
    int output_dim;     /* 1024 / 2048 */
    int n_window;       /* 50 */
    int n_window_infer; /* 800 */
    int chunk_size;     /* 100 = n_window*2 */
} GmbAsrEncConfig;

typedef struct {
    float *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;      /* attn (tutti con bias) */
    float *attn_norm_w, *attn_norm_b;                   /* LayerNorm pre-attn */
    float *fc1, *bfc1, *fc2, *bfc2;                     /* FFN GELU (bias) */
    float *ffn_norm_w, *ffn_norm_b;                     /* LayerNorm pre-ffn */
} GmbAsrEncLayer;

typedef struct {
    float *conv1_w, *conv1_b;  /* [480,1,3,3] + [480] */
    float *conv2_w, *conv2_b;  /* [480,480,3,3] + [480] */
    float *conv3_w, *conv3_b;  /* [480,480,3,3] + [480] */
    float *conv_out_w;         /* [d_model, 7680] (no bias) */
    GmbAsrEncLayer layers[24];
    float *ln_post_w, *ln_post_b;
    float *proj1_w, *bproj1;   /* [d_model,d_model]+[d_model] */
    float *proj2_w, *bproj2;   /* [output_dim,d_model]+[output_dim] */
} GmbAsrEncoder;

/* Forward: mel [GMB_ASR_N_MEL, mel_frames] -> [seq, output_dim] (malloc, free col chiamante).
 * Usa i kernel di questo modulo. Ritorna NULL su errore; *out_seq_len = seq. */
float* gmb_asr_encoder_forward(const GmbAsrEncConfig* cfg, const GmbAsrEncoder* enc,
                               const float* mel, int mel_frames, int* out_seq_len);

/* ===== Entry-point della superficie BASIC (P/Invoke da libgmb_ai_llm) ===== */
/* `ASR.TRANSCRIBE(model$, wav$)`: carica (con cache) modello+tokenizer e trascrive. Ritorna testo
 * malloc'd (liberare con gmb_asr_free) o NULL. */
char* gmb_asr_cmd(const char* model_dir, const char* wav_path, const char* out_path);
void  gmb_asr_free(char* p);

/* ===== Decoder Qwen3 (point 1: GQA + per-head Q/K RMSNorm + NeoX RoPE + SwiGLU + KV cache) ===== */
typedef struct {
    int hidden, layers, heads, kv_heads, head_dim, intermediate, vocab;
    float rms_eps, rope_theta;
} GmbAsrDecConfig;

typedef struct {
    const uint16_t *wq, *wk, *wv, *wo;         /* bf16 [heads*hd,hidden] / [kv*hd,hidden] / [hidden,heads*hd] */
    const float *q_norm, *k_norm;              /* [head_dim] */
    const float *input_norm, *post_attn_norm;  /* [hidden] */
    const uint16_t *gate, *up, *down;          /* bf16 SwiGLU */
} GmbAsrDecLayer;

typedef struct {
    const uint16_t *tok_embeddings;            /* bf16 [vocab,hidden] (tied con lm_head) */
    GmbAsrDecLayer layers[28];
    const float *norm;                         /* [hidden] */
} GmbAsrDecoder;

typedef struct {
    float* k;   /* [layers, max_seq, kv_heads*hd] */
    float* v;
    int len, max;
} GmbAsrKv;

void gmb_asr_kv_init(GmbAsrKv* kv, int layers, int kv_dim, int max_seq);
void gmb_asr_kv_free(GmbAsrKv* kv);
/* Embedding lookup bf16 -> f32 [seq,hidden]. */
void gmb_asr_decoder_embed(const GmbAsrDecConfig* cfg, const GmbAsrDecoder* d, const int* ids, int seq, float* out);
/* Forward di `seq` token a partire da `embeds` [seq,hidden]; appende alla KV; scrive i logits
 * dell'ULTIMO token in `logits` [vocab] (se non NULL). Ritorna 0 / -1. */
int gmb_asr_decoder_forward(const GmbAsrDecConfig* cfg, const GmbAsrDecoder* d, GmbAsrKv* kv,
                            const float* embeds, int seq, float* logits);

/* ===== Modello (pesi HF/safetensors) ===== */
typedef struct GmbAsrModel GmbAsrModel;   /* opaco */
/* Carica i pesi da una dir HF (multi-shard .safetensors). Rileva 0.6B/1.7B dalla presenza di
 * `thinker.audio_tower.layers.18`. Peszi encoder -> F32, decoder -> BF16. NULL su errore. */
GmbAsrModel* gmb_asr_load_model(const char* dir);
void gmb_asr_model_free(GmbAsrModel* m);
const GmbAsrEncConfig* gmb_asr_model_ecfg(const GmbAsrModel* m);
const GmbAsrDecConfig* gmb_asr_model_dcfg(const GmbAsrModel* m);
const GmbAsrEncoder*   gmb_asr_model_encoder(const GmbAsrModel* m);
const GmbAsrDecoder*   gmb_asr_model_decoder(const GmbAsrModel* m);

/* ===== Orchestrazione (point 1/2: mel -> encoder -> prompt -> decoder greedy -> testo) ===== */
/* decode_fn: id token -> stringa UTF-8 (validità fino alla call successiva); il tokenizer lo fornisce
 * il chiamante (riuso LLMToken). Ritorna testo malloc (free col chiamante) o NULL; *out_text_tokens=#. */
typedef const char* (*GmbAsrPieceFn)(int id, void* ctx);
char* gmb_asr_transcribe_segment(const GmbAsrEncConfig* ecfg, const GmbAsrEncoder* enc,
                                 const GmbAsrDecConfig* dcfg, const GmbAsrDecoder* dec,
                                 const float* samples, int n_samples,
                                 const int* prompt_tokens, int n_prompt,
                                 const int* force_tokens, int n_force,
                                 const int* past_tokens, int n_past,
                                 GmbAsrPieceFn decode_fn, void* decode_ctx, int* out_text_tokens);
/* Trascrizione COMPLETA: se `segment_sec`>0 e l'audio è più lungo, taglia ai confini di silenzio
 * (finestra ±`search_sec`) e concatena i segmenti. Altrimenti un solo passaggio. */
char* gmb_asr_transcribe(const GmbAsrEncConfig* ecfg, const GmbAsrEncoder* enc,
                         const GmbAsrDecConfig* dcfg, const GmbAsrDecoder* dec,
                         const float* samples, int n_samples,
                         const int* prompt_tokens, int n_prompt,
                         const int* force_tokens, int n_force,
                         GmbAsrPieceFn decode_fn, void* decode_ctx,
                         float segment_sec, float search_sec, int* out_text_tokens);

/* Streaming (fedele a qwen-asr): chunk di `chunk_sec` (default 2 s) + cache delle finestre
 * encoder (8 s, immutabili) + **prefix rollback** sui token RAW gia' generati (ultimi
 * `rollback_tokens` non "fissati", default 5). Ogni chunk rigenera il contesto con le finestre
 * encoder in cache, cosi' i confini sono stabili (niente artefatti). `emit(piece,len,ctx)` (se non
 * NULL) e' chiamata coi pezzi appena emessi. Ritorna il testo complessivo (malloc). */
typedef void (*GmbAsrStreamEmit)(const char* piece, int len, void* ctx);
char* gmb_asr_transcribe_stream(const GmbAsrEncConfig* ecfg, const GmbAsrEncoder* enc,
                                const GmbAsrDecConfig* dcfg, const GmbAsrDecoder* dec,
                                const float* samples, int n_samples,
                                const int* prompt_tokens, int n_prompt,
                                const int* force_tokens, int n_force,
                                GmbAsrPieceFn decode_fn, void* decode_ctx,
                                int rollback_tokens, float chunk_sec,
                                GmbAsrStreamEmit emit, void* emit_ctx);





#ifdef __cplusplus
}
#endif
#endif /* GMB_ASR_H */
