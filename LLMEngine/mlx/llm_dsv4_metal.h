/* llm_dsv4_metal.h — backend Metal per il compute degli esperti DS4 (IQ2_XXS/Q2_K), glue
 * dell'engine. Zero-copy: aggancia il mmap del GGUF con "view" page-aligned (come ds4_metal.m),
 * niente copia dei 97 GB. Kernel: out[j] = Σ_k dequant(W_row_j,k) * x[k]  (attivazioni f32,
 * parità con llm_dsv4_dot_tensor). ObjC++ (Metal), ABI C. */
#ifndef LLM_DSV4_METAL_H
#define LLM_DSV4_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlmDsv4Metal LlmDsv4Metal;

/* Descrittore di un tensore quantizzato (puntatore raw nella regione mappata + metadati). */
typedef struct { const uint8_t* w; int type; size_t row_bytes; } LlmDsv4GpuView;

/* 1 se esiste un device Metal. */
int llm_dsv4_metal_available(void);

LlmDsv4Metal* llm_dsv4_metal_create(void);
void llm_dsv4_metal_destroy(LlmDsv4Metal* b);

/* View zero-copy su `[file_off, file_off+bytes)` del mmap `mmap_base` (base page-aligned).
 * Ritorna l'handle (da passare a gemv/release) e scrive in *out_view_off l'offset del tensore
 * dentro la view. NULL su errore (mmap non allineato / device). */
void* llm_dsv4_metal_nocopy(LlmDsv4Metal* b, const void* mmap_base, size_t file_off, size_t bytes,
                            size_t* out_view_off);
/* Come nocopy ma da un puntatore arbitrario nella regione mappata (allinea a pagina da solo). */
void* llm_dsv4_metal_nocopy_ptr(LlmDsv4Metal* b, const void* ptr, size_t bytes, size_t* out_view_off);
void llm_dsv4_metal_release(LlmDsv4Metal* b, void* handle);

/* out[j] = dot(W_row_j, x), j in [0,out_dim). `ggml_type`: 16=IQ2_XXS, 10=Q2_K, 8=Q8_0. */
int llm_dsv4_metal_gemv(LlmDsv4Metal* b, void* handle, size_t view_off, int ggml_type,
                        size_t row_bytes, uint32_t n, uint32_t out_dim,
                        const float* x, float* out);

/* Un lavoro GEMV per il batch (view zero-copy transitoria: creata ed eliminata per job,
 * NESSUNA residenza persistente del mmap). */
typedef struct {
    const void* w; size_t byte_off; int ggml_type; size_t row_bytes;
    uint32_t n, out_dim; const float* x; float* out;
} LlmDsv4GpuJob;

/* Esegue `n_jobs` GEMV in una command buffer (commit+wait una volta). 0 = ok. */
int llm_dsv4_metal_gemv_batch(LlmDsv4Metal* b, const LlmDsv4GpuJob* jobs, int n_jobs);

/* MoE fuso ds4 (pair): gate+up in un pass + swiglu*peso -> mid[ne0]. Ritorna 0 = ok.
 * `gate`/`up` = inizio dell'esperto (dati packed); `x` = attivazione f32 [ne00]. */
int llm_dsv4_metal_pair_swiglu(LlmDsv4Metal* b, const void* gate, const void* up,
                               const float* x, float* mid, float route_w, float clamp,
                               uint32_t ne00, uint32_t nb01, uint32_t ne0);

/* Batch: `n` esperti in UNA command buffer + UN wait (gate/up/mid/route per esperto). */
int llm_dsv4_metal_pair_swiglu_batch(LlmDsv4Metal* b, const void* const* gates, const void* const* ups,
                                     const float* x, float* const* mids, const float* rws, int n,
                                     float clamp, uint32_t ne00, uint32_t nb01, uint32_t ne0);

/* MoE ds4 fuso: `n`=6 esperti (gate/up IQ2_XXS, down Q2_K) in UNA command buffer.
 * pair_swiglu (gate+up+swiglu*peso -> mid f32 su device) + sum6 (down dei 6 + somma -> out).
 * Nessun round-trip CPU. `out[embd]` scritto per intero. 0 = ok. */
int llm_dsv4_metal_moe_iq2_q2k(LlmDsv4Metal* b, const void* const* gates, const void* const* ups,
                               const void* const* downs, const float* x, const float* rws, int n,
                               float clamp, uint32_t embd, uint32_t gu_nb01, uint32_t ff,
                               uint32_t down_nb01, const void* sg, const void* su, const void* sd,
                               uint32_t s_nb_g, uint32_t s_nb_d, float* out);

/* Attn mista (raw+comp) su KV device: replica llm_dsv4_attention_mixed. `allowed`/`comp_kv`/`raw_kv`
 * possono essere NULL se n_comp/n_raw = 0. Scrive out[n_head*hd]. 0 = ok. */
int llm_dsv4_metal_attention_mixed(LlmDsv4Metal* b, const float* q, int n_head, int hd,
        const float* raw_kv, int n_raw, const float* comp_kv, int n_comp,
        const int* allowed, const float* sinks, float* out);

/* Compressor su device (replica llm_dsv4_compressor_*): pool dello stato -> pooled[hd]. 0 = ok. */
int llm_dsv4_metal_comp_pool(LlmDsv4Metal* b, const float* state_kv, const float* state_score,
        float* pooled, int hd, int ratio, int co);

/* Compressor: fp8 e4m3 in-place di una riga (n_nope = hd - n_rot, multiplo di 64). 0 = ok. */
int llm_dsv4_metal_comp_fp8(LlmDsv4Metal* b, float* x, int hd, int n_rot);

/* Indexer su device (replica llm_dsv4_indexer_*): QAT in-place, score per le compresse, top-k. */
int llm_dsv4_metal_indexer_qat(LlmDsv4Metal* b, float* x, int n_rows, int hd);
int llm_dsv4_metal_indexer_scores(LlmDsv4Metal* b, const float* q, int nh, int hd,
        const float* kvc, int n_comp, const float* w, float* scores);
int llm_dsv4_metal_indexer_topk(LlmDsv4Metal* b, const float* scores, int n_comp, int top_k, int* allowed);

/* Buffer device opachi (pesi f32 + stato KV persistenti per il middle su GPU). */
void* llm_dsv4_metal_buf_alloc(LlmDsv4Metal* b, size_t bytes);
void  llm_dsv4_metal_buf_write(void* h, size_t off, const void* src, size_t bytes);
void  llm_dsv4_metal_buf_read(void* h, size_t off, void* dst, size_t bytes);
void  llm_dsv4_metal_buf_free(void* h);

/* Shift [src, src+n) -> [0,n) di un buffer device (stato compressor). 0 = ok. */
int llm_dsv4_metal_comp_shift(LlmDsv4Metal* b, void* h, int n, int src);

/* ── Middle attenzione su GPU (compressor+indexer+attn) in UNA command buffer ──
 * KV/stato su device (buffer opachi); pesi compressor/indexer f32 su device; input/output host.
 * Replica llm_dsv4_attn_cached_mid (esclusa la proiezione, gia' fatta dal caller). 0 = ok. */
typedef struct {
    int embd, hd, ihd, ratio, co, n_head, n_rot;
    int idx_nh, idx_hd, idx_top_k, lq, qdim;
    int pos, due, state_row, ape_off, has_idx;
    float rms_eps;
    float rope_cfg[7];
    void* w_ckv; void* w_cgate; void* w_cape; void* w_cnorm;
    void* w_ickv; void* w_icgate; void* w_icape; void* w_icnorm;
    void* w_iqb; void* w_iproj; void* sinks;
    void* st_akv; void* st_asc; void* st_ikv; void* st_isc;
    void* raw;  int raw_cap; int raw_off; int raw_full;
    void* acomp; int acomp_off;
    void* icomp; int icomp_off;
    void* allowed;
    const float* xn; const float* qrn; const float* q; const float* kv_cur;
    int n_raw, n_acomp, n_icomp;
    float* heads;
} LlmDsv4MidParams;

int llm_dsv4_metal_attn_mid(LlmDsv4Metal* b, const LlmDsv4MidParams* p);

/* Output grouped-LoRA FUSO (ds4): out_a (ng gruppi) + out_b in UNA command buffer, `low` su device. */
int llm_dsv4_metal_attn_out_fused(LlmDsv4Metal* b, const void* outa_w, size_t outa_row_bytes,
        size_t outa_lead0, int ng, uint32_t gd, uint32_t rank, const void* outb_w,
        size_t outb_row_bytes, uint32_t ol, uint32_t embd, uint32_t heads_total,
        const float* heads, float* out);

/* Proiezioni Q-LoRA + KV FUSE (ds4): q_a, kv, RMS, q_b, head_rms, RoPE in UNA command buffer. */
int llm_dsv4_metal_attn_proj_fused(LlmDsv4Metal* b, const void* qa_w, size_t qa_row_bytes,
        const void* kv_w, size_t kv_row_bytes, const void* qb_w, size_t qb_row_bytes,
        const float* xn, uint32_t embd, const float* qan_w, const float* kvan_w,
        uint32_t lq, uint32_t hd, uint32_t qd, int n_head, int n_rot,
        const float* rope_cfg, int pos, float eps, float* q_out, float* kv_out, float* qrn_out);

#ifdef __cplusplus
}
#endif

/* ── Tail FUSO del layer (ds4): fin + mhc + route + moe in UNA command buffer ── */
typedef struct {
    const void* outa_w; size_t outa_row_bytes; int ng; uint32_t gd, rank;
    const void* outb_w; size_t outb_row_bytes; uint32_t ol;
    int n_head, hd, n_rot, pos; float rope_cfg[7];
    const float* res_attn;    /* [HC*D]  (residual prima di mhc_pre_attn) */
    const float* post_attn;   /* [HC] */
    const float* comb_attn;   /* [HC*HC] */
    const float* fn;          /* [SIZE*HCD] (hc_ffn_fn, f32) */
    const float* scale;       /* [3] */
    const float* base;        /* [SIZE] */
    const float* norm_w;      /* [D] (ffn_norm) */
    const void* router_w;     /* f32 [n_expert, D] (ffn_gate_inp) */
    const float* exp_probs_b; /* [n_expert] o NULL */
    int n_expert, used; float expert_scale;
    const void* arg_pair; const void* arg_down;  /* argument buffer "table" (ds4) */
    float clamp; uint32_t gu_nb01, ff, down_nb01;
    const void* sg; const void* su; const void* sd; uint32_t s_nb_g, s_nb_d;
    int HC, D, SIZE, HCD, sinkhorn_iters; float hc_eps, rms_eps;
    const float* heads;       /* [n_head*hd] */
    float* out_hc;            /* [HC*D] */
} LlmDsv4TailParams;

/* Esegue l'intero tail del layer in UNA command buffer. 0 = ok, <0 = fallback CPU. */
int llm_dsv4_metal_layer_tail(LlmDsv4Metal* b, const LlmDsv4TailParams* q);

/* Test: se dir != NULL, il tail scrive gli stadi in <dir>/d4_*.bin. */
void llm_dsv4_metal_tail_set_dump(const char* dir);

/* Registra una regione host come risorsa Metal no-copy (retained) per l'accesso via puntatore
 * grezzo dai kernel addr; rilasciare con _release. NULL su errore. */
void* llm_dsv4_metal_register(LlmDsv4Metal* b, const void* ptr, size_t bytes);
void  llm_dsv4_metal_register_release(void* h);

/* gpuAddress (buffer per-esperto registrato, persistente) dell'esperto in `ptr`. */
uint64_t llm_dsv4_metal_expert_gaddr(LlmDsv4Metal* b, const void* ptr, size_t bytes);

/* Argument buffer "table" (ds4): encoda n esperti (view no-copy) e li seleziona per indice. */
void* llm_dsv4_metal_argbuf_pair(LlmDsv4Metal* b, const void* const* gs, const void* const* us, int n,
                                 size_t gbytes, size_t ubytes);
void* llm_dsv4_metal_argbuf_down(LlmDsv4Metal* b, const void* const* ds, int n, size_t dbytes);
void  llm_dsv4_metal_argbuf_bind(void* enc, void* handle, int index);
void  llm_dsv4_metal_argbuf_free(void* handle);

#endif /* LLM_DSV4_METAL_H */
