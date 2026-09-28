/* llmmodelio_infer.c — forward MoE transformer (olmoe) per la calibrazione del routing.
 * Port fedele di llama.cpp/src/models/olmoe.cpp. Pesi via mmap (F16/F32), niente copie. */
#include "llmmodelio_infer.h"
#include "llmcodec_gguf_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define LLMMODELIO_HAVE_F16_NEON 1
#endif
#endif

static float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    float v = (e == 0) ? ldexpf((float)m, -24) : (e == 31) ? (m ? NAN : INFINITY) : ldexpf(1.0f + (float)m / 1024.0f, (int)e - 15);
    return s ? -v : v;
}
typedef struct { const uint8_t* base; int type; uint64_t numel; uint64_t n0; } WT;   /* n0 = dims[0] */
static inline float wget(const WT* w, uint64_t i) {
    return (w->type == 0) ? ((const float*)w->base)[i] : h2f(((const uint16_t*)w->base)[i]);
}
static const LlmCodecGgufTensorInfo* find_ts(const LlmCodecGgufFileInfo* fi, const char* nm) {
    for (uint64_t i = 0; i < fi->n_tensors; i++) if (strcmp(fi->tensors[i].name, nm) == 0) return &fi->tensors[i];
    return NULL;
}
static void mk(WT* w, const LlmCodecGgufFileInfo* fi, const uint8_t* map, const char* nm) {
    const LlmCodecGgufTensorInfo* t = find_ts(fi, nm);
    if (!t) { w->base = NULL; w->type = -1; w->numel = 0; w->n0 = 0; return; }
    w->base = map + fi->data_start + t->offset; w->type = t->ggml_type;
    uint64_t n = 1; for (int d = 0; d < t->ndim; d++) n *= t->dims[d];
    w->numel = n; w->n0 = t->ndim > 0 ? t->dims[0] : 0;
}

/* I tensori che il CORPO del ciclo di run_forward legge, nell'ordine in cui li mk()a. Un nome in
 * piu' o in meno fra questa lista e il ciclo e' il difetto che rendeva un layer con un tensore
 * assente o SILENZIOSO (i `counts` a zero) o FATALE: `attn_q_norm`/`ffn_gate_inp`/`up_exps`
 * venivano dereferenziati con `base == NULL` perche' la guardia ne controllava solo due.
 * Questa e' l'UNICA lista: ciclo, pre-flight e guardia dtype leggono questa. */
#define LLM_LAYER_NT 12
static const char* const LLM_LAYER_SUFFIX[LLM_LAYER_NT] = {
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight",
    "ffn_norm.weight", "ffn_gate_inp.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
enum { L_AN = 0, L_WQ = 1, L_WK = 2, L_WV = 3, L_QN = 4, L_KN = 5, L_WO = 6,
       L_FN = 7, L_GINP = 8, L_GEX = 9, L_UEX = 10, L_DEX = 11 };

/* I pesi che il forward DEREFENZIA: i 12 per layer + `token_embd.weight`. Qui c'erano anche
 * `output.weight` e `output_norm.weight`, ma erano solo passati a mk() e `(void)`ati subito dopo:
 * nessun byte letto. Lasciarli significava rifiutare una calibrazione legittima perche' il lm_head
 * era quantizzato - che e' NORMALE (llama.cpp lo tiene a bit piu' alti dei pesi). */
static int llm_is_read_name(const char* nm) {
    size_t nl = strlen(nm), sl;
    if (strcmp(nm, "token_embd.weight") == 0) return 1;
    for (int s = 0; s < LLM_LAYER_NT; s++) {
        sl = strlen(LLM_LAYER_SUFFIX[s]);
        if (nl >= sl && memcmp(nm + (nl - sl), LLM_LAYER_SUFFIX[s], sl) == 0) return 1;
    }
    return 0;
}

/* Guardia sui dtype — AGENTS §0bis/5: mai risultato silenzioso.
 * `wget` (:26) e le stride di `dot_row`/`:70-75`/`:256`/`:267` dicono "tipo 0 -> F32, ALTRIMENTI
 * F16": con un Q4_K (12), Q6_K (14) o BF16 (30) i byte venivano letti come half con stride 2,
 * senza un messaggio — output spazzatura che sembra un routing calcolato. `LLM PROFILE <model>,
 * <corpus>, <out>` porta il path dell'utente dritto qui (emissione in
 * `llm_compiler_c/src/llm_c99_emit.c:15529`), quindi il caso non è teorico: il modello olmoe
 * nominato dal commento di questo file è distribuito proprio come `…-q4_k_m.gguf`.
 * Ritorna -1 e nomina i primi tensori col tipo non supportato. */
static int llm_check_used_dtypes(const LlmCodecGgufFileInfo* fi) {
    int bad = 0;
    for (uint64_t i = 0; i < fi->n_tensors; i++) {
        const char* nm = fi->tensors[i].name;
        int t = fi->tensors[i].ggml_type;
        if (t == 0 || t == 1) continue;                 /* F32 / F16: gli unici due letti */
        if (llm_is_read_name(nm)) {
            if (bad < 4)
                fprintf(stderr, "LLM PROFILE: %s: tipo GGML %d - questo forward calibra il "
                                "routing leggendo i pesi come F32/F16; dequantizza il modello "
                                "(o passa il GGUF F32 di casa), e non lo legge come F16\n",
                        nm, t);
            bad++;
        }
    }
    if (bad > 4) fprintf(stderr, "LLM PROFILE: ... e altri %d tensori con tipo non supportato\n", bad - 4);
    return bad ? -1 : 0;
}
/* I due avvertimenti NON fatali si dicono UNA volta per MODELLO: run_forward viene chiamato una
 * volta per chunk da llmmodelio_imatrix_compute e il difetto segnalato e' dello STESSO file, quindi
 * ripeterlo per chunk sarebbe rumore che nessuno legge; ma un modello DIVERSO ha un difetto
 * diverso e va sentito lo stesso. */
static char llm_warned_partial[512] = { 0 }, llm_warned_dropped[512] = { 0 };

/* mk()a i 12 tensori del layer `il` in `w`; ritorna quanti NE MANCANO e, se `miss` != NULL,
 * scrive in `miss` i nomi dei tensori che non ha trovato (separati da " /"), troncati a `cap`. */
static int layer_tensors(WT* w, const LlmCodecGgufFileInfo* fi, const uint8_t* map,
                         int il, char* miss, size_t cap) {
    char nm[128];
    int absent = 0;
    if (miss && cap) miss[0] = '\0';
    for (int s = 0; s < LLM_LAYER_NT; s++) {
        snprintf(nm, sizeof nm, "blk.%d.%s", il, LLM_LAYER_SUFFIX[s]);
        mk(&w[s], fi, map, nm);
        if (w[s].base != NULL) continue;
        absent++;
        if (miss && cap) {
            size_t u = strlen(miss);
            if (u + strlen(nm) + 4 < cap) snprintf(miss + u, cap - u, "%s%s", u ? " / " : "", nm);
        }
    }
    return absent;
}
static void rmsnorm(float* o, const float* x, const WT* w, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i]*x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * wget(w, (uint64_t)i);
}
/* y[o] = sum_i W[o*n0 + i] * x[i]  (W: d0=n0) */
#ifdef LLMMODELIO_HAVE_F16_NEON
static inline float dot_f16_neon(const uint16_t* w, const float* x, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(w + i));
        acc = vfmaq_f32(acc, vcvt_f32_f16(vget_low_f16(h)),  vld1q_f32(x + i));
        acc = vfmaq_f32(acc, vcvt_f32_f16(vget_high_f16(h)), vld1q_f32(x + i + 4));
    }
    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += h2f(w[i]) * x[i];
    return s;
}
static inline float dot_f32_neon(const float* w, const float* x, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) acc = vfmaq_f32(acc, vld1q_f32(w + i), vld1q_f32(x + i));
    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += w[i] * x[i];
    return s;
}
#endif
static void mm(const WT* W, const float* x, float* y, int n_out, int n_in) {
    for (int o = 0; o < n_out; o++) {
        const uint8_t* row = W->base + (uint64_t)o * n_in * (W->type == 0 ? 4 : 2);
#ifdef LLMMODELIO_HAVE_F16_NEON
        y[o] = (W->type == 0) ? dot_f32_neon((const float*)row, x, n_in) : dot_f16_neon((const uint16_t*)row, x, n_in);
#else
        double s = 0;
        if (W->type == 0) { const float* r = (const float*)row; for (int i = 0; i < n_in; i++) s += (double)r[i]*x[i]; }
        else { const uint16_t* r = (const uint16_t*)row; for (int i = 0; i < n_in; i++) s += (double)h2f(r[i])*x[i]; }
        y[o] = (float)s;
#endif
    }
}

/* Accumulatore imatrix per layer: sum2 (somma dei quadrati per canale; [n_in*E] per gli esperti) + cnt,
 * piu' i campioni Monte Carlo (§ llmmodelio_infer.h) per la stima ΔWᵀΣΔW in llmmodelio_roundtrip_rel_cov.
 * k_sh/k_ex sono la CAPACITA' di QUESTA istanza (piccola — LLM_SAMP_CHUNK_* — per l'accumulatore
 * per-chunk creato dentro run_forward; grande — LLMMODELIO_SAMP_K_* — per il `tot` che li concatena
 * fra chunk in llmmodelio_imatrix_compute); h_samp_n/att_samp_n/he_samp_n/de_samp_n tracciano quanti
 * slot sono gia' pieni per sapere dove scrivere il prossimo campione. */
typedef struct { float* q; double qc; float* k; double kc; float* v; double vc; float* o; double oc;
                 float* gi; double gic; float* ge; double* gec; float* ue; double* uec; float* de; double* dec;
                 int k_sh, k_ex;
                 float* h_samp; int h_samp_n;       /* [k_sh * ne] — input condiviso di q/k/v */
                 float* att_samp; int att_samp_n;   /* [k_sh * ne] — input di attn_output */
                 float* he_samp; int* he_samp_n;    /* [E * k_ex * ne] — input per-esperto di gate/up */
                 float* de_samp; int* de_samp_n;    /* [E * k_ex * nff] — input per-esperto di down */
               } LAcc;
static void acc_init(LAcc* a, int ne, int nff, int E, int k_sh, int k_ex) {
    memset(a, 0, sizeof *a);
    a->q=(float*)calloc(ne,4); a->k=(float*)calloc(ne,4); a->v=(float*)calloc(ne,4); a->o=(float*)calloc(ne,4); a->gi=(float*)calloc(ne,4);
    a->ge=(float*)calloc((size_t)ne*E,4); a->ue=(float*)calloc((size_t)ne*E,4); a->de=(float*)calloc((size_t)nff*E,4);
    a->gec=(double*)calloc(E,8); a->uec=(double*)calloc(E,8); a->dec=(double*)calloc(E,8);
    a->k_sh = k_sh; a->k_ex = k_ex;
    a->h_samp=(float*)calloc((size_t)k_sh*ne,4); a->att_samp=(float*)calloc((size_t)k_sh*ne,4);
    a->he_samp=(float*)calloc((size_t)E*k_ex*ne,4); a->de_samp=(float*)calloc((size_t)E*k_ex*nff,4);
    a->he_samp_n=(int*)calloc(E,sizeof(int)); a->de_samp_n=(int*)calloc(E,sizeof(int));
}
static void acc_free(LAcc* a) {
    free(a->q);free(a->k);free(a->v);free(a->o);free(a->gi);free(a->ge);free(a->ue);free(a->de);free(a->gec);free(a->uec);free(a->dec);
    free(a->h_samp);free(a->att_samp);free(a->he_samp);free(a->de_samp);free(a->he_samp_n);free(a->de_samp_n);
}
/* NB (2026-09-22): NON cappare l'in_sum2. Provato: un cap a 1e6 rende i valori FINITI ma ~4e11,
 * il guard non li blocca e i pesi giganti di blk.15 ROVINANO la quantizzazione (PPL 59.9 vs 35.3).
 * La scelta corretta e' lasciare il valore estremo -> inf -> il guard `isfinite` in llmmodelio_imatrix_weights
 * ESCLUDE quei tensori dal weighting (fallback non-pesato). Verificato: tiny 35.27, pieno 12.3748. */
static inline void add_ch(float* s, const float* x, int n) { for (int i=0;i<n;i++) s[i]+=x[i]*x[i]; }
static inline void add_ch_e(float* s, const float* x, int n, int e) { float* d=s+(size_t)e*n; for (int i=0;i<n;i++) d[i]+=x[i]*x[i]; }
/* come llama.cpp: l'input degli esperti e' h * peso di routing (build_moe_ffn: ggml_mul(repeated,weights)) */
static inline void add_ch_e_w(float* s, const float* x, int n, int e, float w) { float* d=s+(size_t)e*n; float w2=w*w; for (int i=0;i<n;i++) d[i]+=w2*x[i]*x[i]; }
/* Copia x nel prossimo slot libero (count < cap); slot mai riempiti restano a zero (calloc) — non
 * introducono bias nella somma finale (§ llmmodelio_roundtrip_rel_cov), quindi nessun conteggio da salvare. */
static inline int samp_add(float* s, int n, int count, int cap, const float* x) {
    if (count >= cap) return count;
    memcpy(s + (size_t)count * n, x, (size_t)n * sizeof(float));
    return count + 1;
}
#define LLMMODELIO_SAMP_CHUNK_SH 8
#define LLMMODELIO_SAMP_CHUNK_EX 4

int llmmodelio_imatrix_compute(const char* model_path, const int32_t* tokens, int32_t n_tokens, const char* out_path, int chunk);
static double* run_forward(const char* model_path, const int32_t* tokens, int32_t n_tokens,
                           int* out_n_layer, int* out_n_expert, LAcc** out_acc, int* out_nembd, int* out_nff, int acc_want) {
    if (model_path == NULL || tokens == NULL || n_tokens <= 0) return NULL;
    LlmCodecGgufFileInfo fi; LlmCodecGgufMetaKVOwned* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_read_meta(model_path, &kv, &nkv, &fi) != 0) return NULL;
    int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, n_expert = 0, n_used = 0, n_vocab = 0;
    long dropped = 0;   /* token fuori vocabolario: prima SCARTATI senza contarli, e il denominatore
                         * della calibrazione cambiava senza che nessuno lo dicesse */
    float eps = 1e-5f, rope_base = 10000.0f;
    for (int i = 0; i < nkv; i++) {
        const char* k = kv[i].key;
        if (strstr(k, "block_count") && kv[i].type == LLMCODEC_GGUF_MT_U32) n_layer = (int)kv[i].u32;
        else if (strstr(k, "embedding_length") && kv[i].type == LLMCODEC_GGUF_MT_U32) n_embd = (int)kv[i].u32;
        else if (strstr(k, "feed_forward_length") && kv[i].type == LLMCODEC_GGUF_MT_U32) n_ff = (int)kv[i].u32;
        else if (strstr(k, "attention.head_count_kv") && kv[i].type == LLMCODEC_GGUF_MT_U32) n_head_kv = (int)kv[i].u32;
        else if (strstr(k, "attention.head_count") && kv[i].type == LLMCODEC_GGUF_MT_U32) n_head = (int)kv[i].u32;
        else if (strstr(k, "expert_used_count") && kv[i].type == LLMCODEC_GGUF_MT_U32) n_used = (int)kv[i].u32;
        else if (strstr(k, "expert_count") && kv[i].type == LLMCODEC_GGUF_MT_U32) n_expert = (int)kv[i].u32;
        else if (strstr(k, "rms_epsilon") && kv[i].type == LLMCODEC_GGUF_MT_F32) eps = kv[i].f32;
        else if (strstr(k, "rope.freq_base") && kv[i].type == LLMCODEC_GGUF_MT_F32) rope_base = kv[i].f32;
    }
    if (n_layer <= 0 || n_embd <= 0 || n_expert <= 0) {
        /* Prima: `return NULL` senza una parola -> chi ha lanciato LLM PROFILE non sapeva NEANCHE
         * quale dei tre mancasse. L'errore nomina l'assente e nomina i tre attesi. */
        fprintf(stderr, "LLM PROFILE: %s: calibrazione impossibile, metadata NON dichiarati:%s%s%s"
                        " (attesi come u32: <arch>.block_count, <arch>.attention.embedding_length,"
                        " <arch>.expert_count)\n", model_path,
                n_layer  <= 0 ? " block_count" : "",
                n_embd   <= 0 ? " attention.embedding_length" : "",
                n_expert <= 0 ? " expert_count" : "");
        llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return NULL;
    }
    int hd = n_embd / n_head;
    int fd = open(model_path, O_RDONLY); struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) { llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return NULL; }
    const uint8_t* map = (const uint8_t*)mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { close(fd); llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi); return NULL; }
    /* Prima di toccare un byte: se i pesi usati non sono F32/F16 il forward NON può girare
     * (leggerli come F16 era il difetto, non un caso limite). */
    if (llm_check_used_dtypes(&fi) != 0) {
        munmap((void*)map, (size_t)st.st_size); close(fd);
        llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
        fprintf(stderr, "LLM PROFILE: modello rifiutato, nessun conteggio prodotto\n");
        return NULL;
    }
    /* Guardia sui tensori del layer, DECISA PRIMA di allocare: il `continue` a fondo ciclo saltava
     * il layer senza dire nulla e, se li saltava TUTTI, i `counts` restavano tutti zeri — un
     * profilo che `LLM PRUNE` consuma come se fosse una calibrazione riuscita (AGENTS §0bis/§5:
     * mai un risultato silenzioso). I nomi alternativi NON si cercano "per indovinare il layout":
     * l'errore nomina cio' che cercava. */
    {
        int missing = 0;
        char miss[256] = { 0 };
        WT lay[LLM_LAYER_NT];
        for (int il = 0; il < n_layer; il++) {
            /* `miss` si riempie SOLO al primo layer difettoso: e' quello che il messaggio nomina. */
            if (layer_tensors(lay, &fi, map, il, missing == 0 ? miss : NULL, sizeof miss) > 0) missing++;
        }
        if (missing == n_layer) {
            fprintf(stderr, "LLM PROFILE: %s: NESSUN layer su %d e' calibrabile - expert_count=%d"
                            " dichiarato nei metadata, ma mancano %s. Il forward cerca i nomi GGUF"
                            " canonici (layout per-expert alla Mixtral o attenzione fusa NON sono"
                            " supportati); nessun conteggio prodotto\n",
                    model_path, n_layer, n_expert, miss);
            munmap((void*)map, (size_t)st.st_size); close(fd);
            llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
            return NULL;
        }
        if (missing && strncmp(llm_warned_partial, model_path, sizeof llm_warned_partial) != 0) {
            snprintf(llm_warned_partial, sizeof llm_warned_partial, "%s", model_path);
            fprintf(stderr, "LLM PROFILE: %s: %d layer su %d hanno tensori ASSENTI (primo colpevole:"
                            " %s) -> esclusi dal profilo, i loro conteggi restano a zero\n",
                    model_path, missing, n_layer, miss);
        }
    }

    /* Solo `token_embd.weight`: `output_norm`/`output` erano mk()ati e buttati con `(void)`
     * subito dopo - nessun byte letto (vedi llm_is_read_name). */
    WT emb;
    mk(&emb, &fi, map, "token_embd.weight");
    n_vocab = (int)(emb.numel / (uint64_t)n_embd);

    double* counts = (double*)calloc((size_t)n_layer * (size_t)n_expert, sizeof(double));
    float* x = (float*)malloc((size_t)n_embd * sizeof(float));
    float* h = (float*)malloc((size_t)n_embd * sizeof(float));
    float* q = (float*)malloc((size_t)n_embd * sizeof(float));
    float* k = (float*)malloc((size_t)n_embd * sizeof(float));
    float* v = (float*)malloc((size_t)n_embd * sizeof(float));
    float* att = (float*)malloc((size_t)n_embd * sizeof(float));
    /* g1 OSPITA TRE vettori: gate in [0..n_ff), up in [n_ff..2*n_ff) (:394-395, "riuso") e le
     * probabilita' di attenzione pr[0..n_tokens) a :354. Prima era allocato max(n_ff,n_embd):
     * con n_ff=4096 si scrivevano 4096 float OLTRE il blocco, dentro il heap adiacente (g2,
     * allocato subito dopo). Mai un crash, sempre corruzione silenziosa. */
    float* g1 = NULL;
    {
        size_t g1n = (size_t)(2*n_ff > n_embd ? 2*n_ff : n_embd);
        if ((size_t)n_tokens > g1n) g1n = (size_t)n_tokens;   /* pr[tt], tt <= n_tokens-1 */
        g1 = (float*)malloc(g1n * sizeof(float));
    }
    float* g2 = (float*)malloc((size_t)(n_ff > n_embd ? n_ff : n_embd) * sizeof(float));
    float* kcache = (float*)calloc((size_t)n_layer * n_tokens * n_embd, sizeof(float));
    float* vcache = (float*)calloc((size_t)n_layer * n_tokens * n_embd, sizeof(float));
    float* logits = (float*)malloc((size_t)(n_expert > n_vocab ? n_expert : n_vocab) * sizeof(float));
    int* topi = (int*)malloc((size_t)n_used * sizeof(int));
    float* topw = (float*)malloc((size_t)n_used * sizeof(float));
    LAcc* acc = acc_want ? (LAcc*)calloc((size_t)n_layer, sizeof(LAcc)) : NULL;
    if (acc) for (int l = 0; l < n_layer; l++) acc_init(&acc[l], n_embd, n_ff, n_expert, LLMMODELIO_SAMP_CHUNK_SH, LLMMODELIO_SAMP_CHUNK_EX);
    if (!counts || !x || !h || !q || !k || !v || !att || !g1 || !g2 || !kcache || !vcache || !logits || !topi || !topw) goto done;

    for (int t = 0; t < n_tokens; t++) {
        int tok = tokens[t]; if (tok < 0 || tok >= n_vocab) { dropped++; continue; }
        for (int i = 0; i < n_embd; i++) x[i] = wget(&emb, (uint64_t)tok * n_embd + i);
        for (int il = 0; il < n_layer; il++) {
            float* kc = kcache + ((size_t)il * n_tokens + t) * n_embd;
            float* vc = vcache + ((size_t)il * n_tokens + t) * n_embd;
            /* UNA sola lista dei tensori del layer (LLM_LAYER_SUFFIX): la guardia e il corpo del
             * ciclo leggono la stessa, quindi il "N layer esclusi" detto in testa coincide con
             * cio' che il ciclo salta DAVVERO. */
            WT lay[LLM_LAYER_NT], an, wq, wk, wv, qn, kn, wo, fn, ginp, gex, uex, dex;
            if (layer_tensors(lay, &fi, map, il, NULL, 0) > 0) continue;   /* gia' contato in testa */
            an = lay[L_AN]; wq = lay[L_WQ]; wk = lay[L_WK]; wv = lay[L_WV];
            qn = lay[L_QN]; kn = lay[L_KN]; wo = lay[L_WO]; fn = lay[L_FN];
            ginp = lay[L_GINP]; gex = lay[L_GEX]; uex = lay[L_UEX]; dex = lay[L_DEX];
            /* attn */
            rmsnorm(h, x, &an, n_embd, eps);
            if (acc) { add_ch(acc[il].q, h, n_embd); add_ch(acc[il].k, h, n_embd); add_ch(acc[il].v, h, n_embd); acc[il].qc+=1; acc[il].kc+=1; acc[il].vc+=1;
                       acc[il].h_samp_n = samp_add(acc[il].h_samp, n_embd, acc[il].h_samp_n, acc[il].k_sh, h); }
            mm(&wq, h, q, n_embd, n_embd);
            mm(&wk, h, k, n_embd, n_embd);
            mm(&wv, h, v, n_embd, n_embd);
            rmsnorm(q, q, &qn, n_embd, eps);   /* q_norm sull'intero vettore (prima della reshape) */
            rmsnorm(k, k, &kn, n_embd, eps);
            /* RoPE NEOX (a meta': (d, d+hd/2) per head) — convenzione HF rotate_half di OLMoE.
             * theta_i = base^(-2i/hd) con i = 0..hd/2-1. */
            for (int hh = 0; hh < n_head; hh++) {
                for (int d = 0; d < hd/2; d++) {
                    float th = powf(rope_base, -2.0f * (float)d / (float)hd) * (float)t;
                    float c = cosf(th), s = sinf(th);
                    uint64_t b0 = (uint64_t)hh * hd + d, b1 = b0 + hd/2;
                    float a0 = q[b0], a1 = q[b1];
                    q[b0] = a0*c - a1*s; q[b1] = a0*s + a1*c;
                    float k0 = k[b0], k1 = k[b1];
                    k[b0] = k0*c - k1*s; k[b1] = k0*s + k1*c;
                }
            }
            memcpy(kc, k, (size_t)n_embd * sizeof(float));
            memcpy(vc, v, (size_t)n_embd * sizeof(float));
            float sc = 1.0f / sqrtf((float)hd);
            for (int i = 0; i < n_embd; i++) att[i] = 0;
            for (int hh = 0; hh < n_head; hh++) {
                float best = -1e30f; float* pr = g1;
                for (int tt = 0; tt <= t; tt++) {
                    const float* kb = kcache + ((size_t)il * n_tokens + tt) * n_embd + (size_t)hh * hd;
                    double s = 0; for (int d = 0; d < hd; d++) s += (double)q[(size_t)hh*hd+d] * kb[d];
                    pr[tt] = (float)(s * sc);
                    if (pr[tt] > best) best = pr[tt];
                }
                double sum = 0; for (int tt = 0; tt <= t; tt++) { pr[tt] = expf(pr[tt] - best); sum += pr[tt]; }
                for (int tt = 0; tt <= t; tt++) {
                    float w = (float)(pr[tt] / sum);
                    const float* vb = vcache + ((size_t)il * n_tokens + tt) * n_embd + (size_t)hh * hd;
                    for (int d = 0; d < hd; d++) att[(size_t)hh*hd+d] += w * vb[d];
                }
            }
            if (acc) { add_ch(acc[il].o, att, n_embd); acc[il].oc+=1;
                       acc[il].att_samp_n = samp_add(acc[il].att_samp, n_embd, acc[il].att_samp_n, acc[il].k_sh, att); }
            mm(&wo, att, h, n_embd, n_embd);
            for (int i = 0; i < n_embd; i++) x[i] += h[i];
            if (getenv("LLM_INFER_TRACE")) { float mx=0; for(int i=0;i<n_embd;i++) if(fabsf(x[i])>mx) mx=fabsf(x[i]); fprintf(stderr, "[tr] t=%d l=%d dopo-attn max|x|=%.4g\n", t, il, mx); }
            /* MoE */
            rmsnorm(h, x, &fn, n_embd, eps);
            if (acc) { add_ch(acc[il].gi, h, n_embd); acc[il].gic+=1; }
            mm(&ginp, h, logits, n_expert, n_embd);      /* logits esperto */
            /* softmax sui 64 */
            { float mx = logits[0]; for (int e = 1; e < n_expert; e++) if (logits[e] > mx) mx = logits[e];
              double s = 0; for (int e = 0; e < n_expert; e++) { logits[e] = expf(logits[e] - mx); s += logits[e]; }
              for (int e = 0; e < n_expert; e++) logits[e] = (float)(logits[e] / s); }
            /* top-n_used (norm_topk_prob=false -> pesi = prob softmax, nessuna rinormalizzazione) */
            for (int a = 0; a < n_used; a++) { int bi = -1; float bv = -1; for (int e = 0; e < n_expert; e++) { int u = 0; for (int q2 = 0; q2 < a; q2++) if (topi[q2] == e) { u = 1; break; } if (!u && logits[e] > bv) { bv = logits[e]; bi = e; } } topi[a] = bi < 0 ? 0 : bi; topw[a] = bv < 0 ? 0 : bv; counts[(size_t)il * n_expert + topi[a]] += 1.0; }
            /* esperti: out = sum w * down(silu(gate(h)) * up(h)) */
            for (int i = 0; i < n_embd; i++) g2[i] = 0;   /* accumula l'uscita MoE */
            for (int a = 0; a < n_used; a++) {
                int e = topi[a];
                WT ge = gex, ue = uex, de = dex;   /* esperti: d0=n_embd, d1=n_ff, [.., E] -> esperto = blocco contiguo */
                uint64_t per = (uint64_t)n_embd * n_ff;
                const uint8_t* ge_b = ge.base + (uint64_t)e * per * (ge.type == 0 ? 4 : 2);
                const uint8_t* ue_b = ue.base + (uint64_t)e * per * (ue.type == 0 ? 4 : 2);
                WT gt = ge, ut = ue; gt.base = ge_b; ut.base = ue_b;
                if (acc) { add_ch_e(acc[il].ge, h, n_embd, e); add_ch_e(acc[il].ue, h, n_embd, e); acc[il].gec[e]+=1; acc[il].uec[e]+=1;   /* NB: llama.cpp pesa per routing ma la scala NON torna -> teniamo h nudo (piu' vicino: 8x vs 500x) */
                           acc[il].he_samp_n[e] = samp_add(acc[il].he_samp + (size_t)e*acc[il].k_ex*n_embd, n_embd, acc[il].he_samp_n[e], acc[il].k_ex, h); }
                mm(&gt, h, g1, n_ff, n_embd);
                mm(&ut, h, &g1[n_ff], n_ff, n_embd);   /* riuso: up in g1[n_ff..] */
                for (int i = 0; i < n_ff; i++) { float silu = g1[i] / (1.0f + expf(-g1[i])); g1[i] = silu * g1[n_ff + i]; }
                if (acc) { add_ch_e(acc[il].de, g1, n_ff, e); acc[il].dec[e]+=1;
                           acc[il].de_samp_n[e] = samp_add(acc[il].de_samp + (size_t)e*acc[il].k_ex*n_ff, n_ff, acc[il].de_samp_n[e], acc[il].k_ex, g1); }
                uint64_t dper = (uint64_t)n_ff * n_embd;
                const uint8_t* de_b = de.base + (uint64_t)e * dper * (de.type == 0 ? 4 : 2);
                WT dt = de; dt.base = de_b;
                mm(&dt, g1, h, n_embd, n_ff);
                for (int i = 0; i < n_embd; i++) g2[i] += topw[a] * h[i];
            }
            for (int i = 0; i < n_embd; i++) x[i] += g2[i];
            if (getenv("LLM_INFER_TRACE")) { float mx=0; for(int i=0;i<n_embd;i++) if(fabsf(x[i])>mx) mx=fabsf(x[i]); fprintf(stderr, "[tr] t=%d l=%d dopo-MoE  max|x|=%.4g\n", t, il, mx); }
        }
    }
    if (dropped && strncmp(llm_warned_dropped, model_path, sizeof llm_warned_dropped) != 0) {
        snprintf(llm_warned_dropped, sizeof llm_warned_dropped, "%s", model_path);
        /* n_vocab=0 qui vuol dire `token_embd.weight` ASSENTE: tutti i token fuori vocabolario,
         * cioe' una calibrazione "riuscita" su zero token. Era il caso piu' silenzioso di tutti. */
        fprintf(stderr, "LLM PROFILE: %s: %ld token su %d FUORI vocabolario (n_vocab=%d) -> esclusi."
                        " I conteggi sono su %ld token%s\n", model_path, dropped, n_tokens, n_vocab,
                (long)n_tokens - dropped,
                ((long)n_tokens - dropped == 0 ? ", cioe' NESSUNO: profilo inaffidabile" : ""));
    }
    if (out_n_layer) *out_n_layer = n_layer;
    if (out_n_expert) *out_n_expert = n_expert;
    if (out_acc) { *out_acc = acc; acc = NULL; }   /* ownership al chiamante */
    if (out_nembd) *out_nembd = n_embd; if (out_nff) *out_nff = n_ff;

done:
    if (acc) { for (int l = 0; l < n_layer; l++) acc_free(&acc[l]); free(acc); }   /* solo se non restituito */
    free(x); free(h); free(q); free(k); free(v); free(att); free(g1); free(g2);
    free(kcache); free(vcache); free(logits); free(topi); free(topw);
    munmap((void*)map, (size_t)st.st_size); close(fd);
    llmcodec_gguf_meta_free(kv, nkv); llmcodec_gguf_free_info(&fi);
    return counts;
}


double* llmmodelio_routing_counts(const char* model_path, const int32_t* tokens, int32_t n_tokens,
                               int* out_n_layer, int* out_n_expert) {
    return run_forward(model_path, tokens, n_tokens, out_n_layer, out_n_expert, NULL, NULL, NULL, 0);
}

/* Accumula per canale gli in_sum2 di un layer (somma fra chunk), e CONCATENA (non somma) i campioni
 * Monte Carlo del chunk `a` nello spazio residuo di `d` (rispettando la capacita' k_sh/k_ex di `d`,
 * che e' LLMMODELIO_SAMP_K_* — cioe' molto piu' grande della capacita' per-chunk LLM_SAMP_CHUNK_* di `a`). */
static void acc_accum(LAcc* d, const LAcc* a, int ne, int nff, int E) {
    for (int i=0;i<ne;i++){ d->q[i]+=a->q[i]; d->k[i]+=a->k[i]; d->v[i]+=a->v[i]; d->o[i]+=a->o[i]; d->gi[i]+=a->gi[i]; }
    for (size_t i=0;i<(size_t)ne*E;i++){ d->ge[i]+=a->ge[i]; d->ue[i]+=a->ue[i]; }
    for (size_t i=0;i<(size_t)nff*E;i++){ d->de[i]+=a->de[i]; }
    d->qc+=a->qc; d->kc+=a->kc; d->vc+=a->vc; d->oc+=a->oc; d->gic+=a->gic;
    for (int e=0;e<E;e++){ d->gec[e]+=a->gec[e]; d->uec[e]+=a->uec[e]; d->dec[e]+=a->dec[e]; }
    for (int s=0; s<a->h_samp_n; s++) d->h_samp_n = samp_add(d->h_samp, ne, d->h_samp_n, d->k_sh, a->h_samp + (size_t)s*ne);
    for (int s=0; s<a->att_samp_n; s++) d->att_samp_n = samp_add(d->att_samp, ne, d->att_samp_n, d->k_sh, a->att_samp + (size_t)s*ne);
    for (int e=0; e<E; e++) {
        for (int s=0; s<a->he_samp_n[e]; s++)
            d->he_samp_n[e] = samp_add(d->he_samp + (size_t)e*d->k_ex*ne, ne, d->he_samp_n[e], d->k_ex, a->he_samp + (size_t)e*a->k_ex*ne + (size_t)s*ne);
        for (int s=0; s<a->de_samp_n[e]; s++)
            d->de_samp_n[e] = samp_add(d->de_samp + (size_t)e*d->k_ex*nff, nff, d->de_samp_n[e], d->k_ex, a->de_samp + (size_t)e*a->k_ex*nff + (size_t)s*nff);
    }
}

/* Calcola l'IMATRIX completa (in_sum2 + counts per gli 8 tensori/layer) sul corpus e la scrive in un
 * GGUF nello STESSO formato di llama-imatrix -> poi llmmodelio_imatrix_weights la usa per la quantizzazione. */
int llmmodelio_imatrix_compute(const char* model_path, const int32_t* tokens, int32_t n_tokens,
                            const char* out_path, int chunk) {
    if (!model_path || !tokens || n_tokens <= 0 || !out_path) return -1;
    if (chunk <= 0) chunk = 512;
    LAcc* tot = NULL; int nl=0, ne=0, nembd=0, nff=0, rc=0;
    for (int off=0; off<n_tokens; off+=chunk) {
        int m = (n_tokens-off)<chunk ? (n_tokens-off) : chunk;
        LAcc* a=NULL; int al=0,ae=0,aemb=0,aff=0;
        double* c = run_forward(model_path, tokens+off, m, &al, &ae, &a, &aemb, &aff, 1);
        if (!c || !a) { rc=-1; break; }
        if (!tot) { nl=al; ne=ae; nembd=aemb; nff=aff; tot=(LAcc*)calloc((size_t)nl,sizeof(LAcc)); for(int l=0;l<nl;l++) acc_init(&tot[l],nembd,nff,ne,LLMMODELIO_SAMP_K_SHARED,LLMMODELIO_SAMP_K_EXPERT); }
        for (int l=0;l<nl;l++) acc_accum(&tot[l], &a[l], nembd, nff, ne);
        free(c); for(int l=0;l<nl;l++) acc_free(&a[l]); free(a);
    }
    if (rc || !tot) { if (tot){for(int l=0;l<nl;l++)acc_free(&tot[l]);free(tot);} return -1; }
    /* scrittura: per layer 8 tensori -> in_sum2 + counts (16 per layer, come llama-imatrix) + 7 tensori
     * ".samples" (campioni Monte Carlo per llmmodelio_roundtrip_rel_cov: attn_q/k/v/output condividono h/att,
     * ffn_gate_exps/up_exps/down_exps sono per-esperto — vedi llmmodelio_infer.h). */
    int nT = nl * (16 + 7);
    LlmCodecGgufTensor* ts = (LlmCodecGgufTensor*)calloc((size_t)nT, sizeof(*ts));
    uint64_t* dstore = (uint64_t*)calloc((size_t)nT*2, sizeof(uint64_t));
    char (*names)[80] = (char(*)[80])calloc((size_t)nT, 80);
    float** datas = (float**)calloc((size_t)nT, sizeof(float*));
    LlmCodecGgufMetaKV mkv[2]; memset(mkv,0,sizeof mkv);
    mkv[0].key="general.architecture"; mkv[0].type=LLMCODEC_GGUF_MT_STRING; mkv[0].str="llm-imatrix";
    uint32_t nent=(uint32_t)nT; mkv[1].key="quantize.imatrix.entries_count"; mkv[1].type=LLMCODEC_GGUF_MT_U32; mkv[1].u32=nent;
    int k=0;
#define ADDT(NM, DATA, D0, D1, ND) do { \
        snprintf(names[k],80,"%s",NM); datas[k]=(float*)(DATA); ts[k].name=names[k]; ts[k].ggml_type=0; ts[k].ndim=(ND); \
        ts[k].dims=dstore+k*2; ((uint64_t*)ts[k].dims)[0]=(D0); if((ND)>1) ((uint64_t*)ts[k].dims)[1]=(D1); ts[k].data=datas[k]; k++; } while(0)
    for (int l=0; l<nl && rc==0; l++) {
        char nm[80]; float* t;
        snprintf(nm,80,"blk.%d.attn_q.weight.in_sum2",l); t=(float*)malloc((size_t)nembd*4); for(int i=0;i<nembd;i++) t[i]=tot[l].q[i]; ADDT(nm,t,nembd,0,1);
        snprintf(nm,80,"blk.%d.attn_q.weight.counts",l); t=(float*)malloc(4); t[0]=(float)tot[l].qc; ADDT(nm,t,1,0,1);
        snprintf(nm,80,"blk.%d.attn_k.weight.in_sum2",l); t=(float*)malloc((size_t)nembd*4); for(int i=0;i<nembd;i++) t[i]=tot[l].k[i]; ADDT(nm,t,nembd,0,1);
        snprintf(nm,80,"blk.%d.attn_k.weight.counts",l); t=(float*)malloc(4); t[0]=(float)tot[l].kc; ADDT(nm,t,1,0,1);
        snprintf(nm,80,"blk.%d.attn_v.weight.in_sum2",l); t=(float*)malloc((size_t)nembd*4); for(int i=0;i<nembd;i++) t[i]=tot[l].v[i]; ADDT(nm,t,nembd,0,1);
        snprintf(nm,80,"blk.%d.attn_v.weight.counts",l); t=(float*)malloc(4); t[0]=(float)tot[l].vc; ADDT(nm,t,1,0,1);
        snprintf(nm,80,"blk.%d.attn_output.weight.in_sum2",l); t=(float*)malloc((size_t)nembd*4); for(int i=0;i<nembd;i++) t[i]=tot[l].o[i]; ADDT(nm,t,nembd,0,1);
        snprintf(nm,80,"blk.%d.attn_output.weight.counts",l); t=(float*)malloc(4); t[0]=(float)tot[l].oc; ADDT(nm,t,1,0,1);
        snprintf(nm,80,"blk.%d.ffn_gate_inp.weight.in_sum2",l); t=(float*)malloc((size_t)nembd*4); for(int i=0;i<nembd;i++) t[i]=tot[l].gi[i]; ADDT(nm,t,nembd,0,1);
        snprintf(nm,80,"blk.%d.ffn_gate_inp.weight.counts",l); t=(float*)malloc(4); t[0]=(float)tot[l].gic; ADDT(nm,t,1,0,1);
        snprintf(nm,80,"blk.%d.ffn_gate_exps.weight.in_sum2",l); t=(float*)malloc((size_t)nembd*ne*4); memcpy(t,tot[l].ge,(size_t)nembd*ne*4); ADDT(nm,t,nembd,ne,2);
        snprintf(nm,80,"blk.%d.ffn_gate_exps.weight.counts",l); t=(float*)malloc((size_t)ne*4); for(int e=0;e<ne;e++) t[e]=(float)tot[l].gec[e]; ADDT(nm,t,ne,1,2);
        snprintf(nm,80,"blk.%d.ffn_up_exps.weight.in_sum2",l); t=(float*)malloc((size_t)nembd*ne*4); memcpy(t,tot[l].ue,(size_t)nembd*ne*4); ADDT(nm,t,nembd,ne,2);
        snprintf(nm,80,"blk.%d.ffn_up_exps.weight.counts",l); t=(float*)malloc((size_t)ne*4); for(int e=0;e<ne;e++) t[e]=(float)tot[l].uec[e]; ADDT(nm,t,ne,1,2);
        snprintf(nm,80,"blk.%d.ffn_down_exps.weight.in_sum2",l); t=(float*)malloc((size_t)nff*ne*4); memcpy(t,tot[l].de,(size_t)nff*ne*4); ADDT(nm,t,nff,ne,2);
        snprintf(nm,80,"blk.%d.ffn_down_exps.weight.counts",l); t=(float*)malloc((size_t)ne*4); for(int e=0;e<ne;e++) t[e]=(float)tot[l].dec[e]; ADDT(nm,t,ne,1,2);
        /* campioni Monte Carlo (llmmodelio_roundtrip_rel_cov): q/k/v/output condividono h/att (duplicati per nome,
         * costo trascurabile); gate/up condividono he_samp; down usa de_samp. Slot non riempiti = zero. */
        snprintf(nm,80,"blk.%d.attn_q.weight.samples",l); t=(float*)malloc((size_t)nembd*LLMMODELIO_SAMP_K_SHARED*4); memcpy(t,tot[l].h_samp,(size_t)nembd*LLMMODELIO_SAMP_K_SHARED*4); ADDT(nm,t,nembd,LLMMODELIO_SAMP_K_SHARED,2);
        snprintf(nm,80,"blk.%d.attn_k.weight.samples",l); t=(float*)malloc((size_t)nembd*LLMMODELIO_SAMP_K_SHARED*4); memcpy(t,tot[l].h_samp,(size_t)nembd*LLMMODELIO_SAMP_K_SHARED*4); ADDT(nm,t,nembd,LLMMODELIO_SAMP_K_SHARED,2);
        snprintf(nm,80,"blk.%d.attn_v.weight.samples",l); t=(float*)malloc((size_t)nembd*LLMMODELIO_SAMP_K_SHARED*4); memcpy(t,tot[l].h_samp,(size_t)nembd*LLMMODELIO_SAMP_K_SHARED*4); ADDT(nm,t,nembd,LLMMODELIO_SAMP_K_SHARED,2);
        snprintf(nm,80,"blk.%d.attn_output.weight.samples",l); t=(float*)malloc((size_t)nembd*LLMMODELIO_SAMP_K_SHARED*4); memcpy(t,tot[l].att_samp,(size_t)nembd*LLMMODELIO_SAMP_K_SHARED*4); ADDT(nm,t,nembd,LLMMODELIO_SAMP_K_SHARED,2);
        snprintf(nm,80,"blk.%d.ffn_gate_exps.weight.samples",l); t=(float*)malloc((size_t)nembd*LLMMODELIO_SAMP_K_EXPERT*ne*4); memcpy(t,tot[l].he_samp,(size_t)nembd*LLMMODELIO_SAMP_K_EXPERT*ne*4); ADDT(nm,t,nembd,LLMMODELIO_SAMP_K_EXPERT*ne,2);
        snprintf(nm,80,"blk.%d.ffn_up_exps.weight.samples",l); t=(float*)malloc((size_t)nembd*LLMMODELIO_SAMP_K_EXPERT*ne*4); memcpy(t,tot[l].he_samp,(size_t)nembd*LLMMODELIO_SAMP_K_EXPERT*ne*4); ADDT(nm,t,nembd,LLMMODELIO_SAMP_K_EXPERT*ne,2);
        snprintf(nm,80,"blk.%d.ffn_down_exps.weight.samples",l); t=(float*)malloc((size_t)nff*LLMMODELIO_SAMP_K_EXPERT*ne*4); memcpy(t,tot[l].de_samp,(size_t)nff*LLMMODELIO_SAMP_K_EXPERT*ne*4); ADDT(nm,t,nff,LLMMODELIO_SAMP_K_EXPERT*ne,2);
    }
    if (rc==0) rc = llmcodec_gguf_write(out_path, mkv, 2, ts, k, 32);
    for (int i=0;i<k;i++) free(datas[i]);
    free(ts); free(dstore); free(names); free(datas);
    for(int l=0;l<nl;l++) acc_free(&tot[l]); free(tot);
    return rc==0 ? 0 : -1;
}
