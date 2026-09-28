/* ============================================================================
 * Dominio imatrix + naming tensori: llm_mlx_imatrix_begin/energy_add/accumulate_rows/accumulate/write, tname, tname_gemma4.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <locale.h>
int llm_mlx_imatrix_begin(LlmMlxModel* m) {
    if (m == NULL) return -1;
    const ModelConfig* c = &m->cfg;
    /* Q2: ammessi anche i modelli DENSE (n_experts==0): in quel caso si accumula in un unico slot
     * esperto (0) dal FFN denso; l'unico requisito reale e' avere layer e una larghezza mid nota. */
    if (c->n_layers <= 0) return -1;
    const int e_eff = c->n_experts > 0 ? c->n_experts : 1;
    if (m->im.active) return 0; /* gia' attivo */
    m->im.gate_cols = c->hidden_size;
    /* MoE: intermedio per-esperto. Denso: intermedio del FFN. In entrambi i casi mai 0/hidden se
     * c'e' una misura reale (altrimenti il wiring denso si salta con un avviso esplicito). */
    if (c->n_experts > 0)
        m->im.down_cols = c->moe_intermediate_size > 0 ? c->moe_intermediate_size : c->hidden_size;
    else
        m->im.down_cols = c->intermediate_size > 0 ? c->intermediate_size
                                                   : (c->moe_intermediate_size > 0 ? c->moe_intermediate_size : c->hidden_size);
    if (c->n_experts <= 0 && c->intermediate_size <= 0) return -1; /* denso senza larghezza nota */
    m->im.n_layers = c->n_layers;
    m->im.n_experts = e_eff;
    m->im.top_k = c->top_k;
    size_t gate_n = (size_t)c->n_layers * e_eff * m->im.gate_cols;
    size_t down_n = (size_t)c->n_layers * e_eff * m->im.down_cols;
    size_t cnt_n = (size_t)c->n_layers * e_eff;
    m->im.gate = calloc(gate_n, sizeof(float));
    m->im.down = calloc(down_n, sizeof(float));
    m->im.cnt = calloc(cnt_n, sizeof(int));
    if (!m->im.gate || !m->im.down || !m->im.cnt) {
        free(m->im.gate); free(m->im.down); free(m->im.cnt);
        m->im.gate = m->im.down = NULL; m->im.cnt = NULL;
        return -1;
    }
    /* Q2-b: attention (riga expert=-1). Facoltativa: se il modello non espone n_heads*head_dim
     * validi, i buffer restano NULL e il wiring non accumula (nessun errore). */
    m->im.attn_o_cols = c->n_heads > 0 && c->head_dim > 0 ? c->n_heads * c->head_dim : 0;
    m->im.attn_qkv = (float*)calloc((size_t)c->n_layers * m->im.gate_cols, sizeof(float));
    m->im.attn_cnt = (int*)calloc((size_t)c->n_layers, sizeof(int));
    m->im.attn_o = m->im.attn_o_cols > 0 ? (float*)calloc((size_t)c->n_layers * m->im.attn_o_cols, sizeof(float)) : NULL;
    if (!m->im.attn_qkv || !m->im.attn_cnt || (m->im.attn_o_cols > 0 && !m->im.attn_o)) {
        free(m->im.gate); free(m->im.down); free(m->im.cnt);
        free(m->im.attn_qkv); free(m->im.attn_o); free(m->im.attn_cnt);
        m->im.gate = m->im.down = NULL; m->im.cnt = NULL;
        m->im.attn_qkv = NULL; m->im.attn_o = NULL; m->im.attn_cnt = NULL;
        return -1;
    }
    m->im.active = true;
    return 0;
}

/* Q2-b: energia per-colonna (Σ_righe x²) di x3d [1,N,cols] in dst[cols]. Nessun contatore qui
 * (lo tiene il chiamante). Read host: solo in calibrazione (m->im.active), mai in inferenza. */
void llm_mlx_imatrix_energy_add(LlmMlxModel* m, float* dst, int cols, mlx_array x3d, int n_tokens) {
    if (dst == NULL || cols <= 0 || n_tokens <= 0 || x3d.ctx == NULL) return;
    mlx_array xf = mlx_array_new();
    if (mlx_astype(&xf, x3d, MLX_FLOAT32, m->stream) != 0) { mlx_array_free(xf); return; }
    if (mlx_array_eval(xf) != 0) { mlx_array_free(xf); return; }
    const float* xd = mlx_array_data_float32(xf);
    if (xd != NULL) {
        for (int r = 0; r < n_tokens; r++) {
            const float* row = xd + (size_t)r * cols;
            for (int c = 0; c < cols; c++) dst[c] += row[c] * row[c];
        }
    }
    mlx_array_free(xf);
}

/* Imatrix: accumula per (layer, expert routato, colonna) l'energia dell'input gate/up (x3d:
 * [1,N,hidden]) e della riga mid del down (mid4d: [N,topk,1,moe_intermediate]). top_inds:
 * [N*topk] id esperti (solo quelli effettivamente routati). Fa eval + read host (calibration). */
/* Imatrix: accumula per (layer, expert routato, colonna) l'energia dell'input gate/up (x3d:
 * [1,N,hidden]) e della riga mid del down. `act_rows`: attivazione SwiGLU in ordine di RIGA
 * [n_rows,1,1,moe_intermediate], `row_expert[r]` = esperto della riga r, `row_src[r]` = riga
 * ORIGINALE token-major (j*top_k+k) -> token j = row_src/top_k. Cosi' il collector e' corretto
 * sia col path UNSORTED (row_expert=top_inds, row_src=identita') sia col path SORTED per-esperto
 * (row_expert=idxs, row_src=inverso dell'argsort) -- vedi §E2. */
void llm_mlx_imatrix_accumulate_rows(LlmMlxModel* m, int layer_idx, mlx_array x3d,
                                            mlx_array act_rows, const uint32_t* row_expert,
                                            const uint32_t* row_src, int n_rows) {
    ImatrixState* im = &m->im;
    if (!im->active || im->gate == NULL || layer_idx < 0 || layer_idx >= im->n_layers) return;
    if (!row_expert || !row_src || n_rows <= 0) return;
    if (x3d.ctx == NULL || act_rows.ctx == NULL) return; /* array vuoti: mlx-c lancerebbe */
    mlx_stream st = m->stream;
    mlx_array xf = mlx_array_new(), af = mlx_array_new();
    bool okx = mlx_astype(&xf, x3d, MLX_FLOAT32, st) == 0;
    bool oka = mlx_astype(&af, act_rows, MLX_FLOAT32, st) == 0;
    if (okx && oka) { okx = mlx_array_eval(xf) == 0; oka = mlx_array_eval(af) == 0; }
    const float* xd = okx ? mlx_array_data_float32(xf) : NULL;
    const float* ad = oka ? mlx_array_data_float32(af) : NULL;
    if (xd && ad) {
        const int H = im->gate_cols, M = im->down_cols, K = im->top_k > 0 ? im->top_k : 1, E = im->n_experts;
        const size_t lg = (size_t)layer_idx * E * H;
        const size_t ld = (size_t)layer_idx * E * M;
        const size_t lc = (size_t)layer_idx * E;
        for (int r = 0; r < n_rows; r++) {
            uint32_t e = row_expert[r];
            if (e >= (uint32_t)E) continue;
            int j = (int)(row_src[r] / (uint32_t)K);
            const float* xrow = xd + (size_t)j * H;
            const float* arow = ad + (size_t)r * M;
            float* g = im->gate + lg + (size_t)e * H;
            float* d = im->down + ld + (size_t)e * M;
            for (int col = 0; col < H; col++) g[col] += xrow[col] * xrow[col];
            for (int col = 0; col < M; col++) d[col] += arow[col] * arow[col];
            im->cnt[lc + e]++;
        }
    }
    mlx_array_free(xf); mlx_array_free(af);
}

/* Wrapper retro-compatibile (path UNSORTED token-major): row_expert = top_inds, row_src = identita'. */
void llm_mlx_imatrix_accumulate(LlmMlxModel* m, int layer_idx, mlx_array x3d,
                                       mlx_array mid4d, const uint32_t* top_inds, int n_tokens) {
    ImatrixState* im = &m->im;
    if (!im->active || !top_inds || n_tokens <= 0) return;
    int K = im->top_k > 0 ? im->top_k : 1;
    int n_rows = n_tokens * K;
    uint32_t* src = (uint32_t*)malloc((size_t)n_rows * sizeof(uint32_t));
    if (!src) return;
    for (int r = 0; r < n_rows; r++) src[r] = (uint32_t)r;
    llm_mlx_imatrix_accumulate_rows(m, layer_idx, x3d, mid4d, top_inds, src, n_rows);
    free(src);
}

/* Imatrix: scrive il report (testo, colonne tab-separate) per un futuro quantizer.
 * Per ogni (layer, expert): conteggio rotazioni; poi gate_cols valori (energia normalizzata
 * per il conteggio) e down_cols valori. Ritorna 0 ok / -1. */
int llm_mlx_imatrix_write(LlmMlxModel* m, const char* path) {
    ImatrixState* im = &m->im;
    if (!im->active || im->gate == NULL || path == NULL) return -1;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    /* Numeric locale C: il file deve essere machine-readable (virgola decimale romperebbe
     * un futuro parser). setlocale e' process-global: salva e ripristina subito dopo. */
    char* saved = setlocale(LC_NUMERIC, NULL);
    char saved_buf[64];
    if (saved) { snprintf(saved_buf, sizeof(saved_buf), "%s", saved); setlocale(LC_NUMERIC, "C"); }
    fprintf(f, "# imatrix routed-MoE (DS4-style): layer expert count <gate...> <down...>\n");
    fprintf(f, "# Q2: per i modelli DENSE l'esperto e' sempre 0 (energia del FFN denso).\n");
    fprintf(f, "# Q2-b: riga con expert=-1 = ATTENTION: <qkv (hidden)> <o (n_heads*head_dim)>.\n");
    const int H = im->gate_cols, M = im->down_cols, E = im->n_experts, L = im->n_layers;
    for (int l = 0; l < L; l++) {
        const size_t lg = (size_t)l * E * H, ld = (size_t)l * E * M, lc = (size_t)l * E;
        for (int e = 0; e < E; e++) {
            int c = im->cnt[lc + e];
            if (c <= 0) continue;
            fprintf(f, "%d %d %d", l, e, c);
            for (int col = 0; col < H; col++) fprintf(f, " %.6g", im->gate[lg + (size_t)e * H + col] / (float)c);
            for (int col = 0; col < M; col++) fprintf(f, " %.6g", im->down[ld + (size_t)e * M + col] / (float)c);
            fprintf(f, "\n");
        }
        /* Q2-b: attention (expert = -1). Emessa solo se il blocco attention ha accumulato: i
         * report dei modelli senza wiring attention (o MoE non coperti) restano identici a prima. */
        if (im->attn_cnt != NULL && im->attn_qkv != NULL && im->attn_cnt[l] > 0) {
            const int acnt = im->attn_cnt[l], O = im->attn_o_cols;
            fprintf(f, "%d -1 %d", l, acnt);
            for (int col = 0; col < H; col++) fprintf(f, " %.6g", im->attn_qkv[(size_t)l * H + col] / (float)acnt);
            if (im->attn_o != NULL && O > 0)
                for (int col = 0; col < O; col++) fprintf(f, " %.6g", im->attn_o[(size_t)l * O + col] / (float)acnt);
            fprintf(f, "\n");
        }
    }
    fclose(f);
    if (saved) setlocale(LC_NUMERIC, saved_buf);
    return 0;
}

/* DS4-Q1.b item 4 (roadmap): quantizzatore affine per-gruppo pesato dall'imatrix. Vedi
 * Knowledge/DS4-Antirez-DwarfStar-Analysis.md §3.5. Stesso formato affine di mlx_quantize
 * (dequant_i = code_i*scale+bias, code_i in [0, 2^bits-1]) -- verificato leggendo
 * libs/mlx-deps/mlx-src/mlx/backend/metal/kernels/quantized.h (kernel affine_quantize) e
 * mlx/backend/common/quantized.h (get_pack_factor/get_bytes_per_pack) -- cosi' un tensore
 * quantizzato con questa funzione resta caricabile dal motore esistente (mlx_dequantize/
 * mlx_quantized_matmul/gather_qmm) senza modifiche al path di inferenza.
 *
 * llm_mlx_naive_affine_group replica ESATTAMENTE la formula min/max di affine_quantize (stessa
 * scelta di segno/edge/at_zero) -- usata sia come punto di partenza della ricerca pesata sia
 * come termine di paragone "unweighted" nel test. */
/* Il quantizzatore pesato dall'imatrix (naive/error/weighted) e' ORA nel runtime
 * condiviso `llmcodec_moe_quant.h` (§0bis): qui si usa `llmcodec_moe_weighted_affine_group` (group-level). */

/* Costruisce "model.layers.{layer_idx}.{suffix}" in un buffer del chiamante
 * (era una lambda che restituiva std::string). */
void tname(const char* layer_prefix, int layer_idx, const char* suffix, char* out, size_t out_cap) {
    /* Costruzione MANUALE (niente snprintf/__vfprintf): tname e' chiamata per OGNI tensore di
     * OGNI layer a OGNI token e il printf-engine pesava ~3% del ciclo di decode nel profilo
     * `sample` (157 frame vs 0 di mlx_lm). Stesso formato e stessa troncatura di snprintf. */
    if (out == NULL || out_cap == 0) return;
    size_t n = 0;
    for (const char* p = layer_prefix; p != NULL && *p != '\0' && n + 1 < out_cap; ++p) out[n++] = *p;
    char tmp[12];
    int len = 0, v = layer_idx;
    if (v < 0) { if (n + 1 < out_cap) out[n++] = '-'; v = -v; }
    if (v == 0) tmp[len++] = '0';
    while (v > 0 && len < (int)sizeof(tmp)) { tmp[len++] = (char)('0' + v % 10); v /= 10; }
    while (len > 0 && n + 1 < out_cap) out[n++] = tmp[--len];
    if (n + 1 < out_cap) out[n++] = '.';
    for (const char* p = suffix; p != NULL && *p != '\0' && n + 1 < out_cap; ++p) out[n++] = *p;
    out[n] = '\0';
}

/* FASE 1b.4 (Gemma4): stesso schema di tname, ma con prefisso "language_model.model."
 * invece di "model." -- verificato sull'elenco tensori reale
 * (gemma-4-31b-it-4bit/model.safetensors.index.json): essendo un modello multimodale,
 * il sotto-modello testuale vive sotto "language_model.*", quello visione sotto
 * "vision_tower.*" (ignorato, questo motore non fa vision). Funzione separata invece
 * di parametrizzare tname stesso per non toccare tutti i call site esistenti delle
 * altre architetture. */
void tname_gemma4(int layer_idx, const char* suffix, char* out, size_t out_cap) {
    /* Stesso schema di tname (niente snprintf): prefisso costante + layer + '.' + suffisso. */
    static const char pfx[] = "language_model.model.layers.";
    if (out == NULL || out_cap == 0) return;
    size_t n = 0;
    for (const char* p = pfx; *p != '\0' && n + 1 < out_cap; ++p) out[n++] = *p;
    char tmp[12];
    int len = 0, v = layer_idx;
    if (v == 0) tmp[len++] = '0';
    while (v > 0 && len < (int)sizeof(tmp)) { tmp[len++] = (char)('0' + v % 10); v /= 10; }
    while (len > 0 && n + 1 < out_cap) out[n++] = tmp[--len];
    if (n + 1 < out_cap) out[n++] = '.';
    for (const char* p = suffix; *p != '\0' && n + 1 < out_cap; ++p) out[n++] = *p;
    out[n] = '\0';
}
