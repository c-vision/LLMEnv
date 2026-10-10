/* gmb_asr_transcribe.c — orchestrazione ASR: mel -> encoder -> prompt -> decoder greedy.
 * Port di antirez/qwen-asr `qwen_asr.c` (MIT). Il prompt ASR usa le costanti di token
 * (system/user/audio_start/audio_end/assistant) e inietta l'output dell'encoder nelle posizioni
 * audio. Decodifica id->testo delegata al tokenizer del chiamante (GmbAsrPieceFn).
 * Streaming fedele a qwen-asr: chunk 2s + cache delle finestre encoder (8s, immutabili) +
 * prefix rollback sui token raw gia' generati. */
#include "gmb_asr.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* [<|im_start|>, "system", "\n"] */
static const int PREFIX_HEAD[] = { 151644, 8948, 198 };
/* [<|im_end|>, "\n", <|im_start|>, "user", "\n", <|audio_start|>] */
static const int PREFIX_TAIL[] = { 151645, 198, 151644, 872, 198, 151669 };
/* [<|audio_end|>, <|im_end|>, "\n", <|im_start|>, "assistant", "\n"] */
static const int SUFFIX_BASE[] = { 151670, 151645, 198, 151644, 77091, 198 };
#define PPH 3
#define PPT 6
#define PSB 6
#define T_ENDOFTEXT 151643
#define T_IM_END   151645
#define T_ASR_TEXT 151704

static int argmax_f32(const float* x, int n) { int b = 0; float bv = x[0]; for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; b = i; } return b; }

static void embed_one(float* dst, const uint16_t* tok_emb, int id, int dim) {
    const uint16_t* src = tok_emb + (size_t)id * dim;
    for (int i = 0; i < dim; i++) { uint32_t u = (uint32_t)src[i] << 16; float f; memcpy(&f, &u, 4); dst[i] = f; }
}

/* Genera fino a `max_new` token dal contesto
 *   [PREFIX_HEAD][prompt][PREFIX_TAIL][enc_output][SUFFIX_BASE][force][past_tokens]
 * `past_tokens` sono i token RAW gia' generati (language + <asr_text> + testo), come in qwen-asr:
 * la generazione riparte dopo di essi. Ritorna il numero di token generati (>=0) o -1. */
static int asr_generate(const GmbAsrDecConfig* dcfg, const GmbAsrDecoder* dec,
                        const float* enc_out, int enc_seq,
                        const int* prompt_tokens, int n_prompt,
                        const int* force_tokens, int n_force,
                        const int* past_tokens, int n_past,
                        int max_new, int* out_tokens, int out_cap) {
    const int dim = dcfg->hidden;
    int prefix_len = PPH + n_prompt + PPT;
    int suffix_len = PSB + n_force;
    int total_seq = prefix_len + enc_seq + suffix_len + n_past;
    if (total_seq <= 0 || max_new <= 0) return 0;

    float* input_embeds = (float*)malloc((size_t)total_seq * dim * sizeof(float));
    float* tmp_embed = (float*)malloc((size_t)dim * sizeof(float));
    float* logits = (float*)malloc((size_t)dcfg->vocab * sizeof(float));
    GmbAsrKv kv;
    gmb_asr_kv_init(&kv, dcfg->layers, dcfg->kv_heads * dcfg->head_dim, total_seq + max_new + 16);
    if (!input_embeds || !tmp_embed || !logits || !kv.k) {
        free(input_embeds); free(tmp_embed); free(logits); gmb_asr_kv_free(&kv); return -1;
    }

    int off = 0;
    for (int i = 0; i < PPH; i++) embed_one(input_embeds + (size_t)off++ * dim, dec->tok_embeddings, PREFIX_HEAD[i], dim);
    for (int i = 0; i < n_prompt; i++) embed_one(input_embeds + (size_t)off++ * dim, dec->tok_embeddings, prompt_tokens[i], dim);
    for (int i = 0; i < PPT; i++) embed_one(input_embeds + (size_t)off++ * dim, dec->tok_embeddings, PREFIX_TAIL[i], dim);
    for (int i = 0; i < enc_seq; i++) memcpy(input_embeds + (size_t)(prefix_len + i) * dim, enc_out + (size_t)i * dim, (size_t)dim * sizeof(float));
    int suffix_off = prefix_len + enc_seq;
    for (int i = 0; i < PSB; i++) embed_one(input_embeds + (size_t)(suffix_off + i) * dim, dec->tok_embeddings, SUFFIX_BASE[i], dim);
    for (int i = 0; i < n_force; i++) embed_one(input_embeds + (size_t)(suffix_off + PSB + i) * dim, dec->tok_embeddings, force_tokens[i], dim);
    int text_off = suffix_off + suffix_len;
    for (int i = 0; i < n_past; i++) embed_one(input_embeds + (size_t)(text_off + i) * dim, dec->tok_embeddings, past_tokens[i], dim);

    int prefill_len = total_seq - 1;
    if (gmb_asr_decoder_forward(dcfg, dec, &kv, input_embeds, prefill_len, NULL) != 0 ||
        gmb_asr_decoder_forward(dcfg, dec, &kv, input_embeds + (size_t)prefill_len * dim, 1, logits) != 0) {
        free(input_embeds); free(tmp_embed); free(logits); gmb_asr_kv_free(&kv); return -1;
    }
    free(input_embeds);
    int token = argmax_f32(logits, dcfg->vocab);

    int n = 0;
    for (int gen = 0; gen < max_new; gen++) {
        if (token == T_ENDOFTEXT || token == T_IM_END) break;
        if (n < out_cap) out_tokens[n++] = token;
        embed_one(tmp_embed, dec->tok_embeddings, token, dim);
        if (gmb_asr_decoder_forward(dcfg, dec, &kv, tmp_embed, 1, logits) != 0) break;
        token = argmax_f32(logits, dcfg->vocab);
    }
    free(tmp_embed); free(logits); gmb_asr_kv_free(&kv);
    return n;
}

char* gmb_asr_transcribe_segment(const GmbAsrEncConfig* ecfg, const GmbAsrEncoder* enc,
                                 const GmbAsrDecConfig* dcfg, const GmbAsrDecoder* dec,
                                 const float* samples, int n_samples,
                                 const int* prompt_tokens, int n_prompt,
                                 const int* force_tokens, int n_force,
                                 const int* past_tokens, int n_past,
                                 GmbAsrPieceFn decode_fn, void* decode_ctx, int* out_text_tokens) {
    if (out_text_tokens) *out_text_tokens = 0;
    if (!ecfg || !enc || !dcfg || !dec || !samples || n_samples <= 0) return NULL;

    int mel_frames = 0;
    float* mel = gmb_asr_mel(samples, n_samples, &mel_frames);
    if (!mel) return NULL;
    int enc_seq = 0;
    float* enc_out = gmb_asr_encoder_forward(ecfg, enc, mel, mel_frames, &enc_seq);
    free(mel);
    if (!enc_out) return NULL;

    int cap_gen = 2048;
    int* gen = (int*)malloc((size_t)cap_gen * sizeof(int));
    int ng = gen ? asr_generate(dcfg, dec, enc_out, enc_seq, prompt_tokens, n_prompt,
                               force_tokens, n_force, past_tokens, n_past, cap_gen, gen, cap_gen) : -1;
    free(enc_out);
    if (ng < 0) { free(gen); return NULL; }

    /* sequenza raw = past + generati; il testo e' dopo <asr_text> (assente se linguaggio forzato) */
    int total = n_past + ng;
    int* raw = (int*)malloc((size_t)(total ? total : 1) * sizeof(int));
    if (!raw) { free(gen); return NULL; }
    for (int i = 0; i < n_past; i++) raw[i] = past_tokens[i];
    for (int i = 0; i < ng; i++) raw[n_past + i] = gen[i];
    free(gen);

    int start = 0;
    if (n_force <= 0) for (int i = 0; i < total; i++) if (raw[i] == T_ASR_TEXT) { start = i + 1; break; }

    size_t cp = 4096, len = 0; char* text = (char*)malloc(cp); text[0] = 0; int n_text = 0;
    for (int i = start; i < total; i++) {
        if (decode_fn) {
            const char* piece = decode_fn(raw[i], decode_ctx);
            if (piece) {
                size_t pl = strlen(piece);
                if (len + pl + 1 > cp) { while (len + pl + 1 > cp) cp *= 2; text = (char*)realloc(text, cp); }
                memcpy(text + len, piece, pl); len += pl; text[len] = 0; n_text++;
            }
        }
    }
    free(raw);

    size_t r = strlen(text); while (r > 0 && isspace((unsigned char)text[r - 1])) text[--r] = 0;
    char* s = text; while (*s && isspace((unsigned char)*s)) s++;
    if (s != text) memmove(text, s, strlen(s) + 1);
    if (out_text_tokens) *out_text_tokens = n_text;
    return text;
}

char* gmb_asr_transcribe(const GmbAsrEncConfig* ecfg, const GmbAsrEncoder* enc,
                         const GmbAsrDecConfig* dcfg, const GmbAsrDecoder* dec,
                         const float* samples, int n_samples,
                         const int* prompt_tokens, int n_prompt,
                         const int* force_tokens, int n_force,
                         GmbAsrPieceFn decode_fn, void* decode_ctx,
                         float segment_sec, float search_sec, int* out_text_tokens) {
    if (out_text_tokens) *out_text_tokens = 0;
    int seg = (int)(segment_sec * GMB_ASR_SR);
    if (segment_sec <= 0.0f || n_samples <= seg)
        return gmb_asr_transcribe_segment(ecfg, enc, dcfg, dec, samples, n_samples,
                                          prompt_tokens, n_prompt, force_tokens, n_force, NULL, 0, decode_fn, decode_ctx, out_text_tokens);
    int cuts[256];
    int nc = gmb_asr_segment_split(samples, n_samples, seg, (int)(search_sec * GMB_ASR_SR), cuts, 256);
    size_t cap = 4096, len = 0; char* out = (char*)malloc(cap); out[0] = 0; int tot = 0, start = 0;
    for (int i = 0; i <= nc; i++) {
        int end = (i < nc) ? cuts[i] : n_samples;
        if (end <= start) continue;
        int nt = 0;
        char* s = gmb_asr_transcribe_segment(ecfg, enc, dcfg, dec, samples + start, end - start,
                                             prompt_tokens, n_prompt, force_tokens, n_force, NULL, 0, decode_fn, decode_ctx, &nt);
        start = end;
        if (!s) continue;
        size_t sl = strlen(s);
        if (sl) {
            size_t need = len + sl + 2; if (need > cap) { while (need > cap) cap *= 2; out = (char*)realloc(out, cap); }
            if (len > 0) out[len++] = ' ';
            memcpy(out + len, s, sl); len += sl; out[len] = 0; tot += nt;
        }
        free(s);
    }
    if (out_text_tokens) *out_text_tokens = tot;
    return out;
}

/* ===== Streaming (cache finestre encoder + rollback) ===== */

typedef struct { int64_t start_sample; int n_samples; int seq_len; float* enc_output; } AsrEncWin;

#define S_MAX_ENC_WINDOWS   4
#define S_MAX_PREFIX_TOKENS 150
#define S_MAX_REPEAT_RUN    12
#define S_OVERLAP_MAX       48
#define S_OVERLAP_MIN       4
#define S_UNFIXED_CHUNKS    2
#define S_MAX_NEW_TOKENS    32

static int enc_span(const GmbAsrEncConfig* ecfg, const GmbAsrEncoder* enc,
                    const float* samples, int n_samples, int* out_seq, float** out) {
    *out = NULL; *out_seq = 0;
    if (n_samples <= 0) return 0;
    int mel_frames = 0;
    float* mel = gmb_asr_mel(samples, n_samples, &mel_frames);
    if (!mel) return -1;
    int seq = 0;
    float* e = gmb_asr_encoder_forward(ecfg, enc, mel, mel_frames, &seq);
    free(mel);
    if (!e || seq <= 0) { free(e); return -1; }
    *out = e; *out_seq = seq;
    return 0;
}

char* gmb_asr_transcribe_stream(const GmbAsrEncConfig* ecfg, const GmbAsrEncoder* enc,
                                const GmbAsrDecConfig* dcfg, const GmbAsrDecoder* dec,
                                const float* samples, int n_samples,
                                const int* prompt_tokens, int n_prompt,
                                const int* force_tokens, int n_force,
                                GmbAsrPieceFn decode_fn, void* decode_ctx,
                                int rollback_tokens, float chunk_sec,
                                GmbAsrStreamEmit emit, void* emit_ctx) {
    if (!ecfg || !enc || !dcfg || !dec || !samples || n_samples <= 0) return NULL;
    const int dim = dcfg->hidden;
    int chunk_samples = (int)((chunk_sec > 0.0f ? chunk_sec : 2.0f) * GMB_ASR_SR);
    if (chunk_samples <= 0) chunk_samples = GMB_ASR_SR * 2;
    if (rollback_tokens < 0) rollback_tokens = 5;
    int enc_win_frames = ecfg->n_window_infer;
    if (enc_win_frames < 100) enc_win_frames = 100;
    if (enc_win_frames > 800) enc_win_frames = 800;
    int enc_win_samples = enc_win_frames * GMB_ASR_HOP;

    int* raw = (int*)malloc(8192 * sizeof(int)); int n_raw = 0, raw_cap = 8192;
    int* stable = (int*)malloc(8192 * sizeof(int)); int n_stable = 0, stable_cap = 8192;
    int* emitted = (int*)malloc(8192 * sizeof(int)); int n_emitted = 0, emitted_cap = 8192;
    size_t rcap = 4096, rlen = 0; char* result = (char*)malloc(rcap);
    AsrEncWin* cache = NULL; int n_cache = 0, cache_start = 0, cache_cap = 0, cached_seq = 0;
    int64_t next_win = 0;
    if (!raw || !stable || !emitted || !result) { free(raw); free(stable); free(emitted); free(result); return NULL; }
    result[0] = 0;

    int chunk_idx = 0;
    int64_t cursor = 0;
    while (cursor < n_samples) {
        cursor += chunk_samples; if (cursor > n_samples) cursor = n_samples;
        int is_final = (cursor >= n_samples);
        int64_t full_end = (cursor / enc_win_samples) * (int64_t)enc_win_samples;

        /* --- encoder: completa le finestre allineate [next_win, full_end), poi la coda parziale --- */
        int enc_failed = 0;
        while (next_win < full_end) {
            int ws_seq = 0; float* we = NULL;
            if (enc_span(ecfg, enc, samples + next_win, enc_win_samples, &ws_seq, &we) != 0) { enc_failed = 1; break; }
            if (n_cache == cache_cap) {
                cache_cap = cache_cap ? cache_cap * 2 : 8;
                cache = (AsrEncWin*)realloc(cache, (size_t)cache_cap * sizeof(AsrEncWin));
            }
            cache[n_cache].start_sample = next_win; cache[n_cache].n_samples = enc_win_samples;
            cache[n_cache].seq_len = ws_seq; cache[n_cache].enc_output = we; n_cache++;
            cached_seq += ws_seq;
            next_win += enc_win_samples;
        }
        float* partial = NULL; int partial_seq = 0;
        if (!enc_failed && full_end < cursor) {
            if (enc_span(ecfg, enc, samples + full_end, (int)(cursor - full_end), &partial_seq, &partial) != 0) { partial = NULL; enc_failed = 1; }
        }
        if (enc_failed) { free(partial); chunk_idx++; continue; }

        while (n_cache - cache_start > S_MAX_ENC_WINDOWS) {
            cached_seq -= cache[cache_start].seq_len;
            free(cache[cache_start].enc_output); cache[cache_start].enc_output = NULL; cache_start++;
        }
        int enc_seq = cached_seq + partial_seq;
        if (enc_seq <= 0) { free(partial); chunk_idx++; continue; }
        float* enc_out = (float*)malloc((size_t)enc_seq * dim * sizeof(float));
        if (!enc_out) { free(partial); chunk_idx++; continue; }
        int eo = 0;
        for (int i = cache_start; i < n_cache; i++) { memcpy(enc_out + (size_t)eo * dim, cache[i].enc_output, (size_t)cache[i].seq_len * dim * sizeof(float)); eo += cache[i].seq_len; }
        if (partial_seq > 0) memcpy(enc_out + (size_t)eo * dim, partial, (size_t)partial_seq * dim * sizeof(float));
        free(partial);

        /* --- prefix rollback: token raw precedenti meno gli ultimi `rollback` --- */
        int n_prefix_full = 0, n_prefix = 0, prefix_off = 0;
        if (chunk_idx >= S_UNFIXED_CHUNKS && n_raw > 0) {
            n_prefix_full = n_raw - rollback_tokens; if (n_prefix_full < 0) n_prefix_full = 0;
            n_prefix = n_prefix_full;
            if (n_prefix > S_MAX_PREFIX_TOKENS) { n_prefix = S_MAX_PREFIX_TOKENS; prefix_off = n_prefix_full - n_prefix; }
        }

        int* gen = (int*)malloc((size_t)S_MAX_NEW_TOKENS * sizeof(int));
        if (!gen) { free(enc_out); chunk_idx++; continue; }
        int ng = asr_generate(dcfg, dec, enc_out, enc_seq, prompt_tokens, n_prompt, force_tokens, n_force,
                              raw + prefix_off, n_prefix, S_MAX_NEW_TOKENS, gen, S_MAX_NEW_TOKENS);
        free(enc_out);
        if (ng < 0) { free(gen); chunk_idx++; continue; }

        /* soppressione run ripetuti (degenerazione) */
        int prev_tok = -1, prev_run = 0, dropped = 0;
        if (n_prefix_full > 0) {
            prev_tok = raw[n_prefix_full - 1]; prev_run = 1;
            for (int j = n_prefix_full - 2; j >= 0 && raw[j] == prev_tok && prev_run < S_MAX_REPEAT_RUN; j--) prev_run++;
        }
        int outn = 0;
        for (int i = 0; i < ng; i++) {
            int t = gen[i], sup = 0;
            if (t == prev_tok) { prev_run++; if (prev_run > S_MAX_REPEAT_RUN) sup = 1; }
            else { prev_tok = t; prev_run = 1; }
            if (sup) { dropped++; continue; }
            gen[outn++] = t;
        }
        ng = outn;

        int need = n_prefix_full + ng;
        if (need > raw_cap) { while (need > raw_cap) raw_cap *= 2; raw = (int*)realloc(raw, (size_t)raw_cap * sizeof(int)); }
        memcpy(raw + n_prefix_full, gen, (size_t)ng * sizeof(int)); n_raw = need;
        free(gen);

        /* --- regione testo (dopo <asr_text>) e candidato stabile --- */
        int text_start = 0;
        if (n_force <= 0) for (int i = 0; i < n_raw; i++) if (raw[i] == T_ASR_TEXT) { text_start = i + 1; break; }
        int n_text = n_raw - text_start;
        int cand_len = 0;
        if (is_final) cand_len = n_text;
        else if (chunk_idx >= S_UNFIXED_CHUNKS) {
            cand_len = n_text - rollback_tokens;
            if (cand_len <= 0 && n_text > 0) cand_len = n_text - 1;
            if (cand_len < 0) cand_len = 0;
        }
        int* cand = raw + text_start;

        if (cand_len > stable_cap) { while (cand_len > stable_cap) stable_cap *= 2; stable = (int*)realloc(stable, (size_t)stable_cap * sizeof(int)); }
        int lcp = 0;
        while (lcp < n_stable && lcp < cand_len && stable[lcp] == cand[lcp]) lcp++;
        for (int i = lcp; i < cand_len; i++) stable[i] = cand[i];

        int emit_start = lcp;
        if (emit_start < cand_len && n_emitted > 0) {
            int max_ov = cand_len - emit_start;
            if (max_ov > n_emitted) max_ov = n_emitted;
            if (max_ov > S_OVERLAP_MAX) max_ov = S_OVERLAP_MAX;
            for (int k = max_ov; k >= S_OVERLAP_MIN; k--)
                if (memcmp(emitted + (n_emitted - k), cand + emit_start, (size_t)k * sizeof(int)) == 0) { emit_start += k; break; }
        }

        for (int i = emit_start; i < cand_len; i++) {
            int tok = cand[i];
            if (decode_fn) {
                const char* piece = decode_fn(tok, decode_ctx);
                if (piece) {
                    size_t pl = strlen(piece);
                    if (rlen + pl + 1 > rcap) { while (rlen + pl + 1 > rcap) rcap *= 2; result = (char*)realloc(result, rcap); }
                    memcpy(result + rlen, piece, pl); rlen += pl; result[rlen] = 0;
                    if (emit) emit(piece, (int)pl, emit_ctx);
                }
            }
            if (n_emitted < emitted_cap) emitted[n_emitted++] = tok;
        }
        n_stable = cand_len;
        chunk_idx++;
    }

    for (int i = cache_start; i < n_cache; i++) free(cache[i].enc_output);
    free(cache); free(raw); free(stable); free(emitted);

    size_t len = strlen(result);
    while (len > 0 && isspace((unsigned char)result[len - 1])) result[--len] = 0;
    char* s = result; while (*s && isspace((unsigned char)*s)) s++;
    if (s != result) memmove(result, s, strlen(s) + 1);
    return result;
}
