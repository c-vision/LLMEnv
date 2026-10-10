/* gmb_asr_pipeline.c — segmentazione a confini di silenzio (point 3), bookkeeping dello streaming
 * (point 2) e self-check complessivo del modulo. Logica pura, testabile headless. */
#include "gmb_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===== Segmentazione (point 3) ===== */
int gmb_asr_segment_split(const float* samples, int n, int seg_samples, int search_samples,
                          int* cuts, int max_cuts) {
    if (samples == NULL || seg_samples <= 0 || max_cuts <= 0) return 0;
    const int win = 320; /* 20ms @16k */
    int nc = 0;
    for (int target = seg_samples; target < n && nc < max_cuts; target += seg_samples) {
        int lo = target - search_samples, hi = target + search_samples;
        if (lo < win) lo = win;
        if (hi > n) hi = n;
        if (lo >= hi) break;
        int best = target; double best_e = 1e300;
        for (int c = lo; c < hi; c++) {
            double e = 0;
            for (int i = c - win; i < c; i++) e += (double)samples[i] * samples[i];
            if (e < best_e) { best_e = e; best = c; }
        }
        int prev = nc ? cuts[nc - 1] : 0;
        if (best > prev && best < n) cuts[nc++] = best;
    }
    return nc;
}

/* ===== Streaming bookkeeping (point 2) ===== */
void gmb_asr_stream_init(GmbAsrStream* s, int window_samples, int max_windows,
                         int uncommitted_tokens, int max_prefix_tokens) {
    memset(s, 0, sizeof *s);
    s->window_samples = window_samples;
    s->max_windows = max_windows > 0 ? max_windows : 1;
    s->uncommitted_tokens = uncommitted_tokens;
    s->max_prefix_tokens = max_prefix_tokens;
    s->first_window = 0;
}

void gmb_asr_stream_add_window(GmbAsrStream* s) {
    s->n_windows++;
    /* Eviction sliding: tieni solo le ultime `max_windows`. */
    if (s->n_windows > s->max_windows) { s->n_windows = s->max_windows; s->first_window++; s->evicted_windows++; }
}

void gmb_asr_stream_advance_prefix(GmbAsrStream* s, int fixed_tokens, int new_tokens, int total_tokens) {
    (void)fixed_tokens;
    int want = total_tokens > 0 ? total_tokens : (s->n_prefix + new_tokens);
    if (want > s->max_prefix_tokens) want = s->max_prefix_tokens;    /* capping del prefisso */
    s->n_prefix = want;
}

int gmb_asr_stream_prefix_len(const GmbAsrStream* s, int total_tokens) {
    int want = total_tokens > 0 ? total_tokens : s->n_prefix;
    return want > s->max_prefix_tokens ? s->max_prefix_tokens : want;
}

/* ===== Self-check ===== */
static int chk_q8(void* fo, int* fails) {
    int n = 256; float x[256], y[256]; int8_t q[256]; float sc[256 / GMB_ASR_Q8_BLK];
    for (int i = 0; i < n; i++) x[i] = sinf((float)i * 0.1f) * 3.0f;
    gmb_asr_q8_quantize(x, q, sc, n);
    gmb_asr_q8_dequantize(q, sc, y, n);
    double mxe = 0; for (int i = 0; i < n; i++) { double d = fabs((double)x[i] - y[i]); if (d > mxe) mxe = d; }
    /* max errore di quant a 8 bit <= mezza scala del blocco (amax/127). */
    int ok = mxe <= (3.0 / 127.0 + 1e-4);
    fprintf((FILE*)fo, "[asr] q8 round-trip: maxerr=%.5g %s\n", mxe, ok ? "OK" : "FAIL");
    if (!ok) (*fails)++;
    return ok;
}

static int chk_linear_bf16(void* fo, int* fails) {
    /* 4 righe x 8 in: confronto gmb_asr_linear_bf16 vs riferimento scalare. */
    int out = 4, in = 8, seq = 3;
    uint16_t W[32]; float x[24], b[4], y[12], ref[12];
    for (int i = 0; i < 32; i++) { float v = 0.1f * (i % 7) - 0.3f; uint32_t u; memcpy(&u, &v, 4); W[i] = (uint16_t)(u >> 16); }
    for (int i = 0; i < 24; i++) x[i] = 0.05f * (i % 11) - 0.2f;
    for (int i = 0; i < 4; i++) b[i] = 0.01f * i;
    gmb_asr_linear_bf16(y, x, W, b, seq, in, out);
    for (int s = 0; s < seq; s++)
        for (int o = 0; o < out; o++) {
            float sum = b[o];
            for (int k = 0; k < in; k++) { uint32_t u = (uint32_t)W[o * in + k] << 16; float w; memcpy(&w, &u, 4); sum += w * x[s * in + k]; }
            ref[s * out + o] = sum;
        }
    double mxe = 0; for (int i = 0; i < 12; i++) { double d = fabs((double)y[i] - ref[i]); if (d > mxe) mxe = d; }
    int ok = mxe < 1e-4;
    fprintf((FILE*)fo, "[asr] linear_bf16 vs scalare: maxerr=%.3g %s\n", mxe, ok ? "OK" : "FAIL");
    if (!ok) (*fails)++;
    return ok;
}

static int chk_mel(void* fo, int* fails) {
    int n = 16000; /* 1 secondo */
    float* s = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) s[i] = 0.3f * sinf(2.0f * 3.14159265f * 440.0f * (float)i / 16000.0f);
    int frames = 0; float* mel = gmb_asr_mel(s, n, &frames);
    /* atteso: n_frames_total = (16000+400-400)/160+1 = 101; meno l'ultimo = 100 */
    int ok = (mel != NULL && frames == 100);
    /* la normalizzazione (val+4)/4 non vincola a [0,1]: si verifica solo finitezza e non-degenerazione */
    if (ok) { double mn = 1e30, mx = -1e30; for (int i = 0; i < GMB_ASR_N_MEL * frames; i++) { if (!isfinite(mel[i])) { ok = 0; break; } if (mel[i] < mn) mn = mel[i]; if (mel[i] > mx) mx = mel[i]; } if (ok && !(mx > mn)) ok = 0; }
    fprintf((FILE*)fo, "[asr] mel 1s@440Hz: frames=%d (atteso 100) finito/non-degenere: %s\n", frames, ok ? "OK" : "FAIL");
    if (!ok) (*fails)++;
    free(mel); free(s);
    return ok;
}

static int chk_segment(void* fo, int* fails) {
    /* 2s di tono con un silenzio a ~1s: il taglio deve cadere nel silenzio. */
    int n = 32000, seg = 16000, search = 2000;
    float* s = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) s[i] = (i > 14500 && i < 15500) ? 0.0f : 0.4f * sinf(0.05f * (float)i);
    int cuts[8]; int nc = gmb_asr_segment_split(s, n, seg, search, cuts, 8);
    int ok = (nc == 1 && cuts[0] >= 14500 && cuts[0] <= 15500);
    fprintf((FILE*)fo, "[asr] segmentazione: cuts=%d primo=%d (atteso 1, in [14500,15500]): %s\n",
            nc, nc ? cuts[0] : -1, ok ? "OK" : "FAIL");
    if (!ok) (*fails)++;
    free(s);
    return ok;
}

static int chk_stream(void* fo, int* fails) {
    GmbAsrStream st; gmb_asr_stream_init(&st, 16000, 4, 5, 150);
    for (int i = 0; i < 10; i++) gmb_asr_stream_add_window(&st);
    int ok = (st.n_windows == 4 && st.evicted_windows == 6);
    gmb_asr_stream_advance_prefix(&st, 5, 32, 500);
    ok = ok && (gmb_asr_stream_prefix_len(&st, 500) == 150);
    fprintf((FILE*)fo, "[asr] streaming: n_windows=%d evicted=%lld prefix=%d (atteso 4/6/150): %s\n",
            st.n_windows, (long long)st.evicted_windows, gmb_asr_stream_prefix_len(&st, 500), ok ? "OK" : "FAIL");
    if (!ok) (*fails)++;
    return ok;
}

static float* mkf(size_t n, float seed) {
    float* a = (float*)malloc((n ? n : 1) * sizeof(float));
    if (!a) return NULL;
    for (size_t i = 0; i < n; i++) a[i] = 0.02f * sinf(seed + (float)i * 0.0013f);
    return a;
}

static int chk_encoder(void* fo, int* fails) {
    GmbAsrEncConfig cfg; memset(&cfg, 0, sizeof cfg);
    cfg.d_model = 8; cfg.layers = 1; cfg.heads = 1; cfg.head_dim = 8;
    cfg.ffn_dim = 16; cfg.output_dim = 8; cfg.n_window = 50; cfg.n_window_infer = 800; cfg.chunk_size = 100;
    GmbAsrEncoder enc; memset(&enc, 0, sizeof enc);
    enc.conv1_w = mkf((size_t)480 * 1 * 9, 1.0f); enc.conv1_b = mkf(480, 2.0f);
    enc.conv2_w = mkf((size_t)480 * 480 * 9, 3.0f); enc.conv2_b = mkf(480, 4.0f);
    enc.conv3_w = mkf((size_t)480 * 480 * 9, 5.0f); enc.conv3_b = mkf(480, 6.0f);
    enc.conv_out_w = mkf((size_t)cfg.d_model * 7680, 7.0f);
    GmbAsrEncLayer* L = &enc.layers[0];
    L->wq = mkf((size_t)8 * 8, 8.0f); L->bq = mkf(8, 9.0f);
    L->wk = mkf((size_t)8 * 8, 10.0f); L->bk = mkf(8, 11.0f);
    L->wv = mkf((size_t)8 * 8, 12.0f); L->bv = mkf(8, 13.0f);
    L->wo = mkf((size_t)8 * 8, 14.0f); L->bo = mkf(8, 15.0f);
    L->attn_norm_w = mkf(8, 16.0f); L->attn_norm_b = mkf(8, 17.0f);
    L->fc1 = mkf((size_t)16 * 8, 18.0f); L->bfc1 = mkf(16, 19.0f);
    L->fc2 = mkf((size_t)8 * 16, 20.0f); L->bfc2 = mkf(8, 21.0f);
    L->ffn_norm_w = mkf(8, 22.0f); L->ffn_norm_b = mkf(8, 23.0f);
    enc.ln_post_w = mkf(8, 24.0f); enc.ln_post_b = mkf(8, 25.0f);
    enc.proj1_w = mkf((size_t)8 * 8, 26.0f); enc.bproj1 = mkf(8, 27.0f);
    enc.proj2_w = mkf((size_t)8 * 8, 28.0f); enc.bproj2 = mkf(8, 29.0f);

    int frames = 1600; /* 16 chunk -> 208 token -> 2 finestre (0..103, 104..207) */
    float* mel = (float*)malloc((size_t)128 * frames * sizeof(float));
    for (int i = 0; i < 128 * frames; i++) mel[i] = 0.5f + 0.4f * sinf((float)i * 0.01f);

    int seq = 0; float* out = gmb_asr_encoder_forward(&cfg, &enc, mel, frames, &seq);
    int ok = (out != NULL && seq == 208);
    if (ok) for (size_t i = 0; i < (size_t)seq * cfg.output_dim; i++) if (!isfinite(out[i])) { ok = 0; break; }

    /* isolamento finestre: perturbando la mel del chunk 0, i token della finestra 1 (>=104) non cambiano. */
    if (ok) {
        float* mel2 = (float*)malloc((size_t)128 * frames * sizeof(float)); memcpy(mel2, mel, (size_t)128 * frames * sizeof(float));
        for (int m = 0; m < 128; m++) for (int t = 0; t < 100; t++) mel2[(size_t)m * frames + t] += 0.3f;
        int seq2 = 0; float* out2 = gmb_asr_encoder_forward(&cfg, &enc, mel2, frames, &seq2);
        if (out2 == NULL || seq2 != seq) ok = 0;
        else for (int i = 104; i < seq; i++) for (int d = 0; d < cfg.output_dim; d++) if (out[i * cfg.output_dim + d] != out2[i * cfg.output_dim + d]) ok = 0;
        free(out2); free(mel2);
    }
    fprintf((FILE*)fo, "[asr] encoder (conv+attn finestrata): seq=%d (atteso 208) finito + finestre isolate: %s\n", seq, ok ? "OK" : "FAIL");
    if (!ok) (*fails)++;
    free(out); free(mel);
    return ok;
}

static uint16_t* mkbf(size_t n, float seed) {
    uint16_t* a = (uint16_t*)malloc((n ? n : 1) * sizeof(uint16_t));
    if (!a) return NULL;
    for (size_t i = 0; i < n; i++) { float v = 0.05f * sinf(seed + (float)i * 0.017f); uint32_t u; memcpy(&u, &v, 4); a[i] = (uint16_t)(u >> 16); }
    return a;
}

static int chk_decoder(void* fo, int* fails) {
    GmbAsrDecConfig cfg; memset(&cfg, 0, sizeof cfg);
    cfg.hidden = 8; cfg.layers = 1; cfg.heads = 1; cfg.kv_heads = 1; cfg.head_dim = 8;
    cfg.intermediate = 16; cfg.vocab = 32; cfg.rms_eps = 1e-6f; cfg.rope_theta = 1e6f;
    GmbAsrDecoder d; memset(&d, 0, sizeof d);
    d.tok_embeddings = mkbf((size_t)cfg.vocab * cfg.hidden, 1.0f);
    d.norm = mkf(cfg.hidden, 2.0f);
    GmbAsrDecLayer* ly = &d.layers[0];
    ly->wq = mkbf((size_t)8 * 8, 3.0f); ly->wk = mkbf((size_t)8 * 8, 4.0f);
    ly->wv = mkbf((size_t)8 * 8, 5.0f); ly->wo = mkbf((size_t)8 * 8, 6.0f);
    ly->q_norm = mkf(8, 7.0f); ly->k_norm = mkf(8, 8.0f);
    ly->input_norm = mkf(8, 9.0f); ly->post_attn_norm = mkf(8, 10.0f);
    ly->gate = mkbf((size_t)16 * 8, 11.0f); ly->up = mkbf((size_t)16 * 8, 12.0f); ly->down = mkbf((size_t)8 * 16, 13.0f);

    int H = cfg.hidden; float embeds[4 * 8];
    for (int i = 0; i < 32; i++) embeds[i] = 0.1f * sinf((float)i * 0.3f);

    float logits_step[32], logits_full[32];
    /* A: prefill 3 + step 1 */
    GmbAsrKv kva; gmb_asr_kv_init(&kva, cfg.layers, cfg.kv_heads * cfg.head_dim, 16);
    gmb_asr_decoder_forward(&cfg, &d, &kva, embeds, 3, NULL);
    gmb_asr_decoder_forward(&cfg, &d, &kva, embeds + 3 * H, 1, logits_step);
    /* B: prefill 4 in un colpo */
    GmbAsrKv kvb; gmb_asr_kv_init(&kvb, cfg.layers, cfg.kv_heads * cfg.head_dim, 16);
    gmb_asr_decoder_forward(&cfg, &d, &kvb, embeds, 4, logits_full);

    double mxe = 0; for (int i = 0; i < cfg.vocab; i++) { double e = fabs((double)logits_step[i] - logits_full[i]); if (e > mxe) mxe = e; }
    gmb_asr_kv_free(&kva); gmb_asr_kv_free(&kvb);
    int ok = mxe < 1e-4;
    fprintf((FILE*)fo, "[asr] decoder KV cache (prefill+step vs prefill unico): maxerr=%.3g %s\n", mxe, ok ? "OK" : "FAIL");
    if (!ok) (*fails)++;
    free((void*)d.tok_embeddings); free((void*)d.norm);
    free((void*)ly->wq); free((void*)ly->wk); free((void*)ly->wv); free((void*)ly->wo);
    free((void*)ly->q_norm); free((void*)ly->k_norm); free((void*)ly->input_norm); free((void*)ly->post_attn_norm);
    free((void*)ly->gate); free((void*)ly->up); free((void*)ly->down);
    return ok;
}

static int chk_loader(void* fo, int* fails) {
    GmbAsrModel* m = gmb_asr_load_model("/nonexistent/qwen3-asr-xyz");
    int ok = (m == NULL);          /* dir inesistente -> NULL, nessun crash */
    if (m) gmb_asr_model_free(m);
    fprintf((FILE*)fo, "[asr] loader dir inesistente -> NULL: %s\n", ok ? "OK" : "FAIL");
    if (!ok) (*fails)++;
    return ok;
}

int gmb_asr_selfcheck(void* out_stream) {
    void* fo = out_stream ? out_stream : stdout;
    int fails = 0;
    fprintf((FILE*)fo, "=== GMB ASR self-check ===\n");
    chk_q8(fo, &fails);
    chk_linear_bf16(fo, &fails);
    chk_mel(fo, &fails);
    chk_segment(fo, &fails);
    chk_stream(fo, &fails);
    chk_encoder(fo, &fails);
    chk_decoder(fo, &fails);
    chk_loader(fo, &fails);
    fprintf((FILE*)fo, "=== %s ===\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
