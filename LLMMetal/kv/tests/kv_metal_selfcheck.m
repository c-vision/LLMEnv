// kv_metal_selfcheck.m — verifica di correttezza (vs riferimento CPU fp32) e di banda per
// llmmetal_kv.h: scrittura pagina in-place + decode attention paged online-softmax.
//
// Uso: ./kv_metal_selfcheck [n_kv_heads] [head_dim] [n_q_heads] [context_len] [page_tokens]
// Default: shape stile Llama/Qwen piccolo (Hkv=8, D=128, Hq=32), contesto 8192, pagine da 16.
#import "../llmmetal_kv.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec/1e9; }

static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            // subnormal half -> normal float
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}
static uint16_t float_to_half(float x) {
    uint32_t f; memcpy(&f, &x, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign; // flush to zero (sufficiente per questo test)
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// Riferimento CPU: attenzione standard (softmax completo, non online) in fp32, stesso GQA.
static void cpu_reference_attention(const float* q, const float* K, const float* V,
                                     int n_q_heads, int n_kv_heads, int D, int T, float scale,
                                     float* out) {
    int group = n_q_heads / n_kv_heads;
    float* scores = (float*)malloc(sizeof(float) * T);
    for (int h = 0; h < n_q_heads; h++) {
        int hk = h / group;
        float maxs = -INFINITY;
        for (int t = 0; t < T; t++) {
            const float* kv = K + ((size_t)t * n_kv_heads + hk) * D;
            double s = 0;
            for (int d = 0; d < D; d++) s += (double)q[h*D+d] * kv[d];
            scores[t] = (float)(s * scale);
            if (scores[t] > maxs) maxs = scores[t];
        }
        double sum = 0;
        for (int t = 0; t < T; t++) { scores[t] = expf(scores[t] - maxs); sum += scores[t]; }
        for (int d = 0; d < D; d++) {
            double acc = 0;
            for (int t = 0; t < T; t++) {
                const float* vv = V + ((size_t)t * n_kv_heads + hk) * D;
                acc += (double)scores[t] * vv[d];
            }
            out[h*D+d] = (float)(acc / sum);
        }
    }
    free(scores);
}

int main(int argc, char** argv) {
    int n_kv_heads = argc > 1 ? atoi(argv[1]) : 8;
    int head_dim   = argc > 2 ? atoi(argv[2]) : 128;
    int n_q_heads  = argc > 3 ? atoi(argv[3]) : 32;
    int T          = argc > 4 ? atoi(argv[4]) : 8192;
    int page_tokens= argc > 5 ? atoi(argv[5]) : 16;
    int n_pages = (T + page_tokens - 1) / page_tokens;

    printf("shape: Hq=%d Hkv=%d D=%d T=%d page_tokens=%d n_pages=%d\n",
           n_q_heads, n_kv_heads, head_dim, T, page_tokens, n_pages);

    LlmMetalKvCtx ctx;
    if (llmmetal_kv_init(&ctx, n_pages, page_tokens, n_q_heads, n_kv_heads, head_dim) != 0) {
        fprintf(stderr, "init fallita\n"); return 1;
    }

    // dati deterministici
    srand(1234);
    float* Kf = (float*)malloc(sizeof(float) * (size_t)T * n_kv_heads * head_dim);
    float* Vf = (float*)malloc(sizeof(float) * (size_t)T * n_kv_heads * head_dim);
    for (size_t i = 0; i < (size_t)T * n_kv_heads * head_dim; i++) {
        Kf[i] = ((rand() % 2000) - 1000) / 1000.0f;
        Vf[i] = ((rand() % 2000) - 1000) / 1000.0f;
    }
    float* qf = (float*)malloc(sizeof(float) * n_q_heads * head_dim);
    for (int i = 0; i < n_q_heads * head_dim; i++) qf[i] = ((rand() % 2000) - 1000) / 1000.0f;

    // scrive token per token nella page pool (esercita davvero llmmetal_kv_write, non un
    // bulk-copy diretto nel buffer — cosi' il test copre il path realmente usato a runtime).
    uint16_t* ktok = (uint16_t*)malloc(sizeof(uint16_t) * n_kv_heads * head_dim);
    uint16_t* vtok = (uint16_t*)malloc(sizeof(uint16_t) * n_kv_heads * head_dim);
    double t_write0 = now_s();
    for (int t = 0; t < T; t++) {
        int page = t / page_tokens, off = t % page_tokens;
        for (int i = 0; i < n_kv_heads * head_dim; i++) {
            ktok[i] = float_to_half(Kf[(size_t)t*n_kv_heads*head_dim + i]);
            vtok[i] = float_to_half(Vf[(size_t)t*n_kv_heads*head_dim + i]);
        }
        if (llmmetal_kv_write(&ctx, page, off, ktok, vtok) != 0) { fprintf(stderr, "write fallita t=%d\n", t); return 1; }
    }
    double t_write1 = now_s();
    double write_bytes = (double)T * n_kv_heads * head_dim * 2 /*K+V*/ * sizeof(uint16_t);
    printf("write: %d token in %.3fs (%.1f token/s, %.2f GB/s aggregati incl. overhead dispatch)\n",
           T, t_write1 - t_write0, T/(t_write1-t_write0), write_bytes/(t_write1-t_write0)/1e9);

    // page table completa in ordine logico
    int* page_table = (int*)malloc(sizeof(int) * n_pages);
    for (int p = 0; p < n_pages; p++) page_table[p] = p;
    int last_page_len = T - (n_pages - 1) * page_tokens;

    uint16_t* qh = (uint16_t*)malloc(sizeof(uint16_t) * n_q_heads * head_dim);
    for (int i = 0; i < n_q_heads*head_dim; i++) qh[i] = float_to_half(qf[i]);
    uint16_t* out_gpu = (uint16_t*)malloc(sizeof(uint16_t) * n_q_heads * head_dim);

    // warmup + bench decode attention
    llmmetal_kv_decode_attention(&ctx, qh, page_table, n_pages, last_page_len, 1.0f/sqrtf((float)head_dim), out_gpu);
    int reps = 50;
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        llmmetal_kv_decode_attention(&ctx, qh, page_table, n_pages, last_page_len, 1.0f/sqrtf((float)head_dim), out_gpu);
    double dt = (now_s() - t0) / reps;
    double kv_bytes = (double)T * n_kv_heads * head_dim * 2 /*K+V*/ * sizeof(uint16_t);
    printf("decode_attention: %.3f ms/step, %.2f GB/s (KV letta: %.1f MB/step)\n",
           dt*1e3, kv_bytes/dt/1e9, kv_bytes/1e6);

    // riferimento CPU fp32 sugli stessi dati
    float* out_ref = (float*)malloc(sizeof(float) * n_q_heads * head_dim);
    cpu_reference_attention(qf, Kf, Vf, n_q_heads, n_kv_heads, head_dim, T, 1.0f/sqrtf((float)head_dim), out_ref);

    double max_abs_diff = 0, max_rel_diff = 0;
    for (int i = 0; i < n_q_heads * head_dim; i++) {
        float g = half_to_float(out_gpu[i]);
        float diff = fabsf(g - out_ref[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        float rel = diff / (fabsf(out_ref[i]) + 1e-3f);
        if (rel > max_rel_diff) max_rel_diff = rel;
    }
    printf("correttezza vs riferimento CPU fp32: max_abs_diff=%.5f max_rel_diff=%.5f %s\n",
           max_abs_diff, max_rel_diff, (max_abs_diff < 0.05 && max_rel_diff < 0.05) ? "PASS" : "FAIL");

    llmmetal_kv_free(&ctx);
    return (max_abs_diff < 0.05 && max_rel_diff < 0.05) ? 0 : 2;
}
