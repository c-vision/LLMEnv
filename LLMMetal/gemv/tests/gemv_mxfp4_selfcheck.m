// gemv_mxfp4_selfcheck.m — verifica di CORRETTEZZA (vs riferimento CPU) e banda per il kernel
// Metal GEMV MXFP4-native (llmmetal_gemv_mxfp4.h, kernel #36).
//
// Genera pesi MXFP4 sintetici (nibble e2m1 random + scale e8m0 random) e un vettore x half,
// esegue il kernel e confronta con un riferimento CPU che dequantizza GLI STESSI dati packed
// (valore = e8m0(byte) * e2m1(nibble), dot con x). Questo valida la CORRETTEZZA del kernel
// (dequant + riduzione) sull'effettivo layout byte che il kernel legge, contro un riferimento
// indipendente che replica ESATTAMENTE le espressioni di MLX (fp4.h / fp8.h).
//
// Uso: ./gemv_mxfp4_selfcheck [H] [M] [group_size]
// Default: shape expert-like gpt-oss (H=2880, M=2880, group_size=32).
#import "../llmmetal_gemv_mxfp4.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec/1e9; }

static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, f;
    if (exp == 0) { f = mant == 0 ? sign : ({ uint32_t e=1,m=mant; while(!(m&0x400)){m<<=1;e--;} sign|((e+112)<<23)|((m&0x3FF)<<13); }); }
    else if (exp == 0x1F) f = sign | 0x7F800000 | (mant << 13);
    else f = sign | ((exp + 112) << 23) | (mant << 13);
    float out; memcpy(&out, &f, 4); return out;
}
static uint16_t float_to_half(float x) {
    uint32_t f; memcpy(&f, &x, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// Riferimento CPU: replica ESATTA delle espressioni MLX (fp8.h e8m0, fp4.h e2m1).
static float ref_e8m0(uint8_t b) { uint32_t s = (b == 0u) ? 0x400000u : ((uint32_t)b << 23); float out; memcpy(&out, &s, 4); return out; }
static const float E2M1_LUT[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
static float ref_e2m1(uint8_t nib) { float v = E2M1_LUT[nib & 7u]; return (nib & 8u) ? -v : v; }

int main(int argc, char** argv) {
    uint32_t H = argc > 1 ? (uint32_t)atoi(argv[1]) : 2880;
    uint32_t M = argc > 2 ? (uint32_t)atoi(argv[2]) : 2880;
    uint32_t group_size = argc > 3 ? (uint32_t)atoi(argv[3]) : 32;
    uint32_t n_groups = H / group_size;
    printf("shape: H=%u M=%u group_size=%u n_groups=%u\n", H, M, group_size, n_groups);
    if (H % 64 != 0 || H % group_size != 0) { fprintf(stderr, "H deve essere multiplo di 64 e di group_size\n"); return 1; }

    srand(1337);
    // x half random in [-1,1]
    uint16_t* xh = (uint16_t*)malloc(sizeof(uint16_t) * H);
    float* xf = (float*)malloc(sizeof(float) * H);
    for (uint32_t i = 0; i < H; i++) { xf[i] = ((rand()%2000)-1000)/1000.0f; xh[i] = float_to_half(xf[i]); }

    // pesi MXFP4: nibble random 0..15 (2 per byte), scale e8m0 byte random in range finito
    uint8_t* Wq = (uint8_t*)malloc((size_t)M * (H/2));
    uint8_t* scales = (uint8_t*)malloc((size_t)M * n_groups);
    for (size_t i = 0; i < (size_t)M*(H/2); i++) Wq[i] = (uint8_t)(rand() & 0xFF);   // 2 nibble random
    // Scale e8m0 realistiche (2^-7..2^0), come nei pesi reali normalizzati: il prodotto
    // e2m1(<=6)*scala(<=1) x H termini random-sign resta dentro il range half (<=65504).
    // Con scale a 2^18 il dot (~4e6) traboccherebbe l'output half -> inf, che non e' un bug del
    // kernel ma un dato non rappresentativo (i pesi reali sono piccoli).
    for (size_t i = 0; i < (size_t)M*n_groups; i++) scales[i] = (uint8_t)(120 + (rand()%8)); // 2^-7..2^0

    // Riferimento CPU: per ogni riga, dequant esatto (dai byte scritti) e dot con xf.
    float* out_ref = (float*)malloc(sizeof(float) * M);
    for (uint32_t r = 0; r < M; r++) {
        const uint8_t* rp = &Wq[(size_t)r*(H/2)];
        const uint8_t* sp = &scales[(size_t)r*n_groups];
        double s = 0;
        for (uint32_t h = 0; h < H; h++) {
            uint8_t b = rp[h>>1];
            uint8_t nib = (h & 1u) ? (b >> 4) : (b & 0x0Fu);
            uint32_t g = h / group_size;
            s += (double)(ref_e8m0(sp[g]) * ref_e2m1(nib)) * (double)half_to_float(xh[h]);
        }
        out_ref[r] = (float)s;
    }

    LlmMetalGemvMxfp4Ctx ctx;
    if (llmmetal_gemv_mxfp4_init(&ctx) != 0) { fprintf(stderr, "init fallita\n"); return 1; }
    if (llmmetal_gemv_mxfp4_load_weights(&ctx, Wq, scales, H, M, group_size) != 0) { fprintf(stderr, "load pesi fallito\n"); return 1; }
    uint16_t* out_gpu = (uint16_t*)malloc(sizeof(uint16_t)*M);

    llmmetal_gemv_mxfp4_run(&ctx, xh, out_gpu); // warmup
    int reps = 200;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) llmmetal_gemv_mxfp4_run(&ctx, xh, out_gpu);
    double dt = (now_s()-t0)/reps;
    double bytes = (double)M*(H/2) + (double)M*n_groups; // pesi + scale (niente bias in MXFP4)
    printf("gemv_mxfp4: %.4f ms, %.2f GB/s (pesi+scale: %.2f MB)\n", dt*1e3, bytes/dt/1e9, bytes/1e6);

    const double atol = 0.05, rtol = 0.02;
    double max_abs = 0; int n_fail = 0; uint32_t worst_r = 0; double worst_margin = 0;
    for (uint32_t r = 0; r < M; r++) {
        float g = half_to_float(out_gpu[r]);
        double d = fabsf(g - out_ref[r]);
        if (d > max_abs) max_abs = d;
        double tol = atol + rtol * fabs(out_ref[r]);
        double margin = d - tol;
        if (margin > worst_margin) { worst_margin = margin; worst_r = r; }
        if (d > tol) n_fail++;
    }
    printf("correttezza vs riferimento CPU (dequant esatto MLX): max_abs_diff=%.5f, %d/%u righe fuori tolleranza "
           "(atol=%.3f+rtol=%.3f*|ref|) — peggiore: riga %u gpu=%f ref=%f -- %s\n",
           max_abs, n_fail, M, atol, rtol, worst_r, half_to_float(out_gpu[worst_r]), out_ref[worst_r],
           n_fail == 0 ? "PASS" : "FAIL");

    /* ── #47: A/B del kernel v2 (uint load, bandwidth) contro lo STESSO riferimento CPU ── */
    llmmetal_gemv_mxfp4_run_v2(&ctx, xh, out_gpu); // warmup v2
    double t0b = now_s();
    for (int r = 0; r < reps; r++) llmmetal_gemv_mxfp4_run_v2(&ctx, xh, out_gpu);
    double dtv2 = (now_s()-t0b)/reps;
    printf("gemv_mxfp4_v2: %.4f ms, %.2f GB/s  (speedup vs v1: %.2fx)\n",
           dtv2*1e3, bytes/dtv2/1e9, dt/dtv2);
    double max_abs2 = 0; int n_fail2 = 0;
    for (uint32_t r = 0; r < M; r++) {
        float g = half_to_float(out_gpu[r]);
        double d = fabsf(g - out_ref[r]);
        if (d > max_abs2) max_abs2 = d;
        if (d > atol + rtol * fabs(out_ref[r])) n_fail2++;
    }
    printf("correttezza v2 vs riferimento CPU: max_abs_diff=%.5f, %d/%u fuori tolleranza -- %s\n",
           max_abs2, n_fail2, M, n_fail2 == 0 ? "PASS" : "FAIL");

    /* ── #47: v3 (uint4 load) ── */
    llmmetal_gemv_mxfp4_run_v3(&ctx, xh, out_gpu); // warmup v3
    double t0c = now_s();
    for (int r = 0; r < reps; r++) llmmetal_gemv_mxfp4_run_v3(&ctx, xh, out_gpu);
    double dtv3 = (now_s()-t0c)/reps;
    printf("gemv_mxfp4_v3: %.4f ms, %.2f GB/s  (speedup vs v1: %.2fx, vs v2: %.2fx)\n",
           dtv3*1e3, bytes/dtv3/1e9, dt/dtv3, dtv2/dtv3);
    double max_abs3 = 0; int n_fail3 = 0;
    for (uint32_t r = 0; r < M; r++) {
        float g = half_to_float(out_gpu[r]);
        double d = fabsf(g - out_ref[r]);
        if (d > max_abs3) max_abs3 = d;
        if (d > atol + rtol * fabs(out_ref[r])) n_fail3++;
    }
    printf("correttezza v3 vs riferimento CPU: max_abs_diff=%.5f, %d/%u fuori tolleranza -- %s\n",
           max_abs3, n_fail3, M, n_fail3 == 0 ? "PASS" : "FAIL");

    /* ── #47: v4 (column-parallel multi-warp-per-row) ── */
    llmmetal_gemv_mxfp4_run_v4(&ctx, xh, out_gpu); // warmup v4
    double t0d = now_s();
    for (int r = 0; r < reps; r++) llmmetal_gemv_mxfp4_run_v4(&ctx, xh, out_gpu);
    double dtv4 = (now_s()-t0d)/reps;
    printf("gemv_mxfp4_v4: %.4f ms, %.2f GB/s  (speedup vs v1: %.2fx, vs v2: %.2fx)\n",
           dtv4*1e3, bytes/dtv4/1e9, dt/dtv4, dtv2/dtv4);
    double max_abs4 = 0; int n_fail4 = 0;
    for (uint32_t r = 0; r < M; r++) {
        float g = half_to_float(out_gpu[r]);
        double d = fabsf(g - out_ref[r]);
        if (d > max_abs4) max_abs4 = d;
        if (d > atol + rtol * fabs(out_ref[r])) n_fail4++;
    }
    printf("correttezza v4 vs riferimento CPU: max_abs_diff=%.5f, %d/%u fuori tolleranza -- %s\n",
           max_abs4, n_fail4, M, n_fail4 == 0 ? "PASS" : "FAIL");

    llmmetal_gemv_mxfp4_free(&ctx);
    return (n_fail == 0 && n_fail2 == 0 && n_fail3 == 0 && n_fail4 == 0) ? 0 : 2;
}
