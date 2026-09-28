// gemv_q4_selfcheck.m — verifica correttezza (vs riferimento CPU) e banda per il kernel Metal
// GEMV Q4 affine (llmmetal_gemv_q4.h). Quantizza una matrice random fp32 in Q4 affine per
// gruppo, esegue il kernel, confronta con un dequant+dot di riferimento sugli STESSI dati
// quantizzati (valida la tecnica del kernel, non la compatibilità bit-esatta col formato MLX —
// vedi commento in testa a llmmetal_gemv_q4.h).
//
// Uso: ./gemv_q4_selfcheck [H] [M] [group_size]
// Default: shape del "dato decisivo" del piano (H=2048, M=8192, group_size=64).
#import "../llmmetal_gemv_q4.h"
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

int main(int argc, char** argv) {
    uint32_t H = argc > 1 ? (uint32_t)atoi(argv[1]) : 2048;
    uint32_t M = argc > 2 ? (uint32_t)atoi(argv[2]) : 8192;
    uint32_t group_size = argc > 3 ? (uint32_t)atoi(argv[3]) : 64;
    uint32_t n_groups = H / group_size;
    printf("shape: H=%u M=%u group_size=%u n_groups=%u\n", H, M, group_size, n_groups);

    srand(42);
    float* Wf = (float*)malloc(sizeof(float) * (size_t)M * H);
    for (size_t i = 0; i < (size_t)M*H; i++) Wf[i] = ((rand()%2000)-1000)/1000.0f;
    float* xf = (float*)malloc(sizeof(float) * H);
    for (uint32_t i = 0; i < H; i++) xf[i] = ((rand()%2000)-1000)/1000.0f;

    // quantizza per riga/gruppo: affine min/max -> scale/bias, nibble 0..15
    uint8_t* Wq = (uint8_t*)malloc((size_t)M * (H/2));
    uint16_t* scales = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)M * n_groups);
    uint16_t* biases = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)M * n_groups);
    float* Wdeq_ref = (float*)malloc(sizeof(float) * (size_t)M * H); // per il riferimento CPU
    for (uint32_t r = 0; r < M; r++) {
        for (uint32_t g = 0; g < n_groups; g++) {
            float lo = INFINITY, hi = -INFINITY;
            for (uint32_t j = 0; j < group_size; j++) {
                float v = Wf[(size_t)r*H + g*group_size + j];
                if (v < lo) lo = v; if (v > hi) hi = v;
            }
            float scale = (hi - lo) / 15.0f; if (scale < 1e-8f) scale = 1e-8f;
            float bias = lo;
            scales[(size_t)r*n_groups+g] = float_to_half(scale);
            biases[(size_t)r*n_groups+g] = float_to_half(bias);
            for (uint32_t j = 0; j < group_size; j++) {
                uint32_t h = g*group_size + j;
                float v = Wf[(size_t)r*H + h];
                int nib = (int)roundf((v - bias) / scale);
                if (nib < 0) nib = 0; if (nib > 15) nib = 15;
                uint32_t byte_idx = h/2;
                uint8_t* bptr = &Wq[(size_t)r*(H/2) + byte_idx];
                if (h & 1) *bptr = (uint8_t)((*bptr & 0x0F) | (nib << 4));
                else       *bptr = (uint8_t)((*bptr & 0xF0) | nib);
                // dequant "ideale" col nibble ESATTO scritto (fedele a quel che il kernel legge)
                Wdeq_ref[(size_t)r*H+h] = (float)nib * scale + bias;
            }
        }
    }

    LlmMetalGemvQ4Ctx ctx;
    if (llmmetal_gemv_q4_init(&ctx) != 0) { fprintf(stderr, "init fallita\n"); return 1; }
    if (llmmetal_gemv_q4_load_weights(&ctx, Wq, scales, biases, H, M, group_size) != 0) { fprintf(stderr, "load pesi fallito\n"); return 1; }
    uint16_t* xh = (uint16_t*)malloc(sizeof(uint16_t)*H);
    for (uint32_t i = 0; i < H; i++) xh[i] = float_to_half(xf[i]);
    uint16_t* out_gpu = (uint16_t*)malloc(sizeof(uint16_t)*M);

    llmmetal_gemv_q4_run(&ctx, xh, out_gpu); // warmup
    int reps = 200;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) llmmetal_gemv_q4_run(&ctx, xh, out_gpu);
    double dt = (now_s()-t0)/reps;
    double bytes = (double)M*(H/2) + (double)M*n_groups*sizeof(uint16_t)*2 /*scale+bias*/;
    printf("gemv_q4_affine: %.4f ms, %.2f GB/s (pesi+scale+bias esatti: %.2f MB)\n", dt*1e3, bytes/dt/1e9, bytes/1e6);

    // riferimento CPU: dot(x, Wdeq_ref) per ogni riga
    float* out_ref = (float*)malloc(sizeof(float)*M);
    for (uint32_t r = 0; r < M; r++) {
        double s = 0;
        for (uint32_t h = 0; h < H; h++) s += (double)xf[h] * Wdeq_ref[(size_t)r*H+h];
        out_ref[r] = (float)s;
    }
    // Tolleranza stile numpy.allclose: atol + rtol*|ref| — evita che l'errore relativo esploda
    // quando il riferimento e' vicino a zero (un dot-product casuale su H=2048 elementi puo'
    // capitare vicino a 0 per pura combinazione lineare; un rel-error puro darebbe falsi FAIL
    // anche con un errore assoluto minuscolo, dominato dal rumore di arrotondamento fp16).
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
    printf("correttezza vs riferimento CPU (dequant esatto): max_abs_diff=%.5f, %d/%u righe fuori tolleranza "
           "(atol=%.3f+rtol=%.3f*|ref|) — peggiore: riga %u gpu=%f ref=%f -- %s\n",
           max_abs, n_fail, M, atol, rtol, worst_r, half_to_float(out_gpu[worst_r]), out_ref[worst_r],
           n_fail == 0 ? "PASS" : "FAIL");

    llmmetal_gemv_q4_free(&ctx);
    return n_fail == 0 ? 0 : 2;
}
