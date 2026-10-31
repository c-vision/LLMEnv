#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
// GEMV con dequant FUSO per GGUF q4_K (super-blocco = 256 pesi, 144 byte:
//   half d; half dmin; uchar scales[12]; uchar qs[128]
// e per ogni 64 valori due sotto-blocchi: sc/m da scales[12] (6 bit l'uno, impacchettati).
// Dequant (llama.cpp dequantize_row_q4_K): y = d*sc*q - dmin*m, q = nibble 0..15.
// Lettura dei blocchi GREZZI: nessuna espansione a F16 in RAM (il senso di I2-full).

static inline void get_scale_min_k4(int j, const device uchar* q, thread uchar* sc, thread uchar* m) {
    if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
    else       { *sc = (uchar)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
                 *m  = (uchar)((q[j + 4] >> 4) | ((q[j]     >> 6) << 4)); }
}

kernel void gemv_q4_k(device const uchar* w   [[buffer(0)]],
                      device const float* x   [[buffer(1)]],
                      device float*       out [[buffer(2)]],
                      constant int& N [[buffer(3)]],
                      constant int& K [[buffer(4)]],
                      uint3 tgpig [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg   [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgpig.x * 8 + (int)sg; if (n >= N) return;
    const int NB = K >> 8;                          // super-blocchi da 256 pesi
    const device uchar* wr = w + (size_t)n * (size_t)NB * 144;
    float acc = 0.0f;
    for (int b = (int)lane; b < NB; b += 32) {
        const device uchar* p  = wr + (size_t)b * 144;
        const float d    = (float)(*((const device half*)p));
        const float dmin = (float)(*((const device half*)(p + 2)));
        const device uchar* sc = p + 4;
        const device uchar* qs = p + 16;
        const float* xb = x + (size_t)b * 256;
        #pragma unroll
        for (int g = 0; g < 4; g++) {               // 4 gruppi da 64 valori = 2 sotto-blocchi
            uchar s1, m1, s2, m2;
            get_scale_min_k4(2 * g,     sc, &s1, &m1);
            get_scale_min_k4(2 * g + 1, sc, &s2, &m2);
            const float d1 = d * (float)s1, mf1 = dmin * (float)m1;
            const float d2 = d * (float)s2, mf2 = dmin * (float)m2;
            const device uchar* q = qs + g * 32;
            /* Accumula q·x e Σx SEPARATI: (d1*q - mf1)·x = d1·(q·x) - mf1·(Σx).
             * Toglie una FMA per valore (q4_K paga 2 scale+min per 64 valori, non 1 per 32). */
            float sq_lo = 0.0f, sq_hi = 0.0f, sx_lo = 0.0f, sx_hi = 0.0f;
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int lo = q[l] & 0xF;
                const int hi = (q[l] >> 4) & 0xF;
                const int idx = g * 64 + l;
                const float xl = xb[idx], xh = xb[idx + 32];
                sq_lo += (float)lo * xl;  sx_lo += xl;
                sq_hi += (float)hi * xh;  sx_hi += xh;
            }
            acc += d1 * sq_lo - mf1 * sx_lo + d2 * sq_hi - mf2 * sx_hi;
        }
    }
    float tot = simd_sum(acc);
    if (lane == 0) out[n] = tot;
}

// F16 GEMV di riferimento (stessa shape): 1 simdgroup per riga.
kernel void gemv_f16(device const half* w  [[buffer(0)]],
                     device const float* x [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant int& N [[buffer(3)]],
                     constant int& K [[buffer(4)]],
                     uint3 tgpig [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg   [[simdgroup_index_in_threadgroup]]) {
    int n = (int)tgpig.x*8 + (int)sg; if (n >= N) return;
    const device half* wr = w + (size_t)n*K;
    float acc = 0.0f;
    for (int k = (int)lane; k < K; k += 32) acc += (float)wr[k] * x[k];
    float tot = simd_sum(acc);
    if (lane == 0) out[n] = tot;
}
