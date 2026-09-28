// oq_a8_support.metal — kernel di supporto per il primitive oQ A8 (nessun tensor unit, puro Metal).
// Compilabile anche senza -mmacosx-version-min=26.2 (nessun MPP).
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"   /* bfloat16_t */
#include "oq_a8_decode.h"                      /* oq_group_words (GS64) */
using namespace metal;
using namespace omlx::oq_a8;

#ifndef STEEL_PRAGMA_UNROLL
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

// Repack v8 dell'attivazione dentro ogni GS64: out[m, 16c+4t+j] = in[m, 16c+8*(t>>1)+2j+(t&1)].
// Un thread per byte di output (griglia M*(K/64) x 64).
kernel void oq_repack_v8(const device uchar* in [[buffer(0)]], device uchar* out [[buffer(1)]],
                         constant int& K [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int m = (int)gid.y;                 // riga
    int slot = (int)gid.x;              // 0..(K-1) dentro la riga (gruppo*64+off)
    int g = slot >> 6, off = slot & 63;
    int c = off >> 4, rem = off & 15;
    int t = rem >> 2, j = rem & 3;
    int k = (g << 6) + (c << 4) + ((t >> 1) << 3) + (j << 1) + (t & 1);
    out[(size_t)m * K + slot] = in[(size_t)m * K + k];
}

// Trasposizione 2D f16: out[c, r] = in[r, c]   (in: R x C, out: C x R)
kernel void oq_transpose2d_f16(const device half* in [[buffer(0)]], device half* out [[buffer(1)]],
                               constant int& R [[buffer(2)]], constant int& C [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    int r = (int)gid.y, c = (int)gid.x;
    if (r < R && c < C) out[(size_t)c * R + r] = in[(size_t)r * C + c];
}

// Trasposizione 2D i16
kernel void oq_transpose2d_i16(const device short* in [[buffer(0)]], device short* out [[buffer(1)]],
                               constant int& R [[buffer(2)]], constant int& C [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    int r = (int)gid.y, c = (int)gid.x;
    if (r < R && c < C) out[(size_t)c * R + r] = in[(size_t)r * C + c];
}

// Trasposizione 2D bf16 (dtype del modello, come oMLX)
kernel void oq_transpose2d_bf16(const device bfloat16_t* in [[buffer(0)]], device bfloat16_t* out [[buffer(1)]],
                                constant int& R [[buffer(2)]], constant int& C [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]]) {
    int r = (int)gid.y, c = (int)gid.x;
    if (r < R && c < C) out[(size_t)c * R + r] = in[(size_t)r * C + c];
}

// ─────────────────────────────────────────────────────────────────────────────────────────
// oq_a8_qmv — GEMV MULTI-OUTPUT per M piccolo (decode/verify M<=8).
//
// Perche': a M piccolo il GEMM a tile 16x32 spreca tile e occupa pochi blocchi (misurato:
// ~100-230 GB/s a M=8 contro i ~465 GB/s del GEMV MLX a M=1). Qui la griglia e' per COLONNA
// (N threadgroup) e il peso e' letto UNA volta e riusato su TUTTE le M righe (multi-output).
//
// Struttura: 32 lane per threadgroup = `cols_per_tg` colonne; ogni lane legge UNA parola
// (BITS*32 bit) del gruppo per la sua colonna, decodifica i codici e accumula MM dot-product
// in fp32 (w = q*s + b, correzione inline per gruppo; l'attivazione qa e' int8, scalata a
// fine da sa[m]). Riduzione finale con simd_shuffle_xor sulle lane della colonna.
// Numerica: identica PER RIGA fra M=1 e M=8 (stesso ordine di k) -> AR e verify coerenti.
// ─────────────────────────────────────────────────────────────────────────────────────────
// Mappa k -> slot di memoria per la permutazione v8 fusa nel kernel di quantize.
// Verificata EMPIRICAMENTE (dump qa vs atteso, k=0..15): slot = kPerm16[k], con
// kPerm16 = {0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15} (stessa tabella del quantize).
constant const int kOqV8Slot[16] = {0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15};

template <typename T, int BITS, int MM>
[[kernel]] void oq_a8_qmv(
    const device int8_t* qa [[buffer(0)]],
    const device float* sa [[buffer(1)]],
    const device short* ra [[buffer(2)]],      /* non usata: correzione inline per gruppo */
    const device uint32_t* w [[buffer(3)]],
    const device T* scales [[buffer(4)]],
    const device T* biases [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& g0 [[buffer(10)]],
    const constant int& g1 [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  (void)ra; (void)g0; (void)g1;
  constexpr int words_per_group = oq_group_words(BITS);   /* Q4: 8, Q8: 16 */
  constexpr int codes_per_word = 32 / BITS;               /* Q4: 8, Q8: 4 */
  constexpr int lanes_per_col = words_per_group;          /* Q4: 8, Q8: 16 */
  constexpr int cols_per_tg = 32 / lanes_per_col;         /* Q4: 4, Q8: 2 */
  const int sub = int(lane) % lanes_per_col;
  const int col = int(tgid) * cols_per_tg + int(lane) / lanes_per_col;
  if (col >= N) return;
  const device uint32_t* wrow = w + size_t(col) * size_t(K * BITS / 32);
  const int G = K / 64;
  /* Righe di attivazione clampate a M-1 (le righe oltre M non vengono scritte). */
  const device int8_t* qrow[MM];
  STEEL_PRAGMA_UNROLL
  for (int m = 0; m < MM; ++m)
    qrow[m] = qa + size_t(min(m, M - 1)) * size_t(K);
  float part[MM];
  STEEL_PRAGMA_UNROLL
  for (int m = 0; m < MM; ++m) part[m] = 0.0f;
  for (int g = 0; g < G; ++g) {
    const float s = float(scales[size_t(g) * size_t(N) + col]);
    const float b = float(biases[size_t(g) * size_t(N) + col]);
    const uint32_t wv = wrow[size_t(g) * words_per_group + sub];
    const int k0 = g * 64 + sub * codes_per_word;
    STEEL_PRAGMA_UNROLL
    for (int c = 0; c < codes_per_word; ++c) {
      const uint32_t q = (wv >> (BITS * c)) & ((1u << BITS) - 1u);
      const float wf = metal::fma(float(q), s, b);
      /* qa e' nel layout v8: k -> slot di memoria. */
      const int kk = k0 + c;
      const int slot = (kk & ~15) | kOqV8Slot[kk & 15];
      STEEL_PRAGMA_UNROLL
      for (int m = 0; m < MM; ++m)
        part[m] = metal::fma(wf, float(qrow[m][slot]), part[m]);
    }
  }
  STEEL_PRAGMA_UNROLL
  for (int off = 1; off < lanes_per_col; off <<= 1) {
    STEEL_PRAGMA_UNROLL
    for (int m = 0; m < MM; ++m) part[m] += simd_shuffle_xor(part[m], ushort(off));
  }
  if (sub == 0) {
    STEEL_PRAGMA_UNROLL
    for (int m = 0; m < MM; ++m)
      if (m < M) out[size_t(m) * size_t(N) + col] = static_cast<T>(part[m] * sa[m]);
  }
}

#define instantiate_oq_a8_qmv(bits, mm, type)                                 \
  instantiate_kernel("oq_a8_qmv_q" #bits "_m" #mm "_" #type, oq_a8_qmv,       \
                     type, bits, mm)

#define instantiate_oq_a8_qmv_mm(bits, mm)                                    \
  instantiate_oq_a8_qmv(bits, mm, float16_t);                                 \
  instantiate_oq_a8_qmv(bits, mm, bfloat16_t)

#define instantiate_oq_a8_qmv_bits(bits)                                      \
  instantiate_oq_a8_qmv_mm(bits, 1);                                          \
  instantiate_oq_a8_qmv_mm(bits, 2);                                          \
  instantiate_oq_a8_qmv_mm(bits, 3);                                          \
  instantiate_oq_a8_qmv_mm(bits, 4);                                          \
  instantiate_oq_a8_qmv_mm(bits, 5);                                          \
  instantiate_oq_a8_qmv_mm(bits, 6);                                          \
  instantiate_oq_a8_qmv_mm(bits, 7);                                          \
  instantiate_oq_a8_qmv_mm(bits, 8)

instantiate_oq_a8_qmv_bits(4);
instantiate_oq_a8_qmv_bits(8);
