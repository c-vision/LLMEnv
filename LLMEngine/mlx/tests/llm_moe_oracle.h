/* llm_moe_oracle.h — oracolo 1:1 di qwen4_row_dot (ds4, metal/qwen4.metal) per il NOSTRO
 * formato Q4 affine MLX. Riproduce lo split lane di ds4 (ix = lane/8, it = lane%8, 4 blocchi
 * da 32 per iterazione, elementi h, h+1, h+16, h+17), dequant per elemento e accumulo
 * sequenziale; la somma finale è quella dei 32 lane. Solo lettura: nibble affine invece di
 * q8_0/q4_K. Nessun codice di esecuzione: è il riferimento del gate. */
#ifndef LLM_MOE_ORACLE_H
#define LLM_MOE_ORACLE_H
#include <stdint.h>
#include <string.h>

static inline float llm_moe_oracle_ld(const uint8_t* p, uint32_t idx, int bf) {
    uint16_t u = (uint16_t)(p[2u*idx] | ((uint16_t)p[2u*idx + 1u] << 8));
    if (bf) { uint32_t v = ((uint32_t)u) << 16; float f; memcpy(&f,&v,4); return f; }
    _Float16 h; memcpy(&h,&u,2); return (float)h;
}
static inline float llm_moe_oracle_w(const uint8_t* Wq, const uint8_t* sc, const uint8_t* bi,
                                     uint32_t h, uint32_t group, int bf) {
    uint8_t b = Wq[h >> 1]; uint32_t nib = (h & 1u) ? (b >> 4) : (b & 0xF);
    uint32_t g = h / group;
    return (float)nib * llm_moe_oracle_ld(sc, g, bf) + llm_moe_oracle_ld(bi, g, bf);
}
/* dot di una riga (row) con x: identico per ordine a qwen4_row_dot, somma dei 32 lane. */
static inline float llm_moe_oracle_dot(const uint8_t* Wq, const uint8_t* sc, const uint8_t* bi,
                                       const float* x, uint32_t H, uint32_t group, int bf,
                                       uint32_t row, uint32_t lane) {
    const uint8_t* rp = Wq + (size_t)row * (H / 2);
    const uint32_t ng = H / group;
    const uint8_t* rs = sc + (size_t)row * ng * 2;
    const uint8_t* rb = bi + (size_t)row * ng * 2;
    const uint32_t ix = lane / 8u, it = lane % 8u, nb = H / 32u;
    float acc = 0.0f;
    for (uint32_t ib = ix; ib < nb; ib += 4u) {
        const uint32_t h0 = ib * 32u + it * 2u;
        const float y0 = x[h0], y1 = x[h0+1], y16 = x[h0+16], y17 = x[h0+17];
        acc += y0 * llm_moe_oracle_w(rp, rs, rb, h0,    group, bf)
             + y1 * llm_moe_oracle_w(rp, rs, rb, h0+1,  group, bf)
             + y16* llm_moe_oracle_w(rp, rs, rb, h0+16, group, bf)
             + y17* llm_moe_oracle_w(rp, rs, rb, h0+17, group, bf);
    }
    return acc;
}
static inline float llm_moe_oracle_row(const uint8_t* Wq, const uint8_t* sc, const uint8_t* bi,
                                       const float* x, uint32_t H, uint32_t group, int bf, uint32_t row) {
    float tot = 0.0f;
    for (uint32_t lane = 0; lane < 32u; lane++)
        tot += llm_moe_oracle_dot(Wq, sc, bi, x, H, group, bf, row, lane);
    return tot;
}
#endif
