/* llm_mlx_q4_msl.h — helper MSL condiviso per il gemv Q4 affine su pesi PACKED (mmap) letti come nodi lazy MLX.
 *
 * q4_dotn(): fino a 4 righe INDIPENDENTI contro lo stesso vettore x (f32) in un simdgroup. Rispetto al vecchio kernel (un nibble
 * per lane, un solo carico da 1 byte in volo) ogni lane elabora un intero GRUPPO da 32 colonne per iterazione:
 *   - 4 word da 32 bit per riga (16 byte), piu' carichi indipendenti in volo (4 righe x 4 word);
 *   - x del gruppo in registri (8 float4) e somma di x calcolata una volta, riusate su tutte le righe;
 *   - scala/bias letti una volta per gruppo e riga; prodotto = sc * sum(x*nib) + bias * sum(x).
 * Gli offset dei tensori nelle safetensors non sono allineati (l'header ha lunghezza arbitraria): m = (offset del tensore) & 3
 * e' costante per tensore; si legge dalla griglia allineata a 4 byte e si ricompone con shift (nessun carico non allineato).
 * Scale/bias: letti a byte (offset anche dispari), bf16 o half. Righe di lunghezza H colonne, H multiplo di 32. */
#ifndef LLM_MLX_Q4_MSL_H
#define LLM_MLX_Q4_MSL_H

static const char llm_q4_msl_helpers[] =
"inline float q4_ld_sb(const device uchar* p, uint idx, uint bf) {\n"
"    const ushort u = (ushort)p[2u * idx] | ((ushort)p[2u * idx + 1u] << 8);\n"
"    return bf ? as_type<float>(((uint)u) << 16) : (float)as_type<half>(u);\n"
"}\n"
"inline float q4_nibsum(uint w, float4 xa, float4 xb) {\n"
"    return xa.x * (float)(w & 15u) + xa.y * (float)((w >> 4) & 15u) + xa.z * (float)((w >> 8) & 15u) + xa.w * (float)((w >> 12) & 15u)\n"
"         + xb.x * (float)((w >> 16) & 15u) + xb.y * (float)((w >> 20) & 15u) + xb.z * (float)((w >> 24) & 15u) + xb.w * (float)(w >> 28);\n"
"}\n"
"/* contributo del gruppo gi (32 colonne) della riga che inizia a rp (byte, m = rp & 3) */\n"
"inline float q4_row(const device uchar* rp, const device uchar* rs, const device uchar* rb, uint m, uint gi, uint sidx, uint bf,\n"
"                    float4 xv0, float4 xv1, float4 xv2, float4 xv3, float4 xv4, float4 xv5, float4 xv6, float4 xv7, float xs) {\n"
"    const device uint* wu = (const device uint*)(rp - m) + gi * 4u;\n"
"    uint w0, w1, w2, w3;\n"
"    if (m == 0u) { w0 = wu[0]; w1 = wu[1]; w2 = wu[2]; w3 = wu[3]; }\n"
"    else { const uint sh = 8u * m; uint a0 = wu[0], a1 = wu[1], a2 = wu[2], a3 = wu[3], a4 = wu[4];\n"
"           w0 = (a0 >> sh) | (a1 << (32u - sh)); w1 = (a1 >> sh) | (a2 << (32u - sh));\n"
"           w2 = (a2 >> sh) | (a3 << (32u - sh)); w3 = (a3 >> sh) | (a4 << (32u - sh)); }\n"
"    const float d = q4_nibsum(w0, xv0, xv1) + q4_nibsum(w1, xv2, xv3) + q4_nibsum(w2, xv4, xv5) + q4_nibsum(w3, xv6, xv7);\n"
"    return q4_ld_sb(rs, sidx, bf) * d + q4_ld_sb(rb, sidx, bf) * xs;\n"
"}\n"
"/* fino a 4 righe (nr) contro lo stesso x; out[r] = somma sui lane del simdgroup. rp/rs/rb: inizio riga / scale / bias. */\n"
"inline void q4_dotn(thread float* out, uint nr,\n"
"                    const device uchar* rp0, const device uchar* rp1, const device uchar* rp2, const device uchar* rp3,\n"
"                    const device uchar* rs0, const device uchar* rs1, const device uchar* rs2, const device uchar* rs3,\n"
"                    const device uchar* rb0, const device uchar* rb1, const device uchar* rb2, const device uchar* rb3,\n"
"                    uint m, const device float* x, uint H, uint gs, uint bf, uint lane) {\n"
"    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;\n"
"    const uint ngroups = H / 32;\n"
"    for (uint gi = lane; gi < ngroups; gi += 32) {\n"
"        const device float4* x4 = (const device float4*)(x + gi * 32);\n"
"        float4 xv0 = x4[0], xv1 = x4[1], xv2 = x4[2], xv3 = x4[3], xv4 = x4[4], xv5 = x4[5], xv6 = x4[6], xv7 = x4[7];\n"
"        const float xs = dot(xv0 + xv1 + xv2 + xv3 + xv4 + xv5 + xv6 + xv7, float4(1.0f));\n"
"        const uint sidx = (gi * 32u) / gs;\n"
"        acc0 += q4_row(rp0, rs0, rb0, m, gi, sidx, bf, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xs);\n"
"        if (nr > 1) acc1 += q4_row(rp1, rs1, rb1, m, gi, sidx, bf, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xs);\n"
"        if (nr > 2) acc2 += q4_row(rp2, rs2, rb2, m, gi, sidx, bf, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xs);\n"
"        if (nr > 3) acc3 += q4_row(rp3, rs3, rb3, m, gi, sidx, bf, xv0, xv1, xv2, xv3, xv4, xv5, xv6, xv7, xs);\n"
"    }\n"
"    out[0] = simd_sum(acc0); out[1] = simd_sum(acc1); out[2] = simd_sum(acc2); out[3] = simd_sum(acc3);\n"
"}\n";

#endif
