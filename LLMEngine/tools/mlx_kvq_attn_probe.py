#!/usr/bin/env python3
"""Probe P2 (Splash/KV-Q8): l'attenzione di DECODE costruita con primitive MLX native
quantizzate (`mx.quantized_matmul`) puo' battere la SDPA f16 di Apple?

Motivazione: il kernel custom `llm_mlx_kvq_sdpa` (dequant-on-read, online-softmax scalare)
misura 2,1-2,3x PIU' LENTO della SDPA f16 sul decode denso (8B). Qui si testa, in
isolamento (unit-first), se una strada diversa -- K/V quantizzati letti da `quantized_matmul`
(lo stesso GEMM ottimizzato che usa OQ-A8/W8A8) -- possa invece avvicinarsi/battere la f16.

Uso:  ~/ai/env-mlxlm/bin/python tools/mlx_kvq_attn_probe.py [cap] [bits_k] [bits_v]
"""
import sys
import time

import mlx.core as mx

nh, nkv, hd = 32, 8, 128          # Qwen3-8B GQA (G = nh/nkv = 4)
cap = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
bk = int(sys.argv[2]) if len(sys.argv) > 2 else 8
bv = int(sys.argv[3]) if len(sys.argv) > 3 else 4
g = nh // nkv
group = 64
scale = 1.0 / (hd ** 0.5)
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 50

mx.random.seed(0)
q = mx.random.normal((nh, hd)).astype(mx.float16) * 0.1
k = (mx.random.normal((nkv, cap, hd)) * 0.5).astype(mx.float16)
v = (mx.random.normal((nkv, cap, hd)) * 0.5).astype(mx.float16)

# --- riferimento: SDPA f16 di Apple (GQA: q [1,nh,1,hd], k/v [1,nkv,cap,hd]) ---
q4 = q.reshape(1, nh, 1, hd)
k4 = k.reshape(1, nkv, cap, hd)
v4 = v.reshape(1, nkv, cap, hd)


def sdpa_f16():
    return mx.fast.scaled_dot_product_attention(q4, k4, v4, scale=scale).squeeze((0, 2))


# --- percorso quantizzato: QK^T e PV via quantized_matmul nativo ---
kw, ks, kb = mx.quantize(k, group_size=group, bits=bk)
vw, vs, vb = mx.quantize(v, group_size=group, bits=bv)


def qmm_attn():
    # scores[g, cap] = q_g @ K_kh^T ; un kv-head per volta (G q-head condivisi)
    outs = []
    for kh in range(nkv):
        qq = q[kh * g:(kh + 1) * g]                      # [G, hd]
        s = mx.quantized_matmul(qq, kw[kh], ks[kh], kb[kh],
                                transpose=True, group_size=group, bits=bk)  # [G, cap]
        p = mx.softmax(s * scale, axis=-1).astype(mx.float16)
        o = mx.quantized_matmul(p, vw[kh], vs[kh], vb[kh],
                                transpose=False, group_size=group, bits=bv)  # [G, hd]
        outs.append(o)
    return mx.concatenate(outs, axis=0)


ref = sdpa_f16()
got = qmm_attn()
mx.eval(ref, got)
diff = mx.abs(ref.astype(mx.float32) - got.astype(mx.float32)).max().item()


def bench(fn):
    for _ in range(3):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters * 1e3


ms_sdpa = bench(sdpa_f16)
ms_qmm = bench(qmm_attn)
print(f"cap={cap} bits K{bk}/V{bv} nh={nh} nkv={nkv} hd={hd}")
print(f"  SDPA f16 (Apple)     {ms_sdpa*1000:8.1f} us   ({1/ms_sdpa*1000:7.1f} chiamate/s)")
print(f"  quantized_matmul     {ms_qmm*1000:8.1f} us   ({1/ms_qmm*1000:7.1f} chiamate/s)  ratio={ms_qmm/ms_sdpa:.2f}x")
print(f"  max|diff| (f16 vs q{bk}/{bv}) = {diff:.4g}")
