#!/usr/bin/env python3
"""In-place KV append via buffer condiviso MLX: batte il concat (O(pos)) a contesto lungo?
Scoperta: gli array MLX (Metal) usano ResourceStorageModeShared -> si scrive il nuovo token
direttamente dal CPU nel buffer (O(1), ~us) e SDPA/kernel MLX lo vedono. Qui si misura il costo
per-step di (A) CONCAT [O(pos)] vs (B) IN-PLACE [memcpy CPU + eval per-layer], con attenzione reale.
"""
import time, numpy as np, mlx.core as mx
L, H, D = 16, 8, 64
def bench(N, reps=20):
    print(f"--- N={N} L={L} H={H} D={D} ---")
    x = mx.ones([1, 1, 1024], mx.float16); W = mx.ones([1024, H * D], mx.float16)
    def new_tok():                       # proiezione k/v simulata (piccola) per layer
        return (x @ W).reshape(1, 1, H, D).transpose(0, 2, 1, 3)  # [1,H,1,D]
    # (A) concat
    ca = [mx.zeros([1, H, N, D], mx.float16) for _ in range(L)]
    def fA():
        outs = []
        for i in range(L):
            c = mx.concatenate([ca[i], new_tok()], axis=2)
            q = mx.ones([1, H, 1, D], mx.float16)
            outs.append(mx.fast.scaled_dot_product_attention(q, c, c, scale=1.0/(D**0.5)))
        return outs
    for _ in range(3): mx.eval(fA())
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); mx.eval(fA()); ts.append(time.perf_counter()-t)
    ta=sorted(ts)[len(ts)//2]
    # (B) in-place: buffer persistente condiviso + memcpy CPU del nuovo token
    cb = [mx.zeros([1, H, N + 1, D], mx.float16) for _ in range(L)]
    mx.eval(*cb)
    vw = [np.array(b, copy=False) for b in cb]
    def fB():
        outs = []
        for i in range(L):
            n = new_tok()
            mx.eval(n)                                    # sync per-layer (k/v sulla CPU)
            vw[i][:, :, N:N+1, :] = np.array(n, copy=False)[0, 0]  # memcpy O(1)
            q = mx.ones([1, H, 1, D], mx.float16)
            outs.append(mx.fast.scaled_dot_product_attention(q, cb[i], cb[i], scale=1.0/(D**0.5)))
        return outs
    for _ in range(3): mx.eval(fB())
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); mx.eval(fB()); ts.append(time.perf_counter()-t)
    tb=sorted(ts)[len(ts)//2]
    print(f"  A CONCAT  ms/step={ta*1e3:.3f}  ({N/ta:.0f} tok/s equivalente)")
    print(f"  B INPLACE ms/step={tb*1e3:.3f}  ratio B/A={tb/ta:.3f}")
for N in (2048, 8192, 32768):
    bench(N)
