import mlx.core as mx, numpy as np
# array persistente (capacita' fissa), materializzato
CAP, D = 64, 8
K = mx.zeros([1, 2, CAP, D], dtype=mx.float32)
mx.eval(K)
Kv = np.array(K, copy=False)
assert Kv.flags.writeable, "view non scrivibile"
# append-like: scrivi il token alla posizione 5
Kv[0, 0, 5, :] = 7.0
Kv[0, 0, 6, :] = 9.0
# 1) MLX GPU vede la mutazione?
s = mx.sum(K)                      # operazione GPU sull'array mutato
mx.eval(s)
print("mx.sum(K) =", float(s), "(atteso 2 canali * 2 token * 8 = 32.0 * (7+9)/... )")
# 2) confronto diretto: leggi via MLX
print("K[0,0,5,0] via MLX =", float(K[0,0,5,0]), " K[0,0,6,0] =", float(K[0,0,6,0]))
# 3) una SDPA sull'array mutato (kernel MLX) vede i valori?
q = mx.ones([1,2,1,D]); 
out = mx.fast.scaled_dot_product_attention(q, K, K, scale=0.5)
mx.eval(out)
print("sdpa out[0,0,0,0] =", float(out[0,0,0,0]), "(se != 0 la SDPA legge i valori mutati)")
# 4) stabilità: piu' scritture successive
for t in range(7, 12):
    Kv[0,0,t,:] = float(t)
print("K[0,0,11,0] =", float(K[0,0,11,0]), "(atteso 11)")
# 5) quanto costa un memcpy CPU di un token nella view?
import time
tok = np.arange(D, dtype=np.float32)
t0=time.perf_counter()
for t in range(12, 20): Kv[0,0,t,:] = tok
print("20 memcpy token nella view: %.6f ms" % ((time.perf_counter()-t0)*1e3))
