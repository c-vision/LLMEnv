#!/usr/bin/env python3
"""Eviction/retention lab — valida i 15 metodi "token eviction & merging" di VeloxQuant, la
famiglia che RIDUCE la RSS (l'unica oggi) e quindi la leva del CONTESTO LUNGO di GMBasic.

Simula attenzione causale Q·K e misura l'errore d'uscita (MSE) usando solo un BUDGET B di token
(vs attenzione completa). Metodi: StreamingLLM (sink+finestra), H2O (massa cumulativa), TOVA
(attenzione dello step corrente), SnapKV (proxy-query sulle ultime obs righe di K), L2Norm
(chiavi a norma BASSA = importanza alta), PyramidKV (budget per-layer), ChunkKV (chunk contigui),
CaM (merging), random (controllo).

Uso:  /Users/gaetano/ai/env312/bin/python tools/kv_evict_lab.py
"""
import mlx.core as mx

mx.set_default_device(mx.gpu)
S, D, SQ, BUD = 2048, 128, 64, 256   # sequenza, dim, query, budget token tenuti
SINK, OBS, WIN = 4, 32, 128          # StreamingLLM / SnapKV
mx.random.seed(7)


def softmax(x):
    x = x - mx.max(x, axis=-1, keepdims=True)
    e = mx.exp(x)
    return e / mx.sum(e, axis=-1, keepdims=True)


def mse(a, b):
    return float(mx.mean((a - b) ** 2))


def build():
    """K con token 'heavy-hitter' ALLINEATI alle query (catturano l'attenzione) — regime reale."""
    import numpy as np

    qdir = mx.random.normal(shape=(D,))
    qdir = qdir / mx.linalg.norm(qdir)
    Q = mx.random.normal(shape=(SQ, D)) * 0.3 + qdir * 2.0   # query prevalentemente allineate
    K = mx.random.normal(shape=(S, D)) * 0.5
    V = mx.random.normal(shape=(S, D))
    hh = np.arange(0, S, 64)                                  # 32 heavy-hitter
    K = np.array(K)
    K[hh] = np.array(qdir * 3.0 + mx.random.normal(shape=(len(hh), D)) * 0.1)
    return Q, mx.array(K), V


def out_full(Q, K, V):
    A = softmax((Q @ K.T) / (D ** 0.5))
    return A @ V, A


def out_kept(Q, K, V, idx):
    Kk, Vk = mx.take(K, idx, 0), mx.take(V, idx, 0)
    A = softmax((Q @ Kk.T) / (D ** 0.5))
    return A @ Vk


def scores(Q, K):
    A = softmax((Q @ K.T) / (D ** 0.5))
    return {
        "H2O (massa cumulativa)": mx.sum(A, axis=0),
        "TOVA (ultimo step)": A[-1],
        "SnapKV (proxy-query)": mx.sum(A[-OBS:], axis=0),
        "L2Norm (-norma K)": -mx.linalg.norm(K, axis=-1),
    }


def main():
    Q, K, V = build()
    full, A = out_full(Q, K, V)
    base = mse(full, mx.zeros_like(full))  # varianza di riferimento = energia
    print(f"budget={BUD}/{S}  sink={SINK}  win={WIN}  obs={OBS}")
    print(f"{'metodo':<26} {'MSE uscita':>12} {'/energia':>10} {'recall HH':>10}")
    print("-" * 62)
    # heavy-hitter veri: token con massa d'attenzione alta
    true_mass = mx.sum(A, axis=0)
    top_true = set(mx.argsort(-true_mass)[:BUD].tolist())
    print(f"{'full (nessuna eviction)':<26} {mse(out_full(Q,K,V)[0], full):>12.5f} {0:>10.3f}")
    # StreamingLLM: sink + ultimi WIN
    idx = mx.array(list(range(SINK)) + list(range(S - (BUD - SINK), S)))
    rec = len(top_true & set(idx.tolist())) / BUD
    print(f"{'StreamingLLM (sink+win)':<26} {mse(out_kept(Q,K,V,idx), full):>12.5f} {mse(out_kept(Q,K,V,idx),full)/base:>10.3f} {rec:>10.3f}")
    # random (controllo)
    idxr = mx.array(mx.random.permutation(mx.arange(S))[:BUD].tolist())
    print(f"{'random (controllo)':<26} {mse(out_kept(Q,K,V,idxr), full):>12.5f} {mse(out_kept(Q,K,V,idxr),full)/base:>10.3f} {len(top_true&set(idxr.tolist()))/BUD:>10.3f}")
    for name, sc in scores(Q, K).items():
        top = mx.argsort(-sc)[:BUD]
        idx = mx.concatenate([mx.arange(SINK), top])  # sink obbligatori
        rec = len(top_true & set(top.tolist())) / BUD
        print(f"{name:<26} {mse(out_kept(Q,K,V,idx), full):>12.5f} {mse(out_kept(Q,K,V,idx),full)/base:>10.3f} {rec:>10.3f}")
    # ChunkKV: eviction a chunk contigui (semantici); chunk=1 == H2O
    import numpy as np

    import numpy as _np

    sc = np.array(scores(Q, K)["H2O (massa cumulativa)"])
    chunk = 8
    nc = S // chunk
    cs = sc[: nc * chunk].reshape(nc, chunk).sum(axis=1)
    kc = max(1, (BUD - SINK) // chunk)
    topc = np.argsort(-cs)[:kc]
    idxc = _np.concatenate([_np.arange(SINK), *[_np.arange(c * chunk, c * chunk + chunk) for c in topc]])
    idxc = mx.array(idxc.astype(_np.int32))
    print(f"{'ChunkKV (chunk=8)':<26} {mse(out_kept(Q,K,V,idxc), full):>12.5f} {mse(out_kept(Q,K,V,idxc),full)/base:>10.3f} {len(top_true&set(idxc.tolist()))/BUD:>10.3f}")
    # Keyformer: H2O + rumore di Gumbel sui logits (resuscita i late riser)
    g = -mx.log(-mx.log(mx.random.uniform(shape=(S,)) + 1e-9) + 1e-9)
    topkf = mx.argsort(-(mx.array(sc) + g * mx.mean(mx.abs(mx.array(sc)))))[: BUD - SINK]
    idxkf = mx.concatenate([mx.arange(SINK), topkf])
    print(f"{'Keyformer (H2O+Gumbel)':<26} {mse(out_kept(Q,K,V,idxkf), full):>12.5f} {mse(out_kept(Q,K,V,idxkf),full)/base:>10.3f} {len(top_true&set(topkf.tolist()))/BUD:>10.3f}")


if __name__ == "__main__":
    main()
