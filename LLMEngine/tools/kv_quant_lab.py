#!/usr/bin/env python3
"""KV-quant compression lab — validazione dei metodi zero-calibration di VeloxQuant (43)
applicabili a GMBasic (gmb_host_ai_mlx). Confronta l'errore di ricostruzione (MSE) di K/V
su distribuzioni REALISTICHE (outlier per-canale su K, per-token su V, heavy-tail).

Uso:  /Users/gaetano/ai/env312/bin/python tools/kv_quant_lab.py

I vincenti (tuning-free, no calibrazione) vanno poi implementati come primitive C gated in
llm_mlx_ops.h (come GMB_KVROT / GMB_KIVI). Vedi Knowledge/VeloxQuant-MLX-... §9/§10.
"""
import mlx.core as mx

mx.set_default_device(mx.gpu)
BITS, GS = 4, 64
S, D = 1024, 128


def mse(a, b):
    return float(mx.mean((a - b) ** 2))


def q(x, bits, gs):
    """affine group-quant lungo l'ultimo asse (il default di GMB/MLX)."""
    w, s, b = mx.quantize(x.astype(mx.float32), group_size=gs, bits=bits)
    return mx.dequantize(w, s, b, group_size=gs, bits=bits).astype(mx.float32)


def hadamard(dim):
    if dim & (dim - 1):
        return None
    H = mx.array([[1.0]])
    while H.shape[0] < dim:
        H = mx.concatenate([mx.concatenate([H, H], 1), mx.concatenate([H, -H], 1)], 0)
    return H / (dim ** 0.5)


def rot(x, H):
    return (x.reshape(-1, x.shape[-1]) @ H).reshape(x.shape) if H is not None else x


# --- metodi (tuning-free) -------------------------------------------------
def m_baseline(kv, bits=BITS, gs=GS):
    return q(kv, bits, gs)


def m_rot(kv, bits=BITS, gs=GS):
    H = hadamard(kv.shape[-1])
    return rot(q(rot(kv, H), bits, gs), H.T) if H is not None else q(kv, bits, gs)


def m_kivi_k(kv, bits=BITS, gs=GS):
    """K per-canale: quantizza sull'asse TOKEN (trasposto)."""
    return q(kv.T, bits, gs).T


def m_zipcache(kv, bits=BITS, gs=GS, hi=0.25):
    """ZipCache: mixed-precision per-TOKEN guidata dalla norma L2 (saliency)."""
    n = kv.shape[0]
    norms = mx.linalg.norm(kv, axis=-1)
    order = mx.argsort(norms)
    lo_idx = order[: int(n * (1 - hi))]
    out = q(kv, bits, gs)
    hi_q = q(kv, bits + 2, gs)
    idx = mx.zeros((n,), dtype=mx.int32)
    idx = idx.at[lo_idx].add(1)  # lo_idx=1 -> usa il low-bit
    sel = idx.reshape((n,) + (1,) * (kv.ndim - 1))
    return mx.where(sel.astype(mx.bool_), out, hi_q)


def m_kitty(kv, bits=BITS, gs=GS, hi=0.5):
    """Kitty: mixed-precision per-CANALE guidata dalla varianza online."""
    var = mx.var(kv, axis=0)
    thresh = mx.quantile_ish = mx.sort(var.reshape(-1))[int(var.size * (1 - hi))]
    hi_c = (var >= thresh).astype(mx.float32).reshape((1,) + var.shape)
    lo = q(kv, bits, gs)
    hq = q(kv, bits + 2, gs)
    return mx.where(hi_c.astype(mx.bool_), hq, lo)


def m_skvq(kv, bits=BITS, gs=GS):
    """SKVQ: riordino canali per dynamic range + clip per-group."""
    import numpy as np

    rng = kv.max(axis=0) - kv.min(axis=0)
    perm = mx.argsort(-rng)
    inv = mx.argsort(perm)
    reordered = kv[:, perm]
    lo = mx.array(np.percentile(np.array(reordered), 1, axis=0))
    hi = mx.array(np.percentile(np.array(reordered), 99, axis=0))
    clipped = mx.clip(reordered, lo, hi)
    return q(clipped, bits, gs)[:, inv]


def m_turboquant(kv, bits=BITS, gs=GS):
    """TurboQuant RVQ: codebook ANALITICO Gaussiano (uniforme-in-probabilità) + residuo."""
    levels = (1 << bits) - 1
    import numpy as np

    book = mx.array(np.array([float(v) for v in np.linspace(-1, 1, levels + 1)]))
    scale = mx.linalg.norm(kv, axis=-1, keepdims=True) / (D ** 0.5)
    xn = kv / (scale + 1e-6)
    idx = mx.argmin(mx.abs(xn[:, :, None] - book[None, None, :]), axis=-1)
    rec = book[idx] * scale
    return rec.astype(mx.float32)


def m_qjl(kv, bits=BITS, gs=GS):
    """QJL: sketch 1-bit Johnson-Lindenstrauss (sign di una proiezione casuale)."""
    import numpy as np

    G = mx.array(np.random.RandomState(0).normal(size=(kv.shape[-1], D)).astype(np.float32))
    sk = mx.sign(kv @ G)
    norm = mx.linalg.norm(kv, axis=-1, keepdims=True)
    rec = (sk @ G.T) / D * norm * (mx.pi / 2) ** 0.5
    return rec.astype(mx.float32)


def m_rabitq(kv, bits=1, gs=GS):
    """RaBitQ: rotazione Hadamard randomizzata + 1-bit (sign) + norma esatta."""
    H = hadamard(kv.shape[-1])
    r = rot(kv, H)
    norm = mx.linalg.norm(kv, axis=-1, keepdims=True)
    rec = mx.sign(r) * norm / (D ** 0.5)
    return rot(rec, H.T)


def m_nsn(kv, bits=BITS, gs=GS):
    """NSNQuant: Normalize-Shift-Normalize + rotazione, poi quant."""
    mu = mx.mean(kv, axis=-1, keepdims=True)
    sd = mx.std(kv, axis=-1, keepdims=True) + 1e-6
    x = (kv - mu) / sd
    H = hadamard(kv.shape[-1])
    x = rot(x, H)
    return rot(q(x, bits, gs), H.T) * sd + mu


def m_nuq(kv, bits=BITS, gs=GS):
    """KVQuant-NUQ: codebook NON-uniforme per-group (quantili dei dati = Lloyd-Max online)."""
    import numpy as np

    levels = 1 << bits
    x = np.array(kv)
    out = np.empty_like(x)
    for g0 in range(0, x.shape[-1], gs):
        g = x[..., g0 : g0 + gs]
        qs = np.quantile(g, np.linspace(0, 1, levels))
        idx = np.abs(g[..., None] - qs).argmin(-1)
        out[..., g0 : g0 + gs] = qs[idx]
    return mx.array(out).astype(mx.float32)


def m_gear(kv, bits=BITS, gs=GS, rank=8):
    """GEAR: quant + low-rank (SVD) del residuo di quantizzazione."""
    base = q(kv, bits, gs)
    res = (kv - base).astype(mx.float32)
    U, S, Vh = mx.linalg.svd(res, stream=mx.cpu)
    r = min(rank, S.shape[0])
    lowrank = (U[:, :r] * S[:r]) @ Vh[:r]
    return base + lowrank.astype(mx.float32)


def main():
    mx.random.seed(1)
    H = hadamard(D)
    # K: outlier PERSISTENTI per-canale (regime KIVI/Kitty realistico)
    kscale = mx.abs(mx.random.normal(shape=(1, D))) * 2.0 + 0.4
    K = mx.random.normal(shape=(S, D)) * kscale
    # V: outlier per-TOKEN
    vscale = mx.abs(mx.random.normal(shape=(S, 1))) * 2.0 + 0.4
    V = mx.random.normal(shape=(S, D)) * vscale

    print(f"{'metodo':<18} {'MSE K (per-canale+tail)':>24} {'MSE V (per-token)':>20}")
    print("-" * 66)
    base_k = mse(m_baseline(K), K)
    base_v = mse(m_baseline(V), V)
    print(f"{'baseline':<18} {base_k:>24.5f} {base_v:>20.5f}")
    for name, fk, fv in [
        ("rot(Hadamard)", m_rot, m_rot),
        ("KIVI (K per-canale)", m_kivi_k, m_baseline),
        ("ZipCache per-token", m_zipcache, m_zipcache),
        ("Kitty per-canale", m_kitty, m_kitty),
        ("SKVQ reorder+clip", m_skvq, m_skvq),
        ("NSNQuant (rot+norm)", m_nsn, m_nsn),
        ("KVQuant-NUQ", m_nuq, m_nuq),
        ("GEAR (quant+lowrank)", m_gear, m_gear),
        ("QJL (1-bit sketch)", m_qjl, m_qjl),
        ("RaBitQ (rot+1bit)", m_rabitq, m_rabitq),
        ("TurboQuant RVQ", m_turboquant, m_turboquant),
    ]:
        mk, mv = mse(fk(K), K), mse(fv(V), V)
        print(f"{name:<18} {mk:>24.5f} {mv:>20.5f}   (K ratio {mk/base_k:.3f})")


if __name__ == "__main__":
    main()
