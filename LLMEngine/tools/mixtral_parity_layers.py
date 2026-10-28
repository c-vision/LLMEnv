#!/usr/bin/env python3
"""mixtral_parity_layers.py — localizza la PRIMA divergenza layer-per-layer: engine MLX vs mlx_lm.

Confronta il dump **post-attenzione** dell'engine (GMB_DEBUG_HIDDIR -> /tmp/mxhid/l<N>.bin, f32,
64 token x hidden) con lo stesso stato ricostruito dall'oracolo mlx_lm sullo stesso checkpoint e
sugli stessi 64 token sintetici del bench ((i%1000)+1).

Uso:  python3 tools/mixtral_parity_layers.py <model_dir> [enginedir] [ntok] [nlayers]
      (default: /Users/gaetano/ai/Models/Mixtral-8x22B-4bit /tmp/mxhid 64 32)
"""
import sys, os, numpy as np

def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/gaetano/ai/Models/Mixtral-8x22B-4bit"
    eng = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mxhid"
    ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    nlay = int(sys.argv[4]) if len(sys.argv) > 4 else 32

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask

    model, _tok = load(model_dir)
    inner = getattr(model, "model", model)          # il MixtralModel interno
    print("struttura: model=%s inner=%s n_layers=%d" % (type(model).__name__, type(inner).__name__,
          len(getattr(inner, "layers", []))), flush=True)
    l0 = inner.layers[0]
    print("layer[0] attrs:", [a for a in dir(l0) if not a.startswith('_')][:14], flush=True)

    ids = [(i % 1000) + 1 for i in range(ntok)]
    x = mx.array([ids])
    h = inner.embed_tokens(x)
    cache = model.make_cache() if hasattr(model, "make_cache") else []
    mask = None

    worst = []
    for i, layer in enumerate(inner.layers):
        if i >= nlay: break
        try:
            mask = create_attention_mask(h, cache[i] if i < len(cache) else None)
        except Exception:
            mask = None
        h = h + layer.self_attn(layer.input_layernorm(h), mask, cache[i] if i < len(cache) else None)
        o = np.array(h).astype(np.float32).reshape(-1)
        p = os.path.join(eng, "l%d.bin" % i)
        if os.path.exists(p):
            e = np.fromfile(p, dtype=np.float32)
            n = min(e.size, o.size)
            d = np.abs(e[:n] - o[:n])
            mx_ = float(d.max()); mean = float(d.mean())
            worst.append((i, mx_, mean, float(np.abs(o[:n]).max())))
            print("  layer %2d: maxdiff=%.6g mean=%.6g (|oracolo|max=%.4g)%s" %
                  (i, mx_, mean, worst[-1][3], "   <<< PRIMA DIVERGENZA" if mx_ > 0.05 and not any(w[1] > 0.05 for w in worst[:-1]) else ""), flush=True)
        else:
            print("  layer %2d: (nessun dump engine)" % i, flush=True)
        h = h + layer.block_sparse_moe(layer.post_attention_layernorm(h))

    print("== sintesi ==")
    for i, mx_, mean, omax in worst:
        print("  l%-2d maxdiff=%.6g" % (i, mx_))

if __name__ == "__main__":
    main()
