#!/usr/bin/env python3
"""Reference qwen4_exp MEMORY-SAFE (lazy) per F3.

Carica con lazy=True (i pesi restano mmap/lazy) e valuta SOLO l'hidden per-layer fino a
Q4_N layer: materializza pochi GB, mai i 98 GB. Guardia di memoria (RSS) che uccide il
processo se supera Q4_MEM_GB (default 85). Input FISSO [1,2,3,4] (stesso di ctxdecode 4).

Uso:
  PYTHONPATH=~/ai/Rapid-MLX-official Q4_N=4 ~/ai/env-rapid-official/bin/python /tmp/q4_ref_lazy.py
"""
import os, sys, json, time, threading, resource
from pathlib import Path

LIMIT = int(os.environ.get("Q4_MEM_GB", "85")) * 1024**3
N = int(os.environ.get("Q4_N", "4"))

def _rss_bytes() -> int:
    # macOS: ru_maxrss in byte; Linux: KiB
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024

def _guard() -> None:
    while True:
        r = _rss_bytes()
        if r > LIMIT:
            sys.stderr.write(f"[GUARD] RSS {r/1e9:.1f} GB > {LIMIT/1e9:.0f} GB -> kill\n")
            sys.stderr.flush()
            os._exit(99)
        time.sleep(1)

threading.Thread(target=_guard, daemon=True).start()

sys.path.insert(0, "/Users/gaetano/ai/Rapid-MLX-official")
import numpy as np
import mlx.core as mx
from vllm_mlx.utils.tokenizer import _register_vendored_archs
from mlx_lm.utils import load_model

CKPT = Path("/Users/gaetano/ai/Models/Qwen3.8-Flash-Next-MLX-Serve-4bit")
_register_vendored_archs()
sys.stderr.write("[ref] lazy load...\n"); sys.stderr.flush()
model, _ = load_model(CKPT, lazy=True, strict=False)
layers = model.model.layers
nl = len(layers)
sys.stderr.write(f"[ref] loaded lazy: {nl} layer, RSS={_rss_bytes()/1e9:.1f}GB\n"); sys.stderr.flush()

captured = []
orig = type(layers[0]).__call__
def cap(self, *a, **k):
    out = orig(self, *a, **k)
    captured.append(out[:, -1, :])
    return out
type(layers[0]).__call__ = cap

cache = model.make_cache()
ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
try:
    out = model(ids, cache=cache)
finally:
    type(layers[0]).__call__ = orig

captured = captured[:min(N, len(captured))]
# Valuta SOLO i primi N layer (materializza solo quei pesi).
mx.eval(captured)
layer_last = np.stack([np.asarray(v.astype(mx.float32))[0] for v in captured])
mx.clear_cache()
stats = [[float(np.abs(v).max()), float(v.mean())] for v in layer_last]
sys.stderr.write(f"[ref] evaluated {len(captured)} layer, RSS={_rss_bytes()/1e9:.1f}GB\n"); sys.stderr.flush()
np.savez("/tmp/q4_ref_lazy.npz", layer_last=layer_last)
print(json.dumps({
    "n_layers_total": nl,
    "n_captured": len(captured),
    "per_layer_absmax_mean": [[round(a, 5), round(m, 5)] for a, m in stats],
}))
