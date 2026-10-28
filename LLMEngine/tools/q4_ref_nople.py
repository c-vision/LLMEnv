#!/usr/bin/env python3
"""Reference qwen4_exp SENZA PLE (layers[1].ple=None), lazy+guardia: confronto valido dei
layer GDN/QSA/MoE/HC con GMBasic (GMB_QWEN4EXP_PLE_OFF=1). Memory-safe."""
import os, sys, json, time, threading, resource
from pathlib import Path
LIMIT = int(os.environ.get("Q4_MEM_GB", "85")) * 1024**3
N = int(os.environ.get("Q4_N", "6"))
def _rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024
def _guard():
    while True:
        if _rss() > LIMIT: sys.stderr.write("[GUARD] kill\n"); os._exit(99)
        time.sleep(1)
threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Users/gaetano/ai/Rapid-MLX-official")
import numpy as np, mlx.core as mx
from vllm_mlx.utils.tokenizer import _register_vendored_archs
from mlx_lm.utils import load_model
CKPT = Path("/Users/gaetano/ai/Models/Qwen3.8-Flash-Next-MLX-Serve-4bit")
_register_vendored_archs()
model, _ = load_model(CKPT, lazy=True, strict=False)
layers = model.model.layers
# PLE OFF: la reference non ha la tabella ngram -> la disattiviamo per un confronto VALIDO.
for li in range(len(layers)):
    if getattr(layers[li], "ple", None) is not None:
        layers[li].ple = None
        sys.stderr.write(f"[ref] PLE disattivato al layer {li}\n")
captured = []
orig = type(layers[0]).__call__
def cap(self, *a, **k):
    out = orig(self, *a, **k); captured.append(out[:, -1, :]); return out
type(layers[0]).__call__ = cap
cache = model.make_cache()
ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
try:
    _ = model(ids, cache=cache)
finally:
    type(layers[0]).__call__ = orig
captured = captured[:min(N, len(captured))]
mx.eval(captured)
res = []
for i, v in enumerate(captured):
    a = np.asarray(v.astype(mx.float32))[0]
    res.append([round(float(np.abs(a).max()), 5), round(float(a.mean()), 5)])
mx.clear_cache()
print(json.dumps({"n_layers_total": len(layers), "n_captured": len(captured), "per_layer": res}))
