#!/usr/bin/env python3
"""Confronto PLE qwen4_exp: calcola l'output PLE del reference al layer PLE con input REALE.
Memory-safe (lazy + guardia RSS). Confronta con GMBasic (GMB_QWEN4EXP_PLE_DEBUG -> ple_out)."""
import os, sys, json, time, threading, resource
from pathlib import Path

LIMIT = int(os.environ.get("Q4_MEM_GB", "85")) * 1024**3
def _rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024
def _guard():
    while True:
        if _rss() > LIMIT:
            sys.stderr.write("[GUARD] kill\n"); os._exit(99)
        time.sleep(1)
threading.Thread(target=_guard, daemon=True).start()

sys.path.insert(0, "/Users/gaetano/ai/Rapid-MLX-official")
import numpy as np
import mlx.core as mx
from vllm_mlx.utils.tokenizer import _register_vendored_archs
from mlx_lm.utils import load_model

CKPT = Path("/Users/gaetano/ai/Models/Qwen3.8-Flash-Next-MLX-Serve-4bit")
_register_vendored_archs()
model, _ = load_model(CKPT, lazy=True, strict=False)
layers = model.model.layers
cache = model.make_cache()
ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

# layer 0 output reale (input del layer 1 = layer PLE)
captured = []
orig = type(layers[0]).__call__
def cap(self, *a, **k):
    out = orig(self, *a, **k); captured.append(out); return out
type(layers[0]).__call__ = cap
try:
    _ = model(ids, cache=cache)
finally:
    type(layers[0]).__call__ = orig
mx.eval(captured[0])
h0r = captured[0]                    # [1,4,hc_width] = output layer 0 (input layer 1)

# PLE al layer 1 (indice 1), input reale + ids reali
ple = layers[1].ple
emb = ple.ple_embedding(ids, None)
keys_raw = ple.key_proj(emb)
values = ple.value_proj(emb)
mx.eval(emb, keys_raw, values)
emb_a = np.asarray(emb.astype(mx.float32))
k_a = np.asarray(keys_raw.astype(mx.float32))
v_a = np.asarray(values.astype(mx.float32))
ple_out = ple(h0r, ids, None, None)  # [1,4,hc_width]
mx.eval(ple_out)
last = ple_out[:, -1, :]
mx.eval(last)
a = np.asarray(last.astype(mx.float32))[0]
mx.clear_cache()
print(json.dumps({
    "h0_last_absmax": round(float(np.abs(np.asarray(h0r[:, -1, :].astype(mx.float32))[0]).max()), 5),
    "emb_shape": list(emb_a.shape), "emb_absmax": round(float(np.abs(emb_a).max()), 6),
    "keys_shape": list(k_a.shape), "keys_absmax": round(float(np.abs(k_a).max()), 6),
    "values_shape": list(v_a.shape), "values_absmax": round(float(np.abs(v_a).max()), 6),
    "values_last_absmax": round(float(np.abs(v_a[0, -1]).max()), 6),
    "ple_out_absmax": round(float(np.abs(a).max()), 5),
    "ple_out_mean": round(float(a.mean()), 5),
}))
