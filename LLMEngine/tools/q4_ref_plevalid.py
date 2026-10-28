#!/usr/bin/env python3
"""Reference qwen4_exp con PLE VALIDO: patcha ShardedEmbedding per leggere ngram_table.bin
(4-bit affine g32) invece dei nn.Embedding random. Poi confronta con GMBasic (PLE ON).
Memory-safe (lazy + guardia)."""
import os, sys, json, time, threading, resource, struct
from pathlib import Path
LIMIT = int(os.environ.get("Q4_MEM_GB", "85")) * 1024**3
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
import vllm_mlx.models.qwen4_exp as Q
from mlx_lm.utils import load_model

CKPT = Path("/Users/gaetano/ai/Models/Qwen3.8-Flash-Next-MLX-Serve-4bit")
NG = str(CKPT / "ngram_table.bin")
_f = open(NG, "rb")
_hlen = struct.unpack("<Q", _f.read(8))[0]
_hdr = json.loads(_f.read(_hlen)); _base = 8 + _hlen
_wo = _hdr["weight"]["data_offsets"][0]; _so = _hdr["scales"]["data_offsets"][0]; _bo = _hdr["biases"]["data_offsets"][0]
def _bf16(u): return (u.astype(np.uint32) << 16).view(np.float32)
def _row(row):
    _f.seek(_base + _wo + row * 80); w = np.frombuffer(_f.read(80), dtype=np.uint32)
    _f.seek(_base + _so + row * 10); s = np.frombuffer(_f.read(10), dtype=np.uint16)
    _f.seek(_base + _bo + row * 10); b = np.frombuffer(_f.read(10), dtype=np.uint16)
    sc = np.repeat(_bf16(s), 32); bi = np.repeat(_bf16(b), 32)
    codes = np.array([(int(w[i // 8]) >> ((i % 8) * 4)) & 0xF for i in range(160)], dtype=np.int32)
    return (codes * sc + bi).astype(np.float32)
def _patched(self, indices):
    idx = np.asarray(indices).reshape(-1).astype(np.int64)
    out = np.stack([_row(int(r)) for r in idx]).reshape(*indices.shape, 160)
    return mx.array(out)
Q.ShardedEmbedding.__call__ = _patched

_register_vendored_archs()
model, _ = load_model(CKPT, lazy=True, strict=False)
layers = model.model.layers
cache = model.make_cache()
ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
captured = []
orig = type(layers[0]).__call__
def cap(self, *a, **k):
    out = orig(self, *a, **k); captured.append(out[:, -1, :]); return out
type(layers[0]).__call__ = cap
try:
    out = model(ids, cache=cache)
finally:
    type(layers[0]).__call__ = orig
N = int(os.environ.get("Q4_N", "48"))
captured = captured[:min(N, len(captured))]
mx.eval(captured)
res = []
for i in range(len(captured)):
    a = np.asarray(captured[i].astype(mx.float32))[0]
    res.append([round(float(np.abs(a).max()), 5), round(float(a.mean()), 5)])
logits = getattr(out, "logits", out)
last = logits[:, -1, :]
mx.eval(last)
greedy = int(mx.argmax(last, axis=-1).item())
mx.clear_cache()
print(json.dumps({"per_layer": res, "short_greedy_for_1_2_3_4": greedy}))
