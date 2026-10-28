#!/usr/bin/env python3
"""Reference per-layer hidden states for the OFFICIAL mlx-community qwen4_exp
checkpoint (inline ngram shards), prompt = a single token id (default 9419).

Runs the vendored rapid `vllm_mlx.models.qwen4_exp` via `mlx_lm.load_model`
(lazy=True so the 104GB backbone is materialized on demand, never as a second
resident copy) and captures, for the LAST token of a single forward, the
embedding (post-tile) and the output of every DecoderLayer.

Usage (memory-safe, one model at a time):
  PYTHONPATH=/Users/gaetano/Documents/Projects/Rapid-MLX-NextGen \
  ~/ai/env-rapid-official/bin/python tools/q4_ref_layers_official.py \
      --checkpoint <model_dir> --token 9419 --output /tmp/q4_ref_layers.npz
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

LIMIT = int(os.environ.get("Q4_MEM_GB", "118")) * 1024**3


def rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def guard() -> None:
    while True:
        r = rss_bytes()
        if r > LIMIT:
            sys.stderr.write(f"[GUARD] RSS {r/1e9:.1f}GB > {LIMIT/1e9:.0f}GB -> kill\n")
            sys.stderr.flush()
            os._exit(99)
        time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--token", type=int, default=9419)
    ap.add_argument("--output", type=Path, default=Path("/tmp/q4_ref_layers.npz"))
    args = ap.parse_args()

    threading.Thread(target=guard, daemon=True).start()

    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from vllm_mlx.utils.tokenizer import _register_vendored_archs
    from mlx_lm.utils import load_model

    _register_vendored_archs()

    def pretouch(module):
        """Materializza i pesi di `module` FUORI dal command buffer Metal.

        Con lazy=True i pesi sono mmap sul volume esterno (~67MB/s); se la prima
        lettura avviene DENTRO un command buffer, Metal va in timeout
        (kIOGPUCommandBufferCallbackErrorTimeout) perche' il buffer resta in volo
        decine di secondi. Eval di array puri (nessuna compute) li pagina prima."""
        for path, arr in tree_flatten(module.parameters()):
            p = path if isinstance(path, str) else ".".join(map(str, path))
            if "ngram_embedding.shards" in p:
                continue  # gestiti a parte (solo gli shard usati, mai i 32GB)
            mx.eval(arr)
    sys.stderr.write("[ref] lazy load...\n"); sys.stderr.flush()
    model, _ = load_model(args.checkpoint.resolve(), lazy=True, strict=False)
    layers = model.model.layers
    nl = len(layers)
    sys.stderr.write(f"[ref] loaded lazy: {nl} layers, RSS={rss_bytes()/1e9:.1f}GB\n"); sys.stderr.flush()

    # Pretouch dell'INTERO backbone (tutti i pesi tranne i 32GB di ngram shard) PRIMA del
    # forward: cosi' nessuna prima-lettura lazy finisce dentro un command buffer Metal
    # (era la causa del GPU timeout). ~72GB letti dal volume esterno, una volta.
    t0 = time.time()
    pretouch(model)
    sys.stderr.write(f"[ref] backbone pretouched in {time.time()-t0:.0f}s, RSS={rss_bytes()/1e9:.1f}GB\n")
    sys.stderr.flush()

    ids = mx.array([[args.token]], dtype=mx.int32)
    hc = int(model.model.args.hc_count)
    emb = mx.tile(model.model.embed_tokens(ids), (1, 1, hc))

    captured: list = []
    orig = type(layers[0]).__call__

    def cap(self, *a, **k):
        out = orig(self, *a, **k)
        v = out[:, -1, :]
        # Eval per layer: senza questo mlx costruisce un unico grafo da 48 layer e il
        # command buffer Metal va in timeout. Eval dell'INTERO output del layer (non solo
        # la slice) cosi' il layer successivo non deve ricalcolare nulla.
        mx.eval(out, v)
        captured.append(v)
        return out

    # Pretouch degli ngram shard EFFETTIVAMENTE usati dagli id (al piu' 16), mai i 32GB.
    try:
        from vllm_mlx.models.qwen4_exp import ShardedEmbedding

        _se_orig = ShardedEmbedding.__call__

        def _se_cap(self, indices):
            rps = self.rows_per_shard
            used = sorted({int(v) for v in (indices // rps).reshape(-1).tolist()})
            for sid in used:
                pretouch(self.shards[int(sid)])
            return _se_orig(self, indices)

        ShardedEmbedding.__call__ = _se_cap
    except Exception as exc:  # non fatale: se manca, il PLE paginera' lazy
        sys.stderr.write(f"[ref] ShardedEmbedding pretouch hook non installato: {exc}\n")

    type(layers[0]).__call__ = cap
    try:
        cache = model.make_cache()
        out = model(ids, cache=cache)
    finally:
        type(layers[0]).__call__ = orig

    mx.eval(emb)
    mx.eval(captured)
    mx.eval(out)
    emb_last = np.asarray(emb.astype(mx.float32))[0, -1, :]
    layer_last = np.stack([np.asarray(v.astype(mx.float32))[0] for v in captured])

    def stats(v):
        return dict(absmax=float(np.abs(v).max()), mean=float(v.mean()), sum=float(v.sum()),
                    first4=[float(x) for x in v[:4]])

    logits = out[:, -1, :] if not hasattr(out, "logits") else out.logits[:, -1, :]
    greedy = int(mx.argmax(logits, axis=-1).item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, emb_last=emb_last, layer_last=layer_last)
    print(json.dumps({
        "n_layers": nl,
        "emb": stats(emb_last),
        "per_layer": [stats(layer_last[i]) for i in range(nl)],
        "greedy_next": greedy,
        "rss_gb": rss_bytes() / 1e9,
    }, indent=1))
    sys.stderr.write(f"[ref] done, RSS={rss_bytes()/1e9:.1f}GB\n"); sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
