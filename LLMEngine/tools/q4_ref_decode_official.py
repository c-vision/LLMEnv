#!/usr/bin/env python3
"""Reference greedy decode for the OFFICIAL mlx-community qwen4_exp checkpoint
(inline ngram shards), prompt = a single token id (default 9419), 8 greedy tokens.

Mirrors LLMEnv/LLMEngine/mlx session_append(prefill)+session_generate(8,-1):
prefill the prompt, then greedily argmax one token at a time, feeding each back.
Captures, for every step: full last-position logits (f32), top-8 (id,logit), and
the last-position hidden state after each decoder layer -- so the first decode
step where our engine diverges can be localized layer by layer.

Memory-safe: lazy load + RSS guard; backbone pretouched outside Metal command
buffers; ngram shards pretouched only for the ids actually used (never 32GB).

Usage:
  PYTHONPATH=/Users/gaetano/Documents/Projects/Rapid-MLX-NextGen \
  ~/ai/env-rapid-official/bin/python tools/q4_ref_decode_official.py \
      --checkpoint <model_dir> --token 9419 --n 8 --output /tmp/q4_ref_decode.npz
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
        if rss_bytes() > LIMIT:
            sys.stderr.write(f"[GUARD] RSS {rss_bytes()/1e9:.1f}GB > {LIMIT/1e9:.0f}GB -> kill\n")
            sys.stderr.flush()
            os._exit(99)
        time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--token", type=int, default=9419)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("/tmp/q4_ref_decode.npz"))
    args = ap.parse_args()

    threading.Thread(target=guard, daemon=True).start()

    import numpy as np
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from vllm_mlx.utils.tokenizer import _register_vendored_archs
    from mlx_lm.utils import load_model

    _register_vendored_archs()

    def pretouch(module):
        for path, arr in tree_flatten(module.parameters()):
            p = path if isinstance(path, str) else ".".join(map(str, path))
            if "ngram_embedding.shards" in p:
                continue
            mx.eval(arr)

    sys.stderr.write("[ref] lazy load...\n"); sys.stderr.flush()
    model, _ = load_model(args.checkpoint.resolve(), lazy=True, strict=False)
    layers = model.model.layers
    nl = len(layers)
    sys.stderr.write(f"[ref] loaded lazy: {nl} layers, RSS={rss_bytes()/1e9:.1f}GB\n"); sys.stderr.flush()

    t0 = time.time()
    pretouch(model)
    sys.stderr.write(f"[ref] backbone pretouched in {time.time()-t0:.0f}s, RSS={rss_bytes()/1e9:.1f}GB\n")
    sys.stderr.flush()

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
    except Exception as exc:
        sys.stderr.write(f"[ref] ShardedEmbedding pretouch hook non installato: {exc}\n")

    cache = model.make_cache()

    steps_tokens: list[int] = []
    steps_logits: list = []
    steps_topk: list = []
    steps_hidden: list = []

    # Cattura l'hidden last-position dopo OGNI layer per ogni forward (prefill + decode),
    # per localizzare il layer del primo step divergente. L'hook non altera i numeri.
    cap_active = [True]
    cap_buf: list = []
    orig_layer_call = type(layers[0]).__call__

    def cap_layer(self, *a, **k):
        o = orig_layer_call(self, *a, **k)
        if cap_active[0]:
            cap_buf.append(o[:, -1, :])
        return o

    type(layers[0]).__call__ = cap_layer

    def logits_of(out):
        return out if not hasattr(out, "logits") else out.logits

    ids = mx.array([[args.token]], dtype=mx.int32)
    out = model(ids, cache=cache)
    lg = logits_of(out)
    mx.eval(lg)

    for step in range(args.n):
        lg_last = lg[:, -1, :]
        mx.eval(lg_last)
        mx.eval(*cap_buf) if cap_buf else None
        steps_hidden.append(np.stack([np.asarray(v.astype(mx.float32))[0] for v in cap_buf]))
        cap_buf.clear()
        order = mx.argsort(lg_last, axis=-1)[0]
        top = order[-8:][::-1]
        top_ids = [int(v) for v in np.asarray(top)]
        top_vals = [float(v) for v in np.asarray(lg_last[0][top].astype(mx.float32))]
        nxt = top_ids[0]
        steps_tokens.append(nxt)
        steps_logits.append(np.asarray(lg_last.astype(mx.float32))[0])
        steps_topk.append({"ids": top_ids, "logits": top_vals})
        sys.stderr.write(f"[ref] step {step}: next={nxt} top8={list(zip(top_ids, [round(v,4) for v in top_vals]))}\n")
        sys.stderr.flush()
        if step == args.n - 1:
            break
        out = model(mx.array([[nxt]], dtype=mx.int32), cache=cache)
        lg = logits_of(out)
        mx.eval(lg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        tokens=np.asarray(steps_tokens, dtype=np.int32),
        logits=np.stack(steps_logits),
        hidden=np.stack(steps_hidden),
    )
    print(json.dumps({
        "n_layers": nl,
        "prompt_token": args.token,
        "tokens": steps_tokens,
        "topk": steps_topk,
        "rss_gb": rss_bytes() / 1e9,
    }, indent=1))
    sys.stderr.write(f"[ref] done, RSS={rss_bytes()/1e9:.1f}GB\n"); sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
