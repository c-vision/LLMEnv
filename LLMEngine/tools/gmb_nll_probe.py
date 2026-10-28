#!/usr/bin/env python3
"""gmb_nll_probe.py — misura la NLL (negative log-likelihood, piu' basso = meglio) di un modello MLX
su un corpus di testo fisso. Serve a confrontare varianti di quantizzazione (§E3 mixed per-layer,
§E2 imatrix) e modelli.

Nato dalla sessione §E3 (2026-09-15): il confronto va fatto con lo STESSO corpus e gli stessi token.

Uso:
  python gmb_nll_probe.py <model_dir> [<model_dir> ...] [--corpus FILE] [--ctx 512] [--ntok 4096]

- `--corpus` default = `AGENTS.md` nella CWD (prosa reale; NON usare file degenerati: un corpus di
  soli newline falsa il confronto).
- Richiede `mlx_lm` (usare la venv ~/ai/env-mlxlm: mlx-lm 0.29.1 + transformers 4.57.6).
- Modelli: cartelle MLX (f16, 4/8-bit, mixed con `quantization` per-path).
"""
import argparse
import os
import sys


def nll(path, text, tok_encode, model_fn, ctx, ntok):
    ids = tok_encode(text)[:ntok]
    total, n = 0.0, 0
    for i in range(0, len(ids) - 3, ctx):
        chunk = ids[i:i + ctx]
        tgt = ids[i + 1:i + ctx + 1]
        if not tgt:
            break
        logits = model_fn(chunk)[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sel = lp[mx.arange(len(tgt)), mx.array(tgt)]
        total += -mx.sum(sel).item()
        n += len(tgt)
        if n >= ntok:
            break
    return total / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--corpus", default="AGENTS.md")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--ntok", type=int, default=4096)
    args = ap.parse_args()

    global mx
    import mlx.core as mx  # noqa: E402
    from mlx_lm import load  # noqa: E402

    if not os.path.exists(args.corpus):
        sys.exit(f"corpus non trovato: {args.corpus} (passa --corpus FILE)")
    text = open(args.corpus, encoding="utf-8", errors="ignore").read()
    if len(text.strip()) < 1000:
        print(f"[warn] corpus molto corto/degenere ({len(text)} char): il confronto puo' non essere significativo")

    print(f"corpus: {args.corpus} ({len(text)} char) | ctx={args.ctx} ntok={args.ntok}")
    for p in args.models:
        try:
            model, tok = load(p)
            import glob as _g
            size = sum(os.path.getsize(f) for f in _g.glob(os.path.join(p, "*.safetensors"))) / 1e6
            v = nll(p, text, tok.encode, lambda ids: model(mx.array([ids])), args.ctx, args.ntok)
            print(f"{os.path.basename(p.rstrip('/')):14s} NLL={v:.4f}  ({size:.0f} MB)")
            del model
            mx.clear_cache()
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(p.rstrip('/')):14s} ERRORE: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
