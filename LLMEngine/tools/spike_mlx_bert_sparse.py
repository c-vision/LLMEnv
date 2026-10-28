#!/usr/bin/env python3
"""spike_mlx_bert_sparse.py — spike di fattibilità fase 4a (todo Tokenizer-WordPiece §6).

ESITO 2026-09-13: **PASS** — encoder bidirezionale cosine min 0.999998, testa MLM 1.000000,
top-10 sparse identico a torch, testa cls.predictions PRESENTE nel safetensors.
Riesecuzione: richiede `pip install mlx torch transformers huggingface_hub safetensors` e rete
per il download modello (cache HF; se Xet dà 404: HF_HUB_DISABLE_XET=1). Nessun file del repo
toccato dall'esecuzione. Promosso da /tmp il 2026-09-13 (§0quinquies: i tool vivono in tools/).

Domanda: MLX può eseguire un encoder BERT BIDIREZIONALE (attention non causale) con parità
numerica contro torch (cosine ≥ 0.9997), e il checkpoint doc-v2-mini conserva la testa MLM
(cls.predictions) che serve al retrieval sparse appreso?

Nessun file del repo GMBasic toccato: solo /tmp + cache HF. Non usa l'host C gmb_host_ai_mlx
(in modifica attiva da altra sessione) — dimostra il CALCOLO sull'engine MLX (python mlx =
stessi kernel Metal del C).
"""
import json
import sys

CANDIDATES = ["opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini", "google-bert/bert-base-uncased"]


def download():
    from huggingface_hub import snapshot_download
    last = None
    for repo in CANDIDATES:
        try:
            p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "pytorch_model.bin", "vocab.txt", "*.model"])
            print(f"[dl] {repo} -> {p}")
            return repo, p
        except Exception as e:  # noqa: BLE001
            print(f"[dl] {repo} FALLITO: {type(e).__name__}: {str(e)[:160]}")
            last = e
    raise SystemExit(f"nessun modello scaricabile ({last})")


def inspect_head(path):
    import glob, torch
    files = glob.glob(f"{path}/*.safetensors")
    if files:
        from safetensors import safe_open
        with safe_open(files[0], framework="np") as f:
            keys = list(f.keys())
    else:
        bins = glob.glob(f"{path}/pytorch_model.bin")
        if not bins:
            print("[head] nessun peso trovato")
            return None
        keys = list(torch.load(bins[0], map_location="cpu", weights_only=True).keys())
    mlm = [k for k in keys if k.startswith("cls.") or "predictions" in k]
    print(f"[head] chiavi totali={len(keys)} | testa MLM/cls: {mlm if mlm else 'ASSENTE'}")
    print(f"[head] sample chiavi: {keys[:6]}")
    return keys


def main():
    import numpy as np
    import mlx.core as mx
    import mlx.nn as mxnn
    import torch
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import load_file as st_load

    repo, path = download()
    keys = inspect_head(path)

    cfg = json.load(open(f"{path}/config.json"))
    print(f"[cfg] architectures={cfg.get('architectures')} hidden={cfg.get('hidden_size')} "
          f"layers={cfg.get('num_hidden_layers')} heads={cfg.get('num_attention_heads')}")

    text = "Il gatto nero dorme sul divano rosso mentre piove."
    tok = AutoTokenizer.from_pretrained(path)
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"]

    # ── riferimento torch ──
    ref_model = AutoModel.from_pretrained(path).eval()
    with torch.no_grad():
        ref = ref_model(**enc).last_hidden_state[0].numpy()   # (T, H) f32

    # ── encoder BERT in MLX (bidirezionale: NESSUNA causal mask) ──
    import os, glob as _g
    st_files = _g.glob(f"{path}/*.safetensors")
    if st_files:
        w = st_load(st_files[0])
    else:
        w = torch.load(f"{path}/pytorch_model.bin", map_location="cpu", weights_only=True)
    W = {(k[5:] if k.startswith("bert.") else k): mx.array(v.numpy()) for k, v in w.items()}

    H = cfg["hidden_size"]
    NL = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    DH = H // NH
    eps = cfg.get("layer_norm_eps", 1e-12)

    def layer_norm(x, wk, bk):
        return mx.fast.layer_norm(x, W[wk], W[bk], eps)

    def linear(x, pre):
        return x @ W[pre + ".weight"].T + W[pre + ".bias"]

    x = ids.numpy()[0]
    T = len(x)
    e = (W["embeddings.word_embeddings.weight"][mx.array(x)]
         + W["embeddings.position_embeddings.weight"][:T]
         + W["embeddings.token_type_embeddings.weight"][0])
    e = layer_norm(e, "embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias")

    h = e
    for i in range(NL):
        p = f"encoder.layer.{i}"
        q = linear(h, f"{p}.attention.self.query").reshape(T, NH, DH).transpose(1, 0, 2)
        k = linear(h, f"{p}.attention.self.key").reshape(T, NH, DH).transpose(1, 0, 2)
        v = linear(h, f"{p}.attention.self.value").reshape(T, NH, DH).transpose(1, 0, 2)
        # BIDIREZIONALE: softmax su TUTTE le chiavi (nessuna mask causale)
        att = mx.softmax(q @ k.transpose(0, 2, 1) / (DH ** 0.5), axis=-1) @ v
        att = att.transpose(1, 0, 2).reshape(T, H)
        att = linear(att, f"{p}.attention.output.dense")
        h1 = layer_norm(att + h, f"{p}.attention.output.LayerNorm.weight",
                        f"{p}.attention.output.LayerNorm.bias")
        inter = linear(h1, f"{p}.intermediate.dense")
        inter = mxnn.gelu(inter)
        out = linear(inter, f"{p}.output.dense")
        h = layer_norm(out + h1, f"{p}.output.LayerNorm.weight", f"{p}.output.LayerNorm.bias")

    got = np.array(h)

    # ── testa MLM in MLX (gate reale fase 4: sparse logits = encoder + head) ──
    hh = linear(h, "cls.predictions.transform.dense")
    hh = mxnn.gelu(hh)
    hh = layer_norm(hh, "cls.predictions.transform.LayerNorm.weight",
                    "cls.predictions.transform.LayerNorm.bias")
    logits_mlx = np.array(hh @ W["embeddings.word_embeddings.weight"].T
                          + W["cls.predictions.bias"])   # (T, vocab) — decoder tied

    # riferimento torch CON testa (BertForMaskedLM)
    from transformers import BertForMaskedLM
    tmodel = BertForMaskedLM.from_pretrained(path).eval()
    with torch.no_grad():
        logits_ref = tmodel(**enc).logits[0].numpy()

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    per_tok = [cos(got[t], ref[t]) for t in range(T)]
    per_logit = [cos(logits_mlx[t], logits_ref[t]) for t in range(T)]
    print(f"[esito] encoder: token={T} hidden={H} | cosine min={min(per_tok):.6f} "
          f"mean={sum(per_tok)/len(per_tok):.6f} | max|Δ|={float(np.max(np.abs(got - ref))):.2e}")
    print(f"[esito] head MLM: cosine min={min(per_logit):.6f} "
          f"mean={sum(per_logit)/len(per_logit):.6f} | max|Δ|={float(np.max(np.abs(logits_mlx - logits_ref))):.2e}")

    # embedding sparse finale (SPLADE-style): max-pool sui token → log1p(relu) → top-10
    def sparse(logits):
        pooled = np.maximum(logits.max(axis=0), 0.0)
        w = np.log1p(pooled)
        top = np.argsort(-w)[:10]
        return {int(i): round(float(w[i]), 4) for i in top if w[i] > 0}
    sm, sr = sparse(logits_mlx), sparse(logits_ref)
    print(f"[esito] sparse top-10 MLX : {sorted(sm.items(), key=lambda x:-x[1])[:6]}")
    print(f"[esito] sparse top-10 torch: {sorted(sr.items(), key=lambda x:-x[1])[:6]}")
    stessi = set(sm) == set(sr)
    print(f"[esito] top-10 term-set identici: {stessi}")

    ok = min(per_tok) >= 0.9997 and min(per_logit) >= 0.9997
    print(f"[esito] spike {'PASS' if ok else 'FAIL'} (soglia 0.9997 encoder+head, modello {repo})")
    if keys is not None:
        has_mlm = any(k.startswith("cls.predictions") for k in keys)
        print(f"[esito] testa MLM cls.predictions nel safetensors: {'PRESENTE' if has_mlm else 'ASSENTE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
