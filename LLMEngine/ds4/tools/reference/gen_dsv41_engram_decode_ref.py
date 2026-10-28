#!/usr/bin/env python3
"""gen_dsv41_engram_decode_ref.py — riferimento tiny prefill+decode CON Engram (no download).

Stesso modello di gen_dsv41_engram_fwd_ref (engram su layer1, token_map identita') ma con cache
persistente: prefill S=3 + 5 decode, per verificare Engram nel path decode (history slot 6).

Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_engram_decode_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_engram_decode_ref.h
"""
import os
import math
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel

    VOCAB, DEPTH, HEADS, HD, COMP = 20, 1, 2, 8, 64
    seen, cur = set(), VOCAB - 1
    for _ in range(HEADS):
        cur += 1
        while not is_prime(cur) or cur in seen:
            cur += 1
        seen.add(cur)
    num_emb = sum(seen)

    cfg = ModelConfig(
        dim=8, n_layers=2, n_heads=2, o_groups=2, head_dim=32, rope_head_dim=4,
        index_head_dim=32, index_n_heads=2, q_lora_rank=8, o_lora_rank=8, moe_inter_dim=8,
        n_routed_experts=4, n_activated_experts=2, n_shared_experts=1, window_size=4,
        compress_ratios=(0, 0), kv_source_layers=(), index_source_layers=(),
        candidate_source_layer=-1, index_topk=64,
        engram_layer_ids=(1,), engram_num_embeddings=(num_emb,),
        engram_max_ngram_size=DEPTH + 1, engram_vocab_size=VOCAB,
        engram_n_heads=HEADS, engram_head_dim=HD, engram_pad_id=2,
        engram_compressed_vocab_size=COMP,
        vocab_size=64, hc_mult=4, n_mtp_layers=1, preserve_mtp=False, max_seq_len=16,
    )
    cfg.validate()
    model = LanguageModel(cfg)
    model.set_token_map(np.arange(COMP, dtype=np.int64))
    mx.random.seed(1234)

    def rnd(t):
        if isinstance(t, dict):
            return {k: rnd(v) for k, v in t.items()}
        if isinstance(t, list):
            return [rnd(v) for v in t]
        if isinstance(t, mx.array) and t.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.1, 0.1, t.shape).astype(t.dtype)
        return t

    model.update(rnd(model.parameters()))
    pref, dec = [3, 7, 5], [2, 9, 4, 11, 8]
    cache = model.make_cache()
    lp = model._forward(mx.array([pref], mx.int32), cache=cache); mx.eval(lp)
    ld = []
    for t in dec:
        g = model._forward(mx.array([[t]], mx.int32), cache=cache); mx.eval(g)
        ld.append(np.array(g, np.float32).reshape(-1))
    ld = np.stack(ld, 0)

    out = ["/* GENERATO da tools/gen_dsv41_engram_decode_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_ENGRAM_DECODE_REF_H\n#define GMB_DSV41_ENGRAM_DECODE_REF_H\n",
           "#define REF_ENGD_SPREFILL 3\n#define REF_ENGD_SDECODE 5\n#define REF_ENGD_VOCAB 64\n"]
    out.append("static const int REF_ENGD_IDS_PREFILL[3] = {%s};\n" % ", ".join(str(x) for x in pref))
    out.append("static const int REF_ENGD_IDS_DECODE[5] = {%s};\n" % ", ".join(str(x) for x in dec))
    out.append(cvec("REF_ENGD_LOGITS_PREFILL", lp))
    out.append(cvec("REF_ENGD_LOGITS_DECODE", ld))
    out.append("#endif /* GMB_DSV41_ENGRAM_DECODE_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_engram_decode_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| prefill", np.array(lp).shape, "| decode", ld.shape)


if __name__ == "__main__":
    main()
