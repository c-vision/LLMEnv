#!/usr/bin/env python3
"""gen_dsv41_engram_fwd_ref.py — riferimento tiny del forward V4.1 CON Engram attivo (no download).

Config: 2 layer, engram su layer 1 (max_ngram_size=2, n_heads=2, head_dim=8, vocab 20),
token_map identita' sintetica (niente tokenizer). Dumpa pesi Engram + parametrici di hash
(primes/multipliers/offsets) + token_map + ids + logits, per verificare l'integrazione Engram
nel forward MLX.

Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_engram_fwd_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_engram_fwd_ref.h
"""
import os
import math
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def i64vec(name, arr, per=8):
    vals = [str(int(v)) for v in np.asarray(arr, np.int64).reshape(-1)] or ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const long long {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def primes_needed(vocab, depth, heads):
    seen, groups = set(), []
    for _ in range(depth):
        cur, group = vocab - 1, []
        for _ in range(heads):
            cur += 1
            while not is_prime(cur) or cur in seen:
                cur += 1
            seen.add(cur)
            group.append(cur)
        groups.append(group)
    return np.array(groups, np.int64)


def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel

    VOCAB = 20
    DEPTH = 1          # max_ngram_size - 1
    HEADS = 2
    HD = 8
    COMP = 64          # compressed vocab (token_map identita')

    pr = primes_needed(VOCAB, DEPTH, HEADS)          # (depth, heads)
    num_emb = int(pr.sum())

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

    token_map = np.arange(COMP, dtype=np.int64)
    model.set_token_map(token_map)
    h = model._hasher
    primes = np.asarray(h.primes[0], np.int64)          # (depth, heads)
    mults = np.asarray(h.multipliers[0], np.int64)      # (depth+1,)
    offs = np.asarray(h.offsets[0], np.int64)           # (depth, heads)

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
    ids = mx.array([[3, 7, 5, 2, 9, 4]], mx.int32)
    logits = model(ids)
    mx.eval(logits)

    out = ["/* GENERATO da tools/gen_dsv41_engram_fwd_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_ENGRAM_FWD_REF_H\n#define GMB_DSV41_ENGRAM_FWD_REF_H\n",
           f"#define REF_ENGF_S 6\n#define REF_ENGF_DIM 8\n#define REF_ENGF_HC 4\n"
           f"#define REF_ENGF_DEPTH {DEPTH}\n#define REF_ENGF_NHEADS {HEADS}\n#define REF_ENGF_HD {HD}\n"
           f"#define REF_ENGF_ROWS {num_emb}\n#define REF_ENGF_NCOLS {DEPTH * HEADS}\n#define REF_ENGF_VOCAB 64\n"]
    out.append("static const int REF_ENGF_IDS[6] = {%s};\n" % ", ".join(str(int(v)) for v in ids[0]))
    out.append(cvec("REF_ENGF_EMBED_G", model.embed.weight))
    out.append(cvec("REF_ENGF_FNORM", model.norm.weight))
    out.append(cvec("REF_ENGF_HEAD", model.head.weight))
    for i in range(2):
        L, A, F = model.layers[i], model.layers[i].attn, model.layers[i].ffn
        s = str(i)
        for nm, v in [("HCAFN", L.hc_attn_fn), ("HCAS", L.hc_attn_scale), ("HCAB", L.hc_attn_base),
                      ("ANORM", L.attn_norm.weight), ("WQA", A.wq_a.weight), ("QN", A.q_norm.weight),
                      ("WQB", A.wq_b.weight), ("WKV", A.wkv.weight), ("KVN", A.kv_norm.weight),
                      ("WOA", A.wo_a.weight), ("WOB", A.wo_b.weight), ("SINK", A.attn_sink),
                      ("HCFFN", L.hc_ffn_fn), ("HCFFS", L.hc_ffn_scale), ("HCFFB", L.hc_ffn_base),
                      ("FNORM", L.ffn_norm.weight), ("GW", L.ffn.gate.weight), ("GB", L.ffn.gate.bias),
                      ("EW1", F.experts.w1.weight), ("EW3", F.experts.w3.weight), ("EW2", F.experts.w2.weight),
                      ("SW1", F.shared_experts.w1.weight), ("SW3", F.shared_experts.w3.weight),
                      ("SW2", F.shared_experts.w2.weight)]:
            out.append(cvec("REF_ENGF_" + nm + s, v))
    E = model.layers[1].engram
    out.append(i64vec("REF_ENGF_TOKEN_MAP", token_map))
    out.append(i64vec("REF_ENGF_PRIMES", primes))
    out.append(i64vec("REF_ENGF_MULT", mults))
    out.append(i64vec("REF_ENGF_OFFSETS", offs))
    out.append(cvec("REF_ENGF_EMBED", E.embed.weight))
    out.append(cvec("REF_ENGF_WKV", E.wkv.weight))
    out.append(cvec("REF_ENGF_QW", E.q_weight))
    out.append(cvec("REF_ENGF_KW", E.k_weight))
    out.append(cvec("REF_ENGF_LOGITS", logits))
    out.append("#endif /* GMB_DSV41_ENGRAM_FWD_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_engram_fwd_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| rows", num_emb, "| logits", logits.shape)


if __name__ == "__main__":
    main()
