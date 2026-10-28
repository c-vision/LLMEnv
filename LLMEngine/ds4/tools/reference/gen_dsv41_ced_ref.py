#!/usr/bin/env python3
"""gen_dsv41_ced_ref.py — riferimento tiny del forward V4.1 con CED prefill-skip (no download).

`ced_prefill=True`, `_ced_prefill=True`: la meta' decoder (layer >= mid) forwarda solo gli ultimi
`window_size` token, mentre il layer mid (kv_source) proietta il KV globale dagli hidden finali
dell'encoder (full). Config verificata da `ced_layout_supported()`: 4 layer, mid=2,
compress_ratios=(0,2,1,1), kv_source=(1,2), index_source=(1,2,3), window=4, S=8 (> window).

Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_ced_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_ced_ref.h
"""
import os
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel

    cfg = ModelConfig(
        dim=8, n_layers=4, n_heads=2, o_groups=2, head_dim=32, rope_head_dim=4,
        index_head_dim=32, index_n_heads=2, q_lora_rank=8, o_lora_rank=8, moe_inter_dim=8,
        n_routed_experts=4, n_activated_experts=2, n_shared_experts=1, window_size=4,
        compress_ratios=(0, 2, 1, 1), kv_source_layers=(1, 2), index_source_layers=(1, 2, 3),
        candidate_source_layer=-1, index_topk=2, ced_prefill=True,
        engram_layer_ids=(), engram_num_embeddings=(), vocab_size=64, hc_mult=4,
        n_mtp_layers=1, preserve_mtp=False, max_seq_len=32,
    )
    cfg.validate()
    assert cfg.ced_layout_supported(), "CED layout non supportato"
    model = LanguageModel(cfg)
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
    ids = mx.array([[3, 7, 5, 2, 9, 4, 11, 8]], mx.int32)
    logits = model._forward(ids, cache=model.make_cache(), _ced_prefill=True)
    mx.eval(logits)

    out = ["/* GENERATO da tools/gen_dsv41_ced_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_CED_REF_H\n#define GMB_DSV41_CED_REF_H\n",
           "#define REF_CED_L 4\n#define REF_CED_MID 2\n#define REF_CED_S 8\n#define REF_CED_TAIL 4\n"
           "#define REF_CED_DIM 8\n#define REF_CED_HC 4\n#define REF_CED_D 32\n#define REF_CED_QD 8\n"
           "#define REF_CED_VOCAB 64\n"]
    out.append("static const int REF_CED_IDS[8] = {%s};\n" % ", ".join(str(int(v)) for v in ids[0]))
    out.append(cvec("REF_CED_EMBED", model.embed.weight))
    out.append(cvec("REF_CED_FNORM", model.norm.weight))
    out.append(cvec("REF_CED_HEAD", model.head.weight))
    for i in range(4):
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
            out.append(cvec("REF_CED_" + nm + s, v))
        if i in cfg.kv_source_layers:
            out.append(cvec("REF_CED_CMPWKV" + s, A.compressor.wkv.weight))
            out.append(cvec("REF_CED_CMPNORM" + s, A.compressor.norm.weight))
            if cfg.compress_ratios[i] > 1:
                out.append(cvec("REF_CED_CMPWGATE" + s, A.compressor.wgate.weight))
        if i in cfg.index_source_layers:
            out.append(cvec("REF_CED_IDXWQB" + s, A.indexer.wq_b.weight))
            out.append(cvec("REF_CED_IDXWP" + s, A.indexer.weights_proj.weight))
            if i in cfg.kv_source_layers:
                out.append(cvec("REF_CED_IDXWK" + s, A.indexer.wk.weight))
                out.append(cvec("REF_CED_IDXKN" + s, A.indexer.k_norm.weight))
    out.append(cvec("REF_CED_LOGITS", logits))
    out.append("#endif /* GMB_DSV41_CED_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_ced_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| logits", logits.shape)


if __name__ == "__main__":
    main()
