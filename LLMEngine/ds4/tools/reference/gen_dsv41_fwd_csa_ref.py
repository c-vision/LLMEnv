#!/usr/bin/env python3
"""gen_dsv41_fwd_csa_ref.py — riferimento tiny del forward V4.1 con CSA2 ratio>0 (no download).

Config: 2 layer, `compress_ratios=(0,2)`, layer1 = kv_source + index_source (compressor +
indexer attivi), candidate_source_layer=-1, index_topk=64 (>= chiavi compresse -> selezione
top-k degenere = tutte, cronologiche). S=4 <= window_size=4 (evita il ramo di troncamento del
prefill). Dumpa pesi (inclusi compressor), ids e logits per verificare il forward MLX CSA2.

Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_fwd_csa_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_fwd_csa_ref.h
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
        dim=8, n_layers=2, n_heads=2, o_groups=2, head_dim=32, rope_head_dim=4,
        index_head_dim=32, index_n_heads=2, q_lora_rank=8, o_lora_rank=8, moe_inter_dim=8,
        n_routed_experts=4, n_activated_experts=2, n_shared_experts=1, window_size=4,
        compress_ratios=(0, 2), kv_source_layers=(1,), index_source_layers=(1,),
        candidate_source_layer=-1, index_topk=64,
        engram_layer_ids=(), engram_num_embeddings=(), vocab_size=64, hc_mult=4,
        n_mtp_layers=1, preserve_mtp=False, max_seq_len=16,
    )
    cfg.validate()
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
    c = model._config
    ids = mx.array([[3, 7, 5, 2]], mx.int32)
    logits = model(ids)
    mx.eval(logits)

    out = ["/* GENERATO da tools/gen_dsv41_fwd_csa_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_FWD_CSA_REF_H\n#define GMB_DSV41_FWD_CSA_REF_H\n",
           "#define REF_CSA_L 2\n#define REF_CSA_S 4\n#define REF_CSA_DIM 8\n#define REF_CSA_HC 4\n"
           "#define REF_CSA_SIZE 24\n#define REF_CSA_H 2\n#define REF_CSA_D 32\n#define REF_CSA_DROPE 4\n"
           "#define REF_CSA_QD 8\n#define REF_CSA_OG 2\n#define REF_CSA_OLORA 8\n#define REF_CSA_WIN 4\n"
           "#define REF_CSA_RATIO 2\n#define REF_CSA_M 2\n#define REF_CSA_E 4\n#define REF_CSA_ACT 2\n"
           "#define REF_CSA_DI 8\n#define REF_CSA_VOCAB 64\n"]
    out.append(cvec("REF_CSA_EMBED", model.embed.weight))
    out.append("static const int REF_CSA_IDS[4] = {%d, %d, %d, %d};\n" % tuple(int(v) for v in ids[0]))
    out.append(cvec("REF_CSA_LOGITS", logits))
    out.append(cvec("REF_CSA_FNORM", model.norm.weight))
    out.append(cvec("REF_CSA_HEAD", model.head.weight))

    comp = {0: False, 1: True}
    for i in range(2):
        L = model.layers[i]
        A = L.attn
        E = L.ffn.experts
        S = L.ffn.shared_experts
        s = str(i)
        for nm, v in [("HCAFN", L.hc_attn_fn), ("HCAS", L.hc_attn_scale), ("HCAB", L.hc_attn_base),
                      ("ANORM", L.attn_norm.weight), ("WQA", A.wq_a.weight), ("QN", A.q_norm.weight),
                      ("WQB", A.wq_b.weight), ("WKV", A.wkv.weight), ("KVN", A.kv_norm.weight),
                      ("WOA", A.wo_a.weight), ("WOB", A.wo_b.weight), ("SINK", A.attn_sink),
                      ("HCFFN", L.hc_ffn_fn), ("HCFFS", L.hc_ffn_scale), ("HCFFB", L.hc_ffn_base),
                      ("FNORM", L.ffn_norm.weight), ("GW", L.ffn.gate.weight), ("GB", L.ffn.gate.bias),
                      ("EW1", E.w1.weight), ("EW3", E.w3.weight), ("EW2", E.w2.weight),
                      ("SW1", S.w1.weight), ("SW3", S.w3.weight), ("SW2", S.w2.weight)]:
            out.append(cvec("REF_CSA_" + nm + s, v))
        if comp[i]:
            out.append(cvec("REF_CSA_CMPWKV" + s, A.compressor.wkv.weight))
            out.append(cvec("REF_CSA_CMPNORM" + s, A.compressor.norm.weight))
            out.append(cvec("REF_CSA_CMPWGATE" + s, A.compressor.wgate.weight))
        if i in cfg.index_source_layers:
            out.append(cvec("REF_CSA_IDXWQB" + s, A.indexer.wq_b.weight))
            out.append(cvec("REF_CSA_IDXWP" + s, A.indexer.weights_proj.weight))
            if i in cfg.kv_source_layers:
                out.append(cvec("REF_CSA_IDXWK" + s, A.indexer.wk.weight))
                out.append(cvec("REF_CSA_IDXKN" + s, A.indexer.k_norm.weight))
    out.append("#endif /* GMB_DSV41_FWD_CSA_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_fwd_csa_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| logits", logits.shape)


if __name__ == "__main__":
    main()
