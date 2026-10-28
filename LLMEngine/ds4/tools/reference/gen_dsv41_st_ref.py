#!/usr/bin/env python3
"""gen_dsv41_st_ref.py — fixture per il loader safetensors V4.1 (no download).

Scrive:
  /tmp/dsv41_st/tiny.safetensors — checkpoint sintetico MONO-FILE con i pesi del modello tiny
    CSA (config di gen_dsv41_fwd_csa_ref.py, seed 1234) in F32 e i nomi canonici del release,
    piu' tensori "probe" quantizzati (fp8+ue8m0, fp4+ue8m0, bf16).
  LLMEnv/LLMEngine/mlx/llm_dsv41_st_ref.h — valori attesi dei probe (dequant) per il test C.

Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_st_ref.py
"""
import os
import sys
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.9g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def u8vec(name, arr, per=24):
    vals = [str(int(v)) for v in np.asarray(arr, np.uint8).reshape(-1)] or ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const unsigned char {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


# E2M1 signed table (dequant.py _E2M1).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def main():
    no_probes = "--no-probes" in sys.argv
    outdir_custom = None
    if "--out" in sys.argv:
        outdir_custom = sys.argv[sys.argv.index("--out") + 1]
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel

    outdir = outdir_custom or "/tmp/dsv41_st"
    os.makedirs(outdir, exist_ok=True)

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

    arrs = {}
    add = lambda name, a: arrs.__setitem__(name, a.astype(mx.float32))
    add("embed.weight", model.embed.weight)
    add("norm.weight", model.norm.weight)
    add("head.weight", model.head.weight)

    for i in range(2):
        L, A, F = model.layers[i], model.layers[i].attn, model.layers[i].ffn
        p = f"layers.{i}."
        add(p + "hc_attn_fn", L.hc_attn_fn); add(p + "hc_attn_scale", L.hc_attn_scale)
        add(p + "hc_attn_base", L.hc_attn_base)
        add(p + "hc_ffn_fn", L.hc_ffn_fn); add(p + "hc_ffn_scale", L.hc_ffn_scale)
        add(p + "hc_ffn_base", L.hc_ffn_base)
        add(p + "attn_norm.weight", L.attn_norm.weight); add(p + "ffn_norm.weight", L.ffn_norm.weight)
        add(p + "attn.wq_a.weight", A.wq_a.weight); add(p + "attn.q_norm.weight", A.q_norm.weight)
        add(p + "attn.wq_b.weight", A.wq_b.weight); add(p + "attn.wkv.weight", A.wkv.weight)
        add(p + "attn.kv_norm.weight", A.kv_norm.weight); add(p + "attn.wo_a.weight", A.wo_a.weight)
        add(p + "attn.wo_b.weight", A.wo_b.weight); add(p + "attn.attn_sink", A.attn_sink)
        add(p + "ffn.gate.weight", F.gate.weight); add(p + "ffn.gate.bias", F.gate.bias)
        add(p + "ffn.gate.bias_vl", mx.zeros_like(F.gate.bias))
        add(p + "ffn.shared_experts.w1.weight", F.shared_experts.w1.weight)
        add(p + "ffn.shared_experts.w2.weight", F.shared_experts.w2.weight)
        add(p + "ffn.shared_experts.w3.weight", F.shared_experts.w3.weight)
        for e in range(cfg.n_routed_experts):
            add(p + f"ffn.experts.{e}.w1.weight", F.experts.w1.weight[e])
            add(p + f"ffn.experts.{e}.w2.weight", F.experts.w2.weight[e])
            add(p + f"ffn.experts.{e}.w3.weight", F.experts.w3.weight[e])
        if i == 1:
            add(p + "attn.compressor.wkv.weight", A.compressor.wkv.weight)
            add(p + "attn.compressor.norm.weight", A.compressor.norm.weight)
            add(p + "attn.compressor.wgate.weight", A.compressor.wgate.weight)
            add(p + "attn.indexer.wq_b.weight", A.indexer.wq_b.weight)
            add(p + "attn.indexer.weights_proj.weight", A.indexer.weights_proj.weight)
            add(p + "attn.indexer.wk.weight", A.indexer.wk.weight)
            add(p + "attn.indexer.k_norm.weight", A.indexer.k_norm.weight)

    # ── probe quantizzati (saltati con --no-probes) ───────────────────────
    fp8_expected = fp4_expected = bf16_expected = None
    if not no_probes:
      rng = np.random.default_rng(7)
      # fp8 e4m3 [64,64] + ue8m0 [2,2]
      fw = rng.integers(0, 256, (64, 64), dtype=np.uint8)
      fw[(fw >> 3 & 15) == 15] &= 0xF8  # azzera i pattern NaN (exp=15, mant=7 -> mant=0)
      fsc = rng.integers(120, 136, (2, 2), dtype=np.uint8)
      fw_m = mx.array(fw, mx.uint8)
      fsc_m = mx.array(fsc, mx.uint8)
      fp8_expected = mx.from_fp8(fw_m, mx.float32) * mx.array(
          np.ldexp(1.0, fsc.astype(np.int32) - 127).repeat(32, 0).repeat(32, 1), mx.float32
      )
      # fp4 e2m1 packed [64,32] + ue8m0 [64,2]
      pw = rng.integers(0, 256, (64, 32), dtype=np.uint8)
      psc = rng.integers(120, 136, (64, 2), dtype=np.uint8)
      lut = np.array(_E2M1, np.float32)
      lo, hi = pw & 15, (pw >> 4) & 15
      vals = np.stack([lut[lo], lut[hi]], -1).reshape(64, 64)
      fp4_expected = vals * np.ldexp(1.0, psc.astype(np.int32) - 127).repeat(32, 1)
      # bf16 [8,8]
      bw = mx.random.uniform(-1, 1, (8, 8)).astype(mx.bfloat16)
      bf16_expected = bw.astype(mx.float32)

      arrs["probe.fp8.weight"] = fw_m
      arrs["probe.fp8.scale"] = fsc_m
      arrs["probe.fp4.weight"] = mx.array(pw, mx.uint8)
      arrs["probe.fp4.scale"] = mx.array(psc, mx.uint8)
      arrs["probe.bf16.weight"] = bw

    mx.eval(*arrs.values())
    st_path = os.path.join(outdir, "tiny.safetensors")
    mx.save_safetensors(st_path, arrs)
    print("scritto:", st_path, "|", len(arrs), "tensori")

    if not no_probes:
      # Secondo riferimento: S=8 ... (serve solo al .h dei probe)
      cfg2 = ModelConfig(
          dim=8, n_layers=2, n_heads=2, o_groups=2, head_dim=32, rope_head_dim=4,
          index_head_dim=32, index_n_heads=2, q_lora_rank=8, o_lora_rank=8, moe_inter_dim=8,
          n_routed_experts=4, n_activated_experts=2, n_shared_experts=1, window_size=4,
          compress_ratios=(0, 2), kv_source_layers=(1,), index_source_layers=(1,),
          candidate_source_layer=-1, index_topk=2,
          engram_layer_ids=(), engram_num_embeddings=(), vocab_size=64, hc_mult=4,
          n_mtp_layers=1, preserve_mtp=False, max_seq_len=16,
      )
      cfg2.validate()
      model2 = LanguageModel(cfg2)
      mx.random.seed(1234)
      model2.update(rnd(model2.parameters()))
      ids8 = mx.array([[3, 7, 5, 2, 9, 4, 11, 8]], mx.int32)
      logits8 = model2(ids8)
      mx.eval(logits8)

    # variante sharded (index.json + 2 shard) per esercitare la risoluzione `weight_map`
    import json
    sdir = os.path.join(outdir, "sharded")
    os.makedirs(sdir, exist_ok=True)
    names = list(arrs.keys())
    half = len(names) // 2
    wm = {}
    for si, chunk in enumerate((names[:half], names[half:])):
        sub = {k: arrs[k] for k in chunk}
        fn = f"shard{si}.safetensors"
        mx.save_safetensors(os.path.join(sdir, fn), sub)
        for k in chunk:
            wm[k] = fn
    with open(os.path.join(sdir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"format": "pt"}, "weight_map": wm}, f)
    # config.json in formato HF alias, per esercitare `llm_dsv41_load_model_dir`
    with open(os.path.join(sdir, "config.json"), "w") as f:
        json.dump({"model_type": "deepseek_v41", "text_config": {
            "hidden_size": 8, "moe_intermediate_size": 8, "num_hidden_layers": 2,
            "num_attention_heads": 2, "head_dim": 32, "qk_rope_head_dim": 4,
            "q_lora_rank": 8, "o_lora_rank": 8, "o_groups": 2, "vocab_size": 64,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus", "routed_scaling_factor": 1.0,
            "rms_norm_eps": 1e-20, "sliding_window": 4, "hc_mult": 4,
            "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "compress_ratios": [0, 2],
            "kv_source_layer_ids": [1], "index_source_layer_ids": [1],
            "index_n_heads": 2, "index_head_dim": 32, "index_topk": 64,
            "engram_layer_ids": [], "engram_num_embeddings": [],
            "rope_theta": 10000, "compress_rope_theta": 40000}}, f)
    print("scritto:", sdir, "| 2 shard + config.json")

    if not no_probes:
      out = ["/* GENERATO da tools/gen_dsv41_st_ref.py — NON modificare a mano. */\n",
             "#ifndef GMB_DSV41_ST_REF_H\n#define GMB_DSV41_ST_REF_H\n",
             "#define REF_ST_FP8_OUT 64\n#define REF_ST_FP8_IN 64\n",
             "#define REF_ST_FP4_OUT 64\n#define REF_ST_FP4_IN 64\n",
             "#define REF_ST_BF16_N 64\n",
             "#define REF_CSA8_S 8\n#define REF_CSA8_VOCAB 64\n"]
      out.append(cvec("REF_ST_FP8", np.array(fp8_expected, np.float32)))
      out.append(cvec("REF_ST_FP4", np.array(fp4_expected, np.float32)))
      out.append(cvec("REF_ST_BF16", np.array(bf16_expected, np.float32)))
      out.append("static const int REF_CSA8_IDS[8] = {%s};\n" %
                 ", ".join(str(int(v)) for v in ids8[0]))
      out.append(cvec("REF_CSA8_LOGITS", logits8))
      out.append("#endif /* GMB_DSV41_ST_REF_H */\n")
      dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_st_ref.h")
      open(dst, "w").write("".join(out))
      print("scritto:", os.path.realpath(dst))


if __name__ == "__main__":
    main()
