#!/usr/bin/env python3
"""gen_dsv41_vision_ref.py — riferimento tiny del ViT + Aligner DeepSeek V4.1 (no download).
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_vision_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_vision_ref.h
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
    from omlx.patches.deepseek_v41.vision import ViT, Aligner

    # tiny vision config
    cfg = ModelConfig(
        dim=16, vision_n_layers=2, vision_dim=16, vision_n_heads=2, vision_inter_dim=32,
        vision_patch_size=2, vision_downsample_ratio=2, vision_rope_theta=10000.0,
    )
    mx.random.seed(1234)
    vit, aligner = ViT(cfg), Aligner(cfg)

    def rnd(t):
        if isinstance(t, dict):
            return {k: rnd(v) for k, v in t.items()}
        if isinstance(t, list):
            return [rnd(v) for v in t]
        if isinstance(t, mx.array) and t.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.2, 0.2, t.shape).astype(t.dtype)
        return t

    vit.update(rnd(vit.parameters()))
    aligner.update(rnd(aligner.parameters()))

    n_h, n_w = 4, 4
    patches = mx.random.uniform(-0.2, 0.2, (n_h * n_w, 3 * cfg.vision_patch_size**2))
    vit_out = vit(patches, n_h, n_w)
    aligner_out = aligner(vit_out, n_h, n_w)
    mx.eval(vit_out, aligner_out)

    out = ["/* GENERATO da tools/gen_dsv41_vision_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_VISION_REF_H\n#define GMB_DSV41_VISION_REF_H\n",
           f"#define REF_VIS_DIM {cfg.vision_dim}\n#define REF_VIS_HEADS {cfg.vision_n_heads}\n"
           f"#define REF_VIS_INTER {cfg.vision_inter_dim}\n#define REF_VIS_PATCH {cfg.vision_patch_size}\n"
           f"#define REF_VIS_LAYERS {cfg.vision_n_layers}\n#define REF_VIS_RATIO {cfg.vision_downsample_ratio}\n"
           f"#define REF_VIS_HD {cfg.vision_dim // cfg.vision_n_heads}\n"
           f"#define REF_VIS_NH {n_h}\n#define REF_VIS_NW {n_w}\n#define REF_VIS_N {n_h * n_w}\n"
           f"#define REF_VIS_OUT {cfg.vision_dim}\n#define REF_VIS_ALIGN_OUT {cfg.dim}\n"]
    out.append(cvec("REF_VIS_PATCHES", np.array(patches, np.float32)))
    out.append(cvec("REF_VIS_VIT_OUT", np.array(vit_out, np.float32)))
    out.append(cvec("REF_VIS_ALIGNER_OUT", np.array(aligner_out, np.float32)))
    # pesi
    P = vit.patch_embed
    out.append(cvec("REF_VIS_PATCH_W", P.proj.weight))
    out.append(cvec("REF_VIS_PATCH_B", P.proj.bias))
    for i, B in enumerate(vit.blocks):
        s = str(i)
        for nm, v in [("N1", B.norm1.weight), ("WQKV", B.attn.wqkv.weight), ("WQKVB", B.attn.wqkv.bias),
                      ("WO", B.attn.wo.weight), ("WOB", B.attn.wo.bias), ("N2", B.norm2.weight),
                      ("W1", B.ffn.w1.weight), ("W2", B.ffn.w2.weight)]:
            out.append(cvec("REF_VIS_" + nm + s, v))
    out.append(cvec("REF_VIS_NORM", vit.norm.weight))
    out.append(cvec("REF_VIS_AW1", aligner.w1.weight))
    out.append(cvec("REF_VIS_AW1B", aligner.w1.bias))
    out.append(cvec("REF_VIS_AW2", aligner.w2.weight))
    out.append(cvec("REF_VIS_AW2B", aligner.w2.bias))
    out.append("#endif /* GMB_DSV41_VISION_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_vision_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| vit", vit_out.shape, "| aligner", aligner_out.shape)


if __name__ == "__main__":
    main()
