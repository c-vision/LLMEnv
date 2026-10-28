#!/usr/bin/env python3
"""gen_dsv41_dspark_ref.py — riferimento tiny per la matematica DSpark (no download):
get_dspark_topk_idxs, MarkovHead (bias), ConfidenceHead, proposta argmax(logits+bias).
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_dspark_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_dspark_ref.h
"""
import os
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def i32vec(name, arr, per=16):
    vals = [str(int(v)) for v in np.asarray(arr, np.int32).reshape(-1)] or ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const int {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.dspark import MarkovHead, ConfidenceHead

    V, R, D = 16, 4, 8
    blk, win, start, bsz = 3, 4, 5, 2
    mx.random.seed(1234)
    cfg = ModelConfig(dim=D, vocab_size=V, dspark_markov_rank=R)
    markov = MarkovHead(cfg)
    conf = ConfidenceHead(cfg)

    def rnd(t):
        if isinstance(t, dict):
            return {k: rnd(v) for k, v in t.items()}
        if isinstance(t, list):
            return [rnd(v) for v in t]
        if isinstance(t, mx.array) and t.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.2, 0.2, t.shape).astype(t.dtype)
        return t

    markov.update(rnd(markov.parameters()))
    conf.update(rnd(conf.parameters()))

    ids = mx.array([[3]], mx.int32)
    logits, embedding = markov(ids)          # [1,1,V], [1,1,R]
    h = mx.random.uniform(-0.2, 0.2, (1, 1, D))
    confidence = conf(h, embedding)          # [1,1]
    bias = logits[0, 0]                       # [V]
    row = np.concatenate([np.arange(min(win, start + 1)), win + np.arange(blk)]).astype(np.int32)
    topk = np.broadcast_to(row.reshape(1, 1, -1), (bsz, blk, row.size))
    mx.eval(logits, embedding, confidence, bias)
    bias = np.array(bias, np.float32)
    proposal = int(np.argmax(bias))           # proposta con logits nulli + bias
    mx.eval()

    out = ["/* GENERATO da tools/gen_dsv41_dspark_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_DSPARK_REF_H\n#define GMB_DSV41_DSPARK_REF_H\n",
           f"#define REF_DS_VOCAB {V}\n#define REF_DS_RANK {R}\n#define REF_DS_DIM {D}\n"
           f"#define REF_DS_BLK {blk}\n#define REF_DS_WIN {win}\n#define REF_DS_START {start}\n"
           f"#define REF_DS_BSZ {bsz}\n#define REF_DS_TOPK_N {int(topk.shape[-1])}\n"]
    out.append("static const int REF_DS_ID[1] = {3};\n")
    out.append(cvec("REF_DS_EMBED", markov.embed.weight))          # [V,R]
    out.append(cvec("REF_DS_HEAD", markov.head.weight))            # [V,R]
    out.append(cvec("REF_DS_PROJ", conf.proj.weight))              # [1,D+R]
    out.append(cvec("REF_DS_H", h))
    out.append(cvec("REF_DS_MARKOV_EMB", embedding))
    out.append(cvec("REF_DS_BIAS", bias))
    out.append(cvec("REF_DS_CONF", confidence))
    out.append(i32vec("REF_DS_TOPK", np.array(topk, np.int32)))
    out.append(f"static const int REF_DS_PROPOSAL = {proposal};\n")
    out.append("#endif /* GMB_DSV41_DSPARK_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_dspark_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| topk", topk.shape, "| proposal", proposal)


if __name__ == "__main__":
    main()
