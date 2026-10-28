#!/usr/bin/env python3
"""gen_dsv41_fwd_step1_reference.py — riferimento per lo step 1 del forward V4.1 C.
Step 1 = embed + repeat(hc_mult) + pre. Legge il .npz di
tools/gen_dsv41_forward_reference.py e scrive LLMEnv/LLMEngine/mlx/llm_dsv41_fwd_step1_ref.h.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_fwd_step1_reference.py
"""
import os
import numpy as np

NPZ = "/tmp/dsv41-fwd-ref/dsv41_fwd_ref.npz"

def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def main():
    d = np.load(NPZ)
    emb = d["w:embed.weight"].astype(np.float32)      # (vocab, dim)
    ids = d["input_ids"].astype(np.int64).reshape(-1)  # (L,)
    vocab, dim = emb.shape
    L = len(ids)
    hc = 4  # hc_mult della config tiny
    # h[l,c,d] = emb[ids[l], d] ; pre[l,c] = (c==0)
    h = np.zeros((L, hc, dim), np.float32)
    for l in range(L):
        h[l, :, :] = emb[ids[l]]
    pre = np.zeros((L, hc), np.float32); pre[:, 0] = 1.0

    out = ["/* GENERATO da tools/gen_dsv41_fwd_step1_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_FWD_STEP1_REF_H\n#define GMB_DSV41_FWD_STEP1_REF_H\n",
           "#define REF_S1_VOCAB %d\n#define REF_S1_DIM %d\n#define REF_S1_L %d\n#define REF_S1_HC %d\n"
           % (vocab, dim, L, hc)]
    out.append(cvec("REF_S1_EMBED", emb))
    out.append(("static const int REF_S1_IDS[%d] = {%s};\n"
                % (L, ", ".join(str(int(i)) for i in ids))))
    out.append(cvec("REF_S1_H", h))
    out.append(cvec("REF_S1_PRE", pre))
    out.append("#endif /* GMB_DSV41_FWD_STEP1_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_fwd_step1_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst), "| L=%d dim=%d hc=%d" % (L, dim, hc))

if __name__ == "__main__":
    main()
