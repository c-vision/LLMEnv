#!/usr/bin/env python3
"""gen_dsv41_moe_reference.py — valori di riferimento per llm_dsv41_moe.h (step 4).

Replica VERBATIM di language.py: Gate.__call__, Expert.__call__ (swiglu), MoE combine.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_moe_reference.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_moe_ref.h
"""
import math
import os

MASK = 0x7FFFFFFF

def lcg(seed, n, lo=-1000, hi=1000):
    out, s = [], seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & MASK
        out.append((s % (hi - lo + 1) + lo) / 1000.0)
    return out

def softplus(a):
    return math.log(math.exp(a) + 1.0)

def gate(x, weight, bias, D, n_experts, gate_temp, score_func, n_act, norm_topk_prob, route_scale):
    raw = [sum(x[d] * weight[e * D + d] for d in range(D)) / gate_temp for e in range(n_experts)]
    if score_func == 0:  # softmax
        mxv = max(raw)
        ex = [math.exp(r - mxv) for r in raw]
        s = sum(ex)
        scores = [e / s for e in ex]
    elif score_func == 1:  # sigmoid
        scores = [1.0 / (1.0 + math.exp(-r)) for r in raw]
    else:  # sqrtsoftplus
        scores = [math.sqrt(softplus(r)) for r in raw]
    comb = [scores[e] + bias[e] for e in range(n_experts)]
    order = sorted(range(n_experts), key=lambda e: (-comb[e], e))[:n_act]
    w = [scores[e] for e in order]
    if norm_topk_prob and n_act > 1:
        s = sum(w) + 1e-20
        w = [v / s for v in w]
    w = [v * route_scale for v in w]
    return order, w

def expert_swiglu(gt, up, limit, weight):
    y = []
    for g, u in zip(gt, up):
        if limit > 0:
            g = min(g, limit)
            u = max(-limit, min(limit, u))
        y.append((g / (1.0 + math.exp(-g))) * u * weight)
    return y

def moe_combine(routed, shared, D):
    return [sum(routed[e * D + d] for e in range(len(routed) // D)) + shared[d] for d in range(D)]

def cvec(name, arr, per=10, fmt="%.7g"):
    def flat(a):
        for v in a:
            if isinstance(v, (list, tuple)):
                yield from flat(v)
            else:
                yield v
    vals = [fmt % float(v) for v in flat(arr)]
    if not vals:
        vals = ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def ivec(name, arr, per=16):
    vals = [str(int(v)) for v in arr] or ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const int {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def main():
    out = ["/* GENERATO da tools/gen_dsv41_moe_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_MOE_REF_H\n#define GMB_DSV41_MOE_REF_H\n"]

    D, n_exp, n_act = 3, 6, 2
    x = lcg(10, D)
    weight = lcg(11, n_exp * D)
    bias = lcg(12, n_exp)
    out.append(cvec("REF_GATE_X", x))
    out.append(cvec("REF_GATE_W", weight))
    out.append(cvec("REF_GATE_B", bias))
    out.append("#define REF_GATE_D %d\n#define REF_GATE_E %d\n#define REF_GATE_ACT %d\n" % (D, n_exp, n_act))
    for ci, (sf, nrm, rs) in enumerate([(2, 1, 1.0), (0, 1, 1.0), (1, 0, 1.3)]):
        idx, w = gate(x, weight, bias, D, n_exp, 1.0, sf, n_act, nrm, rs)
        out.append("#define REF_GATE_SF%d %d\n#define REF_GATE_NRM%d %d\n#define REF_GATE_RS%d %g\n"
                   % (ci, sf, ci, nrm, ci, rs))
        out.append(ivec("REF_GATE_IDX%d" % ci, idx))
        out.append(cvec("REF_GATE_WGT%d" % ci, w))

    # Expert swiglu
    dff, limit, ew = 4, 1.5, 0.7
    gt = lcg(20, dff)
    up = lcg(21, dff)
    out.append(cvec("REF_SW_GATE", gt))
    out.append(cvec("REF_SW_UP", up))
    out.append("#define REF_SW_LIMIT %g\n#define REF_SW_WEIGHT %g\n#define REF_SW_D %d\n" % (limit, ew, dff))
    out.append(cvec("REF_SW_OUT", expert_swiglu(gt, up, limit, ew)))

    # MoE combine
    routed = lcg(30, n_act * D)
    shared = lcg(31, D)
    out.append(cvec("REF_MOE_ROUTED", routed))
    out.append(cvec("REF_MOE_SHARED", shared))
    out.append("#define REF_MOE_ACT %d\n#define REF_MOE_D %d\n" % (n_act, D))
    out.append(cvec("REF_MOE_OUT", moe_combine(routed, shared, D)))

    out.append("#endif /* GMB_DSV41_MOE_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_moe_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst))

if __name__ == "__main__":
    main()
