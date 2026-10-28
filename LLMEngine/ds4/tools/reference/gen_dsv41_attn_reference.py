#!/usr/bin/env python3
"""gen_dsv41_attn_reference.py — valori di riferimento per llm_dsv41_attn.h (step 3).

Replica VERBATIM la semantica di:
  - language.py Attention.__call__ : SWA `local`/`valid`/`idx`
  - language.py Compressor.__call__ : concat cache[:rem] + cutoff + pooling
  - kernels.py packed_index_topk    : argsort(-scores)[:k] + sort cronologico + mask -inf
  - kernels.py packed_sparse_attention : softmax su [score, sink] -> weighted sum
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_attn_reference.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_attn_ref.h
"""
import math
import os

MASK = 0x7FFFFFFF
NEG = float("-inf")

def lcg(seed, n, lo=-1000, hi=1000):
    out, s = [], seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & MASK
        out.append((s % (hi - lo + 1) + lo) / 1000.0)
    return out

def swa(start, length, window, old_len):
    W = min(length, window) if start == 0 else window
    lo = max(0, start - old_len)
    rows = []
    for i in range(length):
        pos = start + i
        row = []
        if start == 0:
            base = max(pos - window + 1, 0)
            for j in range(W):
                local = base + j
                row.append(local if (0 <= local <= pos) else -1)
        else:
            for j in range(W):
                local = pos - window + 1 + j
                row.append((local - (start - old_len)) if (lo <= local <= pos) else -1)
        rows.append(row)
    return W, rows

def compressor_step(kv_in, gate_in, L, H, r, start, cache_kv, cache_gate, cache_len):
    rem = start % r
    keep = min(rem, cache_len) if (rem and cache_kv) else 0
    kv = (cache_kv[:keep] if keep else []) + kv_in
    gate = (cache_gate[:keep] if keep else []) + gate_in
    total = keep + L
    cutoff = (total // r) * r
    cnew = total - cutoff
    new_cv = kv[cutoff:]
    new_cg = gate[cutoff:]
    m = cutoff // r
    pooled = []
    for i in range(m):
        seg_kv = kv[i * r:(i + 1) * r]
        seg_g = gate[i * r:(i + 1) * r]
        row = []
        for d in range(H):
            mxv = max(seg_g[k][d] for k in range(r))
            w = [math.exp(seg_g[k][d] - mxv) for k in range(r)]
            s = sum(w)
            row.append(sum(seg_kv[k][d] * (w[k] / s) for k in range(r)))
        pooled.append(row)
    return pooled, new_cv, new_cg, cnew

def index_topk(scores, k):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i] if scores[i] != NEG else float("inf"), i))
    chosen = [i for i in order[:k] if scores[i] != NEG]
    chosen.sort()
    return chosen

def sdpa_sink(q, keys, indices, heads, dim, sink, scale):
    out = [[0.0] * dim for _ in range(heads)]
    for h in range(heads):
        sc = []
        mxv = sink[h]
        for m in range(len(indices)):
            if indices[m] < 0:
                sc.append(NEG); continue
            dot = sum(q[h][e] * keys[m][h][e] for e in range(dim)) * scale
            sc.append(dot)
            mxv = max(mxv, dot)
        w = [0.0 if indices[m] < 0 else math.exp(sc[m] - mxv) for m in range(len(indices))]
        ssum = sum(w) + math.exp(sink[h] - mxv)
        for e in range(dim):
            out[h][e] = sum((w[m] / ssum) * keys[m][h][e] for m in range(len(indices)) if indices[m] >= 0)
    return out

def cvec(name, arr, per=12, fmt="%.7g"):
    def flat(a):
        for v in a:
            if isinstance(v, (list, tuple)):
                yield from flat(v)
            else:
                yield v
    vals = [("NEG" if v == NEG else (fmt % float(v))) for v in flat(arr)]
    if not vals:
        vals = ["0.0"]  # C non ammette array di dimensione 0
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def ivec(name, arr, per=16):
    vals = [str(int(v)) for v in arr]
    if not vals:
        vals = ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const int {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def main():
    out = ["/* GENERATO da tools/gen_dsv41_attn_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_ATTN_REF_H\n#define GMB_DSV41_ATTN_REF_H\n",
           "#define NEG (-INFINITY)\n"]

    # SWA cases
    cases = [(0, 5, 3, 0), (4, 2, 3, 2), (10, 3, 4, 4)]
    out.append("#define REF_SWA_NCASE %d\n" % len(cases))
    for ci, (st, ln, win, ol) in enumerate(cases):
        W, rows = swa(st, ln, win, ol)
        out.append("#define REF_SWA_W%d %d\n" % (ci, W))
        out.append(ivec("REF_SWA_IDX%d" % ci, [x for r in rows for x in r]))

    # compressor: (L,H,r,start,cache_len)
    L, H, r = 4, 3, 2
    for ci, start in enumerate([0, 1]):
        cache_len = start % r
        kv_in = [lcg(100 + ci, H) for _ in range(L)]
        gate_in = [lcg(200 + ci, H) for _ in range(L)]
        cache_kv = [lcg(300 + ci, H) for _ in range(cache_len)]
        cache_gate = [lcg(400 + ci, H) for _ in range(cache_len)]
        pooled, ncv, ncg, cnew = compressor_step(kv_in, gate_in, L, H, r, start, cache_kv, cache_gate, cache_len)
        out.append(cvec("REF_CP_KV_IN%d" % ci, kv_in))
        out.append(cvec("REF_CP_GATE_IN%d" % ci, gate_in))
        out.append("#define REF_CP_CACHE_LEN%d %d\n" % (ci, cache_len))
        out.append("#define REF_CP_CNEW%d %d\n" % (ci, cnew))
        out.append("#define REF_CP_M%d %d\n" % (ci, len(pooled)))
        out.append(cvec("REF_CP_CACHE_KV%d" % ci, cache_kv))
        out.append(cvec("REF_CP_CACHE_GATE%d" % ci, cache_gate))
        out.append(cvec("REF_CP_POOLED%d" % ci, pooled))
        out.append(cvec("REF_CP_NEW_KV%d" % ci, ncv))
        out.append(cvec("REF_CP_NEW_GATE%d" % ci, ncg))

    # index topk
    scores = [0.5, NEG, 0.9, 0.1, NEG, 0.7]
    out.append(cvec("REF_IT_SCORES", scores))
    out.append("#define REF_IT_N %d\n#define REF_IT_K %d\n" % (len(scores), 3))
    out.append(ivec("REF_IT_IDX", index_topk(scores, 3)))

    # candidate block ids
    def candidate_block_ids(scores, lengths, count, block):
        W = len(scores)
        padded = list(scores) + [NEG] * ((block - W % block) % block)
        nblk = len(padded) // block
        grouped = [max(padded[b * block:(b + 1) * block]) for b in range(nblk)]
        if lengths > 0:
            pos = (lengths - 1) // block
            if 0 <= pos < nblk:
                grouped[pos] = float("inf")
        nn = min(count, nblk)
        order = sorted(range(nblk), key=lambda b: (-grouped[b], b))[:nn]
        order.sort()
        return [b if grouped[b] > NEG else -1 for b in order]

    sc = lcg(700, 10)
    LEN, CNT, BLK = 7, 2, 4
    out.append(cvec("REF_CB_SCORES", sc))
    out.append("#define REF_CB_W %d\n#define REF_CB_LEN %d\n#define REF_CB_CNT %d\n#define REF_CB_BLK %d\n"
               % (len(sc), LEN, CNT, BLK))
    out.append(ivec("REF_CB_OUT", candidate_block_ids(sc, LEN, CNT, BLK)))

    # sdpa sink
    heads, dim, M = 2, 3, 4
    q = [[lcg(500, dim) for _ in range(heads)]]
    q = q[0]
    keys = [[lcg(600 + m, dim) for _ in range(heads)] for m in range(M)]
    indices = [0, 2, -1, 3]
    sink = [0.2, -0.1]
    scale = dim ** -0.5
    out_sdpa = sdpa_sink(q, keys, indices, heads, dim, sink, scale)
    out.append(cvec("REF_SD_Q", q))
    out.append(cvec("REF_SD_KEYS", keys))
    out.append(ivec("REF_SD_IDX", indices))
    out.append(cvec("REF_SD_SINK", sink))
    out.append(cvec("REF_SD_SCALE", [scale]))
    out.append(cvec("REF_SD_OUT", out_sdpa))

    out.append("#endif /* GMB_DSV41_ATTN_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_attn_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst))

if __name__ == "__main__":
    main()
