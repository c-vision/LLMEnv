#!/usr/bin/env python3
"""gen_dsv41_engram_reference.py — valori di riferimento per llm_dsv41_engram.h (step 5).

Replica VERBATIM di engram.py: NgramHash (primi/multipliers/offsets + hashing) ed Engram.__call__
(gate + iniezione). Le tabelle usano numpy.default_rng(10007*layer_id) come l'originale.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_engram_reference.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_engram_ref.h
"""
import math
import os
import numpy as np

MASK = 0x7FFFFFFF
I64MAX = np.iinfo(np.int64).max

def lcg(seed, n, lo=-1000, hi=1000):
    out, s = [], seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & MASK
        out.append((s % (hi - lo + 1) + lo) / 1000.0)
    return out

def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))

def build_tables(layer_ids, max_ngram, n_heads, vocab_size):
    depth = max_ngram - 1
    primes, multipliers, seen = [], [], set()
    for layer_id in layer_ids:
        groups = []
        for _ in range(depth):
            current, group = vocab_size - 1, []
            for _ in range(n_heads):
                current += 1
                while not is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                group.append(current)
            groups.append(group)
        primes.append(groups)
        rng = np.random.default_rng(10007 * layer_id)
        bound = max(1, (I64MAX // vocab_size) // 2)
        multipliers.append(rng.integers(0, bound, max_ngram, dtype=np.int64) * 2 + 1)
    primes = np.array(primes, dtype=np.int64)          # (layers, depth, n_heads)
    multipliers = np.array(multipliers, dtype=np.int64)  # (layers, depth+1)
    flat = primes.reshape(len(primes), -1)
    offsets = np.cumsum(np.concatenate([np.zeros((len(primes), 1), np.int64), flat[:, :-1]], -1), -1)
    return primes[0], multipliers[0], offsets[0], flat[0]

def ngram_hash(tokens, history, depth, pad_id, multipliers, primes, offsets):
    n_heads = primes.shape[-1]
    joined = np.concatenate([history, tokens]).astype(np.int64)
    L = tokens.shape[0]
    positions = np.arange(L) + depth
    blocked = np.zeros(L, dtype=bool)
    lookback = []
    for shift in range(depth + 1):
        source = joined[positions - shift]
        blocked |= (source == -1)
        lookback.append(np.where(blocked, pad_id, source))
    product = np.stack(lookback, -1)[:, None, :] * multipliers   # (L,1,depth+1)
    rolling, hashes = product[..., 0], []
    for shift in range(1, depth + 1):
        rolling = rolling ^ product[..., shift]
        hashes.append(rolling[..., None] % primes[shift - 1])
    hashes = np.concatenate(hashes, -1) + offsets               # (L, depth*n_heads)
    return hashes.astype(np.int64), joined[-depth:].astype(np.int64)

def engram_gate(h, key, value, qw, kw, norm_eps):
    x = h.astype(np.float64)
    sqx = np.mean(x * x, -1)
    sqk = np.mean(key.astype(np.float64) ** 2, -1)
    inv = 1.0 / np.sqrt(sqx + norm_eps) * (1.0 / np.sqrt(sqk + norm_eps))
    dot = np.sum(x * qw * kw * key, -1) * inv * (h.shape[-1] ** -0.5)
    gate = np.where(dot == 0, 1.0 / (1.0 + math.exp(-0.001)),
                    1.0 / (1.0 + np.exp(-np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), 1e-6)))))
    return x + gate[..., None] * value[None, :]

def ivec64(name, arr, per=6):
    vals = [str(int(v)) + "LL" for v in np.asarray(arr).reshape(-1)] or ["0LL"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const int64_t {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def main():
    layer_ids = (5,)
    max_ngram, n_heads, vocab_size, pad_token = 3, 2, 1000, 2
    depth = max_ngram - 1
    primes, multipliers, offsets, flat = build_tables(layer_ids, max_ngram, n_heads, vocab_size)
    num_embeddings = int(flat.sum())

    # token_map: identita' su V=40 token (compressed vocab)
    V = 40
    token_map = np.arange(V, dtype=np.int64)
    pad_id = int(token_map[pad_token])

    ids = np.array([3, 7, 7, 11, 3], dtype=np.int64)
    tokens = token_map[ids]
    history = np.array([-1, 9], dtype=np.int64)
    H, HIST = ngram_hash(tokens, history, depth, pad_id, multipliers, primes, offsets)

    # caso con image_mask: -1 su alcune posizioni (blocca il lookback)
    mask = np.array([False, True, False, False, True])
    tokens_masked = np.where(mask, -1, tokens)
    H2, HIST2 = ngram_hash(tokens_masked, history, depth, pad_id, multipliers, primes, offsets)

    # gate
    hc, dim = 2, 3
    h = np.array(lcg(50, hc * dim), np.float32).reshape(hc, dim)
    key = np.array(lcg(51, hc * dim), np.float32).reshape(hc, dim)
    value = np.array(lcg(52, dim), np.float32)
    qw = np.array(lcg(53, hc * dim, 0, 2000), np.float32).reshape(hc, dim)  # 0..2
    kw = np.array(lcg(54, hc * dim, 0, 2000), np.float32).reshape(hc, dim)
    gate_out = engram_gate(h, key, value, qw, kw, 1e-20)

    out = ["/* GENERATO da tools/gen_dsv41_engram_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_ENGRAM_REF_H\n#define GMB_DSV41_ENGRAM_REF_H\n",
           "#include <stdint.h>\n"]
    out.append("#define REF_EN_DEPTH %d\n#define REF_EN_NHEADS %d\n#define REF_EN_PAD %d\n" % (depth, n_heads, pad_id))
    out.append("#define REF_EN_NUMEMB %d\n" % num_embeddings)
    out.append(ivec64("REF_EN_TOKEN_MAP", token_map))
    out.append("#define REF_EN_V %d\n#define REF_EN_L %d\n" % (V, len(ids)))
    out.append(ivec64("REF_EN_IDS", ids))
    out.append(ivec64("REF_EN_HIST", history))
    out.append(ivec64("REF_EN_MULT", multipliers))
    out.append(ivec64("REF_EN_PRIMES", primes))
    out.append(ivec64("REF_EN_OFFSETS", offsets))
    out.append(ivec64("REF_EN_HASH", H))
    out.append(ivec64("REF_EN_HISTOUT", HIST))
    out.append(ivec64("REF_EN_MASK", mask.astype(np.int64)))
    out.append(ivec64("REF_EN_HASH2", H2))
    out.append(ivec64("REF_EN_HISTOUT2", HIST2))
    out.append("#define REF_EN_HC %d\n#define REF_EN_DIM %d\n" % (hc, dim))
    out.append(cvec("REF_EN_H", h))
    out.append(cvec("REF_EN_KEY", key))
    out.append(cvec("REF_EN_VALUE", value))
    out.append(cvec("REF_EN_QW", qw))
    out.append(cvec("REF_EN_KW", kw))
    out.append(cvec("REF_EN_GATE_OUT", gate_out))
    out.append("#endif /* GMB_DSV41_ENGRAM_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_engram_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst))

if __name__ == "__main__":
    main()
