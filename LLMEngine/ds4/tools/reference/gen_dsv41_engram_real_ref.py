#!/usr/bin/env python3
"""gen_dsv41_engram_real_ref.py — riferimento Engram REALE per il loader C, senza pesi.

Costruisce il `token_map` compattato dal tokenizer reale (`build_compressed_token_map`), lo scrive
come int32 little-endian in `engram_token_map.bin` nella cartella modello, esegue `NgramHash` del
layer 1 sui token dati e dumpa un header con token_map (slice), primes/offsets/multipliers del
layer 1 e gli hashes attesi. Verifica: `llm_dsv41_engram_real_test.c`.

Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_engram_real_ref.py
Scrive: <model>/engram_token_map.bin e LLMEnv/LLMEngine/mlx/llm_dsv41_engram_real_ref.h
"""
import os
import json
import numpy as np

MODEL = "/Users/gaetano/ai/Models/DeepSeek-V4.1-Flash"


def i64vec(name, arr, per=8):
    vals = [str(int(v)) for v in np.asarray(arr, np.int64).reshape(-1)]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const long long {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def main():
    from transformers import AutoTokenizer
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.engram import build_compressed_token_map, NgramHash

    with open(os.path.join(MODEL, "inference/config.json")) as f:
        cfg = ModelConfig.from_dict(json.load(f))

    tok = AutoTokenizer.from_pretrained(MODEL)
    lookup, comp = build_compressed_token_map(tok)
    lookup = np.asarray(lookup, np.int64)
    assert comp == cfg.engram_compressed_vocab_size, (comp, cfg.engram_compressed_vocab_size)

    # dump token_map reale (int32 LE)
    bin_path = os.path.join(MODEL, "engram_token_map.bin")
    lookup.astype("<i4").tofile(bin_path)
    print("scritto:", bin_path, "| n", lookup.size)

    hasher = NgramHash(cfg, lookup)
    ids = np.array([[3, 7, 5, 2, 9, 4, 11, 8]], np.int64)
    hashes, history = hasher(ids, None, None)   # [1, n_layers, depth*heads]
    hashes = np.asarray(hashes, np.int64)
    history = np.asarray(history, np.int64)
    depth = cfg.engram_max_ngram_size - 1
    heads = cfg.engram_n_heads
    h0 = hashes[0, :, 0, :].reshape(-1)        # layer 1 (indice 0): [L, ncols]
    primes0 = np.asarray(hasher.primes[0], np.int64).reshape(-1)
    offsets0 = np.asarray(hasher.offsets[0], np.int64).reshape(-1)
    mult0 = np.asarray(hasher.multipliers[0], np.int64)

    out = ["/* GENERATO da tools/gen_dsv41_engram_real_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_ENGRAM_REAL_REF_H\n#define GMB_DSV41_ENGRAM_REAL_REF_H\n",
           f"#define REF_ENR_L 8\n#define REF_ENR_DEPTH {depth}\n#define REF_ENR_NHEADS {heads}\n"
           f"#define REF_ENR_NCOLS {depth * heads}\n#define REF_ENR_MAP_N {lookup.size}\n"
           f"#define REF_ENR_PAD {cfg.engram_pad_id}\n#define REF_ENR_VOCAB {cfg.engram_vocab_size}\n"
           f"#define REF_ENR_COMP {cfg.engram_compressed_vocab_size}\n"]
    out.append("static const int REF_ENR_IDS[8] = {%s};\n" % ", ".join(str(int(v)) for v in ids[0]))
    out.append(i64vec("REF_ENR_PRIMES", primes0))
    out.append(i64vec("REF_ENR_OFFSETS", offsets0))
    out.append(i64vec("REF_ENR_MULT", mult0))
    out.append(i64vec("REF_ENR_HASHES", h0))
    out.append(i64vec("REF_ENR_HIST", history[0]))
    # slice del token_map per gli id usati (verifica mapping)
    out.append(i64vec("REF_ENR_TOKENS", lookup[ids[0]]))
    out.append("#endif /* GMB_DSV41_ENGRAM_REAL_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_engram_real_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| hashes", h0.shape)


if __name__ == "__main__":
    main()
