#!/usr/bin/env python3
"""Audit di provenienza del dtype a 16 bit dei checkpoint MLX (Knowledge §31, task #32).

Replica ESATTAMENTE il campionatore dell'engine `llm_mlx_scales_dtype`
(LLMEnv/LLMEngine/mlx/gmb_host_ai_mlx.c: le 4 basi dense/attn + gate `affine`
+ allow-list `want_bf16`), e INOLTRE — dove quel campionatore fallisce (nomi GDN
`language_model.*linear_attn.*` dei qwen3_5) — legge il vero dtype delle
`.scales` di layer-0 per nome reale. Serve a rispondere, prima di ogni promozione
bf16/f16, a: "questo checkpoint ha scale F16? il default gli dà il flavor giusto?".

Uso:
    python3 tools/llm_mlx_dtype_provenance.py <modelDir> [<modelDir> ...]
    python3 tools/llm_mlx_dtype_provenance.py /Users/gaetano/ai/Models/*   # tutti

Nessun caricamento di modello: legge solo config.json + header safetensors.
"""
import json, os, struct, glob, sys

# Le 4 basi campionate da llm_mlx_scales_dtype (gmb_host_ai_mlx.c:11462).
BASES = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.attention.wq",
    "model.layers.0.feed_forward.w1",
]

def read_index(dirpath):
    idx = os.path.join(dirpath, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as f:
            return json.load(f).get("weight_map", {})
    return {}

def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr

def sampler_engine(dirpath, mp, cache):
    """Cosa vede il campionatore dell'engine: la PRIMA delle 4 basi che risolve."""
    for base in BASES:
        key = base + ".scales"
        dt = dtype_of(key, mp, dirpath, cache)
        if dt:
            return base, dt
    return None, None

def dtype_of(key, mp, dirpath, cache):
    shard = mp.get(key)
    if shard:
        if shard not in cache:
            p = os.path.join(dirpath, shard)
            cache[shard] = st_header(p) if os.path.isfile(p) else {}
        return cache[shard].get(key, {}).get("dtype")
    for s in glob.glob(os.path.join(dirpath, "*.safetensors")):
        b = os.path.basename(s)
        if b not in cache:
            cache[b] = st_header(s)
        if key in cache[b]:
            return cache[b][key]["dtype"]
    return None

def all_layer0_scales(dirpath, mp, cache):
    """Histogram dei dtype delle `.scales` di layer-0 per nome REALE (bypassa le 4 basi)."""
    keys = sorted(k for k in mp if ".0." in k and k.endswith(".scales"))
    hist = {}
    for k in keys:
        dt = dtype_of(k, mp, dirpath, cache)
        hist[dt] = hist.get(dt, 0) + 1
    return hist

def quant_cfg(dirpath, has_scales):
    cfgp = os.path.join(dirpath, "config.json")
    if not os.path.isfile(cfgp):
        return {"model_type": "?", "bits": "?", "group_size": "?", "mode": "?", "affine": has_scales}
    with open(cfgp) as f:
        c = json.load(f)
    q = c.get("quantization_config", {}) or {}
    gs = q.get("group_size", "?")
    mode = q.get("mode", "?")
    # mlx puo non scrivere nulla nel config (Mixtral: quantization_config={}): deduci affine dal
    # group_size (==32 -> mxfp4) o dalla presenza di triple .scales nel checkpoint.
    if mode in ("affine", "mxfp4"):
        affine = (mode == "affine")
    elif gs == 32:
        affine = False
    else:
        affine = (gs != "?") or has_scales
    return {
        "model_type": c.get("model_type", "?"),
        "bits": q.get("bits", "?"),
        "group_size": gs,
        "mode": mode,
        "affine": affine,
    }

def flavor_by_rule(mt, affine, bits, per_tensor=False):
    """Replica il branch want_bf16 (11565-11615) + la porta auto-pair (11508-11513)."""
    if mt == "gpt_oss":
        bits_on = bool(per_tensor)
    elif mt in ("qwen3_moe", "qwen3_5_moe", "mixtral", "glm4_moe"):
        bits_on = affine
    elif mt == "qwen3_5":
        bits_on = (affine and bits in (4, 6, 8))
    else:
        bits_on = False
    return "bf16(default)" if bits_on else "fp32"

def main(paths):
    for d in paths:
        d = d.rstrip("/")
        if not os.path.isdir(d):
            continue
        mp = read_index(d)
        cache = {}
        base, dt = sampler_engine(d, mp, cache)
        hist = all_layer0_scales(d, mp, cache)
        real = dict((k, v) for k, v in hist.items() if k)
        has_scales = bool(real)
        cfg = quant_cfg(d, has_scales)
        sampler = dt if dt else "MANCA"
        default = flavor_by_rule(cfg["model_type"], cfg["affine"], cfg["bits"])
        final = default
        # auto-pair a f16 solo se affine E scale F16 (via il campionatore dell'engine)
        if cfg["affine"] and sampler == "F16":
            final = "f16 (auto-pair)"
        # caso pericoloso: affine, scale realmente F16, ma il campionatore a 4 basi è CIECO
        if cfg["affine"] and sampler == "MANCA" and real and set(real) == {"F16"}:
            final = "f16 NASCOSTO (sampler cieco!)"
        name = os.path.basename(d)
        print(f"{name:52s} type={cfg['model_type']:11s} bits={str(cfg['bits']):3s} "
              f"gs={str(cfg['group_size']):3s} affine={str(cfg['affine']):5s} "
              f"| sampler={sampler:6s} layer0_scales={real if real else '-'} "
              f"| default={default} FINAL={final}")

if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob("/Users/gaetano/ai/Models/*/"))
    main(args)
