#!/usr/bin/env python3
"""gen_tiny_ckpt.py — genera un CHECKPOINT "tiny" sintetico per un'arch di mlx_lm.

PERCHE': implementare/validare una arch non deve dipendere dal download di modelli reali
(lenti, GB). Con un tiny (dim minime, pesi casuali deterministici) si valida la STESSA via di
codice dell'arch in pochi secondi, offline. Il modello tiny e' costruito con la classe
`mlx_lm.models.<arch>.Model` (l'ORACOLO) e salvato come config.json + model.safetensors:
l'engine lo carica e si confrontano i logits a parita' di token-ids. Niente tokenizer
(la parita' e' su token-ids espliciti).

Uso:
  gen_tiny_ckpt.py <arch> <out_dir> [--ids 10,20,30]
Stampa, oltre a creare il checkpoint, i riferimenti (top-k logprob) di mlx_lm per quegli ids.
"""
import sys, os, json
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map, tree_flatten

# Config TINY per-arch (dim minime; i parametri arch-specifici con valori "reali" scalati).
# head_dim = hidden/n_heads deve restare sensato per RoPE/GQA.
TINY = {
    "ernie4_5_moe": dict(
        cls="ernie4_5_moe", model_type="ernie4_5_moe",
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        vocab_size=128, max_position_embeddings=512, rope_theta=10000.0,
        use_bias=False, tie_word_embeddings=False,
        moe_num_experts=4, moe_k=2, moe_intermediate_size=64,
        moe_layer_start_index=0, moe_layer_interval=2, moe_num_shared_experts=1,
        moe_gate_act="softmax", head_dim=16,
    ),
    "glm4": dict(
        cls="glm4", model_type="glm4",
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, attention_bias=True,
        head_dim=16, rms_norm_eps=1e-5, vocab_size=128,
        partial_rotary_factor=0.5, rope_theta=10000.0, rope_traditional=True,
        max_position_embeddings=512,
    ),
    "granite": dict(
        cls="granite", model_type="granite",
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        vocab_size=128, logits_scaling=8.0, attention_multiplier=0.015625,
        embedding_multiplier=12.0, residual_multiplier=0.22,
        max_position_embeddings=512, attention_bias=False, mlp_bias=False,
        rope_theta=5000000.0, tie_word_embeddings=True,
    ),
}


def build(arch):
    if arch not in TINY:
        raise SystemExit(f"arch '{arch}' non in TINY (aggiungila)")
    cfg = dict(TINY[arch])
    cls_name = cfg.pop("cls")
    mod = __import__(f"mlx_lm.models.{cls_name}", fromlist=["Model", "ModelArgs"])
    args = mod.ModelArgs(**cfg)
    model = mod.Model(args)
    # pesi casuali deterministici
    mx.random.seed(0)
    model.update(tree_map(lambda p: mx.random.normal(p.shape, dtype=p.dtype) * 0.05,
                          model.parameters()))
    mx.eval(model.parameters())
    return model, cfg


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    arch, out = sys.argv[1], sys.argv[2]
    ids = [10, 20, 30, 40, 50]
    for i, a in enumerate(sys.argv[3:]):
        if a == "--ids" and i + 1 < len(sys.argv):
            ids = [int(x) for x in sys.argv[4].split(",")]
    os.makedirs(out, exist_ok=True)
    model, cfg = build(arch)
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(out, "model.safetensors"), flat)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"tiny '{arch}' -> {out}  ({len(flat)} tensori)")
    lg = model(mx.array([ids]))[0, -1]
    lp = mx.log(mx.softmax(lg))
    print(f"REFERENCE ids={ids}:")
    for t in mx.argsort(lg)[::-1][:8].tolist():
        print(f"  tok {t} logprob {float(lp[t]):.5f}")


if __name__ == "__main__":
    main()
