#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ref_mlx_graph_diff — cattura la COMPOSIZIONE per-primitiva di UN passo di decode del
motore REFERENCE (mlx_lm, che è la base di rapidMLX/vLLM-MLX) e la stampa in forma
confrontabile col nostro GMB_DISPATCH (LLMEnv/LLMEngine/mlx/llm_mlx_dispatch_profile.cpp).

Perché serve: il confronto §23quater aveva solo il CONTEGGIO nodi aggregato (Llama 704 vs 509).
Questo diff-op-per-op risponde alla domanda decisiva e load-indipendente (i CONTEGGI non
dipendono dal carico): il reference fa GKI stessi AsType sulle scale che facciamo noi, o no?

Metodo identico al nostro profiler: CONTARE ALLA COSTRUZIONE del grafo, prima dell'eval
(mlx.export_to_dot cammina le primitive del grafo lazy). Un passo = model(token, cache) dopo
un prefill valutato, SENZA eval -> export.

Uso (env che ha mlx_lm con la stessa arch da testare):
  /Users/gaetano/ai/env-vllmlx/bin/python3 tools/ref_mlx_graph_diff.py \
      /Users/gaetano/ai/Models/Qwen3.8-27B-8bit [N_prompt]
"""
import collections
import os
import re
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/gaetano/ai/Models/Qwen3.8-27B-8bit"
    np = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    base = [760, 6511, 314, 9338, 369]
    prompt_ids = (base * ((np // len(base)) + 1))[:np] or [760]   # lunghezza ESATTA np

    model, tokenizer = load(model_dir)

    # 1. prefill VALUTATO per popolare la KV-cache (dopo eval le cache sono foglie vive).
    cache = mlx_cache.make_prompt_cache(model)
    tok = mx.array([prompt_ids])
    logits = model(tok, cache)
    mx.eval(logits)
    nxt = int(mx.argmax(logits[0, -1], axis=-1))

    # 2. UN passo di decode LAZY (niente eval) -> il grafo è camminabile.
    step_in = mx.array([[nxt]])
    step_out = model(step_in, cache)
    # Stesso scope del profiler GMB: il passo di GENERAZIONE include il sampler (argmax).
    # Il modello da solo finisce ai logits; GMB conta `next_id`. Includere l'argmax qui rende
    # il confronto op-per-op equo (ArgReduce/Squeeze del sampler compaiono su entrambi).
    if os.environ.get("REF_WITH_SAMPLER", "1") != "0":
        step_out = mx.argmax(step_out[0, -1], axis=-1)

    dot_path = "/tmp/ref_decode_step.dot"
    mx.export_to_dot(dot_path, step_out)

    # 3. tally per tipo di primitiva: mlx etichetta i nodi con il nome della classe; le
    #    forme sono sugli archi come "shape, dtype". Contiamo i nodi per label.
    #    FORMATO REALE riga-nodo: `{ 12345 [label ="Broadcast", shape=rectangle]; }`
    #    → c'è uno SPAZIO tra label e '=', e il blocco `{ <id> ` precede `[label`:
    #    serve re.search, NON re.match con `\s*\d+\s*\[label`.
    label_re = re.compile(r'\[\s*label\s*=\s*"([^"]+)"', re.M)
    with open(dot_path) as f:
        dot = f.read()

    # nodi: righe "{ <id> [label="..."] }" dove il label inizia col nome della primitiva
    nodes = collections.Counter()
    astype_shapes = collections.Counter()
    for line in dot.splitlines():
        m = re.search(r'\[\s*label\s*=\s*"([^"]+)"', line)
        if not m:
            continue
        lab = m.group(1)
        prim = lab.split("\\n")[0].strip("{ }").split(",")[0].strip()
        nodes[prim] += 1
    # AsType: conta anche quante sorgono da tensori-peso (2D grandi) vs attivazioni (3D [1,1,H])
    for m in re.finditer(r'\[\s*label\s*=\s*"\{AsType\\nshapes?\s*=\s*\{?([^}"]+)', dot):
        astype_shapes[m.group(1)] += 1

    total = sum(nodes.values())
    print(f"[ref] passo decode: nodi_totali={total} (confronta: il nostro GMB_DISPATCH 'nodi/token')")
    print("[ref] composizione per primitiva (desc):")
    for name, c in nodes.most_common():          # TUTTE: serve per "GMB <= mlx_lm su ogni op"
        print(f"    {name:<34} {c}")
    if astype_shapes:
        print("[ref] AsType per forma-operando:")
        for shp, c in astype_shapes.most_common(16):
            print(f"    AsType {shp:<24} {c}")
    print(f"[ref] dot salvato in {dot_path} (ri-peggabile per confronto MB/token)")


if __name__ == "__main__":
    main()
