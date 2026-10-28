import sys, time, json
from mlx_lm import load
from importlib import import_module
stream_generate = import_module("mlx_lm.generate").stream_generate

model_path = sys.argv[1]
steps = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [512, 1024, 2048, 4096]
ngen = 32

model, tok = load(model_path)
print(f"# loaded {model_path}", flush=True)

for n in steps:
    ids = [(i % 1000) + 1 for i in range(n)]
    # scarta il primo step di warm-up dello stesso N (metal kernel cache)
    for _ in range(1):
        for _ in stream_generate(model, tok, ids, max_tokens=2):
            pass
    t0 = time.perf_counter()
    ttft = None
    got = 0
    for resp in stream_generate(model, tok, ids, max_tokens=ngen + 1):
        if ttft is None:
            ttft = time.perf_counter() - t0
        got += 1
    total = time.perf_counter() - t0
    dec = got - 1 if got > 1 else 1
    decode_s = total - ttft
    print(f"CONTEXT N={n:6d} prefill={ttft:6.2f}s ({n/ttft:7.0f} tok/s) "
          f"decode{dec}={dec/decode_s:7.1f} tok/s  (got={got})", flush=True)
