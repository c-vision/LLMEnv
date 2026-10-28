import sys, time, mlx.core as mx, mlx_lm
path = sys.argv[1]; N = int(sys.argv[2])
model, tok = mlx_lm.load(path)
ids = mx.random.randint(0, 1000, (1, N))  # prefill timing: il contenuto non conta
mx.eval(model(ids))
best = 1e9
for _ in range(2):
    mx.synchronize(); t0 = time.perf_counter(); out = model(ids); mx.eval(out); t = time.perf_counter() - t0
    best = min(best, t*1000)
print(f"MLXLM_PREFILL N={N} {best:.1f} ms")
