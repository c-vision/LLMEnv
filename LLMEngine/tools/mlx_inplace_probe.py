import mlx.core as mx, time
def ev(a): mx.eval(a); return a
def one(r): return r[0] if isinstance(r,(list,tuple)) else r
k = mx.fast.metal_kernel(name="inc", input_names=["x"], output_names=["y"],
    source="uint i = thread_position_in_grid.x; y[i] = x[i] + 1.0f;")
x = ev(mx.zeros([8], dtype=mx.float32))
y = one(k(inputs=[x], output_shapes=[(8,)], output_dtypes=[mx.float32], grid=(8,1,1), threadgroup=(8,1,1))); ev(y)
print("functional: x[:3]=", [float(v) for v in x[:3]], "y[:3]=", [float(v) for v in y[:3]], "-> input NON mutato" if float(x[0])==0 else "INPUT MUTATO")
try:
    ka = mx.fast.metal_kernel(name="inca", input_names=["x"], output_names=["y"],
        source="uint i = thread_position_in_grid.x; y[i] = x[i] + 1.0f;", atomic_outputs=True)
    ya = one(ka(inputs=[x], output_shapes=[(8,)], output_dtypes=[mx.float32], grid=(8,1,1), threadgroup=(8,1,1))); ev(ya)
    print("atomic: x[0]=", float(x[0]), "ya[0]=", float(ya[0]), "-> ", "input NON mutato" if float(x[0])==0 else "INPUT MUTATO")
except Exception as e:
    print("atomic err:", str(e)[:120])
