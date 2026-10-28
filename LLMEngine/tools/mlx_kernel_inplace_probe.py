import mlx.core as mx
# kernel che SCRIVE su un input (device buffer) — test se MLX passa il buffer reale (no copia)
k = mx.fast.metal_kernel(
    name="append_inplace",
    input_names=["buf", "tok", "pos"],
    output_names=["dummy"],
    source="""
        uint t = thread_position_in_grid.x;
        if (t == 0) {
            uint P = pos[0];
            for (uint d = 0; d < 4; d++) buf[P*4 + d] = tok[d];
        }
        dummy[0] = 0.0f;
    """,
    ensure_row_contiguous=False)
buf = mx.zeros([8, 4], dtype=mx.float32); tok = mx.full([4], 9.0, dtype=mx.float32)
pos = mx.array([5], dtype=mx.uint32)
mx.eval(buf, tok, pos)
r = k(inputs=[buf, tok, pos], output_shapes=[(1,)], output_dtypes=[mx.float32],
      grid=(1,1,1), threadgroup=(1,1,1))
mx.eval(r)
print("after kernel, buf[5] =", [float(v) for v in buf[5]], " (atteso [9,9,9,9] se SCRIVE sull'input)")
print("buf sum =", float(mx.sum(buf)))
