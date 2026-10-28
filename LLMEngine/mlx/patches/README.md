# patches

Small patches to the MLX and mlx-c sources. The build applies them to the MLX copy it fetches, so the
engine can use features or fixes that are not in the pinned MLX version.

| Patch | Purpose |
|---|---|
| `mlx_metal41_nax.patch` | Lets the tensor-unit kernels compile with the Metal 4.1 compiler (applied by `tools/build_ai_libs.sh`). |
| `mlx_c_mlx032.patch` | Adapts mlx-c to the MLX 0.32 interface. |
| `mlx_c_inplace_slice*.patch`, `mlx_inplace_slice.patch` | In-place slice and slice-update operations. |
| `mlx_c_resource_count.patch`, `mlx_resource_count.patch` | Expose the number of live Metal resources. |
| `mlx_disable_qmv_wide.patch` | Lets the engine switch off the wide quantized matrix-vector kernel, whose results differ slightly from the normal one and break exact agreement between batched and single-token runs. |
| `mlx_sdpa_hd512.patch` | Adds attention kernels for head size 512. |
