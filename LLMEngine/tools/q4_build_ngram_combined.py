#!/usr/bin/env python3
"""Materialize a single `ngram_table.bin` (mlx-serve-ngram format) from the 128
inline ngram safetensors shards of an official mlx-community qwen4_exp checkpoint.

The engine in LLMEnv/LLMEngine/mlx only knows how to read `ngram_table.bin`
(a single safetensors with tensors `weight`/`scales`/`biases`); the official
checkpoint stores the same table split row-wise into 128 tensors
`...ngram_embedding.shards.N.{weight,scales,biases}`. This tool concatenates the
shards in numeric order (which is exactly the global row order used by the
reference `ShardedEmbedding`: shard = row // rows_per_shard) into one file so the
engine can be compared bit-for-bit against the reference on the real checkpoint.

Output layout (engine expectation, qwen4_exp_ngram_table_open):
  [8B LE header_len][header JSON][weight bytes][scales bytes][biases bytes]
with data_offsets relative to the end of the header.

Usage:
  python3 tools/q4_build_ngram_combined.py <model_dir> <output_file>
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

CHUNK = 32 * 1024 * 1024


def read_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return 8 + n, hdr


def copy_range(src: Path, off: int, nbytes: int, dst, remaining: int) -> int:
    with open(src, "rb") as f:
        f.seek(off)
        while remaining > 0:
            b = f.read(min(CHUNK, remaining))
            if not b:
                raise IOError(f"short read {src}")
            dst.write(b)
            remaining -= len(b)
    return remaining


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    model_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    index = json.load(open(model_dir / "model.safetensors.index.json"))
    weight_map = index["weight_map"]

    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards."
    nshards = 128
    shard_info = {}
    for s in range(nshards):
        base = f"{prefix}{s}"
        files = {leaf: weight_map[f"{base}.{leaf}"] for leaf in ("weight", "scales", "biases")}
        shard_info[s] = files

    # Header info (data offsets/sizes) from each shard's file header.
    file_hdrs: dict[str, tuple[int, dict]] = {}
    def hdr_for(fn: str):
        if fn not in file_hdrs:
            file_hdrs[fn] = read_header(model_dir / fn)
        return file_hdrs[fn]

    rows_per_shard = None
    total_rows = 0
    wbytes = sbytes = bbytes = 0
    for s in range(nshards):
        files = shard_info[s]
        for leaf, fn in files.items():
            _, hdr = hdr_for(fn)
            t = hdr[f"{prefix}{s}.{leaf}"]
            shape = t["shape"]
            nrows = shape[0]
            if rows_per_shard is None:
                rows_per_shard = nrows
            elif nrows != rows_per_shard:
                raise ValueError(f"shard {s} {leaf}: rows {nrows} != {rows_per_shard}")
            if leaf == "weight":
                wbytes += nrows * 4 * shape[1]
            elif leaf == "scales":
                sbytes += nrows * 2 * shape[1]
            elif leaf == "biases":
                bbytes += nrows * 2 * shape[1]
        total_rows += rows_per_shard

    W0, S0, B0 = 0, wbytes, wbytes + sbytes
    header = {
        "weight": {"dtype": "U32", "shape": [total_rows, 20], "data_offsets": [W0, W0 + wbytes]},
        "scales": {"dtype": "BF16", "shape": [total_rows, 5], "data_offsets": [S0, S0 + sbytes]},
        "biases": {"dtype": "BF16", "shape": [total_rows, 5], "data_offsets": [B0, B0 + bbytes]},
        "__metadata__": {"format": "mlx-serve-ngram", "bits": "4", "group_size": "32"},
    }
    hjson = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(hjson)) % 8
    hjson += b" " * pad
    data_start = 8 + len(hjson)
    total_bytes = data_start + wbytes + sbytes + bbytes

    print(f"[build] shards={nshards} rows_per_shard={rows_per_shard} total_rows={total_rows}")
    print(f"[build] weight={wbytes/1e9:.3f}GB scales={sbytes/1e9:.3f}GB biases={bbytes/1e9:.3f}GB")
    print(f"[build] header={len(hjson)}B total_file={total_bytes/1e9:.3f}GB -> {out_path}")

    if out_path.exists() and out_path.stat().st_size == total_bytes:
        print("[build] output already present with expected size; skipping")
        return 0

    part = out_path.with_suffix(out_path.suffix + ".part")
    with open(part, "wb") as dst:
        dst.write(struct.pack("<Q", len(hjson)))
        dst.write(hjson)
        done = 0
        for leaf in ("weight", "scales", "biases"):
            for s in range(nshards):
                fn = shard_info[s][leaf]
                ds, hdr = hdr_for(fn)
                t = hdr[f"{prefix}{s}.{leaf}"]
                off = t["data_offsets"][0]
                nbytes = t["data_offsets"][1] - off
                with open(model_dir / fn, "rb") as src:
                    src.seek(ds + off)
                    rem = nbytes
                    while rem > 0:
                        b = src.read(min(CHUNK, rem))
                        if not b:
                            raise IOError(f"short read {fn}")
                        dst.write(b)
                        rem -= len(b)
                done += nbytes
                if s % 16 == 15:
                    print(f"[build] {leaf}: shard {s+1}/{nshards} written ({done/1e9:.2f}GB)")
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(part, out_path)
    print(f"[build] done: {out_path} ({out_path.stat().st_size/1e9:.3f}GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
