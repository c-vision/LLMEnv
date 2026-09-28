#!/usr/bin/env python3
"""Dumpa round0 del golden DFlash2 in un file binario C-readable per l'harness
di validazione (dflash2_validate.c). Output: round0.bin accanto a questo file.

Layout (little-endian):
  int32  S
  int32  anchor
  int32  n_expected
  int32  expected[7]
  int32  n_ctx                       (S*25600)
  float  ctx[n_ctx]
  int32  n_layers_snap               (6: layer0..4 + final_norm)
  int32  snap_shape[3]               (1,8,5120)
  float  snaps[6][8*5120]
"""
import json
import struct
import sys
import numpy as np

base = sys.argv[1] if len(sys.argv) > 1 else "."
npz = np.load(f"{base}/dflash_goldens.npz")
meta = json.load(open(f"{base}/dflash_goldens.json"))

round0 = meta["rounds"][0]
S = round0["ctxS"]
anchor = int(round0["anchor"])
expected = [int(x) for x in round0["draft_tokens"]]
ctx = np.ascontiguousarray(npz["round0_ctx"], dtype=np.float32)
snaps = []
for tag in ["layer0", "layer1", "layer2", "layer3", "layer4", "final_norm"]:
    snaps.append(np.ascontiguousarray(npz[f"round0_{tag}"], dtype=np.float32))

assert ctx.shape == (S, 25600), ctx.shape
assert len(expected) == 7

out = base + "/round0.bin"
with open(out, "wb") as f:
    f.write(struct.pack("<i", S))
    f.write(struct.pack("<i", anchor))
    f.write(struct.pack("<i", len(expected)))
    for t in expected:
        f.write(struct.pack("<i", t))
    f.write(struct.pack("<i", int(ctx.size)))
    f.write(ctx.tobytes())
    f.write(struct.pack("<i", len(snaps)))
    shp = snaps[0].shape
    f.write(struct.pack("<iii", shp[0], shp[1], shp[2]))
    for s in snaps:
        assert s.shape == shp
        f.write(s.tobytes())

print(f"scritto {out}: S={S} anchor={anchor} expected={expected} ctx={ctx.shape}")
