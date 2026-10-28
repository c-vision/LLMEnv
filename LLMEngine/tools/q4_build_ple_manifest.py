#!/usr/bin/env python3
"""Ricostruisce il manifest PLE per il checkpoint locale (ngram_table.bin = tabella intera)
e crea una VISTA del modello con `text_config.ple_storage`, poi testa l'ufficiale mlx-vlm.
Memory-safe: il manifest punta al file esistente (nessuna copia dei 32 GB)."""
import os, sys, json, struct, tempfile, shutil
from pathlib import Path
SRC = Path("/Users/gaetano/ai/Models/Qwen3.8-Flash-Next-MLX-Serve-4bit")
NG = SRC / "ngram_table.bin"
VIEW = Path("/tmp/q4view")
VIEW.mkdir(exist_ok=True)

# 1) header del safetensors per payload_start + data_offsets
with NG.open("rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hlen))
payload = 8 + hlen
w = hdr["weight"]; s = hdr["scales"]; b = hdr["biases"]
row_count = int(w["shape"][0]); row_width = int(w["shape"][1]) * 8
manifest = {
    "version": 2,
    "source_root": ".",
    "layout": "safetensors_ranges",
    "row_width": row_width,
    "row_count": row_count,
    "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
    "cache_rows": 0,
    "shards": [{
        "row_start": 0, "row_count": row_count,
        "weight": {"file": "ngram_table.bin", "offset": payload + int(w["data_offsets"][0]), "dtype": "U32", "shape": w["shape"]},
        "scales": {"file": "ngram_table.bin", "offset": payload + int(s["data_offsets"][0]), "dtype": "BF16", "shape": s["shape"]},
        "biases": {"file": "ngram_table.bin", "offset": payload + int(b["data_offsets"][0]), "dtype": "BF16", "shape": b["shape"]},
    }],
}
(VIEW / "ple-manifest.json").write_text(json.dumps(manifest, indent=2))
sys.stderr.write(f"[manifest] row_count={row_count} row_width={row_width} payload={payload}\n")

# 2) vista: symlink a tutti i file tranne config.json; config modificato con ple_storage
for f in SRC.iterdir():
    if not f.is_file() or f.name == "config.json": continue
    dst = VIEW / f.name
    if not dst.exists(): os.symlink(f, dst)
cfg = json.loads((SRC / "config.json").read_text())
tc = cfg.get("text_config")
if tc is None:
    sys.stderr.write("[warn] config senza text_config\n"); tc = cfg
tc["ple_storage"] = {"manifest": "ple-manifest.json", "cache_rows": 0}
(VIEW / "config.json").write_text(json.dumps(cfg, indent=2))
sys.stderr.write("[view] pronta in /tmp/q4view\n")

# 3) test ufficiale
sys.path.insert(0, "/tmp/mlxvlm-probe")
import threading, resource, time
LIMIT = int(os.environ.get("Q4_MEM_GB", "85")) * 1024**3
def _rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024
def _guard():
    while True:
        if _rss() > LIMIT: sys.stderr.write("[GUARD] kill\n"); os._exit(99)
        time.sleep(1)
threading.Thread(target=_guard, daemon=True).start()
import numpy as np, mlx.core as mx
from mlx_vlm.utils import load_model
from mlx_lm.utils import load_tokenizer
m = load_model(VIEW, lazy=True, strict=False)
lm = m.language_model
sys.stderr.write(f"[ref] loaded, RSS={_rss()/1e9:.1f}GB\n"); sys.stderr.flush()
tok = load_tokenizer(VIEW)
cache = lm.make_cache()
out = lm(mx.array([[1,2,3,4]], dtype=mx.int32), cache=cache)
lg = getattr(out, "logits", out); mx.eval(lg)
print(json.dumps({"greedy_1_2_3_4": int(mx.argmax(lg[:, -1, :], axis=-1).item())}))
