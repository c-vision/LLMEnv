#!/usr/bin/env python3
"""gen_ggml_iq_grids.py — estrae le codebook grid IQ di ggml (dati di FORMATO, non codice) da
`ggml-common.h` e genera un file C di sole tabelle.

Uso:
  tools/gen_ggml_iq_grids.py [<ggml-common.h>] [<out.c>]

Default ggml-common.h: nella _deps di llama.cpp del repo (build host). Default out:
native/llm_runtime/src/llmcodec_ggml_iq_grids.c.
"""
import re
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "native/llmengine/build/_deps/llama_cpp-src/ggml/src/ggml-common.h")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    PROJ, "src/llmcodec_ggml_iq_grids.c")
SRC_Q = os.path.join(REPO, "native/llmengine/build/_deps/llama_cpp-src/ggml/src/ggml-quants.c")

# (nome, tipo C, NGRID o None) — solo le grid usate dai dequant IQ.
GRIDS = [
    ("iq1s_grid",   "uint64_t", "NGRID_IQ1S"),
    ("iq2xxs_grid", "uint64_t", None),
    ("iq2xs_grid",  "uint64_t", None),
    ("iq2s_grid",   "uint64_t", None),
    ("iq3xxs_grid", "uint32_t", None),
    ("iq3s_grid",   "uint32_t", None),
    ("kmask_iq2xs", "uint8_t",  None),
    ("ksigns_iq2xs", "uint8_t", None),
]

# tabelle "kgrid" (indici), definite come static local dentro iq2xs/iq3xs_init_impl.
KGRIDS = [
    ("kgrid_2bit_256",  "uint16_t", "256"),
    ("kgrid_2bit_512",  "uint16_t", "512"),
    ("kgrid_1bit_2048", "uint16_t", "NGRID_IQ1S"),
    ("kgrid_2bit_1024", "uint16_t", "1024"),
    ("kgrid_256",       "uint16_t", "256"),
    ("kgrid_512",       "uint16_t", "512"),
]


def extract_kgrid(text, name):
    m = re.search(r"static const uint16_t\s+" + re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise SystemExit("kgrid non trovata: " + name)
    nums = ",".join(m.group(1).split())
    return [v.strip() for v in nums.split(",") if v.strip()]


def extract(text, name):
    m = re.search(r"GGML_TABLE_BEGIN\(\s*\w+\s*,\s*" + re.escape(name) + r"\s*,\s*([^)]+)\)(.*?)GGML_TABLE_END\(\)",
                  text, re.S)
    if not m:
        raise SystemExit("tabella non trovata: " + name)
    decl_n = m.group(1).strip()          # e.g. "256" oppure "NGRID_IQ1S"
    nums = ",".join(m.group(2).split())
    vals = [v.strip() for v in nums.split(",") if v.strip()]
    return decl_n, vals


def main():
    text = open(SRC, "r", encoding="utf-8", errors="replace").read()
    text_q = open(SRC_Q, "r", encoding="utf-8", errors="replace").read()
    n_grid_iq1s = None
    m = re.search(r"#define\s+NGRID_IQ1S\s+(\d+)", text)
    if m:
        n_grid_iq1s = int(m.group(1))
    lines = []
    lines.append("/* llmcodec_ggml_iq_grids.c — codebook grid IQ di ggml (DATI DI FORMATO, non codice).")
    lines.append(" * GENERATO da tools/gen_ggml_iq_grids.py a partire da ggml-common.h (dati tabellari,")
    lines.append(" * non espressione creativa). Non modificare a mano: rigenera col tool. */")
    lines.append('#include "llmcodec_ggml_iq_grids.h"')
    lines.append("")
    out_meta = []
    for name, ctype, _ in GRIDS:
        decl_n, vals = extract(text, name)
        if decl_n == "NGRID_IQ1S":
            n = n_grid_iq1s if n_grid_iq1s else len(vals)
        else:
            n = int(decl_n)
        assert len(vals) == n, f"{name}: attesi {n}, trovati {len(vals)}"
        lines.append(f"const {ctype} {name}[{n}] = {{")
        for i in range(0, n, 8):
            lines.append("    " + ", ".join(vals[i:i + 8]) + ",")
        lines.append("};")
        lines.append("")
        out_meta.append((name, n))
    for name, ctype, decl in KGRIDS:
        vals = extract_kgrid(text_q, name)
        n = n_grid_iq1s if decl == "NGRID_IQ1S" else int(decl)
        assert len(vals) == n, f"{name}: attesi {n}, trovati {len(vals)}"
        lines.append(f"const {ctype} {name}[{n}] = {{")
        for i in range(0, n, 12):
            lines.append("    " + ", ".join(vals[i:i + 12]) + ",")
        lines.append("};")
        lines.append("")
        out_meta.append((name, n))
    body = "\n".join(lines)
    open(OUT, "w", encoding="utf-8").write(body)
    # header con le dichiarazioni
    hdr = os.path.join(os.path.dirname(OUT), "llmcodec_ggml_iq_grids.h")
    h = ["/* llmcodec_ggml_iq_grids.h — dichiarazioni delle codebook grid IQ (dati di formato, generati). */",
         "#ifndef LLMCODEC_GGML_IQ_GRIDS_H", "#define LLMCODEC_GGML_IQ_GRIDS_H", "#include <stdint.h>",
         "#ifdef __cplusplus", 'extern "C" {', "#endif"]
    for name, ctype, _ in GRIDS:
        _, vals = extract(text, name)
        n = n_grid_iq1s if _ == "NGRID_IQ1S" else int(extract(text, name)[0])
        h.append(f"extern const {ctype} {name}[{n}];")
    for name, ctype, decl in KGRIDS:
        n = n_grid_iq1s if decl == "NGRID_IQ1S" else int(decl)
        h.append(f"extern const {ctype} {name}[{n}];")
    h += ["#ifdef __cplusplus", "}", "#endif", "#endif"]
    open(hdr, "w", encoding="utf-8").write("\n".join(h) + "\n")
    print("scritto", OUT, "+", hdr, "|", ", ".join(f"{n}={c}" for n, c in out_meta))


if __name__ == "__main__":
    main()
