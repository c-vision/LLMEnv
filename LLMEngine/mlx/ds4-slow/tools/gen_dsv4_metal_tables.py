#!/usr/bin/env python3
"""gen_dsv4_metal_tables.py — estrae le tabelle IQ2 dal CODEC (§0bis: unica sorgente)
e le emette come stringa Metal per il backend GPU DS4 (llm_dsv4_metal.mm).

Uso (da LLMEnv/):  python3 tools/gen_dsv4_metal_tables.py
Output:            LLMEngine/mlx/llm_dsv4_metal_tables.h  (generato, non editare a mano)
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "LLMCodec/src/llmcodec_ggml_iq_grids.c"
OUT = ROOT / "LLMEngine/mlx/llm_dsv4_metal_tables.h"


def extract(src: str, name: str) -> list[str]:
    m = re.search(r"\b" + name + r"\[\d+\]\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        raise SystemExit(f"tabella {name} non trovata")
    body = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    return [t.strip() for t in body.split(",") if t.strip()]


def emit_metal(name: str, ctype: str, vals: list[str], per_line: int) -> str:
    out = [f"constant {ctype} {name}[{len(vals)}] = {{"]
    for i in range(0, len(vals), per_line):
        out.append("    " + ", ".join(vals[i : i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def main() -> None:
    src = SRC.read_text()
    grid = extract(src, "iq2xxs_grid")
    kmask = extract(src, "kmask_iq2xs")
    ksign = extract(src, "ksigns_iq2xs")

    metal = "\n".join(
        [
            emit_metal("iq2xxs_grid", "ulong", grid, 8),
            emit_metal("kmask_iq2xs", "uchar", kmask, 8),
            emit_metal("ksigns_iq2xs", "uchar", ksign, 8),
        ]
    )

    # C string escape
    esc = metal.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    OUT.write_text(
        "/* GENERATO da tools/gen_dsv4_metal_tables.py — NON modificare a mano.\n"
        " * Tabelle di FORMATO IQ2 (byte-exact dal CODEC: LLMCodec/src/llmcodec_ggml_iq_grids.c).\n"
        " * Sorgente unica (§0bis): il backend GPU DS4 le incorpora nel kernel Metal. */\n"
        "#ifndef LLM_DSV4_METAL_TABLES_H\n#define LLM_DSV4_METAL_TABLES_H\n\n"
        "static const char* const LLM_DSV4_METAL_TABLES =\n"
        f'"{esc}";\n\n'
        "#endif /* LLM_DSV4_METAL_TABLES_H */\n"
    )
    print(f"scritto {OUT} ({len(grid)} grid, {len(ksign)} signs, {len(kmask)} mask)")


if __name__ == "__main__":
    main()
