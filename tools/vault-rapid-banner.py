#!/usr/bin/env python3
"""vault-rapid-banner.py — inserisce un banner di disambiguazione (in testa, dopo il frontmatter)
in TUTTI i documenti del vault GMBasic che citano Rapid-MLX, distinguendo:
  1. repo REMOTO ufficiale
  2. copia LOCALE del repo ufficiale (versione + percorsi)
  3. FORK personale (percorsi)
Nota canonica: Knowledge/Rapid-MLX-Repo-Remoto-Copia-Fork-Percorsi.md

Idempotente (salta i file che gia' contengono il marker). Dry-run di default; `--apply` scrive.
Uso: python3 tools/vault-rapid-banner.py [--root /Users/gaetano/Documents/AiWiki/01-Projects/GMBasic] [--apply]
"""
import argparse
import os
import re
import sys

MARKER = "Rapid-MLX-Repo-Remoto-Copia-Fork-Percorsi"
BANNER = (
    "> [!info] Rapid-MLX — tre identità distinte (remoto / copia ufficiale / fork) · agg. 2026-09-17\n"
    "> - **Remoto ufficiale**: https://github.com/raullenchai/Rapid-MLX (tag **v0.14.2**)\n"
    "> - **Copia locale ufficiale**: `/Users/gaetano/ai/Rapid-MLX-official` (v0.14.2) + venv "
    "`/Users/gaetano/ai/env-rapid-official` (`rapid-mlx==0.14.2`); usata da `LLMEngine/tools/bench4_compare.sh`\n"
    "> - **Fork personale**: `/Users/gaetano/Documents/Projects/Rapid-MLX-NextGen` "
    "(branch `dev`, HEAD `fa7f291d`, ~0.13.1; editable in `env312`, `--version` stale 0.10.3)\n"
    "> Canonico: `Knowledge/Rapid-MLX-Repo-Remoto-Copia-Fork-Percorsi.md`\n"
)
PAT = re.compile(r"Rapid-?MLX|rapid[-_]mlx|RapidMLX|rapidmlx|rapim-mlx", re.IGNORECASE)


def _splice(lines, end, first_content):
    head = lines[: end + 1]
    rest = lines[end + 1:]
    while rest and rest[0].strip() == "":
        rest.pop(0)
    return "\n".join(head) + ("\n\n" if first_content else "\n") + BANNER + "\n" + "\n".join(rest)


def insert_banner(text):
    """Banner dopo il frontmatter YAML, o dopo il primo H1, o in testa. None se gia' presente."""
    if MARKER in text:
        return None
    lines = text.split("\n")
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return _splice(lines, i, True)
    # senza frontmatter: subito dopo il primo H1 se e' nelle prime righe
    for i in range(0, min(3, len(lines))):
        if lines[i].startswith("# "):
            return _splice(lines, i, True)
    return BANNER + "\n" + text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/gaetano/Documents/AiWiki/01-Projects/LLMEnv")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    targets = []
    for dirpath, _dirs, files in os.walk(a.root):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read()
            except Exception:
                continue
            if os.path.basename(p).startswith("Rapid-MLX-Repo-Remoto-Copia-Fork"):
                continue  # la nota canonica non porta il banner (auto-riferimento)
            if PAT.search(txt):
                targets.append((p, txt))

    changed = 0
    for p, txt in sorted(targets):
        new = insert_banner(txt)
        if new is None:
            continue
        changed += 1
        rel = os.path.relpath(p, a.root)
        if a.apply:
            with open(p, "w", encoding="utf-8") as f:
                f.write(new)
            print(f"  APPLY {rel}")
        else:
            print(f"  DRY   {rel}")
    print(f"\n{changed} file da modificare su {len(targets)} che citano rapid "
          f"({'APPLICATO' if a.apply else 'DRY-RUN'})")


if __name__ == "__main__":
    sys.exit(main())
