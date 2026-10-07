#!/usr/bin/env python3
"""migrate_env.py — migra i `getenv("X")` di FUNZIONAMENTO a `llm_ai_mlx_opt_env("X")` (registro
condiviso LLMEngine/Options). I nomi di DEBUG (DBG/DUMP/TRACE/TIME/STATS/VERBOSE/PROBE/...) restano
env. Backup in /tmp/llmenv-backup/migrate/ prima di scrivere. Uso: da LLMEnv/  python3 tools/migrate_env.py <glob>..."""
import re, pathlib, shutil, sys

DBG = re.compile(r'DBG|DEBUG|DUMP|TRACE|_LOG|_TIME|STATS|VERBOSE|PROBE|DIAG|WIRELOG|ERRTRACE|WHICH_PATH')
BK = pathlib.Path("/tmp/llmenv-backup/migrate")

def add_include(ns):
    if 'llm_engine_opts.h' in ns: return ns
    # dopo il primo #include
    return re.sub(r'(\n#include [^\n]*)', r'\1\n#include "llm_engine_opts.h"   /* opt_env: registro */', ns, count=1)

def main():
    pats = sys.argv[1:] or ["LLMEngine/mlx/*.c", "LLMEngine/mlx/*.h"]
    files = []
    for p in pats: files += list(pathlib.Path(".").glob(p))
    tot = 0
    for f in files:
        s = f.read_text()
        def repl(m):
            n = m.group(1)
            return m.group(0) if DBG.search(n) else f'llm_ai_mlx_opt_env("{n}")'
        ns = re.sub(r'getenv\("([A-Z0-9_]+)"\)', repl, s)
        if ns == s: continue
        BK.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, BK / f.name)
        ns = add_include(ns) if f.suffix == ".c" else ns
        f.write_text(ns)
        k = len(re.findall(r'llm_ai_mlx_opt_env\("', ns))
        tot += k
        print(f"{f}: {k} siti")
    print(f"TOTALE: {tot}")

if __name__ == "__main__": main()
