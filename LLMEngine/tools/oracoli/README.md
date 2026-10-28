# oracoli/ — script, sorgenti e binari di misura (NON tracciati)

Questa cartella è deliberatamente **fuori da git** (vedi `.gitignore`, voce `oracoli/`): contiene
i tool e i binari locali di misura/oracolo, rigenerabili. La documentazione completa è nel vault:

    AiWiki/01-Projects/LLMEnv/Knowledge/oracoli.md

## Tool (script di utilizzo)
- `wd.sh`              watchdog RAM (uccide se l'RSS supera il tetto in GB).
- `rss_ab.sh`          A/B paired interleaved che misura il picco RSS (`/usr/bin/time -l`).
- `diag_run.sh`        run con oracoli deterministici (`LLM_DISPATCH`, `GEN_IDS`, `DF2_STATS`).
- `gen_long_prompt.py` prompt lungo deterministico per prefill/RSS; in piu':
      `--corpus FILE`   haystack di TESTO NATURALE (i recall su filler ripetuto sono gonfiati);
      `--needle CODE --depth F`  needle multi-token (run intero) a una profondita'.
- `repeat_ids.sh`      stessa richiesta N volte con `GEN_IDS=1` → confronto ID: leak di stato/determinismo.
- `spec_stats.sh`      da un log di diagnosi estrae PASS-TIME + ACCEPTANCE (mai tok/s).
- `build_probes.sh`    compila le sonde (sotto) QUI, dai sorgenti locali.

## Sonde (sorgenti + binari) — model-free
- `llm_gemv_bench_test.c`         GEMV quantizzato sintetico: GB/s effettivi (collo del decode).
- `llm_bw_test.c`                 banda pura read/write/triad.
- `llm_mlx_compile_cost_probe.c`  break-even del costo `closure_apply` (compile per-layer).
- `llm_mlx_attn_roof_probe.c` + `llm_mlx_attn_roof_probe_alias.cpp`  roofline attention.
- `llm_gather_qmm_sorted_probe.c`  parità `gather_qmm(sorted)` vs riferimento per-esperto (bug MLX #3856:
  M>32768 con `M%64≠0` corrompe la coda; forme reali esperti GLM-4.5).

I sorgenti di queste sonde sono stati **spostati qui** da `LLMEngine/mlx/tests/` (non erano
referenziati dal build: nessun `CMakeLists.txt` li compila).

> ⚠️ **Caveat cache-resident**: `llm_gemv_bench_test` misura **una forma in loop**; se sta nella cache
> (MALL/LLC) il GB/s **sovrastima** il modello (fino a 3-7×; oracoli.md §2b/§5.16). Usalo come
> **rapporto A/B**, non come assoluto; dimensiona il working set come il modello.

## Uso
    bash oracoli/build_probes.sh            # rigenera i binari
    # NB: adt/sdt sono NUMERI (0=f32 1=f16 2=bf16), NON nomi ("bf16" -> atoi=0=f32!)
    oracoli/llm_gemv_bench_test 1 8192 8192 4 64 2 2 50 0
    oracoli/llm_bw_test 512 0 10
    oracoli/llm_mlx_compile_cost_probe
    oracoli/wd.sh <cap_GB> <prompt|-> <comando...>
    oracoli/rss_ab.sh 11 /tmp/p16k.txt 32768 LLM_GPTOSS_BF16 1 0 3 -- prefix_cache=off

Documentazione d'uso completa: `AiWiki/01-Projects/LLMEnv/Knowledge/oracoli.md` §2b.
