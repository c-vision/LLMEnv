# metal — kernel Metal PROPRI (motore LLM GMBasic/LLMEnv)

Kernel Metal scritti da noi (nessuna dipendenza da MLX) per l'engine LLM. Questo è il percorso
che **sostituisce** la reimplementazione MLX-in-C per le operazioni calde. Ogni pezzo si implementa
e si misura **OLD (MLX) vs NEW (nostro)** con lo stesso metodo.

Licenza: **Apache-2.0** (esecuzione/inferenza) — parte di LLMEnv.

## Indice
- **Pezzo #1 — GEMV quantizzata 4-bit (decode, m=1)** — FATTO.
- Pezzo #2 — GEMM quantizzata (prefill, m>1) — in corso (`/tmp/qgemm.*`, non ancora consolidato).

---

## Pezzo #1 — GEMV quantizzata 4-bit affine (group 64), m=1

`gemb_kernel.metal` espone 4 kernel:
| kernel | descrizione |
|---|---|
| **`qgemv4s`** | **VINCITORE**: 1 simdgroup (32 lane) per riga, 8 righe per threadgroup (grid `(N+7)/8`); riduzione `simd_sum`. |
| `qgemv4` | 1 threadgroup (256 thread) per riga (grid `N`); riduzione per-simdgroup + threadgroup. |
| `qgemv4v` | carichi vettoriali `uint4`/`float4` — **veloce ma OUTPUT ERRATO (bug), non usare**. |
| `bcopy` | sanity: copia vettoriale pura (misura la banda reale della macchina). |

> ⚠️ **Attenzione grid**: `qgemv4`/`qgemv4v` richiedono grid `N` (1 riga/TG, `n=tgid`); `qgemv4s`
> richiede grid `(N+7)/8`. Una grid sbagliata calcola **meno righe** e gonfia la banda misurata.

### Metodo di misura (CRITICO)
Misurare **batched/throughput**, non per-chiamata: il decode reale **accoda** i dispatch e **non**
sincronizza per-op. Una misura per-call (con `commit`+`waitUntilCompleted` ogni volta) include
~0.1-0.3 ms di overhead di sync che **oscura il compute** (es. `qgemv4` BATCH=1 = 0.145 ms vs
BATCH=50 = 0.012 ms). Il bench fa `BATCH` dispatch per command buffer, un solo `waitUntilCompleted`.

### Risultati — `N=4096, K=4096, 4-bit g64` (8 MB di pesi), grid corretta
| implementazione | ms/call | banda eff. | note |
|---|---|---|---|
| **OLD = MLX** (`llm_ai_mlx_gemv_bench`, `pipe=1`) | **~0.24** | ~35 GB/s | path `quantized_linear_cfg` |
| **NEW `qgemv4s` (vincitore)** | **~0.017-0.019** | **~460** | ≈ 88% del roof |
| NEW `qgemv4` | ~0.020 | ~420 | |
| `qgemv4v` | ~0.016 | ~520 | **output errato (bug)** |
| `bcopy` (riferimento banda reale) | 0.258 | **521 GB/s** | |

→ **NEW ≈ 13× OLD** sulla GEMV. Il kernel `qgemv4s` è **~88% del roof** di banda (521 GB/s);
MLX è ~13× sotto. (NB: correzione del 2026-10-15 — la prima stesura indicava `qgemv4` come
vincitore a 0.012 ms / 18×: era un **bug di grid** (calcolava 1/8 delle righe).)

### Build / run
```bash
bash LLMEngine/metal/build_bench.sh                 # -> build/gemb_bench (+ build/gemb_kernel.metal)
build/gemb_bench [kernel] [N] [K] [reps] [BATCH]    # default: qgemv4 4096 4096 30 50
build/gemb_bench qgemv4  4096 4096 30 50
build/gemb_bench bcopy   4096 4096 30 50            # banda reale della macchina
```

### Confronto con OLD (MLX)
La gamba OLD si ottiene dalla bench dell'engine: `llm_ai_mlx_gemv_bench(m=1, n, k, bits, group,
adt, sdt, reps, pipe)` — usare **`pipe=1`** (pipelined: `reps` call lazy + 1 eval = throughput).
Costruita via `LLMEngine/tools/oracoli/build_probes.sh`.

---

## Pezzo #2 — GEMM quantizzata (prefill, m>1), tensor unit (MXU) — in corso
Prima stesura (`simdgroup_matrix` + dequant 4-bit→f16 in shared): a **parità** con MLX per M=128
(0.92 vs 0.88 ms), **~1.8× sotto** per M=512 (2.17 vs 1.20). Il dequant-in-shared è ridondante e
le barriere starvationano la MXU. Da consolidare dopo il tuning (dequant-once + GEMM MXU puro,
double-buffering, peso residente). Riferimento: `Todo/(open) - Motore-Metal-Proprio-Piano.md`.
