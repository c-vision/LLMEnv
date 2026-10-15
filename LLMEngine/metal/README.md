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
| **`qgemv4`** | **VINCITORE**: 1 threadgroup (256 thread) per riga di output; carichi `uint` coalesced; riduzione per-simdgroup + threadgroup. |
| `qgemv4s` | 1 simdgroup (32 lane) per riga, 8 righe per threadgroup; riduzione `simd_sum`. |
| `qgemv4v` | carichi vettoriali `uint4`/`float4`, un gruppo (64 pesi) per iterazione. |
| `bcopy` | sanity: copia vettoriale pura (misura la banda reale della macchina). |

### Metodo di misura (CRITICO)
Misurare **batched/throughput**, non per-chiamata: il decode reale **accoda** i dispatch e **non**
sincronizza per-op. Una misura per-call (con `commit`+`waitUntilCompleted` ogni volta) include
~0.1-0.3 ms di overhead di sync che **oscura il compute** (es. `qgemv4` BATCH=1 = 0.145 ms vs
BATCH=50 = 0.012 ms). Il bench fa `BATCH` dispatch per command buffer, un solo `waitUntilCompleted`.

### Risultati — `N=4096, K=4096, 4-bit g64` (8 MB di pesi)
| implementazione | ms/call | banda eff. | note |
|---|---|---|---|
| **OLD = MLX** (`llm_ai_mlx_gemv_bench`, `pipe=1`) | **~0.24** | ~35 GB/s | path `quantized_linear_cfg` |
| **NEW `qgemv4`** | **~0.012-0.015** | **~530-670** | ≈ alla banda |
| NEW `qgemv4s` | ~0.020-0.028 | ~300-470 | |
| `bcopy` (riferimento banda reale) | 0.253 | **531 GB/s** | |

→ **NEW ≈ 18× OLD** sulla GEMV. Il kernel `qgemv4` è **alla banda** (il floor a 8 MB / 531 GB/s
è ~0.015 ms); MLX è ~15-18× sotto. Parte dei picchi (fino a ~900 GB/s) è effetto cache: le rep
batched rileggono gli stessi pesi.

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
