# llmmetal_oq_a8 — port dei kernel oMLX (int8-activation × Q4/Q5, tensor unit M5)

**Origine**: `jundot/omlx` `omlx/custom_kernels/qwen35_prefill/csrc/` (copiati VERBATIM, 2026-09-15).
- `csrc/oq_a8_decode.h`, `csrc/oq_a8_qmm_nax.metal` (GEMM NAX), `csrc/oq_a8_quantize.metal` (Stage-A).

**Build** (toolchain Metal CLI; sono obbligatori sia `-std=metal4.0`, che definisce
`__HAVE_TENSOR__`, sia il deployment floor macOS 26.2 richiesto da MPP/NAX):
```sh
MLX=libs/mlx-deps/mlx-src
xcrun -sdk macosx metal -I "$MLX" -I csrc -std=metal4.0 -fno-fast-math -mmacosx-version-min=26.2 -c csrc/oq_a8_qmm_nax.metal -o /tmp/q.air
xcrun -sdk macosx metal -mmacosx-version-min=26.2 /tmp/q.air -o /tmp/oq_a8_qmm_nax.metallib
xcrun -sdk macosx metal -I "$MLX" -I csrc -std=metal4.0 -fno-fast-math -mmacosx-version-min=26.2 -c csrc/oq_a8_quantize.metal -o /tmp/a.air
xcrun -sdk macosx metal -mmacosx-version-min=26.2 /tmp/a.air -o /tmp/oq_a8_quantize.metallib
```
Funzioni: `oq_a8_quantize_float16_t_am0/1`, `oq_a8_qmm_t_nax_v8_q4_am0_float16_t_**wm_2_wn_2**` (variante
"800"), `oq_a8_decode_weights_q4/q5` (riferimento decode). **Nota (aggiornata 2026-09-16)**: le
`*.metallib` sono ora **risorse versionate** in `Resources/` (il progetto risorse le copia
nell'output della build, accanto alla dylib) e il motore le risolve **automaticamente accanto alla
dylib** (`dladdr`), senza env. Il primitive e' **AUTO per modello dal 2026-09-18** (commit
`3346c675d`): env assente -> attivo su tutti gli eleggibili (l'eleggibilita' — affine Q4/Q5 gs64,
M>=128, N%64, K%64 — filtra da sola); `LLMMETAL_OQ_A8=1` forza, `LLMMETAL_OQ_A8=0` disabilita. Prima era
opt-in: la promozione a default ON del 2026-09-16 era basata su una misura **contaminata** (il
vecchio bench appendeva piu' volte nella stessa sessione) e con `llmmetal_batch_scale_probe` (sessione
nuova per N + warm-up) il quadro e' **model-dependent**:
  Qwen3-8B-4bit  N=1024/2048/4096 -> **ON vince +16..28%** (0,72x/0,84x/0,77x)
  Llama-1B-4bit  N=16..2048      -> **ON PERDE 1,9-5,9x** (l'overhead per-call del bridge eager non si
                                    ammortizza sui GEMM piu' piccoli; non e' il packed-qkv)
**Multi-chunk RISOLTO** (2026-09-18): la cache prep `scT/biT` era riallocata per-call nel primitive
(il 2o chunk del prefill leggeva memoria mai scritta); regressione load-free
`tests/oq_a8_prep_cache_selfcheck.c`.
E' **approssimato** (INT8 sulle attivazioni): gate di qualita' (teacher forcing su 256 posizioni di testo
reale) = **3,52% argmax divergenti** (i precedenti accettati: 1,83-2,11%). FAIL-OPEN: senza metallib o su
ineleggibilita' si ricade sul path normale, senza errori. Si rigenerano coi comandi sopra quando si tocca il kernel, e vanno ricopiate in
`Resources/`.

**Stato**: compilano, le metallib si caricano ed eseguono.
- **Stage-A (quantizzazione attivazione): VERIFICATO ESATTO** (`tests/oq_a8_omlx_selfcheck.m`, mismatch=0).
- **Packing pesi Q4: VERIFICATO ESATTO** col decoder oMLX (`mismatch=0`).
- **GEMM Q4 mode 0: VERIFICATO** su M5 Max. Il one-hot produce solo `out[3,7]=1.0`; il caso
  casuale `M=64,N=128,K=128` dà `GEMM mismatch=0`, `maxdiff=0.0307` contro il riferimento affine
  FP64 (rumore dell'output FP16). Il kernel applica internamente `sa[m]` in activation mode 0.
- Senza `-mmacosx-version-min=26.2` il kernel può caricarsi ed eseguire ma usa una convenzione NAX
  non valida: il sintomo osservato era `8141/8192` mismatch e `maxdiff=103.4285`.

**Non in CI**: harness manuale.

**Licenza / provenienza**: i tre file copiati sono **byte-identici** agli
upstream oMLX (solo rinominati) e ricadono sotto **Apache-2.0** (§4(a) licenza
inclusa, §4(b) notifica di modifica). **Audit di licenza 2026-09-16: nessuna
derivazione da MTPLX in questo modulo** (l'ascendenza MTPLX riguarda la
famiglia *verify-shape*, non `qwen35_prefill`) → nessun obbligo "Powered by
MTPLX". Dettagli, hash e regola per il futuro in
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
