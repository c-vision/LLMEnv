# THIRD_PARTY_NOTICES — `llmmetal_oq_a8`

Questo modulo contiene codice di terze parti. Questo file è la notifica di
attribuzione richiesta dalla licenza Apache-2.0 (sezioni 4(a), 4(b), 4(c)) e
registra l'esito dell'**audit di licenza** eseguito il 2026-09-16.

## 1. Codice oMLX (Apache-2.0) — copiato VERBATIM

Origine: [`jundot/omlx`](https://github.com/jundot/omlx),
`omlx/custom_kernels/qwen35_prefill/csrc/`.

| File in questo modulo | File upstream | Blob SHA (upstream) | Stato | SHA-256 locale |
|---|---|---|---|---|
| `csrc/oq_a8_decode.h` | `oq_a8_decode.h` | `e9bffe3b0e7651f221493571a382fe792c59cce9` | **byte-identico** | `2ff10170fa2e45a289bf9f93e750ea051f85e19004f9e79dbaeda9cded906a21` |
| `csrc/oq_a8_quantize.metal` | `qwen35_oq_a8.metal` | `e5cd3118d69ca3f7e68daa8c7c4fe87d8c9e0df8` | **byte-identico** | `e7b34fb1fcf27983d294ca825fe5e60a7999f9e037c9de22b2f29292d17901fe` |
| `csrc/oq_a8_qmm_nax.metal` | `qwen35_oq_a8_nax.metal` | `458fcc31c5163935b56254100e3debcff8070d92` | **byte-identico** | `20d3ed16de7ec9881f3946aaa0972378f4b51981028564e0adfa5a57904c7c15` |

### Modifiche apportate (§4(b))

L'unica modifica è il **rinominare** i file (nessuna modifica al contenuto,
verificata byte-per-byte con gli hash sopra):

- `qwen35_oq_a8.metal` → `oq_a8_quantize.metal`
- `qwen35_oq_a8_nax.metal` → `oq_a8_qmm_nax.metal`

Conseguenza cosmetica nota: il commento interno di `oq_a8_quantize.metal`
(«Only the GEMM in `qwen35_oq_a8_nax.metal` needs the 26.2 deployment floor»)
cita ancora il nome upstream del file rinominato.

### Copyright e NOTICE (§4(c), §4(d))

- I tre file upstream **non contengono header di copyright, SPDX o
  attribuzione** (verificato: nessuna occorrenza di `copyright`/`spdx`/
  `license`/`apache` nei file). Non c'era quindi alcuna nota da trattenere.
- `jundot/omlx` **non distribuisce un file `NOTICE`** (verificato: 404 su
  `NOTICE`). La sezione 4(d) quindi **non si applica**.
- Testo completo della licenza: [`LICENSE-oMLX-Apache-2.0.txt`](./LICENSE-oMLX-Apache-2.0.txt)
  in questa cartella (§4(a)).

## 2. Esito dell'audit: **nessuna derivazione da MTPLX** in questo modulo

> [!IMPORTANT] Verdetto verificato alla fonte
> La nota precedente secondo cui i kernel `qwen35_prefill` potessero essere
> "MTPLX-derived" è **falsificata**.
>
> - Il `NOTICE` di MTPLX dichiara la relazione nella **direzione opposta**:
>   MTPLX vende kernel **da** oMLX (`Qwen4 QSA sparse-GQA Steel` kernel,
>   PR oMLX #3244) — non viceversa.
> - L'acknowledgment di oMLX circoscrive l'ascendenza MTPLX a una **famiglia
>   diversa**: *"Lightning MTP's **verify-shape** Metal kernels are powered by
>   MTPLX"*. La famiglia `qwen35_prefill` (oQ mixed-bit QxA8) che abbiamo
>   copiato è codice oMLX proprio.
>
> **Quindi: nessun obbligo di attribuzione in-prodotto "Powered by MTPLX"
> ricade su questo modulo.**

### Regola per il futuro (§4(d) di MTPLX)

MTPLX è Apache-2.0 (**Copyright 2026 Youssof Altoukhi**) con una **NOTICE** che
impone l'attribuzione **in-prodotto** (non basta README/repo/sito):

```
Powered by MTPLX
https://github.com/youssofal/mtplx
```

Se e quando verrà vendorizzato codice della famiglia **verify-shape** di oMLX
(che è MTPLX-derived), o qualunque codice MTPLX, tale attribuzione diventa
**obbligatoria in-prodotto** per LLMEnv.

## 3. Codice proprio

Sono opera di questo progetto (non copiati) e non ricadono sotto le note sopra:

- `csrc/oq_a8_support.metal` — kernel di supporto (repack v8), puro Metal;
- `llmmetal_oq_a8_bridge.mm` — bridge ObjC++ zero-copy e dispatch del primitive;
- `README.md`, `build_metallibs.sh` e la struttura del modulo.

Le `*.metallib` sono artefatti di build e **non** sono versionate.

## 4. Altre dipendenze

- **MLX** (`libs/mlx-deps/mlx-src`) — licenza propria già presente in
  `libs/mlx-deps/mlx-src/LICENSE`.

## Riferimenti

- `jundot/omlx` — https://github.com/jundot/omlx (Apache-2.0)
- `youssofal/MTPLX` — https://github.com/youssofal/MTPLX (Apache-2.0 + NOTICE)
- Registro repo esterni (vault): `Knowledge/Repo-Esterni-Speculative-OutOfCore-Kernel-Registro.md`
