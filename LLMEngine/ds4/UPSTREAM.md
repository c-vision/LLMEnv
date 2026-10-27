# ds4 upstream (antirez) — snapshot vendorizzato

- Repo: `https://github.com/antirez/ds4` (riferimento locale: `/Users/gaetano/ds4`)
- Commit: `ee7e8405232c8971000cc7ba2b945b00bf6d20d3`
- Data: 2026-10-21 09:30:33 +0200
- Licenza: MIT (vedi `vendor/LICENSE` e `vendor/licenses/`)

## Regole
1. `vendor/` è una copia **pristine**: nessuna modifica manuale.
2. Le nostre modifiche vivono in `patches/*.patch` (serie ordinata) oppure in `bridge/`.
3. Aggiornamento: `tools/ds4_sync.sh <upstream-dir|git-url> [ref]` (verifica il manifest, ricopia,
   riapplica le patch, rigenera il manifest).
4. Verifica integrità: `tools/ds4_check.sh`.
