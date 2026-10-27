# Patch locali a ds4
Serie numerata, applicata in ordine lessicografico da `tools/ds4_sync.sh` con `patch -p1` dentro
`vendor/`. Ogni patch deve avere un header con motivo e, se possibile, un riferimento alla issue.
Vietato modificare `vendor/` a mano: il check del manifest fallisce.
