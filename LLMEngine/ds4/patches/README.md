# patches

Our changes to ds4, kept apart from the original sources in `../vendor/`.

The patches are numbered (`0001-*.patch`, `0002-*.patch`, and so on). `tools/ds4_sync.sh` applies them in
that order with `patch -p1` inside `vendor/`. Each patch should start with a header that explains why it
exists. Do not edit `vendor/` directly: the manifest check (`tools/ds4_check.sh`) will fail.
