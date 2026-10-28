# ds4

A fast engine for large GGUF models (mixture-of-experts in particular), built on
[ds4](https://github.com/antirez/ds4) by Salvatore Sanfilippo (antirez), MIT license. ds4 is included
here as a copy, together with a few changes of ours, laid out so that updating to a newer ds4 never
touches our own code.

| Folder | Contents |
|---|---|
| `vendor/` | The ds4 sources, **unmodified**, with its Metal kernels, licenses and a checksum manifest. |
| `patches/` | Our changes to ds4, as a numbered series of patches. |
| `bridge/` | Our glue code: a token-level API on top of ds4 and the adapter to the common interface. |
| `tools/` | `ds4_sync.sh` (update from upstream) and `ds4_check.sh` (verify the manifest). |

## Updating ds4

```sh
LLMEngine/ds4/tools/ds4_check.sh                  # must report a clean tree
LLMEngine/ds4/tools/ds4_sync.sh /path/to/ds4      # or a git URL and a ref
```

The sync copies the new sources, reapplies the patches and rewrites the manifest. If a change to ds4 is
needed, add a patch in `patches/`. Never edit `vendor/` by hand, or the manifest check fails.

The snapshot in use is recorded in `UPSTREAM.md`.
