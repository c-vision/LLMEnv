# tools

Scripts and test data for building, packaging and releasing LLMEnv.

| File | Purpose |
|---|---|
| `build_ai_libs.sh` | Builds all the engine libraries and collects them in `build/dist/<platform>/`. |
| `make_macos_package.sh` | Makes the macOS release files from that folder: a `.tar.gz`, a `.pkg`, checksums and the Homebrew formula. |
| `publish_libs_*.sh` | Maintainers only: uploads the built libraries to the package store used by the maintainers' other projects. |
| `Fixtures/` | Reference data used by the tests. See its README. |

## Typical use

```sh
tools/build_ai_libs.sh               # build everything (incremental)
tools/make_macos_package.sh          # build/package/llmenv-<version>-macos-arm64.{tar.gz,pkg}
```

The version comes from the `VERSION` file at the root of the repository.
