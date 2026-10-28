# plugin

How new model features are added without changing the engine core.

- A new **number format** is a *codec* plugin: a shared library `libllmcodec_<name>.dylib` that exports
  `llmcodec_describe()`.
- A new **model building block** (an attention variant, for example) is a *primitive* plugin:
  `libllmengine_plugin_prim_<name>.dylib`, exporting `llmengine_plugin_prim_describe()`.

The model description file (`.llmarch`, see `LLMDesc`) names the plugins it needs and the engine loads
them at run time with `dlopen`. Plugins are searched in the folder named by `LLMENGINE_PLUGIN_DIR`,
then in the repository's plugin resources, then next to the executable.

Files: `llmengine_plugin.h` (the interface) and `llmengine_plugin.c` (the loader).
