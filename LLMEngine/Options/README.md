# Options

The engine's settings registry. Every setting has a name, a type, a built-in default, and a current
value. Programs set options **by name, before loading a model**, and the engine reads them from the
registry. Engine behaviour is deliberately not driven by environment variables.

```c
llm_ai_mlx_opt_set("weights_backend", "packed");   /* 0 = accepted, -1 unknown name, -2 bad value */

int v;
llm_ai_mlx_opt_get_int("prefix_min_match", &v);
```

Readers exist for int, float, bool and string values, and `llm_ai_mlx_opt_is_set()` tells whether the
user changed a default. The command-line tool FastLLM lists every option with `fastllm options` and
sets them with `--opt name=value`.

Files: `llm_engine_opts.h` (interface), `llm_engine_opts.c` (the registry and the list of options).
