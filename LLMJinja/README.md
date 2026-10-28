# LLMJinja

A small Jinja template engine written in C, used to render **chat templates**. Chat models ship a
template that turns a conversation (system message, user turns, assistant turns) into the exact prompt
text the model was trained on. LLMJinja renders those templates.

It implements the part of Jinja2 that chat templates use: variables, loops, conditions, filters, macros
and a few built-in functions. Anything it does not support makes the render fail with a clear error
instead of producing an approximate result.

## Using it

```c
#include "llmjinja.h"
/* build variables with llmjinja_obj / llmjinja_arr / llmjinja_str ..., then: */
char* prompt = llmjinja_render(template_text, vars, err, sizeof err);
/* free(prompt) when done; NULL means an error, described in err */
```

Context functions such as `strftime_now` can be supplied as values.

## Files

- `llmjinja.h`: public interface.
- `llmjinja.c`, `llmjinja_builtins.c`, `llmjinja_internal.h`: the engine.

## Building

```sh
cmake -S LLMJinja -B build/jinja
cmake --build build/jinja
```

The result is a static library, `llmjinja`.
