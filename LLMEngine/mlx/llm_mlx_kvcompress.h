// llm_mlx_kvcompress.h — CateKV/Minima-KV T3.2 (F0 spike)
#pragma once
#include <stdbool.h>
#include <stddef.h>

// F0 spike: verifica supporto dtype fp8/float8 in mlx locale
bool llm_mlx_kvcompress_has_fp8(void);
bool llm_mlx_kvcompress_has_tq3(void);

// F1/F2 flag (default OFF, env LLM_KVFP8 / LLM_KVTQ3)
bool llm_mlx_kvcompress_fp8_enabled(void);
bool llm_mlx_kvcompress_tq3_enabled(void);
bool llm_mlx_kvcompress_verify_enabled(void);
