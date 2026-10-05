"""BenchChronos CLI — wrapper di `mlx-chronos` (fork vendorizzato con i motori ds4 e fmlx).

Esempi:
  python -m benchchronos.cli engines
  python -m benchchronos.cli models --engine ds4
  python -m benchchronos.cli run --engine ds4  --model <id> --trials 5 --format all
  python -m benchchronos.cli run --engine fmlx --model <id> --trials 5 --format all
  python -m benchchronos.cli compare results/local/<a>.json results/local/<b>.json

I motori `ds4` e `fmlx` sono registrati direttamente nel fork
(`vendor/mlx-chronos/mlx_chronos/{constants,schema,engines}.py`).
"""

import sys

from mlx_chronos.cli import main

if __name__ == "__main__":
    sys.exit(main())
