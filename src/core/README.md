# `core/` layout

The dag-fft-compiler core (the canonical FFT tree) is organized into subfolders
by role, layered by dependency — each layer depends only on the ones above it:

```
core/
  support/      platform foundation: env (timing/alloc), threads (pool), strided codelet externs
  engine/       the c2c kernel: executor(s), planner, twiddle, plan types, compat bridge
  planning/     plan SEARCH + cost model + wisdom + measurement (exhaustive / dp / estimate / measure)
  transforms/   everything built ON the engine:
    real/         r2c / c2r / rfft (+ dispatchers, registries)
    trig/         DCT-I/II/III/IV, DST-I/II/III, DHT (+ N=8 codelets, externs)
    fft2d/        2D c2c / 2D r2c / transpose
  primes/       Rader + Bluestein (algorithms, calibrator, wisdom, dispatch)
  oop/          out-of-place c2c engine (plan / auto / dp / wisdom / execute / registries)
```

## Include convention — BARE includes, the build provides `-I`

Headers cross-reference each other **bare**: `#include "executor.h"`, not
`#include "engine/executor.h"`. The build system puts **every** `core/`
subfolder on the `-I` search path, so a bare include resolves regardless of which
subfolder the target lives in. Consequences:

- **Moving a file between subfolders needs no `#include` edits** — only the `-I`
  list must list the subfolders, and `build_tuned/build.py:build_includes()`
  walks `core/` recursively so even that is automatic. (CMake will mirror this.)
- **Header basenames must stay globally unique** across all of `core/` (they are
  today) — otherwise a bare include is ambiguous (first `-I` wins).
- Consumers (benches, future public `.c`) also use bare includes:
  `#include "r2c.h"`, not `#include "core/r2c.h"`.

SIMD codelets are **not** here — they live under `dag-fft-compiler/codelets/`
and compile as linked `.c` files (they include no core headers).

## Key entry points

- **c2c**: `engine/planner.h` (`vfft_proto_auto_plan`) → `engine/executor.h`
  (`vfft_proto_execute_fwd/bwd`). MT via the `support/threads.h` pool (K-split).
- **r2c/c2r**: `transforms/real/r2c_dispatch.h` (`vfft_r2c_plan_create` /
  `vfft_r2c_execute_fwd`) — hybrid rfft vs decoupled-stride; auto-threads when
  `stride_set_num_threads()>1` at plan-create (picks a sub-K `block_K`).
- **trig/DSP**: `transforms/trig/{dct,dct1,dct4,dst,dht}.h` — three-phase MT
  (pre-K-split → inner FFT threads → post-K-split).
- **2D**: `transforms/fft2d/{fft2d,fft2d_r2c}.h` — tile-parallel rows.
- **OOP c2c**: `oop/oop_auto.h` / `oop/oop_wisdom.h`.
- **prime N**: `primes/prime_dispatch.h` → Rader / Bluestein.
