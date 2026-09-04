# `core/` — the canonical FFT tree

This is the production core of the dag-fft-compiler library: the **public front
door** (`vfft.c`, implementing `include/vfft.h`) plus the engines, planners,
and transform layers it dispatches into. Organized into subfolders by role,
layered by dependency — each layer depends only on the ones above it.

```
core/
  vfft.c        THE public API implementation (see "Front door" below)
  support/      platform foundation
  engine/       the in-place c2c kernel
  planning/     plan SEARCH + wisdom (everything measured, nothing estimated)
  transforms/   everything built ON the engine
  primes/       Rader + Bluestein for prime N
  oop/          out-of-place c2c engines (incl. the K=1 z-cascades)
```

## Front door: `vfft.c` (public API = `include/vfft.h`)

One create/execute/destroy surface over **four axes** committed at create time:
`transform` (C2C / R2C / C2R / DCT-I..IV / DST-I..III / DHT) ×
`placement` (in-place / out-of-place) ×
`layout` (SPLIT re/im planes / INTERLEAVED z, MKL DFTI_COMPLEX_STORAGE analog) ×
`order` (DEFAULT / NATURAL / SCRAMBLED, 1D/2D C2C). Plus `dims` 1..4,
`howmany` (K lanes), the opt-in padded-batch handle (`vfft_alloc_batch_for` →
`vfft_batch_planes` → `vfft_batch_stride` → `vfft_free_batch`), and rigor.

Contract: every unsupported or invalid cell is **refused at create with an
actionable stderr message**; `vfft_execute` validates the pointer signature
against the committed layout/placement/direction and computes NOTHING on a
mismatch. Never a silent no-op, never silent garbage. The support matrix lives
in `include/vfft.h` (the capabilities table + the SIGNATURE TABLE /
SUPPORT MATRIX blocks above `vfft_execute`); the machine proof is the gate
battery (`api_matrix_gate` (the serve/refuse table, benches/api_matrix_gate.c)).

Wisdom: per-feature files auto-loaded as a bundle; canonical home is
`src/dag-fft-compiler/generator/generated/` (copies elsewhere are operational
leftovers). `VFFT_WISDOM_DIR` points the bundle at another directory (gates and
benches use a scratch dir); misses calibrate at `config.rigor` and persist.

Runtime knobs (diagnostics/kill switches): `VFFT_NO_ZTURN` (fall back to the
legacy zsplit cascade), `VFFT_FORCE_ZROUTE` (pin the K=1 cascade route),
`VFFT_NO_IL2P` (disable the pure-IL 2-pass route), `VFFT_IL_PAD` (force the IL
padded arm), `VFFT_ZRACE_VERBOSE` (create-time race logging).

## Subfolders

### `support/` — platform foundation
`env.h` (timing — QPC on Windows, CLOCK_MONOTONIC elsewhere — aligned alloc,
env knobs) · `threads.h` (worker pool + pinning; caller owns core 0) ·
`strided_codelets.h` (externs for the generated SIMD codelets).

### `engine/` — the in-place c2c kernel
`plan.h` (plan/stage types) · `planner.h` (`vfft_proto_auto_plan`: plan build
from wisdom/search) · `executor.h` / `executor_generic.h` (the stage walkers) ·
`twiddle.h` (the three measured
twiddle methods: FLAT / T1S / LOG3, mixed per stage by wisdom) · `compat.h` /
`proto_stride_compat.h` (bridges between the proto and stride plan worlds) ·
`il_execute.h` (interleaved z↔z boundary folds over a `stride_plan_t` — lives
here, not in `oop/`, because it is typed on the ENGINE plan; its derived-IL
codelet population was deleted 2026-07-24, so every resolver returns 0 and every
wrapper returns −1 by design, and the fold machinery is kept as the wiring point
for a future IL-native family).

### `planning/` — plan search + wisdom (all MEASURED)
`dp_planner.h` (split-plan DP;
"DP prunes the search; it never composes costs") · `dp_planner_il.h` (IL
whole-chain DP + the cascade engine/route axis) · `exhaustive_plan.h` (the EXHAUSTIVE tier) ·
`measure.h` (paced measurement harness) · `wisdom_reader.h` (spike v6
format) · `adopt_wisdom.h` (`VFFT_ADOPT_WISDOM_DIR` import).

### `transforms/` — built on the engine
- `real/` — `r2c.h`/`r2c_dispatch.h`, `c2r.h`/`c2r_dispatch.h` (NATURAL vs
  STRIDE per-cell), `rfft.h` + `rfft_calibrate.h`/`rfft_trace.h`.
- `trig/` — `dct.h` (II/III), `dct1.h`, `dct4.h`, `dst.h`, `dht.h`,
  `dct2/3_n8_avx2.h` codelets, `trig_codelets.h` externs. Three-phase MT.
- `fft2d/` — 2D c2c/r2c/c2r: `fft2d.h`, `fft2d_r2c.h`, per-feature planners +
  wisdom, `transpose.h`, `strided_tw.h`.
- `fft3d/` — 3D c2c: `fft3d.h`, `fft3d_wisdom.h` ((N1,N2,N3) table),
  `strided_rows.h`.
- `fftnd/` — rank-general ND engine (`fndr`, rank ≤ 4; §6a47 3D real, §6a62
  rank-4 exposure): `fftnd.h`, `fftnd_r2c.h`, `fftnd_planner.h`,
  `fftnd_wisdom.h`, `fftnd_natorder.h`, `conv.h`.
- `natorder/` — the VFFT_ORDER_NATURAL machinery (per-cell measured verdict):
  `natorder_perm.h` (cycle/pair tapes), `natorder_exec.h`,
  `natorder_scatter.h`, `natorder_calibrate.h` (expensive — probe few cells),
  `natorder_2d.h` (per-axis reorder tapes).
- `conv/` — convolution + `il_layout.h` interleave/deinterleave helpers.

### `primes/` — prime-N machinery
`prime_dispatch.h` (factorable → CT/wisdom; prime → override) · `rader.h` ·
`bluestein.h` + `bluestein_calibrator.h` ((M,B) calibrate-on-miss) +
`bluestein_wisdom.h`. **Wired into the IN-PLACE c2c path only** — out-of-place
C2C refuses prime N loudly (OOP prime wiring is a planned feature).

### `oop/` — out-of-place c2c engines
`oop_plan.h` (kinds: MODEB scrambled / LEAF / BAILEY2 natural) · `oop_auto.h`
(champion build) · `oop_dp.h` (KIND×FACT joint search) · `oop_execute.h` ·
`oop_codelets.h` / `oop_leaf_registry.h` · `oop_wisdom.h` (kind-tagged cells;
kind-4 = K=1 cascade route lines `N 1 4 t2q cc_chain ns [zs_route zt_t2q]`) ·
**K=1 ≥2048 cascades**: `zturn.h` (ZTURN-S, the production engine — corner-turn
fused into ingest stores, MKL's sectioned geometry; beats MKL at 2048/16384) ·
`zsplit.h` (legacy block-split cascade, `VFFT_NO_ZTURN` fallback + offline
reference) · `zturn_proto.h` (memcmp-exact derivation prototype, permanent
reference) · **K=1 NATURAL pure IL**: `il2p.h` (il2p 2-pass pair route, BOTH
directions since 2026-07-29, plus the il3p 3-stage chain that gives odd·2^k N a
native route) · `il_prime.h` (prime N via Rader/Bluestein over il2p/il3p inners).
(`il_execute.h` moved to `engine/` — it is typed on `stride_plan_t`, not on any
OOP plan.)

Several subfolders carry their own README.md with deeper notes
(`engine/`, `oop/`, `planning/`, `primes/`, `support/`, `transforms/real/`,
`transforms/fft2d/`).

## Include convention — BARE includes, the build provides `-I`

Headers cross-reference each other **bare**: `#include "executor.h"`, not
`#include "engine/executor.h"`. The build system puts **every** `core/`
subfolder on the `-I` search path (`build_tuned/build.py:build_includes()`
walks `core/` recursively), so a bare include resolves regardless of which
subfolder the target lives in. Consequences:

- **Moving a file between subfolders needs no `#include` edits.**
- **Header basenames must stay globally unique** across all of `core/` —
  otherwise a bare include is ambiguous (first `-I` wins).
- Consumers (benches, the public build) also use bare includes:
  `#include "vfft.h"`, `#include "r2c.h"`.

SIMD codelets are **not** here — they live under `dag-fft-compiler/codelets/`
(generated by the OCaml emitters in `dag-fft-compiler/generator/`) and compile
as linked `.c` files; they include no core headers.

## Key entry points

- **Public API** (use this unless working on internals): `vfft_create` /
  `vfft_execute` / `vfft_destroy` in `vfft.c` — everything below is reached
  through it, chosen by wisdom.
- **c2c in-place**: `engine/planner.h` (`vfft_proto_auto_plan`) →
  `engine/executor.h`. MT via the `support/threads.h` pool (K-split).
- **c2c out-of-place**: `oop/oop_auto.h` champions; K=1 ≥2048 → `oop/zturn.h`.
- **r2c/c2r**: `transforms/real/r2c_dispatch.h` / `c2r_dispatch.h`.
- **trig/DSP**: `transforms/trig/{dct,dct1,dct4,dst,dht}.h`.
- **2D/3D/4D**: `transforms/fft2d/`, `transforms/fft3d/`, `transforms/fftnd/`.
- **prime N**: `primes/prime_dispatch.h` → Rader / Bluestein (in-place only).
- **natural order**: `transforms/natorder/` (1D), `natorder_2d.h` (2D).

## Gates

API surface: `api_matrix_gate` (the serve/refuse table, benches/api_matrix_gate.c)
(session scratchpad; walk the full support matrix + misuse diagnostics + the
header's compiled QUICK START). Feature gates live in `build_tuned/benches/`
(`zsplit_wis_gate`, `zsplit_api_gate`, `gate_vfft_rz`, `gate_4d`,
`gate_fndr_q1`, natorder/natmt tests, `regression_vs_mkl`). Run them with
`VFFT_WISDOM_DIR` pointed at a scratch dir so banked wisdom stays untouched.

## Migration headers at the top level

Three files sit directly in `core/` rather than in a module, because each is
about the library as a whole rather than about one transform family.

| file | role |
|---|---|
| `vfft_internal.h` | the three private structs — `vfft_plan_s`, `vfft_wisdom_s`, `vfft_batch_s`. Lifting these out of `vfft.c` is what let every later module header exist |
| `vfft_execute.h` | **THE execute entry point — every transform, BOTH layouts**, plus the execute-side helpers and `vfft_destroy`. 🔴 `vfft_execute` has EXTERNAL linkage, so the body is guarded by `VFFT_EXECUTE_IMPL` and exactly one TU defines it |
| `vfft_batch.h` | the owned-batch allocator behind `config.owned_buffers` / `config.batch`. Three descriptor shapes (c2c in-place, real, OOP 4-plane); a mismatched handle is refused, never reinterpreted |
