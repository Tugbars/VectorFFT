# The JIT (`src/dag-fft-compiler/jit/`)

Plan-time code generation for zturn-ts, the split library's 1D C2C engine
(`src/core/split/rank1/zts.h`). One file: `zts_jit_runtime.h`. The earlier
runtimes of this folder (the stride engine's executors, the K=1 route twins,
the rfft and c2r emitters, their Python emitters and smoke tests) went with the
stride engine on 2026-11-06.

## What it does

zturn-ts runs a cell in one of two forms. The registry
(`generator/generated/zts_registry_avx2.h`) holds a **fused driver** for every
pow2 cell of ZTURN-T's list: one whole-transform function per (family, N,
chain) with the stage kernels of `codelets/split/ztt` inlined and literal trip
counts (`generator/lib/gen/zts_drivers.ml`, built at build time). Every other
cell — the odd band, the lane-major band, a pow2 chain outside the list — runs
the **staged walk**: the same kernels called one by one through the plan's
stage table.

The JIT gives a deployed plan of the second kind its fused driver at create:

1. it writes the driver the build-time emitter would have written for that
   chain, with the kernels' shipped sources `#include`d so their bodies inline
   exactly as in the baked drivers (the emission is C, in the header; no
   Python);
2. compiles it with the library's own codelet flags (`gcc -O3 -mavx2 -mfma
   -march=native -mno-avx512f`), archives it, runs the kernels' stack
   alignment on it (`src/tools/build/align_kernel_stacks.py`, the step the
   library's own archive gets), links a shared library and loads it;
3. hands the plan a registry cell: from then on the plan runs exactly as a
   baked cell, one function pointer, no execute-path cost.

The model is the one the owner set for the JITs here: the race ranks
candidates on the staged walk; **only the deployed plan is compiled**
(`split/rank1/c2c_create_split.h`, `_zts_deployed`), once per process and once
ever on disk. The cache is `jit/generated/` (git-ignored), keyed
`(family, N, chain, isa, version)`: `zts_jit_<key>.{c,o,a,dll|so,log}`. The
staged walk stays on any failure (no toolchain, a compile error, a missing
symbol): the JIT is a speed cache, never a correctness dependency. The
driver's arithmetic is the staged walk's (the in-tile test is
`tile % (R*L) == 0`), so the two forms are bitwise.

## Switch, requirements, cost

- `VFFT_USE_JIT` at compile time; `gauntlet/build.py` sets it with
  `VFFT_JIT=1`. CMake has no option for it yet.
- At create: gcc (`C:\mingw152\mingw64\bin\gcc.exe` by default; `VFFT_PROTO_JIT_GCC`),
  `ar` beside it, python (the alignment step, Windows only). All paths are
  `#ifndef`-overridable in the header (`VFFT_PROTO_JIT_REPO`, `_DIR`,
  `VFFT_ZTS_JIT_CFLAGS`, `_ALIGN`, `_VERSION`).
- A cold compile is 0.75 s (a small cell) to 1.4 s (a radix-15 cell), once
  ever per cell; a fresh process loads the library in well under a
  millisecond. Logs under `VFFT_NAT_LOG`.

## Measured (2026-11-06, K = 4, one thread)

`docs/research/split_zts_jit_2026_11_06.md`. The JIT'd driver beats the staged
walk where the work is cache-resident — 2 to 13 % on the odd cells at
48..480 contiguous and 45..240 lane-major, 1.33x at 1575 lane-major — and is
at parity past that (the memory floor). The DLL'd chain is within 5 % of the
baked static on most pow2 cells: the JIT path itself costs nothing.

## Bump `VFFT_ZTS_JIT_VERSION`

after any change to the emission, to the stage kernels, or to the codelet
flags: the on-disk cache is keyed by it.
