# Bench Harnesses

This directory contains benchmark drivers for comparing generated codelets against hand-coded references and across variants.

## Structure

- `bench_*.c` — bench drivers (compile and run individually)
- `references/` — hand-coded reference codelets (`radix4/8/16/32/64_handcoded.h`), referenced by `#include "../references/..."`
- `generated/` — generated codelets (output of `gen_radix.exe`), linked into bench binaries

## Building a bench

Each bench is a standalone executable. Example for the headline R=32 SU+Spill comparison:

```bash
# Generate codelets (run from project root)
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place 32 \
    > bench/generated/radix32_gen_inplace.c
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place --spill 32 \
    > bench/generated/radix32_spill_inplace.c
dune exec bin/gen_radix.exe -- --twiddled --emit-c --in-place --spill --su 32 \
    > bench/generated/radix32_spill_su_inplace.c

# Compile codelets
cd bench
gcc -O3 -march=native -mavx512f -mfma -funroll-loops -c generated/radix32_gen_inplace.c -o radix32_gen_inplace.o
gcc -O3 -march=native -mavx512f -mfma -funroll-loops -c generated/radix32_spill_inplace.c -o radix32_spill_inplace.o
gcc -O3 -march=native -mavx512f -mfma -funroll-loops -c generated/radix32_spill_su_inplace.c -o radix32_spill_su_inplace.o

# Build bench binary
gcc -O3 -march=native -mavx512f -mfma -funroll-loops \
    bench_r32_spill_su.c \
    radix32_gen_inplace.o radix32_spill_inplace.o radix32_spill_su_inplace.o \
    radix32_spill_su_fuse2_inplace.o radix32_spill_su_fuse8_inplace.o \
    -o bench_r32_spill_su -lm

# Run with K (batch size)
./bench_r32_spill_su 1024
```

Note: paths in `#include "../radix32_handcoded.h"` are relative to the repo root historically; edit to `#include "references/radix32_handcoded.h"` if running from this `bench/` directory.

## Naming conventions

| Pattern | Meaning |
|---------|---------|
| `radix{N}_gen_inplace.c` | Topo, AVX-512, in-place (default) |
| `radix{N}_spill_inplace.c` | Spill (block-sequential), AVX-512, in-place |
| `radix{N}_su_inplace.c` | SU (without spill), AVX-512, in-place |
| `radix{N}_spill_su_inplace.c` | Full recipe (spill + SU + cluster-sequential), AVX-512, in-place |
| `radix{N}_avx2_topo.c` | Topo, AVX2, in-place |
| `radix{N}_avx2_su_spill.c` | Full recipe, AVX2, in-place |
| `radix{N}_oop_topo.c` | Topo, AVX-512, OUT-OF-PLACE (only used for R=64 hand comparison) |
| `radix{N}_oop_su.c` | Full recipe, AVX-512, OUT-OF-PLACE |

## Key benches

- `bench_r{N}.c` — basic 2-way Hand vs Topo
- `bench_r{N}_su.c` — 2-way Hand vs SU (no spill)
- `bench_r{N}_su_spill.c` — Hand vs Topo vs SU+Spill (full recipe)
- `bench_r{N}_spill.c` — Hand vs Topo vs Spill alone
- `bench_r32_fuse.c` — exhaustive fuse-level sweep at R=32
- `bench_r64_three.c` — R=64 OOP three-way (Hand vs Topo vs full recipe)
- `bench_avx2_su.c` — multi-radix AVX2 sweep

## Methodology

All benches use:
- 100-iteration warmup
- 7 trials of `repeat` iterations each, taking the minimum to reduce timer noise
- `clock_gettime(CLOCK_MONOTONIC)` for nanosecond timing
- Random-input correctness check vs hand-coded (or Topo if no hand reference) before timing

Reported times are nanoseconds per call. SU/H < 1 means recipe is faster than Hand.
