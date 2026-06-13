# N=128 r2c monolithic vs 3-pass head-to-head bench

See `docs/51_mono_vs_3pass.md` for context and analysis.

## How to run

```bash
# From repo root:
GEN=./_build/default/bin/gen_radix.exe
$GEN 128 --r2c --emit-c > /tmp/r128_r2c.c
$GEN 64 --emit-c          > /tmp/r64_c2c.c

gcc-11 -O3 -mavx512f -mavx512dq -mfma -march=icelake-server \
    -flive-range-shrinkage \
    /tmp/r128_r2c.c /tmp/r64_c2c.c bench/r2c_mono/bench_r128.c \
    -o /tmp/bench_r128 -lm

/tmp/bench_r128
```

For repeatable numbers on ICX, pin frequency and core:

```bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
taskset -c 0 /tmp/bench_r128
```

## What it measures

Path A: monolithic R=128 r2c codelet (1 function call, fused).
Path B: synthetic 3-pass mirror of r2c.h structure (pack + R=64 c2c
        codelet + vectorized butterfly).

Both use the same compiler, flags, codelet generator. The
architectural delta is fusion at math-layer time vs three discrete
passes over memory.

## Expected result

Mono wins 1.3-3.3× across K = 8..1024 (validated in container
hardware; ICX numbers should be comparable or better due to its 2×
L1D bandwidth favoring the monolithic's denser memory access).
