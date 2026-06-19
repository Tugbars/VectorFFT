# MKLBench — dag-fft-compiler (JIT) vs Intel MKL, 1D C2C

Benchmarks the new dag-fft-compiler against Intel MKL on the calibrated wisdom
cells, driving execution through the **JIT** path (`vfft_proto_plan_jit_fwd` —
baked static executor if present, else a gcc-JIT-compiled one). Single gcc
executable: gcc exe + in-repo gcc codelets + gcc-JIT'd `.dll`s + MKL via `mkl_rt`.

## Files
- `bench_jit_vs_mkl.c` — the bench. Per wisdom cell: build the plan from its
  factors+variants, resolve via `plan_jit_fwd`, check **roundtrip** accuracy
  (fwd+bwd = input×N — the dag forward is digit-reversed vs MKL's natural order,
  so a direct fwd-vs-MKL compare is invalid), then time dag vs MKL (best-of-5,
  cache-busted between engines). Writes a CSV.
- `build.ps1` — compile + link (gcc + `mkl_rt`).
- `run.ps1` — **isolated per-cell** runner → combined `results.csv`.
- `results.csv` — output (N,K,factors,path,vfft_ns,mkl_ns,gflops,ratio,rt_err).

## Build & run
```
powershell -ExecutionPolicy Bypass -File MKLBench\build.ps1
powershell -ExecutionPolicy Bypass -File MKLBench\run.ps1
```
`run.ps1` defaults to the calibrated wisdom
(`../generator/generated/spike_wisdom.txt`) and writes `results.csv`.

## Gotchas (learned the hard way)
1. **LP64, not ILP64.** Build WITHOUT `-DMKL_ILP64`. `mkl_rt`'s DFTI defaults to
   LP64 (4-byte `MKL_LONG`); compiling ILP64 (8-byte) corrupts the strides array
   element-by-element → `DftiCommit` fails with "Inconsistent configuration
   parameters". (Scalar args survive via low bytes, so it's the arrays that break.)
2. **Runtime PATH** needs `…\mkl\latest\bin` (`mkl_rt.2.dll`) AND the MinGW bin
   (`libwinpthread-1.dll`, pulled in by `nanosleep`). Missing either → exit 53
   (loader failure, no output). `run.ps1` sets both.
3. **Run cells ISOLATED.** A single sequential run has cross-cell cache/thermal
   carryover that `cachebust()` does not fully clear — it produced bogus outliers
   (8192 showed 5.0× in-sequence but **1.0×** isolated; both engines were off).
   `run.ps1` runs one fresh process per cell. Trust the isolated numbers.
4. **Accuracy = roundtrip**, not fwd-vs-MKL (output-order mismatch). DFT
   correctness vs FFTW is validated separately in the `benchmarks/test_oop_*` suite.

## Result (K=4 pow2 8→131072, isolated, 2026-06-14)
dag (JIT) beats MKL on **all 16 cells**; roundtrip err ~1e-14. JIT exercised on the
10 cells whose calibrated factorization isn't baked. Ratios dag/MKL: 8:15.8x
16:8.3x 32:3.5x 64:2.7x 128:2.9x 256:1.8x 512:2.3x 1024:2.0x 2048:1.5x 4096:1.7x
8192:1.1x 16384:1.2x 32768:1.4x 65536:1.3x 131072:1.4x | 1024/128:2.2x.
Geomean ~1.7x (multi-stage) / ~2.4x (all). Consistent with production's
207/207-vs-MKL record. (prod=ICX, dag=gcc — toolchain differs.)
