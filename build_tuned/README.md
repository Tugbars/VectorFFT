# build_tuned/ — calibrated benchmarks and tooling

This directory contains the **dev/calibration toolchain** for VectorFFT:
the calibrator that produces `vfft_wisdom_tuned.txt`, the build harness
(`build.py`) that wraps icx/cl/gcc, and the bench programs that produce
the headline numbers in [`docs/performance/v1_0_results.md`](../docs/performance/v1_0_results.md).

## Layout

```
build_tuned/
├── build.py                  — bench/test build harness (icx/cl/gcc)
├── calibrate.py              — wisdom calibrator entry point
├── make_perf_txt_fftw.py     — CSV → human-readable .txt formatter
├── vfft_wisdom_tuned.txt     — calibrated wisdom for the i9-14900KF
├── fftw3.dll                 — runtime DLL (vcpkg FFTW3 build)
│
├── benches/                  — USER-FACING competitive benches
│   ├── bench_1d_vs_mkl.c          1D C2C  vs MKL    (the headline 207 cells)
│   ├── bench_1d_vs_fftw.c         1D C2C  vs FFTW3  (the 207-cell vs-FFTW sweep)
│   ├── bench_dct2_vs_fftw.c       DCT-II  vs FFTW3
│   ├── bench_dct3_vs_fftw.c       DCT-III vs FFTW3
│   ├── bench_dct4_vs_fftw.c       DCT-IV  vs FFTW3
│   ├── bench_dct4_vs_mkl.c        DCT-IV  vs MKL TT
│   ├── bench_dst23_vs_fftw.c      DST-II/III vs FFTW3
│   ├── bench_dst23_vs_mkl.c       DST-II/III vs MKL TT
│   ├── bench_fft2d_r2c_vs_fftw.c  2D R2C  vs FFTW3
│   └── bench_mt_dct.c             MT scaling (T=1/2/4/8) for DCT/DST/DHT
│
├── dev/                      — internal experiments and diagnostics
│   ├── bench_all_est_vs_wis.c     ESTIMATE vs MEASURE across all transforms
│   ├── bench_gap_check.c          per-transform flag-honoring diagnostic
│   ├── bench_estimate_vs_wisdom.c estimate-mode quality measurement
│   ├── bench_mt_compare.c         direct-MT (C2C) vs wrapper-MT (DCT) scaling
│   ├── bench_mt_overrides.c       MT plumbing experiment
│   ├── bench_2d_mt.c              2D MT scaling experiment
│   ├── bench_r2c.c                R2C standalone bench
│   ├── measure_smoke_test.c       CPE-table sanity check
│   └── calibrate_tuned.c          standalone calibrator (companion to calibrate.py)
│
├── results/                  — generated bench output and plots
│   ├── vfft_perf_tuned_1d.{csv,txt}        VectorFFT vs MKL (207 cells)
│   ├── vfft_perf_tuned_1d_fftw.{csv,txt}   VectorFFT vs FFTW3 (207 cells)
│   ├── vfft_acc_tuned_1d.csv               roundtrip accuracy
│   ├── *.png                                throughput / speedup / scatter / precision
│   └── plot_vfft.ipynb                      plot generator notebook
│
├── bench_compilers/          — three-compiler comparison harness (MSVC/ICX/GCC)
├── test/                     — internal regression / correctness tests
└── backup/                   — old wisdom snapshots (pre-calibration baselines)
```

## Common workflows

### Reproduce the 1D-vs-MKL headline

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_1d_vs_mkl.c --mkl
build_tuned/benches/bench_1d_vs_mkl.exe \
    build_tuned/vfft_wisdom_tuned.txt \
    build_tuned/results/vfft_perf_tuned_1d.csv \
    build_tuned/results/vfft_acc_tuned_1d.csv
```

### Reproduce the 1D-vs-FFTW3 headline

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_1d_vs_fftw.c --fftw
build_tuned/benches/bench_1d_vs_fftw.exe \
    build_tuned/vfft_wisdom_tuned.txt \
    build_tuned/results/vfft_perf_tuned_1d_fftw.csv \
    build_tuned/results/vfft_acc_tuned_1d_fftw.csv
python build_tuned/make_perf_txt_fftw.py \
    build_tuned/results/vfft_perf_tuned_1d_fftw.csv \
    build_tuned/results/vfft_perf_tuned_1d_fftw.txt
```

### Reproduce a single r2r-vs-FFTW3 cell sweep

```
python build_tuned/build.py --vfft --src build_tuned/benches/bench_dct2_vs_fftw.c --fftw
build_tuned/benches/bench_dct2_vs_fftw.exe
```

(Same pattern for `bench_dct3`, `bench_dct4`, `bench_dst23`, `bench_fft2d_r2c`.)

### Recalibrate wisdom on this host

```
python build_tuned/calibrate.py
```

Writes `build_tuned/vfft_wisdom_tuned.txt`. Slow (minutes to hours);
needs a clean P-core, performance power plan, no concurrent load.

## Notes

- Build artifacts (`.exe`, `.obj`, `.pdb`, `.log`) are not tracked.
  Re-running `build.py --src ...` regenerates them.
- `fftw3.dll` is co-located in `benches/` and `dev/` so direct exe
  invocation Just Works. Running through `build.py` adds the vcpkg
  FFTW3 bin dir to `PATH` automatically.
- The `dev/` benches are scratch — outputs aren't preserved, paths
  are hardcoded, error reporting is terse. Use them for debugging,
  not for headline numbers.
- For three-compiler comparison (MSVC vs ICX vs GCC), see
  [`bench_compilers/run.bat`](bench_compilers/run.bat).
