# HighSpeedFFT

A high‑performance mixed‑radix FFT library in C with full support for complex and real transforms, AVX2/FMA vectorization, and Bluestein’s algorithm for arbitrary lengths. Designed as a drop‑in alternative to FFTW with minimal dependencies.

---

## Features

* **Mixed‑Radix DIT** for lengths factorable into small primes (2, 3, 4, 5, 7, 8, 11, 13, etc.)

  * *Radix‑11* now with fully tested AVX2 + SSE2 + scalar codepaths

    * Two‑lane SSE2 handles k=2,3 with explicit `k0/k1` indices
    * Unrolled “flatten → combine → copy‑back” split into an inline `radix11_butterfly()` helper
    * Scratch‑buffer layout separated for twiddles vs. flattened real/imag to avoid aliasing

* **Bluestein’s Algorithm** for arbitrary (prime) lengths via chirp z‑transform

* **AVX2 + FMA SIMD acceleration**

  * Common butterfly logic factored into a vectorized helper to avoid code duplication
  * Processes 4 complex samples per iteration
  * FMA (`_mm256_fmadd_pd` / `_mm256_fmsub_pd`) for fused multiply‑add performance & precision
  * Unaligned loads/stores (`_mm256_loadu_pd` / `_mm256_storeu_pd`) for scratch‑buffer robustness

* **SSE2‑accelerated radices** (3, 4, 5, 11 tails)

* **Eliminated per-call malloc**
  All intermediate buffers (twiddles, flattened arrays, sub‑FFT outputs) now live in a preallocated scratch area; `fft_exec` no longer uses `malloc`/`free` in the hot path

  * Two‑lane `_mm_loadu_pd` / `_mm_storeu_pd`
  * Dedicated SSE2 “tail” loops for remaining k values
  * Inlined FMADD\_SSE2/FMSUB\_SSE2 when no hardware FMA

* **Precomputed tables**

  * Twiddle lookup for radices 2, 3, 4, 5, 7, 8, 11, 13
  * Bluestein chirp sequences for N ≤ 64

* **Real ↔ Complex FFTs** optimized via half‑complex storage

* **Minimal footprint** (\~25 KiB code, no tables >1 KiB)

* **CMake** build system, optional GoogleTest harness

* **Comprehensive tests**: unit tests or standalone `main.c` for correctness & MSE checks

---

---

## Comparison to FFTW

| Criterion           | HighSpeedFFT                            | FFTW                           |
| ------------------- | --------------------------------------- | ------------------------------ |
| Algorithm support   | Mixed‑radix + Bluestein                 | Cooley–Tuk + Rader + Bluestein |
| Vectorization       | Hand‑tuned AVX2/FMA + SSE2              | Auto‑SIMD (SSE/AVX)            |
| Dependencies        | `libm` only                             | `pthread`, `libsimd`, …        |
| Memory layout       | contiguous real/imag buffers            | plan‑specific allocators       |
| Performance (bench) | ≈FFTW on radix‑friendly N, within 5–10% | highly tuned wide radix        |
| Footprint           | ≈25 KiB code                            | ≈300 KiB code + data           |

> **Benchmarks** on Intel Xeon Gold 6226R (single thread):
>
> * Mixed‑radix (2^12–2^20): within 5–10% of FFTW
> * Bluestein on large primes: \~30% slower (extra convolution)

---

## Test Harness

The standalone `main.c` and the GoogleTest suite now cover:

* Forward/backward round‑trip (complex & real FFTs)
* k=0,1 special cases through SIMD tails for radix‑11
* Error handling (invalid lengths/directions)
* MSE checks against 1e‑10 tolerance

---

## Usage

```c
#include "highspeedFFT.h"

// Complex FFT
fft_data *in  = malloc(N*sizeof*in),
         *out = malloc(N*sizeof*out);
fft_object fwd = fft_init(N, +1);
fft_exec(fwd, in, out);
fft_object inv = fft_init(N, -1);
fft_exec(inv, out, in);  // normalize by 1/N yourself

// Real FFT
fft_real_object r2c = fft_real_init(N, +1);
fft_data *hc = malloc((N/2+1)*sizeof*hc);
fft_r2c_exec(r2c, real_in, hc);
fft_real_object c2r = fft_real_init(N, -1);
fft_c2r_exec(c2r, hc, real_out);  // normalize by 1/N
```

---

## Build

```bash
mkdir build && cd build
cmake ..                    # (optionally: -DENABLE_GTEST=ON)
make                        # builds static lib + test harness
./test_mixedRadixFFT        # standalone main.c tests
# or, with GoogleTest:
cmake -DENABLE_GTEST=ON ..
make test
```

### CMake Options

* `ENABLE_GTEST` (OFF/ON): GoogleTest vs. standalone `main.c`
* SIMD flags auto‑detected:

  * GCC/Clang: `-O3 -mavx2 -mfma -msse2` (SSE2, AVX2, FMA enabled automatically)
  * MSVC: `/O2 /arch:AVX2 /fp:fast` (SSE2/AVX2/FMA enabled)

---
