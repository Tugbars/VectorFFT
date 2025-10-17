# VectorFFT

### VectorFFT is a vectorized, pure C FFT library optimized for x86 processors (AVX-512, AVX2, SSE2) with zero external dependencies. It implements mixed-radix algorithms for common sizes and Bluestein's method for arbitrary lengths, with OpenMP multi-threading for large transforms.
### Designed for both digital signal processing and financial applications requiring high-performance spectral analysis. VectorFFT combines the speed of hand-optimized SIMD code with the simplicity of standalone C, making it ideal for embedded systems, real-time applications, and projects where FFTW/MKL are overkill.
---

## Features

* **Mixed-Radix DIT** for lengths factorable into small primes (2, 3, 4, 5, 7, 8, 11, 13, …)

  * *Radix-2, 3, 4, 5, 7, 8, 11, 13, 16, 32 now with fully tested AVX2 + SSE2 + scalar codepaths

    * Vector core processes 4 points per iteration
    * SSE2 tails handle 1–2 leftover samples
    * Scalar tail covers 0..3 leftovers or non-SIMD builds
    * Explicit prefetching of outputs and twiddles for cache-friendliness
    * FMADD/FMSUB macros used to leverage FMA when available
    * Error handling for insufficient scratch or missing precomputed twiddles

* **Bluestein’s Algorithm** for arbitrary (prime) lengths via chirp z-transform

* **AVX2 + FMA SIMD acceleration**

  * Common butterfly math factored into helpers (e.g. `cmul_soa_avx`, `rot90_soa_avx`)
  * Works with unaligned data (`_mm256_loadu_pd`, `_mm256_storeu_pd`)
  * Prefetching added to hide latency in deeper radices

* **SSE2 acceleration**

  * Two-lane fallback for tails in radices 3, 5, 11
  * Safe unaligned loads/stores (`_mm_loadu_pd` / `_mm_storeu_pd`)

* **Robust scratch-buffer model**

  * No per-call `malloc` — all recursion stages and local twiddles use a preallocated scratch buffer
  * Layout partitioned: sub-FFT outputs + (optional) stage twiddles
  * Prints an error and aborts stage if alignment or sizing is invalid

* **Precomputed or on-the-fly twiddles**

  * Radices 2, 3, 4, 5, 7, 8, 11, 13 supported
  * If no precompute is provided, stage-local twiddles are generated in the scratch tail

* **Real ↔ Complex FFTs** via half-complex storage

* **Minimal footprint** (\~25 KiB code, no giant tables)

* **CMake** build system with optional GoogleTest harness

* **Comprehensive test suite**

  * Round-trip forward/backward FFTs
  * Radix-3 and 5 vector/scalar tails
  * Radix-11 special-case coverage (k=0,1 SSE2 lanes)
  * Error conditions (invalid N, missing scratch, bad alignment)
  * MSE checks against reference FFTW within 1e-10

---

## Comparison to FFTW

| Criterion           | HighSpeedFFT                            | FFTW                           |
| ------------------- | --------------------------------------- | ------------------------------ |
| Algorithm support   | Mixed-radix + Bluestein                 | Cooley–Tuk + Rader + Bluestein |
| Vectorization       | Hand-tuned AVX2/FMA + SSE2              | Auto-SIMD (SSE/AVX)            |
| Dependencies        | `libm` only                             | `pthread`, `libsimd`, …        |
| Memory layout       | contiguous real/imag buffers            | plan-specific allocators       |
| Performance (bench) | ≈FFTW on radix-friendly N, within 5–10% | highly tuned wide radix        |
| Footprint           | ≈25 KiB code                            | ≈300 KiB code + data           |

> **Benchmarks** on Intel Xeon Gold 6226R (single thread):
>
> * Mixed-radix (2^12–2^20): within 5–10% of FFTW
> * Bluestein on large primes: \~30% slower (extra convolution)

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
cmake ..                    # optionally: -DENABLE_GTEST=ON
make                        # builds static lib + test harness
./test_mixedRadixFFT        # standalone tests
# or, with GoogleTest:
cmake -DENABLE_GTEST=ON ..
make test
```

### CMake Options

* `ENABLE_GTEST` (OFF/ON): GoogleTest vs. standalone harness
* SIMD flags auto-detected:

  * GCC/Clang: `-O3 -mavx2 -mfma -msse2`
  * MSVC: `/O2 /arch:AVX2 /fp:fast`

---
