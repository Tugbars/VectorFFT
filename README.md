# HighSpeedFFT

A high-performance mixed-radix FFT library in C with full support for complex and real transforms, AVX2/FMA vectorization, and Bluestein for arbitrary lengths. Designed as a drop-in alternative to FFTW with minimal dependencies.

---

## Features

* **Mixed-Radix DIT** for lengths factorable into small primes (2, 3, 4, 5, 7, 8, 11, 13, etc.)
* **Bluestein’s Algorithm** for arbitrary (prime) lengths via chirp z-transform
* **AVX2 SIMD acceleration**:

  * Flattened real/imag arrays for contiguous loads/stores
  * Unaligned `_mm256_loadu_pd` / `_mm256_storeu_pd` for robustness
  * Vectorized butterfly loops processing four complex samples at a time
* **Precomputed tables**:

  * Twiddle lookup tables for common radices (2, 3, 4, 5, 7, 8, 11, 13)
  * Bluestein chirp sequences up to N = 64
* **Real ↔ Complex FFTs** optimized via half-complex storage
* **Minimal memory footprint**: stack & heap buffers, no large lookup tables (>1 KiB)
* **CMake** build with optional GoogleTest harness
* **Comprehensive tests**: unit tests (GoogleTest) or standalone `main.c` for correctness and MSE checks

---

## Usage

```c
#include "highspeedFFT.h"

// Complex FFT
fft_data *in = malloc(N*sizeof*in), *out = malloc(N*sizeof*out);
fft_object fwd = fft_init(N, +1);
fft_exec(fwd, in, out);
fft_object inv = fft_init(N, -1);
fft_exec(inv, out, in);  // normalize by 1/N in your code

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
cmake ..                    # configure (optionally: -DENABLE_GTEST=ON)
make                        # build static lib and test executable
./test_mixedRadixFFT        # runs main.c harness
# or, with GoogleTest
cmake -DENABLE_GTEST=ON ..
make test                   # runs gtests
```

### CMake Options

* `ENABLE_GTEST` (OFF/ON): build GoogleTest unit tests vs. standalone `main.c`
* SIMD flags auto-detected (GCC/Clang: `-O3 -mavx -mfma`; MSVC: `/O2 /arch:AVX2`)

---

## Comparison to FFTW

| Criterion           | HighSpeedFFT                                 | FFTW                           |
| ------------------- | -------------------------------------------- | ------------------------------ |
| Algorithm support   | Mixed-radix + Bluestein                      | Cooley–Tuk + Rader + Bluestein |
| Vectorization       | Hand‐tuned AVX2/FMA                          | Auto‐SIMD (SSE/AVX)            |
| Dependencies        | `libm` only                                  | `pthread`, `libsimd`...        |
| Memory layout       | contiguous buffers                           | plan-specific allocators       |
| Performance (bench) | ≈FFTW on power‑of‑2, slower on prime lengths | Highly tuned wide radix        |
| Footprint           | ≈25 KiB code                                 | ≈300 KiB code/data             |

> **Benchmarks**: on Intel Xeon Gold 6226R, N=2<sup>12</sup>–2<sup>20</sup>, single thread:
>
> * **MixedRadix** vs. FFTW: within 5–10% for radix‐friendly N
> * **Bluestein** vs. FFTW: \~30% slower for large primes due to extra convolution

---

## Test Harness (main.c)

* Generates sinusoidal signals
* Runs forward/backward FFT or real FFT roundtrip
* Computes MSE & checks against `1e-10` tolerance
* Tests mixed‑radix and Bluestein cases: N={4,8,15,20,64}
* Real FFTs for N={4,8,16,32,64}
* Error‐handling tests: invalid lengths/directions

---


