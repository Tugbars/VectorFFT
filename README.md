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

## Architecture
┌─────────────────────────────────────────────────────┐
│  FFT_INIT (Planning)                                │
│                                                     │
│  1. Factorize N into radices                       │
│  2. For each stage:                                │
│     ┌──────────────────────────────────────┐      │
│     │ Twiddle Manager                       │      │
│     │ - Computes stage_tw[r,k] = W^(r*k)   │      │
│     │ - Stores in plan.stages[i].tw        │      │
│     └──────────────────────────────────────┘      │
│     ┌──────────────────────────────────────┐      │
│     │ Rader Manager (if prime radix)        │      │
│     │ - Computes conv_tw[q] = exp(...)     │      │
│     │ - Stores in rader_plan_fwd/inv        │      │
│     └──────────────────────────────────────┘      │
│  3. Store plan.direction = FORWARD/INVERSE         │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│  FFT_EXEC (Execution)                               │
│                                                     │
│  Dispatcher:                                        │
│    if (plan.direction == FORWARD):                 │
│      radix7_fv(stage.tw, stage.rader_fwd)          │
│    else:                                           │
│      radix7_bv(stage.tw, stage.rader_inv)          │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│  RADIX BUTTERFLY (Arithmetic)                       │
│                                                     │
│  void radix7_fv(out, in, stage_tw, rader_tw) {    │
│    // Just compute DFT using provided twiddles     │
│    // No knowledge of direction!                   │
│    // Direction is baked into constants            │
│  }                                                 │
└─────────────────────────────────────────────────────┘

void radix7_fv(out, in, stage_tw, rader_tw, sub_len) {
    for (k = 0; k < sub_len; k++) {
        // 1. Load inputs
        x0 = in[k + 0*sub_len];
        x1 = in[k + 1*sub_len];
        ...
        
        // 2. Apply stage twiddles (pre-computed!)
        if (sub_len > 1) {
            x1 *= stage_tw[k*6 + 0];  // W^(1*k)
            x2 *= stage_tw[k*6 + 1];  // W^(2*k)
            ...
        }
        
        // 3. Rader convolution (pre-computed twiddles!)
        tx = permute(x1, x3, x2, x6, x4, x5);
        for (q = 0; q < 6; q++) {
            conv[q] = 0;
            for (l = 0; l < 6; l++) {
                // ✅ USE PRE-COMPUTED rader_tw!
                conv[q] += tx[l] * rader_tw[(q-l) % 6];
            }
        }
        
        // 4. Output (using pre-computed convolution result)
        out[k + 0*sub_len] = x0 + sum(conv);
        out[k + 1*sub_len] = x0 + conv[0];
        ...
    }
}
```

**KEY:** The function doesn't know if it's forward or inverse!
- It just uses the twiddles it's given
- Those twiddles have the correct sign (computed by twiddle manager)

---

## 🧩 Rader Special Case

### **Two Twiddle Types in Rader:**

| Twiddle Type | Purpose | Who computes? | When used? |
|-------------|---------|---------------|-----------|
| **Stage twiddles** | Connect sub-FFTs (Cooley-Tukey) | Twiddle Manager | DIT multiply |
| **Convolution twiddles** | Rader cyclic convolution | Rader Manager | Inside Rader loop |

### **Rader Plan Per Direction:**
```
Planning creates TWO Rader plans for each prime:

rader_plan_fwd_7:
  conv_tw[6] = exp(-2πi * out_perm[q] / 7)  // Negative!

rader_plan_inv_7:
  conv_tw[6] = exp(+2πi * out_perm[q] / 7)  // Positive!

Dispatcher selects:
  if FORWARD: pass rader_plan_fwd_7.conv_tw to radix7_fv()
  if INVERSE: pass rader_plan_inv_7.conv_tw to radix7_bv()

  src/
├── planning/
│   ├── fft_planning.h          # Main planning API (fft_init umbrella)
│   ├── fft_planning.c          # Factorization + stage setup
│   ├── fft_twiddles.h          # Twiddle Manager (single source)
│   ├── fft_twiddles.c          # AVX2 twiddle computation
│   └── fft_rader_plans.h/c     # Rader Manager (single source)
│
├── execution/
│   ├── fft_executor.h/c        # Dispatcher (calls _fv/_bv)
│   └── fft_exec.c              # Main fft_exec() function
│
├── radix_forward/
│   ├── fft_radix2_fv.c
│   ├── fft_radix3_fv.c
│   └── ...
│
└── radix_inverse/
    ├── fft_radix2_bv.c
    └── ...