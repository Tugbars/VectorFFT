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
fft_init(N, direction)
    │
    ├─ Validate N > 0, direction valid
    │
    ├─ Allocate fft_plan struct (calloc)
    │   ├─ plan->n_input = N
    │   ├─ plan->n_fft = N
    │   ├─ plan->direction = direction
    │   └─ plan->use_bluestein = 0
    │
    ├─ factorize(N, plan->factors) → num_stages
    │     │
    │     ├─ Try radices in priority order:
    │     │   [32, 16, 13, 11, 9, 8, 7, 5, 4, 3, 2,
    │     │    17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
    │     │
    │     ├─ While (n > 1):
    │     │   ├─ Find largest radix r that divides n
    │     │   ├─ factors[num_factors++] = r
    │     │   ├─ n /= r
    │     │   └─ IF (num_factors >= MAX_FFT_STAGES): ERROR
    │     │
    │     └─ Return: num_stages (success) or -1 (unfactorizable)
    │
    │
    ├─ ══════════════════════════════════════════════════════════
    │  BRANCH 1: BLUESTEIN PATH (unfactorizable N)
    │  ══════════════════════════════════════════════════════════
    │
    ├─ IF num_stages < 0:
    │   │
    │   └─ DISPATCH → plan_bluestein(plan, N, direction)
    │        │
    │        ├─ M = next_pow2(2*N - 1)  // Pad to power-of-2
    │        │   └─> Example: N=509 → M=1024
    │        │
    │        ├─ plan->use_bluestein = 1
    │        ├─ plan->n_input = N
    │        ├─ plan->n_fft = M
    │        │
    │        ├─ Allocate chirp twiddles: bluestein_tw[M]
    │        │   │
    │        │   ├─ sign = (direction == FFT_FORWARD) ? -1.0 : +1.0
    │        │   │
    │        │   └─ FOR k = 0..N-1:
    │        │       ├─ angle = sign * π * k² / N
    │        │       └─ bluestein_tw[k] = exp(i * angle)
    │        │           └─> Store as {cos(angle), sin(angle)}
    │        │
    │        │   └─ Zero-pad: bluestein_tw[N..M-1] = {0, 0}
    │        │
    │        ├─ ✅ Create SEPARATE forward/inverse plans for M:
    │        │   │
    │        │   ├─ IF direction == FFT_FORWARD:
    │        │   │   ├─ plan->bluestein_fwd = fft_init(M, FFT_FORWARD)  ← RECURSIVE
    │        │   │   └─ plan->bluestein_inv = fft_init(M, FFT_INVERSE)  ← RECURSIVE
    │        │   │
    │        │   └─ (Same for FFT_INVERSE direction)
    │        │       └─> ✅ Separate opaque types enforce type safety
    │        │
    │        └─ Return 0 (success) or -1 (error)
    │
    │   └─> RETURN plan (Bluestein planning complete)
    │
    │
    ├─ ══════════════════════════════════════════════════════════
    │  BRANCH 2: COOLEY-TUKEY PATH (factorizable N)
    │  ══════════════════════════════════════════════════════════
    │
    └─ ELSE (Cooley-Tukey path):
         │
         ├─ plan->num_stages = num_stages
         │
         ├─ Log factorization:
         │   └─> "Factorization: 13×11×7" (example)
         │
         │
         ├─ ════════════════════════════════════════════════════
         │  STAGE CONSTRUCTION LOOP
         │  ════════════════════════════════════════════════════
         │
         ├─ N_stage = N  // Initial transform size
         │
         └─ FOR stage i = 0 to num_stages-1:
              │
              ├─ radix = plan->factors[i]
              ├─ sub_len = N_stage / radix
              │
              ├─ stage_descriptor *stage = &plan->stages[i]
              ├─ stage->radix = radix
              ├─ stage->N_stage = N_stage
              ├─ stage->sub_len = sub_len
              │
              │
              ├─ ════════════════════════════════════════════════
              │  TWIDDLE MANAGER: Cooley-Tukey Stage Twiddles
              │  ════════════════════════════════════════════════
              │
              ├─ DISPATCH → compute_stage_twiddles(N_stage, radix, direction)
              │    │
              │    ├─ Validate: radix >= 2, N_stage >= radix
              │    │
              │    ├─ num_twiddles = (radix - 1) × sub_len
              │    │   └─> Layout: tw[k*(radix-1) + (r-1)] = W_N^(r*k)
              │    │       where W_N = exp(sign * 2πi/N_stage)
              │    │
              │    ├─ Allocate: tw = aligned_alloc(32, num_twiddles × sizeof(fft_data))
              │    │   └─> 32-byte aligned for AVX2
              │    │
              │    ├─ sign = (direction == FFT_FORWARD) ? -1.0 : +1.0
              │    ├─ base_angle = sign × 2π / N_stage
              │    │
              │    ├─ ┌─────────────────────────────────────────┐
              │    │   │ TWIDDLE COMPUTATION DISPATCH           │
              │    │   └─────────────────────────────────────────┘
              │    │
              │    ├─ IF __AVX2__ AND sub_len > 8:
              │    │   │
              │    │   └─ AVX2 Vectorized Path:
              │    │       │
              │    │       ├─ FOR r = 1 to radix-1:
              │    │       │   │
              │    │       │   ├─ offset = r - 1
              │    │       │   │
              │    │       │   └─ compute_twiddles_avx2(&tw[offset], sub_len, 
              │    │       │                             base_angle, r, radix-1)
              │    │       │        │
              │    │       │        ├─ vbase = _mm256_set1_pd(base_angle)
              │    │       │        ├─ vr = _mm256_set1_pd((double)r)
              │    │       │        │
              │    │       │        └─ FOR i = 0 to sub_len-1 (step 4):
              │    │       │            │
              │    │       │            ├─ Prefetch: &tw[(i+16) * interleave]
              │    │       │            │
              │    │       │            ├─ vi = {i, i+1, i+2, i+3}
              │    │       │            ├─ vang = vbase × vr × vi  (FMA)
              │    │       │            │
              │    │       │            ├─ Extract 4 angles
              │    │       │            │
              │    │       │            └─ FOR j = 0..3:
              │    │       │                ├─ idx = (i+j) × interleave
              │    │       │                └─ sincos_auto(angles[j], 
              │    │       │                     &tw[idx].im, &tw[idx].re)
              │    │       │                     │
              │    │       │                     ├─ IF |angle| ≤ π/4:
              │    │       │                     │   └─> sincos_minimax() 
              │    │       │                     │       (0.5 ULP, FMA polynomials)
              │    │       │                     │
              │    │       │                     └─ ELSE:
              │    │       │                         └─> sincos() / sin()+cos()
              │    │       │                             (system libc)
              │    │       │
              │    │       └─ Scalar tail for remainder
              │    │
              │    └─ ELSE (Scalar Path):
              │        │
              │        └─ FOR k = 0 to sub_len-1:
              │            └─ FOR r = 1 to radix-1:
              │                ├─ idx = k × (radix-1) + (r-1)
              │                ├─ angle = base_angle × r × k
              │                └─ sincos_auto(angle, &tw[idx].im, &tw[idx].re)
              │    │
              │    └─ Return tw (OWNED by stage)
              │
              ├─ stage->stage_tw = tw
              │
              ├─ Log: "Stage %d: radix=%d, N=%d, sub_len=%d, twiddles=%d"
              │
              │
              ├─ ════════════════════════════════════════════════
              │  RADER MANAGER: Prime Radix Convolution Twiddles
              │  ════════════════════════════════════════════════
              │
              ├─ IF is_prime(radix) AND radix >= 7:
              │   │
              │   │   // Known primes: 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67
              │   │
              │   └─ DISPATCH → get_rader_twiddles(radix, direction)
              │        │
              │        ├─ mutex_lock()  // Thread-safe cache access
              │        │
              │        ├─ IF !g_cache_initialized:
              │        │   └─ init_rader_cache()
              │        │       ├─ memset(g_rader_cache, 0, ...)
              │        │       ├─ Pre-populate: create_rader_plan_for_prime(7)
              │        │       ├─ Pre-populate: create_rader_plan_for_prime(11)
              │        │       ├─ Pre-populate: create_rader_plan_for_prime(13)
              │        │       └─ g_cache_initialized = 1
              │        │
              │        ├─ ┌────────────────────────────────────────┐
              │        │   │ CACHE LOOKUP                          │
              │        │   └────────────────────────────────────────┘
              │        │
              │        ├─ FOR i = 0 to MAX_RADER_PRIMES-1:
              │        │   └─ IF g_rader_cache[i].prime == radix:
              │        │       ├─ entry = &g_rader_cache[i]
              │        │       │
              │        │       ├─ result = (direction == FFT_FORWARD)
              │        │       │           ? entry->conv_tw_fwd
              │        │       │           : entry->conv_tw_inv
              │        │       │
              │        │       ├─ mutex_unlock()
              │        │       └─ RETURN result  ← CACHE HIT ✅
              │        │
              │        ├─ // Cache miss - create new entry
              │        │
              │        ├─ mutex_unlock()  // Release during creation
              │        │
              │        ├─ ┌────────────────────────────────────────┐
              │        │   │ CREATE NEW RADER PLAN                 │
              │        │   └────────────────────────────────────────┘
              │        │
              │        └─ create_rader_plan_for_prime(radix)
              │             │
              │             ├─ Find primitive root g from database:
              │             │   │
              │             │   └─ g_primitive_roots[] = {
              │             │        {7,3}, {11,2}, {13,2}, {17,3}, {19,2},
              │             │        {23,5}, {29,2}, {31,3}, {37,2}, {41,6},
              │             │        {43,3}, {47,5}, {53,2}, {59,2}, {61,2}, {67,2}
              │             │      }
              │             │
              │             ├─ IF g < 0: ERROR (prime not in database)
              │             │
              │             ├─ Find free cache slot:
              │             │   └─ FOR i=0..MAX_RADER_PRIMES-1:
              │             │       IF g_rader_cache[i].prime == 0: slot=i; break
              │             │
              │             ├─ IF slot < 0: ERROR (cache full)
              │             │
              │             ├─ entry = &g_rader_cache[slot]
              │             │
              │             ├─ ┌──────────────────────────────────┐
              │             │   │ COMPUTE PERMUTATIONS            │
              │             │   └──────────────────────────────────┘
              │             │
              │             ├─ Allocate:
              │             │   ├─ entry->perm_in = malloc((radix-1) × sizeof(int))
              │             │   └─ entry->perm_out = malloc((radix-1) × sizeof(int))
              │             │
              │             ├─ compute_permutations(radix, g, perm_in, perm_out)
              │             │   │
              │             │   ├─ Input permutation (generator powers):
              │             │   │   └─ FOR i = 0 to radix-2:
              │             │   │       └─ perm_in[i] = g^i mod radix
              │             │   │           └─> [g^0, g^1, g^2, ..., g^(p-2)] mod p
              │             │   │
              │             │   └─ Output permutation (inverse mapping):
              │             │       └─ FOR i = 0 to radix-2:
              │             │           ├─ idx = perm_in[i] - 1  // Map to 0..(p-2)
              │             │           └─ perm_out[idx] = i
              │             │
              │             ├─ ┌──────────────────────────────────┐
              │             │   │ COMPUTE CONVOLUTION TWIDDLES    │
              │             │   └──────────────────────────────────┘
              │             │
              │             ├─ Allocate (32-byte aligned for AVX2):
              │             │   ├─ entry->conv_tw_fwd = aligned_alloc(32, (radix-1)×...)
              │             │   └─ entry->conv_tw_inv = aligned_alloc(32, (radix-1)×...)
              │             │
              │             ├─ FOR q = 0 to radix-2:
              │             │   │
              │             │   ├─ idx = perm_out[q]
              │             │   │
              │             │   ├─ FORWARD twiddle: exp(-2πi × idx / radix)
              │             │   │   ├─ angle_fwd = -2π × idx / radix
              │             │   │   └─ sincos_auto(angle_fwd, 
              │             │   │        &conv_tw_fwd[q].im, &conv_tw_fwd[q].re)
              │             │   │
              │             │   └─ INVERSE twiddle: exp(+2πi × idx / radix)
              │             │       ├─ angle_inv = +2π × idx / radix
              │             │       └─ sincos_auto(angle_inv, 
              │             │            &conv_tw_inv[q].im, &conv_tw_inv[q].re)
              │             │
              │             ├─ entry->prime = radix
              │             ├─ entry->primitive_root = g
              │             │
              │             ├─ Log: "Created Rader plan for prime %d (g=%d) in slot %d"
              │             │
              │             └─ Return 0 (success)
              │
              │        └─ Recursive call: get_rader_twiddles(radix, direction)
              │            └─> Now in cache, will return immediately ✅
              │
              ├─ stage->rader_tw = (returned pointer from cache)
              │   └─> ✅ NOT OWNED by stage (shared from global cache)
              │
              └─ ELSE:
                  └─ stage->rader_tw = NULL
                      └─> Non-prime or radix < 7 (e.g., 2,3,4,5,8,9,16,32)
              │
              ├─ Log: "  → Rader: prime=%d, conv_twiddles=%d" (if applicable)
              │
              └─ N_stage = sub_len  // Update for next stage
         
         │
         ├─ ════════════════════════════════════════════════════
         │  SCRATCH BUFFER ALLOCATION
         │  ════════════════════════════════════════════════════
         │
         ├─ Compute maximum scratch needed across all stages:
         │   │
         │   ├─ scratch_max = 0
         │   ├─ N_stage = N
         │   │
         │   └─ FOR i = 0 to num_stages-1:
         │       ├─ radix = plan->factors[i]
         │       ├─ sub_len = N_stage / radix
         │       ├─ stage_need = radix × sub_len
         │       ├─ IF stage_need > scratch_max:
         │       │   └─ scratch_max = stage_need
         │       └─ N_stage = sub_len
         │
         ├─ Add margin for Rader convolutions:
         │   └─ scratch_needed = scratch_max + 4×N
         │
         ├─ plan->scratch_size = scratch_needed
         ├─ plan->scratch = aligned_alloc(32, scratch_needed × sizeof(fft_data))
         │
         ├─ IF !plan->scratch: ERROR (allocation failed)
         │
         ├─ Log: "Scratch buffer: %zu elements (%.2f KB)"
         │
         └─ Log: "Planning complete!"
    
    └─ RETURN plan  ✅ PLANNING COMPLETE
```

---

## Memory Ownership Summary
```
fft_plan
├─ stages[i].stage_tw          ✅ OWNED (freed by free_stage_twiddles)
├─ stages[i].rader_tw          ❌ BORROWED (pointer to g_rader_cache[].conv_tw_*)
├─ scratch                     ✅ OWNED (freed by aligned_free)
├─ bluestein_tw               ✅ OWNED (if Bluestein, freed by aligned_free)
├─ bluestein_plan_fwd         ✅ OWNED (recursive fft_plan, freed by free_fft)
└─ bluestein_plan_inv         ✅ OWNED (recursive fft_plan, freed by free_fft)

g_rader_cache[i]  (GLOBAL, thread-safe)
├─ conv_tw_fwd                ✅ OWNED (freed by cleanup_rader_cache)
├─ conv_tw_inv                ✅ OWNED (freed by cleanup_rader_cache)
├─ perm_in                    ✅ OWNED (freed by cleanup_rader_cache)
└─ perm_out                   ✅ OWNED (freed by cleanup_rader_cache)