<img width="2560" height="960" alt="image" src="https://github.com/user-attachments/assets/0a206a63-f7d3-45f4-8e1d-be3aca3e2cb5" />

<p align="center">
  A permutation-free mixed-radix Fast Fourier Transform library in C with hand-tuned AVX2/AVX-512 codelets.<br>
  <b>Beats Intel MKL on every tested size — 207/207 benchmarks.</b> No external dependencies.
</p>

---

## Benchmark Results

> **Platform:** Intel Core i9-14900KF, 48 KB L1d, DDR5, AVX2, single-threaded  
> **Competitor:** Intel MKL 2025 (sequential, `mkl_set_num_threads(1)`)  
> **207 data points** across 9 categories, 3 batch sizes (K=4, K=32, K=256), N=8 to N=823,543

### 1D FFT Throughput — VectorFFT vs Intel MKL

![Throughput](docs/performance/vfft_throughput_1d.png)

Three panels showing GFLOP/s at each batch size. Blue = VectorFFT, Red = MKL. Different marker shapes per category. VectorFFT sits above MKL across the board.

| Category | Cells | Best Win | Closest MKL Gets |
|----------|-------|----------|-------------------|
| **Small pow2** (8-128) | 15 | **8.98x** (N=8, K=4) | 2.38x (N=128, K=4) |
| **Power-of-2** (256-131K) | 30 | **2.74x** (N=512, K=256) | 1.17x (N=131072, K=4) |
| **Composite** (60-100K) | 33 | **5.15x** (N=60, K=32) | 1.69x (N=50000, K=256) |
| **Prime powers** (3,5,7) | 30 | **3.95x** (N=390625, K=4) | 1.26x (N=243, K=4) |
| **Prime powers** (R=11,13) | 15 | **3.06x** (N=14641, K=32) | 1.50x (N=2197, K=256) |
| **Rader primes** | 24 | **3.42x** (N=641, K=32) | 1.05x (N=127, K=4) |
| **Bluestein primes** | 24 | **3.09x** (N=83, K=4) | 1.01x (N=179, K=256) |
| **Odd composites** | 18 | **4.20x** (N=175, K=256) | 1.86x (N=6615, K=4) |
| **Mixed deep** | 18 | **3.09x** (N=4620, K=32) | 1.70x (N=6930, K=4) |

### Speedup over Intel MKL — All Categories

![Speedup](docs/performance/vfft_speedup_vs_mkl.png)

Every point above the dashed line is a VectorFFT win. Marker size indicates batch count (small=K=4, medium=K=32, large=K=256). **All 207 points are above parity.**

| Metric | Value |
|--------|-------|
| **Win rate** | 207 / 207 (100%) |
| **Median speedup** | 2.36x |
| **Best speedup** | 8.98x (N=8, K=4) |
| **Closest MKL** | 1.01x (N=179, K=256, Bluestein) |
| **Peak GFLOP/s** | 103.0 (N=8, K=256) |

### Combined Dense Scatter — All Sizes & Batch Counts

![Scatter](docs/performance/vfft_scatter_all.png)

All 207 data points overlaid. Blue cloud (VectorFFT) consistently above red cloud (MKL). Peak throughput at small N with large K where codelets run entirely from L1.

### Highlight Results

| N | K | Category | Factors | VectorFFT | MKL | Speedup |
|---|---|----------|---------|-----------|-----|---------|
| 8 | 256 | small | 8 | 103.0 GF/s | 13.3 GF/s | **7.75x** |
| 8 | 4 | small | 8 | 49.1 GF/s | 5.5 GF/s | **8.98x** |
| 60 | 32 | composite | 12x5 | 82.0 GF/s | 15.5 GF/s | **5.30x** |
| 390625 | 4 | prime_pow (5^8) | 25x25x25x25 | 39.3 GF/s | 11.0 GF/s | **3.95x** |
| 641 | 32 | rader | (override) | 15.1 GF/s | 4.4 GF/s | **3.42x** |
| 175 | 256 | odd_comp | 5x5x7 | 56.6 GF/s | 15.3 GF/s | **4.20x** |
| 4620 | 32 | mixed_deep | 7x6x10x11 | 47.2 GF/s | 14.6 GF/s | **3.09x** |
| 83 | 4 | bluestein | (override) | 6.2 GF/s | 2.0 GF/s | **3.09x** |
| 131072 | 32 | pow2 | 16x4x4x4x4x4x8 | 18.4 GF/s | 11.3 GF/s | **1.49x** |
| 131072 | 4 | pow2 | 4x4x4x4x8x4x4x4 | 22.8 GF/s | 21.0 GF/s | 1.18x |

### 2D FFT — Tiled SIMD Transpose

VectorFFT's 2D FFT uses a tiled gather/scatter approach with cache-oblivious SIMD transpose kernels (8x4 line-filling + 4x4 AVX2). Beats MKL at all tested sizes.

| Size | VectorFFT | MKL | Speedup |
|------|-----------|-----|---------|
| 32x32 | 0.9 us | 1.5 us | **1.63x** |
| 64x64 | 5.5 us | 6.5 us | **1.18x** |
| 128x128 | 30.3 us | 33.4 us | **1.10x** |
| 256x256 | 127.3 us | 145.6 us | **1.14x** |
| 512x512 | 875.1 us | 948.1 us | **1.08x** |
| 1024x1024 | 3,900 us | 5,512 us | **1.41x** |
| 100x200 | 40.7 us | 60.9 us | **1.50x** |

Multi-threaded 2D (tile-parallel, per-thread scratch, zero barriers):

| Size | 1 thread | 8 threads | Speedup |
|------|----------|-----------|---------|
| 256x256 | 147.5 us | 37.0 us | **3.99x** |
| 1024x1024 | 3,787 us | 1,002 us | **3.78x** |

### Prime-N Performance (Rader and Bluestein)

Both prime-handling paths use wisdom-driven inner FFTs — calibrating the inner cell sizes lets the prime path pick up the same per-stage variant codes (LOG3, T1S, BUF) as the regular composite path.

**Rader (smooth N-1):**

| Prime N | K | vs MKL |
|---------|---|--------|
| 127 | 32 | **2.11x** |
| 257 | 32 | **3.29x** |
| 641 | 32 | **3.42x** |
| 1009 | 32 | **1.95x** |
| 2801 | 32 | **2.46x** |
| 4001 | 32 | **1.28x** |

**Bluestein (non-smooth N-1, chirp-z):**

| Prime N | K | vs MKL |
|---------|---|--------|
| 47 | 32 | **1.71x** |
| 83 | 32 | **2.98x** |
| 167 | 32 | **1.66x** |
| 263 | 32 | **1.61x** |
| 311 | 32 | **1.57x** |

---

## Accuracy

![Precision](docs/performance/vfft_precision.png)

Roundtrip error (fwd + bwd / N) across all tested sizes. Errors follow the theoretical O(log2(N) * epsilon) bound. All values within double-precision tolerance.

| Category | Min Error | Max Error |
|----------|-----------|-----------|
| Small pow2 (8-128) | 1.1e-16 | 3.3e-16 |
| Power-of-2 (256-131K) | 2.8e-16 | 6.7e-16 |
| Composite | 2.4e-16 | 8.3e-16 |
| Prime powers (3,5,7) | 4.6e-16 | 8.8e-16 |
| Prime powers (R=11,13) | 4.6e-16 | 8.8e-16 |
| Odd composites | 4.3e-16 | 6.7e-16 |
| Mixed deep | 5.5e-16 | 1.2e-15 |

Errors stay within the theoretical O(log2(N) · ε) bound across the grid. The permutation-free roundtrip guarantees perfect cancellation of digit-reversal — no accumulated permutation error from the Cooley-Tukey decomposition.

Rader and Bluestein prime cells use a convolution-based path; their roundtrip error is comparable (~6e-16 to 2e-15 range, dominated by the inner FFT's accumulated rounding), well within FP64 acceptable bounds.

---

## Architecture

VectorFFT uses a **permutation-free stride-based Cooley-Tukey** architecture.

### Permutation-Free Roundtrip

Traditional FFT libraries (FFTW, MKL) compute the Cooley-Tukey decomposition followed by a **digit-reversal permutation** -- an O(N) memory shuffle that produces no useful computation but costs cache misses. VectorFFT eliminates this entirely:

- **Forward (DIT):** stages process data at increasing strides; output lands in digit-reversed order
- **Backward (DIF):** stages process in reverse order, naturally undoing the permutation
- **Roundtrip (fwd + bwd):** produces natural-order output with zero permutation overhead

This saves one full data pass per transform. At large N with large batch sizes, that's the difference between 1x and 16x over FFTW.

### Executor

- **Zero scratch buffers** -- fully in-place, single buffer for all stages
- **Method C fused twiddles** -- bakes the common factor (product of all outer-stage twiddles) into each per-leg twiddle table at plan time, reducing the forward path to a single twiddle multiply per butterfly leg
- **Scalar twiddle optimization** -- each twiddle row is a single scalar replicated K times; the executor stores only (R-1) unique doubles per group and broadcasts them to SIMD width inside the codelet, cutting twiddle memory from (R-1)\*K\*16 bytes to (R-1)\*16 bytes
- **Split-complex layout** -- separate `re[]` / `im[]` arrays, naturally aligned for SIMD without interleave/deinterleave overhead

### Codelets

- **18 hand-optimized radixes** (2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 25, 32, 64) covering every smooth integer up to 100,000+
- **Composite radix codelets** (R=6, 10, 12, 16, 20, 25) use a two-pass internal CT decomposition with spill-optimized register scheduling
- **Generated by Python scripts** with ISA-specific scheduling: register allocation, spill management, FMA pipelining, and constant folding for internal twiddle factors
- **Prime-radix butterflies** (R=11, 13, 17, 19) derived from FFTW's genfft algebraic DAG output, then re-scheduled using Sethi-Ullman optimal register allocation with explicit spill management to fit AVX2's 16 YMM registers
- **Three ISA targets**: AVX2 (VL=4), AVX-512 (VL=8), scalar fallback -- selected at compile time

### Planner

- **Recursive DP planner** (FFTW-style) for large N: tries each available radix as the first stage, recursively solves sub-problems with memoization, then benchmarks all orderings of the winning factorization (~150 benchmarks for N=100,000 vs ~61,000 for exhaustive search)
- **Exhaustive search** for small N (<=2048): benchmarks every valid factorization and ordering
- **Wisdom file** caches optimal plans per (N, K) pair with incremental saves (crash-safe) and `--recalibrate` support
- **Heuristic fallback** for instant planning when no wisdom is available: greedy largest-first factorization with SIMD-aware reordering (power-of-2 innermost)

### 2D FFT

- **Tiled SIMD gather/scatter** with cache-oblivious recursive transpose
- **8x4 line-filling kernel** writes full 64-byte cache lines, eliminating write-allocate penalties
- **Three-regime dispatch** (L1/L2/L3) with configurable tile size
- **Tile-parallel threading** for row FFTs — per-thread scratch buffers, zero barriers

### Prime-N Support

- **Rader's algorithm** for smooth primes (N-1 is 19-smooth): reduces prime-N DFT to a cyclic convolution of length N-1, computed using the existing mixed-radix engine
- **Bluestein's algorithm** (chirp-z transform) for non-smooth primes: zero-pads to a composite-friendly convolution length

### R2C / C2R

- **Pair-packing R2C** with fused first/last stage (t1_oop / n1_scaled codelets)
- **1.5-1.9x faster** than equivalent complex FFT
- Hermitian postprocess in natural frequency order

---

## Quick Start

```bash
git clone https://github.com/Tugbars/VectorFFT.git
cd VectorFFT && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### C API (opaque handles)

```c
#include <vfft.h>

vfft_init();
vfft_plan p = vfft_plan_c2c(1024, 256);
double *re = vfft_alloc(1024 * 256 * sizeof(double));
double *im = vfft_alloc(1024 * 256 * sizeof(double));
// ... fill re[], im[] ...
vfft_execute_fwd(p, re, im);
vfft_execute_bwd_normalized(p, re, im);  // roundtrip: output == input
vfft_destroy(p);
vfft_free(re); vfft_free(im);
```

### Internal API (header-only)

```c
#include "planner.h"

stride_registry_t reg;
stride_registry_init(&reg);

stride_plan_t *plan = stride_auto_plan(N, K, &reg);
stride_execute_fwd(plan, re, im);
stride_execute_bwd(plan, re, im);
stride_plan_destroy(plan);
```

### Threading

```c
// Option A: internal parallelism (library manages threads)
vfft_set_num_threads(8);
vfft_execute_fwd(plan, re, im);  // call from ONE thread

// Option B: external parallelism (you manage threads)
vfft_set_num_threads(1);
// Each of your threads calls execute on its own data — safe.
```

### 2D FFT

```c
vfft_plan p = vfft_plan_2d(256, 256);
vfft_execute_fwd(p, re, im);
```

See [`examples/`](examples/) for complete working examples including spectrum analyzer with live audio.

---

## Reproducing Benchmarks

```bash
# Build benchmarks
cmake --build build --target vfft_bench_1d_csv vfft_bench_2d_csv

# Run 1D benchmark (calibrates wisdom on first run, ~minutes)
./build/bin/vfft_bench_1d_csv

# Force recalibration after codelet changes
./build/bin/vfft_bench_1d_csv --recalibrate

# Results: build/bin/vfft_perf_1d.csv, build/bin/vfft_acc_1d.csv
# Plot: open src/stride-fft/bench/plot_vfft.ipynb
```

---

## Roadmap

### Near-term
- **ILP optimization for large power-of-2** -- VTune profiling shows MKL achieves better instruction-level parallelism on large pow2 at K=4 (our closest margin: 1.02x at N=16384). Codelet scheduling improvements to increase FMA port saturation.
- **Natural-order DFT output** (`vfft_permute`) -- expose the digit-reversal permutation table so users can inspect individual frequency bins without roundtrip
- **R=9 codelet** -- unlocks 3^N sizes (3^10 = 59049, 3^12 = 531441) that currently exceed the max stage depth with R=3 alone. Fits AVX2's 16 YMMs.
- **Strided-batch codelets for 2D** -- eliminate transpose entirely by allowing non-unit batch stride in codelets. Estimated 1.27x over MKL at 256x256 (currently 1.14x with tiled transpose).
- **K=1 scalar fallback** -- AVX2 codelets currently require K>=4 (SIMD width). Add scalar path for single-transform use cases.
- **Native interleaved C2C** -- dedicated codelets for interleaved complex layout (re+im adjacent), avoiding deinterleave overhead for users with packed data.

### Medium-term
- **1D Bailey 4-step** -- natural-order output for large 1D transforms using transpose + twiddle infrastructure (already built and benchmarked)
- **3D FFT** -- extend tiled 2D approach to three dimensions
- **ARM NEON / SVE codelets** -- port codelet generators to ARM SIMD targets
- **Single-precision (float32)** support -- separate codelet set with 8-wide AVX2 / 16-wide AVX-512

### Long-term
- **GPU backend** (Vulkan compute / CUDA) -- VkFFT-style GPU execution sharing the same planner and wisdom infrastructure
- **Distributed FFT** (MPI) -- multi-node decomposition for very large transforms
- **White paper** -- microarchitectural profiling (PMU counters, spill analysis, roofline) documenting why VectorFFT beats MKL at the instruction level

---

## Acknowledgments

- [FFTW](http://www.fftw.org/) by Matteo Frigo and Steven G. Johnson -- the gold standard for decades. VectorFFT's prime-radix butterflies (R=11, 13, 17, 19) are derived from FFTW's genfft algebraic output, then re-scheduled using Sethi-Ullman register allocation with explicit spill management to minimize register pressure on AVX2 (16 YMM) and AVX-512 (32 ZMM).
- [VkFFT](https://github.com/DTolm/VkFFT) by Dmitrii Tolmachev -- inspiration for the benchmarking methodology and presentation style.
