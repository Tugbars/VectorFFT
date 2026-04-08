<img width="1280" height="640" alt="vectorfft_banner (1)" src="https://github.com/user-attachments/assets/27eeb333-f0a9-41d4-8b00-8624b1f1ee07" />

<p align="center">
  A permutation-free mixed-radix Fast Fourier Transform library library in C with hand-tuned AVX2/AVX-512 codelets.<br>
  <b>Beats Intel MKL and FFTW on every tested size.</b> No external dependencies.
</p>

The classical FFT literature optimized a cost model where arithmetic operations were the dominant resource. Mature libraries descending from that tradition — FFTW, MKL, and their kin — inherit codelet structures and factorization heuristics shaped by that assumption. On modern SIMD CPUs the binding constraint has shifted from arithmetic throughput to register pressure, memory bandwidth, and instruction-level parallelism; algorithms that were optimal under the old cost model are no longer optimal under the new one. VectorFFT was designed from the hardware backward: the butterfly, radix, and twiddle-storage decisions are driven by what the machine is actually bottlenecked on, not by operation count minimization

---

## Benchmark Results

### VectorFFT vs Intel MKL vs FFTW

![Throughput](src/stride-fft/bench/vfft_throughput.png)

> **Platform:** Intel Core i9-14900KF, 48 KB L1d, DDR5, AVX2, single-threaded  
> **Competitors:** FFTW 3.3.10 (FFTW_MEASURE), Intel MKL (sequential)

### Speedup over Intel MKL and FFTW

![Speedup](src/stride-fft/bench/vfft_speedup_bars.png)

### VectorFFT vs Intel MKL

![vs MKL](src/stride-fft/bench/vfft_vs_mkl.png)

| N | K | vs FFTW | vs MKL |
|---|---|---------|--------|
| 60 | 32 | 4.81x | **5.36x** |
| 49 | 32 | 4.60x | **4.11x** |
| 200 | 256 | 3.78x | **3.36x** |
| 1000 | 256 | 3.17x | **2.78x** |
| 5000 | 32 | 5.88x | **2.95x** |
| 10000 | 256 | 7.92x | **1.82x** |
| 10000 | 1024 | 9.67x | **2.19x** |
| 50000 | 256 | 16.02x | **1.77x** |
| 100000 | 32 | 13.98x | **2.55x** |
| 16384 | 1024 | 7.00x | **1.82x** |
| 4096 | 1024 | 2.69x | **1.75x** |

### Prime-N Performance (Rader & Bluestein)

<img width="3172" height="1452" alt="vfft_primes" src="https://github.com/user-attachments/assets/cbc42dd5-a983-4e2e-be41-10bf5328ca28" />

Smooth primes (N-1 is 19-smooth) use **Rader's algorithm** with hand-optimized codelets for the convolution. Non-smooth primes use **Bluestein's algorithm** (chirp-z transform).

| Prime N | K | Method | vs FFTW | vs MKL |
|---------|---|--------|---------|--------|
| 61 | 32 | Rader | 3.89x | **5.41x** |
| 337 | 256 | Rader | 2.37x | **2.58x** |
| 2053 | 256 | Rader | 2.52x | **2.24x** |
| 263 | 32 | Bluestein | 1.04x | **1.10x** |

**Rader primes: 1.39x - 5.41x vs MKL (mean 2.50x)**. Bluestein handles the hard cases where no efficient decomposition exists.

---

## Accuracy

<img width="3174" height="1230" alt="vfft_accuracy" src="https://github.com/user-attachments/assets/ba2dafe4-127b-4acb-b3b4-fe482cf53fd6" />

All three libraries achieve comparable accuracy against brute-force O(N^2) DFT reference. VectorFFT's errors are 1.3-2.5x higher than FFTW/MKL due to the multi-stage stride-based decomposition (more intermediate twiddle multiplications), but remain well within double-precision tolerance and follow the theoretical O(N * epsilon * log N) bound.

Roundtrip error (fwd + bwd / N) is at machine epsilon (~1e-16) for all sizes -- the permutation-free architecture guarantees perfect cancellation.

> Run `vfft_bench` to see accuracy results for your hardware.

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
- *permutation-free tensor-stride FFT** -- bakes the common factor (product of all outer-stage twiddles) into each per-leg twiddle table at plan time, reducing the forward path to a single twiddle multiply per butterfly leg
- **Scalar twiddle optimization** -- each twiddle row is a single scalar replicated K times; the executor stores only (R-1) unique doubles per group and broadcasts them to SIMD width inside the codelet, cutting twiddle memory from (R-1)\*K\*16 bytes to (R-1)\*16 bytes
- **Split-complex layout** -- separate `re[]` / `im[]` arrays, naturally aligned for SIMD without interleave/deinterleave overhead

**VTune Microarchitecture Profile (N=10000, K=256):**

| Metric | Value |
|--------|-------|
| L2 Bound | 4.7% |
| L3 Bound | 7.1% |
| Store Bound | 6.7% |
| DTLB Overhead | 2.0% |
| Retiring | 29.3% |

### Codelets

- **18 hand-optimized radixes** (2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 25, 32, 64) covering every smooth integer up to 100,000+
- **Composite radix codelets** (R=6, 10, 12, 16, 20, 25) use a two-pass internal CT decomposition with spill-optimized register scheduling
- **Generated by Python scripts** with ISA-specific scheduling: register allocation, spill management, FMA pipelining, and constant folding for internal twiddle factors
- **Prime-radix butterflies** (R=11, 13, 17, 19) derived from FFTW's genfft algebraic DAG output, then re-scheduled using Sethi-Ullman optimal register allocation with explicit spill management to fit AVX2's 16 YMM registers
- **Three ISA targets**: AVX2 (VL=4), AVX-512 (VL=8), scalar fallback -- selected at compile time

### Planner

- **Recursive DP planner** (FFTW-style) for large N: tries each available radix as the first stage, recursively solves sub-problems with memoization, then benchmarks all orderings of the winning factorization (~150 benchmarks for N=100,000 vs ~61,000 for exhaustive search)
- **Exhaustive search** for small N (<=1024): benchmarks every valid factorization and ordering
- **Wisdom file** caches optimal plans per (N, K) pair, with versioning
- **Heuristic fallback** for instant planning when no wisdom is available: greedy largest-first factorization with SIMD-aware reordering (power-of-2 innermost)

### Prime-N Support

- **Rader's algorithm** for smooth primes (N-1 is 19-smooth): reduces prime-N DFT to a cyclic convolution of length N-1, computed using the existing mixed-radix engine
- **Bluestein's algorithm** (chirp-z transform) for non-smooth primes: zero-pads to a composite-friendly convolution length

---

## Quick Start

```bash
git clone https://github.com/Tugbars/VectorFFT.git
cd VectorFFT && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

```c
#include "planner.h"

stride_registry_t reg;
stride_registry_init(&reg);

stride_plan_t *plan = stride_auto_plan(N, K, &reg);
stride_execute_fwd(plan, re, im);   // forward DIT
stride_execute_bwd(plan, re, im);   // backward DIF (divide by N to normalize)
stride_plan_destroy(plan);
```

See [`examples/basic_fft.c`](examples/basic_fft.c) for a complete working example covering all API entry points: environment setup, registry, heuristic and wisdom-based planning, forward/backward execution, prime-N support (Rader & Bluestein), and memory management.

---

## Project Structure

```
src/stride-fft/
  core/               Runtime engine (header-only)
    executor.h        In-place stride-based executor (Method C)
    planner.h         Top-level API: auto_plan, wise_plan, exhaustive_plan
    registry.h        ISA-aware codelet registry (AVX-512 > AVX2 > scalar)
    factorizer.h      CPU-aware heuristic factorizer
    exhaustive.h      Exhaustive factorization search
    compat.h          Portable timer + aligned alloc
  codelets/           Generated SIMD headers (~150k lines)
    avx2/             47 AVX2 codelet headers
    avx512/           47 AVX-512 codelet headers
    scalar/           47 scalar fallback headers
  generators/         Python codelet generators
    gen_radix2.py .. gen_radix64.py
    generate_all.bat
  bench/
    bench_planner.c   Full benchmark (vs FFTW + MKL)
    plot_results.py   Comparison graphs
```

---

## Acknowledgments

- [FFTW](http://www.fftw.org/) by Matteo Frigo and Steven G. Johnson -- the gold standard for decades. VectorFFT's prime-radix butterflies (R=11, 13, 17, 19) are derived from FFTW's genfft algebraic output, then re-scheduled using Sethi-Ullman register allocation with explicit spill management to minimize register pressure on AVX2 (16 YMM) and AVX-512 (32 ZMM).
- [VkFFT](https://github.com/DTolm/VkFFT) by Dmitrii Tolmachev -- inspiration for the benchmarking methodology and presentation style.
