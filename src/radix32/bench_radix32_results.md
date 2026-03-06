# VectorFFT Radix-32 vs FFTW 3.3.10 — AVX-512 Codelet Benchmark

## Setup

| | VectorFFT | FFTW 3.3.10 |
|---|---|---|
| **ISA** | AVX-512F + AVX-512DQ + FMA | AVX-512F + FMA (--enable-avx512) |
| **Layout** | Split-real SoA (`re[]`, `im[]`) | Interleaved AoS (`[re,im,re,im,...]`) |
| **Vector width** | 8 doubles / zmm (8 DFTs parallel) | 4 complex / zmm (4 DFTs parallel) |
| **Algorithm** | 8×4 Cooley-Tukey DIT | Split-radix (genfft DAG-optimized) |
| **Twiddle** | Flat (K≤64) or binary ladder (K≥128) | Flat, 31 pairs per k-step |
| **Compiler** | gcc -O3 -march=native | gcc -O3 -march=native |
| **Planning** | N/A (direct call) | FFTW_PATIENT |

**What's being measured:** K independent twiddled DFT-32 codelets called in a loop. VectorFFT applies inter-stage twiddle factors *and* butterflies. FFTW batch-32 applies butterflies only (no inter-stage twiddles). VectorFFT is doing strictly more work per call.

## Results

| K | N=32K | VecFFT best | FFTW batch-32 | Speedup | VecFFT GF/s | FFTW GF/s |
|---:|------:|------------:|--------------:|--------:|------------:|----------:|
| 8 | 256 | 15.3 ns/dft | 21.6 ns/dft | **1.41×** | 52.2 | 37.1 |
| 16 | 512 | 16.1 ns/dft | 20.4 ns/dft | **1.27×** | 49.8 | 39.1 |
| 32 | 1024 | 17.8 ns/dft | 20.3 ns/dft | **1.14×** | 44.9 | 39.5 |
| 64 | 2048 | 27.1 ns/dft | 49.8 ns/dft | **1.84×** | 29.5 | 16.1 |
| 128 | 4096 | 25.6 ns/dft | 48.1 ns/dft | **1.88×** | 31.3 | 16.6 |
| 256 | 8192 | 30.2 ns/dft | 52.0 ns/dft | **1.72×** | 26.5 | 15.4 |
| 512 | 16384 | 34.3 ns/dft | 55.0 ns/dft | **1.60×** | 23.3 | 14.6 |
| 1024 | 32768 | 36.8 ns/dft | 54.4 ns/dft | **1.48×** | 21.7 | 14.7 |

Correctness verified against scalar reference: max |error| < 6×10⁻¹⁴ for both implementations.

## Kernel variant breakdown

| K | Flat | Ladder U1 | Ladder U2 | Winner |
|---:|-----:|----------:|----------:|--------|
| 8 | **15.3** | 17.4 | — | Flat |
| 16 | 19.3 | 16.5 | **16.1** | Ladder U2 |
| 32 | 20.2 | 22.0 | **17.8** | Ladder U2 |
| 64 | 30.7 | 28.9 | **27.1** | Ladder U2 |
| 128 | 32.6 | **25.6** | 32.3 | Ladder U1 |
| 256 | 33.8 | 30.6 | **30.2** | Ladder U2 |
| 512 | 35.4 | **34.3** | 34.8 | Ladder U1 |
| 1024 | 41.8 | **36.8** | 39.5 | Ladder U1 |

The flat kernel wins at K=8 where the twiddle table fits L1 and there's no ladder derivation overhead. The ladder variants take over at K≥16 due to the 6× smaller twiddle table (10K vs 62K doubles). U2 helps at mid-range K where the extra ILP offsets the doubled spill buffer, but U1 wins at K≥128 where U2's register pressure causes compiler spills.

## Why VectorFFT wins

**8-wide split-real vs 4-wide interleaved.** On AVX-512, a `__m512d` holds 8 doubles. VectorFFT packs 8 independent DFTs per vector (split re/im arrays). FFTW packs 4 complex numbers per vector (interleaved `[re,im]` pairs). This is a 2× data parallelism advantage for butterflies, partially offset by FFTW's cheaper interleaved twiddle multiply (`_mm512_fmsubadd_pd`).

**Contiguous memory access.** VectorFFT's stride-K layout gives contiguous aligned loads for each `in_re[n*K + k]` access within a k-step. FFTW's batch-32 with stride-K interleaved access requires gathering from `x[WS(rs, n)]` which scatters across cache lines as K grows. This explains the widening gap at K≥64.

**Twiddle table compression.** The binary ladder loads 5 base twiddles and derives the other 26 via multiplication chains. The table is 10K doubles vs FFTW's 62K. At K=128 the flat table is 62×128×8 = 62KB (exceeds L1), while the ladder table is 10×128×8 = 10KB (fits L1 comfortably).

## Where FFTW still has an edge

**Arithmetic complexity.** FFTW's genfft uses split-radix which achieves ~10-15% fewer flops than 8×4 Cooley-Tukey for N=32. This nearly closes the gap at K=32 (1.14×) where both codes are L1-hot and compute-bound.

**DAG-level scheduling.** genfft's pipeline-aware scheduler interleaves loads with compute. VectorFFT's two-pass structure has a hard spill barrier (32 stores, then 32 reloads) that the OOO engine must pipeline across.

**Full-transform planning.** FFTW's planner can decompose N=32K using different factorizations (e.g. radix-16 × radix-2 × K instead of radix-32 × K), choosing the best plan at runtime. VectorFFT's advantage here is codelet-level only.

## Potential improvements (FFTW-inspired)

1. **Split-radix inner kernel.** Replace 8×4 Cooley-Tukey with split-radix DFT-32. Would save ~40-50 FMA ops per 8-DFT batch, widening the lead at small K where the margin is thinnest.

2. **Reduced spill barrier.** Instead of pass-1-all then pass-2-all, interleave pass-2 radix-4 combines as soon as their inputs are ready from pass-1 sub-FFTs. Reduces live spill set from 32 to ~16 slots.

3. **Fix U2 register pressure at large K.** Interleave the two pipelines at sub-FFT granularity rather than running them sequentially, keeping derived twiddles live in registers.

4. **U=3 variant.** Process 24 doubles per k-step. AVX-512 has 32 zmm registers; U=3 may find a better balance between ILP and spill pressure than U=2. Bench pending on ICX/SPR/Zen4.
