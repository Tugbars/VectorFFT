# VTune Profile: Full FFT N=16384 K=4 — VectorFFT vs MKL

**Date:** 2026-04-10  
**CPU:** Intel i9-14900KF (Raptor Lake), P-core only, 5.67-5.71 GHz turbo  
**Binary:** `vfft_bench_vtune.exe 16384 4 [ours|mkl] 10`  
**Data size:** N=16384, K=4, total=65536 elements = 1 MB (re+im)  
**Factorization:** VectorFFT uses `2x8x16x64` (from wisdom)  
**Analysis:** Microarchitecture Exploration (uarch-exploration)  
**Wallclock ratio:** VectorFFT 1.02x over MKL (our closest margin)

---

## Top-Level Comparison

| Category | VectorFFT | MKL | Winner |
|----------|-----------|-----|--------|
| **Retiring** | **48.4%** | 38.2% | **VectorFFT (+10.2%)** |
| Front-End Bound | 4.6% | 3.0% | MKL |
| Bad Speculation | 2.3% | 0.2% | MKL |
| **Back-End Bound** | **44.8%** | **58.6%** | **VectorFFT** |

## CPI / IPC

| Metric | VectorFFT | MKL |
|--------|-----------|-----|
| **CPI** | **0.330** | 0.429 |
| IPC | 3.03 | 2.33 |
| Clockticks | 55.2B | 55.2B |
| Instructions Retired | 167.0B | 128.8B |

VectorFFT executes 30% more instructions than MKL but completes them 27% faster (lower CPI). More efficient use of the pipeline.

## Memory Bound Breakdown

| Metric | VectorFFT | MKL | Winner |
|--------|-----------|-----|--------|
| **Memory Bound (total)** | **33.0%** | **50.4%** | **VectorFFT** |
| L1 Bound | 7.2% | 7.0% | Tied |
| L2 Bound | 4.4% | 5.6% | VectorFFT |
| L3 Bound | 0.5% | 2.2% | VectorFFT |
| DRAM Bound | 0.1% | 0.1% | Tied |
| **Store Bound** | **24.7%** | **36.5%** | **VectorFFT** |
| Store Latency | 37.3% | 41.7% | VectorFFT |
| FB Full | 3.0% | 7.2% | VectorFFT |

## DTLB Analysis (key asymmetry)

| Metric | VectorFFT | MKL | Notes |
|--------|-----------|-----|-------|
| **DTLB Load Overhead** | **14.5%** | **26.7%** | VectorFFT wins |
| Load STLB Hit | 14.4% | 26.6% | |
| **DTLB Store Overhead** | **32.4%** | **2.7%** | MKL wins |
| Store STLB Hit | 32.4% | 2.6% | |
| **Combined DTLB** | **46.9%** | **29.4%** | MKL wins overall |

MKL shows 26.7% load DTLB but only 2.7% store DTLB — suggesting MKL writes to fewer distinct pages (possibly using a small write buffer or a different store pattern). VectorFFT's store DTLB at 32.4% is the single largest improvement opportunity.

## Port Utilization

| Port | VectorFFT | MKL | Winner |
|------|-----------|-----|--------|
| **Port 0 (FMA)** | **35.5%** | 31.5% | VectorFFT |
| **Port 1 (FMA)** | **47.6%** | 36.1% | VectorFFT |
| Port 6 (ALU) | 17.6% | 15.7% | VectorFFT |
| **3+ Ports Utilized** | **60.8%** | **47.0%** | **VectorFFT** |
| 2 Ports Utilized | 11.2% | 14.5% | |
| 1 Port Utilized | 8.1% | 16.2% | VectorFFT |
| 0 Ports Utilized | 0.6% | 0.4% | |

VectorFFT utilizes FMA ports significantly better: 35.5%/47.6% vs MKL's 31.5%/36.1%. MKL spends 16.2% of cycles utilizing only 1 port (vs our 8.1%) — indicating more serialization in MKL's execution.

## Core Bound

| Metric | VectorFFT | MKL |
|--------|-----------|-----|
| Core Bound | 11.7% | 8.2% |
| Port Utilization | 19.0% | 26.0% |
| Serializing Ops | 0.8% | 1.6% |

## Front-End

| Metric | VectorFFT | MKL |
|--------|-----------|-----|
| DSB Coverage | 90.9% | 56.4% |
| LSD Coverage | 3.0% | 41.4% |
| ICache Misses | 0.1% | 0.3% |

VectorFFT achieves 91% DSB coverage (uop cache hits) vs MKL's 56%. MKL relies more heavily on the LSD (41% vs 3%). This suggests MKL has smaller, tighter inner loops (LSD-friendly) while VectorFFT has larger codelet bodies that fill the DSB well.

## L1 Latency Dependency

| Metric | VectorFFT | MKL |
|--------|-----------|-----|
| L1 Latency Dependency | 28.3% | 18.9% |

VectorFFT shows higher L1 latency dependency — consistent with in-place stride-based execution where consecutive stages read and write the same buffer. MKL may use an out-of-place stage or different buffer layout.

---

## Key Findings

### 1. VectorFFT is microarchitecturally superior
Despite near-identical wallclock (1.02x), VectorFFT retires 48.4% vs MKL's 38.2% — 26% more efficient pipeline utilization. VectorFFT achieves this with better FMA port saturation and less memory stalling overall.

### 2. MKL executes fewer instructions
MKL retires 128.8B instructions vs our 167.0B — 23% fewer. This means MKL uses a different algorithmic decomposition (possibly split-radix or larger monolithic codelets) that requires fewer total operations, but executes them less efficiently.

### 3. DTLB store overhead is our weakness
32.4% of clockticks wasted on store DTLB vs MKL's 2.7%. This 30% gap is the single largest opportunity. MKL appears to concentrate stores to fewer pages — possibly through:
- Write-combining buffer
- Different stage traversal order (blocking)
- Smaller working set per stage via tiling

### 4. Both libraries are memory-bound at this size
VectorFFT 33% memory-bound, MKL 50%. Neither is compute-bound. The 1 MB working set (N=16384, K=4, split-complex) fits in L2 (2 MB) but not L1 (48 KB). Cache reuse between stages is the limiting factor for both.

---

## Improvement Targets for VectorFFT

| Target | Current | Goal | Expected Impact |
|--------|---------|------|-----------------|
| DTLB Store Overhead | 32.4% | <5% | ~15-20% speedup |
| L1 Latency Dependency | 28.3% | <20% | ~5% speedup |
| Store Bound | 24.7% | <15% | ~5-10% speedup |

### Approaches
1. **Stage blocking / tiling** — process FFT in cache-resident blocks, completing all stages per block before moving on. Concentrates stores to same pages.
2. **Huge pages (2 MB)** — eliminates DTLB entirely for 1 MB working set. Not a fair benchmark comparison, but useful for production users.
3. **Prefetch stores** — `_mm_prefetch` with `_MM_HINT_T0` before store to warm TLB entries.
4. **Investigate MKL's store pattern** — disassemble MKL's inner loop to see how they achieve 2.7% store DTLB.
