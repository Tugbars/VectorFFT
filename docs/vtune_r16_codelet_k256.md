# VTune Profile: R=16 Isolated Codelet (t1_dit_fwd, K=256)

**Date:** 2026-04-11  
**CPU:** Intel i9-14900KF (Raptor Lake), P-core only, 5.69 GHz turbo  
**Binary:** `vfft_bench_codelet.exe 256 5000 16 t1_dit_fwd`  
**Codelet:** `radix16_t1_dit_fwd_avx2` — in-place DIT twiddle + butterfly, AVX2  
**Data size:** R=16, K=256, total=4096 elements = 64 KB (re+im), L1 boundary  
**Analysis:** Microarchitecture Exploration (uarch-exploration)

---

## Top-Level Breakdown

| Category | % Pipeline Slots | vs R=4 | vs R=8 |
|----------|-----------------|--------|--------|
| **Retiring** | **21.8%** | 85.9% | 72.2% |
| Front-End Bound | 1.5% | 2.3% | 2.8% |
| Bad Speculation | 0.5% | 2.0% | 0.2% |
| **Back-End Bound** | **76.2%** | 9.8% | 24.7% |

## Retiring Detail

| Metric | Value | vs R=4 | vs R=8 |
|--------|-------|--------|--------|
| Light Operations | 21.8% | 85.8% | 72.2% |
| FP Arithmetic | 53.4% of uOps | 62.9% | 58.8% |
| FP Vector (256-bit) | 53.7% of uOps | 62.5% | 59.2% |
| Memory Operations | 10.6% | 38.9% | 30.1% |

## Back-End Detail — Memory Bound (74.0%)

The dominant bottleneck. R=16 needs 32 complex values (64 doubles) for the butterfly inputs — the codelet uses explicit `double spill_re[64], spill_im[64]` stack arrays because 16 YMM registers cannot hold the working set.

| Metric | Value | vs R=4 | vs R=8 |
|--------|-------|--------|--------|
| **Memory Bound** | **74.0%** | 1.0% | 1.4% |
| **Store Bound** | **39.3%** of clockticks | 0.0% | 0.0% |
| **Store Latency** | **34.2%** of clockticks | 2.7% | — |
| **DTLB Store Overhead** | **23.9%** of clockticks | 0.1% | 0.1% |
| Store STLB Hit | 23.9% | 0.1% | 0.1% |
| **L1 Bound** | **30.3%** of clockticks | 0.4% | 0.9% |
| L1 Latency Dependency | 14.3% | 61.6% | 36.8% |
| **L2 Bound** | **9.6%** of clockticks | 0.0% | 0.5% |
| L2 Hit Latency | 100.0% | — | — |
| FB Full | 3.3% | — | 3.2% |
| DTLB Load Overhead | 1.5% | 0.1% | 0.0% |
| Split Stores | 5.7% | — | 5.3% |

## Back-End Detail — Core Bound (2.2%)

| Metric | Value | vs R=4 | vs R=8 |
|--------|-------|--------|--------|
| Core Bound | 2.2% | 8.8% | 23.4% |

Core Bound is negligible because the pipeline never gets far enough to hit compute bottlenecks — it's stuck on memory.

## Port Utilization

| Port | Usage | vs R=4 | vs R=8 | Role |
|------|-------|--------|--------|------|
| **Port 0** | **23.2%** | 96.0% | 76.0% | FMA unit 1 |
| **Port 1** | **31.4%** | 91.0% | 80.0% | FMA unit 2 |
| Port 6 | 3.8% | 11.1% | 18.7% | Branch/ALU |
| 3+ Ports Utilized | 23.9% of cycles | 96.5% | 86.2% |
| 2 Ports Utilized | 11.8% | 1.0% | 7.9% |
| **1 Port Utilized** | **18.4%** | 0.6% | 2.9% |
| 0 Ports Utilized | 0.1% | 0.0% | 0.0% |

FMA ports at 23-31% — the hardware could do **4x** more computation but is starved for data.

## Instruction Mix

| Metric | Value | vs R=4 | vs R=8 |
|--------|-------|--------|--------|
| Clockticks | 138,112,000,000 | 142,425,600,000 | 144,982,400,000 |
| Instructions Retired | 182,368,000,000 | 755,318,400,000 | 635,811,200,000 |
| **CPI** | **0.757** | 0.189 | 0.228 |
| IPC | 1.32 | 5.30 | 4.39 |
| Average CPU Frequency | 5.690 GHz | 5.71 GHz | 5.683 GHz |

## Front-End Detail

| Metric | Value |
|--------|-------|
| DSB Coverage | 99.3% |
| LSD Coverage | 0.1% |
| DSB Misses | 3.9% |
| ICache Misses | 0.1% |

Front-end is clean — the bottleneck is entirely back-end.

## Benchmark Result

| Metric | Value | vs R=4 | vs R=8 |
|--------|-------|--------|--------|
| ns/call | 4474.5 | 118.6 | 437.8 |
| CPE | 2.580 | 0.277 | 0.516 |
| GFLOP/s | 12.01 | 73.4 | 50.9 |

---

## Interpretation

### R=16 is memory-bound from register spills (74.0%)

This is a fundamentally different bottleneck from R=4 (compute-efficient) and R=8 (dependency-limited). R=16 requires 32 complex values (64 doubles) for the butterfly, while AVX2 provides only 16 YMM registers. The codelet uses explicit stack spill arrays (`__attribute__((aligned(32))) double spill_re[64]`). The result:

1. **39.3% Store Bound** — stores to the spill buffer dominate execution time
2. **23.9% DTLB Store** — the spill area crosses page boundaries, every store misses L1 TLB and hits STLB
3. **34.2% Store Latency** — stores queue behind each other, creating back-pressure
4. **30.3% L1 Bound** — reloading spilled values from L1 adds latency
5. **9.6% L2 Bound** — some spilled values evict past L1 (64 KB spill area approaches L1 capacity)

### FMA units are starving

Port 0/1 at 23%/31% vs R=4's 96%/91%. The hardware has 4x more compute bandwidth than R=16 can use. The spill traffic creates a memory bottleneck that prevents the FMA units from being fed.

### L1 Latency Dependency paradox

L1 Latency Dependency dropped to 14.3% (vs R=8's 36.8%, R=4's 61.6%). This is not an improvement — it means the pipeline is so dominated by memory stalls that the compute dependency chains (which R=4/R=8 expose) are never the limiting factor. The CPU never gets far enough into the computation to hit dependency bottlenecks.

### R=16 on AVX2 should be avoided

The calibration system should avoid R=16 as a twiddle stage on AVX2. Factoring as R=4×R=4 or R=8×R=2 would be massively faster per-element. R=16 is viable as a first-stage (N1, no twiddles) where register pressure is lower, and on AVX-512 where 32 ZMM registers eliminate all spills.

### R=4 → R=8 → R=16 trend on AVX2

| Metric | R=4 (16 values) | R=8 (16 values) | R=16 (32 values) |
|--------|-----------------|-----------------|-------------------|
| Retiring | 85.9% | 72.2% | 21.8% |
| Bottleneck | Compute (peak) | Dependency chains | **Spill traffic** |
| CPI | 0.189 | 0.228 | 0.757 |
| CPE | 0.277 | 0.516 | 2.580 |
| Port 0/1 | 96%/91% | 76%/80% | 23%/31% |
| Register fit | Exact fit (16) | Tight (16+spills) | Overflow (32 needed) |

The cliff between R=8 and R=16 is dramatic: CPE goes from 0.516 to 2.580 (5x worse), retiring from 72% to 22%. R=8 is the largest radix that fits in AVX2's register file.

---

## Optimizations Applied

Three micro-optimizations were tested on the R=16 t1_dit codelet. Two produced significant wins.

### #1: Defer W16 constants to Pass 2 (applied, -4.4%)

The original codelet loaded 8 constant YMM registers (`sign_flip`, `sqrt2_inv`, 6 x `tw_W16_*`) at function scope. These consumed 8 of 16 YMM during Pass 1's twiddle+DFT-4 phase, where they are never used. Moving them to just before Pass 2 freed registers during Pass 1, reducing compiler-generated spills.

| | ns/call | CPE |
|---|---|---|
| Before | 4474.5 | 2.580 |
| After | 4275.9 | 2.481 |
| Delta | **-4.4%** | |

### #2: Prefetch next column's twiddles during DFT-4 (applied, -63.5%)

The external twiddle loads (`W_re[n*me+m]`) access data at stride `me` = K = 256 doubles = 2KB apart. The hardware prefetcher cannot predict this strided pattern. Adding `_mm_prefetch` for the next column's twiddles during the current column's DFT-4 computation (which is compute-bound, leaving load ports idle) hides the load latency entirely.

| | ns/call | CPE |
|---|---|---|
| Before (#1 only) | 4275.9 | 2.481 |
| After (#1+#2) | 1631.6 | 0.941 |
| Delta | **-63.5%** | |

VTune confirmed the mechanism: L1 Bound dropped 30.3% to 11.6%, DTLB Load dropped 1.5% to 0.4%. Store Bound rose from 39.3% to 53.0% as stores became the new relative bottleneck.

### #3: Skip spill for k1=0 bin-0 values (applied, neutral)

After each Pass 1 column DFT-4, bin 0 is stored directly to `rio` instead of the spill buffer. Pass 2's column k1=0 then loads from the TLB-warm `rio` array. Expected to save 8 STLB lookups per iteration. Result: within noise (~1612 vs 1631 ns). The spill buffer was already L1-hot; the bottleneck is the 16 stride-K rio stores in Pass 2, not spill traffic.

### Post-optimization VTune profile

| Metric | Original | Optimized (#1+#2+#3) |
|--------|----------|---------------------|
| **Retiring** | 21.8% | 22.5% |
| **CPI** | 0.757 | 0.735 |
| **CPE** | 2.580 | **0.941** |
| Back-End Bound | 76.2% | 75.4% |
| **L1 Bound** | 30.3% | **11.6%** |
| **DTLB Load** | 1.5% | **0.4%** |
| L2 Bound | 9.6% | 11.8% |
| **Store Bound** | 39.3% | **53.0%** |
| Store Latency | 34.2% | 34.1% |
| **DTLB Store** | 23.9% | **18.5%** |
| Port 0/1 | 23%/31% | 18%/27% |

The remaining bottleneck is **Store Bound 53.0%** — 16 stride-K stores to `rio` in Pass 2 that cannot be prefetched (no store-prefetch on x86). The DTLB Store at 18.5% is from these rio output stores hitting different pages.

### R=4 -> R=8 -> R=16 trend (updated)

| Metric | R=4 (16 values) | R=8 (16 values) | R=16 original | R=16 optimized |
|--------|-----------------|-----------------|---------------|----------------|
| Retiring | 85.9% | 72.2% | 21.8% | 22.5% |
| Bottleneck | Compute (peak) | Dependency chains | Spill + load stalls | **Store bound** |
| CPI | 0.189 | 0.228 | 0.757 | 0.735 |
| CPE | 0.277 | 0.516 | 2.580 | **0.941** |
| Port 0/1 | 96%/91% | 76%/80% | 23%/31% | 18%/27% |

The prefetch optimization brought R=16 from 5x worse than R=8 to **1.8x worse** — still not competitive per-element, but no longer catastrophic. The calibration system's stage-count reduction benefit may now justify R=16 at more (N, K) pairs.

**Verdict: R=16 on AVX2 improved from 2.58 to 0.94 CPE (2.7x faster) via deferred constants and twiddle prefetch. Still store-bound at 53% due to stride-K output writes. AVX-512 with 32 registers would eliminate both the spill buffer and the register pressure that forces the 2-pass architecture.**
