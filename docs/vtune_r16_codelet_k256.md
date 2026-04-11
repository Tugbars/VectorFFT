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

**Verdict: R=16 on AVX2 is broken by register pressure. 78% of pipeline capacity wasted on spill traffic. The codelet works correctly but at 1/4th the efficiency of R=8. Reserve R=16 for AVX-512.**
