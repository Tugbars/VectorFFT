# VTune Profile: R=8 Isolated Codelet (t1_dit_fwd, K=256)

**Date:** 2026-04-11  
**CPU:** Intel i9-14900KF (Raptor Lake), P-core only, 5.68 GHz turbo  
**Binary:** `vfft_bench_codelet.exe 256 5000 8 t1_dit_fwd`  
**Codelet:** `radix8_t1_dit_fwd_avx2` — in-place DIT twiddle + butterfly, AVX2  
**Data size:** R=8, K=256, total=2048 elements = 32 KB (re+im), fits L1  
**Analysis:** Microarchitecture Exploration (uarch-exploration)

---

## Top-Level Breakdown

| Category | % Pipeline Slots | vs R=4 |
|----------|-----------------|--------|
| **Retiring** | **72.2%** | 85.9% |
| Front-End Bound | 2.8% | 2.3% |
| Bad Speculation | 0.2% | 2.0% |
| **Back-End Bound** | **24.7%** | 9.8% |

## Retiring Detail

| Metric | Value | vs R=4 |
|--------|-------|--------|
| Light Operations | 72.2% | 85.8% |
| FP Arithmetic | 58.8% of uOps | 62.9% |
| FP Vector (256-bit) | 59.2% of uOps | 62.5% |
| FP Scalar | 0.0% | 0.0% |
| Memory Operations | 30.1% | 38.9% |
| Heavy Operations | 0.0% | 0.1% |

## Back-End Detail

| Metric | Value | vs R=4 |
|--------|-------|--------|
| Memory Bound | 1.4% | 1.0% |
| **Core Bound** | **23.4%** | 8.8% |
| L1 Bound | 0.9% | 0.4% |
| L2 Bound | 0.5% | 0.0% |
| L3 Bound | 0.1% | 0.1% |
| DRAM Bound | 0.1% | 0.1% |
| Store Bound | 0.0% | 0.0% |
| **L1 Latency Dependency** | **36.8%** of clockticks | 61.6% |
| DTLB Load Overhead | 0.0% | 0.1% |
| DTLB Store Overhead | 0.1% | 0.1% |
| Split Stores | 5.3% | — |
| FB Full | 3.2% | — |

## Port Utilization

| Port | Usage | vs R=4 | Role |
|------|-------|--------|------|
| **Port 0** | **76.0%** | 96.0% | FMA unit 1 |
| **Port 1** | **80.0%** | 91.0% | FMA unit 2 |
| Port 6 | 18.7% | 11.1% | Branch/ALU |
| 3+ Ports Utilized | 86.2% of cycles | 96.5% |
| 2 Ports Utilized | 7.9% | 1.0% |
| 1 Port Utilized | 2.9% | 0.6% |
| 0 Ports Utilized | 0.0% | 0.0% |

## Instruction Mix

| Metric | Value | vs R=4 |
|--------|-------|--------|
| Clockticks | 144,982,400,000 | 142,425,600,000 |
| Instructions Retired | 635,811,200,000 | 755,318,400,000 |
| **CPI** | **0.228** | 0.189 |
| IPC | 4.39 | 5.30 |
| Vector Capacity Usage (FPU) | 50.0% | 50.0% |
| Average CPU Frequency | 5.683 GHz | 5.71 GHz |

## Front-End Detail

| Metric | Value | vs R=4 |
|--------|-------|--------|
| **DSB Coverage** | **99.9%** | 31.0% |
| LSD Coverage | 0.0% | 67.8% |
| DSB Misses | 8.9% | 12.7% |
| ICache Misses | 0.0% | 0.1% |
| Branch Resteers | 1.0% | 0.5% |

## Benchmark Result

| Metric | Value | vs R=4 |
|--------|-------|--------|
| ns/call | 437.8 | 118.6 |
| CPE (cycles per element) | 0.516 | 0.277 |
| GFLOP/s | 50.9 | 73.4 |

---

## Interpretation

### Core Bound is the bottleneck (23.4% vs R=4's 8.8%)

R=8 uses a DFT-8 = 2 x DFT-4 + W8 combine butterfly, which creates deeper dependency chains than R=4. The OOO engine cannot fully overlap dependent operations — the FMA ports sit at 76%/80% (vs R=4's 96%/91%) because they are starving for independent work.

### L1 Latency Dependency: 36.8%

One third of cycles are spent waiting for a result from the previous instruction. This is not from cache misses (L1 hit rate is perfect) — it is from **data dependency chains** in the butterfly. Instruction B needs the result of instruction A, and the pipeline stalls until A completes. The DFT-8 butterfly has a critical path roughly 2x longer than DFT-4.

### DSB vs LSD shift

R=4's loop body fits in the Loop Stream Detector (LSD=68%), bypassing both decode and the DSB entirely. R=8's loop body is too large for the LSD (0%) but fits in the Decoded Stream Buffer (DSB=99.9%). This costs slightly more front-end bandwidth but is still near-optimal — no decode bottleneck.

### DIF is 10% faster than DIT

Timing benchmarks show R=8 DIF consistently outperforms DIT:

| Variant | ns/call | CPE |
|---------|---------|-----|
| t1_dit_fwd | 437.8 | 0.516 |
| t1_dif_fwd | 401.9 | 0.474 |

DIF performs the butterfly first (add/sub only) then applies twiddles at the end. This structure likely has a shorter critical path — the twiddle FMA chains sit at the tail where they overlap with stores rather than feeding into the combine phase.

### Register pressure (attempted optimization)

A fused-half layout was tested: compute even twiddles + even DFT-4 first (freeing x0/x2/x4/x6) before loading odds. Theory: reduce peak live registers from 32+ to 16 (AVX2 limit), eliminating spills. Result: **+5.5% regression** — the non-sequential memory access pattern (0,4,2,6,1,5,3,7) hurt the hardware prefetcher more than register pressure reduction helped. ICX compiler already handles register scheduling well on the sequential layout. Reverted.

### R=8 vs R=4 summary

| Metric | R=4 | R=8 | R=8 is... |
|--------|-----|-----|-----------|
| Retiring | 85.9% | 72.2% | 16% less efficient |
| CPI | 0.189 | 0.228 | 21% worse |
| Core Bound | 8.8% | 23.4% | 2.7x worse |
| Port 0/1 | 96%/91% | 76%/80% | Underutilized |
| FP Vector | 62.5% | 59.2% | Similar |
| Memory Bound | 1.0% | 1.4% | Both clean |
| DTLB | ~0% | ~0% | Both clean |

---

## Optimizations Attempted

### Deferred vc/vnc constants (applied, -2 to -4%)

The W8 combine constants `vc` (√2/2) and `vnc` (-√2/2) were loaded at function scope, consuming 2 of 16 YMM registers throughout the twiddle and butterfly phases where they are unused. Moving them to just before the W8 combine frees 2 registers during the most pressure-heavy phase.

| | Best ns/call | Median ns/call |
|---|---|---|
| Before | 434.9 | 445.8 |
| After | **425.7** | **427.6** |
| Delta | **-2.1%** | **-4.1%** |

### Fused-half layout (reverted, +5.5% regression)

Compute even twiddles + even DFT-4 before loading odds, reducing peak live registers. The non-sequential memory access pattern (0,4,2,6,1,5,3,7) hurt performance more than register pressure reduction helped. ICX already schedules the sequential layout well.

### U=2 interleaving (reverted, +39% regression)

Process two k-blocks per iteration for cross-block ILP. Two R=8 blocks need 32 YMM values — catastrophic spilling on AVX2's 16 registers. Only viable on AVX-512 (32 ZMM registers); a U=2 AVX-512 variant was written and is ready for testing.

### Twiddle prefetch — aggressive (reverted, +15% regression)

14 prefetch instructions at loop start for all 7 twiddle rows. R=8 is 72% retiring with only 1.4% Memory Bound — the OOO engine already overlaps twiddle loads with computation. The prefetch instructions competed for front-end slots and load ports, adding pure overhead.

### Twiddle prefetch — lightweight (reverted, +8% regression)

3 prefetch instructions for tw5-tw7 im components during early multiplies. Still hurt — even 3 extra instructions are measurable when the pipeline is 72% busy with useful work. No idle slots to absorb them.

### Key insight: R=8 is NOT load-bound

The R=16 codelet saw 2.7x speedup from prefetch because it was 30% L1 Bound (loads stalling on cache misses from stride-K access). R=8 is only 0.9% L1 Bound — the OOO engine's 512-entry ROB can see ~6 iterations ahead and naturally overlaps all twiddle loads. Prefetch helps load-bound codelets, hurts compute-bound ones.

**Verdict: R=8 DIT codelet is good (72.2% retiring) but not at peak like R=4 (85.9%). The gap is entirely from deeper dependency chains in the DFT-8 butterfly — algorithmic, not fixable by instruction reordering or prefetch. The applied win is deferred constants (-2 to -4%). The DIF variant is consistently 10% faster than DIT due to shorter critical path.**
