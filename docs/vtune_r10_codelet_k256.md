# VTune Profile: R=10 Isolated Codelet (t1_dit_fwd, K=256)

**Date:** 2026-04-13  
**CPU:** Intel i9-14900KF (Raptor Lake), P-core only, 5.69 GHz turbo  
**Binary:** `vfft_bench_codelet.exe 256 5000 10 t1_dit_fwd`  
**Codelet:** `radix10_t1_dit_fwd_avx2` — 2×5 decomposition, in-place DIT twiddle + butterfly, AVX2  
**Data size:** R=10, K=256, total=2560 elements = 40 KB (re+im), fits L1  
**Analysis:** Microarchitecture Exploration (uarch-exploration)

---

## Baseline performance

| Metric | Value |
|---|---|
| ns/call | 892 |
| CPE (cycles / (R × K)) | **1.98** |
| GFLOP/s | 37 |
| Retiring | **63.4%** of pipeline slots |
| CPI Rate | 0.259 |

For comparison, `radix10_n1_fwd_avx2` (no twiddle table) runs at 488 ns /
CPE 1.08. The 45% gap between `t1_dit` and `n1` comes from the stride-K
twiddle loads required by the two-stage decomposition.

---

## Pipeline breakdown

| Category | Value |
|---|---|
| **Retiring** | **63.4%** |
| Front-End Bound | 1.2% |
| Bad Speculation | 0.1% |
| Back-End Bound | 35.2% |
| — Memory Bound | 10.7% |
| — Core Bound | 24.5% |

High retiring, minimal front-end or speculation cost. The remaining 37% of
lost slots split roughly 2:1 between compute-chain waits and memory. R=10
sits in the "small composite, well optimized" regime alongside R=20 and
R=25 — bottleneck is FMA dependency chains in the radix-5 butterfly plus
radix-2 combine.

---

## Memory Bound breakdown

| Source | % of Clockticks |
|---|---|
| L1 Bound | 1.4% |
| — L1 Latency Dependency | 39.0% |
| — FB Full | 11.2% |
| — DTLB Overhead (loads) | 0.1% |
| — Split Loads | 0.0% |
| L2 Bound | 2.3% |
| L3 Bound | 0.1% |
| DRAM Bound | 0.0% |
| Store Bound | 7.9% |
| — Store Latency | 7.2% |
| — Split Stores | 12.3% of store-cycles |
| — DTLB Store Overhead | 0.1% |

Key observations:

- **L1 Bound is only 1.4%** — load latency is fully hidden by the OOO engine.
  No software prefetch would help here; the prefetch heuristic predicts
  regression.
- **DTLB overhead is effectively zero** for both loads and stores — the
  10-row output × 2 arrays ≈ 20 pages working set fits comfortably in the
  L1 DTLB.
- **L1 Latency Dependency at 39.0%** indicates FMA dependency chains (not
  cache misses). Source: the radix-5 butterfly's internal chain combined
  with the radix-2 outer combine.
- **Split Stores 12.3% of store-cycles** — some stores cross 4 KB page
  boundaries at stride K=256 = 2 KB. At only 7.9% total Store Bound this
  is not a meaningful contributor.

---

## Core Bound breakdown

| Source | % of Clockticks |
|---|---|
| Port Utilization | 24.5% |
| — 0 Ports Utilized | 0.0% |
| — 1 Port Utilized | 6.6% |
| — 2 Ports Utilized | 11.8% |
| — **3+ Ports Utilized** | **78.3%** |
| ALU Operation Utilization | 42.1% |
| — **Port 0** | **64.7%** |
| — **Port 1** | **66.8%** |
| — Port 6 | 11.3% |
| Load Operation Utilization | 40.0% |
| Store Operation Utilization | 25.0% |
| Vector Capacity Usage (FPU) | 50.0% |

**Excellent OOO parallelism**: 78.3% of cycles dispatch uops on 3 or more
ports simultaneously. Both FMA ports (0 and 1) at ~65%, indicating balanced
pressure from radix-5 and radix-2 butterfly work.

---

## Cross-radix comparison (K=256, AVX2 t1_dit_fwd)

| Radix | CPE | Retiring | Memory Bound | Dominant bottleneck |
|---|---|---|---|---|
| R=4 | 0.277 | 85.9% | — | peak |
| R=8 | 0.516 | 72.2% | 1% | R=8 FMA chain |
| **R=10** | **1.98** | **63.4%** | **10.7%** | **R=5 + R=2 FMA chains** |
| R=16 post | 0.94 | ~25% | ~30% | was store bound, fixed |
| R=20 | 1.17 | 53.8% | 10.3% | R=5 FMA chain |
| R=25 | 1.93 | 50.0% | 20.1% | R=5 + emerging store DTLB |
| R=32 post | 1.97 | 34.4% | 46.0% | DTLB store 66% |
| R=64 | 3.07 | 27.2% | 57.5% | stacked memory |

R=10 slots in cleanly:

- Retiring **above** R=20 and R=25 (63% vs 54%/50%)
- Memory Bound **similar** to R=20 (10.7% vs 10.3%)
- CPE per element **higher** than R=20 (1.98 vs 1.17) because the twiddle-load
  per-element overhead of a 2-stage decomposition dominates over the DFT-5
  work when there are only 10 elements to amortize across
- Clearly compute-bound, not memory-bound

---

## Optimization analysis (no changes applied)

### Deferred constants — not applicable, would be null

R=10 has minimal hoisted constants (smaller DFT-5 butterfly than R=20/R=25,
which themselves showed null signal from deferred constants). No register
pressure problem to solve.

### Twiddle prefetch — predicted to regress

L1 Bound is 1.4% — no load latency to hide. Same situation as R=8, R=20,
R=25, all of which regressed with prefetch. Not attempted.

### `PREFETCHW` on outputs — not applicable

Store Bound is 7.9% total, DTLB Store Overhead 0.1%. No store-side capacity
problem to address.

### Split Stores 12.3% — investigation not worthwhile

The 12.3% of store-cycles that split across 4 KB pages at K=256 stride is a
small fraction of a small bucket (Store Bound is only 7.9% overall). Even
eliminating all splits would recover < 1% of total runtime. Not worth the
layout restructuring.

---

## Final conclusion

R=10 on AVX2 is **already near its achievable ceiling** for this
microarchitecture. The 2×5 decomposition keeps the working set small (no
memory pressure) and the OOO engine extracts excellent parallelism across
three ports. The remaining 37% of lost pipeline slots are intrinsic FMA
dependency chains in the radix-5 and radix-2 butterflies.

**Baseline locked (unchanged)**: `t1_dit_fwd` 892 ns / CPE 1.98.

---

## Heuristic reinforced

The prefetch heuristic continues to hold across composite radixes:

- **L1 Bound > ~15% + front-end budget**: prefetch helps (confirmed on R=16 only)
- **L1 Bound < ~5% + high retiring**: prefetch hurts (confirmed on R=8, R=20, R=25)
- **R=10**: L1 Bound 1.4%, retiring 63.4% — squarely in the "do not
  prefetch" regime

R=10, R=20, R=25 together form a cluster of "well-optimized small composite
radixes at ceiling." None benefit from prefetch, deferred constants, or
blocked executor. Their performance is bounded by DFT-5 FMA chain length —
intrinsic to the algorithm, not fixable by instruction-level tuning.
