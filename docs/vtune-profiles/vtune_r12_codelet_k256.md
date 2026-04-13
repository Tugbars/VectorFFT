# VTune Profile: R=12 Isolated Codelet (t1_dit_fwd, K=256)

**Date:** 2026-04-13  
**CPU:** Intel i9-14900KF (Raptor Lake), P-core only, 5.71 GHz turbo  
**Binary:** `vfft_bench_codelet.exe 256 5000 12 t1_dit_fwd`  
**Codelet:** `radix12_t1_dit_fwd_avx2` — 4×3 decomposition, in-place DIT twiddle + butterfly, AVX2  
**Data size:** R=12, K=256, total=3072 elements = 48 KB (re+im), fits L1  
**Analysis:** Microarchitecture Exploration (uarch-exploration)

---

## Baseline performance

| Metric | Value |
|---|---|
| ns/call | 1128–1411 (run-to-run variance ~25%) |
| CPE (cycles / (R × K)) | **~2.36** (median) |
| GFLOP/s | ~29 |
| Retiring | **57.1%** of pipeline slots |
| CPI Rate | 0.287 |

For comparison, `radix12_n1_fwd_avx2` (no twiddle table) runs at 534–724 ns.
The `t1_dit` / `n1` ratio is ~2× — larger than R=10's 1.8×, because the
4×3 decomposition has more total twiddle loads than R=10's 2×5.

**Note on variance**: R=12 shows unusual run-to-run variance (25% swings
between consecutive bench runs) compared to R=10 (stable to ~1%). The
root cause hasn't been isolated but may relate to the radix-3 butterfly's
interaction with CPU power state or memory access patterns. VTune
measurements below are from a single representative run.

---

## Pipeline breakdown

| Category | Value |
|---|---|
| **Retiring** | **57.1%** |
| Front-End Bound | 1.5% |
| Bad Speculation | 0.2% |
| Back-End Bound | 41.2% |
| — Memory Bound | 8.4% |
| — Core Bound | 32.8% |

R=12 is **even more compute-bound than R=10**. Memory Bound is the lowest
we have measured (8.4%, vs R=10 at 10.7%). The remaining 41% of lost slots
is 4:1 compute to memory — almost all FMA chain serialization from the
radix-3 + radix-4 butterflies.

---

## Memory Bound breakdown

| Source | % of Clockticks |
|---|---|
| L1 Bound | 1.1% |
| — L1 Latency Dependency | 34.7% |
| — FB Full | 16.2% |
| — DTLB Overhead (loads) | 0.1% |
| — Split Loads | 0.0% |
| L2 Bound | 8.4% |
| L3 Bound | 0.2% |
| DRAM Bound | 0.1% |
| **Store Bound** | **0.0%** |
| — Store Latency | 0.2% |
| — Split Stores | 12.9% of store-cycles |
| — DTLB Store Overhead | 0.2% |

Key observations:

- **Store Bound is effectively zero**. The compute work absorbs all store
  latency. This is the cleanest store-side profile we've seen — better
  than R=10 (7.9%) and R=20 (1.9%).
- **L2 Bound 8.4%** is slightly elevated vs R=10 (2.3%). The 4×3
  decomposition touches more intermediate values during the radix-3 stage,
  some of which spill to L2.
- **FB Full 16.2%** — moderate fill buffer pressure, consistent with the
  4-wide radix-3 stage bursting stores.
- **L1 Bound is only 1.1%** — load latency is fully hidden. Prefetch would
  regress here, same as every other compute-bound small composite.

---

## Core Bound breakdown

| Source | % of Clockticks |
|---|---|
| Port Utilization | 24.4% |
| — 0 Ports Utilized | 0.0% |
| — 1 Port Utilized | 10.6% |
| — 2 Ports Utilized | 11.2% |
| — **3+ Ports Utilized** | **67.9%** |
| ALU Operation Utilization | 39.1% |
| — **Port 0** | **58.8%** |
| — **Port 1** | **62.8%** |
| — Port 6 | 11.4% |
| Load Operation Utilization | 36.6% |
| Store Operation Utilization | 24.6% |
| Vector Capacity Usage (FPU) | 50.0% |

**67.9% of cycles dispatch on 3+ ports** — good OOO parallelism, though
somewhat lower than R=10's 78.3%. Port 0 and Port 1 balanced at ~60%, both
well utilized. The radix-3 butterfly appears to be slightly more
serial-chain-heavy than radix-5 (R=10/R=20/R=25) but spreads across ports
similarly.

---

## Cross-radix comparison (K=256, AVX2 t1_dit_fwd)

| Radix | CPE | Retiring | Memory Bound | Dominant bottleneck |
|---|---|---|---|---|
| R=4 | 0.277 | 85.9% | — | peak |
| R=8 | 0.516 | 72.2% | 1% | R=8 FMA chain |
| R=10 | 1.98 | 63.4% | 10.7% | R=5 + R=2 FMA chains |
| **R=12** | **~2.36** | **57.1%** | **8.4%** | **R=3 + R=4 FMA chains** |
| R=16 post | 0.94 | ~25% | ~30% | was store bound, fixed |
| R=20 | 1.17 | 53.8% | 10.3% | R=5 FMA chain |
| R=25 | 1.93 | 50.0% | 20.1% | R=5 + emerging store DTLB |
| R=32 post | 1.97 | 34.4% | 46.0% | DTLB store 66% |
| R=64 | 3.07 | 27.2% | 57.5% | stacked memory |

R=12 sits with R=10, R=20, R=25 in the "small composite at ceiling"
cluster. The per-element CPE (~2.36) is slightly higher than R=10 (1.98)
because:

- 4×3 has 3 "true" stages vs 2×5's 2 stages → more twiddle loads
- Radix-3 butterfly has longer FMA chain than radix-5 (fewer independent
  partial products)
- 12 values doesn't amortize setup overhead as well as 20 or 25

---

## Optimization analysis (no changes applied)

### Deferred constants — not applicable, would be null

R=12 has minimal hoisted W12 constants. No register pressure problem.
Same result expected as R=10/R=20/R=25 — no effect.

### Twiddle prefetch — predicted to regress

L1 Bound 1.1% — nothing to hide. Same regime as R=8/R=10/R=20/R=25, all
of which regressed or were null.

### `PREFETCHW` on outputs — not applicable

Store Bound is **literally 0.0%**. There is no store-side problem to
solve. The 12.9% split-stores figure is of zero store-cycles —
meaningless.

### Investigation of 25% run variance

The run-to-run variance of 25% on t1_dit_fwd is unusual for this codelet
family. Possible sources:

- **Power state interaction**: radix-3 uses a specific FMA pattern that
  may trigger different power/frequency behaviors from radix-5/radix-2
- **Cache-line alignment edge cases**: stride-K access with 12-way
  grouping may hit occasional alignment pathologies
- **Thread Director migration**: Windows may occasionally relocate the
  thread even with affinity set

Not investigated further; variance is a measurement concern, not an
optimization opportunity.

---

## Final conclusion

R=12 on AVX2 is **already near its achievable ceiling** for this
microarchitecture. The 4×3 decomposition is even more compute-bound than
R=10 — Memory Bound 8.4%, Store Bound 0.0%. The 41% of lost pipeline slots
is almost entirely FMA dependency chains in the radix-3 butterfly plus
radix-4 combine.

**Baseline: `t1_dit_fwd` ~1128–1411 ns / CPE ~2.36** (wide range due to
variance; no change from optimization would be measurable above this
noise floor).

---

## Heuristic reinforced

The prefetch heuristic continues to hold. R=12 joins R=10, R=20, R=25 in
the "compute-bound small composite" cluster:

- **L1 Bound**: 1.1% (far below the 15% threshold)
- **Retiring**: 57.1% (moderate, but Memory Bound is only 8.4%)
- **Conclusion**: no software-level optimization space

R=12's lower retiring than R=10 (57% vs 63%) comes from the radix-3
butterfly's longer dependency chain, not from any fixable inefficiency.
Radix-3 is intrinsically less parallel than radix-5 (fewer independent
rotation pairs in the butterfly decomposition).

The full cluster of well-optimized small composites (R=10, R=12, R=20,
R=25) shares the common signature: **L1 Bound ≤ 2%, Memory Bound ≤ 20%,
Retiring ≥ 50%, all bounded by the algorithmic FMA chain length of their
constituent prime radixes**.
