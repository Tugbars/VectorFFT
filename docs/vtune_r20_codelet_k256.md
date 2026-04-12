# R=20 AVX2 Codelet — VTune Analysis (K=256)

Target: `radix20_t1_dit_fwd_avx2` at K=256 (L2-resident).
Hardware: Intel Core i9-14900KF (Raptor Lake), P-core pinned, 5.695 GHz turbo.
Collector: `uarch-exploration` (event-based sampling).

## Baseline performance

| Metric | Value |
|---|---|
| ns/call | 2500.6 |
| CPE (cycles / (R × K)) | 1.170 |
| GFLOP/s | 28.66 |
| Retiring | **53.8%** of pipeline slots |
| CPI Rate | 0.306 |

R=20 is a composite radix (4×5). Baseline is **already well-optimized** — CPE
1.17 is close to R=16's post-optimization ceiling (0.94). Prior optimization
work on the odd-radix generators has paid off here.

For comparison, `radix20_n1_fwd_avx2` (no twiddle table) runs at 1834 ns /
CPE 0.858. The `t1_dit`/`n1` gap of 36% comes from twiddle-load work, not
from memory overhead.

## Pipeline breakdown

| Category | Value |
|---|---|
| **Retiring** | **53.8%** |
| Front-End Bound | 1.8% |
| Bad Speculation | 1.5% |
| Back-End Bound | 42.8% |
| — **Memory Bound** | **10.3%** |
| — **Core Bound** | **32.6%** |

This is categorically different from the spill-heavy family. R=20 is
**compute-bound**, not memory-bound. Only 10.3% of pipeline slots are
stalled on memory.

## Why R=20 is different

Structurally R=20 is 4×5 with 80-entry spill buffer per component. That's
small relative to R=32 (128) and R=64 (256). Working set fits comfortably
in L1, so memory pressure stays low. The remaining cost is **radix-5
butterfly dependency chains** — similar to R=8's radix-8 compute-bound
signature.

## Memory Bound breakdown

| Source | % of Clockticks |
|---|---|
| L1 Bound | 0.7% |
| — L1 Latency Dependency | 34.2% |
| — FB Full | 16.4% |
| — DTLB Overhead (loads) | 6.2% |
| — Split Loads | 0.3% |
| L2 Bound | 9.6% |
| L3 Bound | 0.4% |
| DRAM Bound | 0.2% |
| Store Bound | 1.9% |
| — Split Stores | 13.6% of store-cycles |
| — DTLB Store Overhead | 69.8% of store-cycles |

Store Bound is only 1.9% of total clockticks, so the 69.8% DTLB-store-overhead
figure is relative to that small 1.9% — negligible in absolute terms. Memory
is fundamentally not the bottleneck here.

**L1 Latency Dependency 34.2%** — this is NOT cache misses. It is the
pipeline waiting for results from L1-served loads that feed the next
instruction in a chain. In this codelet it signals the radix-5 butterfly's
FMA chain length rather than any memory issue.

## Core Bound breakdown

| Source | % of Clockticks |
|---|---|
| Port Utilization | 26.2% |
| — 0 Ports Utilized | 0.0% |
| — 1 Port Utilized | 10.5% |
| — 2 Ports Utilized | 13.7% |
| — **3+ Ports Utilized** | **62.9%** |
| ALU Operation Utilization | 38.1% |
| — Port 0 | 57.5% |
| — **Port 1** | **67.4%** |
| — Port 6 | 8.6% |
| Load Operation Utilization | 35.9% |
| Store Operation Utilization | 23.3% |
| Vector Capacity Usage (FPU) | 50.0% |

High port utilization (62.9% of cycles executing 3+ uops) indicates the OOO
engine is successfully extracting parallelism. Port 1 at 67.4% is the FMA
throughput ceiling — the radix-5 butterfly is saturating the FMA unit.

## Cross-radix comparison (K=256, AVX2 t1_dit_fwd)

| Radix | CPE | Retiring | Dominant bottleneck | Prefetch outcome |
|---|---|---|---|---|
| R=4 | 0.277 | 85.9% | — (peak) | n/a |
| R=8 | 0.516 | 72.2% | FMA deps (R=8 butterfly) | **-15% (hurts)** |
| R=16 post | 0.94 | ~25% | store bound | **+2.7× (helps)** |
| **R=20** | **1.17** | **53.8%** | **FMA deps (R=5 butterfly)** | **not tried — prefetch predicted to hurt** |
| R=32 post | 1.97 | 34.4% | DTLB store 66% | -10% (hurts) |
| R=64 | 3.07 | 27.2% | stacked memory | -4% (hurts) |

## Optimization results

### Optimization #1: Deferred W20 twiddle broadcasts (tested, no signal)

**Change attempted**: Moved the 16 hoisted W20 broadcasts (8 indices ×
re/im) from function scope to the start of PASS 2, matching the pattern
that gave R=32 a 21.5% win.

**Result** (K=256, three independent runs of the full R=20 bench):

| Run | Config | n1_fwd | t1_dit_fwd |
|---|---|---|---|
| 1 | baseline | 1834 | 2501 |
| 2 | deferred-constants | 1802 | 2440 |
| 3 | baseline (reverted) | 1805 | 2463 |

Run-to-run variance on the **unchanged** `n1_fwd` is ~2% (1834 → 1802 → 1805).
The apparent `t1_dit_fwd` improvement from deferred-constants (2501 → 2440,
-2.4%) sits within that noise band. No real signal above variance.

**Why no signal**: R=20 is compute-bound (Memory Bound 10.3%, Core Bound
32.6%). The 16 hoisted broadcasts weren't causing meaningful spill pressure
in the first place — the compiler had room to manage them without stack
spills. Moving them to PASS 2 doesn't free anything actually contested.

Contrast with R=32, where 22 hoisted broadcasts + 2 other constants + 16
working registers catastrophically overflowed 16 YMM, causing massive spill
traffic during PASS 1. R=20 just doesn't have that pressure.

**Status**: Reverted. No change committed.

### Optimization #2: Twiddle prefetch (not attempted)

**Prediction**: Would regress, per the heuristic established across R=8/16/32/64.

Criteria for prefetch to help:
- L1 Bound > ~15%
- Front-end has budget (Retiring < ~70%)

R=20 satisfies neither strongly:
- L1 Bound is only **0.7%** → no load latency to hide
- Retiring is 53.8% → moderate FE budget, but prefetch instructions would
  compete with a heavily-utilized FMA chain (Port 1 at 67.4%)

Expected outcome: small regression, similar to R=8's -15% scenario. Not worth
the cycle cost to confirm. Heuristic is well-validated at this point.

## Final conclusion

R=20 on AVX2 is **already near its achievable ceiling** for this
microarchitecture. The 4×5 decomposition keeps the working set small enough
to fit in L1 (10.3% Memory Bound), and the radix-5 butterfly's dependency
chains are what limit further CPE improvement.

The R=32 playbook (deferred constants) doesn't apply because there is no
register-pressure problem to solve. Prefetch is predicted to regress
because load latency is already hidden and the FMA chain is saturated.

**Baseline locked (unchanged)**: `t1_dit_fwd` 2500.6 ns / CPE 1.170.

## Heuristic reinforced

The [prefetch heuristic](../../../.claude/projects/c--Users-Tugbars-Desktop-highSpeedFFT/memory/prefetch_heuristic.md)
from the broader investigation holds: software twiddle prefetch helps only
when **L1 Bound > ~15%** AND there is front-end budget. R=20's L1 Bound of
0.7% firmly places it outside the useful zone, alongside R=8.

R=16 remains the sole codelet where prefetch has paid off on this hardware.
No expansion of the prefetch auto-tuning framework is justified by R=20.
