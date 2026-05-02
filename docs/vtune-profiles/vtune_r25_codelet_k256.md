# R=25 AVX2 Codelet — VTune Analysis (K=256)

Target: `radix25_t1_dit_fwd_avx2` at K=256 (L2-resident).
Hardware: Intel Core i9-14900KF (Raptor Lake), P-core pinned, 5.684 GHz turbo.
Collector: `uarch-exploration` (event-based sampling).

## Baseline performance

| Metric | Value |
|---|---|
| ns/call | 5113.1 |
| CPE (cycles / (R × K)) | 1.927 |
| GFLOP/s | 19.03 |
| Retiring | **50.0%** of pipeline slots |
| CPI Rate | 0.328 |

For comparison, `radix25_n1_fwd_avx2` (no twiddle table) runs at 4662 ns /
CPE 1.757. The `t1_dit`/`n1` gap of 10% is small — smaller than any other
radix we profiled, suggesting the twiddle-load path is well-hidden.

## Pipeline breakdown

| Category | Value |
|---|---|
| **Retiring** | **50.0%** |
| Front-End Bound | 1.9% |
| Bad Speculation | 1.4% |
| Back-End Bound | 46.6% |
| — Memory Bound | **20.1%** |
| — Core Bound | 26.6% |

R=25 is a **hybrid** between R=20's compute-bound profile (Memory 10.3%) and
R=32's memory-bound profile (Memory 46.0%). The radix-5 butterfly dependency
chains dominate, but a non-trivial store-side component is starting to appear.

## Memory Bound breakdown

| Source | % of Clockticks |
|---|---|
| L1 Bound | 2.3% |
| — L1 Latency Dependency | 29.5% |
| — FB Full | 15.8% |
| — DTLB Overhead (loads) | 2.9% |
| L2 Bound | 9.3% |
| L3 Bound | 0.2% |
| DRAM Bound | 0.2% |
| **Store Bound** | **11.2%** |
| — Store Latency | 30.3% |
| — DTLB Store Overhead | **84.3%** of store-cycles |
| — Split Stores | 0.0% |

### New signature: moderate Store Bound

Where R=20's Store Bound was only 1.9% (tiny tail), R=25 reaches 11.2% of
total clockticks with 84.3% of those store-cycles hit by DTLB walks. This is
the same pattern R=32 showed, just smaller in absolute scale:

- R=25 touches 25 output rows × 2 arrays ≈ 50 output pages
- R=32 touches 32 output rows × 2 arrays ≈ 64 output pages
- R=20 touches 20 output rows × 2 arrays ≈ 40 output pages

50 pages is close to the L1 DTLB's ~48-entry store capacity — overflow starts
to appear but isn't catastrophic. Same mechanism as R=32/R=64, first visible
signature in the R=25 radix.

## Core Bound breakdown

| Source | % of Clockticks |
|---|---|
| Port Utilization | 26.1% |
| — 3+ Ports Utilized | 60.6% |
| — Port 0 | 59.5% |
| — Port 1 | 62.0% |
| — Port 6 | 6.7% |
| Load Operation Utilization | 31.6% |
| Store Operation Utilization | 20.1% |
| Vector Capacity Usage (FPU) | 50.0% |

Port 0 and Port 1 are nearly balanced (59.5% / 62.0%), unlike R=20 where
Port 1 dominated (67.4% / 57.5%). Suggests the R=25 DFT-5 butterfly is
placing more symmetric pressure on the FMA units. Still compute-heavy, but
not FMA-bottlenecked.

## Cross-radix comparison (K=256, AVX2 t1_dit_fwd)

| Radix | CPE | Retiring | Memory Bound | Store Bound | Dominant bottleneck |
|---|---|---|---|---|---|
| R=16 post | 0.94 | ~25% | ~30% | 53% | store bound |
| R=20 | 1.17 | 53.8% | 10.3% | 1.9% | FMA deps (R=5) |
| **R=25** | **1.93** | **50.0%** | **20.1%** | **11.2%** | **FMA deps + emerging store DTLB** |
| R=32 post | 1.97 | 34.4% | 46.0% | 37.2% | DTLB store 66% |
| R=64 | 3.07 | 27.2% | 57.5% | 38.7% | stacked memory |

R=25 sits between R=20 (pure compute) and R=32 (clearly memory-bound).

## Optimization results

### Optimization #1: Deferred W25 twiddle broadcasts (reverted, mild regression)

**Change**: Moved the 18 hoisted W25 broadcasts (9 indices × re/im) from
function scope to the start of PASS 2, matching the pattern that gave R=32 a
21.5% win.

**Result** (K=256):

| Codelet | Baseline | + Deferred | Δ |
|---|---|---|---|
| n1_fwd (unchanged code) | 4662 | 4593 | -1.5% (variance) |
| n1_bwd (unchanged code) | 4599 | 4655 | +1.2% (variance) |
| **t1_dit_fwd** | **5113** | **5220** | **+2.1% slower** |

Variance on the unchanged `n1` runs is ±1.5%. The `t1_dit_fwd` change is
+2.1% — just above noise, trending mildly negative rather than null. No net
positive.

**Why no win**: R=25 is not register-pressure-limited the way R=32 is. R=25's
18 hoisted broadcasts + 5 DFT constants + 10 working registers = 33 live
`__m256d` at peak. Compiler handles this acceptably because PASS 1's DFT-5
butterflies don't need all 10 working slots simultaneously — scheduling gives
the compiler room.

Contrast with R=32, where 22 broadcasts + 16 working DFT-8 registers =
catastrophic overflow with no slack. Moving constants to PASS 2 freed real
spill traffic. R=25's structure doesn't hit that wall.

**Status**: Reverted.

### Optimization #2: Twiddle prefetch (not attempted)

Prediction: regress. L1 Bound is only 2.3% — no load latency to hide — and
the FMA chain has moderate pressure. Same situation as R=20 and R=8, both of
which saw regressions with prefetch. Not worth the cycle cost to confirm.

### Optimization #3: `PREFETCHW` on outputs (not attempted)

Store Bound 11.2% with DTLB capacity issues makes this superficially
attractive. But R=32 tried this with worse capacity pressure and saw -71%.
R=25 has less room for regression (baseline already near ceiling), so the
downside risk outweighs the uncertain upside. Would need huge pages to fix
the DTLB overflow fundamentally.

## Final conclusion

R=25 on AVX2 sits at **CPE 1.93 / 50% retiring** — already efficient, close
to its achievable ceiling for this microarchitecture. The R=32 playbook
doesn't apply (no register-pressure bottleneck). The R=16 playbook doesn't
apply (no load-latency bottleneck). The remaining cost is split between:

- Radix-5 butterfly dependency chains (~27% Core Bound)
- Emerging store DTLB overflow at 50 output pages (~11% Store Bound)

Neither has a clean software fix. Huge pages would address the store-side
component when eventually implemented.

**Baseline locked (unchanged)**: `t1_dit_fwd` 5113 ns / CPE 1.927.

## Heuristic reinforced

The prefetch heuristic holds: R=25 satisfies **neither** "L1 Bound > 15%"
nor "clear memory-bound with load-latency to hide." It's compute-bound with
store-DTLB backpressure, a profile not served by software prefetch on this
hardware.

R=20 and R=25 together demonstrate that the deferred-constants trick from
R=32 does **not** generalize to composite radixes — it only applies when
register pressure is pathological, not merely high.
