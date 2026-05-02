# VTune Profile: Full FFT N=65536 K=256 — High-K Regime Characterization

**Binary:** `vfft_bench_vtune.exe 65536 256 ours 10`
**Configuration:** VectorFFT, full fwd+bwd roundtrip at N=65536 K=256
**Platform:** i9-14900KF (Raptor Lake), AVX2, single P-core (AFFINITY 1)
**Frequency:** 5.678 GHz (full turbo)
**Elapsed:** 199 s, 68.2 billion clockticks, 86.2 billion instructions retired

---

## Purpose

The prior full-FFT profile at N=16384 K=4 showed compute-bound behavior (48.4% retiring) with DTLB-store as the dominant residual bottleneck (32.4%). This profile at N=65536 K=256 characterizes the **high-K regime** to determine whether the same bottleneck structure holds or whether K=256 FFTs face a fundamentally different constraint.

Working-set comparison:
- N=16384 K=4 → 512 KB (L2-resident)
- N=65536 K=256 → 128 MB (DRAM-resident)

---

## Top-Down Metrics

| Category | Value |
|---|---|
| Retiring | **20.1%** |
| Front-End Bound | 4.1% |
| Bad Speculation | 0.7% |
| Back-End Bound | **75.1%** |
| Memory Bound | **60.2%** |
| Core Bound | 14.9% |

Retiring is less than half the K=4 case. The pipeline is dramatically less productive per cycle — bandwidth-bound rather than compute-bound.

## Memory-Bound Breakdown

| Sub-metric | % of Clockticks |
|---|---|
| L1 Bound | 6.4% |
| — DTLB Load Overhead | 1.3% |
| — L1 Latency Dependency | 8.3% |
| — FB Full | 21.3% |
| L2 Bound | 1.8% |
| L3 Bound | 17.8% |
| — SQ Full | **34.7%** |
| — L3 Latency | 18.7% |
| **DRAM Bound** | **22.2%** |
| — **Memory Bandwidth** | **77.2%** |
| — Memory Latency | 10.2% |
| Store Bound | 17.3% |
| — DTLB Store Overhead | **11.7%** |
| — Store Latency | 18.5% |
| — Split Stores | 2.6% |

Three memory-subsystem pressure points light up:

1. **DRAM Memory Bandwidth at 77.2%** of clockticks inside DRAM Bound — the program is approaching main-memory bandwidth limits.
2. **SQ Full at 34.7%** — the Super Queue (L2→Uncore request buffer) is full, back-pressuring the core.
3. **FB Full at 21.3%** — L1 fill buffers saturated.

The store side remains present but reduced: Store Bound 17.3%, DTLB Store Overhead dropped from 32.4% (K=4) to 11.7% (K=256). The load DTLB overhead practically disappeared (18.2% → 1.3%).

## Compute Characterization

| Metric | Value |
|---|---|
| CPI | 0.791 |
| FP Vector 256-bit | 41.4% of uOps |
| Vector Capacity (FPU) | 50% |
| Port 0 | 16.8% |
| **Port 1 (FMA)** | **21.8%** |
| Port 6 | 10.3% |
| Load utilization | 10.1% |
| Store utilization | 11.8% |
| DSB Coverage | 80.6% |

FMA port utilization dropped from 28.6% (K=4) to 21.8%. The compute engine sits idle waiting on the memory subsystem more often. This is consistent with bandwidth saturation leaving the OOO window underfilled.

---

## Regime Comparison: K=4 vs K=256

| Metric | N=16384 K=4 | N=65536 K=256 | Shift |
|---|---|---|---|
| Retiring | 48.4% | 20.1% | **-28.3 pp** |
| DRAM Bound | 0.5% | 22.2% | **+21.7 pp** |
| DRAM Memory Bandwidth | — | 77.2% | **new bottleneck** |
| SQ Full | 12.0% | 34.7% | **+22.7 pp** |
| DTLB Store Overhead | 32.4% | 11.7% | -20.7 pp |
| DTLB Load Overhead | 18.2% | 1.3% | -16.9 pp |
| Port 1 (FMA) | 28.6% | 21.8% | -6.8 pp |
| Frequency | 5.70 GHz | 5.68 GHz | same |

The bottleneck profile shifts categorically, not incrementally. At K=4 the pipeline is bounded by store-side TLB and L3 contention while running arithmetic at moderate intensity. At K=256 the DRAM path becomes the primary constraint — the core is waiting on memory the majority of cycles, and individual TLB/L3/store issues become secondary.

## Implications

### For the split-radix hypothesis

Split-radix reduces arithmetic operations by 6–11%. In the K=4 compute-bound regime this would nominally translate to ~5–10% wall-clock improvement on our side. In the **K=256 bandwidth-bound regime it translates to ≈0 improvement** — the core already waits on DRAM for the majority of cycles; fewer instructions arrive at the same DRAM wall and wait the same amount of time. Split-radix does not address the K=256 gap in either direction.

Cross-regime summary:

- **K=4 (compute-bound)**: algorithmic/scheduling matters. Split-radix could recover ~7% vs MKL.
- **K=256 (bandwidth-bound)**: algorithmic changes are invisible. Memory-layout and prefetcher-friendly patterns dominate.

### For the OOP-staging hypothesis

Store-DTLB dropped from 32.4% to 11.7% moving from K=4 to K=256 — without any code change. The mechanism: at K=256 the FB/SQ saturates before the store STLB can thrash, so the store side's dominant cost is no longer TLB-miss handling but fill-buffer back-pressure. OOP staging (even if its simpler version weren't semantically broken per our MVP finding) would have targeted a bottleneck that has already moved at K=256.

### For the pow2 gap story in the paper

Our K=256 pow2 results already show a **6% win over MKL at N=65536** (`64x32x32` at 116.6M ns vs MKL's 123.2M ns). That win is attributable to memory-layout advantages (split-complex, cleaner prefetcher behavior) rather than algorithmic edge. At K=4 we sit within ±3% of MKL across large pow2 sizes — compute-bound parity, with the residual gap attributable to MKL's hand-scheduled codelets and (likely) conjugate-pair split-radix arithmetic reduction.

The two-regime picture is a cleaner story than "VectorFFT trails MKL at pow2":

1. **Compute-bound (low K)**: ~0-3% tradeoff range against MKL on pow2. Comfortable win everywhere else.
2. **Bandwidth-bound (high K)**: 6-40% win on pow2. 1.3-5× win elsewhere.
3. **Composite/prime-power/odd-composite**: 2-5× win regardless of K.

MKL's algorithmic advantage is narrow, localized, and disappears when memory dominates.

## Files Referenced

- Prior profile (K=4 regime): [vtune_full_fft_n16384_k4.md](vtune_full_fft_n16384_k4.md)
- OOP staging decision (rejected after MVP): [../future/oop_staging_decision.md](../future/oop_staging_decision.md)
- Paper-level framing: [../future/two_regime_pow2_story.md](../future/two_regime_pow2_story.md) (to be written)
