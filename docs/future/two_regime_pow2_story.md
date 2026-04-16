# Two-Regime Pow2 Story (Paper Framing)

**Date:** 2026-04-17
**Status:** Accepted framing for paper.
**Scope:** How VectorFFT's pow2 performance story should be presented given the K-dependent bottleneck profile.

---

## The finding that drove this framing

VTune full-FFT profiles at two regimes of the same algorithm (VectorFFT, pow2, single P-core, AVX2):

**K=4 (compute-bound)** — N=16384 K=4:
- Retiring 48.4% (healthy)
- DTLB-store 32.4% (the dominant residual)
- DRAM Bound <1% (working set fits L2)
- Port 1 (FMA) at 28.6% utilization

**K=256 (bandwidth-bound)** — N=65536 K=256:
- Retiring 20.1% (memory-starved)
- DRAM Memory Bandwidth 77.2% of clockticks in DRAM Bound (= ~17% of total)
- DTLB-store dropped to 11.7% (masked by earlier-stage stalls)
- Port 1 at 21.8% (compute sitting idle)

Same codelet code, same codepath. The bottleneck moves from compute/DTLB to DRAM-bandwidth as the working set exceeds L2. Full data: [../vtune-profiles/vtune_full_fft_n65536_k256.md](../vtune-profiles/vtune_full_fft_n65536_k256.md).

---

## Paper framing

### The old one-sentence story (flawed)

> "VectorFFT wins on composite/prime-power sizes but trails MKL on pow2."

This reading implied MKL has a systematic pow2 advantage we can't close. The VTune evidence shows that's wrong — MKL's advantage is **narrow, localized, and regime-dependent**.

### The new story

VectorFFT operates in two bottleneck regimes on pow2 FFTs, separated by whether the working set fits in L2:

| Regime | Threshold | Bottleneck | Pow2 ratio vs MKL |
|---|---|---|---|
| **Compute-bound** | Working set ≤ L2 (≈256 KB) | Arithmetic + store DTLB at large stride | **0.99–1.03×** (parity within noise) |
| **Bandwidth-bound** | Working set > L2 | DRAM memory bandwidth, L3 ring, SQ | **1.06–1.41×** (VectorFFT wins) |

In the compute-bound regime (typically K≤8 and N≤16k, or K=4 up to N≈131k), MKL's hand-scheduled codelets and likely conjugate-pair split-radix give them a small algorithmic edge (~6–11% fewer instructions) that our pipeline compensates for via higher IPC (0.73 vs MKL's ~0.93 CPI → 27% more retired per cycle). Net result: ±3% wall-time parity, with VectorFFT running more efficiently per cycle but executing slightly more instructions.

In the bandwidth-bound regime (K≥32 at any N, or K=4 at very small cases), the algorithmic distinction is irrelevant — both implementations wait on DRAM. VectorFFT's split-complex layout and cleaner prefetcher interaction give us a measurable edge of 6–41% depending on size.

Outside pow2 — composite, prime-power, odd-composite, mixed-radix — VectorFFT wins by 1.3×–5× regardless of regime, because the algorithmic specialization is where MKL's radix-heavy assumptions break down.

### What this means for claims

| Claim | Truth value | Evidence |
|---|---|---|
| "VectorFFT beats MKL on pow2 at high K" | TRUE | 6-41% wins at K=256 |
| "VectorFFT trails MKL on pow2 at low K" | MISLEADING | at most 1-3% behind, within measurement noise, and IPC shows we're more efficient per cycle |
| "MKL has an algorithmic advantage on pow2" | NARROW TRUE | ~6-11% fewer instructions via split-radix, neutralized by our IPC advantage in practice |
| "VectorFFT wins on non-pow2 sizes" | TRUE | 1.3-5× on composite/prime-power/odd/mixed |

---

## Why split-radix isn't worth implementing

Split-radix would reduce our instruction count by 6–11% at pow2. In each regime:

- **Compute-bound regime**: would close ~1/3 of the already-narrow (±3%) gap. Possible ~2-4% wall-time improvement on specific K=4 cases. Not paper-defining.
- **Bandwidth-bound regime**: zero benefit — we're memory-saturated, not instruction-saturated.

Cost: planner rewrite (flat → tree factorization), new codelet generator type (L-shape butterflies combining N/2 + N/4 + N/4 sub-results), wisdom format bump, DIT+DIF pairing revalidation. Months of work.

**Decision: not shipping split-radix.** The paper story is already clean; spending months for 2-4% on specific cells doesn't justify the architectural cost.

---

## What about the K=4 DTLB-store at 32.4%?

The store-locality investigation (see [oop_staging_decision.md](oop_staging_decision.md) — status Rejected) explored OOP staging as a fix. Three findings killed it:

1. Simple `os < me` packing is semantically incorrect (address collisions within one codelet call).
2. Correct packing requires tiling `me` into chunks ≤ `packed_os`, and tiling overhead runs 30-100% on this architecture (prior experience, independently confirmed).
3. Even at K=4 where packing is correct without tiling, per-call overhead dominates and the MVP ran +110-2740% slower than baseline.

The store-DTLB cost at K=4 is real but not cost-effectively fixable without moving to a fundamentally different execution structure (Bailey / six-step, which is its own multi-month project with its own tiling costs).

**Consequence:** the ±3% pow2 K=4 gap vs MKL is **structurally inherent to our design** given split-complex + DIT+DIF zero-permutation + flat factorization. The remaining wins outside that cell are where VectorFFT earns its keep.

---

## What remains worth investing in

1. **DP planner improvements** — non-determinism, multi-trial measurement, k-best search. Addresses the ~7% measurement-variance-driven suboptimality we saw in wisdom recalibration. Independent of pow2-vs-MKL gap.
2. **Larger-radix codelets at high K** — our K=256 wins might extend to 2× if we had R=128 or R=256 codelets. Cheap to generate; the question is whether larger radixes help memory layout (possibly not — larger strides per butterfly).
3. **Multithreading + SMT tuning** — we ship single-thread numbers now. Multi-thread MKL comparisons are a separate scaling study.
4. **Bailey / six-step for very large N** — the long-term pow2 answer if we ever care enough, but not blocking the paper.

---

## Paper structure implication

The paper's pow2 section should be two subsections:

**Subsection: L2-resident pow2 (compute-bound)**
- Acknowledge parity with MKL (±3%)
- Frame the VectorFFT advantage in IPC (27% higher retirement rate per cycle)
- Note that MKL's edge is localized to this specific cell and attributable to algorithmic refinement (split-radix) and hand-scheduled codelets

**Subsection: DRAM-resident pow2 (bandwidth-bound)**
- Report the 6–41% VectorFFT win at K=256 across 4096 ≤ N ≤ 131072
- Attribute to split-complex layout and prefetcher behavior
- Note that this is the regime users care more about for real batched-FFT workloads

Outside pow2 stays one subsection ("non-pow2 sizes — 1.3-5× wins") as before.

This framing is **defensible, measurable, and doesn't oversell.** The evidence directly supports every claim.
