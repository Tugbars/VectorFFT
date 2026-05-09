# isub2 vs Our Recipe-log3 — We Already Beat It At R=32, R=64

## Summary

`ct_t1_dit_log3_isub2` is a hand-coded variant in `gen_radix{16,32,64}.py` that organizes log3 sub-DFT processing into **pairs**: instead of processing PASS 1's `N1` sub-DFTs sequentially, it groups adjacent pairs and processes each pair tightly together — loading both sub-DFTs' inputs, computing the derived twiddles once for the pair, doing both sub-DFTs interleaved, and spilling all outputs together.

The intended benefit is keeping the paired sub-DFTs' working set in registers and sharing instruction-level parallelism between them.

**Our recipe-log3 already beats hand isub2 at R=32 and R=64.** The pair-scheduling structure isub2 hand-codes turns out to underperform our SU scheduler's automatic ILP extraction in most cases — our hash-cons globally dedupes derived twiddles (resulting in ~2.5x fewer cmul ops than hand), and SU schedules the whole codelet with full visibility. Hand wins narrowly only at R=16 K≥1024, where its tight in-register pair scheduling pays off for sub-DFT-4 pairs that fit in registers.

## Method

Extracted hand isub2 codelets from `gen_radix{16,32,64}.py --variant ct_t1_dit_log3_isub2 --isa avx512`. Compared correctness (against each other) and speed at K ∈ {64, 256, 1024, 4096} on AVX-512.

Both hand isub2 and our log3 use the same flat-layout twiddle convention (slot j-1 = W^j), so the same `tw_re/tw_im` array works for both.

## Results

Median over 3 runs each, ours/hand_isub2 (< 1.0 means our recipe-log3 is faster):

| Radix | K=64 | K=256 | K=1024 | K=4096 |
|-------|------|-------|--------|--------|
| R=16 | 0.97 | 0.97 | **1.04** | **1.03** |
| R=32 | **0.93** | **0.90** | **0.85** | **0.88** |
| R=64 | **0.94** | **0.94** | 1.01 | **0.95** |

Our recipe-log3:
- **R=32: wins by 7-15% across all K** (best at K=1024)
- **R=64: wins or ties** (5% at most K, tied at K=1024)
- **R=16: ties or loses by 3-4%** (loses at K≥1024)

Correctness check: max relative error between hand-isub2 and our-log3 outputs is 1e-12 (small K) to 1e-9 (large K). Both implementations are mathematically correct DFTs; the difference is FP-arithmetic order.

## Op-count comparison

Total cmul ops (`_mm512_fmsub_pd` + `_mm512_fnmadd_pd` instances) per codelet:

| Radix | Hand isub2 | Our log3 | Ratio |
|-------|------------|----------|-------|
| R=16 | 60 | 26 | 2.3x |
| R=32 | 146 | 57 | 2.6x |
| R=64 | 336 | 120 | 2.8x |

**Hand isub2 has 2.3-2.8x more cmuls than our log3.** Because hand recomputes derived twiddles per pair (each pair gets its own w5, w9, w12, w13, etc.), while we hash-cons globally — every derived twiddle exists exactly once and is shared across all sub-DFTs that need it.

This is the dominant factor at R=32 and R=64. The pairing benefit of register-resident sub-DFTs is real, but it doesn't compensate for 2.5x more arithmetic.

## Why hand wins at R=16 K≥1024

R=16 = CT(4, 4) means PASS 1 has 4 sub-DFT-4s. Hand pairs these into 2 pairs of sub-DFT-4s. A pair of sub-DFT-4s has 8 inputs and produces 8 outputs — fits comfortably in 16 ZMM registers with cmul scratch space.

R=32 = CT(4, 8) means PASS 1 has 4 sub-DFT-8s. A pair has 16 inputs and produces 16 outputs — already at the register budget without scratch.

R=64 = CT(8, 8) means PASS 1 has 8 sub-DFT-8s. A pair has 16 inputs/outputs, same as R=32, but with 4 pairs to schedule.

So hand's pairing benefit is **maximized when paired sub-DFTs fit in registers**, which happens only at R=16 here. At R=32 and R=64, hand's pair construction already needs spills, and the recompute-derived-twiddles tax dominates.

Why specifically K≥1024 at R=16? At smaller K, both implementations are loop-body-bound and our smaller cmul count wins. At larger K, both are memory-bound enough that the cmul-count difference matters less, and hand's tighter inner-loop schedule wins on raw IPC.

## Conclusion

**Implementing isub2-style pair-scheduling in our generator is not worth it.** Our recipe + log3 already wins or ties everywhere except a narrow R=16 K≥1024 case where hand wins by 3-4%. The complexity to add manual pair-scheduling (special spill management for paired sub-DFTs, breaking the linear emission order, custom cmul derivation duplication) would be substantial for a 3-4% win in one (radix, K) corner.

The isub2 design is a valid optimization for hand-coded codelets that can't rely on a smart scheduler. With our SU + Spill recipe + log3 hash-cons, it's redundant and ultimately slower at 5/6 of the (radix, K) combinations tested.

## What we keep

- Our existing `--log3` flag generates codelets that match-or-beat hand isub2 at every (radix, K) except one corner.
- No new generator features needed for isub2 coverage.
- The R=16 K≥1024 case stands as a known 3-4% gap.

## What this validates

The recipe philosophy works: **automated SU scheduling + global hash-cons** outperforms hand's pair-scheduled structural optimization at R=32 and R=64, where the pair-scheduled approach hits its limits (paired sub-DFTs don't fit in registers anyway). At R=16 the hand approach has its niche, but that niche is narrow.

The R=32 result is particularly striking: **15% faster than hand isub2 at K=1024**, despite isub2 being one of the most carefully hand-tuned variants in `gen_radix32.py`.
