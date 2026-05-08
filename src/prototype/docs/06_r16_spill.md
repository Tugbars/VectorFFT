# R=16 Spill Validation — `should_spill` Heuristic Falsified

## Summary

Tested forcing the spill variant on R=16 (where the `should_spill` heuristic predicts "don't spill" because peak live ≈ 22 < 32 vec_regs). Result: **spill helps R=16 too — and matches or beats hand-coded at K ≥ 512.**

The heuristic was wrong. Spilling is beneficial even when registers theoretically fit.

## Bench results (AVX-512, R=16, 3 runs per K, median reported)

| K | Hand (ns) | Topo (ns) | Spill (ns) | T/H | Spill/H | Spill/T |
|---|-----------|-----------|------------|-----|---------|---------|
| 64 | 502 | 636 | 570 | 1.28 | 1.14 | 0.90 |
| 128 | 1348 | 1568 | 1383 | 1.16 | **1.03** | 0.88 |
| 256 | 3598 | 4443 | 3938 | 1.24 | 1.09 | 0.88 |
| 512 | 9695 | 10804 | 9445 | 1.11 | **0.99** | 0.89 |
| 1024 | 21191 | 26049 | 21298 | 1.23 | **1.01** | 0.82 |
| 2048 | 54065 | 59393 | 52133 | 1.10 | **0.96** | 0.88 |
| 4096 | 140777 | 154517 | 138152 | 1.02 | **0.97** | 0.93 |

**Two findings:**

1. **Spill always beats Topo.** 7-18% faster across all K. Same pattern as R=32. Variant approach is robust across radices.
2. **Spill matches or beats Hand at K ≥ 512.** Spill/H is 0.99 / 1.01 / 0.96 / 0.97 at K = 512 / 1024 / 2048 / 4096.

## Why the heuristic was wrong

Current `should_spill` rule:

```ocaml
let should_spill (n : int) (vec_regs : int) : bool =
  n + 6 > vec_regs
```

For R=16 AVX-512: `22 > 32 = false`. Heuristic says "don't spill."
Empirical: spill is a clear win across all K.

The issue: the heuristic asks **"does the live set exceed register count?"** That's the wrong question. The right question is **"does explicit boundary spilling outperform GCC's register allocator on our DAG?"**

Even when peak live (~22) fits in 32 ZMM registers, GCC produces suboptimal code on our R=16 DAG — likely because:
- Our DAG topology (large SoA with shared NK_Const leaves) confuses GCC's live-range analysis
- Without an explicit barrier, GCC can't tell PASS 1 outputs from intermediate PASS 2 values; it pessimistically extends lifetimes across the natural boundary
- Once we mark the boundary explicitly (via `{ }` scope and explicit spill stores), GCC can do simpler per-pass allocation

This is corroborated by hand-coded behavior: the hand-coded R=16 codelet **also spills**, despite peak live fitting in registers. Hand-coded designers spill because it produces better code, not because they have to.

## Updated rule

The simple register-count heuristic should be replaced. Two candidate rules, in order of conservatism:

**Rule A (data-supported):** `spill if CT-decomposed and N >= 16`.
**Rule B (more aggressive):** `spill if CT-decomposed`.

R=8 is the next falsification target. If R=8 spill also helps, Rule B is right. If R=8 spill hurts (peak live ~10 is comfortably below 32), Rule A is right with threshold around 12-16.

For now, the practical recommendation: **default to spill for any R ≥ 16 CT-decomposed codelet on AVX-512**. The spill variant has no clear downside in the K range we've tested (64 to 4096) and frequently matches hand-coded.

## Structural gap that remains

At R=16, the small-K regime (K ≤ 256) still shows Spill 8-14% behind Hand. This mirrors the R=32 pattern but at smaller magnitude.

The cause is the same as at R=32: hand-coded keeps a fraction of PASS 1 outputs in registers across the boundary instead of spilling everything. Our generator currently spills all of them.

This is the **FUSED optimization** — per-spill-target lifetime analysis to identify which PASS 1 outputs are consumed first in PASS 2 and skip the spill round-trip for those. Estimated 5-10% gain at small K, less at large K.

## What this means for the cost model

The variant decision was previously framed as a register-pressure threshold. The R=16 data shows that's the wrong frame.

The actual cost model needs:
- **Static signal**: CT decomposition with N ≥ threshold (likely 16, possibly 12) → spill is beneficial
- **K-dependent signal**: spill becomes more beneficial as K grows (because L1-resident spill array hides latency, and large K amortizes reload costs)

This is simpler than I thought. The cost model can probably be a 5-line rule rather than a fitted curve.

## Status

- ✓ R=16 spill variant emits correctly, matches Hand within FP tolerance
- ✓ Spill always beats Topo at R=16 (7-18%)
- ✓ Spill matches or beats Hand at R=16 K ≥ 512
- ✗ `should_spill` heuristic falsified — needs revision
- → FUSED optimization is next target (closes small-K gap)
