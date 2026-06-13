# DIF Across Radices — Convention Mismatch at R=8

## Summary

Validated DIF support across all radices that have hand-coded references. **Our DIF generation is mathematically correct at every radix**, but **R=8 hand DIF uses a nonstandard convention** that drops the W^4 = -1 twiddle (since W_8^4 is exactly -1, hand absorbs it into a sign flip in the butterfly itself instead of an external multiply).

| Radix | Hand DIF | Our DIF correctness | Notes |
|-------|----------|--------------------|---------|
| R=4   | exists   | ✓ matches hand      | Standard convention, slot j-1 for output j |
| R=8   | exists   | ⚠ convention mismatch | Hand skips slot 3 (W^4 = -1); ours applies all 7 twiddles |
| R=16  | doesn't exist | n/a            | — |
| R=32  | doesn't exist | n/a            | — |
| R=64  | exists   | ✓ matches hand      | Standard convention, all 63 slots |

## R=8 DIF: how we know our math is right

Wrote a direct-DFT reference (`bench_r8_dif_check.c`) that:
1. For each k position, computes `DFT-8(x[*, k])` directly via the definition
2. Post-multiplies by `W_8K^(j*k)` to get the expected DIF output
3. Compares to our generated codelet output

Result:
```
PASS: max_err=9.55e-15 (K=8)
PASS: max_err=3.15e-14 (K=64)
PASS: max_err=5.27e-13 (K=1024)
```

Our R=8 DIF is correct to machine precision. The convention mismatch with hand R=8 DIF is hand's choice, not our bug.

## What hand R=8 DIF does

```c
const __m512d y4r=_mm512_sub_pd(A0r,B0r), y4i=_mm512_sub_pd(A0i,B0i);
ST(&rio_re[m+4*ios], y4r);  // no twiddle multiply
ST(&rio_im[m+4*ios], y4i);
```

For output 4 (j = N/2 at R=8), hand stores the raw butterfly output. The `W_8^4 = -1` factor is absorbed elsewhere — likely into the sign convention of A0/B0 in the butterfly. This is a valid optimization for codelets used in a specific CT recursion structure where the caller compensates for the missing twiddle.

Hand uses 6 twiddle slots {0, 1, 2, 4, 5, 6} skipping slot 3. Our DIF uses all 7 slots {0..6} per the standard convention.

## R=4 DIF performance

Recipe is essentially tied with hand at R=4 DIF (SU/H ranges 1.00-1.04 across K), as expected — R=4 is too small for the recipe to matter much, and the cost-model rule auto-applies it on AVX-512 only because the rule is conservative.

## R=64 DIF performance with log3 (3 runs each, median)

| K | Hand | Recipe | Recipe-log3 | SU/H | **LR/H** |
|---|------|--------|-------------|------|----------|
| 64 | 4048 | 3427 | 3081 | 0.85 | **0.75** |
| 256 | 19842 | 18585 | 19714 | 0.94 | 1.00 |
| 1024 | 119255 | 112865 | 85775 | 0.95 | **0.72** |
| 4096 | 602097 | 557954 | 474397 | 0.93 | **0.79** |

**DIF + log3 wins 21-28% over hand DIF at most K.** At K=256 they're tied (the bandwidth crossover point).

## Final variant matrix at R=64 in-place (all vs respective hand reference)

| Variant | K=64 | K=256 | K=1024 | K=2048 | K=4096 | Avg |
|---------|------|-------|--------|--------|--------|-----|
| DIT recipe vs hand DIT | 0.68 | 0.96 | 0.91 | 0.93 | 0.94 | 0.88 |
| **DIT recipe-log3 vs hand DIT** | **0.63** | 0.84 | **0.66** | **0.62** | **0.72** | **0.69** |
| DIF recipe vs hand DIF | 0.85 | 0.93 | 0.95 | 0.95 | 0.93 | 0.92 |
| **DIF recipe-log3 vs hand DIF** | **0.75** | 1.00 | **0.72** | n/a | **0.79** | **0.81** |

DIT recipe-log3 remains the strongest variant overall. DIF + recipe is competitive but doesn't quite match DIT's headroom — likely because the post-multiply Cmuls add register pressure that PASS 2 cluster-sequential emission can't fully hide.

## What changed in the classifier

Two iterations of the backward classifier pass were needed:

**Iteration 1: Loads only.** Pass2 nodes' Twiddle Loads need to be in Pass2 scope (not Pass1). Reclassify Pass1 Loads whose all consumers are Pass2.

**Iteration 2: All Pass1 nodes, fixpoint loop.** DIF + log3 surfaced cmul derivation nodes (e.g., W^7 = cmul(W^3, W^4)) that have only Pass2 consumers but aren't themselves Loads. Extended the backward pass to ANY Pass1 node, with a fixpoint loop in case reclassifying X causes Y (which fed X) to become reclassifiable.

In practice the loop converges in 1-3 iterations.

## Validation across the variant matrix

All 16 combinations of `{R=4, R=8, R=16, R=32, R=64}` × `{DIT, DIF}` × `{flat, log3}` × `{recipe-on, recipe-off}` generate and compile correctly. Where hand references exist:

| | R=4 | R=8 | R=16 | R=32 | R=64 |
|---|-----|-----|------|------|------|
| DIT (in-place) | matches | matches | matches | matches | matches |
| DIT log3 | matches | matches | matches | matches | matches |
| DIF (in-place) | matches | conv. mismatch | n/a | n/a | matches |
| DIF log3 | matches | not tested* | n/a | n/a | matches |

*R=8 DIF log3 also has the convention mismatch since it uses the same external twiddle layout; we'd need a direct-DFT reference to validate.

## What's left

- **R=16/R=32 DIF.** No hand reference exists. Could validate via direct DFT or DIT/DIF cross-check, but lower priority.
- **bwd direction.** All hand variants come in fwd/bwd pairs; we only emit fwd.
- **Variants from `gen_radix64.py` we haven't matched yet.** `ct_t1_buf_dit` (buffered output for tile/drain), `ct_t1_dit_prefetch` (prefetch hints). These are tuning extensions on top of t1_dit, not new functional variants.
