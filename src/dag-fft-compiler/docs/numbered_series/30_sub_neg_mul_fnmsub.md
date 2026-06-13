# 30. Sub(Neg(Mul), c) → fnmsub: a Modern-FMA-Aware genfft Lifting

## Context

This investigation falls under the broader research question:
**what should genfft's IR-level decisions look like when targeting a modern
FMA-aware compiler (GCC 13/14 + AVX-512/AVX2)?**

Frigo's original genfft (early 2000s) was tuned for compilers without
reliable FMA fusion or sophisticated register allocation. Modern GCC does
both well per-context. The question is which IR-level transformations
should *constrain* that capability vs *expose* it.

## The Anomaly

Cross-radix diagnostic on AVX-512 t1_dit kernels (R ∈ {5, 7, 11, 13, 17, 19, 25}):

| R    | total | fma  | **vxor** | bcast | rod | mem-arith | spill | reload | reg-copy |
|------|-------|------|------|-------|-----|-----------|-------|--------|----------|
| 5    |    89 |   20 |    0 |     4 |   0 |         0 |     0 |      0 |        4 |
| 7    |   147 |   42 |    0 |     6 |   0 |         0 |     0 |      0 |       10 |
| 11   |   281 |  110 |    0 |     5 |   0 |        50 |     6 |      0 |       15 |
| 13   |   395 |  156 |    0 |    11 |   6 |        44 |    24 |      6 |       19 |
| 17   |   661 |  272 |    0 |    22 |  10 |        16 |    43 |     41 |       44 |
| 19   |   797 |  342 |    0 |    28 |  10 |        78 |    59 |     41 |       60 |
| **25** | **810** | **200** | **6** | 10 | 6 | 44 | 45 | 39 | 35 |

Every prime (R=5..19) has zero `vxorpd`. R=25 (the only Cooley-Tukey-
decomposed codelet) has 6.

`vxorpd .LC_neg_zero(%rip), %zmm, %zmm` is GCC's emission for sign-flip
when it can't fuse a negation into an FMA variant. Each occurrence costs:
  - 1 extra vxorpd instruction
  - 1 extra .rodata constant (the -0.0 mask) which GCC keeps register-
    resident, consuming a ZMM slot
  - The mul that feeds it is computed separately rather than fused

## Root Cause

R=25's IR contains 6 `Sub(Neg(Mul(a, b)), c)` patterns (where c is itself
a Mul). These patterns survive simplification because:

1. **dedup_sub_pairs** identifies `Sub(a, b)` and `Sub(b, a)` pairs in the
   DAG and substitutes the loser with `Neg(winner)` (algsimp.ml:578).

2. The substituted Neg is then consumed as the **LHS** of another Sub
   somewhere in the DAG.

3. `mk_sub_binary` has a peephole for Neg on the RHS (`Sub(x, Neg(y))` →
   `Add(x, y)`) but not on the LHS. So `Sub(Neg(z), x)` falls through to
   `hashcons(NK_Sub(Neg, _))`.

4. emit_c renders `Sub(Neg(Mul(a, b)), c)` as
   `_mm512_sub_pd(_mm512_xor_pd(_mm512_mul_pd(a,b), neg_zero), c)`.

5. GCC compiles this to vmulpd + vxorpd + vsubpd instead of recognizing
   the equivalence to vfnmsub231pd.

The pattern doesn't appear in primes because dft_direct_conjugate_pair's
`make_sum` chain emits Neg only at the head of its accumulator (when
the first coefficient is negative), and the smart constructor peephole
on Add catches the resulting `Add(x, Neg(...))` patterns. Subs of a
just-introduced Neg never arise in primes.

CT codelets get them because dedup_sub_pairs has more material to work
with (twiddle muls produce many similar Sub patterns), and the substituted
Neg can land in any Sub's LHS depending on DAG structure.

## The Fix

Add a targeted IR pass `lift_sub_neg_mul`:

```
Sub(Neg(Mul(a, b)), c)  →  NK_Fma(a, b, c, neg_mul=true, neg_add=true)
```

The NK_Fma node with `(neg_mul=true, neg_add=true)` flags emits as
`_mm512_fnmsub_pd(a, b, c)` which compiles to vfnmsub231pd directly.

**The pass runs unconditionally on every codelet** — there is no radix
gate. It walks the full IR and rewrites every occurrence of the pattern.
The reason this looks like an "R=25 fix" empirically is that the pattern
is rare in current codelets, not because we deliberately restricted it:

- **Primes never produce it.** dft_direct_conjugate_pair's `make_sum`
  emits `Neg` only at the head of an accumulator chain that is then
  consumed by an `Add`, where mk_add's `_, NK_Neg b' -> mk_sub_binary a b'`
  peephole catches and eliminates the Neg. Primes reach the simplification
  fixed point with zero Negs.

- **CT-with-prime-sub-block codelets do.** dedup_sub_pairs introduces
  Neg substitutions when it identifies opposite-direction Sub pairs in
  the DAG, and depending on DAG structure, those Negs can land in any
  Sub's LHS. R=25 (CT(5,5)) is the first composite we wired with prime
  sub-blocks, so it surfaces first.

Future codelets in the same family — R=10 (CT(5,2)), R=12 (CT(4,3)),
R=20 (CT(5,4)) — will likely produce similar patterns and benefit
automatically without per-radix flagging.

The pass is correctness-preserving (`-(a*b) - c` is exactly what
fnmsub computes) and strictly better wherever the pattern appears, so
unconditional application is safe. Adding a gate would just be
deferred maintenance work for the next person to wire a CT codelet.

**Run unconditionally** (not gated to primes, unlike fma_lift). The
distinction matters:

- **fma_lift on composites** (gated off, doc 28): REPLACES auto-fusion
  that GCC was already doing well. Forced variant choices constrain
  GCC's RA, costing instructions.
- **lift_sub_neg_mul** (unconditional, this doc): REPLACES emission
  that GCC was already doing badly. The xor-with-mask path is strictly
  worse than fnmsub in instructions, register pressure, and rodata
  pressure.

The general principle: **the right scope for IR-level decisions is
not "above" or "below" the compiler's scope uniformly. It's per-pattern,
based on whether our IR transformation aligns with or fights against
the compiler's existing capabilities.**

## Empirical Validation

R=25 ours/hand ratios at K=256 (the cache-cliff regime where below-
compiler-scope decisions become visible):

| Variant         | Before | After | Δ        |
|-----------------|--------|-------|----------|
| avx512 t1_dit   | 1.220  | 1.113 | -10.7pp  |
| avx512 t1s_dit  | 1.240  | 1.159 |  -8.1pp  |
| **avx2 t1_dit** | 1.310  | **1.048** | **-26.2pp** |
| **avx2 t1_dif** | 1.280  | **1.049** | **-23.1pp** |
| avx2 t1s_dit    | 1.210  | 1.037 | -17.3pp  |

Cross-radix asm metrics (AVX-512 t1_dit) — the pass runs on every codelet,
this table just records where it found patterns to rewrite:

| R  | Δinst | Δvxor | Δvmovapd | Pattern occurrences |
|----|-------|-------|----------|---------------------|
| 5  | 0     | 0     | 0        | 0                   |
| 7  | 0     | 0     | 0        | 0                   |
| 11 | 0     | 0     | 0        | 0                   |
| 13 | 0     | 0     | 0        | 0                   |
| 17 | 0     | 0     | 0        | 0                   |
| 19 | 0     | 0     | 0        | 0                   |
| 25 | -11   | -6    | -5       | 6                   |

R=25-only impact today, exactly matching the 6 IR Negs the pass converts.
Primes are unaffected because their IR construction never produces the
pattern. Expected to find more matches at R=10, R=12, R=20 when those
CT-with-prime-sub-block codelets are wired.

## Causal Mechanism for the Cliff Closure

The fix removes 6 instructions but the K=256 cliff narrows by 8-26
percentage points. The mechanism is not instruction count — it's
register pressure relief:

1. The `-0.0` mask used by the 6 vxorpd was register-resident across
   the function (GCC sees it has multiple uses, broadcasts once).
   Removing it frees one ZMM slot.

2. Each xor-mask sequence required: mask reg + multiplicand reg + result
   reg = 3 simultaneously-live values. Replacing with fnmsub uses only
   the operand registers (no mask).

3. With ~32 ZMM available and ~30+ live at PASS 1/2 boundary, freeing
   even one slot reduces GCC-induced spills.

4. At K=256 (working set just above L1, fully L2-resident), each
   GCC-induced spill is an L1→L2 round trip. The fix's spill reduction
   directly attacks this cost.

## Architectural Generalization

This finding crystallizes a per-pattern distinction for genfft IR design:

**Pattern is OK to leave to compiler:**
- Add(Mul, c) auto-fusion to FMA — GCC handles per-context with full RA awareness
- Variant selection (132 vs 213 vs 231) — GCC picks based on local live-ranges

**Pattern needs IR-level lifting:**
- Patterns the compiler can't recognize (e.g., Sub(Neg(Mul), c) → fnmsub)
- Patterns that span basic-block boundaries (e.g., spill recipe across cluster boundaries)
- Patterns where naive emission has known-bad codegen (this doc's case)

The discriminator: does the IR-level lifting REPLACE good compiler work or REPLACE bad compiler work? Replace bad → unconditional win.
Replace good → conditional, gated, often net-negative.

## Implementation

Pass added to `lib/algsimp.ml` as `lift_sub_neg_mul`, called from
`bin/gen_radix.ml` between `dedup_sub_pairs` and the share/transpose
pipeline. ~30 lines of OCaml. No new node types; reuses existing
NK_Fma infrastructure. Emit_c already handles all four (neg_mul,
neg_add) cases including the (true, true) = fnmsub form.

## Future Work

The cross-radix diagnostic methodology generalizes:
- Build the same metrics table for AVX2, NEON, and other ISAs to find
  ISA-specific pattern leaks.
- Extend to other CT-decomposed sizes (R=10, R=12, R=20) when those are
  wired — likely some will have similar Sub(Neg(Mul), c) patterns.
- The use-count-per-constant metric (R=25 ours has 11/27 single-use vs
  hand's 4/22) suggests another investigation: why are our constants
  fragmented, and is there a CSE-shaped fix.

## Update: R=10, R=12, R=20 cross-radix validation

After wiring R=10 (CT(5,2)), R=12 (CT(4,3)), R=20 (CT(5,4)):

| R  | Ours inst | Hand inst | Δ     | Ours vxor | Hand vxor | Bench best |
|----|-----------|-----------|-------|-----------|-----------|------------|
| 10 | 213       | 221       | -3.6% | 0         | 0         | 0.911      |
| 12 | 242       | 254       | -4.7% | 0         | **3**     | **0.725**  |
| 20 | 521       | 543       | -4.1% | 0         | 0         | 0.886      |

(AVX-512 t1_dit metrics; bench best is best ours/hand ratio across all
K and ISA combinations — values <1.0 mean we beat hand.)

**Headline finding**: at R=12, hand has 3 unfused vxorpd patterns (and
5 GCC-induced spills) that our `lift_sub_neg_mul` pass eliminates. Hand
also has these patterns; the gen_radix12.py author didn't catch them.
Our research-derived genfft optimization is now strictly better than
the hand-tuned reference at this specific micropattern.

R=12 is the strongest demonstration that the methodology produces
actionable findings: we beat hand by up to 27.5% on AVX-512 t1_dif
K=1024 (ratio 0.725).

R=10 and R=20 generalize: we beat or match hand at all instruction-count
metrics, with no cliff regimes. The K=256 cliff that motivated the
investigation at R=25 doesn't appear at R=10/12/20 — their working sets
fit L1 at all measured K values, so the register-pressure pathway that
exposed the issue at R=25 isn't activated.

The pass continues to fire only where the pattern arises. R=10 produces
2 IR Negs (caught), R=12 produces 1, R=20 produces 5. None of these
became visible as a perf regression because the fix was already in
place when they were wired — empirical confirmation that unconditional
application is the right design.
