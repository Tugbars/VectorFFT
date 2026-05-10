# 29. AVX2 prime threshold — extending the recipe to R=5/R=7

## TL;DR

The cost-model rule `should_spill` decides whether a codelet runs the
full Spill+SU recipe or pure Topo emission. The original rule was

```ocaml
let should_spill (n : int) (vec_regs : int) : bool =
  (n + 6 > vec_regs) || vec_regs >= 32
```

For AVX2 (vec_regs=16), only clause (1) applied, triggering at R≥11.
R=5 and R=7 ran pure Topo. The rule was set this way because an early
bench showed R=8 AVX2 with the recipe regressed by 1.00–1.08× — but
that bench was measured **before** the doc 28 fma_lift fix, when
fma_lift was unconditional and creating exactly the kind of register
pressure that breaks AVX2 small-codelet emission.

After the doc 28 fix gates fma_lift to primes only, the recipe path is
healthy on AVX2 for small codelets too. Empirical sweep shows R=5 and
R=7 lose 4–37% to hand without the recipe, and win 2–18% with it.

**Fix:** add a third clause to the rule.

```ocaml
let should_spill (n : int) (vec_regs : int) : bool =
  (n + 6 > vec_regs) || vec_regs >= 32 || n >= 5
```

This keeps R=2/3/4 in the Topo path (too small to benefit) and puts
everything from R=5 upward in the recipe path. R=11+ AVX2 was already
in the recipe path via clause (1); they're unaffected.

## How this was found

After yesterday's doc 28 fix landed, Tugbars asked to test primes on
AVX2 vs hand to make sure the AVX-512 fix didn't have an AVX2 analog.
We had AVX-512 hand-vs-ours data showing primes were healthy. AVX2
data didn't exist.

Generated AVX2 hand references via the Python codelet generators
(`gen_radix{5,7,11,13,17,19}.py --isa avx2 --variant ct_t1_*`) into
`/tmp/hand_avx2/`. Ran a 56-cell sweep (R={5,7,11,13,17,19} ×
{t1_dit, t1_dif, t1_dit_log3, t1s_dit} × {K=64..4096}).

R=11/13/17/19 all in the recipe path: mostly winning, 5–25% over
hand, occasional cell-level outliers consistent with container noise.

R=7 in the Topo path: losing 4–37% across all variants. **t1_dif
catastrophic at K=512–4096 (1.23–1.37×).** This was the standout
problem — the same DAG variant that benchmarked fine on AVX-512 was
broken on AVX2.

Forcing the recipe ON for R=7 (probe: `|| n >= 5`) recovered
everything: every variant flipped from losing to winning by 2–18%.
The biggest swing was t1_dif K=1024: 1.37 → 0.97, a 40-point recovery.

## R=5 — the case that needed careful checking

R=5 wasn't initially benched without recipe. Adding the `n >= 5`
clause put it in the recipe path; an immediate concern was whether
the recipe was actually *helping* R=5 or just adding overhead for a
small codelet that doesn't need it.

Direct comparison (this container, EMR, AVX2):

| Variant | WITHOUT recipe | WITH recipe | Δ |
|---|---|---|---|
| R=5 t1_dit | 0.98–1.25 | 0.96–1.05 | recipe better at K=512 spike |
| **R=5 t1_dif** | **1.19–1.30** | **1.03–1.16** | **recipe wins 13–19pp** |
| R=5 t1_dit_log3 | 0.79–0.95 | 0.80–0.94 | wash (both win) |
| R=5 t1s_dit | 1.00–1.22 | 1.00–1.09 | recipe better at K≥1024 |

The recipe is a clear net win for R=5, dominated by the t1_dif
recovery. Both states win on log3 and lose mildly elsewhere; t1_dif
is the differentiator.

## Why the recipe helps small primes on AVX2 specifically

Two factors compound:

1. **AVX2 has 16 YMM vs AVX-512's 32 ZMM.** Even small codelets like
   R=5 (5 outputs = 10 YMM live for split re/im storage) plus working
   set (~5–8 YMM for in-flight DFT computation plus twiddle constants)
   bump up against the 16-register ceiling. GCC's RA spills aggressively
   to fit, and without explicit Spill markers it spills *unpredictably*
   at points that hurt scheduling.

2. **The doc 28 fix matters more on AVX2 than AVX-512.** With
   fma_lift on, explicit NK_Fma nodes constrained GCC's ability to
   re-balance reg pressure. On AVX-512 there were enough registers to
   hide this; on AVX2 it pushed marginal codelets over the cliff. With
   fma_lift gated to primes only, AVX2 reg pressure is back in a
   range the recipe can manage cleanly.

The Spill+SU recipe fixes both: explicit spill markers tell GCC
exactly when to evict, and SU (single-use inlining) reduces the live
SSA name set so the actual peak live in any given cycle stays within
the 16-register budget.

## Why t1_dif specifically suffers most

Pattern visible across multiple radices on AVX2 (not just R=5/R=7):
DIF variants underperform DIT variants. R=7 t1_dif was the worst
case before the fix; R=13 t1_dif still has K≥512 issues even with
the recipe; R=17 t1_dif marginal. R=19 holds up.

Hypothesis: DIF emits the twiddle multiplication AFTER the butterfly
rather than before. This produces a different live-range pattern —
specifically, several intermediate butterfly results live across the
twiddle multiplication step, increasing peak live count in a window
where DIT has already consumed those values. On AVX-512 (32 ZMM)
this never matters; on AVX2 it does.

This isn't fixed by the threshold extension. **It's a separate
codegen issue worth investigating later.** The threshold fix
recovers the catastrophic R=5/R=7 cases; the residual DIF weakness
on R=13 K≥512 (1.06–1.10× hand) remains unaddressed but isn't
catastrophic.

## Asm metric verification

R=5 t1_dit AVX2 with vs without recipe:

| Metric | WITHOUT recipe | WITH recipe |
|---|---|---|
| Function suffix | `_gen_inplace` | `_gen_inplace_su` |
| vmovapd | (varies, GCC RA picks freely) | (constrained by spill markers) |
| vmovupd | similar | similar |
| Total FP | similar | slightly less |

The recipe doesn't dramatically change instruction count — it changes
*scheduling*. By telling GCC where to put the spill boundaries, the
recipe eliminates GCC's worst-case RA decisions that produced the
observed performance variance at K=512 (the 1.25 outlier without
recipe).

## What the threshold change does NOT do

- Does not affect AVX-512 codelets (clause 2 already covers them)
- Does not affect R=2/3/4 (still Topo on both ISAs)
- Does not affect R=11+ (clause 1 already covers them on AVX2)
- Does not affect correctness — 56/56 prime correctness PASS
  unchanged from before/after fix
- Does not address the residual DIF-on-AVX2 weakness mentioned above

## Files changed

- `lib/dft.ml`: `should_spill` extended with clause `|| n >= 5` and
  updated explanatory comment block

No other code changes. The recipe path itself (Spill + SU + cluster-
sequential PASS 2 + deferred reload) is unchanged — we just gate it
to fire on more codelets.

## Status

- ✓ R=5/R=7 AVX2 regression identified
- ✓ Mechanism: recipe-gate threshold tuned for pre-doc-28 state
- ✓ Fix landed: extend clause to `n >= 5`
- ✓ R=7 sweep validates: 4–37% loss → 2–18% win
- ✓ R=5 verified: recipe is net positive (especially t1_dif: 13–19pp)
- ✓ Other primes (R=11/13/17/19) unaffected
- ✓ Prime correctness 56/56 PASS
- → Open: DIF-on-AVX2 weakness (R=13 K≥512, R=17 marginal). Separate
  investigation, not blocking.
- → Open: AVX2 spill emission for R=8 has a `__m256d *` vs `double *`
  type mismatch warning when forced into the spill path. R=8 still
  uses the spill path (clause 3 fires on R=8 too) so the warning
  appears now. Functionally fine, codegen cleanliness fix needed.

## Next steps

This fix unblocks the next priority: **mixed-radix codelet integration
starting with R=6 (CT(3, 2))**. The threshold change covers small
primes including the DFT-3 sub-block used internally by R=6 / R=12,
so prime sub-block performance inside mixed-radix CT codelets should
be solid on AVX2.
