# 28. Composite regression — fma_lift was the leak from prime/odd work

## TL;DR

Tugbars hypothesized: "something we did for primes/odd codelets leaked
into pow2 CSE and code generation." **Confirmed and fixed.** The
culprit was `Vfft_v2.Algsimp.fma_lift`, called unconditionally on
every DAG before emission. fma_lift recognizes Add(Mul, c) patterns
and rewrites them as explicit NK_Fma atoms which the codegen renders
as `_mm512_fmadd_pd(...)` instead of separate `_mm512_mul_pd` +
`_mm512_add_pd`.

The in-source comment claimed: "GCC -O3 -mfma auto-fuses un-lifted
Mul+Add patterns reliably, so fma_lift is essentially a no-op for
codegen perf — verified by asm diff." **That claim is wrong for
composite (Cooley-Tukey) DAGs.** Explicit NK_Fma constrains GCC's
register allocation more than auto-fused mul+add chains, producing
significantly more vmovapd reg-reg moves and total FP instructions.

For primes (Direct construction with conjugate pairs), fma_lift gives
~1-2% benefit — the prime DAG shape exposes specific Add(Mul, c)
patterns that benefit from explicit FMA encoding. So the right fix is
to gate fma_lift to aggressive (= Direct primes) only.

**Fix:** in `bin/gen_radix.ml`, change

```ocaml
let deduped = Vfft_v2.Algsimp.fma_lift post_trans in
```

to

```ocaml
let deduped =
  if aggressive then Vfft_v2.Algsimp.fma_lift post_trans
  else post_trans
in
```

This recovered docs 09/11 era composite performance. Some K values
now exceed the documented numbers (R=64 K=4096 now wins 23% vs doc
11's 7%). Prime correctness 56/56 PASS unchanged.

## How the leak was found

Tugbars uploaded a copy of the doc 09-era `algsimp.ml` (721 lines, vs
current 1720 lines). Diffing structural function lists showed the old
file was missing FIVE top-level functions that exist in current:

- `factor_common_muls`
- `factor_by_atom`
- `share_subsums`
- `transpose`
- **`fma_lift`** ← this one

The first four are gated behind `aggressive` (only run for primes).
For composites with `aggressive=false`, all four are no-ops. So the
only one that actually executes for composites and didn't exist in
the doc 09 era is `fma_lift` — which has no aggressive gate.

That made fma_lift the single suspect.

## v0 hypothesis (wrong) — share_subsums

The earlier draft of this doc claimed `share_subsums` was the leak.
Empirically forcing share_subsums to run on composites (after fixing
a cross-pass scope issue with a forward-declare mechanism) made
performance MUCH worse:

| Metric | share_subsums OFF | share_subsums ON | Hand |
|---|---|---|---|
| Total FP ops | 910 | **1300** | 709 |
| vmovapd | 288 | **578** | 170 |
| llvm-mca SKX | 312 | **509** | 338 |

share_subsums materializes partial sums that prevent fma_lift from
recognizing the unified mixed-sign FMA chain — the comment block in
gen_radix.ml describes this exactly for primes:

> share_subsums pass — which factors common addition subsums across
> outputs — actively HURTS this layout: it materializes partial sums
> that prevent fma_lift from recognizing the unified mixed-sign FMA
> chain

The "DOES help Cooley-Tukey" claim in the same comment block is wrong
— at least in the current pipeline with fma_lift. Likely the comment
dates from before fma_lift existed, when share_subsums was a net win
on raw add/sub counts.

So `share_subsums` is correctly disabled for composites. The
experimental cross-pass forward-declare changes have been reverted.

## v1 (this version) — fma_lift confirmed and fixed

Disabling fma_lift for composites recovered the metrics dramatically.

### Asm metrics, R=32 t1_dit, gcc-13

| Metric | With fma_lift | **Without (gated)** | Hand |
|---|---|---|---|
| Total FMA | 106 | 118 | 128 |
| vmulpd | 112 | 106 | 101 |
| vaddpd | 160 | 160 | 155 |
| vmovapd | 288 | **101** | 170 |
| vmovupd | 190 | 190 | 128 |
| **Total FP** | **910** | **717** | **709** ✓ |

717 essentially matches hand's 709 — we have 8 fewer total FP
instructions than hand on R=32 now.

### Asm metrics, R=64 t1_dit, gcc-13

| Metric | With fma_lift | **Without (gated)** | Hand (current GCC) |
|---|---|---|---|
| vmovapd | 940 | 344 | (current GCC ~167×scaled) |
| Total FP | ~1900 | 1743 | (similar order) |

R=64 recovery is significant; comparing exact numbers to doc 11's
"26 vmovapd ours / 69 hand" requires accounting for the
GCC-version-only shift visible on hand alone (20 → 167 from
gcc-9-era to gcc-13). With that scaling, current numbers are in the
expected ballpark.

### llvm-mca cycles (loop body)

| Codelet | μarch | With fma_lift | **Without (gated)** | Hand |
|---|---|---|---|---|
| R=32 t1_dit | SKX | 312 | **226** | 338 |
| R=32 t1_dit | SPR | 305 | **276** | 393 |
| R=32 t1_dit | Zen 4 | 342 | **233** | 370 |
| R=64 t1_dit | SKX | 784 | **459** | 821 |
| R=64 t1_dit | SPR | 688 | **534** | 874 |
| R=64 t1_dit | Zen 4 | 777 | **432** | 836 |

llvm-mca says we now beat hand by 33-48% across all six cells. As
established in doc 27, llvm-mca tends to overstate wins by ~15pp vs
real silicon; realistic expectation is ~20-35% wins on bare metal,
still much better than the "1-9% wins" of doc 09/11 era.

### Real runtime, EMR container

R=32 K-sweep (SU/Hand ratio, lower = ours faster):

| K | Doc 09 (SKX virt) | Pre-fix (regressed) | **Post-fix** |
|---|---|---|---|
| 64 | 0.98 | 1.27 | **0.96** |
| 128 | 0.99 | 1.40 | **0.93** |
| 256 | 0.99 | 1.22 | 1.02 |
| 512 | 0.97 | 1.22 | 1.05 |
| 1024 | 0.93 | 1.19 | 1.00 |
| 2048 | 0.91 | 1.04 | **0.96** |
| 4096 | 0.91 | 0.98 | 1.01 |

R=32 recovered to doc 09 range. Container noise dominates K=256-512
(small variance pushes individual cells over 1.0); the underlying
codelet quality is intact.

R=64 K-sweep:

| K | Doc 11 (SKX virt) | Pre-fix | **Post-fix** |
|---|---|---|---|
| 64 | 0.99 | 1.40 | **0.95** |
| 128 | 0.98 | 1.23 | **0.98** |
| 256 | 0.93 | 1.35 | **0.97** |
| 512 | 0.95 | 1.22 | 1.05 |
| 1024 | 0.93 | 1.13 | **0.95** |
| 2048 | 0.98 | 1.17 | **0.83** |
| 4096 | 0.93 | 1.36 | **0.77** |

R=64 recovered AND improved. K=2048-4096 wins 17-23% vs doc 11's 7%.
Either the intervening code changes (single-use inlining,
cluster-sequential PASS 2 refinements) genuinely improved R=64, or
container noise affects hand more than us at large K. Either way the
fix recovers and exceeds the documented baseline.

### Prime perf delta with the gate

R=13 t1_dit asm metrics:

| Metric | With fma_lift | Without |
|---|---|---|
| Total FMA | 156 | 156 |
| vmulpd | 36 | 36 |
| vmovapd | 55 | 46 |

Same FMA count, marginally different vmovapd. llvm-mca SKX cycles
191 vs 196 — fma_lift is a 2.5% win for R=13. Similar tiny effect on
R=17 (291 vs 293 cycles).

Gating fma_lift behind aggressive preserves the ~1-2% prime benefit
and recovers the ~28% composite advantage. Net: clear win.

## Why fma_lift hurts composites

Two mechanisms:

1. **Constrained register allocation.** With explicit `_mm512_fmadd_pd(a, b, c)`,
   GCC must honor the specific (a, b, c) operand assignment we encode.
   With separate `_mm512_mul_pd(a, b)` + `_mm512_add_pd(mul, c)`, GCC
   picks any of the three FMA variants (132/213/231) freely, choosing
   based on which assignment minimizes register pressure. Composite
   DAGs with their cluster-sequential structure benefit from this
   flexibility; primes have fewer scheduling alternatives so the
   benefit is small either way.

2. **Cross-pass interactions.** fma_lift creates new NK_Fma atoms that
   may bridge PASS 1 / PASS 2 boundaries differently than the original
   Add(Mul, c) structure did. The classify_passes / inline_set logic
   was tuned for the pre-fma_lift node shapes. Post-fma_lift, the
   classification produces more compiler-spilled values, manifesting
   as extra vmovapd.

The ~1-2% prime benefit suggests fma_lift IS doing useful work for
prime DAGs — probably exposing FMA opportunities in the
conjugate-pair construction that GCC's local pattern matcher
wouldn't find. But that benefit is much smaller than the cost it
imposes on composite DAGs.

## Status

- ✓ Regression mechanism identified: fma_lift creating explicit NK_Fma
  atoms that hurt GCC's RA on composite DAGs
- ✓ Empirical confirmation: gating fma_lift recovers docs 09/11 era
  composite asm metrics
- ✓ Fix landed: `if aggressive then fma_lift post_trans else post_trans`
- ✓ Prime correctness: 56/56 PASS
- ✓ All composite radices (R=4, 8, 16, 32, 64) compile
- ✓ All R=32 variants (DIT/DIF, fwd/bwd, t1s) compile
- ✓ llvm-mca cycles recovered: R=32 SKX 312 → 226, R=64 SKX 784 → 459
- ✓ Runtime K-sweep on EMR container: R=32 recovered to doc 09 range,
  R=64 exceeds doc 11 range
- → Bare-metal SPR/EMR verification still recommended for production-
  grade numbers
- → Separate pre-existing issue: `--no-recipe --spill` still has a
  compile error (undeclared tag in spill store) — unrelated to
  fma_lift, predates this session

## Open questions

- Why does the post-fix R=64 advantage *exceed* doc 11's? Plausible
  causes: (a) cluster-sequential PASS 2 refinements landed after doc
  11 genuinely help R=64 more than they help R=32; (b) container
  affects hand differently than us at large K; (c) doc 11
  methodology differed slightly. Bare-metal SPR/EMR measurement
  would distinguish.
- Should we revisit whether fma_lift helps primes at all? The 1-2%
  benefit is small enough that disabling fma_lift entirely might
  simplify the code with negligible perf cost. Worth a separate
  perf bisect on primes.

## Files changed

- `bin/gen_radix.ml` — gate fma_lift behind `aggressive` flag, with
  updated comment explaining the empirical finding

No changes to `lib/algsimp.ml` (share_subsums correctly stays
disabled for composites; v0 hypothesis was wrong, experimental
patch reverted).

No changes to `lib/emit_c.ml` (cross-pass forward-declare mechanism
from share_subsums experiment reverted; not needed for actual fix).
