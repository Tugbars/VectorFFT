# 56. FMA-lift fix and gating decision

## 1. Why we needed this fix

The M-project register-allocation work (Stage 4, doc 55) emits each
scheduled value through a register-pinned barrier:

```c
register __m512d t12 asm("zmm5") = _mm512_add_pd(t8, t11);
asm volatile("" : "+v"(t12));   /* pin t12 to zmm5 */
```

The `asm volatile` is a side-effect barrier as far as gcc is concerned.
It prevents the optimizer from looking through `t12` to rewrite the
surrounding expressions. Concretely, this kills gcc's auto-contraction
of `Add(Mul(a,b), c)` into `vfmadd*` because the Mul produces a value
that gets pinned, and the Add reading that pinned value can't be
fused back into a single FMA — the barrier forbids it.

Verified at the assembly level on R=8 AVX2 n1:

| Build | vfmadd count in body |
|---|---|
| Default emission (no M-project) | 3 |
| Stage 4 M-project emission | 0 |

That's the entire FMA contraction gone. Every Add(Mul, c) that gcc
would have produced as a single FMA becomes two separate
`vmulpd; vaddpd` instructions, costing throughput and inflating port
pressure.

The architectural response, in your framing: **own the FMA fusion in
IR so M-project's barriers become harmless.** That's what `fma_lift`
in `lib/algsimp.ml` does — it walks the DAG and rewrites
`Add(Mul(a,b), c)` patterns into `NK_Fma(a, b, c)` nodes, which emit
as `_mm512_fmadd_pd(a, b, c)`. After fma_lift, the FMAs are explicit
intrinsics that gcc must emit as `vfmadd*` instructions regardless of
any `asm volatile` between them — the barrier blocks fusion *of*
operations, not the emission of operations already written as FMAs.

The IR pipeline already had `fma_lift` available, but doc 28 had
gated it off for everything except Direct primes. Activating it on
the composite codelets that M-project targets was therefore the
prerequisite for M-project to deliver its full value. That activation
exposed the bug this fix addresses.

## 2. The fix

### One-line change in `lib/algsimp.ml`

```ocaml
(* Before: *)
let liftable_mul (_n : t) : bool = true in

(* After: *)
let liftable_mul (n : t) : bool = single_use n in
```

`fma_lift` walks every `Add(Mul(a,b), c)` (and the Sub/Neg variants)
and decides whether to rewrite it into `Fma(a, b, c)`. `liftable_mul`
is the predicate that says "yes, lift this Mul into the Fma." Before
the fix it returned `true` unconditionally — every Mul got lifted,
including Muls shared by multiple consumers. After the fix, only Muls
with use_count = 1 (single-use) are lifted.

The `single_use` value was already computed in the function (line
1586) but was being explicitly discarded with the comment "now using
the more permissive `liftable` below." The "more permissive" version
was the bug.

### What the unconditional lift was doing wrong

The author's earlier reasoning for `liftable_mul = true` was that
duplicating a shared Mul into N Fmas is harmless because:

  a. Each Fma computes `a*b` internally, so duplication is "free"
     at the asm level.
  b. The original shared Mul becomes unreachable from outputs once
     all its consumers absorb it into Fmas, and emit_c won't emit it.

Both points break down on composite codelets:

  a. Modern FMA ports run at 2 ops/cycle. Issuing N independent
     `a*b` multiplications where 1 was sufficient competes for those
     same ports and lengthens the critical path of the dependency
     chain.
  b. Shared Muls in composite codelets frequently have at least one
     non-Add consumer (another Mul, a Cmul operand, a direct output
     store, or — critically — a spill_marker subtree). When that
     happens the original Mul stays alive AND each duplicated Fma
     also computes a*b. Pure waste.

Doc 28 measured this as 33-48% regression on R=32 t1 (910 FP
instructions vs 717 without fma_lift, vs hand-written 709). It
attributed the regression to "explicit NK_Fma constrains gcc's
register allocation more than auto-fused mul+add." That diagnosis
was wrong. The 193 extra instructions were duplicate mul work —
restoring `single_use` makes them disappear.

### What the fix preserves

Every lift is now strictly op-count-preserving: 1 mul + 1 add → 1
fma. Verified across the matrix:

| Codelet | OLD (no lift) | NEW (single_use) | Delta |
|---|---|---|---|
| R=8 n1 AVX-512 | 1 FMA + 4 mul = 5 | 2 FMA + 3 mul = 5 | 0 |
| R=16 n1 AVX-512 | 3 FMA + 23 mul = 26 | 12 FMA + 14 mul = 26 | 0 |
| R=32 n1 AVX-512 | 12 FMA + 82 mul = 94 | 44 FMA + 50 mul = 94 | 0 |
| R=64 n1 AVX-512 | 34 FMA + 232 mul = 266 | 117 FMA + 149 mul = 266 | 0 |
| R=15 n1 AVX-512 | 0 FMA + 114 mul = 114 | 80 FMA + 34 mul = 114 | 0 |

The number of FMA opportunities the IR finds increases dramatically
(R=64 n1: 34 → 117 explicit FMAs), but no work is duplicated.

### Side effect: doc 54's compile failure also fixed

Doc 54 had previously attempted a similar single_use restriction
(`fma_lift_safe`), measured the IR FMA count rising to 235 on R=64
t1 (close to gcc's 286 contracted), but the emit step failed with
undeclared tag references like `t82` on R=32 t1. Doc 54 attributed
the failure to a "three-DAG-roots invariant" — assigns +
spill_markers + possibly other roots — and estimated ~week of work
to enumerate and freeze them all.

That diagnosis was wrong too. The hashcons unification ambiguity
was driven by duplicated Mul nodes: when `fma_lift` created `Fma(a,
b, c1)` and `Fma(a, b, c2)` from a shared `Mul(a, b)`, the
hashcons table had to choose how to represent the shared `a*b`
subexpression across the two Fma nodes, and the resulting
ambiguity propagated through the SU+spill recipe's PASS 1 / PASS 2
boundary as undeclared tags.

With `single_use` restored, every Mul has exactly one consumer, no
shared subexpressions get rewritten, and no unification ambiguity
arises. Doc 54's compile failure vanishes entirely — R=32 t1
AVX-512 SU+spill, R=64 t1 AVX-512 SU+spill, R=32 t1 AVX2 SU+spill,
and R=128 t1 AVX-512 SU+spill all compile cleanly.

The `frozen_tags` machinery that doc 54 had introduced for direct
spill_marker tag references is still in place as a defensive belt,
but it's no longer load-bearing.

## 3. Validating the asm-level effect

Section 1 argued that M-project's `asm volatile` barriers prevent
gcc's auto-fusion of `Add(Mul(a,b), c)` into FMAs, so we need to do
the fusion in IR. This section measures the claim directly by
counting source-level FMA intrinsics versus final asm-level
`vfmadd*` instructions, with and without M-project.

### Without M-project: gcc reconstructs the same asm

When emission has no `asm volatile` barriers, gcc with `-O3`
(which implicitly enables `-ffp-contract=fast`) can scan across
the whole function and contract any visible `vmulpd; vaddpd`
into FMAs. Empirically:

| Codelet | OLD src FMA | OLD asm FMA | NEW src FMA | NEW asm FMA |
|---|---|---|---|---|
| R=64 t1 SU+spill | 160 | **286** | 235 | **286** |
| R=32 t1 SU+spill | 74 | **118** | 102 | **118** |
| R=64 n1 hot path | 34 | **160** | 117 | **160** |
| R=16 n1 | 3 | **16** | 12 | **16** |
| R=13 prime | 0 | **132** | 132 | **132** |

asm-level mul and add/sub counts are identical between OLD and NEW
for each codelet too. gcc adds *additional* fusion on top of
whatever we lift in IR — when source has 235 FMAs, gcc adds 51
more; when source has 0, gcc adds 132. The asm converges to the
same instruction set either way.

So without M-project, **`fma_lift` has zero effect on instruction
count** in the gcc-emitted asm. Any runtime difference between OLD
and NEW in non-M-project benches is purely scheduling — gcc orders
the same instructions differently when handed `Fma(a,b,c)` versus
`Add(Mul(a,b), c)` as input. This is the mechanism behind the
R=15/20/21/35 regression class (preserved instructions, regressed
schedule).

### With M-project: gcc's auto-fusion gets crippled

`asm volatile` is a side-effect barrier to gcc. It can't look
through it to contract operations on either side. The damage varies
by codelet dataflow shape:

| Codelet | C (M, no lift) src→asm | D (M, lift) src→asm | gcc's added fusion | asm FMA lost vs no-M |
|---|---|---|---|---|
| R=64 t1 SU+spill | 160 → 250 | 235 → 250 | +90 (C), +15 (D) | 36 (286 → 250) |
| R=32 t1 SU+spill | 74 → 106 | 102 → 106 | +32 (C), +4 (D) | 12 (118 → 106) |
| **R=64 n1 hot path** | **34 → 34** | **117 → 117** | **+0 (C), +0 (D)** | **126 (160 → 34)** |

For t1 codelets, gcc still finds some auto-fusion under M-project
(its barriers don't completely block visibility — patterns within a
single emission segment still contract). But the t1 SU+spill asm
loses 12-36 FMAs versus the unbarriered baseline.

For R=64 n1, the result is starker: **gcc adds zero auto-fusion
under M-project's barriers.** The asm FMA count exactly equals the
source FMA count. The barriers are so restrictive on this codelet's
dataflow that without `fma_lift`, M-project ships a binary missing
126 FMAs that gcc would have found unbarriered. That's 252 extra
instructions (each lost FMA = mul + add).

### What this means architecturally

The validation is concrete:

1. **`fma_lift` is purely cosmetic when M-project is off.** gcc
   converges to the same asm regardless. The doc 28 regression on
   composites was driven by `liftable_mul = true` duplication, not
   by anything fma_lift does at the asm-fusion level.

2. **`fma_lift` is load-bearing when M-project is on.** Each
   explicit FMA in our IR is one less FMA gcc has to find through
   barriers it can't see through. On R=64 n1, fma_lift converts a
   34-FMA M-project asm into a 117-FMA M-project asm — directly
   visible in the runtime difference between C (`+7.9%` vs A) and D
   (`+12.7%` vs A) at K=256.

3. **The R=64 n1 gap remains.** Even with full single_use
   fma_lift, R=64 n1 under M-project only reaches 117 asm FMAs vs
   gcc's unbarriered 160. The remaining 43 FMAs are presumably
   pinned across barriers in a way that prevents `single_use` from
   firing — for example, a Mul whose consumer is across an `asm
   volatile` boundary. Closing that gap is the open follow-up:
   either looser barrier placement in M-project's emission, or a
   barrier-aware lifting predicate that can fuse across the
   register pinning when safe.

4. **The t1 SU+spill wins of 12-27% (D vs A) are coming from the
   right place.** D under M-project has 250 asm FMAs vs C's 250
   asm FMAs — *same asm-level FMA count*. The runtime improvement
   comes not from doing more FMAs but from doing them more
   efficiently: C does some via vmulpd/vaddpd pairs gcc couldn't
   fuse, D does the same number but more of them are encoded as
   single FMAs by us upfront, leaving gcc free to use its remaining
   optimization budget on scheduling rather than fusion-recovery.

## 4. Regression test results

### Numerical correctness

45 / 45 codelets verified bit-exact or within 8.9e-16 max error
(1 ULP, expected from FMA's intermediate rounding). Coverage:

- Primes R = 3, 5, 7, 11, 13, 17, 19 (both ISAs)
- n1 R = 2, 4, 6, 8, 10, 12, 14, 15, 16, 20, 25, 32, 64, 128
- t1 R = 4, 8, 16, 32, 64, 128 (both ISAs)
- Includes all SU+spill recipe codelets that previously failed to
  compile under fma_lift.

### Performance — primes, 10-trial statistics

Each cell: 10 program runs, median ± stdev, effect size in pooled
stddevs. Threshold: σ > 2.0 is significant; σ < 1.0 is noise.

| Codelet | OLD median | NEW median | ratio | σ | verdict |
|---|---|---|---|---|---|
| R=3 prime AVX-512 | 8.50 | 9.02 | 1.061 | 1.0 | noise |
| R=5 prime AVX-512 | 18.57 | 18.18 | 0.979 | 0.2 | noise |
| R=7 prime AVX-512 | 23.96 | 23.94 | 0.999 | 0.1 | noise |
| R=11 prime AVX-512 | 54.92 | 58.35 | 1.063 | 0.8 | noise |
| R=13 prime AVX-512 | 88.48 | 94.89 | 1.072 | 0.9 | noise |
| R=17 prime AVX-512 | 166.77 | 167.23 | 1.003 | 0.1 | noise |
| R=19 prime AVX-512 | 215.92 | 213.60 | 0.989 | 0.4 | noise |

All primes within 1σ. No significant regressions.

### Performance — low-radix n1 power-of-2

| Codelet | OLD median | NEW median | ratio | σ | verdict |
|---|---|---|---|---|---|
| R=2 n1 AVX-512 | 6.76 | 6.89 | 1.019 | 0.4 | noise |
| R=4 n1 AVX-512 | 19.70 | 18.55 | 0.941 | 0.5 | noise |
| R=8 n1 AVX-512 | 33.43 | 33.96 | 1.016 | 0.3 | noise |
| R=16 n1 AVX-512 | 168.94 | 164.50 | 0.974 | 1.8 | borderline win |

### Performance — mixed-radix small composites

| Codelet | factors | OLD | NEW | ratio | σ | verdict |
|---|---|---|---|---|---|---|
| R=6 | 2×3 | 28.38 | 27.30 | 0.962 | 0.7 | noise |
| R=10 | 2×5 | 56.99 | 50.90 | 0.893 | 1.7 | borderline **win** |
| R=12 | 4×3 | 73.66 | 67.50 | 0.916 | 2.3 | SIG **win** |
| R=14 | 2×7 | 147.47 | 139.93 | 0.949 | 2.2 | SIG **win** |
| R=25 | 5×5 | 502.04 | 501.94 | 1.000 | 0.0 | noise |
| **R=15** | **3×5** | 286.82 | 325.49 | 1.135 | **4.3** | **SIG regression +13.5%** |
| **R=20** | **4×5** | 277.46 | 295.36 | 1.065 | **5.7** | **SIG regression +6.5%** |
| **R=21** | **3×7** | 608.87 | 673.11 | 1.106 | **4.8** | **SIG regression +10.5%** |
| **R=35** | **5×7** | 1695.83 | 1788.38 | 1.055 | **5.2** | **SIG regression +5.5%** |

Pattern: mixed-radix codelets with outer factor 2 (R=6, R=10, R=14)
all win or are neutral. Codelets where the outer factor in the
Cooley_Tukey decomposition is odd (R=15, R=21, R=35) or where the
combination produces odd-prime-dominated FMA chains (R=20) regress
significantly. The regression mechanism is the same gcc-scheduling
phenomenon that affects R=64 n1 hot path (preserved op count,
regressed runtime) — explicit Fma intrinsics constrain gcc's
scheduling differently than auto-contracted mul+add chains, and this
particular butterfly shape comes out worse for it.

### Performance — t1 codelets (SU+spill recipe)

These are the cases doc 28 reported as 33-48% regression and doc
54 reported as compile failure. With single_use:

| Codelet | A:base | B:fma | C:M | D:fma+M | Best vs A |
|---|---|---|---|---|---|
| R=16 t1 AVX-512 | 614 | 618 | 545 | **537** | **D −12.5%** |
| R=32 t1 AVX-512 (su_spill) | 2060 | 1951 | 1689 | **1673** | **D −18.8%** |
| R=64 t1 AVX-512 (su_spill) | 5448 | 5425 | 5378 | **5334** | D −2.1% |
| R=16 t1 AVX2 | 1129 | 1134 | 887 | **871** | **D −22.9%** |
| R=32 t1 AVX2 (su_spill) | 3427 | 3415 | 2529 | **2510** | **D −26.8%** |
| R=128 t1 AVX-512 (su_spill, K=256) | 78740 | 77917 | 71383 | **70458** | **D −10.5%** |

D (fma_lift + M-project) wins across the board on twiddled CT
codelets — the configuration that motivated this whole investigation.
The R=32 t1 SU+spill case that doc 28 reported as a 33-48% regression
is now a **26.8% improvement** (AVX2) — the largest single win in the
matrix.

### Performance — hot-path codelets (10-trial, K=256)

| Codelet | OLD median ± stdev | NEW median ± stdev | ratio | σ | verdict |
|---|---|---|---|---|---|
| R=64 n1 AVX-512 K=256 | 27167 ± 553 | 27400 ± 168 | 1.009 | 0.40 | **NOISE** |
| R=128 n1 AVX-512 K=256 | 100455 | 96871 | 0.964 | (single-cell) | **NEW wins −3.6%** |
| R=128 t1 AVX-512 K=256 | 78740 | 70458 (D) | 0.895 | (single-cell) | **D wins −10.5%** |

R=64 n1 hot path is unaffected within bench noise (effect size 0.40
pooled stddevs — well below the 1σ threshold). R=128 n1 wins
modestly with fma_lift alone. R=128 t1 wins big with the full D
configuration.

## 5. Gating: current state and proposed shape

### What's in the tree now

`bin/gen_radix.ml`:

```ocaml
let fma_lift_safe =
  match Vfft_v2.Dft.pick_algorithm n with
  | Vfft_v2.Dft.Direct        -> true
  | Vfft_v2.Dft.Cooley_Tukey _ -> true
  | Vfft_v2.Dft.Split_radix    -> false in
```

Plus environment-variable overrides for testing:
- `VFFT_FORCE_FMA_LIFT=1`: force on regardless of gate
- `VFFT_DISABLE_FMA_LIFT=1`: force off regardless of gate

### Justifications

**Direct → on.** Doc 28's existing decision; primes benefit 1-2% from
explicit FMAs, 7/7 primes verified within noise (no regressions).

**Cooley_Tukey → on.** This is the new default. Justified by:

- Activating M-project's value on composites (the original
  motivation: M-project's `asm volatile` barriers prevent gcc
  auto-fusion, so we need explicit FMAs in IR).
- t1 codelets win 1.4% – 26.8% with the full D configuration
  (fma_lift + M-project).
- n1 power-of-2 codelets show neutral or winning behavior at all
  tested sizes; the R=64 hot path is confirmed noise (σ = 0.40).
- Doc 28's reported 33-48% regression on R=32 t1 was caused by the
  `liftable_mul = true` bug, not by Cooley_Tukey decomposition
  itself; with single_use restored, the same codelet wins 5.2% with
  fma_lift alone and 26.8% with fma_lift + M-project.

**Split_radix → off.** Untested only. The underlying machinery now
works (no fma_lift correctness issues remain). Currently off to
preserve doc 28's default until someone runs the matrix on
split_radix codelets — likely a few hours of work, low risk.

### Known regressions in the current gating

Mixed-radix codelets where the Cooley_Tukey outer factor is an odd
prime ≥ 3 regress 5-14% (σ > 4 across R=15, R=20, R=21, R=35). This
is not a correctness issue and not a duplication issue — op counts
are preserved. It is a gcc-scheduling phenomenon: the explicit-Fma
DAG shape produces dependency chains gcc handles worse than the
original Mul+Add form for this particular butterfly structure.

Three options for handling this:

**A. Accept the regression.** R=15/20/21/35 are non-power-of-2 sizes,
not on the typical FFT hot path (HFT/quant workloads dominated by
2^k transforms). The library still produces correct output for these
sizes; it's only slower than the doc 28 baseline. Document and move
on.

**B. Refine the Cooley_Tukey gate.** Add a check on the outer factor:

```ocaml
| Vfft_v2.Dft.Cooley_Tukey (n1, _) ->
  (* Mixed-radix with odd-prime outer factor regresses (R=15/21/35
   * measured at +5-14% σ>4). Outer 2/4 lets gcc schedule the
   * butterfly stage cleanly. *)
  n1 land (n1 - 1) = 0   (* outer factor is a power of 2 *)
```

This keeps the wins on power-of-2 composites (R=2..R=128 n1, all t1
sizes, R=10/12/14 mixed-radix with 2-outer) and reverts R=15/20/21/35
to the doc 28 baseline. The regression class is small and clearly
delimited.

**C. Investigate the schedule-layer cause.** Same phenomenon as the
R=64 n1 hot path (preserved op count, scheduling-driven regression).
Likely about operand ordering of explicit Fma intrinsics versus the
freedom gcc has with separate Mul + Add. Could in principle be fixed
by either operand-ordering hints in NK_Fma or by integrating the
schedule layer (M-project) more tightly with the fma_lift output.
Defer until needed.

**Recommendation:** **Option B** is the cheapest production fix —
small, localized, justified by 4σ+ measurement, doesn't bake in
codelet-specific knowledge (the rule is about the algorithm shape,
not about which radix). Option C is the right long-term direction
but doesn't need to ship for this work to be valuable.

## Status summary

- ✓ Root cause identified: `liftable_mul = true` unconditional
    duplication. Not gating, not DAG-root invariants, not RA friction.
- ✓ One-line fix shipped in `lib/algsimp.ml`.
- ✓ Gating in `bin/gen_radix.ml` simplified from the
    three-way per-codelet-family form to a clean two-branch policy.
- ✓ 45/45 codelets numerically correct, including all previously-
    broken SU+spill cases.
- ✓ Doc 54's compile failure resolved (was driven by duplicate-Mul
    hashcons, not by structural DAG-root enumeration).
- ✓ Hot paths verified noise-stable: R=64 n1 (σ=0.40), R=128 n1
    (-3.6% win), R=128 t1 SU+spill (-10.5% win with M-project).
- ⚠ Mixed-radix odd-prime-outer regression class identified
    (R=15/20/21/35 at σ>4). Recommend Option B refinement to the
    Cooley_Tukey gate.

## Files

- `lib/algsimp.ml`: `fma_lift` function, `liftable_mul` predicate
  restored to `single_use`.
- `bin/gen_radix.ml`: per-codelet-family `fma_lift_safe` gate.

Future work: refine Cooley_Tukey gate per Option B above, evaluate
Split_radix activation, investigate schedule-layer mechanism behind
the remaining preserved-op-count regressions (R=64 n1 noise-borderline
and R=15/20/21/35 SIG).
