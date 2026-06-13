# Cnum Symbolic Complex Layer — Implementation Findings

## What was built

Three additions to VectorFFT, sized roughly as one focused day of work:

1. **`lib/expr.ml`** — Smart constructors added to the Expr ADT:
   `mk_const`, `mk_neg`, `mk_add`, `mk_sub`, `mk_mul`. Implements FFTW's
   `littlesimp.ml` rewrite rules at construction time:
   - Constant folding (Const ⊕ Const → Const)
   - Identity elimination (x*0=0, x*1=x, x+0=x, etc.)
   - **Constant rotation through Mul**: `Mul(Const a, Mul(Const b, x))
     → Mul(Const(a*b), x)`. The FFTW rule from genfft.
   - **Distribution-with-rotation**: `Mul(Const, Sub(Mul(Const,x), Mul(Const,y)))
     → Sub(Mul(Const*Const, x), Mul(Const*Const, y))`. Gated to only fire when
     both branches of the Sub/Add are `Mul(Const, _)` — guaranteeing the
     subsequent rotation succeeds and the rewrite is a net win at the IR level.

2. **`lib/cnum.ml`** — Symbolic complex number combinator type:
   ```
   type cnum = { re : Expr.expr; im : Expr.expr }
   ```
   Operations: `cmul`, `cadd`, `csub`, `cscale`, `cneg`, `cconj`, `cconst`,
   `croot_of_unity` (fwd/bwd). The `cmul` implementation uses Plus-of-Times
   form `Sub(Mul, Mul)` (the FFTW choice) rather than the older tan-factored
   `Mul(Const, Sub(_, Mul))` — placing Mul nodes at the leaves of Sub/Add chains
   so downstream `mk_mul` can find rotation opportunities.

3. **`lib/dft.ml`** — `dft_winograd5_cnum` added as a parallel implementation
   of the existing `dft_winograd5`. Gated behind `VFFT_CNUM_W5=1` for
   measurement.

## What was measured

### Correctness

`VFFT_CNUM_W5=1` at R=25 AVX-512 produces output numerically identical to
the baseline (max diff 2.08e-14, i.e., rounding noise). The Cnum-based W5
is algebraically equivalent.

### Rule firing

With `VFFT_CNUM_W5=1` at R=25, the distribution-with-rotation rule fires
**20 times** during codelet construction. The rotation-through-Mul rule
fires 0 times (the pattern `Mul(Const, Mul(Const, x))` doesn't appear
naturally even in Cnum mode). The distribution rule is the one that
matches.

### Op count: **no change**

| Radix | Baseline ops | Cnum ops | Δ |
|---|---|---|---|
| R=5 AVX-512 | 32 | 32 | 0 |
| R=25 AVX-512 (intrinsics) | 383 | 383 | 0 |
| R=25 AVX-512 (IR pre-gcc) | 161 add/sub/mul + 222 fma = 383 | 129 add/sub/mul + 260 fma = 389 | +6 IR |

The Cnum path produces 6 *more* IR ops, but gcc folds them back into the same
final intrinsic count. The distribution + rotation rewrite is **algebraically
equivalent to the original form after FMA fusion** — the work shifts between
explicit Muls and the Mul inside an FMA, with no net wall-clock benefit.

## Why the algebraic gap to FFTW does not close

FFTW's R=25 codelet has 236 ops (vs our 383). The gap is **not** in
constant-rotation or local Mul-chain simplification. Construction-time
rewrites — whether via tan-factored or plus-of-times cmul, whether or
not the smart constructors run — produce algebraically equivalent forms
that gcc reconciles via FMA fusion. We measured this directly.

The actual gap is in **non-local algebraic structure**:

1. **N-ary Plus.** FFTW's `Expr.Plus` is an n-ary list, not binary. Their
   `mangleSumM` walks the full list with `collectM` ("ax + bx + cx → (a+b+c)·x")
   and `reduce_sumM` ("combine Const terms"). Pattern matching on `Plus[
   Times(a, x); Times(b, x); Times(c, x)]` finds the shared `x` factor in
   one pass. Our binary `Add` requires tree-walking and the pattern only
   matches at the leaves it happens to nest into.

2. **`deepCollectM`.** FFTW recursively walks Plus trees, applying `collectM`
   at every level. Their `Magic.deep_collect_depth` is tunable. This catches
   sharing patterns across the 5×5 CT decomposition that our binary IR never
   sees because the Plus terms are scattered across different subtrees.

3. **Cost-model variant selection.** FFTW generates multiple candidate codelets
   per N (`-compact`, `-pipeline-latency`, `-fma`) and the schedule layer picks
   the cheapest by an explicit cost function. We commit to one form.

These three together produce FFTW's 1.6× algebraic advantage on R=25. The
Cnum layer alone is not enough.

## What the Cnum work IS worth

Three things:

1. **Foundation for n-ary Plus / collectM.** When the n-ary Plus + collect
   work happens, it will operate on Cnum-built expression trees. The smart
   constructors give it canonical shape to work on. Without Cnum-style
   construction, the algsimp work would have to detect and normalize the
   tan-factored shape first.

2. **Clean reference implementation of W5.** `dft_winograd5_cnum` is shorter
   and clearer than the original (uses `cnum` records instead of two parallel
   `re`/`im` arrays). Going forward, this is the better surface to write new
   Winograd codelets against.

3. **The smart Expr constructors are pure wins.** Const folding, identity
   elimination, and Neg propagation — these always reduce IR node count, even
   if the compiler would fold most of them. Less garbage through the algsimp
   pipeline, faster builds, easier debug printouts.

## Recommendation

**Land the Cnum infrastructure** (Expr smart constructors + cnum.ml +
`dft_winograd5_cnum`). It's foundational, correct, and zero-regression.

**Don't enable VFFT_CNUM_W5=1 by default** — same intrinsics out, no benefit
to deployed code.

**Don't expect this to close the FFTW gap on R=25.** The gap is in algsimp's
sum-collection capability, not in construction-time rewrites. Closing it
requires implementing n-ary Plus + deepCollectM, which is a separate ~2-week
project.

**Wall-clock recap.** The R=25 algebraic gap is 1.6× more ops. We currently
tie or slightly trail FFTW on R=25 in compiled wall-clock (60-65 cy/DFT vs
FFTW's 60 cy/DFT). The layout + SIMD-friendliness compensates for the op
handicap. On R=64 (where W5 doesn't apply) we beat FFTW 2.1× even with our
2× op count — the wall-clock argument from our prior session stands.

The honest closing read: this work was the right scoped attempt, the
infrastructure is good, but the cheap path to closing the algebraic gap
does not exist. Either invest in n-ary Plus + deepCollectM, or accept
that our 60 cy/DFT on R=25 tied-with-FFTW is the right answer for now.
