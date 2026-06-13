# 58. Closing the Op-Count Gap — From Hand-Coded Python to FMA-Saturated OCaml

Status: **`round_13` fix landed in `lib/dft.ml`'s `const_cmul`** (this session).
Cumulative op-count savings at R=64 / 128 / 256 verified, all radices
correctness sub-ulp (≤0.4 ulps relative to R²). Latency benchmarked against
the old `gen_radix32.py` hand-coded codelets: **13-16% faster single-call,
33-35% faster steady-state on R=32 AVX-512.**

## TL;DR

The session began with one question — *why are we +14 over FFTW at R=32 and
+66 at R=64?* — and an explicit ask to "compare t1 codelets vs FFTW, then do
common-twiddle factoring." Over the investigation we:

1. **Traced the +66 gap at R=64 to two root causes**: a 1-ulp precision
   divergence in OCaml's `sin(π/8)` vs `cos(3π/8)` (mathematically equal,
   bit-different) blocking hashcons sharing of `tan(π/8)` ratios across
   symmetric-angle rotations; and a structural residual of `mul_in_fma_addend`
   patterns (the general complex-mul form `K1·X + K2·Y` with K1≠K2) that have
   no algebraic reduction available.

2. **Shipped one fix**: round `cr` and `ci` to 13 sig digits *before* dividing
   them in `const_cmul`'s Path B (tan-factored form). This canonicalizes
   symmetric angles so the resulting inner ratios are bit-identical, letting
   hashcons unify downstream Muls. Saves **16 ops at R=64, 76 at R=128, 249
   at R=256** with zero algorithmic change.

3. **Disproved three plausible-sounding hypotheses**: (a) frozen spill tags
   were blocking absorption — false, zero frozen disqualifications in
   `multi_use_fma_lift`; (b) the 2-pass spill recipe was costing ops vs
   monolithic — false, the delta is 4-10 ops at R=128/256, irrelevant at
   smaller radices; (c) the CT factorization shape was eating ops —
   *partially true* (CT(4, 16) at R=64 produces 950 ops vs CT(8, 8)'s 978),
   but a wall-clock bench showed the lower-op factorization is **13.8%
   slower** due to register pressure and dependency-chain depth — a sharp
   illustration that op count and latency are different optimization
   targets.

4. **Final state**:
   - R=8/R=16: exact match with FFTW (n1 and t1)
   - R=32: 386 / 510 ops (FFTW 372 / 496), gap +14 on both
   - R=64: 978 / 1230 ops (FFTW 912 / 1164), gap +66 on both
   - R=128/256: no FFTW reference; cumulative session savings 76/249 ops at n1
   - vs old `gen_radix32.py`: 386 vs 443 (n1), 510 vs 567 (t1) — **−57 ops
     each**, with 4× more FMAs and zero standalone xor-pd negates

5. **Where the remaining gap lives**: FFTW's `gen_notw.native` uses 22
   unique constants at R=64 to our 16, including ratios like KP471396736
   and KP668178637 that arise from split-radix or conjugate-pair split-radix
   decomposition. Their op count is achievable with that algorithm; ours
   currently isn't because our `lib/split_radix.ml` produces *worse* output
   (1144 ops at R=64 vs CT 978). Closing the rest of the gap is a separate
   algorithm-layer workstream, not a codegen issue.

## The R=64 mystery and where the +66 lives

At session start the picture was:

```
Radix | ours n1 | FFTW n1 | gap | ours t1 | FFTW t1 | gap
------|---------|---------|-----|---------|---------|-----
R=8   | 52      | 52      | +0  | 80      | 80      | +0
R=16  | 144     | 144     | +0  | 204     | 204     | +0
R=32  | 386     | 372     | +14 | 510     | 496     | +14
R=64  | 994     | 912     | +82 | 1246    | 1164    | +82
```

The n1 and t1 gaps are identical (+14 and +82) because they share the inner
DFT structure — t1 is just n1 plus 63 runtime twiddle multiplies (each 4
ops: 2 muls + 2 FMAs, exactly matching FFTW's pattern). So everything
interesting about the gap lives in the n1 inner DFT.

Decomposing our R=64 n1 = 994 ops:

```
ours:  624 add+sub + 66 muls + 304 FMA = 994
FFTW:  520 add+sub +  0 muls + 392 FMA = 912

delta: +104 add+sub, +66 mul, −88 FMA
```

FFTW has zero standalone muls; every multiplication is folded into an FMA
(either as the mul slot or by emitting `_mm512_fmadd_pd(...)` directly with
the multiplication implicit). We have 66 standalone `_mm512_mul_pd` plus the
inline muls inside FMA addend slots.

Classifying our 36 standalone muls by consumer pattern:

```
Type 1 (FMA-addend only): 20 muls
   Pattern: result fed as the c-operand of ≥1 FMA, like
     t2669 = mul(K, expr)
     ...
     t3120 = fmsub(K2, X, t2669)     // t2669 is FMA addend
     t3184 = fmadd(K2, X, t2669)
   These are STRUCTURALLY UNABSORBABLE: there's no FMA opcode that does
   K1·X + K2·Y in one instruction. Per pair: 1 mul + 2 FMAs = 3 ops minimum,
   same as what FFTW emits.

Type 2 (plain Add/Sub only): 16 muls
   Pattern: result consumed by Add/Sub directly, like
     t3127 = mul(K, sub(X, Y))
     t3131 = add(t3127, t3130)
     t3279 = sub(t3130, t3127)
   These SHOULD be absorbable by multi_use_fma_lift — the pass exactly
   handles this case (multi-use Mul with Add/Sub-only consumers, rewrites
   each consumer as FMA, eliminating the Mul).
```

The Type 2 muls were the obvious lever: 16 ops at R=64 if we could absorb
them.

## False lead: the frozen-Mul hypothesis

First hypothesis: `multi_use_fma_lift` refuses to absorb frozen Muls
(`lib/algsimp.ml:2072`: `if is_frozen m.tag then disqualify m.tag`). The
spill recipe marks certain Muls as spill targets — could those be the 16
Type-2 muls?

Added a `disqualify_reason` Hashtbl, instrumented each `disqualify` call
site with a labeled reason, and traced. The result was unambiguous: across
all 4 invocations of `multi_use_fma_lift` (mfl1 through mfl4), **zero Muls
were disqualified for `frozen`**. The disqualifications all came from
`mul_in_fma_addend` (46/50 in the final pass) and `mul_in_fma_b` (4/50) —
both Type-1 patterns that legitimately can't be absorbed.

So the 16 Type-2 muls weren't being disqualified by the existing logic at
all. They simply didn't exist as Type-2 patterns in the IR — they only
appeared that way in the *emitted C output*. Something was producing
multiple Mul nodes with the same value but distinct tags, blocking hashcons
unification at the IR layer.

## The actual cause: 1-ulp precision divergence in symmetric angles

R=64 uses CT(8, 8) — 8 columns of DFT-8 in PASS 1, 49 non-trivial twiddle
multiplies between passes, 8 rows of DFT-8 in PASS 2. The twiddle for
`(n1_idx, k2) = (4, 1)` has angle θ = π/8; for `(n1_idx, k2) = (12, 1)`
the angle is 3π/8. Both rotations require `tan(π/8)` as the inner ratio
in Path B's tan-factored form, since `cos(3π/8)/sin(3π/8) = sin(π/8)/cos(π/8)
= tan(π/8)`.

In exact arithmetic those are identical. In OCaml's FP:

```
cos(π/8)   = 0.92387953251128674   ← used by ω^4 case (|cr| > |ci|)
sin(3π/8)  = 0.92387953251128674   ← same bits — symmetric

sin(π/8)   = 0.38268343236508978   ← used by ω^4 case
cos(3π/8)  = 0.38268343236508984   ← DIFFERS by 1 ulp!
```

The `sin(π/8)` vs `cos(3π/8)` mismatch is a libm artifact: `sin` and `cos`
of mathematically-equal angles are computed via different reduction paths
and the last bit drifts.

Path B (`lib/dft.ml:274-326`) then divides:

```ocaml
let tn = ci /. cr in           (* ω^4:  sin(π/8) / cos(π/8) → tan(π/8) variant A *)
let ct = cr /. ci in           (* ω^12: cos(3π/8) / sin(3π/8) → tan(π/8) variant B *)
```

Variant A: `0.38268343236508978 / 0.92387953251128674 = 0.41421356237308998`
Variant B: `0.38268343236508984 / 0.92387953251128674 = 0.41421356237309003`

After `mk_const`'s `Printf.sprintf "%.13e"` rounding, A and B *still* round
to different 14-sig-fig representations because they straddle the rounding
boundary at digit 14. Result: two distinct Const nodes in the IR for what
should be one shared constant. Hashcons can't unify them. Downstream Muls
that use these constants stay as separate Mul nodes per variant, doubling
the standalone-mul count for every twiddle pair affected.

## The fix: input rounding in `const_cmul`

The structural fix is to round `cr` and `ci` to 13 sig digits *before*
computing the ratio. After rounding:

```
r13(sin(π/8))  = 0.38268343236509
r13(cos(3π/8)) = 0.38268343236509   ← now equal!

(min / max) computed from these rounded inputs produces bit-identical
results across symmetric angles.
```

The code change in `lib/dft.ml:302-326`:

```ocaml
let round_13 x = float_of_string (Printf.sprintf "%.13e" x) in
let cr_r = round_13 cr in
let ci_r = round_13 ci in
let acr = abs_float cr_r in
let aci = abs_float ci_r in
let r_abs = (min acr aci) /. (max acr aci) in
(* ...use cr_r, ci_r, r_abs to build the IR... *)
```

Why this works where the earlier attempt (canonicalizing `r_abs = min/max`
after the division) didn't: rounding the *output* of the division doesn't
help because the inputs to the division differ at the 17th decimal, which
amplifies through the division into a 1-ulp difference at the 14th decimal
of the result — exactly at the boundary `%.13e` rounding will preserve.
Rounding the *inputs* before the division ensures the divisions themselves
become bit-identical.

The numerical cost is a single ulp of constant precision (we lose ~10⁻¹³
relative accuracy on the unified ratio). For an N=64 FFT that's well below
the algorithm's intrinsic error bound; correctness verification post-fix
showed max errors of `1.072e-13` at R=64 (0.1 ulp relative to R²),
indistinguishable from pre-fix accuracy.

### Savings

```
                 before round_13 → after
R=64    n1:      994             → 978     (−16 ops)
R=64    t1:      1246            → 1230    (−16 ops)
R=128   n1:      2404            → 2328    (−76 ops)
R=128   t1:      2912            → 2836    (−76 ops)
R=256   n1:      5709            → 5460    (−249 ops)
R=256   t1:      6729            → 6480    (−249 ops)
R=8/R=16/R=32:   unchanged (no symmetric-angle twiddles affected at these sizes)
```

The savings scale: more symmetric-angle pairs exist at larger N, so
hashcons unification triggers more often.

## False lead: 2-pass blocking as the cause

The user proposed: *our R=32/R=64 are 2-pass blocked (PASS 1 + spill +
PASS 2) while FFTW emits a monolithic codelet — could that explain the
gap?* It's a plausible hypothesis: a pass boundary could prevent algsimp
from absorbing Muls whose value crosses it.

Tested by running with `--no-recipe` (forces monolithic emission, disables
spill markers):

```
            2-pass    monolithic   delta
R=32   n1   386       386          0
R=64   n1   978       978          0
R=128  n1   2328      2324         −4
R=256  n1   5460      5450         −10
```

The 2-pass blocking adds essentially zero op-count overhead at R=32/R=64
and only 4-10 ops at the largest radices. It's a wash. The gap to FFTW
isn't structural to our blocking design — it's algorithmic.

The reason this is so well-behaved: `extend_frozen` propagates each pass's
`tag_remap` through the chain (lib/gen_radix.ml:384), so when a frozen
Add/Sub gets rewritten to an Fma in PASS 1, the spill marker follows the
new tag. The blocked codelet is byte-identical in arithmetic content to
the monolithic version; only the placement of intermediate scratch slots
differs.

## False lead with a real lesson: CT factorization shape

Different CT factorizations of the same N produce different op counts.
Sweeping the choices at R=32, R=64, R=128:

```
R=64 factorizations:
  CT(2, 32)   960 ops   mul=20  fma=432  add+sub=508
  CT(4, 16)   950 ops   mul=30  fma=368  add+sub=552   ← lowest
  CT(8, 8)    978 ops   mul=50  fma=336  add+sub=592   ← current default
  CT(16, 4)   972 ops   mul=52  fma=344  add+sub=576
  CT(32, 2)   988 ops   mul=48  fma=376  add+sub=564

R=32 factorizations:
  CT(2, 16)   380 ops   mul=0   fma=160  add+sub=220   ← lowest, zero muls
  CT(4, 8)    386 ops   mul=10  fma=128  add+sub=248   ← current default
  CT(8, 4)    394 ops   mul=18  fma=120  add+sub=256
  CT(16, 2)   392 ops   mul=12  fma=136  add+sub=244
```

CT(4, 16) at R=64 saves 28 ops; CT(2, 16) at R=32 saves 6 ops *and* has
zero standalone muls. These looked like free wins.

A latency bench on the actual codelets told a different story:

```
R=32 single-call latency (Intel Xeon @ 2.10 GHz nominal, AVX-512):
  CT(4, 8)  current   404 cyc  (192.4 ns)
  CT(2, 16) candidate 438 cyc  (208.6 ns)   ← +8.4% slower despite −6 ops

R=64 single-call latency:
  CT(8, 8)  current   954 cyc  (454.4 ns)
  CT(4, 16) candidate 1086 cyc (517.2 ns)   ← +13.8% slower despite −28 ops

R=128 single-call latency:
  CT(8, 16) current   2218 cyc (1056.5 ns)
  CT(4, 32) candidate 2480 cyc (1181.2 ns)  ← +11.8% slower
```

The lower-op-count factorizations are **uniformly and substantially slower**
in wall-clock terms. Three mechanisms compound:

1. **Register pressure.** CT(8, 8) at R=64 has 8 independent DFT-8s in
   PASS 1 — each sub-DFT's working set fits in ~16 ZMM, well under
   AVX-512's 32-register file. CT(4, 16) has 4 sub-DFTs of size 16, each
   with ~2× the peak-live count. That's the regime where spills start
   appearing.

2. **Dependency-chain depth.** A 16-point DFT has one more butterfly
   layer than an 8-point DFT. The critical path through the codelet is
   *longer*, so even though total ops drop, throughput-per-cycle drops
   more because the OOO engine can't issue dependent ops back-to-back.

3. **ILP across sub-DFTs.** CT(8, 8) gives 8 independent computation
   streams that the scheduler can interleave across ports. CT(4, 16)
   gives only 4 streams, each 4× longer — fewer independent contexts
   to mine for parallelism.

The current factorization defaults (CT(4, 8), CT(8, 8), CT(8, 16),
CT(16, 16)) were chosen — possibly empirically by earlier benchmarking —
for latency, not op count. **They are correct.** The 28 ops "saved" by
CT(4, 16) at R=64 are an op-count fiction; the hardware doesn't see them.

This is one of the most important findings of the session and it
generalizes: op count is a useful proxy but not an objective. Anywhere
op count and latency disagree, *bench wins*.

## Old vs new: the cumulative effect

The bigger picture is harder to see when you optimize incrementally. The
old hand-coded `gen_radix32.py` (last touched 2025-05-08) produces a
codelet with these characteristics at AVX-512:

```
R=32 codelet, AVX-512, forward only
                  FMA   mul   add   sub   xor    total
Old n1 (Python)   32    56    172   172   11     443 ops
New n1 (OCaml)    128   10    124   124   0      386 ops    (−57 ops, −12.9%)

Old t1 (Python)   94    118   172   172   11     567 ops
New t1 (OCaml)    190   72    124   124   0      510 ops    (−57 ops, −10.1%)
```

Three things changed dramatically between the two generators:

- **FMA count quadrupled on n1 (32 → 128), nearly doubled on t1
  (94 → 190).** Every absorbed `Mul + Add` becomes one FMA instead of
  two instructions. The `multi_use_fma_lift` + `fma_addend_factor`
  passes find absorptions the old hand-coded version couldn't.
- **Standalone muls collapsed: 56 → 10 (n1), 118 → 72 (t1).** Most of
  these became inline FMA operands or were eliminated by hashcons after
  the `round_13` unification.
- **Sign-flip negates disappeared entirely: 11 → 0.** `Neg(Mul(K, X))`
  folds into the FMA opcode (fmadd → fnmadd, fmsub → fnmsub) at the IR
  level, so we never emit `_mm512_xor_pd(_, sign_flip)` for negation.
  The old code emitted these as separate port-5 micro-ops.

### Latency bench: old vs new

Same hardware, same AVX-512 host (Intel Xeon @ 2.10 GHz nominal), both
codelets called in-place (the old codelet aliases in==out cleanly because
it loads all inputs to registers before any store). 3-trial minimum:

```
R=32 n1 (no twiddle):
  Old (OOP, separate buffers)      226.5 ns single  /  198.6 ns steady
  Old (in-place via alias)         225.4 ns single  /  196.9 ns steady
  New (in-place by design)         190.0 ns single  /  127.3 ns steady
                                   −15.7% single    /  −35.4% steady

R=32 t1 (twiddled):
  Old (OOP, separate buffers)      314.0 ns single  /  281.3 ns steady
  Old (in-place via alias)         304.7 ns single  /  274.6 ns steady
  New (in-place by design)         265.4 ns single  /  184.4 ns steady
                                   −12.9% single    /  −32.8% steady
```

The latency improvement (13-16% single-call, 33-35% steady-state) is
larger than the op-count reduction (10-13%). Three reasons it
*over-delivers* on the op-count savings:

1. **Each FMA replaces two instructions, not one op.** The op-count
   accounting counts `Mul + Add` as 2 ops and one fused FMA as 1 op —
   a 50% reduction. But at the instruction level the savings are similar:
   2 separate instructions retire vs 1 FMA instruction. The op-count delta
   (−57 ops) understates the instruction-count delta because many of those
   −57 represent pair → FMA fusions, each saving 1 *instruction* even
   when the op count only shows the net.

2. **11 fewer port-5 micro-ops.** The disappeared xor-pd negates were
   port-5 ops competing with `vbroadcastq`, lane shuffles, and other
   port-5 work. Removing them creates port-5 headroom that the rest of
   the codelet can fill, lifting overall throughput.

3. **Shorter dependency chains.** Mul → Add is a 2-instruction critical
   path step (the Add can't issue until the Mul produces). One FMA is a
   1-instruction step. Halving the depth of these segments lets the OOO
   engine retire more per cycle.

Steady-state shows the true compute delta (~34% faster); single-call adds
~100 cycles of rdtsc/CPUID serialization overhead that doesn't shrink with
faster code, so the single-call improvement (~14%) is more
conservative. For HFT-style latency where each call must complete before
the next decision, the single-call number is what matters.

## How FMA fusion interacts with M-project (the regalloc layer)

The codegen pipeline has **three independent FMA-emission paths**, layered:

**Layer 1: explicit IR-level FMA lifting (algsimp passes).** Three
passes work together:

- `fma_lift` (lib/algsimp.ml:1514): single-use `Add(Mul(K,X), Y)` →
  `Fma(K,X,Y)` directly in IR.
- `multi_use_fma_lift` (lib/algsimp.ml:2042): multi-use Muls whose
  consumers are all Add/Sub direct-operand patterns; rewrites every
  consumer to FMA, eliminating the Mul.
- `fma_addend_factor` (lib/algsimp.ml:2290): `Fma(K, X, Mul(K, Y))`
  refactored to `Mul(K, X±Y)` when the Ks match; the new outer Mul is
  then re-absorbable on a subsequent `multi_use_fma_lift` pass.

These create explicit `NK_Fma` nodes that emit as `_mm512_fmadd_pd(...)`.
Crucially, the original Mul is *gone* — there's nothing for M-project to
pin or barrier around. The orchestration runs them in a 4-iteration loop
(`mfl → faf → mfl → faf → mfl → faf → mfl`, at `bin/gen_radix.ml:395-476`)
because each `fma_addend_factor` can produce new Muls that the next
`multi_use_fma_lift` may absorb.

**Layer 2: gcc auto-FMA-fusion for the surviving Muls.** Muls that
algsimp can't eliminate (because their consumers include FMA-addend slots
— the `mul_in_fma_addend` disqualification that accounts for 46/50 of
the final-pass disqualifications at R=64) get preserved as `NK_Mul` nodes
in the IR. These would normally be killed by M-project's
`asm volatile ("" : "+v"(t))` barrier (lib/emit_c.ml:190) which prevents
gcc from auto-contracting `Mul + Add → vfmadd*pd` across the pin.

That's where **selective pinning** (lib/emit_c.ml:225 `compute_unpin_candidates`)
intervenes: any Mul with ≥1 Add/Sub consumer gets *unpinned* (no asm
barrier), letting gcc re-fuse it during code generation. Doc 56 measured
this carefully: at R=64 without selective pinning, M-project's barriers
cost **−126 asm FMAs** (gcc fuses 160 with no pin, 34 with full pin); with
selective pinning, the loss drops to **−43** (117 FMAs survive). The
selective-unpin set is exactly the Muls where Layer 2 can win.

**Layer 3: tag remapping for spill targets.** When Layer 1 rewrites a
frozen `Add` or `Sub` (tag T_old, a spill marker) to an `Fma` (tag T_new),
`extend_frozen` propagates `T_old → T_new` through the 4-pass remap chain
(bin/gen_radix.ml:384). The spill_info eventually consumed by M-project
regalloc (lib/regalloc.ml) points at the live FMA tag, not the dead Add tag,
so the spill machinery and regalloc stay coherent across rewrites.

**Where `round_13` fits**: this session's fix doesn't touch Layers 2 or
3. It operates earlier, at the math layer in `const_cmul`, by ensuring
symmetric-angle twiddles produce bit-identical inner ratios so hashcons
unifies their Muls upstream of any algsimp pass. Once the Muls are
unified, the existing FMA passes do their normal work — Layer 1 absorbs
more of them because there are fewer distinct Mul tags to absorb across.

## Final op-count picture

```
======================================================================
   defaults CT(4,8) / CT(8,8) / CT(8,16) / CT(16,16)
   with round_13 input rounding in dft.ml const_cmul
======================================================================
Radix | n1 (add+mul+fma)       | FFTW  | Δ    | t1 (add+mul+fma)        | FFTW | Δ
------|------------------------|-------|------|-------------------------|------|-----
R=8   |  44 +   0 +   8 = 52   |  52   |  +0  |  44 +  14 +  22 = 80    |  80  |  +0
R=16  | 104 +   0 +  40 = 144  | 144   |  +0  | 104 +  30 +  70 = 204   | 204  |  +0
R=32  | 248 +  10 + 128 = 386  | 372   | +14  | 248 +  72 + 190 = 510   | 496  | +14
R=64  | 592 +  50 + 336 = 978  | 912   | +66  | 592 + 176 + 462 = 1230  | 1164 | +66
R=128 | 1328+ 136 + 864 = 2328 |   —   |  —   | 1328+ 390 +1118 = 2836  |   —  |  —
R=256 | 3048+ 404 +2008 = 5460 |   —   |  —   | 3048+ 914 +2518 = 6480  |   —  |  —
```

All correctness sub-ulp (max errors ≤0.4 ulp relative to R²).

R=8 and R=16 hit FFTW exactly. The +14 / +66 residual at R=32 / R=64 is
the algorithmic delta from FFTW's split-radix-flavored decomposition.

## What remains

Three plausible directions to close the remaining gap, in increasing
implementation cost:

1. **Improve `lib/split_radix.ml`.** Currently produces 1144 ops at R=64
   vs CT(8, 8)'s 978. The algorithm is correct (passes correctness) but
   the IR shape it builds doesn't compose well with the existing algsimp
   passes — most of its 232 standalone muls are in patterns the FMA passes
   can't absorb. Diagnosing this is bounded work: trace which Muls survive
   and why, identify whether the SR construction emits patterns that look
   superficially different from CT's but are algebraically equivalent.

2. **Conjugate-pair split-radix (Johnson-Frigo 2007).** Frigo's
   gen_notw.native almost certainly uses this. It expresses N → (N/2 + 2·N/4)
   with shared computation across the two N/4 sub-transforms, and it's
   what produces FFTW's ratio-style constants (KP471396736 = sin(3π/16)
   reduced ratio, KP668178637 = sin(3π/8) ratio, etc.) that don't appear
   in any plain CT factorization. Implementing this is research-level
   work — Johnson-Frigo is ~30 pages of dense algebra — but it's the
   established path to FFTW-parity op counts at power-of-2 N.

3. **Algorithm-aware factoring passes in algsimp.** Recognize specific
   structural patterns produced by CT decomposition (e.g., the symmetric
   `cos(θ)·X ± sin(θ)·Y` pairs at adjacent rotation columns) and apply
   factorings that aren't visible at the generic IR level. Cheaper than
   #2 but potentially fragile — each new pattern is its own pass to
   maintain.

For now, with the `round_13` fix landed and the FMA-fusion machinery
performing as designed, **the latency wins at R=32 vs the old codelets
are substantial enough (14% single-call, 33% steady-state) that further
work should be benched-first**: each new algsimp pass or algorithm change
needs the same latency comparison treatment the CT factorization
sweep got. Op-count reduction is necessary but not sufficient evidence
for a perf change.

## Files touched this session

- `lib/dft.ml:302-326` — `round_13` input rounding in `const_cmul`'s
  Path B (tan-factored form). This is the one functional change shipped.

All other investigations (the disqualify-reason trace in
`lib/algsimp.ml:2068-2090`, the CT factorization sweeps, the bench
harnesses) were temporary and either reverted or live under `/tmp/`.

## References

- Doc 28 — original `fma_lift` gating decision
- Doc 56 — strided-batch 2D codelets, selective-pinning measurement
- Doc 57 — compiler-agnostic FMA fusion (the negative-result writeup
  whose later `fma_addend_factor` and `multi_use_fma_lift` passes set
  up this session's work)
- FFTW codelet references: `/tmp/fftw-3.3.10/dft/scalar/codelets/n1_*.c`,
  `t1_*.c`
- Johnson, S.G. and Frigo, M. *A modified split-radix FFT with fewer
  arithmetic operations*. IEEE TSP, 2007.
