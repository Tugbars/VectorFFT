# Hand-derived Winograd-25 — Implementation and Findings

## What landed

`dft_winograd25` in `lib/dft.ml`, gated under `VFFT_WINOGRAD25=1`. CT(5,5)
decomposition with PLUS-OF-TIMES twiddles (instead of the tan-factored
form `const_cmul` uses in `dft_ct`).

Numerical correctness verified: **2.18 × 10⁻¹⁴ max diff** vs baseline on
random complex inputs (well within float-precision evaluation-order
tolerance at double).

Op count: 384 instructions vs baseline's 383 — essentially identical.
Both are 32 instructions above FFTW's 352 for the same algorithm.

## Why the gap didn't close

The 31-op gap to FFTW is structural at the multi-use FMA-absorption
level, not at the cmul-form level. Three empirical findings, in order
of discovery:

### Finding 1 — FFTW's R=25 is 352 ops, not 236

Initial premise was a 147-op gap (~38%). Reading the actual FFTW
codelet header (`/tmp/fftw-3.3.10/dft/scalar/codelets/n1_25.c`):

```
* This function contains 352 FP additions, 268 FP multiplications,
* (or, 84 additions, 0 multiplications, 268 fused multiply/add),
```

So FFTW R=25 = **84 add + 0 mul + 268 fma = 352**. Our R=25 =
**130 add + 31 mul + 222 fma = 383**. Real gap: **31 ops (~8%)**,
driven entirely by 31 unfused muls.

### Finding 2 — The lincomb approach hits 0 muls but loses sharing

First attempt at `dft_winograd25` used a symbolic linear-combination
representation: each pass-2 W5 intermediate (t8, ta, ti, tk, etc.)
stored as a 10-element float array of coefficients over Y[j].re and
Y[j].im. W5 constants × twiddle constants pre-multiplied at OCaml time.
Final emission converted each lincomb to a flat Add tree of
`Mul(Const c, Y[j].re/im)` leaves — exactly what fma_lift wants.

Result: **0 unfused muls** (goal achieved), but **472 FMAs + 78 adds =
550 total ops**, vs baseline's 383. The fully-distributed form lost
the intra-pass-2-W5 sharing that the standard W5 algebra provides
(t8 shared across t9 and X[0], tb shared across X[1] and X[4], etc.).
Each output expanded its lincomb independently, materializing the same
algebra many times across outputs.

This is a *valid* design point — minimum-mul codelet — but at a 43%
total-op cost. Discarded in favor of the plus-of-times CT(5,5) form.

### Finding 3 — Plus-of-times CT(5,5) gives the same op count as tan-factored

The shipped `dft_winograd25` uses CT(5,5) with PLUS-OF-TIMES twiddles:

```
z.re = Sub(Mul(cr, Y.re), Mul(ci, Y.im))     // 2 leaf muls + 1 sub
z.im = Add(Mul(cr, Y.im), Mul(ci, Y.re))     // 2 leaf muls + 1 add
```

vs baseline's tan-factored:

```
z.re = Mul(cr, Sub(Y.re, Mul(tan, Y.im)))    // 1 outer mul + 1 inner mul
z.im = Mul(cr, Add(Y.im, Mul(tan, Y.re)))    // (rotation-fold on outer)
```

Both forms have 4 abstract muls per cmul. After algsimp + fma_lift, both
reach the same op count (382/383/384 — equivalent to within the
algebraic noise floor). The IR optimizer canonicalizes both to the
same fixpoint.

## What's actually causing the 31 muls

Two distinct sources, both visible in the emitted C output:

**Source A: spill-marker pinning (~20 of 31 muls at R=25).** When a Mul's
result has both an absorbable use (as Add/Sub operand) AND a non-
absorbable use (spill store via `regalloc_spill`), `multi_use_fma_lift`
correctly refuses to absorb — duplicating into the FMA would leave the
spill marker pointing at a non-existent tag. R=25 has 485 spill stores
(high register pressure), driving this category.

Evidence: R=32 has 0 spills but still 10 muls; R=64 has 0 spills but
50 muls. Spills aren't the only cause.

**Source B: multi-use Muls where one use is in an Fma's multiplicand
slot or root assignment (~10-15 muls at R=25).** The Mul gets duplicated
once into one consumer, but if a second consumer is a position fma_lift
can't fuse into (e.g., the `a` or `b` slot of another Fma), the Mul
stays as a Mul.

FFTW's generator doesn't have either constraint because it doesn't have
our spill-aware register allocator OR our multi-use-aware fma_lift. It
emits FMAs unconditionally at algorithm-emission time, and lets the C
compiler handle register pressure. Our pipeline is more conservative
by design (the SU+spill recipe is what gives us competitive wall-clock
performance across many radices).

## Closing the gap — what it would take

To match FFTW's 352 ops, we'd need either:

1. **Spill-aware fma_lift**: instead of refusing to absorb when a spill
   marker is present, update the marker to point at the FMA result (the
   value at the spill point is algebraically the same — the marker is
   just naming the spill slot). This is a ~50-line change in
   `multi_use_fma_lift` + the spill-marker update logic. Likely closes
   ~20 of the 31 muls.

2. **Aggressive Mul duplication at codelet-emit time**: instead of
   relying on fma_lift after the fact, have `dft_winograd25` (and other
   hand-derived codelets) emit FMA-friendly forms directly, never
   creating multi-use Mul intermediates. This is what FFTW does. For
   R=25 this would mean computing `Mul(k_sin_2pi5, ti)` separately for
   the X[1] and X[4] consumers rather than sharing — paying 1 extra Mul
   at construction time, but 0 muls post fma_lift.

Both paths are research projects in their own right. For now,
`dft_winograd25` is a clean reference implementation that demonstrates
the algorithm at our IR's current optimization ceiling.

## Status

- `dft_winograd25` in `lib/dft.ml` (plus-of-times CT(5,5))
- Dispatched at top of `dft` when `VFFT_WINOGRAD25=1`
- All other radices unchanged (R=5,7,11,13,16,20,32,64 identical with
  flag set vs unset)
- 17/17 unit tests pass
- Numerical correctness: 2.18×10⁻¹⁴ max diff vs baseline
- Op count: 384 (baseline 383, FFTW 352)

The lincomb-based first attempt is preserved in git history for
reference but not shipped; the plus-of-times form is what's in the
final file.
