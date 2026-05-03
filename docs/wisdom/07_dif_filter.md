# 07 — DIF: whole-plan-or-nothing

DIT and DIF are not interchangeable per stage. The wisdom system handles
that constraint at the orientation level — one flag per (N, K) entry,
no per-stage DIF substitution.

## The DIT/DIF asymmetry

Two ways to do a Cooley-Tukey radix-R butterfly:

- **DIT (Decimation In Time)**: twiddle multiply happens *before* the
  butterfly. Standard for forward FFTs.
- **DIF (Decimation In Frequency)**: twiddle multiply happens *after*
  the butterfly. Standard for inverse FFTs.

Despite the symmetric naming, **DIT and DIF are not interchangeable
per-stage**. Given the same input and the same twiddle table, a DIT
codelet and a DIF codelet produce different output buffers — they
compute different intermediate transforms.

Mathematically the two algorithms are duals (DIF is a rearrangement
of DIT with the twiddle multiplication moved across the butterfly),
but the *intermediate state* between stages differs. A plan that
mixes a DIT stage 0 with a DIF stage 1 produces wrong output unless
an explicit transpose-or-permutation sits between them — which the
executor doesn't have.

So orientation is **whole-plan-or-nothing**:
- DIT-forward: all forward stages are DIT (`use_dif_forward = 0`)
- DIF-forward: all forward stages are DIF (`use_dif_forward = 1`)
- The calibrator picks one orientation per (N, K) cell

## How the calibrator handles it

The variant Cartesian iterator (`vfft_variant_iter_*`) runs **once per
orientation**:

```c
for (int orient = 0; orient < 2; orient++) {
    /* refine variant cartesian within this orientation */
    walk_variants_in_orientation(orient);
}
```

Within DIT, the iterator emits {FLAT, LOG3, T1S, BUF}. Within DIF,
only {FLAT, LOG3} — T1S and BUF have no DIF analogs in the codelet
generator's output.

The calibrator picks the (orientation, variant assignment) tuple
with the lowest measured ns. The wisdom entry records both:

- `use_dif_forward` — the orientation flag
- `variant_codes[]` — interpreted in that orientation's context

A DIF wisdom entry will only have FLAT or LOG3 codes; a DIT wisdom
entry can have any of the four.

## Codelet coverage

Per-radix DIF codelet availability:

| Radix | DIF FLAT | DIF LOG3 |
|-------|----------|-----------|
| 2, 4, 8, 16, 32, 64 | ✓ | only R=16/32/64 |
| Other radixes | — | — |

DIF orientation is **pow2-only**. A DIF plan must factor into pow2
radixes; mixed-radix plans must use DIT. The codelet generator hadn't
been extended for DIF on non-pow2 radixes at v1.0 freeze.

DIF-LOG3 is restricted to R=16/32/64 because those are where the
twiddle-derivation chain wins big (smaller radixes don't have enough
twiddle traffic for log3 to amortize the chain setup cost).

## At lookup time

`_stride_build_plan_explicit` reads `use_dif_forward` from the wisdom
entry and dispatches to the matching codelet pointer:

```c
if (use_dif_forward) {
    switch (variants[s]) {
        case FLAT: t1f[s] = reg->t1_dif_fwd[R];      break;
        case LOG3: t1f[s] = reg->t1_dif_log3_fwd[R]; break;
        default:   return NULL;  /* T1S, BUF — no DIF analog */
    }
} else {
    /* DIT — full set */
    ...
}
```

If a wisdom entry has `use_dif_forward = 1` and a `variant_codes[s]`
of T1S or BUF, the build returns NULL — caller sees a registry-shape
mismatch and falls through. This shouldn't happen in practice since
the calibrator's iterator filters those combinations, but the build
function is defensive.

## Why no per-stage DIF substitution

The forward executor in `executor.h` is structurally DIT — permutation
lives at the input, butterflies are bottom-up. To execute a DIF plan,
the executor uses `executor_dif.h`, a parallel implementation with
mirrored structure. The two executors aren't interleavable mid-plan.

Building this would require a transpose-or-permutation between adjacent
DIT and DIF stages, eating any architectural win DIF would provide.
The cost-benefit doesn't justify it.

## When this could change

A future executor with cheap inter-stage transposes could enable
per-stage DIF substitution, at which point:

- The wisdom file format would need a per-stage `orientation_codes[]`
  array alongside `variant_codes[]`
- The calibrator's iterator would walk a 2× larger search space
- Layer 1 predicates (already being deleted) would gain a meaningful
  `prefer_dif_*` distinction

None of those is on a current roadmap. DIF whole-plan-or-nothing
stays for v1.x.

## See also

- [02_codelet_taxonomy.md](02_codelet_taxonomy.md) — DIT/DIF as orthogonal axis to protocol
- [04_layer2_plan_level.md](04_layer2_plan_level.md) — `use_dif_forward` field in the wisdom entry
- [`src/core/planner.h:_stride_build_plan_explicit`](../../src/core/planner.h) — the orientation switch
- [`src/core/executor_dif.h`](../../src/core/executor_dif.h) — the parallel DIF executor implementation
