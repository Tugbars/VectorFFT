# N-ary Plus + collectM — Commits 1-3 Findings

## Summary

Three commits landed, all behind tests, no production-codepath regressions.
The work establishes the infrastructure for FFTW-style sum collection in
VectorFFT's algsimp pipeline. Empirical finding: **shallow collectM is
correct but does not find opportunities in current codelets** — the savings
FFTW achieves require deepCollectM (recursive distribution through nested
Plus subtrees), which is the next step.

## Commits

### Commit 1: NK_Plus type + match-site migration

Added `NK_Plus of (int * t) list` to the algsimp node_kind ADT, where each
`int` is `+1` or `-1` (signed term). Patched 31 match sites across
`algsimp.ml`, `schedule.ml`, and `emit_c.ml` to handle NK_Plus — each gets
an explicit `| NK_Plus _ -> Algsimp.nk_plus_unreachable "<site>"` branch.

The `nk_plus_unreachable` helper raises with a clear message so any
unmigrated production path that accidentally generates NK_Plus fails loudly
with the call-site identified.

Zero behavior change in this commit — NK_Plus is defined but no production
code generates it.

### Commit 2: mk_plus + lower_plus smart constructors

Added `mk_plus : (int * t) list -> t` enforcing 8 invariants:

1. Empty list → `Const 0.0`
2. Single-term list collapses (with sign applied via `mk_neg` for -1)
3. Nested `NK_Plus` flattens
4. `NK_Neg` absorbed into the sign (`(+1, Neg x) → (-1, x)`)
5. Multiple `NK_Const` terms summed
6. Terms sorted by tag for canonical hash-cons keys
7. Zero terms dropped
8. Opposite-sign tag-identical terms cancel

Added `lower_plus : t -> t` for round-trip back to balanced binary
`NK_Add`/`NK_Sub`, used by consumers that don't (yet) handle NK_Plus.

`bin/test_mk_plus.exe` provides 13 invariant tests, all passing. Production
codegen unchanged (mk_plus is unused so far).

### Commit 3: collect_m pass

Implemented the shallow variant of FFTW's `collectM`. For each Add/Sub
subtree in the DAG:

1. Flatten via `flatten_sum` into `(sign, leaf)` pairs
2. For each leaf, extract `(coefficient, atom)` where atom is the
   non-constant factor
3. Group by atom tag; sum coefficients
4. Emit one `Mul(Const(sum), atom)` per group; preserve constants
5. Lower the resulting `NK_Plus` back to binary

**Critical guard**: `subtree_has_collectible` pre-checks whether the
subtree contains any tag-shared atom (or multiple constants). If not, the
original binary tree shape is preserved. This guard exists because the
original tree was built by `mk_add`'s pair-fold (balanced for FMA fusion);
re-flattening and re-emitting would linearize it and **dramatically hurt**
fma_lift downstream (measured: R=64 went 978 → 3162 ops without the guard,
a 3.2× regression from re-linearization alone).

Gated behind `VFFT_COLLECT_M=1`. 4 synthetic unit tests in
`test_mk_plus.exe` confirm:
- `2x + 3x → 5x` ✓
- `2x + 3x - x → 4x` ✓
- `2x - 2x → 0` ✓
- `x + y → x + y` (preserved when no sharing) ✓

Production codegen unchanged at all 9 tested radices (5, 7, 11, 13, 16,
20, 25, 32, 64) with `VFFT_COLLECT_M=1` because the guard correctly
identifies that no atoms share within local Add/Sub subtrees.

Numerical equivalence verified: R=25 AVX-512 with collect_m produces
**zero difference** from baseline (bit-identical output, max diff 0.0).

## Empirical finding

Simple shallow collectM doesn't fire in our codelets. The reason is
structural: in pass-2 Winograd-5 of R=25, terms are like
`Mul(cr_1, Sub(xr_1, Mul(ci_1, xi_1))) + Mul(cr_2, Sub(xr_2, ...))` where
the constants (cr_i) are different per term and the atoms inside the Subs
are also distinct. No collectible sharing.

FFTW's 1.6×–2.4× algebra advantage doesn't come from shallow collect either.
It comes from **deepCollectM** — recursive distribution of Muls through
nested Plus subtrees. Specifically:
- `Mul(c, Plus[Mul(c1,x), Mul(c2,y)])` distributes to
- `Plus[Mul(c*c1, x), Mul(c*c2, y)]`
This exposes new (x, y) atoms that might share with outer Plus terms,
enabling collection that wasn't visible before.

We confirmed this experimentally: shallow collect found exactly 0
collectible opportunities at R=25. The savings must therefore come from
the distribute-then-collect coupling, not shallow collect alone.

## What's next: deepCollectM (Commit 4)

The next step is implementing FFTW's `deepCollectM` — distribution at
configurable depth into nested Plus structures.

Algorithm:
1. For each `Mul(c, X)` where `c` is `Const` and `X` is a Plus:
   distribute one level: `Mul(c, Plus[t1, t2, ...]) → Plus[Mul(c, t1), Mul(c, t2), ...]`
2. Each inner `Mul(c, Mul(c', x))` rotates via the existing `mk_mul`
   rotation rule (already implemented in Expr smart constructors) to
   `Mul(c*c', x)`.
3. Apply shallow collect on the resulting flat Plus.
4. Recurse to depth N (FFTW uses 5).
5. Gate on guards: distribution adds ops upfront; only emit if the
   resulting collect step finds enough sharing to net out.

The risk is that distribution is expensive and the guards have to be
right — same lesson as the R=64 linearization regression in Commit 3,
just at the next layer.

Estimated cost: 3-5 days for deep distribute + tests + regression sweep.

## Files

- `lib/expr.ml` — smart constructors (rotation + distribution rules) [Commit 0]
- `lib/cnum.ml` — Cnum combinator type [Commit 0]
- `lib/algsimp.ml` — NK_Plus type + mk_plus + lower_plus + collect_m [Commits 1-3]
- `lib/schedule.ml`, `lib/emit_c.ml` — match-site migration [Commit 1]
- `bin/test_mk_plus.ml` — 17 unit tests [Commits 2-3]
- `bin/gen_radix.ml` — pipeline wiring [Commit 3]

All tests pass: 17/17.
All radix codegen unchanged: 9/9 op counts match baseline with or without
`VFFT_COLLECT_M=1`.
Numerical correctness: R=25 baseline vs collect_m max diff = 0.0.
