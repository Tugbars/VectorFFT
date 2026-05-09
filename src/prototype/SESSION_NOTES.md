# Session: Distributive factoring + subsum sharing for monolithic primes

## What shipped

Two new passes in `lib/algsimp.ml`, both with explicit `~aggressive` flag,
both no-ops by default (safe for CT-decomposed codelets):

### `factor_common_muls ?aggressive`

Distributive factoring on flat sums:

    Σ ± c · x_i  →  c · (Σ ± x_i)   when c is constant

Operates on flat sums (not binary Add/Sub pairs) — pair-fold orders by
tag, so same-constant Muls aren't adjacent siblings; a binary peephole
on `Add(Mul(_,c), Mul(_,c))` never fires for primes ≥ 5.

The `aggressive=true` mode disables the use-count safety check. In
primes, source muls are shared across outputs (use_count > 1) precisely
because they aren't factored yet; both outputs migrate to the same
factored form after rewriting, so the "sharing" the safety check would
protect is illusory. Enabling factoring creates the Winograd s/d
structure (s = x_j + x_{N-j}, d = x_j - x_{N-j}).

### `share_subsums ?aggressive`

Recognize pre-existing 2-term sub-expressions inside larger flat sums
and reuse them. The motivating case is the X[0] output:

    X[0].re = x[0] + x[1] + x[2] + x[3] + x[4]   (5 terms, 4 binary adds)

After factoring fires, the DAG already contains pair sums:
    s14 = x[1] + x[4]    (built for 0.309·s14 inner sum)
    s23 = x[2] + x[3]    (built for 0.809·s23 inner sum)

X[0].re could be expressed as `x[0] + s14 + s23` (3 terms, 2 binary
adds) by greedily replacing pairs with pre-existing Add nodes that
have other users. This pass does that lookup-based rewriting.

## Op-count results

After `dft_direct + algsimp + factor_common_muls + dedup_sub_pairs +
share_subsums` (aggressive=true for primes only):

| R  | Before | After | Δ      | gen_radix*.py target |
|----|--------|-------|--------|----------------------|
| 3  | 26     | 18    | −8     | ~16                  |
| 5  | 85     | 66    | −19    | ~24-46               |
| 7  | 220    | 140   | −80    | ~50                  |
| 11 | 644    | 404   | −240   | ~110                 |

| R  | Before | After | Δ      |
|----|--------|-------|--------|
| 4  | 16     | 16    | +0 ✓   |
| 8  | 57     | 57    | +0 ✓   |
| 16 | 171    | 171   | +0 ✓   |
| 32 | 476    | 476   | +0 ✓   |
| 64 | 1210   | 1210  | +0 ✓   |

R=3 is within 2 ops of hand-coded (likely just FMA-vs-separate
accounting). R=5/7/11 still have substantial gaps — the remaining
target is DAG transposition, which Frigo specifically identifies
as saving muls on sizes 5/13/15.

## Structural separation: primes vs CT

Both new passes default to no-op (`aggressive=false`). The dispatcher
selects based on `pick_algorithm`:

    let aggressive = match Dft.pick_algorithm n with
      | Dft.Direct -> true
      | Dft.Cooley_Tukey _ -> false in

CT-decomposed codelets are already in FMA-friendly form: each twiddle
multiplication produces 4 muls with distinct constants — no factoring
opportunity. Monolithic primes are the inverse: cyclic DFT symmetry
means many same-constant multiplications across outputs that must
unify via factoring to expose the Winograd structure.

The use_count <= 1 safety check (initial design) was insufficient for
CT codelets — even fires that "look safe" can hurt because the
resulting factored term doesn't share globally. The clean answer is
the explicit flag.

## Files touched

- `lib/algsimp.ml`:
  - Added `lookup_node` (hash-cons table peek without create).
  - Added `factor_common_muls ?aggressive` (~150 lines).
  - Added `share_subsums ?aggressive` (~110 lines).
- `bin/prime_opcount.ml`: measurement harness, runs primes (R=3,5,7,11)
  through aggressive path and CT (R=4,8,16,32,64) through safe path.
- `bin/dune`: registered prime_opcount.

## Next session: DAG transposition

Frigo's third simplifier pass — reverse all edges, simplify, reverse
back. Per his Table 7, saves muls on sizes 5, 10, 13, 15 specifically.
Estimated 100-200 lines.

The intuition: a linear network computing `y = Mx` can be transposed
to compute `x = M^T y`, which simplifies differently. The catch is
that our DAG isn't a pure linear network — it has Mul(_, _) nodes,
not just edge weights. Need to convert to/from linear-network form
or implement a more direct transposition on our DAG representation.
