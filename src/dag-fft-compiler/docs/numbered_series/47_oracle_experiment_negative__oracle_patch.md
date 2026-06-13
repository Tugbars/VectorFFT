# 47. Oracle CSE Experiment — Negative Result

## What was tested

Doc 46 identified Oracle-based randomized CSE as the highest-potential
FFTW feature we lack. The Oracle catches algebraic equivalences that
pure structural CSE misses — expressions that evaluate to the same
value despite being syntactically different.

Classic example: `(x+y)*(x-y)` and `x*x - y*y` are mathematically equal.
Structural CSE (which is what we use, via hash-consing) sees them as
different DAG nodes and computes both. The Oracle assigns random doubles
to all leaves, evaluates both subtrees, and detects the equality via
matching numerical hashes.

If our structural CSE were leaving algebraic sharing on the table,
adding Oracle would shrink DAGs by some percentage. Doc 46 estimated
5-15% reduction would translate to ~3-8% fewer emitted instructions.

This doc reports the experiment to find out.

## Implementation

Minimal Oracle in `lib/oracle.ml` (~150 lines):

- Each unique `Load(elem_ref)` gets a deterministic random double from
  `sin(seed * golden_ratio) * 1000` where seed is derived from
  `(slot, is_re, kind)`.
- Each DAG node is evaluated recursively bottom-up via standard FP
  arithmetic; results memoized by node tag.
- Bucket key uses `frexp` to extract mantissa+exponent, rounds the
  mantissa to 40 bits (~12 decimal digits), and combines with exponent
  and sign. This avoids the int64 overflow problem of naive `v * 1e12`
  scaling that would saturate all large values into one bucket.
- A "collision bucket" with ≥ 2 distinct DAG tags = an algebraic
  equivalence our structural CSE missed.

Added `--oracle-diag` flag to `bin/gen_radix.ml` that runs the diagnostic
on the final post-algsimp DAG and reports the count.

## Sanity check

Built synthetic test:
```ocaml
let e1 = mk_mul (mk_add x y) (mk_sub x y)   (* (x+y)*(x-y) *)
let e2 = mk_sub (mk_mul x x) (mk_mul y y)   (* x*x - y*y *)
```

Both Oracle-evaluate to -6.114e5 with the same bucket key. Diagnostic
correctly reports 1 missed CSE chance. Confirms Oracle implementation
can detect the kind of equivalences it's designed to find.

## Experiment results

Ran Oracle diagnostic on every default codelet across the production
radix set:

| R   | Nodes | Buckets | Missed CSE | Largest collision |
|-----|------:|--------:|-----------:|------------------:|
| 5   |    66 |      44 |          0 |                 0 |
| 7   |   110 |      78 |          0 |                 0 |
| 11  |   222 |     170 |          0 |                 0 |
| 13  |   293 |     228 |          0 |                 0 |
| 16  |   262 |     197 |          0 |                 0 |
| 25  |   736 |     618 |          0 |                 0 |
| 32  |   653 |     520 |          0 |                 0 |
| 64  |  1556 |    1286 |          0 |                 0 |
| 128 |  3607 |    3064 |          0 |                 0 |
| 256 |  8194 |    7107 |          0 |                 0 |

Also tested variants on R=32: `--log3`, `--bwd`, `--dif`, `--log3 --t1s`.
All report zero missed CSE.

**Zero missed CSE at every tested size, every variant.**

## What this means

Our structural CSE is **algebraically complete** for the DAGs our
generator produces. The Oracle-based randomized CSE in FFTW would not
shrink our DAGs further. The 5-15% reduction doc 46 hypothesized is
not present.

In retrospect this makes sense:

1. **Hash-consing of constants and Loads** is at the leaf level. Every
   distinct leaf is created exactly once.
2. **`flatten_sum` + `cancel_signs`** normalize sums to a canonical
   `(sign, term)` list before reconstruction. Two sums that differ only
   in evaluation order or signed-zero cancellation collapse to the same
   DAG.
3. **Tag-ordered Mul canonicalization** gives commutative CSE for free —
   `a*b` and `b*a` always hash to the same node.
4. **`factor_common_muls`** factors out shared constant coefficients,
   producing canonical form before CSE.
5. **`share_subsums`** finds shared 2-term subsums within flattened
   chains (when aggressive=true for primes; not for composites).
6. **`transpose` fixed-point loop** for composites: factor →
   share_subsums → repeat up to 6 times.
7. **`dedup_sub_pairs`** at the assignment level catches `Sub(a, b)`
   vs `Sub(b, a)` duplicates.

These passes collectively normalize the DAG so that any two
algebraically-equivalent expressions land at the same node. The Oracle's
"smarter" hashing has nothing to detect because the work has already
been done structurally.

The interesting subtlety: FFTW uses Oracle to compensate for not having
some of these passes. FFTW doesn't do as aggressive structural
canonicalization — `reduce_sumM` and `collectM` are less normalizing
than our `flatten_sum + cancel_signs + emit_pair_fold`. The Oracle is
their backstop. We use direct structural normalization instead and
land at the same place.

Different mechanism, same result. **Both approaches reach the
algebraically-complete DAG.**

## Implication for the R=25 gap

The AVX-512 regression bench (doc 45) showed R=25 ~7-14% slower than
hand on AVX-512 — the only consistent performance gap. This experiment
rules out "missed CSE in the algsimp" as the cause. R=25 has 736 nodes
post-algsimp; our Oracle confirms every one of them is structurally
unique up to algebraic equivalence.

The gap must therefore be in:
- **Scheduler** (SU+spill behavior on 5×5 CT structure)
- **Spill recipe** (the cluster placement might be wrong for 5×5)
- **Instruction selection** at emit time (FMA-lift gating, micro-ops)
- **Compiler** (gcc-11 register allocator behavior on the specific
  shape of the R=25 DAG)

None of these would be improved by Oracle. If we ever care about
closing the R=25 gap, the next investigations should be:

1. Run llvm-mca on the R=25 codelet body, compare against the hand
   version. Identify which µop sequences differ.
2. Try `--bb` (branch-and-bound scheduler) on R=25 specifically — for
   5×5 CT the search space might be small enough that BB finds a
   schedule beating SU.
3. Try emitting without spill markers (smaller R=25 might not need
   them) — at 736 nodes it's borderline.

But none of this is Oracle-related.

## Decision: drop Oracle

Per the criterion in doc 46:

> "Diagnostic test: generate R=64 with and without Oracle, compare DAG
> node count. If <5% reduction, not worth doing. If 10%+, valuable."

We measured 0% across all sizes. **Not worth implementing the full
rational-arithmetic version of Oracle. Drop and move on.**

The minimal Oracle in `lib/oracle.ml` is left in place as a diagnostic
tool. Anyone questioning whether structural CSE is complete can run
`gen_radix.exe N --twiddled --in-place --oracle-diag` on any new
codelet to verify. If a future generator change introduces algebraic
equivalences our structural CSE misses, the diagnostic will catch it.

The full FFTW algsimp comparison in doc 46 is also useful as a
reference — it documents what we have parity on vs. what's genuinely
different. The other items (collectM factoring non-constants,
deepCollectM depth-1, generateFusedMultAddM) are lower-priority and
probably also wouldn't move the needle, but haven't been measured.

## Files changed

- `lib/oracle.ml` (new, ~150 lines) — Oracle diagnostic module
- `lib/dune` — register oracle module
- `bin/gen_radix.ml` — add `--oracle-diag` flag and hookup

No production code paths affected. The Oracle only runs when
`--oracle-diag` is explicitly requested. Default codelet generation is
unchanged.
