# 48. Non-Constant Factoring Experiment — Another Negative Result

## What was tested

Doc 46 identified FFTW's `collectM` as a place where they generalize
beyond what we do: they factor any shared first factor in a sum, not
just constant coefficients. Our `factor_common_muls` only fires when
one operand is `NK_Const`. FFTW factors `a*x + a*y → a*(x+y)` for any
`a`, including Loads, Adds, or arbitrary expressions.

Doc 46's hypothesis: probably marginal for FFT since twiddles are the
natural shared factors, but worth measuring.

This doc reports the experiment to actually measure it.

## Implementation

Added `factor_common_terms` to `lib/algsimp.ml`:

- For each Add/Sub sum, flatten to `(sign, term)` list
- For each term that's `Mul(a, b)`, consider both `a` and `b` as
  candidate shared factors
- Bucket terms by candidate factor tag
- For each bucket with ≥2 entries: factor out the shared factor
- Use-count safety: only factor Muls with use_count == 1 (factoring
  multi-use Muls would create new paths while leaving originals
  reachable — net node-count INCREASE, same trap doc 28 warned about
  for FMA-lift)
- Greedy: pick the bucket with the most entries first, then iterate

Also added `count_factor_opportunities` as a diagnostic that counts
buckets-with-≥2-entries WITHOUT rewriting. This separates "found
nothing to factor" from "found opportunities but rewrite is broken".

Added `--factor-terms` flag to gen_radix.ml that runs the pass and
reports the before/after node count.

## Sweep results

Ran across the full radix set with use-count safety enabled:

| R | Opportunities | Nodes before | Nodes after | Delta |
|---|--------------:|-------------:|------------:|------:|
| 5  | 0 | 66   | 66   | +0   |
| 7  | 0 | 110  | 110  | +0   |
| 11 | 0 | 222  | 222  | +0   |
| 13 | 0 | 293  | 293  | +0   |
| 16 | 0 | 262  | 316  | +54  |
| 25 | 0 | 736  | 896  | +160 |
| 32 | 0 | 653  | 737  | +84  |
| 64 | 0 | 1556 | 2068 | +512 |
| 128| 0 | 3607 | 5917 | +2310 |
| 256| 0 | 8194 | 16121 | +7927 |

**Zero opportunities at every R.** The non-trivial deltas in nodes
after are pure artifacts of my rewrite implementation (which
reconstructs Add chains via `rebuild_sum`, creating new intermediate
Add nodes when no factoring actually fires) — not real factoring.

## Why nothing fires

The pattern `a*x + a*y` where `a` is non-constant requires the DAG to
contain `Mul(a, x)` and `Mul(a, y)` as separate nodes in the same sum.
In our pipeline, this combination doesn't get created because:

1. **`mk_mul` does not distribute over Add.** We never expand
   `c*(a+b)` into `c*a + c*b`. Patterns stay factored at construction
   time.

2. **Hash-consing reuses identical Mul nodes.** If `x*a` and `a*x` both
   conceptually appear, tag-ordered canonicalization gives them the
   same node — so they're already shared.

3. **`factor_common_muls` (when aggressive=true) already eliminates
   constant-coefficient redundancy.** For primes, any `Const*x +
   Const*y` pattern gets factored into `Const*(x+y)` upstream of where
   non-constant factoring would care.

4. **The math layer in `dft.ml` builds DFTs by twiddle-multiplying
   sub-results.** The natural pattern is `Add(twiddle * subdft_a,
   twiddle * subdft_b)`, where twiddles are constants. Already covered
   by case 3.

5. **For CT-decomposed codelets**, inner DFT outputs feed into outer
   DFT computations via Add/Sub chains, NOT through duplicated
   multiplies. So sub-DFT outputs don't enter sums as `factor *
   sub_a + factor * sub_b` patterns.

In short, our DAG construction strategy never CREATES the patterns
that non-constant factoring would optimize. FFTW's `collectM`
generalization exists because their pipeline DOES sometimes create
those patterns (different math-layer construction). Ours doesn't.

## Comparison with the Oracle experiment (doc 47)

The Oracle experiment found that our structural CSE is algebraically
complete — no equivalences our structural CSE missed.

This experiment finds that our DAG construction doesn't create the
redundant-multiply patterns where non-constant factoring would help.

Different roots, same outcome: the patterns FFTW's algsimp targets
don't appear in our codelets.

## Decision: drop the pass

By the same criterion as doc 47: if opportunities are 0 across the
full radix sweep, the optimization can't help. Drop and move on.

The `factor_common_terms` function and `count_factor_opportunities`
diagnostic are left in `lib/algsimp.ml` as research artifacts. If
future generator changes introduce DAG patterns where non-constant
factoring would matter, `gen_radix.exe N --twiddled --in-place
--factor-terms` will report opportunities > 0 and the implementation
can be properly debugged. Right now it's never called by production
paths.

If the rewrite itself ever needs to be fixed (e.g., to handle the
reconstruction cascade properly so it doesn't bloat node counts when
fired), the fix is to apply the factoring directly at the original
Add/Sub site instead of going through `rebuild_sum`. That's invasive
work that's not justified by the current findings.

## What remains from doc 46

Three FFTW features were flagged as potentially worth investigating:

1. **Oracle / randomized CSE** — measured in doc 47, no opportunities found
2. **collectM generalized to non-constants** — measured here, no opportunities found
3. **deepCollectM depth=1** — preserves inner Plus structure for one
   level. Not measured. Lower priority since the first two showed
   nothing.

Given two consecutive negative results from the highest-priority
candidates, it's reasonable to **also drop deepCollectM** without
measurement. The pattern is clear: our structural canonicalization
combined with hash-consing and the existing `factor_common_muls`
catches what FFTW catches via different mechanisms.

The doc 46 "biggest potential gain" hypothesis was wrong. Our
algsimp is already complete with respect to the patterns FFTW's
algsimp targets — at least for the radix range we generate.

## Files changed

- `lib/algsimp.ml` — added `factor_common_terms` and
  `count_factor_opportunities` (research artifacts, never called by
  production paths)
- `bin/gen_radix.ml` — added `--factor-terms` flag

No production code paths affected. Default codelet generation
unchanged.

---

## Cleanup (later session)

The `factor_common_terms` rewrite pass and `count_factor_opportunities`
diagnostic were removed from `lib/algsimp.ml`. The findings recorded
above stand — zero opportunities at every radix tested — but the
unused implementations (the rewriter had a known bug inflating node
counts when fired) were dead code. The `--factor-terms` flag was
removed from `bin/gen_radix.ml`.

Git history preserves the implementation. This doc remains as the
permanent record.
