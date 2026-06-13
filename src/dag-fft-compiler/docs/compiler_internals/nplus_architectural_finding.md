# N-ary Plus, collectM, deepCollectM — Final Architectural Finding

## The hypothesis worth landing on

Tugbars's question — *"maybe the regression comes from the interaction between
nplus and other features we have"* — turned out to be the right question, just
not in the way I initially interpreted it.

There WERE interactions worth fixing:
1. `lower_plus_terms` built a left-linear chain via `fold_left mk_add_binary`,
   while `mk_add` builds a balanced tree via `emit_pair_fold`. Fixed by
   making lower_plus_terms use `emit_pair_fold` too.
2. `flatten_sum` stops at `NK_Fma` leaves (the Sub-Neg-Mul peephole creates
   60 of these at R=25), hiding their internal Mul+addend structure from
   collection. Fixed by adding `flatten_sum_through_fma` that decomposes
   these and is used by `collect_m` and `deep_collect`.

But neither interaction was preventing wins. After fixing both, deep_collect
still finds zero opportunities at any radix. The deeper interaction is
architectural — not a bug to fix but a design tradeoff to acknowledge.

## The architectural finding

VectorFFT's algsimp pipeline runs `factor_common_muls` BEFORE `deep_collect`:

```ocaml
of_assignments
  → dedup_sub_pairs
  → factor_common_muls (aggressive — for Direct/prime DFTs)
  → factor_by_atom
  → dedup_sub_pairs
  → collect_m
  → deep_collect    ← my new pass
  → share_subsums
  → fma_lift
  → emit
```

`factor_common_muls` does the rewrite:

```
Σ ± c · x_i  →  c · (Σ ± x_i)
```

— it FACTORS leaf-level Muls into a single Mul-of-Sum form. This is
intentional: that form is what `fma_lift` consumes to produce FMA instructions
(`Mul + Add → FMA`).

`deep_collect`'s distribute step does:

```
c · (x + y)  →  c·x + c·y
```

— it DISTRIBUTES the same Mul-of-Sum back out to leaf-level Muls. This is
intentional in FFTW's pipeline: that form is what their algsimp pattern-matches
on for cross-output CSE and constant-product hash-cons sharing.

**These two passes do exactly opposite transformations.** They cannot both
provide value in the same DAG. One produces the form the other dismantles.

Our codelets are built with factor-then-FMA as the optimization philosophy.
FFTW's codelets are built with distribute-then-CSE as the optimization
philosophy. Both achieve high quality codegen, by different routes.

## Why we can't have both

Within a single algsimp pipeline, you have to choose:

**Path A (our current design)** — *factor toward Mul-of-Sum*:
- `factor_common_muls` consolidates Σ c·xᵢ patterns
- `fma_lift` fuses Mul(c, Add(x,y)) into FMAs and propagates upward
- Leaf-level Muls are rare; most Muls live inside FMA instructions
- Algebra count stays moderate; FMA count is high
- Op count = good (we observe ~383 for R=25)
- Gap to FFTW = the FFTW-style CSE wins we don't capture

**Path B (FFTW's design)** — *distribute toward leaf-level Muls*:
- Every Mul-of-Sum gets distributed eagerly
- Leaf-level Muls (with combined constants like `k_sin_2π5 · cos(2π/25)`)
  proliferate
- CSE catches the ones that appear in multiple outputs
- FMA fusion happens AFTER CSE catches the sharing
- Algebra count is small (lots of CSE); FMA count is high
- Op count = excellent (FFTW observes 236 for R=25)
- Gap to ours = the factor-side wins they don't capture

The two paths produce structurally different intermediate forms. Neither
algsimp pass (theirs nor ours) is incorrect — they're optimizing for
different intermediate-form invariants.

## Empirical confirmation

With deep_collect's loosest gate (`any_mul_exists`: distribute if EITHER
resulting Mul already exists in the hash-cons table), we see:

```
R=25 deep_collect decisions: 0 firings (every check finds neither
                              resulting Mul in the table)
```

That's the evidence. In our IR, when we see `Mul(c, Add(x, y))`, neither
`Mul(c, x)` nor `Mul(c, y)` already exists. Because we never distribute,
those leaf-level products were never created. Because they were never
created, hash-cons can never share them. Deep_collect has nothing to find.

To populate the hash-cons table with the products deep_collect needs, we'd
have to distribute first — but then `factor_common_muls` would have nothing
to factor, and `fma_lift` would have a different (worse-suited) starting
shape to fuse from.

## What this means for closing the FFTW gap

The 1.6×–2.4× source-op gap is **not** something deep_collect can close.
It's a function of our IR design choice. To close it, you have to either:

**Option 1: Switch IR philosophy** (massive rewrite — many weeks)
- Remove `factor_common_muls` from the pipeline
- Add eager distribution at radix-algorithm emit time
- Reshape `fma_lift` to consume distributed form
- Add FFTW-style cost-model variant selection
- Reshape `share_subsums` to expect distributed shape
This is essentially "port FFTW's algsimp." Possible but very high cost.

**Option 2: Algorithm-specific Winograd derivations** (small focused work
per radix — 3-5 days per radix that matters)
- Keep current IR philosophy
- Write `dft_winograd25.ml` (similar to `dft_winograd5.ml`) that emits the
  algebra in pre-derived form with constants like
  `k_root5_4 · cos(2πk/25)` pre-multiplied at emit time
- For each N where we care about the gap, the radix algorithm itself does
  what FFTW's algsimp does — but only for that one N, and in our IR style
This is what FFTW also does for special cases — they have hand-derived
algebraic shortcuts mixed in with the generic pipeline.

For VectorFFT as a research project, Option 2 is the right path. The
infrastructure built in commits 1-4 (NK_Plus, mk_plus, lower_plus, collect_m,
deep_collect) is reusable in Option 2: the radix-specific code can build
n-ary sums symbolically, collect-merge known shared terms, and lower to
binary at the right point.

## Status snapshot

- **17/17 unit tests pass**
- **9/9 radices**: op counts unchanged with any combination of
  `VFFT_COLLECT_M=1` and/or `VFFT_DEEP_COLLECT=1` (vs baseline)
- **Numerical correctness**: R=25 with `VFFT_DEEP_COLLECT=1` produces
  bit-identical output (max diff 0.0)
- **No regression at any radix**
- **No wins at any radix** — the architectural finding above explains why

## What landed (recap)

1. NK_Plus type + 31-site migration with `nk_plus_unreachable` guards
2. `mk_plus` / `lower_plus` smart constructors with 8 invariants + pair-fold
3. `collect_m` (shallow) with `subtree_has_collectible` pre-check
4. `deep_collect` with `lookup_node`-based distribution gate
5. `flatten_sum_through_fma` for both collect_m and deep_collect
6. `lower_plus_terms` uses `emit_pair_fold` (balanced, matches `mk_add`)
7. `extract_coefficient` for (coefficient, atom) decomposition
8. Iterative deep_collect + collect_m loop (matches FFTW's fixpoint algsimp)
9. 4 environment-variable gates (`VFFT_COLLECT_M`, `VFFT_DEEP_COLLECT`,
   `VFFT_DEEP_COLLECT_TRACE`, `VFFT_CNUM_W5`)

All of this is research-quality infrastructure usable by future algsimp work.
The empirical finding — that the collectM family cannot close the FFTW gap
on our IR — is the deliverable.
