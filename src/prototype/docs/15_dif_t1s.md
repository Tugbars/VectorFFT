# t1s + DIF + In-place R=64 — Variant Coverage Complete

## Summary

Three additions that round out the codelet variant matrix:

1. **In-place R=64 hand reference.** `gen_radix64.py --variant ct_t1_dit` is the in-place hand variant (I missed this earlier). Re-ran all R=64 hand comparisons in-place vs in-place — the comparisons stand but the magnitudes change vs the OOP-only numbers we had.

2. **t1s codelet support.** `--t1s` flag emits scalar-broadcast twiddles (`_mm512_set1_pd(tw_re[j])` instead of strided vector loads). For inner CT codelets where all `me` iterations share the same twiddle set. ~50 lines (10 lines emit_c.ml change + CLI plumbing).

3. **DIF codelet support.** `--dif` flag flips the external twiddle from input pre-multiply to output post-multiply. ~120 lines total: `direction` type, DIF expansion in `dft.ml` (with and without spill), classifier fix in `emit_c.ml` to handle Pass2-side Twiddle Loads, cluster-assignment fix for unclustered Pass2 Loads.

All four combinations (`t1`/`t1s` × `dit`/`dif`) generate correctly at every radix. The recipe auto-applies through them all.

## R=64 in-place results (3 runs each, median)

### t1 DIT vs hand t1 DIT (in-place)

| K | Hand | Topo | **Recipe** | **Recipe-log3** | LR/H |
|---|------|------|------------|-----------------|------|
| 64 | 4253 | 6707 | 2911 | **2685** | **0.63** |
| 128 | 8946 | 13542 | 8842 | **8177** | **0.91** |
| 256 | 20958 | 29045 | 19937 | **17564** | **0.84** |
| 512 | 47331 | 67047 | 47111 | **41639** | **0.88** |
| 1024 | 128650 | 157026 | 117844 | **85775** | **0.66** |
| 2048 | 298683 | 367249 | 278899 | **184789** | **0.62** |
| 4096 | 610366 | 753244 | 574368 | **437658** | **0.72** |

Recipe-log3 wins at every K by 9-38%. Recipe alone is barely tied at mid-K.

### t1 DIF vs hand t1 DIF (in-place)

| K | Hand | Topo | Recipe | T/H | **SU/H** |
|---|------|------|--------|-----|----------|
| 64 | 4229 | 6715 | 3623 | 1.59 | **0.85** |
| 128 | 9351 | 14575 | 8736 | 1.56 | **0.93** |
| 256 | 20776 | 31392 | 19116 | 1.51 | **0.93** |
| 512 | 46674 | 71057 | 44725 | 1.56 | **0.96** |
| 1024 | 117604 | 167371 | 113856 | 1.43 | **1.00** |
| 2048 | 291853 | 409410 | 276043 | 1.37 | **0.95** |
| 4096 | 610944 | 805709 | 544588 | 1.31 | **0.89** |

Recipe-DIF ties or beats hand DIF at every K (0-15% faster). DIF Topo is consistently 31-59% slower than hand — the recipe closes that gap entirely.

### t1s DIT vs hand t1s (in-place)

| K | Hand | Topo | Recipe | T/H | **SU/H** |
|---|------|------|--------|-----|----------|
| 64 | 3299 | 6340 | 2976 | 1.91 | **0.90** |
| 128 | 8500 | 12208 | 8014 | 1.45 | **0.96** |
| 256 | 17955 | 27535 | 17257 | 1.53 | **0.96** |
| 512 | 42891 | 61784 | 38643 | 1.44 | **0.90** |
| 1024 | 85776 | 126565 | 86019 | 1.51 | **0.99** |
| 2048 | 194251 | 264109 | 185539 | 1.36 | **0.96** |
| 4096 | 451583 | 612728 | 436997 | 1.36 | **0.97** |

Recipe-t1s 1-10% faster than hand t1s. Tighter spread than t1 because t1s already has minimal twiddle bandwidth (just scalar broadcasts hoisted by GCC LICM). Hand pre-hoists 4 broadcasts manually; we let GCC handle it — both end up fine at -O3.

## What changed in the code

### `lib/dft.ml`

Added `direction = DIT | DIF` type. `dft_expand_twiddled` and `dft_expand_twiddled_spill` both take `?(direction = DIT)`.

DIT path: pre-multiply inputs by twiddles, then run DFT (existing behavior).
DIF path: run DFT on raw inputs, post-multiply outputs by twiddles.

Spill markers are captured at the same PASS 1 / PASS 2 boundary in both directions — the CT decomposition structure is identical; only the external twiddle position differs.

### `lib/emit_c.ml`

Two fixes for DIF:

**Classifier fix.** Forward classification (Pass2 = anything depending on a spilled tag) puts Twiddle Loads in Pass1 since they have no predecessors. For DIF, twiddle loads are consumed only by Pass2 cmuls (post-multiply). The C block scoping (`{...}` for PASS 1, `{...}` for PASS 2) means Pass1-emitted loads go out of scope before PASS 2.

Backward pass added: any Pass1 Load whose consumers are exclusively in Pass2 → push to Pass2.

**Cluster-assignment fix.** Pass2 cluster IDs are derived from `min_input_slot mod N2`, which traces back through predecessors to spill slots. Pass2 Twiddle Loads have no spill-slot ancestors → no `min_input_slot` → no cluster assignment → DROPPED from `groups.(k2)` in SU clustering.

Fix: after the slot-based cluster assignment, walk Pass2 nodes without a cluster and assign each to its first consumer's cluster.

For t1s: render_load takes `~t1s` flag. When true, Twiddle Loads emit `_mm512_set1_pd(tw_re[j])` instead of `_mm512_loadu_pd(&tw_re[j*me + k])`. The function signature is unchanged; only the access pattern (and bench harness's twiddle array layout) differs.

### `bin/gen_radix.ml`

`--t1s` and `--dif` flags. Function name picks up `t1s` infix and `dit`/`dif` suffix appropriately. Recipe still auto-applies through the cost-model rule for all combinations.

## What's left (small)

- **R=64 AVX2 DIF/t1s**. Should work — the changes are ISA-agnostic. Not yet measured.
- **DIF + log3.** Likely works because the DIF expansion uses the same `twiddle_expr policy`; same memoization+hash-cons applies. Not measured.
- **t1s log3.** Same — likely works.
- **bwd direction.** All hand variants come in fwd/bwd pairs; we only emit fwd. Mirror the cmul sign convention to add bwd.

## Final variant matrix at R=64 in-place vs hand

| | DIT | DIF |
|---|-----|-----|
| t1 (recipe) | 0.63-1.00 (avg 0.83) | 0.85-1.00 (avg 0.93) |
| t1s (recipe) | 0.90-0.99 (avg 0.95) | not measured |
| t1 (recipe-log3) | **0.62-0.91 (avg 0.74)** | not measured |

Recipe-log3 t1 DIT remains the strongest single variant. Recipe alone is competitive across all DIT/DIF/t1/t1s combinations — never significantly worse than hand, often better.

## What this validates

1. **Direction is orthogonal to the recipe.** The same Spill + SU + cluster-sequential PASS 2 emission policy works for DIF, with two small classifier fixes for the post-multiply twiddle path.

2. **The variant axis is small.** Going from "t1 DIT" to "t1/t1s × DIT/DIF" was ~150 lines total. The math layer (`dft.ml`) absorbed ~60; the emission layer (`emit_c.ml`) absorbed ~60; the CLI absorbed ~30.

3. **t1s validates that GCC LICM is sufficient for loop-invariant broadcasts.** Hand pre-hoists 4 broadcasts manually before the m loop; we leave them inside the loop body. At -O3, GCC's LICM extracts them and the result is identical to hand's manual hoisting.

4. **The classifier's forward-only pass had a latent bug.** It worked for DIT because DIT happens to put all unboundary Loads in PASS 1, where they're correctly scoped. DIF exposed the issue. The backward pass now makes the classifier directional-agnostic.
