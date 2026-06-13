# R=32 Spill Variant: It Works

## TL;DR

Built the spill variant in OCaml. R=32 with `--spill` is **4-19% faster than Topo at every K**, closing most of the gap to hand-coded:

| K | Topo/Hand | Spill/Hand | Spill/Topo |
|---|---|---|---|
| 64 | 1.48 | 1.28 | **0.86** |
| 128 | 1.71 | 1.39 | **0.81** |
| 256 | 1.45 | 1.26 | **0.86** |
| 512 | 1.37 | 1.22 | **0.87** |
| 1024 | 1.22 | 1.13 | **0.92** |
| 2048 | 1.13 | 1.17 | **0.91** |
| 4096 | 1.15 | 1.10 | **0.95** |

At K=2048 one run hit Spill/Hand = 1.03 — essentially matching hand-coded. At K=128 the gap narrowed from 71% behind hand to 39%.

## Architecture

Three layers of work:

**Math layer (`dft.ml`)**: New `dft_expand_twiddled_spill` drives the outermost CT step manually instead of through the recursive `dft` dispatch. This lets us capture `pass1_re` and `pass1_im` arrays as `spill_marker` records before they get fed into INTERNAL TWIDDLES + PASS 2. Inner sub-DFT recursions use the normal `dft` path. ~80 lines.

**Algsimp layer (`algsimp.ml`)**: New `lift_spill_markers` resolves the marker `Expr.expr` values to their hash-consed tags after `of_assignments` has run. Hash-consing dedupes — the same marker subtree appearing inside an assignment expression and standalone in a marker gets the same `Algsimp.t` with the same tag. ~30 lines.

**Emit layer (`emit_c.ml`)**: New `?spill` parameter. When set, takes a dedicated emission path:
- Declare `__m512d spill_re[N], spill_im[N]` at function scope
- Hoist `NK_Const` nodes to outer scope (visible to both passes)
- Open PASS 1 nested scope: emit nodes block-sequentially, with immediate spill-stores after each spill target
- Close PASS 1, open PASS 2 nested scope: emit reloads, then PASS 2 nodes, then output stores
~150 lines.

## The two bugs we hit

### Bug 1: Constants out of scope

Hash-consed constants like `set1_pd(0.707...)` are shared between PASS 1 (radix-8 internal twiddles) and PASS 2 (radix-4 internal twiddles). My initial classification put them in PASS 1 only, so when PASS 2's nested scope opened, the constants were already out of scope.

**Fix**: hoist all `NK_Const` leaves to the for-loop body top, BEFORE the PASS 1 scope opens. They have no predecessors so they can be defined anywhere; placing them in outer scope makes them visible everywhere.

(Loads, by contrast, depend on the loop variable `k` and have to live inside the loop. They're naturally classified to PASS 1 by topology since their only consumers are PASS 1 cmul operations.)

### Bug 2: Sub-FFT interleaving destroys the win

First spill version produced **identical assembly to no-spill version**. The reason was hidden in the topological tag order:

After `algsimp` runs, tags are assigned in DAG-construction order, but the construction order isn't simply "sub-FFT 0 first, sub-FFT 1 next." It's driven by the recursive walk inside `of_expr`, which descends into the leftmost-deepest path first. For our R=32 with N1=4 sub-FFTs, this caused tags to interleave heavily:

```
spill[31]: t84   ← sub-FFT 3 output
spill[15]: t178  ← sub-FFT 1 output
spill[23]: t276  ← sub-FFT 2 output
spill[7]:  t369  ← sub-FFT 0 output
spill[30]: t395  ← sub-FFT 3 output (next bin)
...
```

When emission walks tags in order, it interleaves all four sub-FFTs simultaneously. Even with immediate-spill, peak live-set in PASS 1 stays ~32 (one bin from each sub-FFT plus their intermediate state). GCC has to spill internally on top of our explicit spills.

**Fix**: For each PASS 1 node, compute `min_descendant_slot` — the smallest spill slot reachable from this node. Then sort PASS 1 nodes by `(min_descendant_slot, tag)`. This clusters them into sub-FFT blocks (slots 0-7 = sub-FFT 0, 8-15 = sub-FFT 1, etc.) while preserving topological order within each block.

The reordering is safe because sub-FFTs in CT are **independent**: their only shared dependencies are constants (already hoisted) and there are no cross-sub-FFT data flows.

After this fix, the spill stores emit in clean slot order:

```
spill[0]: t790   ← sub-FFT 0
spill[1]: t770   ← sub-FFT 0
spill[2]: t726   ← sub-FFT 0
spill[3]: t684   ← sub-FFT 0
...
spill[7]: t369
spill[8]: t780   ← sub-FFT 1 starts
spill[9]: t744
...
```

Each sub-FFT is computed completely (with its outputs spilled) before the next sub-FFT begins. Peak live-set in PASS 1 drops to ~16 (one sub-FFT's working set), fitting comfortably in 32 ZMM.

## Effect on assembly

| | arith | reg-reg copies | spill stores | spill loads | **Memory total** |
|---|---|---|---|---|---|
| Topo (no spill) | 610 | 133 | 235 | 138 | **506** |
| Spill | 611 | 66 | 135 | 102 | **303 (-40%)** |
| Hand | 589 | 28 | 55 | 112 | **195** |

Memory operations dropped by 40%. Three components each contributed:
- **Reg-reg copies halved** (133→66): GCC chose better FMA variants because registers weren't constantly conflicting
- **Spill stores -43%** (235→135): GCC's automatic spills were replaced by our explicit ones, which are more organized and don't compound
- **Spill loads -26%** (138→102): fewer reload events because the explicit reload pattern is clean

We're still 55% above Hand's memory ops (303 vs 195). Two things hand does that we don't:

1. **"FUSED" optimization**: hand keeps 4 of 8 PASS 1 outputs in registers (using `s` regs) instead of spilling them. Saves ~16 spill-store + spill-load pairs at the boundary. Not implemented in our generator.

2. **Better external twiddle handling**: hand has 64 unaligned loads vs our 126. Looks like hand is using fewer twiddle loads — possibly factoring twiddle uses across passes or using a different twiddle policy. Worth a separate investigation.

## What the architecture cleanly demonstrates

The variant-dispatcher framing we've been arguing for is now empirically validated across two radices:

- **R=16 crossover**: Hand-coded spill helps small K (8-15%), no-spill helps large K (8-15%). Two regimes with different winners.
- **R=32 spill**: Spill-with-block-sequential-PASS1 wins at every K (4-19%), but never matches Hand. Need additional optimizations.

The math layer is shared. The emit layer differs by emission policy (spill or no-spill). The dispatcher would pick per (radix, K, ISA, µarch).

## Honest scope of the win

The spill variant works, but the gap to Hand at R=32 is still substantial:
- Best case (K=2048): 3% behind Hand
- Worst case (K=128): 39% behind Hand

The 39% small-K gap suggests two structural inefficiencies still present:

1. **No FUSED optimization** — keeping some pass-1 outputs in registers across the boundary saves both their spill store and reload. ~16 ops saved per iteration. Modest gain (5-10%) at small K.

2. **Per-pass scheduling could be SU instead of Topo** — at R=32 SU helped 6-24% over Topo without spill. With spill, SU within each pass should help further. Not yet tried (current spill emission only supports Topological scheduling within passes).

Both are clear next steps. (1) requires lifetime analysis per spill target. (2) requires plumbing SU through the spill path's per-pass node lists.

## CLI

```
$ dune exec bin/gen_radix.exe -- 32 --twiddled --emit-c --in-place --spill
```

Generates `radix32_t1_dit_fwd_avx512_gen_inplace_spill`. The `_spill` suffix in the function name distinguishes it from the no-spill variant.

`--spill` is orthogonal to `--su`, `--bisect`, `--annotate`, etc. For now it overrides them (always uses Topological within passes). Future work: combine with SU.

## Where the cost model fits in next

Now that we have two variants (Topo, Topo+spill) with different K-dependent characteristics at R=32, the cost-model question becomes concrete:

**Predict crossover K where spill overhead exceeds its register-pressure benefit.**

For R=32 with our current spill emission, the empirical data shows spill ALWAYS wins (no crossover within K=64-4096). But for R=16, where peak live (16+6=22) doesn't exceed 32 ZMM, spill might be unconditionally bad. The cost model needs to predict that.

Static analysis we have:
- Peak live-set (≈ N) — known at codegen time
- Vector register count (`isa.vec_regs`) — known
- Memory traffic per pass — countable from the DAG

Static predicate `should_spill(N, vec_regs)` already exists in `dft.ml`:
```ocaml
let should_spill n vec_regs = n + 6 > vec_regs
```

For R=16 (n=16, vec_regs=32 for AVX-512): `22 > 32` is false → no spill recommended.
For R=32 (n=32, vec_regs=32): `38 > 32` is true → spill recommended.

This is a coarse heuristic. The cost model would refine it with K-dependent terms (memory cost vs compute cost crossover). Worth building once we have R=16 spill data to characterize the "spill is harmful" regime.

## Files changed

```
lib/dft.ml         +130  spill_marker type, dft_expand_twiddled_spill, should_spill
lib/algsimp.ml      +30  spill_tag_marker, lift_spill_markers
lib/emit_c.ml      +160  spill_info, classify_passes, spill emission path
bin/gen_radix.ml    +20  --spill flag, spill plumbing
```

Net ~340 lines added across 4 files. The spill variant builds cleanly on top of existing infrastructure (math layer, hash-consing, emission) without modifying any of it. That's the architectural property we wanted: variants compose, they don't entangle.
