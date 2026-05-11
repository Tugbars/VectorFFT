# 36. Spill Controller Phase 2 Prototype: Design B Was Already Implemented

## Context

[Doc 35](35_spill_phase1_diagnostic.md) closed Phase 1 of the spill
controller work with a diagnostic of where stack ops come from. The
hypothesis going into Phase 2 was that 60-83% of stack ops are
"GCC-extras" — spills GCC adds beyond what the recipe places, driven
by peak live count exceeding the register file within Pass 1 and Pass 2.

Two design directions were on the table:
- **Design A**: Extend `spill_marker` with sub-cluster IDs, modify
  `classify_passes` in emit_c.ml to handle multi-level boundaries
- **Design B**: Plumb cluster_id through to the scheduler, prefer ready
  nodes from the current sub-DFT

Design B was preferred as the first experiment because it's a
reordering rather than a structural change — worst case, no impact;
best case, substantial stack op reduction.

This doc records the result: **Design B turned out to already be
implemented**, and the current 5216 stack ops at R=512 AVX-512 is the
*floor* of what cluster-aware scheduling can produce. Phase 2 needs
to pivot to Design A.

## What was already in the codebase

Reading `lib/emit_c.ml` more carefully, three pieces of cluster
awareness exist today:

**1. `pass1_blocked_topo` sorted by min_slot.** Around line 717,
emit_c computes `min_slot` for each Pass 1 node (bottom-up walk:
each non-spilled node inherits the smallest spill slot from its
successors), then sorts Pass 1 nodes by `(min_slot, tag)`. Because
slot = n1_idx * N2 + k2, sorting by min_slot ASC clusters by
sub-DFT-N2 automatically — even in the default Topological scheduler.

**2. SU per-cluster refinement.** Around line 748, when the scheduler
is `SU` and `sp.ct_n2 > 0`, emit_c splits `pass1_blocked_topo` into
runs of the same cluster (by min_slot / N2) and runs `su_schedule_subset`
on each cluster's nodes against that cluster's spill sinks.

**3. Pass 2 symmetric clustering.** Around line 912, the same logic
applies to Pass 2 nodes — groups by k2 cluster, runs SU per group.

In short: cluster-aware scheduling is the default behavior when spill
is active, regardless of whether the user passes `--su`. The min_slot
sort imposes cluster boundaries; the SU mode adds within-cluster
ordering refinement.

## Empirical confirmation

Tested R=512 AVX-512 with three scheduler choices:

```
scheduler         stack ops
default           5216
--su              5216
--bisect          29215  (bisection collapses with spill — no surprise)
```

Default and `--su` produce *identical* stack op counts. The default
emission path already imposes cluster boundaries through min_slot
sorting; SU mode adds polish that doesn't change the spill count.

Tested various `--fuse` values (the existing register-resident hint
mechanism that keeps the last K outputs of each sub-DFT-N2 in registers
instead of spilling them):

```
fuse value        stack ops
0  (default)      5216
1                 5216
2                 5216
4                 5216
8                 5216
16                5216
```

The fuse parameter has zero effect on the stack op count at this
size. GCC is doing its own register pressure analysis regardless of
the recipe's hint, and arrives at the same allocation either way.

These two tests together confirm: **5216 stack ops is the floor of
cluster-aware scheduling + register-resident hints on AVX-512 R=512.**
The current pipeline is already producing the best it can within its
design.

## Why the cluster floor isn't lower

The recipe spills every Pass 1 output. For R=512 = CT(16, 32):
- 16 clusters in Pass 1, each is one sub-DFT-32
- Each sub-DFT-32 produces 32 outputs (= 32 spill slot pairs = 64 ZMM-bytes)
- Internal computation of one sub-DFT-32 has peak live ~30-50 register-live
  values
- Total per-cluster peak live: ~60-90 values

This exceeds 32 ZMM. GCC must add extras inside the cluster regardless
of how well we cluster.

The N1 × N2 = N constraint forces *some* factor to be large. For R=512:

```
factorization    Pass-1 cluster   Pass-2 cluster   total stack ops
                 (sub-DFT-N2)     (sub-DFT-N1)     (measured, AVX-512)
CT(16, 32)       DFT-32 (large)   DFT-16 (small)   5216
CT(32, 16)       DFT-16 (small)   DFT-32 (large)   8752
CT(8, 64)        DFT-64 (huge)    DFT-8 (tiny)     (untested at recipe level)
```

CT(32, 16) was tested as a Phase 2 prototype to see if swapping factors
would help. It's *worse*: the small Pass 1 clusters reduce Pass 1 extras
but the large Pass 2 clusters more than compensate. Picker reverted to
CT(16, 32).

The pattern generalizes: at R=512, any 2-level CT has at least one
factor ≥ √512 ≈ 22.6, large enough to overflow the register file
within that pass's clusters. **No single-level recipe can drive stack
ops below the current floor.**

## What Design A does that Design B can't

The structural fix is **multi-level recipe**: apply spill markers not
just at the outermost CT level but at nested levels too. For R=512
with inner recipe at the CT(4, 8) factorization of each sub-DFT-32:

```
Outer level:  spill all 512 Pass 1 outputs (current behavior)
Inner level:  within each sub-DFT-32 = CT(4, 8), also spill its 32
              sub-sub-DFT-8 outputs

Cluster sizes that GCC sees:
- Outermost Pass 1: sub-DFT-32 → broken into sub-sub-DFT-8 clusters
  (~8 outputs + working ≈ 16 live, fits in 32 ZMM)
- Outermost Pass 2: sub-DFT-16 (16 outputs + working ≈ 22 live,
  marginal but fits)
```

Predicted stack op breakdown at R=512 AVX-512 with single inner level:
- Outer recipe: 2048 ops (same as today)
- Inner recipe: 16 outer × 32 inner outputs × 4 = 2048 additional ops
- GCC-extras: should drop substantially since each cluster fits

Net: ~4500-5000 ops. Possibly a 5-15% reduction, not the dramatic
50% Phase 1 hoped for. The recipe-mandated portion *grows* because
we're adding spill points, partially offsetting the GCC-extras reduction.

For larger reductions we'd need to *fuse* some of the inner spills —
the existing fuse mechanism applies at the outer level but doesn't
have an inner-level analog. A multi-level recipe with selective fusion
(keep small subsets register-resident across the inner boundary) could
push lower.

## The other path: change the algorithm choice

A separate observation: the cluster floor depends on the algorithm,
not just the recipe. If R=512 used split-radix instead of CT, the
fan-in/fan-out pattern would differ and the spill structure would too.
But SR doesn't currently have a recipe (doc 31 explicitly punted on
that), so this would be its own multi-week project.

A more practical separate path: **make the Pass 1/Pass 2 structure
itself non-binary**. Currently the recipe has exactly two passes
divided by a spill boundary. A 3-pass or 4-pass structure would
keep per-pass live counts lower automatically. This is approximately
what Design A produces, just with explicit pass boundaries rather
than nested ones.

## Revised plan

Phase 2 pivot:

**Phase 2a (next session): Design A prototype.** Extend `spill_marker`
with a `level` field (0 = outermost, 1 = next inner, etc.). Modify
`dft_expand_twiddled_spill` to recurse one level: when the
sub-DFT-N2's chosen algorithm is also CT, capture its Pass 1 outputs
as additional spill markers. Modify `classify_passes` to handle
multi-level boundaries (Pass1_level0, Pass1_level1, Pass2_level1,
Pass2_level0).

Scope: probably 1-2 days. Touches dft.ml (recursive marker capture),
algsimp.ml (lift through inner CTs), emit_c.ml (multi-level
classification + emission). Validation: prime correctness 56/56,
R=16/R=32 vs hand no regression, stack op count target ≤ 4500 at
R=512 AVX-512.

**Phase 2b: inner-level fuse mechanism.** If 2a gets stack ops to
~4500 but mono still doesn't beat multi-stage's 1728 at R=512 B=512
on AVX2, add a `fuse_inner` parameter that keeps a tunable fraction
of inner spill slots register-resident. Similar to the existing outer
fuse but applies at the inner CT level.

**Phase 2c: validate runtime impact.** Re-run the real-shuffle bench
from doc 34 with the improved spill counts. Confirm whether the
AVX2 B=512 crossover (multi-stage wins by 33%) shifts to parity or
mono win.

## What stays from this session

The diagnostic infrastructure in `/tmp/spill_phase1` and the discovery
that 5216 / 12066 is the cluster-scheduling floor are the carryovers.
Design B is closed — its claim that "scheduler reordering can help" is
false because the reordering is already happening.

Doc 35 stands as written; this doc supplements it with the prototype
result. The Phase 2 entry point in dft.ml's `dft_expand_twiddled_spill`
identified in doc 35 is correct, but the work to do there is
multi-level marker capture rather than the simpler cluster-id
annotation originally sketched.

## Predictions to falsify in the next session

The Phase 2a prototype should confirm or refute:

The first prediction is that GCC-extras drop sharply when inner-level
spill markers are placed. The 3168 extras at R=512 AVX-512 should
fall toward 500-1000 because each cluster fits in registers.

The second is that recipe-mandated ops roughly double (from 2048 to
~4000-4500), partially offsetting the gain.

The third is that net stack ops land around 4500-5500 — a modest 5-15%
improvement, not a transformation.

The fourth is that **runtime improvement is smaller than stack op
reduction would suggest**, because the spill controller is no longer
the only bottleneck — function call overhead, icache pressure, and
µop cache eviction remain.

If predictions 1-3 are roughly correct but 4 is too pessimistic
(runtime improves more than the stack op reduction predicts), it
means the *structure* of spills matters more than the count — recipe
spills are placed at predictable points where GCC's extras are
scattered. That's a separately useful finding.

If predictions 1-3 are wrong (e.g., GCC adds new extras as fast as
we eliminate them), the floor is even harder to push down and we'd
need to think about scheduling-level pressure reduction instead.

Either outcome is informative.
