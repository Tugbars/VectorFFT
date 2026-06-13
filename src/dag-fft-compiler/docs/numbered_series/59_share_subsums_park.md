# Doc 59 — Cluster-aware share_subsums: safety machinery in place, no win realized

## Goal

Activate `share_subsums` for composite codelets (R ≥ 32, n1 and t1). The pass
was effectively dead code in the post-doc-58 pipeline due to a gating bug
(`aggressive=true && !is_direct` but `is_direct = aggressive`). The hypothesis
was a 3–5% arithmetic-count reduction once the pass could fire on composites.

## Result

Safe to enable, but always net-negative on op count. The guard activates on
every codelet and reverts to `factored`. End state is performance-identical
to baseline.

| R   | variant | baseline fp | with share_subsums fp |
|-----|---------|-------------|------------------------|
| 32  | n1      | 430         | 430                    |
| 32  | t1      | 554         | 554                    |
| 64  | n1      | 1085        | 1085                   |
| 64  | t1      | 1337        | 1337                   |
| 128 | n1      | 2645        | 2645                   |
| 128 | t1      | 3153        | 3153                   |
| 256 | n1      | 6229        | 6229                   |
| 256 | t1      | 7249        | 7249                   |

Bit-exact correctness vs baseline across all tested codelets (max_err = 0.0).

## What was built (and kept in tree)

`lib/algsimp.ml`:

1. `compute_cluster_map` — given `Dft.spill_marker list` and `assigns`, walks
   markers (PASS1 bit) and assigns-stopping-at-markers (PASS2 bit), returns
   tag → cluster-bits map. Marker tags themselves receive both bits (they
   bridge clusters via spill/reload).

2. Module-level constants `pass1_bit = 1`, `pass2_bit = 2`.

3. `share_subsums` extended with `?frozen_tags` and `?cluster_of_tag`
   optional parameters. When `cluster_of_tag = None`, behaves as before;
   when supplied, restricts substitutions to PASS-2-visible nodes.

4. `substitute_safe : t -> bool` — substituting a PASS 2 term with existing
   node Y is safe iff Y's cluster includes PASS 2. Plugged into
   `find_shareable_pair` alongside `used_elsewhere`.

5. `is_frozen : int -> bool` — frozen-tag predicate. `rewrite` returns frozen
   nodes unchanged and stops descending; their subtrees are PASS 1 work that
   PASS 2 rewriting must not touch.

6. `valid_in_pass2 : t -> bool` — post-rewrite validator. Walks the rewritten
   tree, fails if any non-frozen tag has cluster bits that don't include
   PASS 2 (catches the case where `rebuild_sum_binary`'s `mk_add_binary`
   hashconses with an existing PASS-1-only node, silently creating a
   cross-cluster reference). If validation fails, the per-output rewrite
   falls back to the original tree.

7. `rebuild_sum_binary` switched from left-leaning `fold_left mk_add_binary`
   chain to interleaved-split build mirroring `emit_pair_fold`. Includes a
   pre-sort by tag for canonical ordering (no measurable effect, but
   structurally cleaner — preserves the option to share prefixes if term
   sets ever do match across outputs).

`bin/gen_radix.ml`:

1. `frozen_tags` and `cluster_of_tag` computed once before `share_subsums`
   call (when `spill_markers <> []`); the same `frozen_tags` is reused by
   `fma_lift` later (replaces the previous duplicate computation).

2. `share_subsums` invoked with `~aggressive:true ~frozen_tags ~cluster_of_tag`
   on composites (no longer gated to `is_direct = false`, which was always
   true under the prior gating bug).

3. Top-level op-count guard: `if count_ops shared < count_ops factored then
   shared else factored`. This is the safety net that keeps us at parity
   when the rewrite is net-negative.

## Diagnostic numbers from instrumented runs (R = 64 n1, before final cleanup)

```
share_subsums: safe=258 unsafe=1337 fallback=86 (of 128 assigns)
share_subsums: net loss (1160 -> 1224), reverting
```

- **1595 substitution candidates** considered by `find_shareable_pair`
  across both halves of R=64.
- **258 (16%)** passed the cluster-safety gate (existing Add(a,b) visible
  in PASS 2).
- **1337 (84%)** were rejected: existing Add(a,b) lived in PASS 1 only,
  unsafe to import into PASS 2.
- Of the 258 substitutions actually performed, **86 of 128 assigns (67%)**
  hit the post-rewrite per-output validation fallback — the
  `rebuild_sum_binary`'s `mk_add_binary` hashconsed with a PASS-1-only
  node despite the gate, and the per-output validator reverted.
- The ~30–40 outputs that survived all checks contributed a **+74 op net
  loss** (1160 vs 1085 baseline = ~1.29 new ops created per substitution
  while each substitution saves only 1 op).

## Why the rewrite never wins (root cause)

Three independent measurements established a complete picture:

**1. The rebuild cost dominates the substitution savings.**
Per-substitution savings: 1 op (the substituted Add(a,b) is reused
instead of recomputed). Per-substitution rebuild cost: ~1.29 new
intermediate ops on average (empirical, R = 64 n1). Net: negative on every
codelet.

**2. Canonical sorting before interleaved split changes nothing.**
Sorting the post-substitution term list by tag before `mk_add_binary`'s
interleaved split produces identical op counts to unsorted (R=64 n1: 1159
in both cases). The post-substitution term lists across different outputs
don't share long prefixes, so canonical ordering can't recover the
cross-output sharing that `emit_pair_fold` provides at construction time.

**3. No high-value substitution candidates exist.**
Tightening `used_elsewhere` from `≥ 1` to `≥ 3` produces ZERO
substitutions — meaning no `Add(a,b)` in our codelets has use_count ≥ 3.
emit_pair_fold's interleaved split creates intermediates that are
mostly shared between exactly 2 outputs (use_count = 2) or used only
once (use_count = 1). No `Add(a,b)` is shared widely enough for a
substitution to save more than 1 op.

These three together mean share_subsums fundamentally cannot deliver a
win in this pipeline. The "3–5% arithmetic count save" expectation was
incorrect for the post-doc-58 DAG topology: emit_pair_fold + the
factor_common_muls / dedup_sub_pairs / fma_lift chain already finds the
sharings worth finding. There's no remaining structural slack for
share_subsums to recover.

## What was tried in this investigation (in order)

1. **Add `frozen_tags` to `share_subsums`** — prevents descending into
   spill-marker subtrees. Required for safety but not sufficient.

2. **Add `cluster_of_tag` + `substitute_safe`** — gates `find_shareable_pair`
   on PASS 2 visibility of substitute target. Filters 84% of candidates.
   Still produces +6.8% op count vs baseline at R=64 n1.

3. **Switch `rebuild_sum_binary` from left-leaning to interleaved split** —
   mirrors `emit_pair_fold`'s shape. Doesn't change op count (still +6.8%).
   The interleaved shape alone can't recover sharing when term lists
   differ between outputs.

4. **Add `valid_in_pass2` post-rewrite validator** — catches cross-cluster
   refs introduced by `mk_add_binary`'s hashcons unification with
   PASS-1-only nodes. Reverts per-output to original when caught. About
   67% of rewritten outputs hit fallback. Still +6.8% net.

5. **Top-level op-count guard** — reverts the whole rewrite if global op
   count increased. Always fires. Brings the codelet back to baseline
   parity. This is the final safety net.

6. **Canonical sort by tag in `rebuild_sum_binary`** — no measurable effect.

7. **Tighter `used_elsewhere` thresholds (≥ 2, ≥ 3, ≥ 4)** — `≥ 2` filtered
   nothing additional (cluster gate already dominant). `≥ 3` and `≥ 4`
   produced zero substitutions, confirming no high-use-count candidates
   exist.

## What's in the tree as a result

The machinery is correct and ready to activate. It currently never makes
things better because the guard always reverts, but it also never makes
them worse. If a future change to construction (e.g., a different pair
selection in `emit_pair_fold`, or `factor_common_muls` being made more
aggressive) creates higher-use-count Add nodes, the guard would naturally
let those wins through.

To fully revert: remove `compute_cluster_map`, `pass1_bit`/`pass2_bit`
constants, the `?frozen_tags`/`?cluster_of_tag` parameters and their uses
in `find_shareable_pair`/`rewrite`/`valid_in_pass2` from
`lib/algsimp.ml`; remove the early `frozen_tags + cluster_of_tag`
computation and the op-count guard from `bin/gen_radix.ml`. The
`rebuild_sum_binary` interleaved-build is unrelated to the
share_subsums work and can be kept (it's structurally cleaner than the
old left-leaning chain regardless).

Alternatively, keep the machinery in place as documented dead-code
infrastructure with the negative result attached, in case a future
restructuring (e.g. a speculative pre-pair construction pass that
boosts Add(a,b) use-counts) makes share_subsums viable.

## Files

- `lib/algsimp.ml` — `compute_cluster_map`, cluster-aware `share_subsums`,
  `valid_in_pass2`, interleaved `rebuild_sum_binary`. Snapshotted to
  `algsimp_doc59.ml`.
- `bin/gen_radix.ml` — early `frozen_tags`/`cluster_of_tag` computation,
  share_subsums called with cluster awareness, op-count guard. Snapshotted
  to `gen_radix_doc59.ml`.

## Correctness verification

R = 32/64/128 × n1/t1 × AVX-512: bit-exact match against baseline at
K = 8 (max_err = 0.0e+00 in all cases). Output byte-level differs from
baseline (hashcons tag ordering varies due to the extra `lift_spill_markers`
call inside `compute_cluster_map`), but the semantic output is identical.

## Pointer to related work

- Doc 28 — original "share_subsums hurts composites" misdiagnosis.
- Doc 56 — `fma_lift`'s `liftable_mul` bug (the actual root cause Doc 28
  misidentified). Doc 56 is the legitimate win in this neighborhood.
- Doc 57 — retracted "three-DAG-roots invariant" misdiagnosis. The cluster
  awareness implemented here is the correct generalization of what Doc 57
  was trying to articulate.
- Doc 58 — n1 blocking. Independent of this work; both can coexist.
