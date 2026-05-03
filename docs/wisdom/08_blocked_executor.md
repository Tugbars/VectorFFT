# 08 — Blocked executor

The third dimension in plan space — orthogonal to factorization and
variant choice. Used at small-K, medium-N cells where the standard
executor's data-traversal pattern thrashes cache.

## Why "third dimension"

Plan-level wisdom records three orthogonal choices per `(N, K)`:

1. **Factorization + variants** — the chosen multiset, stage order,
   and per-stage codelet variant
2. **Orientation** — DIT-forward vs DIF-forward (`use_dif_forward`)
3. **Executor structure** — standard vs blocked (`use_blocked`)

The first two compose freely. The third currently doesn't compose
with variants in v1.1 — when blocked wins, the wisdom entry is stamped
with default variants (`has_variant_codes = 0`).

## What the standard executor does

Ordinary execution: walk the plan stage-by-stage, each stage processes
all `N*K` elements top-to-bottom. Within a stage, the codelet sees `K`
columns and loops over them; `K` is the inner loop bound.

For `K = 256`, this is fine — each stage's working set is bounded by
codelet operand size + the K-strided access pattern stays in L1/L2.

For very small `K` (K=4, K=8) at large N, this changes. The codelet
loops over K=4 elements, then advances to the next radix-R group at
stride `K`. Stride-4 access at large N means data that should be hot
in L1 from the previous stage gets cold by the time the next group
needs it.

## What the blocked executor does

Process K in **blocks of `block_groups` groups** starting at
`split_stage`:

```
Standard executor:
  Stage 0: process all N*K elements
  Stage 1: process all N*K elements
  ...
  Stage L-1: process all N*K elements

Blocked executor (split at stage 2, block_groups = 4):
  for block_start in range(0, N/R_2, block_groups):
    Stage 0: process this block's elements
    Stage 1: process this block's elements
    Stage 2..L-1: process this block's elements
```

By interleaving stage execution within a block, the data stays in L1
across stages. The block size is tuned so the per-block working set
fits L1.

The split point matters: stages before `split_stage` run in the
standard pattern (their access pattern doesn't benefit from blocking),
stages from `split_stage` onward run blocked.

## When blocking wins (empirical)

Per the calibrator's `STRIDE_BLOCKED_K_THRESHOLD`:

```c
#define STRIDE_BLOCKED_K_THRESHOLD 8  /* try blocking only when K ≤ 8 */
```

Empirical, calibrated against i9-14900KF (Raptor Lake). The threshold
is host-specific — a CPU with different L1 size or different
prefetcher behaviour might want a different cutoff. On the calibration
host:

- **K ≤ 8 + N ≥ 512**: blocking can win 30–60% over standard at some
  cells
- **K ≥ 16**: standard executor's natural inner loop already keeps
  things hot enough; blocking adds overhead without help
- **N < 512**: working set fits L1 anyway, no benefit

The threshold is empirical, not theoretical — it captures what the
14900KF actually does, not a generalizable property of the K-blocking
algorithm. On AMD Zen 4 or Sapphire Rapids the right cutoff might
differ.

## Calibrator integration

`calibrate_tuned.c:try_blocked_refine` runs after the variant cartesian
search:

```c
if (K <= STRIDE_BLOCKED_K_THRESHOLD && N > 512) {
    /* try the blocked executor with various split + block_groups */
    int blocked_split = 0, blocked_bg = 0;
    double blocked_ns = 1e18;
    int use_blocked = try_blocked_refine(N, K, &dec, reg, deploy_ns,
                                          &blocked_split, &blocked_bg,
                                          &blocked_ns);
    if (use_blocked) {
        /* blocked won — overwrite wisdom entry */
        deploy_ns = blocked_ns;
    }
}
```

If blocked wins, the wisdom entry's `use_blocked = 1` with the chosen
`split_stage` and `block_groups`. Since v1.1 doesn't compose blocking
with variants, the entry's `has_variant_codes = 0` and the variant
selection at runtime falls through to `wisdom_bridge` predicates.

## Phase B catches large-N small-K cases

Phase A of the calibrator (the wisdom-driven `stride_wisdom_calibrate_full`)
only runs blocking for `N <= EXHAUSTIVE_MAX_N = 2048`. Phase B
explicitly covers `N > 2048 && K ≤ 8`:

```c
/* ── Phase B: large-N joint blocked for small K ─────────────────── */
if (K <= STRIDE_BLOCKED_K_THRESHOLD && N > EXHAUSTIVE_MAX_N) {
    stride_factorization_t jb_fact;
    int jb_use_blocked = 0, jb_split = 0, jb_bg = 0;
    double joint_ns = stride_dp_plan_joint_blocked(...);
    if (joint_ns < 1e17) {
        /* refine-bench the joint winner */
        ...
        if (refined < std_ns) {
            stride_wisdom_add_full(wis, ..., jb_use_blocked, jb_split, jb_bg);
        }
    }
}
```

Without this, blocking wins at N=4096 K=4 etc would be missed because
Phase A's blocked search is gated below 2048.

## Why it doesn't compose with variants in v1.1

Architectural mismatch: the blocked executor's per-block traversal
expects each stage's codelet to be the **same type** across the block
(otherwise the inner loop's expectations break). When variants vary
per-stage, the codelet pointers vary too, and the blocked executor
would need different inner-loop shapes per stage.

Solving this is straightforward — emit a per-block dispatch that
matches the stage's variant. But it's not a v1.0 priority because
cells where blocking wins are mostly K=4 / K=8 plans, where T1S
already dominates and the variant-axis flexibility loss is small.

When the blocked executor gains variant composition (v1.1+), the
wisdom format gets `has_variant_codes = 1` for blocked entries too,
and the planner builds via `_stride_build_plan_explicit` which sets
the executor's blocked flags afterwards.

## Lookup-time behaviour

When `stride_wise_plan` hits a wisdom entry with `use_blocked = 1`:

```c
if (e->has_variant_codes) {
    /* Build with explicit variants; then stamp blocked flags. */
    plan = _stride_build_plan_explicit(...);
    plan->use_blocked = e->use_blocked;
    plan->split_stage = e->split_stage;
    plan->block_groups = e->block_groups;
}
else if (e->use_dif_forward) {
    plan = _stride_build_plan_dif(...);
    plan->use_blocked = ...;
}
else {
    /* Legacy v3/v4 path (transitional, scheduled for removal) */
    plan = _stride_build_plan(...);
    plan->use_blocked = ...;
}
```

The blocked flag is set on the plan struct *after* construction. The
executor reads the flag at execute time:

```c
void stride_execute_fwd_auto(stride_plan_t *plan, double *re, double *im) {
    if (plan->use_blocked) {
        _stride_execute_fwd_blocked(plan, re, im,
                                     plan->split_stage,
                                     plan->block_groups);
    } else {
        stride_execute_fwd(plan, re, im);
    }
}
```

So `use_blocked` is a runtime dispatch, not a build-time codelet
choice. Variants are build-time; blocking is execute-time.

## Knobs visible in the wisdom file

Three integer fields per entry:

```
... use_blocked split_stage block_groups ...
... 0          0           0            ...   /* standard executor */
... 1          2           4            ...   /* blocked, split at stage 2, 4 groups/block */
```

For the v1.0 grid, expect blocked entries to cluster at:

- N ∈ {512, 1024, 2048, 4096, 8192} × K=4
- N ∈ {2048, 4096, 8192} × K=8

Outside that band, `use_blocked = 0` is universal.

## See also

- [05_calibrator_pipeline.md](05_calibrator_pipeline.md) — where blocked refine fits in the calibrator's phase structure
- [04_layer2_plan_level.md](04_layer2_plan_level.md) — wisdom file's `use_blocked` / `split_stage` / `block_groups` columns
- [`src/core/executor_blocked.h`](../../src/core/executor_blocked.h) — the blocked executor implementation
- [`src/core/dp_planner.h:stride_dp_plan_joint_blocked`](../../src/core/dp_planner.h) — Phase B's joint blocked search
