# 04 — Layer 2: plan-level wisdom

The primary wisdom system. Per (N, K) cell, an explicit record of
the winning factorization, per-stage variant codes, and orientation,
backed by `bench_plan_min` measurements.

## What gets stored per (N, K)

```c
typedef struct {
    int N;
    size_t K;
    int factors[FACT_MAX_STAGES];     /* the chosen factorization */
    int nfactors;                     /* its length */
    double best_ns;                   /* deploy-bench ns/iter */
    int use_blocked;                  /* 0 = standard, 1 = blocked executor */
    int split_stage;                  /* first blocked stage  (if use_blocked) */
    int block_groups;                 /* groups per block at split stage */
    int use_dif_forward;              /* 0 = DIT, 1 = DIF */
    int has_variant_codes;            /* 1 = explicit codes, 0 = legacy */
    int variant_codes[STRIDE_MAX_STAGES];  /* per-stage variant: 0/1/2/3 */
} stride_wisdom_entry_t;
```

The `variant_codes[]` array is the v5 addition that makes plan-level
wisdom meaningful — without it, the entry just records "use this
factorization", same as v3/v4. With it, the calibrator's exact
per-stage variant choice is preserved.

## File format (v5)

```
@version 5
# VectorFFT stride wisdom — N entries
# N K nf factors... best_ns use_blocked split_stage block_groups use_dif_forward variant_codes... (v=0:FLAT 1:LOG3 2:T1S 3:BUF)
8 4 1 8 9.83 0 0 0 0 0
8 32 1 8 41.43 0 0 0 1 0
16 256 2 4 4 1045.08 0 0 0 0 0 2
32 256 2 4 8 2963.93 0 0 0 0 0 2
60 4 2 12 5 129.16 0 0 0 0 0 1
64 256 3 4 4 4 8439.34 0 0 0 0 0 2 2
1024 256 5 4 4 4 4 4 412765.00 0 0 0 0 0 2 2 2 2
4096 256 7 2 4 2 4 4 4 4 2226600.01 0 0 0 0 0 2 0 2 2 2 2
```

Per entry, fields in order:

| Field | Width | Meaning |
|-------|-------|---------|
| `N` | 1 | transform size |
| `K` | 1 | batch size |
| `nf` | 1 | factorization length |
| `factors` | nf | the radixes, in plan order |
| `best_ns` | 1 | deploy-bench `ns/iter` |
| `use_blocked` | 1 | blocked-executor flag |
| `split_stage` | 1 | first blocked stage (when `use_blocked=1`) |
| `block_groups` | 1 | groups per block at split stage |
| `use_dif_forward` | 1 | orientation flag |
| `variant_codes` | nf | per-stage variant code (one per stage) |

Total fields per entry: `8 + 2*nf`. Stage 0's variant code is by
convention `FLAT` (it has no twiddle codelet) but is preserved in the
file for parsing regularity.

Special placeholder: `variant_codes[s] = -1` means *legacy entry, no
explicit variant code recorded* (v3/v4 entries pre-loaded into v5
format). The loader sets `has_variant_codes = 0` for any entry with
`-1` codes; the planner falls through to predicate-driven plan-build.

## Variant code semantics

Per `vfft_variant_t`:

| Code | Variant | Stage type |
|------|---------|------------|
| 0 | FLAT | n1 at stage 0; t1 elsewhere |
| 1 | LOG3 | t1_dit_log3 |
| 2 | T1S | flat + t1s overlay |
| 3 | BUF | t1_buf_dit (R=16/32/64 only) |

Important detail: code 2 (T1S) means the flat codelet *and* t1s are
both attached to the stage. The executor's runtime dispatch picks t1s
because `t1s_fwd` is non-NULL. The plan-level wisdom doesn't have a
"pure t1s" code — `T1S` always means "flat-with-t1s-overlay."

Code 3 (BUF) is different: the flat codelet pointer is replaced with
the buf dispatcher pointer. No overlay. The within-flat dispatcher
selection is encoded directly at this level.

## Blocked-executor flag

`use_blocked = 1` enables a different executor that processes K in
blocks of `block_groups` groups starting at `split_stage`. This is a
cache-friendliness optimization for small-K large-N cells.

When `use_blocked = 1`, the variant codes are *not* applied — the
blocked executor doesn't compose with variants in v1.1. The wisdom
entry has `has_variant_codes = 0` and the planner falls through to
the legacy build path.

See [08_blocked_executor.md](08_blocked_executor.md) for details.

## Orientation flag

`use_dif_forward = 1` means the whole forward pass uses DIF codelets
(and the backward pass uses DIT). Default (`= 0`) is DIT-forward,
DIF-backward — the original orientation.

DIF is **whole-plan-or-nothing**. You can't mix DIT and DIF stages
within the same plan because they compute different output buffers
from the same input. The calibrator benches both orientations per
cell and records the winner; per-stage variant choice happens *within*
the chosen orientation.

DIF-only constraints:

- T1S and BUF protocols don't have DIF analogs → variant codes 2 / 3
  are invalid in DIF orientation; the calibrator filters them out.
- LOG3 in DIF is registered for R=16/32/64 only.
- Other radixes' DIF stages can only use FLAT.

See [07_dif_filter.md](07_dif_filter.md).

## Version history

The format has evolved as the calibrator gained sophistication:

| Version | Date | Adds | Reason |
|---------|------|------|--------|
| v3 | pre-2026-04 | factorization + ns + blocked metadata | initial |
| v4 | 2026-04 | `use_dif_forward` | DIT-vs-DIF whole-plan bench landed |
| **v5** | 2026-04-26 | `variant_codes[]` + `has_variant_codes` | plan-level joint search produced explicit per-stage codes |

The loader reads all three versions; older entries get `has_variant_codes
= 0` and fall through to predicate-driven plan-build at lookup time.
The writer always emits v5.

`stride_wisdom_save` always writes the latest version; mismatched
versions trigger a re-calibration on next launch.

```c
#define WISDOM_VERSION 5
```

When this bumps, old files are silently rejected on load — the
incompatible entries get re-measured rather than mis-applied.

## In-memory representation

```c
typedef struct {
    stride_wisdom_entry_t entries[WISDOM_MAX_ENTRIES];
    int count;
} stride_wisdom_t;
```

Linear array, lookup by linear scan. `WISDOM_MAX_ENTRIES = 256` —
plenty for the v1.0 grid (about 200 entries shipped) but a hardcoded
ceiling that would need bumping for a much larger grid.

Lookup is `O(count)` — fine for a 200-entry table called once at plan
creation.

## How entries get added

Three paths:

1. **`stride_wisdom_calibrate_full`** (`planner.h`) — internal
   factorization-search-then-add helper. Used by Phase A of the
   calibrator. Produces v3-shaped entries (no variant codes).

2. **`stride_wisdom_add_v5`** (`planner.h`) — direct add with all
   v5 fields. Used by Phase D of the calibrator after the variant
   cartesian search picks per-stage codes.

3. **`stride_wisdom_load`** (`planner.h`) — reads from disk, adds each
   entry into the in-memory table.

The calibrator goes 1 → 2: Phase A populates the entry with default
variants (legacy v3/v4 shape); Phase D updates it with explicit codes
(promotes to v5 shape). `stride_wisdom_add_v5` only commits when
`best_ns` decreases — idempotent across re-runs.

## Sample entries with annotations

```
8 256 1 8 334.02 0 0 0 0 0
```
N=8 K=256, single-stage R=8, 334 ns, no blocking, DIT, stage 0 = FLAT
(stage 0 always FLAT — no twiddle codelet anyway).

```
16 256 2 4 4 1045.08 0 0 0 0 0 2
```
N=16 K=256, 4×4, 1045 ns, DIT, stage 0 FLAT (n1), stage 1 T1S (t1+t1s
overlay).

```
4096 256 7 2 4 2 4 4 4 4 2226600.01 0 0 0 0 0 2 0 2 2 2 2
```
N=4096 K=256, 7-stage 2×4×2×4×4×4×4, 2.23 ms, DIT.
Variants: stage0 FLAT, stage1 T1S, stage2 FLAT, stages 3-6 T1S.
Notice stage 2 is FLAT despite being a twiddled stage — the calibrator
specifically measured T1S at this stage and FLAT won.

```
1024 1024 5 4 4 4 4 4 412765.00 0 0 0 0 0 2 2 2 2
```
N=1024 K=1024, 4×4×4×4×4, 413 µs. Stage 0 FLAT, stages 1-4 T1S. The
typical pow2 pattern at K=1024.

Statistics across the shipped wisdom file (198 cells, 735 stages 1+):

- T1S wins **84%** of stages 1+
- LOG3 wins **10%** (mostly R=13/17/25/32/64)
- FLAT wins **6%** (mostly R=12/16/20)
- BUF wins **<1%** at this calibration density (would rise with more
  cells in the K=4 / large-N range)

See `docs/cost_model/figures/variant_share_by_radix.png` for the
breakdown by radix.

## See also

- [03_layer1_per_radix.md](03_layer1_per_radix.md) — the legacy layer this one replaced
- [05_calibrator_pipeline.md](05_calibrator_pipeline.md) — how entries are produced
- [06_lookup_pipeline.md](06_lookup_pipeline.md) — how entries are consumed
- [`src/core/planner.h`](../../src/core/planner.h) — `stride_wisdom_t`, `stride_wisdom_entry_t`, `stride_wisdom_save/load`
- [`build_tuned/vfft_wisdom_tuned.txt`](../../build_tuned/vfft_wisdom_tuned.txt) — current shipped wisdom file
