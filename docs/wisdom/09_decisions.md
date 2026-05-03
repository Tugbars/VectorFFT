# 09 — Decision rationale

ADR-style record of the wisdom system's non-obvious decisions. Each
entry captures *what we considered, what we picked, and why* — so
future maintainers don't re-litigate already-settled trade-offs from
an incomplete view of the alternatives.

Conventions: same as `docs/cost_model/09_decisions.md`.
- **ACTIVE** — current decision; reverting requires a real reason
- **SUPERSEDED** — preserved for history; the superseder has its own ADR
- Each entry should be self-contained

---

## ADR-001 — Reject per-codelet isolation wisdom; plan-level joint search is the verdict

**Status**: ACTIVE (the headline decision; v1.2 pivot, 2026-04-26)
**Date**: 2026-04-26

### Context

The original wisdom architecture: orchestrator runs each codelet *in
isolation* across a `(me, ios)` grid for each radix R, picks per-(R,
me, ios) winners, emits per-radix predicate headers
(`vfft_r{R}_plan_wisdom.h`), distributes them to consumers via
`wisdom_bridge.h`. The planner consults the predicates per stage at
plan-build time.

Clean, modular, fast to compute. **Empirically wrong.**

### What we considered

1. **Keep per-codelet wisdom** as primary, accept the inaccuracy.
2. **Build plan-level wisdom on top** as an override layer; per-codelet
   stays as primary.
3. **Replace per-codelet wisdom** with plan-level joint search; demote
   per-codelet to coarse-prior + cell-miss fallback.
4. **Eliminate per-codelet wisdom** entirely; use cost model + plan-
   level wisdom only.

### What we picked

Option 3. Plan-level joint search became primary; per-codelet
predicates demoted.

### Why

Validation pilot at N=4096 (memory: `v1_2_joint_search_validation.md`):

- Per-codelet wisdom said one factorization with one set of variants.
- Plan-level joint search said a different factorization with different
  variants.
- The plan-level winner was **+40% at K=4 / +16% at K=256** over the
  per-codelet pick.

That's far beyond measurement noise — it's a structural signal. The
per-codelet bench measures a quiet pipeline; the real plan runs with
contention from previous stages (cache pollution, DTLB warm/cold,
load-port pressure, OoO window state). Per-codelet isolation can't
predict per-codelet-in-plan performance on modern OoO uarches.

Option 1 was the easy default but knowingly leaving 16-40% on the
table is unacceptable. Option 2 felt safe but doubles maintenance
cost without solving the underlying composition problem. Option 4 was
too disruptive — the predicates have real coarse-prior value in the
MEASURE pipeline (Upgrade F, see `docs/wisdom/05_calibrator_pipeline.md`).

Option 3 demotes per-codelet wisdom to two narrowly-scoped roles:

- **Coarse-pass prior in MEASURE** (LOG3-aware probing seeds candidates
  the noisy default-variant coarse pass would miss).
- **Cell-miss fallback** when wisdom doesn't cover an `(N, K)`.

Both roles are explicitly time-limited: as Layer 2 wisdom coverage
expands and Layer 1 retirement plan progresses (Phase 2: shrink to
static table, Phase 3: delete), per-codelet predicates are scheduled
for full removal.

### Trade-offs

- **Loss**: Cell-miss fallback gives a "best-guess" plan from per-
  codelet data. Quality is worse than calibrated wisdom but better
  than cost-model-only. Acceptable as a transitional state.
- **Loss**: Coarse-pass prior is implicit and could break silently if
  the predicate emit ever drifts from what's measured. Mitigated by
  the predicate emit being deterministic from bench output.
- **Gain**: Wisdom quality at calibrated cells. Plan-level numbers
  beat per-codelet picks by 16-40% at the validation cells.
- **Gain**: Reproducibility — plan-level wisdom is what the executor
  actually runs, so it's directly verifiable by re-benching.

### Future

Per-codelet wisdom (`vfft_r{R}_plan_wisdom.h` and `wisdom_bridge.h`)
is **scheduled for deletion** once:

- Layer 2 wisdom covers all production (N, K) combinations
- The MEASURE coarse pass switches from predicate-driven LOG3 priming
  to unconditional LOG3 probing (already mostly the case via
  `vfft_variant_available`)
- Bluestein/Rader inner cells are explicitly calibrated rather than
  relying on cell-miss fallback

Tracked in `memory/roadmap_wisdom_bridge_retirement.md`.

---

## ADR-002 — Top-K = 5 at the coarse → refine boundary

**Status**: ACTIVE
**Date**: 2026-04-29 (Upgrade H landed alongside this default)

### Context

The MEASURE planner is two-pass: coarse over factorizations × permutations
with default variants, refine with full variant cartesian on the top-K
coarse survivors. What `K`?

### What we considered

| K_top | Behaviour |
|-------|-----------|
| 1 | legacy v1.2 (variant cartesian on the single best multiset) |
| 3 | early default, captures most variant-axis-flips-multiset cases |
| 5 | current default |
| 10 | additional safety, refine cost grows linearly |
| ∞ | EXTREME (every coarse pair gets full refine) |

### What we picked

5 (`MEASURE_TOPK_DEFAULT`). Behind this is Upgrade H's deploy-rebench
that further filters to within 10% of refine-best, capped at 5
deploy candidates.

### Why

The cost model is `K_top × V × 2 × bench`. At V ≈ 256 (variants per
multiset/orient at a 5-stage plan), bench ≈ 100 ms:

| K_top | refine wall (one cell) | quality vs EXTREME |
|-------|-----------------------|---------------------|
| 1 | ~50 s | ~60% (loses on variant-flips-multiset cells) |
| 3 | ~150 s | ~88% |
| **5** | **~250 s** | **~92% (within 3-12% of EXTREME)** |
| 10 | ~500 s | ~95% (negligible improvement past 5) |

Beyond K=5, the marginal multiset added to refine has too high a
coarse rank to plausibly win after variant search. The 95% → 92%
gap at K=5 vs K=10 is mostly noise, not real misses.

The EXTREME mode (full joint cartesian, ~3000 s/cell) is preserved as
opt-in for users who want PATIENT-quality wisdom. Default ships from
`K_top = 5`.

### Trade-offs

- **Loss**: Pathological cells where the "right" multiset is at coarse
  rank > 5. Mitigated by Upgrade D's top-K-at-every-level recursive DP,
  which keeps sub-problem runners-up alive across recursion frames.
- **Gain**: 15× cheaper than EXTREME at most-of-the-quality.

---

## ADR-003 — DIF whole-plan-or-nothing, not per-stage

**Status**: ACTIVE
**Date**: pre-v1.0 (architectural, baked into executor design)

### Context

DIT and DIF radix codelets compute different intermediate buffers
given the same input/twiddles. They're *algorithmically* duals
(rearranged butterfly + twiddle ordering) but *operationally* not
substitutable per-stage without an explicit transpose-or-permutation
between them. The forward executor in `executor.h` is structurally
DIT-only.

The wisdom system has to decide: support per-stage DIF substitution
(let the planner mix DIT and DIF stages within a single plan), or
constrain DIF to be whole-plan-or-nothing (`use_dif_forward = 1`
applies to all forward stages uniformly).

### What we considered

1. **Per-stage DIF substitution.** Requires an executor that can
   transpose-or-permute between adjacent DIT and DIF stages. Adds
   significant executor complexity and may regress data-layout
   continuity guarantees.
2. **Whole-plan DIF.** One flag in the wisdom entry + plan struct;
   executor dispatches between two parallel implementations
   (`executor.h` for DIT, `executor_dif.h` for DIF).

### What we picked

Option 2. `use_dif_forward` is a whole-plan flag.

### Why

Per-stage DIF would expose more search space — but the actual win from
DIF is in *whole-plan structure* (different access patterns across
the entire plan), not per-stage codelet substitution. A DIT-DIF-DIT
sandwich would force two transposes that eat the gain.

See [07_dif_filter.md](07_dif_filter.md) for what this means at lookup
time.

### Trade-offs

- **Loss**: A small set of cells where DIF wins as one-stage-only
  (rare, by construction — DIF's wins are typically architectural,
  spanning the whole plan).
- **Gain**: Simpler executor; predicates that are honest about what
  the planner can act on.

---

## ADR-004 — Wisdom file format v5: explicit per-stage variant codes

**Status**: ACTIVE (supersedes v3/v4 per-cell formats)
**Date**: 2026-04-26 (alongside ADR-001)

### Context

Pre-2026-04-26 wisdom format (v3/v4) stored only the factorization,
orientation, and blocked-executor flags. Variant choice was deferred
to plan-build, which consulted the per-codelet predicates.

After ADR-001 made plan-level joint search the verdict, we needed
the wisdom file to record the **plan-level chosen variants** — not
re-derive them from predicates at lookup.

### What we considered

1. **Keep v4** — store factorization only, let predicates handle
   variants. Loses information that the calibrator measured per-stage.
2. **v5 with explicit variant codes** — one code per stage, written to
   wisdom. Lookup builds the plan exactly as measured.
3. **Compose v4 + a side-channel variant table** — keep variant codes
   separate from the main wisdom file. Two files to load, easier to
   diff.

### What we picked

Option 2 — codes inline in the wisdom file as an additional set of
columns per entry.

### Why

The wisdom file is one row per cell already; adding `nf` more integers
costs near-nothing in file size (~10 bytes/row). Splitting into two
files would create a reload-coherency burden — what if you load v5
data with mismatched variant table from a previous run? Inline keeps
all wisdom for a cell in one record.

The `has_variant_codes` flag distinguishes v5 entries (with codes)
from v3/v4 entries that got loaded as-is into v5 format (codes set
to placeholder `-1`). The planner uses the flag to decide between
explicit-build (v5 path) and predicate-driven build (legacy path).

### Trade-offs

- **Loss**: Format complexity at the parser level — v3/v4/v5 each have
  slightly different field counts. Mitigated by writing always at v5
  and treating older versions as silent re-calibration triggers.
- **Gain**: Wisdom hits build the exact plan the calibrator measured.
  No drift between calibration and execution.

---

## ADR-005 — Per-codelet wisdom infrastructure scheduled for deletion

**Status**: ACTIVE (closes ADR-001 with a deletion plan)
**Date**: post-v1.0

### Context

ADR-001 rejected per-codelet wisdom as a methodology and made
plan-level joint search the primary system. The per-codelet
infrastructure (`vfft_r{R}_plan_wisdom.h` headers, `wisdom_bridge.h`
dispatcher, the predicate-driven branches in `_stride_build_plan`)
was kept temporarily because it carries two transitional roles:

1. Coarse-pass LOG3 priming in MEASURE — informed by which radixes
   have LOG3 codelets (already mostly handled by
   `vfft_variant_available`)
2. Cell-miss fallback at lookup time — used when wisdom doesn't
   cover a particular (N, K)

### What we picked

Delete the per-codelet predicate infrastructure post-v1.0. Both
transitional roles get replaced:

- LOG3 priming becomes unconditional registry-driven (probe LOG3
  on every stage where a LOG3 codelet exists; no predicate consult).
- Cell-miss fallback uses the cost model (`stride_estimate_plan`)
  for variant selection at uncalibrated cells, not Layer 1
  predicates.

### Why

The per-codelet predicates were a bad design choice we kept as a
stop-gap. Their continued existence in the codebase invites future
maintainers to write code that depends on them as if they were
stable infrastructure — they aren't. Cleaner to delete them entirely
than leave them as a perennial "we'll get to it" artifact.

### Gating before deletion

Two gates currently keep deletion blocked:

- Calibration grid coverage of Bluestein/Rader inner (M, B) cells
  is incomplete; cell-miss fallback runs in production for primes
  whose inner FFT signatures aren't directly calibrated.
- Cost-model accuracy at uncalibrated cells is good (mean ratio
  ~1.2x of measured) but its variant choice is still informed by
  `wisdom_bridge` predicates today (see `cost_model/` ADR-001).
  When that dependency is severed, cost model takes over the
  miss-path role cleanly.

Both gates are work items in the active roadmap. Once cleared, the
deletion is mechanical: remove the headers, remove `wisdom_bridge.h`,
remove the predicate-call sites in `_stride_build_plan`, regenerate
build files.

### Trade-offs

- **Loss during transition**: legacy code carries documented dead
  weight that future maintainers may misread as live. Mitigated by
  the framing in `00_thesis.md` and this ADR.
- **Gain after deletion**: cleaner architecture, fewer ways for
  cost-model and wisdom-system decisions to drift, smaller
  generated-code surface to maintain.

### Trade-offs

- **Loss**: Codebase still carries the legacy headers. Maintenance
  burden of generating them (orchestrator's per-radix bench).
- **Gain**: Wisdom quality on uncalibrated cells doesn't fall off a
  cliff at v1.0 ship.

The user has indicated the per-radix headers will be deleted from
the project ultimately (post-v1.0). This ADR captures the gating
conditions; the deletion itself will get its own ADR when scheduled.

---

## ADR-006 — Calibration grid centered on production K values

**Status**: ACTIVE
**Date**: bake-in for v1.0

### Context

The calibrator must pick a finite grid of `(N, K)` cells to measure.
Larger grid → more wisdom coverage but quadratically more wall time.
What grid?

### What we considered

1. Dense grid (every power of 2 × every K from 1 to 4096). Hours of
   calibration per radix. Massive coverage but prohibitive wall.
2. Production-distribution grid: K ∈ {4, 32, 256} × N from
   `bench_1d_csv`'s benchmark cells. Targeted coverage of cells users
   actually run.
3. Sparse grid + extrapolation. Few cells, predict the rest. Fragile
   under non-monotonic CPE behaviour.

### What we picked

Option 2 — `GRID_K = {4, 32, 256}` × `GRID_N` matching
`bench_1d_csv.c::all_sizes[]`.

### Why

The benchmark is what we're judged on, not abstract coverage. Cells
in the bench grid are the cells that determine perception. Anything
outside falls through to predicate fallback or cost model — both
work, just at lower quality.

The K choices are:

- **K=4** — minimum batch, used in low-latency contexts. Has the
  worst noise floor, hence MEASURE_REFINE_RUNS=4.
- **K=32** — typical small-batch (audio frames around this size).
- **K=256** — large-batch (image rows, radar pulse-Doppler). Most
  production loads.

Three K values × ~50 N values × ~250 s/cell ≈ 10 hours wall per
calibration run. Acceptable for a quarterly recalibration cadence.

### Trade-offs

- **Loss**: Cells outside the grid (K=8, K=16, K=512, K=1024 for many
  N) get predicate fallback, not direct calibration. Quality at those
  cells is good-but-not-great.
- **Gain**: Calibration completes in tractable wall time. Production
  cells get direct hits.

The grid is documented inline at `calibrate_tuned.c:GRID_N` and
`GRID_K`; users with different production K patterns should adjust
accordingly.

---

## ADR-007 — Variance check delegated to deploy-rebench, not coarse/refine

**Status**: ACTIVE
**Date**: 2026-04-29 (Upgrade H)

### Context

Where in the pipeline do we sanity-check that the chosen plan is
actually fastest, given measurement noise?

### What we considered

1. Strict coarse-pass variance gate — refuse to commit if any coarse
   bench's CV > X%.
2. Strict refine variance gate — same at refine level.
3. Deploy-rebench with independent harness — bench all top-K survivors
   at the end, pick deploy-fastest.

### What we picked

Option 3.

### Why

Coarse and refine use the same `_dp_bench` harness. Their noise is
**correlated** — a candidate that ranked high by accident in coarse
will likely rank high by accident in refine, because the same noise
sources affect both passes. Variance gates within the same harness
detect noise-amplified rankings but can't distinguish "this is
genuinely fastest" from "this happened to win the coin flip twice."

`bench_plan_min` (the deploy harness) has different warmup, fresh
plans, and a min-of-trials protocol. Its noise is **decorrelated**
from `_dp_bench`. If candidate A wins both refine and deploy,
that's two independent estimates agreeing — a much stronger signal
than refine winning twice.

The 10% threshold (`MEASURE_DEPLOY_THRESHOLD_PCT`) is the noise floor
this design admits as "tied" — within 10%, deploy decides; beyond
10%, the refine winner is unambiguous.

### Trade-offs

- **Loss**: Deploy adds wall time (~5 candidates × 100 ms = 500 ms
  per cell). Negligible vs the main pipeline (~250 s).
- **Gain**: Wisdom is reproducible across recalibration runs because
  the final winner depends on two decorrelated bench protocols
  agreeing, not one noise pattern winning twice.

---

## ADR-008 — `STRIDE_BLOCKED_K_THRESHOLD = 8` is empirical, host-specific

**Status**: ACTIVE
**Date**: bake-in for v1.0

### Context

The blocked executor's K-threshold (above which we don't try blocking
because the standard executor's natural inner loop already wins) was
calibrated on i9-14900KF. Should it be a runtime-detected value? A
configurable constant? A theoretical bound?

### What we considered

1. **Empirical fixed constant** (`#define STRIDE_BLOCKED_K_THRESHOLD 8`).
2. **Runtime-detected** based on L1 size, prefetcher behaviour.
3. **Configurable per-host** in the wisdom file (one threshold per
   calibration host).

### What we picked

Option 1.

### Why

Option 2 sounds clean but is fragile: blocking's win comes from the
*specific* interaction of stride-K access patterns with the
prefetcher, not from L1 size alone. A theoretical model would be
wrong on enough cells to be worse than a hardcoded number.

Option 3 is overkill — wisdom file maintenance already requires
recalibration on host change; the threshold can change as part of
that recalibration via `#define` override before compile.

Option 1 documents that the threshold is empirical: tuned for one
host, ships as the default for users on similar hosts (Raptor Lake +
Sapphire Rapids share enough microarchitecture that the threshold
generalizes). Users on materially different hosts (Zen 4, AMD 4465P,
ARM) should re-measure and override the threshold.

### Trade-offs

- **Loss**: Hardcoded constant doesn't adapt. Users on different
  hosts may get sub-optimal blocked decisions.
- **Gain**: Predictable, documented threshold. Easy to override via
  compile-time `#define`. No fragile runtime detection that could
  silently drift.

---

## See also

- [00_thesis.md](00_thesis.md) — the framing all these decisions live inside
- [`memory/v1_2_post_calibration_state.md`](../../memory/) — the historical record of v1.2 decisions
- [`memory/roadmap_wisdom_bridge_retirement.md`](../../memory/) — the Layer 1 deletion plan
- `docs/cost_model/09_decisions.md` — sister ADR file for the cost model
