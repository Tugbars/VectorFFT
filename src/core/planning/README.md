# `core/planning/` — plan search, cost model, wisdom

`core/engine/` *builds and runs* a plan once you hand it a `(factorization, per-stage
variants, orientation)`. **`core/planning/` decides what that triple should be** — and
persists the verdict. It is the layer between "I want an FFT of size N at batch K" and
"here is the exact staged plan that runs fastest on *this* host."

Two kinds of output:
- a **plan**, directly (the cost-model `estimate` path — no measurement), or
- **wisdom** — the measured winners persisted to a file, which `engine/planner.h`'s
  `auto_plan` then consumes at runtime as a pure lookup.

Everything that *measures* (DP, exhaustive, MEASURE) is a **calibration-time** tool:
search once, write wisdom, and the deployed runtime never measures again. `estimate` is
the exception — pure model, usable live for cold cells.

---

## The fidelity ↔ cost spectrum

Three strategies are live — DP, MEASURE and flat exhaustive. (Three more — estimate,
screened exhaustive, patient exhaustive — plus an orchestrator sketch were designed but never
wired into any translation unit; they were removed 2026-09-01.) They all answer the
same question (which `(factors, variants, orientation)` is fastest at `(N,K)`) with
different budgets:

| strategy | file | how it picks | cost | searches variants? |
|----------|------|--------------|------|:------------------:|
| **DP** | `dp_planner.h` | recursive measured search + sub-problem **memoization** | ~150 benches | no (T1S/DIT) |
| **MEASURE** | `measure.h` | DP/exhaustive coarse → **variant + DIT/DIF refine** | ~few hundred | **yes** |
| **flat exhaustive** | `exhaustive_plan.h` | bench **every** (multiset × permutation) at parent (N,K) | ~500–1500 | (factorization) |

The non-obvious split: **DP and exhaustive only search the *factorization* axis** (they
build every candidate all-T1S/DIT). **MEASURE adds the *variant* axis** (per-stage
FLAT/LOG3/T1S) **and orientation** (DIT/DIF) on top of a coarse factorization search —
that is what produces production-grade wisdom (an all-T1S plan leaves the LOG3-tail and
DIF wins on the table; see `engine/README.md` §twiddle for why mixing matters).

**Maps onto FFTW's rigor flags** — and crucially, **the DP planner is itself a *dial*, not
a single point**: its `beam` width + `believe_subplan_cost` toggle span from moderate to
patient within one engine.

| FFTW flag | our equivalent |
|-----------|----------------|
| `FFTW_ESTIMATE` | `estimate` (V4 model, no measurement) |
| `FFTW_MEASURE` | **DP, default** (`set_measure`: beam 3, trust cached sub-costs) |
| `FFTW_PATIENT` | **DP, patient** (`set_patient`: beam 8, re-measure top-K on every cache hit) |
| `FFTW_EXHAUSTIVE` | the exhaustive engines (full multiset × permutation enumeration) |

---

## DP planner (`dp_planner.h`) — the measured core

FFTW-style **recursive decomposition with memoization**. Instead of enumerating every
factorization × ordering (exponential), it decomposes recursively and caches sub-problem
solutions.

**Algorithm** (`_vfft_proto_dp_solve_topk`):
1. To plan `N` at batch `K`, try each registered radix `R` as the **first stage**.
2. Recurse on `M = N/R`, asking for its top-`beam` sub-plans (from the **cache** if seen).
3. Assemble `[R, sub_plan_i]`, **benchmark the full plan** at the real `(N,K)`.
4. Keep the top-`beam` candidates; cache the row keyed by `(N, K_eff)`.
5. Permute the winning factor set and bench each ordering; store the best.

**Complexity:** ~150 benchmarks for N≈100k (vs ~61,000 exhaustive). Cache holds up to
`VFFT_PROTO_DP_CACHE_MAX = 512` rows, each row up to `TOPK_MAX = 8` plans sorted by cost.

**The rigor dial** — the same engine spans FFTW's `MEASURE`↔`PATIENT` range via two knobs,
`beam` (how many sub-plans propagate per node) and `believe_subplan_cost` (whether to trust
cached costs):
- **MEASURE (default, ≈ `FFTW_MEASURE`)** — `beam = 3`; trust the cached sub-plan cost on a
  hit. Cheaper, slightly noisier ranking.
- **PATIENT (`vfft_proto_dp_set_patient`, ≈ `FFTW_PATIENT`)** — `beam = 8`; on every cache
  hit, **re-measure all cached top-K** (best-of-`PATIENT_REMEASURE_RUNS = 2`) and re-sort,
  so a noise-mis-ranked runner-up can climb back. Wider search + jitter/thermal-drift
  antidote over a long calibration. Same code, just turned up.

(Flip with `vfft_proto_dp_set_measure` / `vfft_proto_dp_set_patient` before `dp_plan`.)

**The benchmark harness** (`_vfft_proto_dp_bench`) is the FFTW-style adaptive timer:
best-of-`TIME_REPEAT = 6` trials, each trial ≥ `TIME_MIN_NS = 2 ms` of wall-clock (reps
auto-scaled), capped at `TIME_LIMIT_NS = 0.5 s` per bench. Buffers (`re/im/orig_re/orig_im`)
are allocated once per context for the max `N·K` and reused. **Thermal pacing**: for big-N
low-K cells (`N·K ≥ PACE_TOTAL_THRESHOLD`, or `K ≤ PACE_K_THRESHOLD`), sleep `PACE_MS = 200`
every `PACE_EVERY = 25` benches to keep the package thermal envelope stable.

**Known bias:** first-stage-first recursion + sub-cache pruning makes DP lean *against*
wide-radix-innermost decompositions (it commits to a first stage before seeing the tail).
That's exactly the gap the exhaustive/screened paths and the V4 model are designed to cover.

---

## MEASURE (`measure.h`) — the variant-aware two-pass

This is the **wrapper the DP port deliberately skipped** ("MEASURE wrapper skipped —
separate variant-cartesian workstream"). Without it, dag's calibrator searched only the
factorization axis and emitted all-T1S/DIT wisdom that lost to production's variant+DIF
plans *with identical codelets*. `measure.h` closes that gap.

**Two passes:**
- **COARSE** — enumerate `(factorization × permutation)`; bench with default **T1S/DIT**
  (plus a **LOG3-forced probe** so LOG3-friendly multisets aren't eliminated before
  refine); sort; take `MEASURE_TOPK = 5`. (For huge cells `> MEASURE_EXH_THRESHOLD =
  16384`, the coarse frame comes from DP's top-K multisets rather than full enumeration.)
- **REFINE** — on each coarse survivor, run the **variant cartesian** `(FLAT/LOG3/T1S)^(nf-1)`
  × `{DIT, DIF}`, tracking the global best **and** a deploy pool of everything within
  `DEPLOY_PCT = 10%` of the refine-best (capped at `DEPLOY_MAX = 5`) for the caller to
  deploy-rebench.

`vfft_proto_variant_available()` reads the **generated** registry slots
(`registry_{avx2,avx512}.h`, emitted by the OCaml pipeline) so REFINE only tries variants
that actually have a codelet. The output is a `vfft_proto_plan_decision_t`
`{factors, variants, use_dif_forward, ns}` — exactly what a wisdom entry stores.

---

## Exhaustive (`exhaustive_plan.h`)

Brute-force the factorization axis, benching at the **real parent (N,K) context** (which
is what captures the full plan's cache/TLB interaction — the signal isolated sub-plan
measurement loses):

- **flat** (`exhaustive_plan.h`) — bench **every** `(multiset × permutation)`; 3 warmups,
  best-of-3, a 1.5× quick **pre-screen** to drop obvious losers. ~500–1500 benches/cell.
- The **patient** and **screened** variants (no-pre-screen thermal-paced, and V4-cost-model
  pre-ranked) were designed alongside flat but never wired; removed 2026-09-01.

---

---

## Wisdom (`wisdom_reader.h`) — the persistence format

The in-memory table + the on-disk format that closes the calibrate loop (search a cell →
fill an entry → `set()` → … → `save()`; `load()` round-trips with `save()`).

**v5 line format:**
```
N K nf  f0..f(nf-1)  best_ns  use_blocked split_stage block_groups  use_dif_forward  v0..v(nf-1)
```
Variant codes `0=FLAT 1=LOG3 2=T1S 3=BUF`. `use_blocked/split_stage/block_groups` are
legacy blocked-executor fields (carried for format compatibility). `load`/`lookup` consume
wisdom; `set`/`add`/`save` produce it. Each transform family keeps its **own** wisdom file
(c2c `spike_wisdom`, `rfft_wisdom`, `c2r_wisdom`, `oop_wisdom`, Bluestein) — different
optima, same format machinery.

---

---

## Config values (defaults)

| macro | value | meaning |
|-------|:-----:|---------|
| `VFFT_PROTO_DP_CACHE_MAX` | 512 | memoization rows (keyed by N, K_eff) |
| `VFFT_PROTO_DP_TOPK_MAX` | 8 | plans stored per cache row |
| `VFFT_PROTO_DP_BEAM_MEASURE / _PATIENT` | 3 / 8 | sub-plans propagated per node |
| `VFFT_PROTO_DP_TIME_REPEAT` / `_MIN_NS` / `_LIMIT_NS` | 6 / 2 ms / 0.5 s | bench best-of / min-trial / cap |
| `VFFT_PROTO_DP_PACE_MS` / `_EVERY` | 200 ms / 25 | thermal pacing (big-N low-K) |
| `VFFT_PROTO_MEASURE_TOPK` | 5 | coarse survivors sent to variant refine |
| `VFFT_PROTO_MEASURE_DEPLOY_PCT` / `_MAX` | 10% / 5 | deploy-rebench pool window / cap |
| `VFFT_PROTO_MEASURE_EXH_THRESHOLD` | 16384 | above this, coarse frame = DP top-K, not full enum |

---

## Files

| file | role |
|------|------|
| `dp_planner.h` | recursive measured DP + memoization (the search core) |
| `measure.h` | variant-aware (FLAT/LOG3/T1S × DIT/DIF) two-pass on top of the coarse search |
| `exhaustive_plan.h` | flat exhaustive (every multiset × permutation, parent-context bench) |
| `wisdom_reader.h` | wisdom file format + load/lookup/set/save (closes the calibrate loop) |
| `cascade_calibrate.h` | t2q calibrators — the cascade terminator choice, measured per cell (placement luck is ±5%, so it is never hand-set) |
| `pad_calibrate.h` | pad-vs-tail calibration and the `_VFFT_PADVW` width stamp |

## Gotchas

- **Every strategy here measures → calibration-time only.** Runtime is a wisdom
  lookup; never measure on the deploy host's hot path.
- **Cost-model constants are Raptor-Lake / AVX2-specific** (bandwidth tiers @5.7 GHz, DTLB,
  the measured `radix_memboundness` table). Re-probe on other hosts.
- **DP/exhaustive are factorization-only; MEASURE adds variants+DIF.** Calibrate through
  MEASURE for production wisdom, or you ship all-T1S plans that leave wins on the table.

See also: `core/engine/README.md` (what consumes the chosen plan + the twiddle variants),
`docs/cost_model/` (the V4 derivation + variant-selection model).
