# CP Exact Cluster Scheduler — Implementation Plan (OCaml)

> Plan for replacing `bb.ml`'s hand-rolled branch-and-bound with a real
> constraint-programming exact scheduler, run **per cluster** via the
> decomposition the compiler already has. Strategy context:
> [`scheduling_strategy.md`](scheduling_strategy.md). This doc is about *how* to
> build it in `.ml`, and — more importantly — the **pitfalls, details, and
> scope** before any code.

---

## 0. The one-paragraph shape

The scheduler stays a DAG pass in the generator. We add a module
`Cp_schedule` whose entry point has the **same signature as
`Schedule.su_schedule_subset`** (so it drops into `emit_c`'s dispatch), but
internally: build the cluster's precedence graph + value lifetimes, **encode it
as a constraint model**, hand it to a CP solver (with a time budget and an `su`
warm-start), and parse the returned order back into `Algsimp.t list`. It runs at
**build time, per cluster**, so it never sees a whole R=256 codelet — only the
small subsets the recipe cut already produces. Every output is a legal
topological reorder → **bit-exact by construction**; the measured-asm harness is
the referee.

---

## A. Decomposition — the part we already own

The exact solve is per **cluster**, not per codelet, and the compiler already
produces the clusters:

- `Schedule.su_schedule_subset` is called **once per pass/group**
  (R=32 → 12 subsets; verified). `bb.ml` is already *cluster-local*. So the CP
  inherits the exact same unit of work and the exact same **contract**:
  > schedule a `subset` of nodes; treat predecessors *outside* the subset as
  > already-available, and the `sinks` as values that must stay live to the end.
- **The cut granularity is the size dial.** The same recipe cut that bounds
  per-pass register pressure also bounds the per-cluster node count → the
  solver's tractability. R=256's "is it solvable" question collapses to "how big
  is the biggest cluster," which the cut controls.
- `Schedule.connected_components_of` splits a subset into independent sub-DFTs →
  even smaller, fully-independent CP instances.

**Implication for scope:** we do **not** need to solve large codelets. We need a
solver that's optimal-or-near on clusters of ~tens of nodes, and a fallback
(`su`) for any cluster that's still too big. With a fine enough cut, that
fallback almost never fires.

---

## B. The two build paths (the key scoping decision)

### Path 1 — OCaml *encoder* + external CP-SAT (recommended start)

The `.ml` builds the model; a real solver does the search.

- **OCaml side:** `Cp_schedule` walks the cluster DAG and **emits a model** — a
  MiniZinc model + data, or an OR-Tools CP-SAT protobuf. Then `Unix`-subprocess
  to `minizinc --solver cp-sat` (or the CP-SAT runner), and parse the solution
  back to a node order.
- **Why this first:** CP-SAT's propagators (precedence, `cumulative`, LP bounds,
  no-good learning) are decades of engineering we will *not* match by hand. We
  get a real solver in days, and can **measure the gain** before deciding whether
  a custom engine is even worth it.
- **Pitfalls:** a build-time dependency (the `minizinc`/CP-SAT binary must be
  present — same kind of dependency as gcc); the subprocess boundary (model
  serialization, parsing); and **determinism** (see §D-5).

### Path 2 — custom OCaml CP/B&B engine (evolve `bb.ml`)

Self-contained: extend `bb.ml`'s DFS with real **propagators** — precedence
(domain pruning on positions), a **register-pressure propagator** (prune when the
partial schedule's live-set already forces > R), critical-path/saturation lower
bounds, and symmetry breaking.

- **Why maybe:** no external dependency, fully deterministic, fully in your OCaml,
  tailored to FFT structure.
- **Pitfalls:** this is a *research-grade* effort. A naive custom CP barely beats
  `bb.ml`; matching CP-SAT's `cumulative` + learning is months. High risk of
  reinventing a solver badly.

**Recommendation:** **Path 1 to learn the ceiling**, keep `bb.ml`/`su` as
fallback. Only invest in Path 2 if (a) the external dependency is unacceptable for
your build/distribution, and (b) Path 1 proved the gain is real and worth a custom
engine. Don't write a solver to find out if you need a solver.

---

## C. The model (what the `.ml` encodes)

For one cluster, with `R = uarch.vec_regs`:

- **Decision variables:** `pos[i] ∈ [0, n-1]` — the emission position of node `i`
  (a permutation; or `cycle[i]` on a modeled machine if we later add latency).
- **Precedence:** for every DAG edge `i → j` with both in the subset,
  `pos[i] < pos[j]`. (From `preds n` filtered to the subset — exactly what the
  injector already dumps.)
- **Register pressure (the crux):** value `v` is **live** from its definition
  position to the max position of its in-subset users (and to the *end* if `v` is
  a sink / crosses the cluster boundary). The count of simultaneously-live values
  at every point must be `≤ R`. This is a **`cumulative`-style** constraint over
  *variable* intervals — the integrated schedule+pressure model, and the hard part
  of the encoding.
- **Objective:** start with **minimize makespan** or **minimize peak-live**
  (see §D-6 for why this is fraught). Later: minimize modeled **cycles** (ports +
  latency) for the throughput dimension.
- **Warm-start:** feed `su_schedule_subset`'s order as the initial solution / hint
  (incumbent → massive pruning).

### Proposed OCaml interface

```ocaml
(* Cp_schedule.ml — drop-in alternative to su_schedule_subset. *)
val cp_schedule_subset :
  Uarch.t ->
  inline_set:(int, unit) Hashtbl.t ->   (* values emit_c inlines = NOT registers *)
  subset:Algsimp.t list ->
  sinks:Algsimp.t list ->
  budget_sec:float ->                   (* anytime cap; falls back to su on timeout *)
  Algsimp.t list                        (* the scheduled subset, legal topo order *)
```

It mirrors `su_schedule_subset` plus two inputs the pressure model needs
(`inline_set`, `budget_sec`). `emit_c`'s dispatch gets a new `CP` scheduler
variant alongside `SU`/`Bisection`; a `--cp` flag selects it.

---

## D. Pitfalls & details (the part that matters)

**1. The model→asm gap (THE pitfall — same one that sank GH and bb).**
The CP finds the *model* optimum; **gcc still does the final register
allocation**, and we measured that the IR's `peak_live` optimum ≠ gcc's realized
spills (`bb` minimized `peak_live` and got *2× the asm spills* at R=64). A
provably-optimal-for-the-model schedule can still lose in the asm. **Mitigations:**
(a) keep the objective as close to the real metric as feasible (modeled cycles >
peak-live > makespan as proxies for gcc behavior); (b) **always validate CP output
through the measured-asm harness** — the CP *proposes*, the harness *disposes*;
(c) treat "provably optimal" as "provably optimal for our model," not "fastest
asm." This single point is why the harness is not optional.

**2. Hard `≤ R` can be infeasible.** A cluster's register saturation may exceed
`R` (it genuinely needs spills — that's *why* the spill recipe exists). A hard
"pressure ≤ R" constraint then has **no solution**. Options: **minimize spill
count** as a soft objective (count values forced above R), or **Touati-pre-bound**
the saturation ≤ R by adding serialization arcs *before* the CP. Start with
*minimize peak-live* (always feasible) and read off whether it's ≤ R.

**3. Inlining mismatch (subtle, will silently corrupt the pressure model).**
`emit_c` **inlines single-use intermediates** — they never occupy a register. If
the CP counts every node as a live value, it **over-counts pressure** and
optimizes a fiction. The model must exclude `inline_set` values from the
`cumulative`. This is why `inline_set` is in the signature. (Cross-check: the
pressure the CP computes for the `su` order must equal what `Regalloc.peak_live_*`
reports, or the model is wrong.)

**4. Cross-cluster (spill-slot) values.** Inputs that come from a prior pass are
"available at position 0"; outputs that feed the next pass (`sinks`) are "live to
the end." Their lifetime contribution to the `cumulative` is fixed by the seam,
not chosen by the CP. Model them as boundary intervals; don't let the CP try to
"reschedule" them.

**5. Determinism (build-reproducibility).** CP-SAT with multiple workers is
**non-deterministic** — different runs return different optimal-cost schedules,
which would break your bit-reproducible codelets. Force `num_search_workers=1` +
fixed `random_seed`, and on ties prefer the lowest-tag order (as `bb.ml`/`su`
already do) so the *chosen* optimum among equal-cost ones is deterministic.

**6. Which objective? (genuinely unsettled — decide deliberately.)**
- *makespan* (schedule length): easy, but only loosely tied to spills/runtime.
- *peak-live*: the natural pressure proxy, but `peak_live ≠ asm spills` (pitfall 1).
- *spill count*: closer, but needs an allocation model in the constraint system.
- *modeled cycles* (ports + latency + pressure): closest to runtime, hardest to
  model, and the unification we actually want.
Plan: **start with peak-live/makespan to stand the pipeline up, validate with the
harness, then move the objective toward modeled cycles** (or `uiCA`-in-the-loop)
once the plumbing works. Don't block the first artifact on the perfect objective.

**7. Build-time cost.** One solve per cluster × many clusters × ~1000 codelets.
A per-cluster **time budget** (anytime: best-found + optimality gap) bounds it;
**cache** solved orders (a wisdom file keyed by cluster hash — the injector
already keys by `subset_key`) so the expensive solve runs once per codelet, not
per build.

**8. Symmetry — power *and* trap.** FFT DAGs have heavy symmetry (identical
butterflies, interchangeable lanes); symmetry-breaking constraints collapse the
search and are a *big* FFT-specific win — but a wrong symmetry constraint can
**cut off the optimum**. Defer to a later phase; it's an accelerator, not needed
for correctness or the first result.

**9. Correctness is free — *if* precedence is right.** A CP output is a reorder of
the same nodes, so it's **bit-exact by construction** *provided* the precedence
constraints exactly match the DAG (`preds`). Validate with the same round-trip the
injector uses (inject CP order → diff vs su body must compile + match), then the
harness for the win check. The one correctness risk is a missing precedence edge →
use-before-def → compile error (caught) — so model precedence from the *same*
`preds` the emitter uses.

---

## E. Scope

**In scope (first artifact):**
- `Cp_schedule.cp_schedule_subset` (Path 1): cluster DAG → MiniZinc/CP-SAT model
  (precedence + `cumulative` pressure with `inline_set`), `su` warm-start,
  single-thread deterministic, time-budgeted, parse order back.
- A `--cp` scheduler variant in `emit_c` dispatch (parallels `--bb`).
- Objective = minimize peak-live (always feasible) or makespan.
- Validation: round-trip bit-exactness + the measured-asm harness on R=13 and one
  R=32 cluster.

**Out of scope (later phases):**
- Port/resource (`cumulative` over ports) and the modeled-**cycles** objective.
- Symmetry-breaking constraints.
- Large-Neighborhood Search for oversized clusters.
- Touati saturation pre-bounding (its own workstream — see strategy doc).
- Wisdom caching of solved orders (do once the per-cluster gain is proven).
- A custom OCaml CP engine (Path 2) — only if Path 1 proves the gain *and* the
  external dependency is unacceptable.

---

## F. Build order (concrete first steps)

1. **Stand up the encoder on the smallest real cluster.** Pick one R=13 cluster
   (it's monolithic — one subset — so the simplest), encode precedence + pressure
   in MiniZinc, solve, parse, inject via `VFFT_SCHED_ORDER`, confirm round-trip
   bit-exact. *Goal: prove the pipeline, not the win.*
2. **Validate the model.** Confirm the CP's computed peak-live for the `su` order
   equals `Regalloc.peak_live_*` — catches the inlining/lifetime bugs (pitfall 3/4).
3. **Solve + harness.** Solve to optimal peak-live, inject, run the harness vs
   `su`. Does the *model* optimum reduce *gcc's* asm spills/instructions? This is
   the moment of truth for pitfall 1.
4. **Per-cluster on a blocked codelet.** Loop the solve over R=32's 12 subsets,
   stitch, harness vs `su`. Confirms the decomposition path end-to-end.
5. **Decide.** If the harness shows real, transferable wins → add anytime budgets,
   caching, then move the objective toward cycles. If not → we learned the model
   optimum doesn't survive gcc (a real result), and the lever is the objective
   (cycles/uiCA), not the solver.

The whole plan rests on one honest expectation carried over from the experiments:
**a better solver doesn't escape the model→asm gap — it just finds the model's
optimum faster.** So step 3 (does it survive gcc?) gates everything after it.
