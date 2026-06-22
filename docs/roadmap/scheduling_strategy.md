# Codelet Scheduling — Methods Pruned & The Strategy

> What we explored for the codelet scheduler, what we ruled out (with evidence),
> and the path forward. Companion to the experimental record in
> [`docs/performance/schedule_search_phase0_results.md`](../performance/schedule_search_phase0_results.md).

## Frame

The scheduler is a **pass inside our compiler** — OCaml in
[`generator/lib/schedule.ml`](../../src/dag-fft-compiler/generator/lib/schedule.ml)
that walks the hash-consed `Algsimp.t` DAG and emits the operation order that
becomes a codelet. It is **not** a gcc/LLVM flag. Two facts shape everything:

1. **gcc does the final register *allocation* downstream.** Whatever order our
   scheduler emits is a *hint* gcc partially follows, then re-allocates. So no
   scheduler here directly controls spills — it shapes gcc's input.
2. **All scheduling is offline** — codelets are generated, scheduled, and
   compiled at *build time*, once each, then shipped. There is no runtime
   scheduling. So the axis is never online-vs-offline; it's **fast-heuristic vs
   slow-optimal**, and we can afford a slow optimal solve per codelet *if it
   scales*.

Binding constraint (memory-bound thesis): **register pressure → spills** and
**port throughput**. The OoO engine hides most *latency* for window-sized blocks.

## The central tension

Two forces pull opposite directions, and a real scheduler must balance both:

- **Register-pressure minimization** → *serialize* (fewer values live at once).
- **ILP / latency-hiding** → *parallelize* (keep many chains in flight so the
  OoO machine overlaps them).

Serializing to bound pressure can starve the OoO engine and *expose* latency.
SU/GH/Touati treat pressure as the **objective** (one side). The unification is
to flip it: **pressure becomes a CONSTRAINT (≤ vec_regs); throughput/cycles
becomes the OBJECTIVE.** Then one optimization balances both — over-serialize and
you pay in cycles, over-parallelize and you violate the constraint.

---

## Methods explored and PRUNED (with evidence)

| Method | What it is | Verdict | Evidence |
|---|---|---|---|
| **Sethi–Ullman (SU)** | 1974 register-only list heuristic (priority = `cp_dist`, `su_num`) | **Kept as baseline** — beats the alternatives below on this IR, but single-resource and 1970s | wins vs GH/BB/bisect everywhere measured |
| **Goodman–Hsu (GH)** | pressure-mode switch on top of SU | **Pruned as a lever** | lowering the threshold 12→4 left asm spills *flat* (R=32 138–142, R=64 457–474) and runtime flat — because `peak_live` ≠ realized spills |
| **Branch-and-bound (`bb.ml`)** | hand-rolled B&B minimizing `saturated_peak` | **Pruned** | loses to SU; **2× the asm spills at R=64** (942 vs 477) despite minimizing `peak_live` — wrong metric; ~50-op wall |
| **Frigo recursive bisection** | cache-oblivious RED/BLUE/YELLOW | **Pruned for this IR** | hash-consed `Const`/twiddle nodes weld the sub-DFTs together → ~2× SU's spills (R16 94 vs 68, R32 270 vs 147) |
| **`peak_live` as the objective** | minimize IR register pressure | **Pruned** | the model→asm gap — gcc re-allocates, so minimizing the IR's peak-live (GH *and* BB) does not minimize gcc's realized spills |
| **M-project regalloc** (`regalloc.ml`, pins + scheduling fence) | take allocation from gcc | **Dormant by design** | post-FMA-fusion it's *harmful*: the fence defeats operand folding (+9%), residual reg-reg `vmov` are rename-eliminated (free); net-negative/tie in every cell |
| **Finer cut topology** (3+ pass internal blocking) | recursively block the codelet | **Pruned** | already handled by the executor's K-adaptive **plans** (N=64 → `[4,4,4]` at high K); building it into codelet blocking duplicates the plan layer |
| **Schedule search** (simulated annealing / iterative compilation) | search legal orders, score on measured asm | **Works, but marginal** | beats SU on *primes* (R=13 −19 insns / −7 spills) but **~no pow2 headroom** (R=32 found 0 spill reduction — SU already optimal there). Validated the method; headroom is in under-served codelets, not pow2 |

### What the experiments taught us (kept as principles)

- **asm spills predict runtime only *coarsely*** (Spearman ρ≈0.94 across
  structurally-different schedulers with 2–20× spill spreads) — **not at fine
  grain** (small reorderings of one schedule).
- **Runtime can't arbitrate fine-grain wins on this host** (frequency/contention
  noise swamps ~2% deltas). → use a **static win condition**.
- **The noise-robust objective:** *reduce spills without increasing total
  instructions or saturating a port.* Implemented as: **minimize total
  instructions** (subsumes spills + reg-reg moves), **gated** on spills-not-up +
  **FMA-count invariant** (the port guard — reordering can't change the
  arithmetic mix).
- **Reg-reg moves are rename-eliminated (free)** on Raptor Lake → for *pressure*,
  the real metric is **spills**; for *ports/latency*, you need a **throughput
  model**, not an instruction count.

---

## The method landscape (families)

- **1970s single-resource heuristics** (SU, GH) — register objective, reactive.
- **Cache-oblivious recursive bisection** (Frigo) — locality; loses on our IR.
- **The unified methods** (pressure = constraint, throughput = objective):
  - **Exact ILP/CP resource-constrained scheduling** (van Beek–Wilken) — one
    model with precedence (latency), port/resource capacity, *and* register
    pressure; minimize cycles. Optimal. The real floor.
  - **Touati SIRA** (register saturation + serialization arcs chosen to minimize
    critical-path increase) — *guarantees* pressure ≤ regs at minimal ILP cost.
    A decoupled balance of both ends.
  - **Unified multi-resource list scheduler** (the LLVM MachineScheduler
    *algorithm*, implemented in our generator) — one greedy priority combining
    pressure-delta + latency + port pressure.
  - **Swing modulo scheduling** — unifies throughput + pressure for *loops*
    (applies to the K-batch loop, not the codelet basic block).

---

## The STRATEGY

The scheduler is a DAG pass that reasons about **register pressure + ILP + ports**
and emits the order, at build time. Three implementable DAG algorithms, **size-
gated**:

**1. Unified multi-objective DAG list scheduler — evolve `su_schedule`.**
Same skeleton (topological list scheduling, ready-set, pick-one), richer priority
computed live as we walk the DAG:
- *register-pressure delta* (`births − kills`) from a live-set tracker (GH half
  does this) — prefer pressure-reducers near the budget;
- *latency / ILP* (`cp_dist`) — already present;
- *port pressure* — running per-port counts (p0/p1 FMA, p2/p3 load, p5 shuffle)
  from a per-node port vector added to `uarch.ml`.
Production-fast; subsumes SU+GH as a special case. **Smallest diff, biggest
conceptual upgrade over 1974 SU.**

**2. Touati SIRA — the pressure-guaranteed option.**
DAG saturation analysis (max-live over all schedules) → insert serialization
DAG edges to bound it ≤ vec_regs while minimizing critical-path increase → list
schedule the augmented DAG. Spill-free *by construction*, ILP cost minimized.

**3. ILP/CP exact (CP-SAT) — optimal where it scales.**
Encode the DAG schedule as a constraint model (variable per node position;
precedence from DAG edges, latency-weighted; per-cycle **port capacity**;
**#live ≤ vec_regs**; objective = minimize cycles) and solve from the generator.
One artifact that is *both* the unified scheduler *and* the provable floor.

### Scaling the exact solve to big codelets (R=128/256) — via decomposition we already have

The solver never sees the whole codelet; it solves **per cluster**.

- **2-pass blocking + cluster-sequential emission *is* the decomposition.**
  `bb.ml` already runs cluster-local. R=256 is many small cluster problems, not
  one 5000-node problem.
- **Cut granularity is the dial** — it bounds register pressure *and* solver
  problem size simultaneously (one lever, two jobs). Finer cut → smaller, solver-
  tractable clusters.
- **Connected-components** (`connected_components_of`) — independent sub-DFTs are
  separate, smaller instances.
- **Solver engineering:** strong CP propagation (cumulative/precedence — why CP
  scales to hundreds of ops where `bb.ml` dies at ~50); SU/SIRA **warm-start**
  incumbent; **register-saturation lower bound** (Touati) for pruning;
  **symmetry-breaking** (FFT butterflies are highly symmetric); **anytime /
  time-budget** solving (returns best-found + optimality gap → deterministic
  build time).
- **Large-Neighborhood Search** (exact solver on a sliding window) + LP-relax-and-
  round for any residual oversized cluster.

### The objective: throughput, with pressure as a constraint

Use a **Raptor-Lake throughput model — `uiCA` (or `llvm-mca`)** as the cost, or
model ports+latency directly in the CP. It captures what SU's register-only view
and our spill-count both miss — **port pressure + the critical-path latency the
OoO can't hide** — and it's **deterministic**, sidestepping the noisy-host
problem. Pressure stays a hard constraint (Touati / CP / list-scheduler bound).

### gcc is downstream → the harness is the referee

Every method emits an *order*; gcc re-allocates. So the **measured-asm harness we
built** stays essential — it's the only piece that confirms which method's *model*
actually survived to silicon. Components (in the gitignored experiment sandbox):
the `VFFT_SCHED_DUMP`/`VFFT_SCHED_ORDER` injector (monolithic + blocked), the
paired-in-process `compare_codelets` bench, and the `objdump` static counters.

---

## Recommended build order

1. **Evolve `su_schedule` → unified multi-resource list scheduler** (live-set +
   port vector folded into the priority). Real multi-resource DAG scheduler,
   smallest diff, replaces 1974 SU.
2. **CP-SAT per-cluster exact** to replace `bb.ml` — optimal where the cluster is
   small enough, via the decomposition already in the compiler.
3. **Touati SIRA** as the pressure-*guarantee* path and as a CP lower-bound
   source.
4. **`uiCA`/`llvm-mca` throughput objective** + harness validation, to bring the
   ILP/port/latency dimension into the cost (and to test whether pow2's "no
   headroom" was real or a spill-count blind spot).

## Honest expectation

pow2 is SU's home turf — its 2-pass + GH + spill-recipe machinery already tunes it
(R=32 showed *zero* spill headroom). The schedule search's real headroom is in the
codelets that machinery underserves — **primes, odd composites, large N** (R=13
gave −19 insns / −7 spills; primes bypass the spill recipe). Aim the heavier
methods there, not at pow2.
