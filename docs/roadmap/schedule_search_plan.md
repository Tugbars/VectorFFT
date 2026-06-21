# Schedule-Search Experiment — Plan

> Metaheuristic search over legal codelet schedules, scored by measured cost,
> to find schedules that spill less / run faster than the production SU+GH
> scheduler — built and validated as **isolated sandbox experiments** so wrong
> conclusions can't leak into production.

**Status:** planning. No search code until Phase 0's GO/NO-GO gate passes.

---

## Hypothesis & kill criteria

**Hypothesis:** a search over the legal-schedule space, scored by a *validated*
cost, beats SU+GH on Raptor Lake AVX2 on the large codelets (R≥32) where SU
leaves spills on the table and `bb.ml` exhaustive search times out (~50-op wall).

**We STOP and conclude "no win" if any of these hold (honest negatives are valid
outcomes, like the split-radix / op-count findings):**
- **No headroom:** on the codelets where `bb_search` completes, its *provable*
  optimum already ties SU+GH everywhere → SU+GH is at the floor; nothing to find.
- **Proxy lies and runtime-search finds nothing:** spills don't predict runtime
  (Phase 0), *and* searching directly on measured runtime yields no win above the
  noise floor.
- **No transfer:** wins exist at the tuned K but vanish on held-out K / in the
  real executor (overfit).

---

## What we already have (don't rebuild)

| Asset | Where | Use in this plan |
|---|---|---|
| SU+GH list scheduler | `schedule.ml:su_schedule`/`su_schedule_subset` (`~gh`) | incumbent baseline |
| Frigo bisection | `schedule.ml:bisection_schedule` | alt baseline (loses on this IR) |
| **Exact B&B scheduler** | `bb.ml:bb_search` / `bb_schedule_subset` | **ground-truth oracle on small codelets** |
| **Cost function** | `bb.ml:compute_peak_live` + `compute_progress` (lex: `saturated_peak`, `progress`) | **inner-search objective** |
| Scheduler selection | `gen_main.ml` `--su`/`--bisect`/`--bb`(+`bb_budget`)/`--gh` (line 846, `~gh` at 903) | add `--sched-order` here |
| Timing harness | `generator/cost_model/score_and_time_plans.c` | adapt into the representative bench |
| JIT path (1D C2C) | `jit/emit_jit.py` (+ runtime `dlopen`) | fast per-candidate turnaround |
| Generator runs natively | OCaml/dune in **WSL** | scripted candidate generation |
| Sandbox capability | separate tree + symbol prefix | experimental codelets, isolated from `generated/` + wisdom |

---

## Phase 0 — Measurement substrate (GO/NO-GO gate)

**No search code until this passes.** This is the phase that prevents wrong
conclusions; everything downstream trusts it.

- **0.1 Representative isolated bench.** A standalone harness that runs ONE
  codelet over a **production-shaped K-loop** (split-complex `re[]`/`im[]`
  layout, caller pinned to core 0, `QueryPerformanceCounter`, best-of-N, fixed
  gcc version + flags). Output: median + variance per `(codelet, K)`. *Isolate
  the code, but bench at the production embedding* — a naked codelet bench is the
  trap (high-K behavior is executor/memory-bound, invisible at the codelet level).
- **0.2 Noise floor.** Same binary, repeated runs; characterize per-cell variance
  → the **minimum detectable effect (MDE)**. Any "win" below MDE is noise.
- **0.3 Proxy validation (the decisive study).** For R=16/32/64, generate a
  *spread* of legal schedules — SU, GH, BB-optimal, + 50–100 random-legal — and
  measure **both** `peak_live`/asm-spill-count **and** runtime. Compute
  Spearman(spills, runtime) per K.

**GATE:**
- corr **strong** → inner search may use the model (`compute_peak_live`); cheap.
- corr **weak** → model is a liar; outer search must judge on **measured
  runtime** (or a calibrated multi-term cost). Proceed, but knowing which.
- BB-opt **ties SU+GH everywhere** on solvable codelets → **STOP** (no headroom).

**Exit:** a trustworthy representative bench + a known MDE + a decided objective +
evidence that headroom exists.

---

## Phase 1 — Sandbox experiment infrastructure

- **1.1 Experimental codelet namespace.** Generator flag to emit into a sandbox
  tree with a distinct symbol prefix (e.g. `expsched_n{N}_k{K}_v{NNNN}_…`). Never
  writes `generated/` or any wisdom file.
- **1.2 Schedule-injection mode.** Add `--sched-order <file>` to `gen_main.ml`
  (alongside `--su`/`--bb`): emit the codelet from an explicit node order.
  **Legal-by-construction** — validate the order respects `Algsimp.preds`; reject
  illegal orders. This is the knob the search drives.
- **1.3 Fast turnaround.** Per-candidate path: generator → emit C → gcc → `dlopen`
  in the bench. Optionally extend the JIT (`emit_jit`) to the experimental
  schedule for sub-second candidate compile (currently 1D C2C only).
- **1.4 Correctness harness.** Every candidate validated **bit-exact vs the
  reference (production) codelet** before timing (reuse existing roundtrip/ref
  check). A schedule that changes the result is a bug, not a candidate.

**Exit:** stamp out → correctness-check → bench an arbitrary legal schedule for a
given `(N,K,ISA)` in a scripted loop.

---

## Phase 2 — The two-tier search

- **2.1 Inner search (cheap, in OCaml).** `schedule_anneal` mode: simulated
  annealing over legal orders, scored by `Bb.compute_peak_live`
  (`saturated_peak`, `progress` tie-break — the same cost BB minimizes).
  Operates on the ready-set so every move is correct-by-construction. Emits the
  **top-K distinct** candidates per codelet. **Fixed RNG seed** (reproducible).
- **2.2 BB anchor (correctness of the search itself).** On codelets where
  `bb_search` completes, assert the annealer reaches BB's **optimal cost**. Fail
  → fix the engine, do **not** proceed to large codelets. This is the built-in
  check that the search isn't fooling itself.
- **2.3 Outer validation (expensive).** Each top-K candidate → representative
  bench (Phase 0.1) → runtime distribution → pick the runtime-best that also
  clears Phase 3.

**Exit:** for a given codelet, the search produces a gate-passing candidate ≥
SU+GH, or proves none exists.

---

## Phase 3 — Validation gates (an accepted schedule clears ALL)

| Gate | Check | Guards against |
|---|---|---|
| 3.1 Effect | runtime gain > Phase-0 MDE | noise masquerading as signal |
| 3.2 Significance | paired test across trials; report effect size + CI | best-of-N luck |
| 3.3 Held-out K | tune on train K, win transfers to held-out K (per-K report) | K-overfit (the GH↔BB crossover is real) |
| 3.4 In-executor | win survives in the production K-batched executor | microbench ≠ production |
| 3.5 Attribution | asm spill/move count, chain depth, VTune retiring%; **diff asm** to confirm gcc *reordered* not *reallocated* | artifacts + the gcc-mediation confound |
| 3.6 Regression | ≥ SU+GH on **every** K; tie → keep incumbent | shipping a per-cell loss; churn |

**Exit:** clears all → real win; else → documented honest negative for that codelet.

---

## Phase 4 — Decision & productionization

- **Wins real + transfer:** wire the search as an **offline calibration step**
  that bakes the winning order into wisdom (per codelet/cell), regenerate those
  production codelets, re-validate bit-exact. Search stays **offline** → the
  deterministic, bit-reproducible build is preserved.
- **Wins marginal / cell-specific / absent:** document the negative (with the
  proxy-correlation and headroom data) and keep SU+GH. A clean negative is a
  publishable result and a valid stop.

---

## Risk → mitigation map

| Risk | Mitigation (phase) |
|---|---|
| Optimizing a proxy that doesn't predict runtime | Phase 0.3 correlation gate; fall back to runtime objective |
| Measurement noise → false wins | Phase 0.2 MDE + Phase 3.2 significance |
| Overfit to one K / the microbench | Phase 3.3 held-out K + 3.4 in-executor |
| gcc re-allocation confound | Phase 3.5 asm diff (reorder vs realloc) |
| No headroom (SU+GH already optimal) | Phase 0 BB-tie kill |
| Search itself buggy / non-optimal | Phase 2.2 BB-optimum anchor |
| Breaking deterministic builds | Phase 4 offline-only; fixed seed throughout |
| Contaminating production | Phase 1.1 sandbox namespace; bit-exact gate (1.4) |

---

## First concrete step (zero search code)

**Phase 0.1 + 0.2 + 0.3** on R=16/32/64: build the representative bench, measure
the noise floor, and run the spills-vs-runtime correlation across SU/GH/BB/random
schedules. That single study is the GO/NO-GO: it tells us whether the objective is
"spills" or "runtime," whether headroom exists at all, and what effect size is
even detectable — before a line of annealer is written.
