# Notes: the split engine's DP planner (dp_planner.h) → what the z-cascade production planner should inherit

*2026-07-24. Study of `src/core/planning/dp_planner.h` (822 lines, "vfft_proto_dp_planner"),
requested as the reference for productionizing the z-cascade chain search. The spike-level
planner (`build_tuned/benches/zil_chain_dp.c`) is exhaustive full-plan measurement — correct
and affordable at today's scale (13–175 chains × ≤10 variants, 4 cells, K=1 pow-2). This doc
records what the advanced planner does beyond that, why each mechanism exists (each one is a
scar from a measured failure), and which pieces the z planner should adopt at which scale.*

## 1. Architecture in one paragraph

FFTW-style **recursive decomposition with memoization**: to plan N, try each registered radix
R as the first stage, recursively obtain the best plans for N/R, assemble `[R, sub…]`,
**benchmark the FULL assembled plan** through the production `plan_create → execute` path, and
cache winners. Complexity ~O(S·|R|·beam) benches (S = #unique sub-sizes ≈ log N) instead of
exhaustive — ~150 benches for N=100000 vs ~61000. Crucially: **DP prunes the search; it never
composes costs** — every reported cost is a whole-plan measurement (the cost-model-ceiling
doctrine is enforced structurally).

## 2. The six load-bearing mechanisms (each fixes a measured failure)

1. **(Upgrade A) Cache key = (N, K_eff), not N.** K_eff = K_outer × product(prefix radices
   already consumed). A sub-plan's best factorization depends on the batch context it will
   execute in; keying by N alone caused the v1.1 "lock-in" failure (sub-winner best in
   isolation, suboptimal as substage). **z mapping**: the analog context isn't batch (K=1) but
   *stage geometry* — a sub-chain's cost depends on its stride/depth context; if the z planner
   ever goes recursive it must key on (M, D-context), or stay whole-chain like today.
2. **(Upgrade D) Top-K beam per node + propagation of runners-up.** Each (N,K_eff) row keeps
   up to 8 plans; the recursion hands runners-up to outer levels, because a factorization that
   loses in isolation can WIN composed under a different outer radix (the N=32768 K=4
   regression: [4,32,64] lost standalone, [4,4,32,64] won overall). Beam width is a knob:
   MEASURE=3 (fast), PATIENT=8 (wide).
3. **(PATIENT dedup) Beam diversity by MULTISET.** Keep the cheapest *ordering* of each
   distinct factor-set in the beam — otherwise the beam fills with re-orderings of one set and
   misses different sets entirely (the 4096 K=32 lesson: beam=8 collapsed to 2 multisets and
   missed 4×4×4×64).
4. **(Phase 2) Ordering search at the top.** The recursion only emits radix-first orderings;
   the final phase permutes every unique retained multiset and benches ALL orderings. **z
   mapping**: our grid-preserving cascade is strongly ordering-sensitive (4.8.16.8 ≠ 16.8.4.8
   — measured), so any recursive z planner MUST keep this phase; today's exhaustive
   enumeration covers orderings by construction.
5. **(Upgrade C) believe_subplan_cost = FFTW BELIEVE_PCOST.** MEASURE trusts cached costs on
   hit; PATIENT re-measures every cached top-K (best-of-2) on every encounter and re-sorts, so
   a noise-mis-ranked runner-up can climb back. Variance is re-absorbed instead of frozen in.
6. **(Pacing) Intra-search thermal pacing.** Sleep 200 ms every 25 benches when K>64 OR
   N·K ≥ 32768. Sustained pinned-core load heat-soaks the package enough to **drift the
   ranking itself** (verified 2026-06-16: a 700-candidate 16384 sweep mis-ranked without it;
   ~33% wall overhead at K=4 is the accepted price for a steady clock).

Plus the timing harness (Upgrade B, FFTW `measure_execution_time` mirror): adaptive reps
(double until trial ≥ 2 ms), best-of-6 trials, 0.5 s hard cap per bench, per-trial buffer
reset from a pristine copy; and the buffer invariant N·K_eff = const across recursion → one
allocation serves the whole search.

## 3. How variants are handled — the important structural choice

The DP planner does NOT multiply its search space by kernel variants. It benches through
`vfft_proto_plan_create`, which internally consults **codelet-side plan wisdom** to pick the
per-stage protocol (flat / t1s / DIT-log3) — so variant selection is DELEGATED to plan-time
wisdom, and the DP searches factorization shape only. **z mapping**: today the spike planner
joint-enumerates chain × interior × twiddle × terminator (≤10 variants) because the space is
tiny and the interactions are real (t2c changed the winning CHAIN at 16384; split-interior
eligibility constrains radices). At production scale the split engine's pattern applies:
per-stage variant wisdom resolved inside z-plan-create, DP over chains only — with the caveat
that interior layout (z vs block-split) is chain-COUPLED (eligibility + crossover), so layout
should stay a searched axis, not delegated.

## 4. What the z planner should adopt, by scale

| scale | search | adopt from dp_planner.h |
|---|---|---|
| today: K=1 pow-2, 4 cells, ≤175×10 arms | exhaustive full-plan (zil_chain_dp.c) | **thermal pacing** (the 8192 t2c ±20% run-swing is plausibly heat/layout — pace + re-measure), adaptive-reps timing, PATIENT re-measure of finalists |
| + more cells (2^11..2^17, odd-radix, r2c) | exhaustive per cell still OK (~10³ benches) | + multiset/variant dedup discipline, wisdom emission per cell |
| + K>1 × placement × layout axes | recursive DP required | full inheritance: (N, context) cache, top-K beam + diversity dedup, ordering phase, BELIEVE toggle; z-plan-create with per-stage variant wisdom |

Immediate actionable: add `_maybe_pace`-style pacing + the adaptive-reps harness to
`zil_chain_dp.c` finals, and re-measure the 8192 cell (its variant ranking swung between runs).

## 5. Pointers

- `src/core/planning/dp_planner.h` — this file (wholesale port of the older
  `src/core/dp_planner.h`, mechanical renames; MEASURE-wrapper workstream separate).
- Companions: `exhaustive_plan.h` / `exhaustive_screened.h` / `exhaustive_patient.h` (the
  non-DP searchers), `measure.h`, `estimate_plan.h` (the modeled path — superseded by
  measurement per the cost-model-ceiling lesson).
- z side today: `build_tuned/benches/zil_chain_dp.c` (exhaustive chain×variant),
  z_cascade_plan.md §4.95–4.99 (results).
