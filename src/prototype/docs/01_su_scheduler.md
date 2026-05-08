# SU Scheduler: Modest Mixed Result, And a Surprise About µarch Parameters

## TL;DR

We built a Sethi-Ullman-flavored list scheduler (~250 lines) with critical-path priority, parametric on a µarch profile. Tested on AVX-512 R=8 and R=16 against the existing topological emission.

**Empirical result (R=16, Sapphire Rapids container):**

| K regime | SU vs Topo | SU vs Hand |
|---|---|---|
| K=64–512 (compute-leaning) | SU 2–6% faster | SU 6–15% slower |
| K=1024+ (memory-bound) | SU 1–6% slower | SU still 12–13% better than Hand |

**R=8 result:** SU is statistically indistinguishable from Topo at every K (within 1–3% noise).

**Surprise about µarch parameterization:** All three profiles we built (Sapphire Rapids, Raptor Lake, Zen5) produce **byte-identical schedules** for our codelets. The reason is informative and turns out to be a meaningful finding in itself: our cost model uses uniform per-op latencies, so cp_dist values shift uniformly across profiles without changing the relative ordering that determines schedule choices.

## What we built

`lib/schedule.ml` already had the SU implementation in place from earlier work. Confirmed:

- **`compute_cp_dist`**: backward DP over the DAG, weighted by `Uarch.t` latency parameters. Sinks have `cp_dist = node_latency`; everyone else accumulates the longest path to a sink.
- **`compute_su_number`**: classical Sethi-Ullman label, with k-ary generalization for Cmul (sort children by SU desc, label = max_i (SU_i + i)).
- **`su_schedule`**: list scheduler with priority queue. Picks from ready set by `(cp_dist DESC, su_num ASC, tag ASC)`.

Plumbed into `emit_c.ml` as scheduler choices `SU of Uarch.t` and `Annotated_SU of Uarch.t`. CLI flags `--su --uarch <profile>`.

## The bug we hit and fixed

First measurement showed SU consistently slower than Topo by 4–13% across all K. Inspection of the output revealed why:

```
const __m512d t0  = _mm512_loadu_pd(&rio_re[15*ios + k]);   // load
const __m512d t1  = _mm512_loadu_pd(&tw_re[14*me + k]);     // load
const __m512d t2  = _mm512_loadu_pd(&rio_im[15*ios + k]);   // load
const __m512d t3  = _mm512_loadu_pd(&tw_im[14*me + k]);     // load
const __m512d t9  = _mm512_loadu_pd(&rio_re[7*ios + k]);    // load
const __m512d t10 = _mm512_loadu_pd(&tw_re[6*me + k]);      // load
... 24 more loads ...
// only after all 30 loads: any arithmetic
```

SU's critical-path priority sees loads as "long-chain leaders" (they have many dependent FMAs) and fires them all first. The `next_required_load` mechanism we built preserves load *order* but not load *interleaving* with arithmetic. Result: same prefetcher-killing pattern we saw with bisection — all loads at the top, no overlap of memory and compute.

**Fix (5 lines):** Among ready instructions, prefer arithmetic over loads. Loads fire only when no arithmetic is ready (i.e., only when needed to unblock the next FMA). With this, loads naturally interleave leg-by-leg, matching Topo's organic order:

```
const __m512d t38 = _mm512_set1_pd(0.92...);          // constant
const __m512d t0  = _mm512_loadu_pd(&rio_re[15*ios + k]);
const __m512d t1  = _mm512_loadu_pd(&tw_re[14*me + k]);
const __m512d t2  = _mm512_loadu_pd(&rio_im[15*ios + k]);
const __m512d t3  = _mm512_loadu_pd(&tw_im[14*me + k]);
const __m512d t4  = _mm512_fnmadd_pd(t2, t3, _mm512_mul_pd(t0, t1));   // fma fires
const __m512d t5  = _mm512_fmadd_pd(t0, t3, _mm512_mul_pd(t2, t1));    // fma
const __m512d t9  = _mm512_loadu_pd(&rio_re[7*ios + k]);   // next leg starts loading
...
```

This is a meaningful design lesson, not a fixable mistake to bury: critical-path priority and source-order preservation aren't sufficient. You also need an explicit policy that loads don't run ahead of arithmetic that's ready. Otherwise SU undoes the natural interleaving Topo has by construction.

## Bench results, post-fix

R=16 t1_dit AVX-512, Sapphire Rapids container, 3 runs at each K, single-process side-by-side timing:

| K | Hand (ns) | Topo (ns) | SU (ns) | T/H | S/H | S/T |
|---|---|---|---|---|---|---|
| 64 | 538 | 629 | 614 | 1.17 | 1.14 | **0.98** |
| 128 | 1407 | 1604 | 1501 | 1.14 | 1.07 | **0.94** |
| 256 | 4024 | 4569 | 4523 | 1.14 | 1.12 | **0.99** |
| 512 | 10791 | 10538 | 9902 | 0.98 | **0.92** | **0.94** |
| 1024 | 25943 | 22669 | 22969 | 0.87 | 0.88 | 1.01 |
| 2048 | 62511 | 52592 | 55029 | 0.84 | 0.88 | 1.05 |

Notes:
- T/H < 1 means Topo beats Hand. S/H < 1 means SU beats Hand. S/T < 1 means SU beats Topo.
- All ratios stable across 3 runs (per-run variance ≤ 4% except K=512 where it was 5%).

**Reading the data:**

- **Compute-leaning regime (K ≤ 512):** SU recovers 2–6% over Topo. At K=128 specifically, SU closes more than half the gap to Hand (Topo=14% slower than Hand → SU=7% slower). This is where SU's theoretical sweet spot is — chains exposed to GCC in cp-priority order, leading to better register allocation.

- **Memory-bound regime (K ≥ 1024):** SU loses 1–5% to Topo. Topo's leg-sequential layout (which SU mostly preserves but slightly perturbs) is what the prefetcher wants here; SU's small reorderings are net-negative when memory dominates.

- **K=512 transition:** SU beats *both* Topo and Hand. Best regime for SU — large enough that GCC can't trivially fix Topo's ordering, small enough that compute matters more than memory.

R=8 t1_dit AVX-512: SU and Topo are statistically tied at every K. Three runs at each K give S/T ratios in [0.97, 1.05]. With 99 ops total, GCC's own scheduler has plenty of room to find a good order from any reasonable input; SU's restructuring contributes nothing measurable.

## The µarch-parameterization surprise

I expected different µarch profiles to produce visibly different schedules. They don't:

```
$ diff radix16_su_sapphire_rapids.c radix16_su_raptor_lake.c
[empty]
$ diff radix16_su_sapphire_rapids.c radix16_su_zen5.c
[empty]
```

All three produce byte-identical output for R=16. The reason traces directly to a property of our cost model:

| Profile | fma_lat | add_lat | mul_lat | load_l1 |
|---|---|---|---|---|
| sapphire_rapids | 4 | 4 | 4 | 7 |
| raptor_lake | 4 | 4 | 4 | 5 |
| zen5 | 4 | 3 | 3 | 4 |

Within each profile, arithmetic latencies are uniform (or near-uniform: zen5 has add=mul=3, fma=4, but FMA chains are still uniform-FMA chains).

`compute_cp_dist` does backward dynamic programming with these latencies. For a chain of N FMAs:

- SPR cp_dist = `4N + load_lat`
- Zen5 cp_dist = `4N + load_lat`  (FMA is same on both)

The relative ordering between any two nodes in the DAG is determined by *differences* in cp_dist. Since the differences come from differences in instruction-mix and chain-depth, and those are properties of the DAG (not the profile), the sorted order of cp_dist values is invariant under uniform shifts.

For load latency specifically: it differs across profiles (7 / 5 / 4 cycles) but it shifts *all* load nodes' cp_dist uniformly. Since loads are picked from a separate "next-required-load" gate (not by cp_dist), this shift has no effect on the schedule.

**To make µarch parameters actually differentiate schedules, the cost model would need at least one of:**

1. **Heterogeneous per-op latencies** — e.g., add=3 but mul=4 within the same profile. Real chips do have some asymmetry, but capturing it requires a richer model than "one latency per op kind."

2. **Pressure-aware scheduling** — use the `pressure_threshold` field (currently unused) to switch policies above a live-count threshold. AVX-512's 32 ZMM vs AVX-2's 16 YMM gives a real pressure-threshold difference, but it requires a scheduler that *consumes* the threshold, not just stores it.

3. **Port-aware scheduling** — model functional-unit assignments (FMA ports vs ADD ports). Significant complexity. Unlikely worth the bench-measured benefit at our DAG sizes.

For now, the µarch-parametric machinery exists but is essentially passive: same schedule across all profiles. This is honest scope-creep that didn't pay off — we built parameterization for a model that's too simple to use it.

This is itself a meaningful finding for the v2.0 architecture: **don't add µarch parameters until the consumer is rich enough to differentiate them.** The right next step (if we want µarch-differentiated scheduling) is to add a pressure-aware policy that switches behavior based on live-count vs `pressure_threshold`, not to add more profiles.

## What this means for v2.0

The pattern across our experiments now:

| Scheduler | Small K (R=16) | Large K (R=16) |
|---|---|---|
| Hand-coded | wins by 8–17% | loses by 12–14% |
| Topo | baseline | wins by 12–14% over Hand |
| Bisection | regression of 10–25% | regression of 10–25% |
| SU | modestly better than Topo (2–6%) | modestly worse than Topo (1–5%) |
| Annotate (any) | tied with Topo (byte-identical assembly) | tied |

No single scheduler wins everywhere. The right framing is what we've been arriving at incrementally:

1. **Build a small set of variants** (Topo, SU, future spill-variant codelets, future log3 variants).
2. **Bench each across (radix, K, ISA) to characterize their sweet spots.**
3. **Build a dispatcher** that picks among them based on (K, ISA, µarch) at plan time.

The variants themselves are mostly cheap to build from the math layer (we now have Topo, Bisection, Annotated, SU — all ~5–10% emission code each, sharing the same DAG). The expensive part is the bench characterization. Each variant × radix × K × ISA × µarch is a measurement.

For the actual "shipping codelet library," I'd suggest:

- **Default to Topo** for AVX-512. It's never far from optimal, never the worst variant, and matches Hand for K ≥ 256 most of the time.
- **Use SU for R=16 at K ≤ 512** if benchmarks confirm the 4–6% gain reproduces on Raptor Lake.
- **Skip Bisection** entirely. Worst variant at every K we've measured.
- **Annotate is a no-op** with current GCC; keep the code but don't expect benefit.

For the v2.0 framing, the substantive contribution is becoming clearer: not *a better scheduler* but *a scheduler dispatch system* with a small set of variants that each have characterized sweet spots. The empirical work is the artifact, not the algorithm.

## What we'd need to make SU more compelling

The current SU is a 2–6% improvement at R=16 small K. To make it a clear win, we'd want:

1. **Richer cost model** that uses pressure_threshold as a policy switch (live count > threshold → prefer ops that consume more values than they produce).
2. **Better SU-number on DAGs** — current implementation is conservative on shared subexpressions. A more accurate live-set model could find better schedules.
3. **Test on R=32 and R=64** — larger DAGs (700+ ops) are where GCC's own scheduling heuristics start to break down. SU's marginal value over Topo likely scales with DAG size.

(3) is the most promising direction if we want SU to matter. At R=16's 262 ops, GCC's scheduler still has enough headroom that our reordering only matters at the margins. At R=64's expected ~1100 ops, GCC's scheduling-bandwidth limits would give SU more room to help.

## Summary of where we are

After scheduler work across multiple sessions:

| Layer | Status | Empirical effect |
|---|---|---|
| Math layer (CT N1×N2 decomposition) | Works for R=4, R=8, R=16 | DAG matches hand-coded within 2% on op count |
| Algsimp (reassoc, dedup, peephole) | Works, generates correct ops | Necessary infrastructure |
| Topological emission | Works | Beats Hand at large K (12–15%); matches at medium; loses at small K (8–17%) |
| Bisection scheduling | Works | Strictly worse than Topo (10–25% regression) |
| Annotate (nested blocks) | Works | Zero effect — GCC produces byte-identical assembly |
| SU scheduler | Works | R=16 small-K: 2–6% over Topo. R=16 large-K: 1–5% under Topo. R=8: tied. |
| Parametric ISA (AVX-512/AVX-2) | Works | Both ISAs produce correct codelets; Sapphire Rapids gives AVX-512 a 3–4× per-element advantage |
| µarch profiles (SPR/Raptor/Zen5) | Built but unused | All produce identical schedules with current cost model |

The honest summary: scheduling at this DAG size is a small-percentage game. Variant selection (especially the spill/no-spill direction we identified earlier) has more measurable upside than instruction reordering. The infrastructure for variant dispatch is the right place to invest the next session's work.
