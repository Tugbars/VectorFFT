# 00 — Thesis

The conceptual foundation behind the VectorFFT cost model. Read this
before the implementation docs (01–07) — every architectural decision
in the rest of the folder traces back to ideas in this one.

## The claim

> **Empirical, per-radix cycle costs combined with executor-mirrored variant
> selection produce ESTIMATE-mode plans within 1.2× of measured plans, in
> sub-millisecond plan time, with no first-run cost to the user.**

This is novel. Most FFT planners either skip cost modeling entirely
(measurement-only — minutes per plan) or use static heuristics that
are wildly inaccurate (FFTW ESTIMATE is commonly 2–5× off the real
optimum). The middle ground we sit in — *measurement-quality estimates
without measurement cost* — is the contribution.

The rest of this doc explains why getting there required throwing
away three things people normally take for granted:

1. The intuition that FFT is compute-bound
2. Static op counts as a cost proxy
3. The independence of cost model and executor

## 1. Memory-bound, not compute-bound

The most important observation underlying the entire architecture:

> **Modern high-end CPUs are memory-bound on FFT, not compute-bound.**

This contradicts most FFT-library folklore, which assumes the dominant
cost is the arithmetic in the butterfly (the multiplies and adds in a
DFT-R kernel). On older hardware that assumption was correct. On
Raptor Lake / Sapphire Rapids / Zen 4 it isn't, and the gap widens
every generation.

### What VTune actually shows at K=256

From `docs/vtune-profiles/` (P-core pinned, performance plan):

| Radix | Retiring % | Bottleneck |
|-------|-----------|------------|
| R=4 | **86%** | Compute-peak — port 0/1 saturated |
| R=8 | **72%** | Dependency chains in DFT-8 critical path |
| R=10–13 | **50–65%** | Winograd FMA chains (compute, but limited) |
| R=16 (post-prefetch) | **25%** | Store-bound + L1 latency |
| R=32 | **34%** | L1 store-DTLB overflow (~80 pages > capacity) |
| R=64 | **27%** | Load + store DTLB overflow (~160 pages) |

Read column 2: **only R=4 retires anywhere near peak**. Everything from
R=8 up is bottlenecked by something other than arithmetic — dependency
chains, store ports, DTLB capacity. R=32 and R=64 spend most of their
cycles waiting on memory subsystem state, not doing FLOPs.

For a full 1D FFT at N=16384, K=4 (1 MB working set), VTune shows
**Memory Bound 33% for VectorFFT vs 50% for MKL** — both are
memory-bound; we're just less so. Even the "compute-peak" R=4 codelet
spends 38% of its uops on memory operations.

### Why this changes how the cost model has to work

If the bottleneck is memory traffic, the cost model needs to track
memory traffic. The op-count approach (`extract.py`) gives us that
*structurally* — load/store ops are counted alongside arithmetic — but
the **per-cycle weight** of a load vs an add is hardware-specific. A
CPU with deeper buffers and bigger DTLB would let the same codelet
run faster.

That's why op counts alone aren't enough. The same codelet has very
different costs on different hardware *despite identical instruction
streams*, and the difference is in how the memory subsystem absorbs
the traffic. We need empirical numbers — from the actual host — to
get cost rankings right.

### The implication for codelet design

A second, perhaps more counter-intuitive consequence: **arithmetic
optimization yields diminishing returns past a certain radix**.
Investing in a faster R=64 butterfly via better register allocation
won't move the needle if 73% of the cycles are spent on DTLB walks.
The biggest wins on R=16+ have come from address-pattern changes
(prefetch hiding), not from instruction-count reductions.

This reframing — *memory-first, compute-second* — is the lens
everything else in the cost model is designed through.

## 2. Why static op counts fail as a cost proxy

A natural instinct is to use the op count from `radix_profile.h` as a
direct cost: `cost = ops / SIMD_width`. We tried this. The bench shows:

| Cost model | Mean estimate/wisdom ratio |
|------------|---------------------------|
| Greedy factorizer (no scoring) | 1.85× |
| **`ops / SIMD_width` (pure static)** | **1.69×** |
| Sqrt-throttled `ops / SIMD_width` | 1.33× |
| Linear-throttled `ops / SIMD_width` | 1.33× |
| Empirical CPE per radix | **1.19× ✓** |

The ops-only model gets only 9% of the way from baseline (1.85) to
target (1.20). That's because op counts are a structural feature of
the codelet source, but execution time is a hardware-dependent
function of the source. The static count under-counts hardware-induced
costs that scale super-linearly.

### Concrete failure modes

**Decoder pressure**: A 717-op codelet (R=32 t1) doesn't run 8× faster
than a 90-op codelet (R=8 t1) on the same hardware. Real ratio: 6.3×
in cycles. The op count over-promises by ~25%.

**DTLB overflow**: Static counts don't see address space. R=64 at K=256
touches ~160 distinct 4 KB pages per iteration, blowing through the
L1 DTLB's ~96 entries → STLB-walk on every store → 43% of clockticks
lost to TLB. The op count says R=64 should be ~14.6× R=8; reality at
K=256 is 49×.

**Dependency chain length**: R=11 (Winograd prime) has 180 ops; the
critical path through its FMA chain is ~30 dependent operations long.
No SIMD vectorization saves you here — the chain runs serially. Op
count says this should be 2× R=8 (132 ops); reality is 4× in cycles.

### Why we keep `radix_profile.h` anyway

The static profile exists as a **fallback** for radixes whose CPE
hasn't been measured (unusual — typically only happens between codelet
regeneration and CPE regeneration). It's also useful as a **sanity
check** on the CPE numbers — if `cyc_t1` for R=8 looks suspiciously
similar to R=4's, the static profile flags that the codelets aren't
that different in instruction count.

For everyday cost-model decisions: the CPE table is the source of
truth, the profile is the shadow.

## 3. The cost model must mirror the executor

A subtle but structural point: **what the cost model scores has to
match what the executor actually runs.**

This sounds obvious. It isn't, because there's a tempting
simplification — the cost model just scores plans against an
"average" or "default" codelet, and the executor independently picks
the variant that wins at runtime via wisdom predicates.

That decoupling looks clean. It's wrong.

### Why decoupling breaks the cost model

`_stride_build_plan` (the function both ESTIMATE and MEASURE paths go
through) consults `stride_prefer_t1s(R, me, ios)` per stage and
attaches t1s if the predicate fires. The executor uses t1s. So the
**actual cycle cost of that stage is `cyc_t1s`**, not `cyc_t1`.

If the cost model scores against `cyc_t1` everywhere:

- Stages where t1s is cheaper (84% of production cells, with t1s
  saving 5–35%) get scored higher than they actually run.
- The model thinks "more stages" is more expensive than reality
  because it's pricing each extra stage at t1's cost.
- Plans with deeper R=4 chains get systematically under-picked, even
  when they'd actually win.

We saw this before fixing it: pre-mirroring, the model preferred
shallower R=8/R=16 plans, missing the wisdom's R=4 chains.

### The fix: same predicate, same input

The cost model has to make exactly the same call:

```c
/* Inside _radix_butterfly_cost (factorizer.h) */
if (stride_prefer_dit_log3(R, me, ios)) { return cyc_log3; }
if (stride_prefer_t1s(R, me, ios))     { return cyc_t1s;  }
return cyc_t1;
```

with the same `(R, me, ios)` triple `_stride_build_plan` will use
when constructing the actual plan. Both arrive at the same predicate
inputs → both pick the same variant → no drift.

This is a **structural property**, not an optimization. If a future
change to plan-build adds a new variant or reorders precedence, the
cost model must change in lockstep. The two are coupled by the
runtime contract: the cost model models the executor.

### Why this is novel

Most planners either don't have variant predicates at all (FFTW's
ESTIMATE picks a single codelet per radix) or hide the variant
selection behind a black box (MKL's pre-baked plans). We expose the
variant selection as a public predicate (`wisdom_bridge.h`) that the
cost model and executor *both* call.

The predicate file is a small bit of public surface area: it's the
contract between cost modeling and execution. If one diverges from
the other, the bench (`bench_estimate_vs_wisdom.c`) catches it as a
regression.

## 4. Why K=256 is the baseline

The CPE table is measured at K=256, not K=4 or K=1024. This is a
deliberate choice that reflects the actual usage distribution — but
it does need explanation, since it could look like an arbitrary
constant.

### Three reasons

1. **Production usage centers on K=256.** Audio frame batches, image
   tile counts, radar pulse-Doppler matrices — most workloads sit
   around K=256. Most wisdom entries are there. Most bench cells
   are there.

2. **Big enough to amortize loop overhead.** At K<64, the per-call
   constant overhead (function dispatch, register save/restore)
   distorts per-butterfly numbers. At K=4, we'd be measuring overhead
   as much as compute.

3. **Small enough to fit in L1.** R*K complex doubles for R=64, K=256
   = 32 KB — fits in 48 KB L1 on Raptor Lake. Bigger K would push
   the working set into L2, conflating codelet timing with cache
   effects. We'd be measuring "what happens when the codelet has to
   miss L1" rather than "what does the codelet do."

### Why this works for K ≠ 256

The cost model adjusts for stride pressure via `cache_factor` (the
3-tier L1/L2 step function in `stride_score_factorization`). When the
plan's per-stage working set differs from the K=256 baseline, the
model multiplies the per-butterfly cost by `cache_factor`.

This is how a single-K calibration extends to all-K plans. It's not
perfect — the cache_factor is a step function, not a gradient, so
50 KB and 1.5 MB working sets both score 3.0 — but it's adequate.
The bench shows good ratios at K=1024 cells (which were never
directly measured), confirming the extension works.

### When this breaks down

Two cases the K=256 baseline misses:

- **Very small K (K=4)**: per-element overhead dominates. The cost
  model's cycles-per-butterfly include K=256-amortized overhead, so
  we under-cost K=4 stages where overhead is relatively bigger.
- **Very large K (K≥4096)**: cross-stage cache pressure becomes
  dominant in ways the cache_factor doesn't fully capture. The bench
  doesn't include these cells today, so this is uncalibrated territory.

These are documented v1.x improvements in
[09_decisions.md](09_decisions.md) under "Why we measure at K=256
only" — the trade-off was accepting these edges in exchange for a
single, cheap baseline.

## How the pieces fit

Everything in the cost model is downstream of the four ideas above:

| Architectural decision | Justified by |
|------------------------|--------------|
| Empirical CPE table (`measure_cpe.c`) | Memory-boundedness — needs hardware measurement |
| Static profile is fallback only | Op counts diverge from cycles by 4–8× on memory-bound radixes |
| Variant lookup mirrors plan-build | Cost model has to model what runs, not what *might* |
| K=256 baseline | Trade-off between calibration cost and coverage |
| Cache factor on stride pressure | Single-K calibration → all-K plans |
| Wisdom-bridge predicates as public surface | Cost-model / executor contract |
| Variance check at 5% on `measure_cpe` | Bad CPE numbers cascade — refuse before commit |
| Calibration done by orchestrator, not at runtime | Codelet generation is the right point in the lifecycle |

If the memory-bound thesis turns out to be wrong on some future
hardware (say, a CPU with so much DTLB / L2 / load bandwidth that
arithmetic re-becomes the bottleneck), most of these decisions need
revisiting. That's the load-bearing claim.

## Why this approach is novel

To anchor the claim explicitly:

| Existing approach | Plan-time cost | Plan quality | Limit |
|-------------------|---------------|--------------|-------|
| FFTW PATIENT/EXHAUSTIVE | minutes (full measurement) | optimal-ish | unusable in interactive contexts |
| FFTW ESTIMATE | µs (static heuristics) | poor (commonly 2–5× off) | doesn't capture hardware |
| Intel MKL | µs (pre-baked dispatch) | good but black-box | not portable to new hardware |
| FFTPACK / KISSFFT | n/a (fixed factorization) | depends entirely on the size | no choice = no cost model |
| Spiral | offline DSL search | very accurate | not a runtime estimator |
| **VectorFFT ESTIMATE** | **µs (table lookup + recursion)** | **1.0–1.3× of measured** | **needs CPE regen on hardware change** |

The combination — runtime cost = µs, plan quality = within 1.3× of
measured, hardware-portable via regen — is what makes this novel.
The cost is paid by codelet maintainers (re-running `cpe_measure`
when the hardware changes), not by users (zero first-run cost).

## See also

- [docs/vtune-profiles/](../vtune-profiles/) — the empirical evidence
  for the memory-bound claim
- [01_architecture.md](01_architecture.md) — how the ideas here become components
- [05_variant_selection.md](05_variant_selection.md) — the
  cost-model / executor mirroring in code
- [09_decisions.md](09_decisions.md) — non-obvious choices captured
  for future maintainers
