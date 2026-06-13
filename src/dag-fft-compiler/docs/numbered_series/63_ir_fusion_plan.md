# 63. Deferred FMA materialization — IR rewrite plan (Option C)

Status: Plan only. No code changes. This doc captures the design and the
open decisions that need to be settled before implementation begins.

## Motivation

Two open gaps to FFTW that share a common root cause:

```
R=25  +31 ops over FFTW (+8.8%, after Winograd-5)
R=64  +66 ops over FFTW (+7%)
```

Both gaps live in the same algsimp pattern: a shared `Mul(c, d)` consumed by
two FMA-class operations with opposite signs (the classic butterfly-pair
shape). FFTW emits 2 FMAs per pair (each FMA recomputes `c·d` internally as
one µop). We emit 1 Mul + 2 FMA-class ops, costing 3 IR-level ops per pair
where FFTW spends 2. At R=25, there are 31 such pairs; at R=64, the count
matches the broader gap with some extra structure.

A second open issue motivates the same fix: the **AVX-512 R=25/R=7
Winograd-vs-baseline regression** documented in Doc 62. The regression is
real (~4% reproducible), and it exists because our conjugate-pair Direct
codelet has wider parallel structure than Winograd, so on a register-rich
ISA it wins on port utilization. Closing the butterfly-pair gap eliminates
the structural advantage of conj-pair Direct; Winograd is then unambiguously
better at every ISA.

So one IR change addresses both:

- Closes the +31 R=25 gap to FFTW
- Closes most of the +66 R=64 gap to FFTW
- Eliminates the AVX-512 Winograd regression as a side effect
- Removes our last reason to keep dual code paths for prime-N

## Current state

Our IR (in `lib/expr.ml`, roughly):

```
type node =
  | NK_Const of float
  | NK_Load of int
  | NK_Neg of t
  | NK_Add of t * t
  | NK_Sub of t * t
  | NK_Mul of t * t
  | NK_Fma of t * t * t * bool * bool   (* a, b, addend, neg_m, neg_a *)
```

`NK_Fma` is constructed by several passes in `algsimp.ml`:

```
factor_const_muls         creates Fma for Add(K*X, Y)/Sub patterns
multi_use_fma_lift        creates Fma during multi-use mul lifting
fma_addend_factor         creates Fma during addend factorization
flatten_fma_mul_addend    rewrites existing Fma chains
```

Once an Fma node exists, its three operands are committed. Downstream
(regalloc, schedule, emit) treats Fma as a primitive with one register
write. The "frozen tags" cross-pass spilling machinery in the recipe
identifies values by Fma node identity.

For comparison, FFTW's `genfft/expr.ml`:

```
| Plus of expr list      (* n-ary, signed children via Uminus *)
| Times of expr * expr   (* binary *)
| Uminus of expr
| Load / Store / Num
(* no Fma in the IR *)
```

FMA is an emit-time pattern in FFTW's `c.ml`:

```
| Plus (Times(a,b) :: c :: rest) -> fma(a, b, emit (Plus (c :: rest)))
| Plus (Uminus(Times(a,b)) :: c :: rest) -> fnmadd(a, b, emit (Plus (c :: rest)))
...
```

The Plus walker is greedy: every Plus child is independently considered for
FMA fusion. The same `Times` appearing in two different Plus parents emits
as two FMAs.

## Goal

Eliminate `NK_Fma` from our IR. Defer FMA pattern matching to emit time
without copying FFTW's n-ary Plus list. The IR stays binary; the emitter
learns to walk Add/Sub chains and absorb Mul children.

## Why Option C, not A or B

Three options were considered (Doc 62 references this as "n-ary IR work"):

- **Option A** — Remove `NK_Fma`, keep binary Add/Sub/Mul. Emitter pattern-
  matches Add/Sub trees at print time. Same IR shape as today minus the Fma
  node. Estimated 3-5 days.
- **Option B** — Migrate to n-ary `NK_Sum of (t * bool) list`. Full IR
  rewrite. Algsimp learns to operate on Sum nodes. Estimated 1-2 weeks.
- **Option C** — Like A, but pattern matching happens in a dedicated walker
  that builds a virtual flat sum on the fly, rather than ad-hoc per-node
  patterns. Same IR shape as today minus Fma. Estimated 4-6 days.

Option C is the chosen path. Rationale:

- Closes the same fundamental gap as A and B (butterfly-pair fusion at emit
  time). The R=25/R=64 wins don't require Sum-aware algsimp.
- Doesn't touch IR construction across all passes; only emit changes
  substantially.
- Can be promoted to B later if specific algsimp-on-Sum patterns emerge as
  valuable. Most likely we never need to.

## Design

### Pipeline change

Today's order:

```
IR construction → algsimp (creates Fma) → schedule → recipe → regalloc → emit
```

After Option C:

```
IR construction → algsimp (no Fma created) → schedule → fusion planning
                → recipe → regalloc → emit (Fma-aware)
```

The new step is **fusion planning**. It runs between schedule and recipe.
Recipe sees only Muls that are committed to materialize; regalloc sees the
post-fusion live ranges.

### The fusion planning pass

Each Mul gets annotated with a status:

```
type mul_status =
  | Fused          (* absorbed into every consumer FMA; never materializes *)
  | Materialized   (* lives as a stored register value *)
```

Classification rules, applied in order:

```
RULE 1 (safety): if any consumer is cross-pass (recipe says this value
       crosses a pass boundary) → Materialized

RULE 2 (win):    if all consumers have FMA-absorbable shape (Add(Mul,X) or
       Sub(X, Mul) or Fma-class with Mul in addend slot) → Fused

RULE 3 (pressure): if Fusing this Mul would push peak-live above the
       register budget (vec_regs) → Materialized

RULE 4 (default): Fused
```

Rule 1 is non-negotiable: cross-pass values must materialize because
`c·d` cannot be recomputed inside an FMA in pass 2 unless `c` and `d` are
themselves available in pass 2 (and our recipe machinery moves single
values, not their factor pairs).

Rule 2 captures the butterfly-pair win. The rule generalizes: as long as
every consumer can absorb the Mul, fusion is a strict op-count improvement
(`1 Mul + N consumers → N FMAs`, save 1 op net).

Rule 3 prevents regression. The peak-live calculation under fusion treats
the Mul's inputs `c, d` as having extended live ranges — they must remain
live at every consumer site, not just at the would-be Mul site. In adverse
schedules this can increase peak; Rule 3 backs off.

Rule 4 is the optimistic default.

### Cross-pass invariant

The recipe machinery (today's "frozen tags") identifies values that cross
pass boundaries. Under Option C, the fusion planner reads the recipe's
crossing set and force-materializes those Muls via Rule 1.

This means:

```
spill set under Option C  ⊇  { mul | mul is in the recipe crossing set }
                         ≈   the same set we have today
```

The set may grow (if Rule 3 fires defensively), never shrinks. Mechanical
equivalence with today's spill behavior at pass boundaries is guaranteed.

### Peak-live calculation under deferred FMA

The current peak-live model is straightforward: each IR node is a register
write, live from its def to its last use. Under fusion this changes for
both fused Muls and their inputs.

Concrete butterfly-pair example. Today:

```
m  = Mul(c, d)        ← m occupies one register slot, live from here…
y1 = Add(K*x, m)
y2 = Sub(K*x, m)      ← …to here. Then m can be freed.
```

Peak contribution from m: 1 slot, over the y1→y2 interval.

After fusion:

```
y1 = Fma(c, d, K*x)   ← c, d, K*x all live at y1's emit point
y2 = Fnmadd(c, d, K*x) ← c, d, K*x all live at y2's emit point
                       ← m never lives
```

Peak contribution:
- m: 0 slots (gone)
- c: extended live range — must reach both y1 and y2 instead of just the m site
- d: same
- K*x: same liveness as before (it was already needed at both y1 and y2)

Net change per fused Mul:
- Strict −1 if c and d's live ranges were already extending past the m
  site for other reasons (Rule 3 cannot fire).
- −1 to +1 depending on schedule if c and d's ranges have to be extended
  past their would-be natural end. (Rule 3 evaluates this.)

This is data-dependent. The fusion planner cannot make Rule 3 decisions
without seeing the schedule.

### The fixed point

There is a dependency cycle:

```
fusion decisions  →  affects peak-live
peak-live         →  determines whether Rule 3 fires
Rule 3 firing     →  forces materialization, removes a fusion
                  → affects peak-live again
```

Resolution: iterate. Each iteration can only ADD materialization
(monotonically grow the Materialized set), never remove it. Bounded by
total Mul count.

```
Iteration 0: all Muls optimistically Fused (except Rule 1 cross-pass)
Iteration k: compute peak-live; for each Fused Mul, check Rule 3;
             if Rule 3 fires, change to Materialized; repeat
Stop:       no Mul changed status this iteration
```

In practice expected to converge in 0-2 extra iterations after the initial
pass. Worst case: everything materializes → equivalent to today's behavior.

## Failure modes and how to handle them

### 1. Schedule sensitivity

Two equivalent schedules of the same IR can give different peak-live numbers
under fusion. If `c` and `d` happen to be short-lived in one schedule
(naturally die soon after `m`), fusion is cheap; if they happen to be
long-lived in another, fusion is expensive.

**Mitigation**: planner runs after schedule, when emit order is known. The
planner doesn't try to influence the scheduler; the scheduler doesn't know
about fusion. This is the simplest separation. A more sophisticated approach
(scheduler-aware fusion) is future work — not needed for the gap we're
trying to close.

### 2. Tag stability

The recipe machinery references specific node tags to identify spill slots
and load points. Today these tags are stable from algsimp through emit.

Under Option C, a fused Mul still has a tag (the IR isn't deleted), but the
tag never appears in emitted code — there's no stored value associated with
it. Recipe must not assign a spill slot to a Fused Mul.

**Mitigation**: planner runs *before* recipe. Recipe sees only the
Materialized Mul set; Fused Muls are invisible to it. Pipeline ordering is
load-bearing.

### 3. Phantom dependencies in regalloc

Today's regalloc uses def-to-last-use ranges keyed by node. Under fusion,
a Mul's inputs `c, d` have extended ranges that aren't visible from the
IR alone — they're determined by the planner's annotations.

**Mitigation**: the planner emits an annotation table:

```
type fusion_annotation = {
  fused_muls: Mul.t set;
  extended_live: (Node.t * Range.t) list;  (* c, d ranges per fused Mul *)
}
```

Regalloc consults the annotation table when computing live ranges. The
change is local — one new input to regalloc's range builder, no algorithm
change.

## Conservative vs Aggressive variants

Two ways to ship this:

### Conservative (recommended for first cut)

Rule 3 fires whenever fusion *might* increase peak-live, even if peak hasn't
yet reached the budget. Equivalent to:

```
RULE 3': if extending c, d ranges past their non-fused last-use → Materialize
```

Forfeits some wins but eliminates the planner ↔ regalloc iteration. Closes
maybe 70-80% of the FFTW gap. Probably ~22 of the 31 R=25 muls fuse.

Engineering: ~4 days.

### Aggressive

Full Rule 3 with peak-live iteration. Closes ~95% of the gap (29-31 of
the 31 R=25 muls fuse). Requires the iteration loop and regalloc
annotation interface.

Engineering: ~6 days.

**My recommendation**: ship Conservative first. Measure. If the residual gap
is significant on EPYC bench, upgrade to Aggressive. The conservative→
aggressive promotion is a localized change in the planner and a small
regalloc surface.

## Testing strategy

Three layers:

### Correctness (sub-ulp diff)

For every radix in the current test matrix (R=2..256, plus the non-pow2
radices we added today: 14, 15, 20, 21, 25), generate the codelet under
Option C and compare bit-for-bit numerical output against the current main.
Tolerance: max |Δ| < 1e-12. Anything looser is a sign of an arithmetic
restructure that shouldn't have happened.

Run for both Fwd and Bwd.

### Op-count regression

Today's op counts at every radix become baseline. Option C should produce:

- Same or fewer ops at every radix (no regressions allowed)
- Strictly fewer at R=25 (target: 383 → ~352, matching FFTW)
- Strictly fewer at R=64 (target: 978 → ~912, matching FFTW)

A simple regression script that runs at every commit; any radix that
increases is a hard fail.

### Spill-count regression

Same approach: pin AVX2 and AVX-512 spill counts at every radix. Under
Option C they should:

- Decrease modestly (fewer materialized Muls = fewer values to spill)
- At minimum, never increase

Adversarial test: write a synthetic IR with worst-case butterfly patterns
(many shared Muls with non-cooperating consumer shapes) and confirm Rule 3
fires correctly.

### Wall-clock regression

After everything else passes, re-run the bench harness from Doc 62:

- AVX2 R=7, R=25: expect ≥ today's measurements (Winograd already wins here)
- AVX-512 R=7, R=25: expect to flip from regression to neutral or win

If AVX-512 doesn't flip, the IR work isn't doing what we expected and
warrants investigation before merge.

## Open decisions to settle before implementation

These are the architectural choices that affect later work and can't easily
be revisited:

### 1. Planner ordering

Plan: planner runs after schedule, before recipe and regalloc. Non-negotiable
for the design as written. Confirm before starting.

### 2. Conservative or Aggressive variant for first cut

Plan: Conservative. Trades ~25% of the achievable win for simpler regalloc
interface. Promote later if needed.

### 3. NK_Neg handling

FFTW uses `Uminus` as a Plus-child decoration (each child carries its own
sign). Our `NK_Neg` is a separate node that the emitter has to handle as
either standalone or absorbed into a parent Add/Sub.

Three options:
- **Leave NK_Neg alone.** Emitter handles it as today. Simplest.
- **Eliminate NK_Neg** by replacing each `Neg(x)` with `Sub(Const 0.0, x)`.
  Reduces node-type count. Adds a Sub for every Neg.
- **Push sign onto Add/Sub children.** Would require new IR shape. Too far.

Plan: leave NK_Neg alone for first cut. Revisit if it causes pattern-matching
complexity in the emit-time walker.

### 4. Algsimp passes that create Fma — keep or remove

Current algsimp creates Fma in 4 places. Under Option C none of them should
fire (no Fma in IR). Options:

- **Remove the passes entirely.** Cleaner but harder to back out if the
  rewrite has issues.
- **Make them no-ops conditional on a build flag.** Adds a flag we said
  we'd avoid.
- **Replace them with comments referencing this doc.** Preserves the
  algebraic insight without dead code. Easy to restore if needed.

Plan: physically remove the passes. The algebra they discover (e.g.,
`Add(Mul(K,X), Y) → Fma(K,X,Y)`) is exactly what the emit-time walker
rediscovers. Keeping both is redundant.

### 5. Recipe interaction with the fusion set

The recipe machinery needs an API to say "this value crosses a pass
boundary; do not fuse." Today recipe-decisions happen before algsimp's
final passes run. Under Option C, recipe must inform the planner.

Plan: add a `crossing_set: Node.t set` produced by recipe and consumed by
the planner. Single new piece of data flowing through the pipeline.

## Phase plan summary

The plan is two phases. Phase 1 reaches parity with FFTW; Phase 2 (only if
Phase 1's measurements warrant it) consults the existing uarch cost model
to surpass FFTW's emit-time greedy. Detailed day-by-day plans are at the
end of this document, after the Phase 2 design.

## What this doesn't address

For honesty's sake, things Phase 1 does NOT fix:

- **The R=15, R=20 gaps to FFTW.** Those are mostly twiddle-stage
  inefficiencies that exist with or without Fma in the IR. The
  Winograd-5 cascade in Doc 62 already pushed them as far as the
  binary IR allows.
- **Non-CT radices we don't support.** R=11, R=13 still go through
  Direct conjugate-pair. Adding Winograd codelets for those is a
  separate piece of work.

Phase 1 closes the FFTW gap by matching them; Phase 2 (below) is where
we go past them.

## Phase 2 — Beyond FFTW: critical-path-aware joint planning

Phase 1's greedy emit-time fusion gets us to parity with FFTW. Two
extensions can go further. Both reuse the same cost-model infrastructure
that's already partially in tree (`lib/uarch.ml`, `lib/schedule.ml`).

The case for Phase 2: FFTW's `c.ml` greedy walker takes the first `Times`
child it sees in each Plus, independently per Plus. This is the simplest
correct policy. It's not optimal:

- When a Plus has multiple Times children, the choice of which to absorb
  affects critical-path length through the codelet.
- When a Mul is shared across multiple consumers, all-fuse vs
  all-materialize is a binary choice that misses the mixed strategy.

Both problems share a cost model — fusion is a tradeoff between op count,
critical path, and register pressure. Existing infrastructure makes the
cost data cheap:

```
lib/uarch.ml
  type t = { add_latency, mul_latency, fma_latency,
             load_l1_latency, ... }
  Profiles for Sapphire Rapids, Raptor Lake AVX-512/AVX2,
  Zen 5, generic AVX-512, generic AVX2.

lib/schedule.ml
  compute_cp_dist : Uarch.t → Expr.t → cp_dist_map
  Already computes critical path distance per node using uarch latencies.
  Drives the existing scheduler's bisection choices.
```

So microarch awareness is "free" — Phase 2's cost model is a thin wrapper
over `Uarch.t` and `cp_dist`. The earlier mention of #3 in the discussion
that motivated this section is effectively already shipped; what's missing
is consulting it from the fusion planner.

### Extension #1 — critical-path-aware fusion order

FFTW emits this:

```
Plus [Times(a,b); Times(c,d); x]   →   fma(a, b, fma(c, d, x))
```

Critical path through this expression:

```
inner fma(c, d, x):  max(cp(c), cp(d), cp(x)) + fma_latency
outer fma(a, b, _):  max(cp(a), cp(b), cp(inner)) + fma_latency
```

If `c, d` arrive late (e.g., they come out of an earlier FMA chain) and
`a, b` arrive early, the outer FMA stalls waiting for the inner. The
alternative ordering:

```
fma(c, d, fma(a, b, x))
```

has the early-arriving FMA on the critical path, and the late-arriving
ones feeding it. Often shorter.

**Design**: when the emit walker encounters a Plus (virtual flat sum) with
multiple Mul children, sort absorption order by `cp_dist` of the Mul's
inputs — late-arriving first into the deepest position, early-arriving last
into the outer FMA. Greedy with a cost-model tiebreaker.

Compatible with Phase 1's structure; the walker stays a single pass, just
sorts its candidates before consuming them.

**Estimated win**: 1-5% wall-clock on register-rich ISAs (AVX-512), where
we're more often latency-bound. On AVX2 (register-starved), under 1% —
spill traffic still dominates.

**Cost**: 2-3 days after Phase 1 ships. Single new function in the emitter;
consults the existing `cp_dist` map.

### Extension #2 — joint optimization across shared Muls

This is the extension that addresses our R=25/R=64 gap shape directly,
and where we could plausibly beat FFTW rather than match them.

FFTW emits each Plus independently. A `Times(a,b)` appearing in 4 Plus
parents emits as 4 FMAs, each recomputing `a*b` internally. Op count:
4 FMAs vs (1 Mul + 4 Adds) = 4 vs 5. FMA wins on op count.

But op count isn't the only metric. Consider:

- **FMA port pressure**: if we've saturated FMA-capable ports, the
  4-FMA emission stalls. Materializing once frees 3 ops for other ports.
- **Critical path**: 4 independent FMAs run in parallel given enough
  registers. 1 Mul + 4 dependent Adds serializes. FMA usually wins
  here unless the Mul is on the critical path of all 4 consumers
  anyway.
- **Live ranges**: materializing extends the Mul's range; fusing
  extends `a` and `b`'s ranges instead. Either can be worse depending
  on the schedule.

The right policy is per-Mul, based on local cost. Possible strategies for
a Mul `m = Times(a, b)` with consumer set `C`:

```
strategy_all_fuse:     m has no register write; every consumer absorbs
strategy_all_materialize:  m is one register write; every consumer reads it
strategy_mixed(S ⊆ C): m materializes; consumers in S read the value;
                       consumers not in S absorb (recomputing a*b internally)
```

The mixed strategy is interesting and not in FFTW's repertoire. It comes up
when most consumers are FMA-shape and one isn't — today we'd materialize
to satisfy the non-FMA-shape consumer, costing the FMA-shape consumers
their potential fusion. Mixed gives the materialized value to the
non-FMA consumer and lets the others fuse.

**Design**: per Mul, the planner computes cost of each candidate strategy
using the model:

```
cost(strategy) = α · ops_added
               + β · critical_path_extension
               + γ · peak_live_extension
               + δ · port_pressure
```

Coefficients α, β, γ, δ derived from the uarch profile (FMA-port-rich
microarchs weight β lower; register-starved weight γ higher). For each
Mul, pick the lowest-cost strategy.

For R=25's 31 butterfly-pair Muls: today each is 2-use with one Add(K*X, m)
and one Sub(K*X, m) consumer. All-fuse closes the +31 op gap to FFTW. The
planner would discover this without being told.

For R=64's mixed Cat-A and Cat-B patterns: the planner picks per-Mul,
sometimes recovering the mixed strategy.

**Estimated win**: 0-3% on average, 5-8% on high-share codelets like R=64.
The latter is the case where we'd genuinely surpass FFTW. They use the same
greedy walker we'd use in Phase 1; this extension makes per-Mul decisions
they don't make.

**Cost**: 3-4 days after Extension #1. Builds on the same cost-model
plumbing. New planner pass slot between Phase 1's planner and emit.

### Combined Phase 2 architecture

Both extensions plug into the same place:

```
Phase 1:
  algsimp (no Fma) → schedule → planner (Rules 1-3-4) → recipe → regalloc → emit

Phase 2:
  algsimp (no Fma) → schedule → planner (cost-model, Ext #2) → recipe
                   → regalloc → emit (cost-model walker, Ext #1)
```

The cost model is a single shared module (`lib/fusion_cost.ml`, new),
consumed by both the planner and the emit walker. Inputs: `Uarch.t`,
`cp_dist` map from schedule. Outputs: per-strategy cost number.

This isolates the cost-model code in one place. If the model is wrong (and
it will be — cost models always are), tuning happens in one file rather
than scattered across planner and emitter.

### When to ship Phase 2

After Phase 1 measures on EPYC and shows what's left. The signal to upgrade
to Phase 2 is one of:

- FFTW parity reached at major radices but residual gaps at R=64/R=256
  haven't fully closed (Extension #2 territory).
- AVX-512 wall-clock benchmarks are latency-bound (Extension #1 territory).
- Per-microarch tuning becomes valuable (the uarch profiles are already
  there — would just mean exercising them more).

If Phase 1 already overshoots FFTW (unlikely but possible if our regalloc
turns out better than theirs on some specific codelets), Phase 2 may not
be worth doing immediately. Measure first.

### Combined estimated win envelope

Stacked, all phases plus their extensions:

```
Phase 1                  : close 70-95% of current FFTW gaps. Match parity.
+ Extension #1 (CP-aware): +1-5% wall-clock on AVX-512
+ Extension #2 (joint)   : +0-3% average, +5-8% on R=64-class codelets

Realistic combined ceiling: ~3-8% beyond FFTW on best cases.
Adversarial worst case: same as FFTW (cost model declines to fuse where
  FFTW does, but ends up equivalent in practice).
```

These numbers are upper bounds; real codelets will see less. FFTW has been
tuned for 30 years and is hard to leapfrog on emit quality alone. The
realistic place to surpass them substantially is in cross-codelet
optimization, target-specific code generation (already partially shipped
via uarch profiles), and choosing fundamentally different algorithms for
specific radices.

## Implementation outline

If we commit to the plan:

### Phase 1 — match FFTW (~1-2 weeks)

```
Day 1   Remove NK_Fma from IR and all algsimp passes that create it.
        Verify build is clean and all tests still produce numerically
        correct output (will have higher op counts because no fusion
        anywhere yet — that's expected).

Day 2   Write the fusion planner pass: Rules 1-2-4 (no peak-live check).
        Wire it after schedule. Add `mul_status` annotation table that
        emit consults.

Day 3   Write the emit-time FMA walker. Pattern-match Add/Sub trees;
        when a child is a Mul with Fused status, absorb. When a child
        is Materialized, emit it as a register load.

Day 4   Run regression suite. Fix correctness issues.

Day 5   Add Rule 3 (Conservative form). Verify spill counts don't
        regress. Iterate on edge cases.

Day 6   Wall-clock bench. Document results. Decide whether to upgrade
        to Aggressive variant or move to Phase 2.
```

### Phase 2 — beyond FFTW (~1 week additional, only if Phase 1 measurements warrant)

```
Day 1   Extract `Fusion_cost` module. Consumes Uarch.t and cp_dist;
        produces strategy-cost numbers. Initial coefficients chosen
        analytically; tuned empirically later.

Day 2   Extension #2: planner consults cost model per Mul; picks
        all-fuse / all-materialize / mixed. Replace Rules 1-3-4 with
        cost-model-driven decision (Rules 1 and 3 remain as hard
        constraints; Rule 4 becomes "lowest cost strategy").

Day 3   Extension #1: emit walker sorts absorption candidates by
        critical-path metric before greedy consumption.

Day 4   Regression suite. Confirm no codelet regresses; measure wins
        at R=25, R=64, R=128, R=256.

Day 5   Wall-clock bench. Compare Phase 1 vs Phase 2. Document.
        Tune cost-model coefficients if needed.
```

Realistic combined effort: 2-3 weeks with debugging.

- Doc 59 — n-ary IR sizing discussion in the AVX2 R=64 addendum
- Doc 62 — Winograd-5/7 results; the R=25 +31 gap that motivates this
- FFTW `genfft/expr.ml` and `genfft/c.ml` — reference implementation
  of the Plus-list + emit-time greedy walker pattern
- Frigo, M. — "A Fast Fourier Transform Compiler" (PLDI 1999), §3 covers
  the genfft IR design
