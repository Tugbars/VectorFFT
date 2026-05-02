# VectorFFT: DIT vs DIF Codelets and the Per-Stage Mixing Question

**Date:** April 2026
**Author:** Tugbars (with notes from a session-long collaboration)
**Status:** Design note. Not committed work. Future-direction reference.

---

## Context

By April 2026, VectorFFT's codelet portfolio was effectively complete: 17
radixes (R = 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 20, 25, 32, 64),
three twiddle protocols (flat, t1s, log3), specialized AVX-512 variants
(isub2, log_half), and Phase-B buffered variants. Per-chip autotuning ran in
~9 minutes on Raptor Lake AVX2 (RL) with proper power-plan and affinity
hygiene. The bench wisdom emit produced per-radix `prefer_log3(me, ios)`
predicates that the executor's planner could consult to activate the log3
twiddle protocol where it won.

A subtle issue had been deferred throughout: the codelet bench treats
DIT-log3 and DIF-log3 as same-protocol alternatives (both are "log3"), and
the wisdom emit picked whichever was faster per cell without distinguishing
between them. This is fine for codelet-level reporting — we want to know
which codelet is fastest. But when the executor consumes the wisdom to
activate log3 at a given stage, the question of *which* log3 codelet it
should call became unavoidable.

This note documents what we discovered when we tried to integrate DIF-log3
codelets into the executor's existing per-stage protocol selection, why
the naive approach failed mathematically, and what the proper architecture
would look like if we ever decided to build it.

---

## The codelet-level finding

On Raptor Lake AVX2 across the 17-radix portfolio, log3 codelets won 6/18
cells at R=16, 9/18 at R=32, and 13/18 at R=64. Within those wins, DIF-log3
was the fastest variant in roughly a third of the cells — clustered at large
me with power-of-two ios. At R=64, DIF-log3 was decisively faster than
DIT-log3 in several high-me cells, with margins comparable to log3's win
over flat.

The mechanism behind DIF-log3's wins at R=64 is mostly twiddle scheduling.
Pre-butterfly twiddles (DIT) sit on the data dependency chain before any
butterfly arithmetic begins. Post-butterfly twiddles (DIF) overlap with
output stores, hiding twiddle latency behind memory work. At R=64 where the
butterfly itself is long enough to absorb derivation latency, this scheduling
advantage becomes decisive. At R=16 the butterfly is too short and the
advantage disappears — explaining why DIT-log3 dominates at smaller radixes.

This data made integration tempting. If DIF-log3 wins 13 cells across the
portfolio — and several with margins of 15-25% over DIT-log3 — capturing
those wins should follow naturally from the existing protocol-selection
machinery. The Phase 2 integration of DIT-log3 had recently shipped; adding
DIF-log3 to the menu felt incremental.

It wasn't.

---

## The naive integration attempt and what failed

The intended Phase 3 integration was minimal:

1. Add a `prefer_dif_log3(me, ios)` predicate alongside the existing
   `prefer_dit_log3` and the union `prefer_log3`.
2. At plan time, when log3 is selected for a stage, query both
   `prefer_dit_log3` and `prefer_dif_log3` and dispatch to whichever wins.
3. The codelet has the same 6-arg signature; it should be a drop-in
   replacement.

Implementation took maybe 15 minutes. Validation took longer.

We wrote a cross-validation diagnostic: feed the same input buffer and the
same twiddle table to both `radix16_t1_dit_log3_fwd` and
`radix16_t1_dif_log3_fwd`, then compare outputs. The result:

```
DIT-log3 vs DIF-log3, same input, same twiddles:
  max_diff = 21.76
  energy_match: yes (both unitary)
  leg-0 column matches exactly (no twiddle factor at m=0)
  other columns: differ by structured (i, m)-dependent factor
```

DIT-log3 and DIF-log3 produce *different output buffers* given the same
input. They are not implementing the same function.

---

## Why they differ: pre- vs post-butterfly twiddle math

The CT decomposition of an `R·me`-point DFT (with `Nbig = R·me`) writes
each output bin as

```
X[m + i·me] = Σ_j x[j·me + m] · W_{Nbig}^(j·(m + i·me))
            = Σ_j x[j·me + m] · W_{Nbig}^(j·m) · W_R^(i·j)
```

Two things multiply each input leg `x[j·me + m]`: the *external twiddle*
`W_{Nbig}^(j·m)`, and the *internal butterfly factor* `W_R^(i·j)`. The
external twiddle depends on the leg `j` and the slot `m` but not on the
output index `i`; the internal factor depends on the leg and the output
index but not on the slot.

DIT and DIF differ in *where* the external twiddle is applied:

- **DIT**: pre-butterfly. Each input leg `j` is multiplied by `W_{Nbig}^(j·m)`
  *before* the size-R butterfly. The butterfly then sums
  `Σ_j (x[j·me+m] · W_{Nbig}^(j·m)) · W_R^(i·j)` and writes to position
  `i·ios + m`.

- **DIF**: post-butterfly. The butterfly first computes
  `Y[i] = Σ_j x[j·me+m] · W_R^(i·j)` (no input twiddle). Then output `Y[i]`
  is multiplied by `W_{Nbig}^(i·m)`.

The crucial observation: in DIT, the per-leg factor `W_{Nbig}^(j·m)` is
inside the sum (varies with the summation index `j`). In DIF, the
per-output factor `W_{Nbig}^(i·m)` is outside the sum (constant within the
sum). These cannot be made equivalent by any post-multiply or any output
permutation.

What DIF-log3 actually computes, then, is

```
DIF[i, m] = W_{Nbig}^(i·m) · DFT_R(x[·, m])[i]
```

This is a perfectly well-defined transform. It is *not* the bin
`m + i·me` of an `Nbig`-point DFT of the input. It's the size-R DFT of
the input slice at slot `m`, post-multiplied by `W_{Nbig}^(i·m)`. As a
standalone operation, this isn't what a CT-DFT step is supposed to compute.

This was not what we expected. The R=16 codelet generator's DIF branch
contains a comment claiming "DIT's `_log3_subfft_chain(n2=k1)` applies
verbatim here — same derivations, same live-set, same cmul count." That
comment is true at the *code-organization* level — the chain of cmuls that
derives w3, w5, w7, ... is structurally identical between DIT and DIF — but
the *mathematical effect* of the resulting kernel is different. The
generator's correctness check (validate.c) only compares each codelet against
its own family's reference, never DIT vs DIF; this asymmetry was invisible
until we cross-validated.

---

## Why DIF-log3 isn't broken — it's just for a different schedule

The natural followup question: is DIF-log3 wrong? If it computes a different
function from DIT-log3, isn't the bench data measuring the wrong thing?

No. DIF-log3 is correct as the building block for a *DIF schedule*. A DIF
schedule reorganizes the multi-stage CT decomposition so that the per-leg
external twiddle of stage `s+1` becomes the per-output post-twiddle of stage
`s`. The factor `W_{Nbig}^(i·m)` that DIF-log3 produces at output `(i, m)`
is exactly the input twiddle that the next DIF stage would have applied if
it were DIT — moved one stage upstream and absorbed into the previous stage's
output multiply.

In a homogeneous DIF chain — every stage uses DIF codelets — the post-
twiddle of stage `s` flows into stage `s+1` as its (already-applied)
external twiddle, and stage `s+1` does a pure DFT_R with no further input
twiddle of its own. The result is the same as a homogeneous DIT chain, just
with the twiddles applied at different points along the data path. Both
schedules produce the same final DFT (modulo any output ordering convention,
which is handled by how the executor wires stages together).

The codelets compose correctly *within* their schedule. They do not compose
across schedules.

---

## Mixing DIT and DIF stages: the transpose-bridge problem

If we wanted to mix per-stage — say, DIT-log3 for stage 0 because it wins
at that (R, me, ios) cell, then DIF-log3 for stage 1 — the buffer state
between the stages is wrong. Stage 0 (DIT) leaves the buffer in
DIT-canonical form: each position holds the DFT bin value with no extra
factors. Stage 1 (DIF) expects to find inputs that already carry the
post-twiddle factor that a previous DIF stage would have left behind.
They don't match.

Bridging the mismatch requires a pass over the buffer that converts DIT-
canonical state into DIF-expected state (or vice versa). For a DIT→DIF
transition, that pass is a per-position multiply by the missing post-
twiddle factor. For DIF→DIT, it's the conjugate factor. Either way, it's
an `O(N·K)` pass — full memory bandwidth, one complex multiply per element.

For a representative case — `N = 4096` with factorization `64 × 64` and
batch `K = 8` — the buffer is `4096 · 8 · 16 bytes = 524 KB` complex
doubles. A bridge pass touches all of it: load + multiply + store.
Compare to one stage's cost in the same transform: a radix-64 codelet at
`me = 64`, `ios = 64`, executes in ~80μs on Raptor Lake AVX2, dominated
by the same memory traffic. The bridge is roughly **half a stage's work**
in time, depending on whether memory or FMA is the bottleneck.

If the per-stage codelet substitution gains 20% on the stage it's applied
to (a generous estimate at the cells where DIF-log3 wins), and the bridge
costs half a stage, then mixing breaks even only when more than half the
transform's stages are DIF wins. Below that threshold, bridges erase the
optimization.

In practice, on RL AVX2 most stages favor DIT-log3 and only a minority
favor DIF-log3. Mixed schedules with bridges would slow the transform down
relative to homogeneous DIT.

---

## The proper architecture: schedule-level decision, codelet-level autotuning

The right granularity for the DIT/DIF decision is **the whole transform**.
A schedule (DIT or DIF) is chosen for the entire factorization; per-stage
codelet selection (flat / t1s / log3) operates *within* the chosen schedule
independently.

```
                    EXECUTOR
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
       DIT executor          DIF executor
       (existing)            (would need building)
             │                    │
       per-stage              per-stage
       codelet pick           codelet pick
       ┌──┬──┬──┐             ┌──┬──┬──┐
       │  │  │  │             │  │  │  │
     flat t1s DIT  …         flat t1s DIF …
          DIT  log3                 DIF  log3

                       ▲
                       │
                    PLANNER
              consults whole-plan
              wisdom: which schedule
              wins for (N, K, fact)?
```

The codelet-level wisdom we already produce (`prefer_dit_log3`,
`prefer_dif_log3`, `prefer_t1s`, etc.) feeds into both executor paths
unchanged — within DIT, the planner consults `prefer_dit_log3`; within DIF,
it consults `prefer_dif_log3`. The cross-protocol comparison happens at the
*schedule* level, comparing whole-transform end-to-end timings of DIT plans
vs DIF plans for each (N, K, factorization) configuration.

This is roughly the FFTW model. FFTW's planner compares "direct", "DIT", and
"DIF" execution paths at the plan level, not per-stage.

The work needed to implement this:

1. **DIF executor** (~500-800 lines): a parallel execution path that runs
   the transform stages in DIF order, with per-stage twiddle setup matching
   what DIF codelets expect. The existing executor's backward path is
   *partly* DIF-shaped (hand-coded n1 + post-multiply), but doesn't actually
   call the `t1_*_dif` codelets we have — that wiring is missing.

2. **Whole-transform bench harness** (~200 lines orchestration): for each
   (N, K, factorization) tuple in a chosen range, build a DIT plan and a
   DIF plan, run both end-to-end, record the winner. Output is whole-plan
   wisdom: per (N, K, factorization), which schedule wins.

3. **Top-level dispatcher** (~50 lines): consults whole-plan wisdom at
   transform setup time and routes to the appropriate executor.

Estimated total effort: 1-2 weeks of focused development, plus comprehensive
cross-validation against an external reference (FFTW or naive DFT).

---

## Why we're not committing to building it yet

Two reasons.

**First**, the win is uncertain across architectures. On Raptor Lake AVX2,
DIT-log3 won the majority of log3-winning cells decisively, especially at
high me. DIF-log3's wins were a minority and clustered at specific (me, ios)
combinations. If most production transforms on this hardware would land in
"DIT-schedule wins" territory, the DIF executor would be a complex addition
that rarely fires. The engineering investment isn't justified by a 15-25%
gain on a small fraction of cells.

But this is one CPU class. On AVX-512 hosts (Sapphire Rapids, EPYC Zen4/5,
Xeon SPR+) the calculus may flip. AVX-512's wider register file (32 ZMM)
better accommodates DIF-log3's post-butterfly live-set, and DIF-log3's
twiddle-store overlap is more profitable when the SIMD width is larger.
On Apple Silicon (NEON, in-order issue) the answer might flip again. We
genuinely don't know.

**Second**, the codelet-level optimization story is already complete and
deployable as-is. Phase 2 captures all DIT-log3 wins. That's a real
performance contribution and a finished piece of work. Adding DIF-log3
would be marginal on the hardware we've measured. The honest decision
is: ship Phase 2, plan cross-architecture benchmarks, and revisit the DIF
question when there's data showing it pays off somewhere.

This is the article's main contribution as a personal-archive note: a
record of what we figured out so we don't have to re-derive it. If a future
bench run on Sapphire Rapids or M3 shows DIF-log3 dominating large swaths
of the wisdom grid, the design framework above is ready. If not, the
codelets we have remain useful as building blocks for some other application
(reverse FFT paths, partial transforms, etc.) and the work isn't wasted.

---

## Loose ends and cross-references

A few things worth recording for future reference:

- **The R=16 generator's DIF-log3 chain-sharing comment** (in
  `gen_radix16.py` near line 587) claims the DIT chain "applies verbatim"
  in DIF. This is true at the code-reuse level — the same cmul derivation
  pattern produces the right twiddle values for both — but the resulting
  codelets compute different functions because the application point
  (pre- vs post-butterfly) is different. The comment isn't wrong but it's
  easy to misread as "the codelets are equivalent." Worth annotating
  when we're back in that file.

- **R=32 has a custom column chain plan** (`_r32_dif_log3_column_plan`)
  designed for DIF-specific column-leg sharing. This was a clever
  optimization but is only useful within a DIF schedule. If we never
  build the DIF executor, the codelet sits unused in the registry —
  not deleted (it might be reactivated) but not invoked from any code
  path.

- **Backward FFT** in the existing executor is structurally DIF-shaped
  (hand-coded n1 + post-multiply), but doesn't currently call the
  generated `t1_*_dif` codelets. A natural first step for a DIF executor
  would be wiring the backward path to use the existing DIF codelets,
  which would (a) validate that the codelets are wired correctly into
  the executor before tackling the forward DIF path, and (b) potentially
  improve backward FFT performance for free as a side benefit.

- **The diagnostic programs** that established this finding live in
  `/tmp/dif_test/` and `/tmp/dit_dif_diagnose.c`. Worth preserving in the
  vectorfft_tune tree under `tools/diagnostics/` if a future investigation
  picks this up.

- **Cross-validation gap in `validate.c`**: the validator compares each
  codelet to its own protocol's reference, never cross-protocol. This is
  fine for what it's testing (codelet correctness within its protocol)
  but it means the DIT-log3 vs DIF-log3 mismatch went undetected by the
  bench's own validation. Worth adding a cross-protocol assertion if we
  ever resume this work — it would have caught the problem on day one.

---

## Summary

DIT-log3 and DIF-log3 codelets compute different linear transforms of
their inputs. They are correct as building blocks within their respective
schedules but cannot be substituted for each other within a schedule
without a transpose-bridge pass that costs roughly half a stage's work
and erases the optimization.

The proper architecture for capturing both types of wins is a dual-
schedule executor: a DIT path and a DIF path, each with its own per-stage
codelet autotuning. The planner picks DIT or DIF for the entire transform
based on whole-plan benchmarks, not per-stage codelet bench cells.

We're not building this. The Phase 2 integration ships DIT-log3 wins
across the portfolio. DIF-log3 capturing is deferred to a future decision
informed by cross-architecture bench data on AVX-512 and ARM hosts.

The codelet portfolio remains complete. The wisdom emit predicates remain
correct. Phase 3 is a future-direction reference, not an open task.
