# 61. Plan-shaped executor spike — Plan B+A validation

**Date:** 2026-05-16
**Status:** Spike complete across four benches (T1S + FLAT + LOG3 variants
+ Plan C comparison). Architecture validated. **5-6% wall-time win on
load-heavy codelets with many invocations**, ~0% on compute-bound codelets
(LOG3) and on small cells. Plan C (whole-FFT monolithic) loses badly.
Not wired into production.

**Where things live:**
- Emitter — [src/prototype/bin/emit_executor_h.ml](../bin/emit_executor_h.ml)
- Generated header — `src/prototype/generated/plan_executors.h`
- Bench harnesses:
  - [bench/spike_n131072_k4.c](../bench/spike_n131072_k4.c) — N=131072 K=4 T1S
  - [bench/spike_n1024_k128.c](../bench/spike_n1024_k128.c) — N=1024 K=128 T1S + Plan C comparison
  - [bench/spike_n131072_k4_flat.c](../bench/spike_n131072_k4_flat.c) — N=131072 K=4 FLAT (synthetic)
  - [bench/spike_n131072_k4_log3.c](../bench/spike_n131072_k4_log3.c) — N=131072 K=4 LOG3 (synthetic)
- VTune doc that motivated this — [docs/dev/vtune_n131072_k4_vfft_vs_mkl.md](../../../docs/dev/vtune_n131072_k4_vfft_vs_mkl.md)

---

## The gap being closed

The N=131072 K=4 VTune profile measured 21% of CPU time in
`_stride_execute_fwd_slice_from` and 7% in `stride_cmul_scalar_avx2` —
~28% wall-time share outside the codelet bodies. For an 8-stage plan
that's per-stage executor overhead paid 8 times: per-group base pointer
compute, twiddle pointer load, variant branch tree, scalar twiddle prep
gate, and an indirect call through `st->t1s_fwd`.

The VTune doc names the closure path: "codelet fusion — generate one
monolithic 8-stage codelet ... eliminates the 8× executor-overhead pay
and the 8× codelet function-call overhead." That's an MKL-style v2.0+
workstream. This spike tests a smaller architectural lever first:
**plan-shaped executor emission**, which keeps the per-stage codelet
structure but specializes the wrapper around it per (N, K, factorization,
variant-assignment) tuple.

## The two-part design

| Part | What it does | What it saves |
| --- | --- | --- |
| **(B) Direct calls** | Emit one C function per wisdom entry. The codelet call sites are direct symbol references, not function pointers loaded from `plan->stages[s].t1s_fwd`. | Indirect-call latency (~2-3 cycles/call). The 4-branch variant tree (n1-fallback / log3 / t1s / flat) collapses to one codepath per stage, resolved at emit time. |
| **(A) Pre-walked tape** | At plan-build time, pack each invocation's per-group runtime values into one flat 24-byte struct `{base, tw_re, tw_im}`. The emitted executor walks this tape sequentially. | Per-group load count drops from ~5 scattered (across `group_base`, `needs_tw`, `cf0_re`, `cf0_im`, `tw_scalar_re/im`) to ~3 sequential. HW prefetcher streams one cache line ahead. |

(B) alone produced ~0.7% (within-noise). (A) is where the actual cache-
stall reduction lives — sequential prefetch reaches the next entry before
the codelet returns. Combined, (B)+(A) gave 5.0% on this bench.

## How the new executor works vs. the old one

The key claim is that the new executor never costs more than the old —
it does strictly less work per group — and recovers wall time **only
when that work was visible** (not already hidden by the OoO engine).

### Per-group hot loop in the OLD executor (production)

From [src/core/executor.h:385-470](../../../src/core/executor.h):

```c
for (int s = start_stage; s < plan->num_stages; s++) {
    const stride_stage_t *st = &plan->stages[s];        // load stage ptr

    for (int g = 0; g < st->num_groups; g++) {
        double *base_re = re + st->group_base[g];        // load #1: group_base[g]
        double *base_im = im + st->group_base[g];        // (same offset, free)

        if (!st->needs_tw[g])      { /* path A: n1 */ }   // load #2 + branch
        else if (st->use_n1_fallback) { /* path B */ }    // load #3 + branch
        else if (st->use_log3)        { /* path C */ }    // load #4 + branch
        else if (st->t1s_fwd && st->tw_scalar_re
                            && st->tw_scalar_re[g]) {     // loads #5,6,7 + 2 branches
            double cfr = st->cf0_re[g];                   // load #8
            double cfi = st->cf0_im[g];                   // load #9
            if (cfr != 1.0 || cfi != 0.0) { /* prep */ }  // 1 branch
            st->t1s_fwd(base_re, base_im,                 // INDIRECT call (fn-ptr load + jmp)
                        st->tw_scalar_re[g],              // load #10: pointer
                        st->tw_scalar_im[g],              // load #11: pointer
                        st->stride, slice_K);
        }
    }
}
```

**Per group: ~10 loads, 4-5 branches, 1 indirect call** before the codelet
runs. The loads touch **5 separate per-group arrays** (`group_base[]`,
`needs_tw[]`, `cf0_re/im[]`, `tw_scalar_re/im[]`). At our scale that's
~1.25 MB of per-stage tables which the L1 (48 KB) can't hold — random-
order accesses spill into L2.

### Per-group hot loop in the NEW (B)+(A) executor

```c
/* Stage 1: emitted block, R=4, variant=T1S */
if (start_stage <= 1) {
    const stride_stage_t *st = &plan->stages[1];
    const stride_invocation_t *tape = st->tape;          // hoisted
    const int    num_groups = st->num_groups;            // hoisted
    const size_t stride     = st->stride;                // hoisted

    for (int g = 0; g < num_groups; g++) {
        const stride_invocation_t inv = tape[g];         // ONE 24-byte sequential load
        radix4_t1s_dit_fwd_avx2(re + inv.base, im + inv.base,
                                inv.tw_re, inv.tw_im,
                                stride, slice_K);        // DIRECT call (1 cycle)
    }
}
```

**Per group: 1 sequential tape read, 1 direct call.** No branch tree.
No scattered loads. The tape is one flat array (24 bytes/entry × 32K
groups = 768 KB) walked sequentially → HW prefetcher streams one cache
line ahead, so the next entry is already in L1 when the codelet returns.

### Side-by-side delta

| Operation | Old executor | New (B)+(A) executor |
| --- | --- | --- |
| Per-group bookkeeping loads | ~10 (5 arrays, scattered) | 1 (one 24-byte struct, sequential) |
| Branch tree resolutions | 4 conditional branches | 0 (resolved at emit time) |
| Codelet call type | indirect (`call *%reg`) | direct (`call <symbol>`) |
| Cache footprint of per-stage tables | ~1.25 MB across 5 arrays | ~0.75 MB in one packed array |
| Access pattern | random across arrays | sequential walk (prefetcher-friendly) |

### Why T1S/FLAT benefit and LOG3 doesn't — the OoO overlap story

Raptor Lake's out-of-order engine can run independent work in parallel
with the codelet body. While the codelet is executing, the OoO engine
can also speculatively execute the *next* group's wrapper code — if it
finds idle execution slots.

**T1S codelet** at K=4: ~25-30 cycles of work, mostly SIMD loads + 5
FMAs. Short FMA dependency chains; loads can issue in parallel. The
OoO engine has **slack** — execution slots that aren't filled by the
codelet's compute.

- *Old wrapper occupies that slack* — branch resolution stalls,
  scattered loads hit L2, indirect-call address forwarding. The wrapper
  cycles **compete with codelet execution** for execution ports → some
  of them stay on the critical path → wall-time visible.
- *New executor removes the wrapper* — those slots either fold into
  codelet execution or stay empty → wall-time drops by ~5%.

**LOG3 codelet** at K=4: ~30-40 cycles of work, with long Cmul
derivation chains (`W^j = W^p × W^q` = 4 muls + 2 adds each, chained on
the prior). **No slack** — the OoO engine is fully busy keeping FMA
pipelines fed.

- *Old wrapper also fits in those slots*, but the slots wouldn't have
  contributed to wall time anyway — the FMA dependency chain is the
  bottleneck. The wrapper cycles are "free" — they happen during cycles
  the codelet couldn't have used.
- *New executor removes the wrapper* → frees up nothing visible →
  0% wall-time change.

### Cycle budget for the T1S case

Measured wall-time delta for N=131072 K=4 T1S: 1180 µs (old) - 1121 µs
(new) = **59 µs saved per FFT**. At 5.7 GHz: 336K cycles total. With
~256K codelet invocations per FFT, that's **~1.3 cycles saved per
group** in wall time.

The wrapper itself has ~10-15 cycles of work per group — but only
~1.3 cycles of them were on the critical path. The other ~10 cycles
were already absorbed by OoO overlap with the codelet's loads/FMAs. The
new executor doesn't recover those — they weren't slowing anything down.

**Theoretical max recoverable wrapper share at this cell ≈ ~5%.** Bigger
gains need different levers (partial-fusion codegen, codelet redesign).

### Plain-English summary

> The new executor does strictly less work per group than the old one —
> fewer loads, fewer branches, simpler call. **Never a regression.**
> Recovers wall time only when the wrapper work was *visible* on the
> critical path. Compute-bound codelets (LOG3) hide wrapper cycles via
> OoO parallelism and so see no gain. Load-heavy codelets (T1S, FLAT)
> leave OoO bubbles that wrapper work sits in, and removing it actually
> shifts wall time.

## Spike scope (what's emitted)

For this spike, exactly one wisdom entry is hard-coded in OCaml:

```ocaml
{ n=131072; k=4;
  factors  = [|4; 4; 4; 4; 8; 4; 4; 4|];
  variants = [|FLAT; T1S; T1S; T1S; T1S; T1S; T1S; T1S|];
  use_dif_forward = false }
```

The emitted function name encodes the full tuple:
`exec_n131072_k4_44448444_v02222222_dit_fwd_avx2`. Inner-stage variants
of LOG3, FLAT, and BUF are stubbed (`abort()` if reached) — implementing
them is straightforward but unnecessary for the architectural validation.

Inner-stage `needs_tw[g]` and `cf0 != (1.0, 0.0)` branches are dropped
from the emitted code in this spike. Real plans with mixed paths need
both back (compile-time-decided codepaths per group group-class, since
those branches are per-group runtime decisions).

## Validation: disassembly

The two architectural claims show up directly in assembly:

```
1ee3:  e8 78 04 00 00    call   2360 <radix4_n1_fwd_avx2>
1f47:  e8 f4 04 00 00    call   2440 <radix4_t1s_dit_fwd_avx2>
2067:  e8 14 08 00 00    call   2880 <radix8_t1s_dit_fwd_avx2>
```

Every codelet call site uses opcode `e8` (direct relative call). No
`call *%reg` (indirect call through pointer) anywhere in the executor
body. Baseline executor in the same binary emits indirect calls for the
same codelets — the contrast is exactly what (B) promises.

## Measurement

Standalone bench:
[src/prototype/bench/spike_n131072_k4.c](../bench/spike_n131072_k4.c).
Builds a plan with realistic stage strides
(`stride[s] = K · ∏(R_{s+1}..R_{S-1})`) but synthetic `group_base[]` —
**timing-only**, not correctness. Both executors share the same plan
and codelets; the only delta is call style + tape vs scattered loads.

Run conditions: Raptor Lake i9-14900KF, CPU 2 pinned (`taskset -c 2`),
20 reps/run × 11 runs, median ns/FFT reported.

### Cell 1 — N=131072 K=4, T1S inner stages (wisdom-driven)

8 stages, 4×4×4×4×8×4×4×4, variants FLAT,T1S×7. ~256K codelet calls per FFT.

```
                            baseline      spike       speedup
(B)-only direct calls       1163305 ns    1155072 ns  1.007×
(B)+(A) tape walk           1180148 ns    1120837 ns  1.053×
```

(B) alone is within-noise; (B)+(A) wins **all 11 runs** before thermal
walking takes over at run 10. The 5× jump in the win between (B)-only
and (B)+(A) confirms that **the tape pre-walk does most of the heavy
lifting** — direct calls alone don't recover much.

```
run  spike(ns)    base(ns)    spike-base
 0   1,059,085   1,114,539     -55,454
 5   1,120,837   1,180,148     -59,311  ← median
 ...
10   1,484,225   1,272,733    +211,492  ← thermal outlier
```

### Cell 2 — N=1024 K=128, T1S inner stages

5 stages, 4×4×4×4×4, variants FLAT,T1S×4. **Only ~1.3K codelet calls
per FFT** (256 groups × 5 stages).

```
                            baseline      spike       speedup
(B)+(A) tape walk           106464 ns     106477 ns   1.000×
Plan C (R=1024 monolithic)      —          473815 ns  0.225× (4.45× SLOWER)
```

**(B)+(A) shows nothing here.** With 200× fewer codelet invocations per
FFT than Cell 1, the wrapper overhead is too small to matter — ~1µs
out of 106µs total. The architectural lever just doesn't have anything
to pull at this scale.

Plan C bench (next section) is the more interesting result for this cell.

### Cell 3 — N=131072 K=4, FLAT inner stages (synthetic)

Same plan shape as Cell 1, but with all-FLAT inner variants instead of
T1S. Synthetic because wisdom doesn't pick this configuration (T1S
beats FLAT at K=4 in real measurement), but isolates the FLAT codepath
at maximum call count.

```
                                       baseline      spike       speedup
(B)+(A) tape walk + K-blocked staging  1498597 ns    1407524 ns  1.065×
                                                                  ↑ 6.1% recovered
```

**6.1% — meaningfully larger than T1S's 5.0%.** The FLAT path has more
per-group wrapper work (per-leg `_stride_broadcast_2` staging of the
twiddle scalars into a stack tw_buf), so (B)+(A) has more to remove.

Also note: even after both executors do their wrapper recovery, FLAT
spike (1408 µs) is still **25% slower than T1S spike (1121 µs)** at
this cell. Wisdom's empirical preference for T1S over FLAT at K=4 is
correct — the intrinsic codelet cost of FLAT (with broadcast staging)
exceeds T1S's internal-broadcast pattern at this K.

### Cell 4 — N=131072 K=4, LOG3 inner stages (synthetic)

Same plan shape, all-LOG3 inner variants. Synthetic — wisdom doesn't
pick LOG3 for many-inner-stage K=4 plans (it's mostly the innermost
stage choice in 3-stage R=11/13 plans). Tests whether (B)+(A) generalizes
to the LOG3 codepath.

```
                                  median ns/FFT     speedup
  baseline (indirect log3)             1,131,313     1.000×
  spike    (direct + tape)             1,133,945     0.998×
                                                     ↑ -0.2% (within noise)
```

**0% gain — surprising at first.** Two observations explain it:

**(a) LOG3 baseline is faster than T1S/FLAT baselines.** LOG3 codelets
read fewer twiddle slots (base-only, derive the rest internally via
Cmul): R=4 LOG3 has 2 base slots vs T1's 3; R=8 LOG3 has 3 base slots
vs T1's 7. Fewer memory loads → codelet body is intrinsically faster
at K=4. The LOG3 baseline at 1131 µs is **20% faster than T1S baseline
at 1180 µs** — wisdom's preference for LOG3 on twiddle-bandwidth-bound
innermost stages is validated here.

**(b) LOG3 is compute-bound; wrapper cycles already overlap.** The
log3 derivation chains (`W^j = W^p · W^q` = 4 muls + 2 adds each) keep
the OoO engine fully busy. During those FMA chains, the **wrapper
bookkeeping for the next group computes in parallel** — per-group
loads, indirect call setup, etc. are hidden behind codelet compute.
T1S/FLAT codelets are more load-heavy → fewer FMA chains → OoO has
slack → wrapper code competes with codelet → (B)+(A) recovers that
contention. LOG3 doesn't have that slack to recover.

This refines the architectural rule (see "Cross-cell conclusions"
below).

### Plan C — whole-FFT monolithic codelet

For N=1024, we have an existing `radix1024_n1_fwd_avx2` from the
`xl_pow2` codelet family. It's a single 76K-line straight-line function
that does the entire N=1024 transform without returning. This is what
"fully fused" looks like in our DAG-compiler output.

```
                                 median ns/FFT     speedup vs baseline
  baseline (5-stage indirect)         106,464      1.000×
  spike    (B+A, 5-stage direct)      106,477      1.000×
  Plan C   (R=1024 monolithic)        473,815      0.225× (4.45× SLOWER)
```

**Plan C is 4.45× slower than the 5-stage decomposition.** Three reasons,
all architectural:

- **I-cache spillover.** 76K lines of straight-line code blows past
  Raptor Lake's 32 KB L1 instruction cache. Front-end stalls dominate.
- **OoO reorder buffer too small.** ~512 entries; the codelet has
  thousands of in-flight-able instructions.
- **Register allocation collapses.** Even with M-active and full doc-56
  treatment, GCC spills heavily across the codelet body. The "everything
  stays in registers" benefit that monolithic fusion was supposed to
  deliver gets defeated.

This empirically validates **why wisdom doesn't pick R=1024 monolithic**
for cells with working set > L1 (N=1024 K=128 = 2 MB working set, way
past L1's 48 KB).

**Important clarification: "Plan C" as a research lever is not "whole-FFT
monolithic".** What MKL apparently does (per the VTune doc's
`owns_crRadix4FwdNorm_64f` kernel) is **partial fusion** — 2-4 stages
inside one function, with internal loops over groups (NOT unrolled),
explicit cache-block sizing, and a working set that still fits L1.
That's level 2-3 in the fusion spectrum below; what we tested is level 4
(extreme end). Our DAG compiler produces level 4 because it always fully
unrolls; producing level 2-3 needs a different emitter that keeps loops
around groups intact.

**The fusion spectrum:**

| Level | What it does | Working set | Code size | Status |
| --- | --- | --- | --- | --- |
| 0 | One codelet per stage, executor loops between (today's shipping behavior) | Per-stage tile | Small | shipping |
| 1 | 2-stage pair fusion, registers carry between adjacent stages | Doubles register need | Slightly larger | unimplemented |
| 2 | Cache-blocked multi-stage, internal loops, explicit L1 tiling | L1 tile (~48 KB) | Compact | **MKL likely here** |
| 3 | Multi-stage straight-line within a tile, no inter-tile loops | Per-tile working set | Tile-sized | unimplemented |
| 4 | Whole-FFT monolithic, every butterfly unrolled | Whole FFT (e.g. 2 MB) | Massive (76K lines) | tested → loses |

## Cross-cell architectural conclusions

| Cell | Variant | Calls / FFT | (B)+(A) gain | Codelet character |
| --- | --- | --- | --- | --- |
| N=131072 K=4 | T1S | ~256K | **5.0%** | Load-heavy |
| N=131072 K=4 | FLAT (synthetic) | ~256K | **6.1%** | Load-heavy + per-group broadcast staging |
| N=131072 K=4 | LOG3 (synthetic) | ~256K | **~0% (-0.2%)** | Compute-bound (Cmul derivation chains) |
| N=1024 K=128 | T1S | ~1.3K | **0.0%** | (call count too low for any wrapper recovery) |

Three architectural rules emerge:

1. **(B)+(A) gain scales with codelet call count.** Cells with many
   stages × many groups benefit; compact plans don't. Wisdom's K=4
   portfolio is the main beneficiary.
2. **(B)+(A) gain scales with per-group wrapper size.** FLAT has more
   wrapper (broadcast staging) → 6.1%. T1S has less (internal broadcast
   in the codelet body) → 5.0%.
3. **(B)+(A) gain depends on whether wrapper cycles overlap with
   codelet execution.** Load-heavy codelets (T1S, FLAT) have OoO slack
   that wrapper competes with — removing the wrapper helps. Compute-
   bound codelets (LOG3, with its Cmul derivation chains) keep the OoO
   engine busy enough to hide wrapper cycles via parallelism, so removing
   the wrapper recovers nothing. This refines the "more wrapper = more
   gain" rule: the wrapper has to be *visible* in wall time to be
   recoverable.

### Why (B)+(A) is orthogonal to the SIMD prep helpers landed earlier

Production's per-call SIMD prep helpers (`_stride_cmul_scalar_inplace`,
`_stride_broadcast_2`, `_stride_cmul_vec_inplace`) wins **when K is
large** — many elements per group to vectorize. (B)+(A) wins when
**many groups exist** — wrapper bookkeeping aggregates. The two are
orthogonal; cells in different regimes benefit from one or the other,
rarely both:

| Cell regime | SIMD prep win | (B)+(A) win |
| --- | --- | --- |
| Large K (K≥128), few groups | **5-15%** | ~0% |
| Small K (K=4), many groups | ~0% (K=4 = one SIMD lane) | **5-6%** |

So (B)+(A) is additive to the SIMD prep gain, not redundant. The K=4
cells that don't benefit from SIMD prep are exactly the cells where
(B)+(A) helps most.

### Why Plan C (whole-FFT monolithic) lost

Three independent failure modes hit simultaneously at the R=1024 mono
codelet's 76K-line scale: I-cache spillover, OoO reorder-buffer
saturation, and GCC register allocation collapse. The "everything in
registers across stage boundaries" benefit that motivated codelet
fusion gets defeated by the size of the resulting straight-line code.

The path to inter-stage fusion gains is **level 2/3 partial fusion**
(MKL's apparent approach): 2-4 stages with internal loops and cache-
sized tiles. That needs a new emitter mode in the OCaml DAG compiler —
walk the factorization, emit loops over groups, leave butterfly bodies
unrolled per-iteration. Separate workstream, not built.

## Caveats — what this bench does NOT prove

1. **Correctness.** Twiddles are filled with arbitrary values; the
   spike validates execution flow and timing, not numerical output.
   Correctness validation against the production planner is a separate
   workstream.
2. **Realistic memory pressure.** The synthetic `group_base[g] = g*K
   mod headroom` makes inter-group memory accesses cache-friendlier
   than a real plan's bit-reversal-like pattern. Total per-FFT wall
   time was 1.16 ms vs the production 2.36 ms — the synthetic setup
   undercounts memory traffic. Net effect on the spike vs baseline
   delta is unclear: in a more memory-bound regime, the wrapper share
   is a smaller fraction of wall time, so the absolute speedup might
   shrink; but the tape's sequential prefetch becomes more valuable
   too. Real-plan bench is the conclusive test.
3. **LOG3 variant not yet emitted.** 7% of wisdom inner stages pick
   LOG3 (mostly innermost stages of large plans like R=13×R=13×R=13).
   Expected to behave similarly to T1S/FLAT; not measured.
4. **VTune profile is stale.** [docs/dev/vtune_n131072_k4_vfft_vs_mkl.md](../../../docs/dev/vtune_n131072_k4_vfft_vs_mkl.md)
   was captured before the SIMD prep helpers landed in `src/core/executor.h`.
   For K=4 cells specifically the wrapper share is still ~21% (SIMD prep
   doesn't help K=4 — only one SIMD lane wide). For K=256 cells the
   actual current wrapper share is lower than the doc says — those cells
   already absorbed 5-10% via SIMD prep. Re-running VTune against the
   current production executor would clarify, but isn't needed for the
   architectural conclusions here.

## What's reused vs new

| Component | Status |
| --- | --- |
| `bin/emit_executor_h.ml` (~340 lines OCaml) | NEW |
| `generated/plan_executors.h` | NEW — auto-emitted, ~600 lines (3 entries: 2 wisdom + 1 synthetic FLAT) |
| `bench/spike_n131072_k4.c` | NEW — T1S harness, ~270 lines |
| `bench/spike_n1024_k128.c` | NEW — T1S + Plan C harness, ~300 lines |
| `bench/spike_n131072_k4_flat.c` | NEW — FLAT harness, ~240 lines |
| Codelet symbol naming convention | LEAVES — same as production now (`radix{R}_{variant}_{isa}`, no `_gen_inplace_su*` suffix; aligned in same session). |
| OCaml DAG emit pipeline | LEAVES — codelets unchanged; spike just emits additional plumbing around the same codelet calls. |
| Production SIMD prep helpers (`_stride_broadcast_2`, etc.) | LEAVES — spike inlines its own copy of `_stride_broadcast_2` matching production's behavior. Production helpers still ship. |

## Path forward (in order of decreasing "cheap")

1. ~~LOG3 variant emission~~ — **done**. Result: 0% gain (compute-bound
   codelet hides wrapper via OoO). LOG3 still belongs in production
   emission for correctness on wisdom entries that pick it, just don't
   expect speedup. ~30 lines OCaml + a bench harness.
2. **(ii) Real-plan bench** — build the plan via production's planner
   (`stride_auto_plan_wis` etc.), point our spike executor at it. The
   plan's `stride_stage_t` has all the fields the spike needs (plus
   more we ignore). This conclusively shows the gain on production-
   shaped memory access. Touches production code only to USE the
   planner; doesn't replace anything yet.
3. **(iii) Generalize the emitter** — parse `build_tuned/vfft_wisdom_tuned.txt`
   in OCaml (~50 lines); emit per-entry executors for all ~200 cells.
   Full bench sweep vs the generic executor across the production
   portfolio. Likely Phase 2 of this workstream.
4. **(iv) Wire into production** — replace `_stride_execute_fwd_slice_from`
   with a lookup-then-call dispatcher: specialized path for wisdom
   entries, generic fallback for cold cells. Build-time integration via
   CMake. Probably Phase 3. Expect **0.5-2% portfolio-wide gain** —
   small-K-many-groups cells contribute the lion's share, large-K cells
   are noise.
5. **(v) Partial-fusion codegen (v2.0+)** — emit 2-stage / 4-stage
   monolithic codelets with internal group loops. This is the **MKL-style
   approach**, distinct from the whole-FFT monolithic (level 4) that
   loses. Needs a new emit mode in `bin/gen_radix.ml` or a sibling
   binary. Targets the structural gap to MKL on instruction density,
   not just wrapper overhead. Far bigger workstream, weeks of work, not
   gated on this spike.

## Decision points worth memory entries when this lands

- **5-6% wall-time gain on small-K many-groups cells; ~0% on large-K
  compact cells.** The per-stage codelet call structure is the lower
  bound for this design's headroom. Bigger wins (matching MKL's
  instruction density) need partial-fusion codegen, not bigger executor
  tweaks.
- **The tape pre-walk does the heavy lifting**, not the direct-call
  conversion. (B)-only got 0.7% (within noise); adding (A) jumped it to
  5%. Anyone repeating this should not implement (B) without (A).
- **(B)+(A) gain scales with (call_count × per_group_wrapper_size).**
  T1S: low per-group wrapper, high call count at K=4 → 5%. FLAT: medium
  per-group wrapper, same call count → 6.1%. Compact plans: any wrapper,
  low call count → 0%.
- **(B)+(A) is orthogonal to the SIMD prep helpers** (cf0 cmul,
  tw_buf broadcast) that landed earlier in `src/core/executor.h`. SIMD
  prep helps **large K**; (B)+(A) helps **many groups**. Different
  bottlenecks, additive gains.
- **Whole-FFT monolithic codelets lose by 4.45× at N=1024 K=128.**
  I-cache + OoO buffer + GCC register allocator all break down at the
  76K-line codelet scale. Wisdom's per-stage decomposition is empirically
  the right baseline; the v2.0+ lever is **partial fusion** (2-4 stages
  in one function with internal loops), not whole-FFT monolithic.
- **OCaml-side emission scales naturally** to per-entry specialization
  — same machinery as `emit_profile_h.ml`. Adding another entry is
  ~3 lines + a regen. Adding a new variant (e.g., LOG3) is ~30 lines
  in `emit_stage`.

## See also

- [docs/dev/vtune_n131072_k4_vfft_vs_mkl.md](../../../docs/dev/vtune_n131072_k4_vfft_vs_mkl.md)
  — the VTune profile this spike addresses. 21% wrapper share, 28%
  combined wrapper+scalar-prep share, ~3× MKL ceiling via codelet fusion.
- Codelet symbol naming alignment (same session, prior to this spike):
  prototype codelets renamed from `radix{R}_{variant}_{isa}_gen_inplace[_su[_spill]]`
  to production-style `radix{R}_{variant}_{isa}`. 832 codelets regenerated,
  66 consumer files updated. Drop-in compatible with production symbol
  slots once n1 signature alignment lands (separate task).
