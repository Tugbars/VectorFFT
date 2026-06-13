# M-project: spill-prevention strategy, costs, and mitigation paths

> **Note (May 2026): the central claim of this document — that
> M-project's pinning prevents high-value YMM/ZMM evictions — is
> incorrect.** Per-variable spill counts (re-measured with
> `gcc -fverbose-asm` annotations) top out at 2-3 reloads per
> variable, not the 18-20 the hot-slot data appeared to show; the
> 18-20 figure was slot recycling, not per-variable eviction
> hammering. A subsequent fence-decomposition test (strip
> `asm("regN")` clauses while keeping `asm volatile("" : "+v"(t))`
> fences) showed that the inline-asm fence — not the register pin —
> is the actual win mechanism on most cases. See
> `fence_pin_decomposition.md` for the corrected analysis and the
> resulting two-rule policy. The instruction-counting and
> diagnostic-methodology sections below remain correct as
> measurement data; the mechanism interpretation and the Option C
> proposal are superseded.

This document captures what M-project (manual register allocation in
`lib/regalloc.ml`) is designed to do, how its execution diverges from
its design intent on AVX2 vs AVX-512, and what we've measured for the
three classes of side effects it generates. It also sketches the
mitigation roadmap (Options C, D, and a small PRE_SPILL fix).

Companion to the calibration table in `lib/emit_c.ml:740-797` (the gate
that decides *whether* to enable M-project). This doc is about *what
M-project does* once enabled, and why the gate exists in the first
place.

All measurements below are from R=25 log3 on both AVX2 and AVX-512,
across the four direction × phase variants (DIT-Fwd, DIT-Bwd, DIF-Fwd,
DIF-Bwd). The R=25 log3 case is the cleanest test because it has
borderline pin density (~1.04 on AVX2), so M-project's failure modes
are observable without being swamped by either small-codelet noise or
catastrophic spill behavior.

## 1. What M-project does, in one paragraph

M-project pins each candidate vector value (Load, FMA result, Cmul
output, Add/Sub intermediate) to a specific physical YMM/ZMM register
via the `register __m256d t### asm("ymm15") = ...; asm volatile("" :
"+v"(t###))` pattern. The goal is to keep high-value variables (high
fanout, long live range) resident in the vector register file across
their full lifetime, so they don't get evicted to stack and reloaded
on the critical path. When two pinned values would collide on the same
physical register, M-project emits a `vmovapd` to route one of them
through a temporarily-free register. That move replaces a stack
round-trip (~5-7 ns at L1 latency) with a near-free reg-reg move
(~0-1 cycles under OoO rename).

## 2. The intent succeeds — measured

Counting, for each variant, how many times the *hottest* stack spill
slot is referenced. A heavily-reused slot indicates a high-value
variable being repeatedly spilled and reloaded — exactly what
M-project is designed to prevent.

| Direction | M-on max | M-off max | M-on top-5 spread | M-off top-5 spread |
|---|---:|---:|---|---|
| AVX-512 DIT-Fwd | 8 | **13** | [8,6,6,5,5] | [13,13,11,6,5] |
| AVX-512 DIT-Bwd | **5** | **18** | [5,5,5,5,5] | [18,6,5,4,4] |
| AVX-512 DIF-Bwd | 8 | **16** | [8,8,5,5,3] | [16,11,9,5,4] |
| AVX2 DIT-Fwd | 10 | **14** | [10,10,9,8,7] | [14,13,13,13,9] |
| AVX2 DIT-Bwd | 13 | **20** | [13,11,8,8,8] | [20,12,10,9,8] |
| AVX2 DIF-Fwd | 12 | **20** | [12,11,10,10,9] | [20,12,12,12,11] |
| AVX2 DIF-Bwd | 11 | **17** | [11,10,9,9,8] | [17,16,11,10,10] |

The pattern is consistent: under M-off, the hottest slot is hammered
13-20 times — one high-value variable repeatedly evicted from the
register file and reloaded. Under M-on, the hottest slot is referenced
5-13 times, and the spill distribution flattens. The total *count* of
spills is similar (sometimes M-on has more), but the *value* of what
gets spilled is much lower. M-on takes many cold one-shot spills
to keep the hot multi-use values in YMM/ZMM. That's the strategy
working as designed.

## 3. The three problems

M-project emits three classes of overhead while implementing its
strategy. We've measured each via static asm analysis on the eight
log3 R=25 variants (DIT/DIF × Fwd/Bwd × AVX-512/AVX2).

### 3.1 Greedy uniform pinning (mechanism 2)

The biggest cost. M-project pins every candidate without modeling its
own routing budget. Each pin contributes to register pressure during
its live range. When the cumulative pressure exceeds the physical
register file size, M-project's routing strategy degrades: it emits
`vmovapd` moves to route around collisions, but runs out of free
registers to route to, and GCC ends up spilling *additional* values to
make room.

Measured as the "OTHER" reg-to-reg category — moves not explained by
FMA encoding, broadcast fanout, or pre-spill staging.

| Direction | OTHER delta (M-on minus M-off) |
|---|---:|
| AVX-512 DIT-Fwd | +44 |
| AVX-512 DIT-Bwd | +17 |
| AVX-512 DIF-Fwd | -7 |
| AVX-512 DIF-Bwd | +52 |
| AVX2 DIT-Fwd | **+121** |
| AVX2 DIT-Bwd | **+85** |
| AVX2 DIF-Fwd | **+86** |
| AVX2 DIF-Bwd | **+96** |

On AVX-512 the OTHER delta is small (-7 to +52) and falls within the
free-rename budget. On AVX2 it's an order of magnitude larger (+85 to
+121) and consumes the 16-YMM file. The increase in total spill ops
(+15 to +50 on three of four AVX2 directions) is the visible
consequence: M-project's preservation strategy stops working when
there isn't enough register headroom to route around its own pins.

### 3.2 FMA encoding tax (mechanism 1)

Every pinned FMA result forces GCC into the 132 encoding (`dest =
multiplicand × accumulator + addend`). To set this up, GCC emits one
`vmovapd` per FMA to move the multiplicand into the pinned dest
register *before* the FMA executes. Without pinning, GCC freely picks
the 231 encoding (`dest = accumulator`) where no setup move is needed
because the accumulator is already live in the chosen dest.

Pattern in asm:

    vmovapd  %ymm5, %ymm14         # setup: move multiplicand to pinned dest
    vfmadd132pd %ymm6, %ymm9, %ymm14  # 132 encoding: dest = ymm14 * ymm6 + ymm9

vs M-off's natural choice:

    vfmadd231pd %ymm5, %ymm6, %ymm14  # 231 encoding: dest already in ymm14

Measured as the PRE_FMA reg-to-reg category — moves where the dest
register becomes an FMA destination within the next 3 instructions.

| Direction | PRE_FMA delta (M-on minus M-off) |
|---|---:|
| AVX-512 DIT-Fwd | +82 |
| AVX-512 DIT-Bwd | +64 |
| AVX-512 DIF-Fwd | +74 |
| AVX-512 DIF-Bwd | +46 |
| AVX2 DIT-Fwd | +68 |
| AVX2 DIT-Bwd | +55 |
| AVX2 DIF-Fwd | +59 |
| AVX2 DIF-Bwd | +62 |

The cost is consistent across ISAs (+46 to +82 per direction). It
scales with the number of pinned FMAs, not with the register file
size. Free under rename on modern uarch, but consumes uop dispatch
slots and the move-elimination budget. Visible on AVX2 because there
the move budget is already constrained by mechanism-2 pressure.

### 3.3 PRE_SPILL tax (M-project's spill staging)

When M-project's routing fails and a value goes to stack anyway, the
emission routes through the pinned register first:

    vmovapd  %ymm3, %ymm12          # route through pinned dest
    vmovapd  %ymm12, -64(%rsp)      # spill from pinned dest

M-off would have written directly:

    vmovapd  %ymm3, -64(%rsp)       # spill src directly

Measured as the PRE_SPILL category — reg-to-reg moves immediately
followed by a spill of the same dest.

| Direction | PRE_SPILL delta |
|---|---:|
| AVX-512 DIT-Fwd | +18 |
| AVX-512 DIT-Bwd | +37 |
| AVX-512 DIF-Fwd | +37 |
| AVX-512 DIF-Bwd | +18 |
| AVX2 DIT-Fwd | +34 |
| AVX2 DIT-Bwd | +32 |
| AVX2 DIF-Fwd | +32 |
| AVX2 DIF-Bwd | +35 |

A clean +18 to +37 across all variants. Smaller than the other two
mechanisms but cleanly identifiable and independently fixable.

## 4. The asymmetry: AVX-512 wins, AVX2 loses

Two ISA-specific factors flip the verdict.

### 4.1 Free-rename budget

Modern uarchs (Sapphire Rapids, Zen 4) handle `vmovapd reg, reg` via
the rename unit in zero-cycle moves. The rename unit has finite
bandwidth per cycle, but for typical FFT codelets the extra moves from
M-project sit well within the budget. On AVX-512, all three mechanisms
above contribute moves that the rename unit absorbs without execution
cost.

### 4.2 Register file size

The routing strategy in mechanism 2 requires a free physical register
to route into. With 32 ZMMs on AVX-512, the free pool is usually 8-15
registers wide at any point in the codelet — plenty for M-project's
~100-150 routing moves per direction. With 16 YMMs on AVX2, the free
pool is often 2-5 wide — and M-project needs 200+ moves on log3-scale
codelets. The shortage forces GCC to spill working values to stack to
free up routing registers, breaking the strategy.

### 4.3 Net effect

For R=25 log3:

| ISA | M-on perf vs M-off | Mechanism balance |
|---|---|---|
| AVX-512 | +20 to +32% | All three taxes paid; spill-prevention benefit dominates |
| AVX2 | -3 to -14% (3/4 dirs) | Mechanism 2 breaks down; +15-50 extra spills overwhelm benefit |

Same DAG, same pin density (~1.04), same three mechanisms — but
doubling the register file flips the trade-off completely. The
current gate captures this asymmetry via `vec_regs <= 16` plus the
pin-density threshold.

## 5. Mitigation roadmap

Three changes, ordered by yield-vs-effort. C and D are largely
independent; the PRE_SPILL fix is a small addition to whichever lands
first.

### 5.1 Option C — knapsack-style live-range-aware pinning

Target: mechanism 2 (greedy uniform pinning).

The current M-project decides what to pin without modeling its own
cost. Option C makes the decision *closed-loop*: estimate each
candidate's preservation value, estimate the routing budget, pin
greedily by value-to-cost ratio until budget is exhausted.

Concretely:

- **Value of pinning V** ≈ `fanout(V) × live_range_length(V)` — a
  rough estimate of how many stack round-trips M-project saves by
  keeping V pinned. Cmul outputs and base twiddles (high fanout,
  long live range) score high. Short-lived single-use FMA
  intermediates score low.

- **Cost of pinning V** = V's contribution to peak register pressure
  during its live range. Computed by tracking simultaneous live
  pinned values across the scheduled DAG.

- **Routing budget** = `vec_regs − reserved_count − peak_non_pin_live`.
  On AVX-512 this is ~28; on AVX2 ~10-12.

- **Decision rule**: sort candidates by value-to-cost ratio; admit in
  order; stop when cumulative cost reaches the budget.

Expected behavior:

- **AVX-512** — high budget admits nearly all candidates; matches
  current "pin everything" behavior; wins preserved.
- **AVX2 R=16 t1** — moderate budget; admits ~30-40 high-value
  candidates (twiddles, multi-use intermediates); current +22% win
  preserved.
- **AVX2 R=25 log3** — budget exceeded by candidate count; rejects
  ~20-30 low-value single-use FMA-result pins; OTHER count drops
  from +85-121 to ~+20-40; perf converges toward forced-M-off and
  may exceed it (because high-value pins are still preserved).
- **AVX2 R=11/R=13 primes** — budget heavily exceeded; rejects most
  candidates; effectively converges to current density-gate behavior
  but via a more principled mechanism.

Implementation cost: ~500-800 LOC across `regalloc.ml` (live-range
tracking + budget allocator) and `emit_c.ml` (gate the asm-pin
emission per candidate, not per codelet). Risk: the value metric is
heuristic; we need to validate it against measured behavior on a
diverse radix sweep before committing.

### 5.2 Option D — force FMA encoding

Target: mechanism 1 (FMA encoding tax).

C reduces the number of pinned FMAs but doesn't eliminate the
per-pin encoding tax. D addresses the encoding choice directly.

Two possible implementations:

- **D-emit**: when emitting a pinned FMA, write `__asm__ volatile
  ("vfmadd231pd %1, %2, %0" : "+v"(dst) : "v"(mul), "v"(acc))` to
  force the 231 encoding regardless of pinning. Bypasses GCC's
  encoding choice entirely. Risk: asm template needs to be ISA-aware
  (avx2 vs avx512 vs avx512+masks); needs careful validation
  across the radix sweep.

- **D-hint**: emit a non-pin-clobber inline asm fence after the FMA
  (`asm volatile("" : "+v"(t42))`) without the `asm("zmmX")` clause.
  Lets GCC choose the encoding freely while preserving the
  scheduling fence. Risk: loses the deterministic placement that
  the spill recipe relies on at phase boundaries.

D-emit is cleaner for the bulk of FMAs. D-hint could be used at
phase boundaries where deterministic placement matters less but the
fence is still needed.

Estimated saving: 46-82 moves per direction (the PRE_FMA delta).
Visible mostly on AVX2 once C has freed up move budget; on AVX-512
the saving is invisible because rename absorbs the moves anyway.

Implementation cost: ~200-400 LOC in `regalloc.ml` (FMA-aware
emit path). Risk: GCC's instruction selection is sensitive to
inline-asm constraint strings; small changes can cause regressions
elsewhere.

### 5.3 PRE_SPILL fix

Target: mechanism 3 (spill staging).

When M-project's allocator decides a value will be spilled (not
preserved through routing), the current emit path still routes it
through the pinned register first. The fix: when the allocator marks
a value as spill-bound, skip the routing step and emit the spill
directly.

Implementation cost: ~50 LOC in the regalloc emit path. Risk:
minimal — pure simplification. Saves 18-37 moves per direction
consistently across both ISAs.

This is the cheapest of the three fixes and probably the first to
land. Independent of C and D.

## 6. What we don't yet know

- **Whether the value metric `fanout × live_range` is right.** Other
  candidate metrics: `fanout` alone, `fanout × number_of_critical_path_uses`,
  or a learned weighting. Needs validation against measured spill
  hotness on a radix sweep before committing to C's implementation.

- **Whether the routing budget can be computed cheaply at emit time.**
  Peak non-pin live count requires a pre-pass over the scheduled DAG.
  The current SU+GH scheduler produces an ordered list; computing
  peak-live is O(N) over that list, should be tractable.

- **Whether D-emit's asm templates compose cleanly with the existing
  M-project pinning machinery.** GCC's inline-asm constraint handling
  has edge cases; we'd need to validate that mixing inline-asm FMAs
  with surrounding pinned asm fences doesn't produce surprising
  scheduling artifacts.

- **Whether the AVX-512 win on log3 (and elsewhere) actually comes
  from spill prevention, or from something else** (better scheduling
  shape from deterministic register choice, critical-path
  shortening). The slot-reuse data shows M-project *does* prevent
  hot-value evictions, but we haven't confirmed whether that's the
  main contributor to the measured 20-32% perf advantage or whether
  there's a secondary mechanism (improved dependency chains, fewer
  bypass-network hazards) doing significant work too. Worth running
  perf counters on M-on vs M-off to lock this down before investing
  in C.

## 7. Diagnostic methodology

For reproducibility, the measurements in this doc were collected by:

1. Generate M-on and M-off variants of each codelet:

       gen_radix.exe 25 --twiddled --log3 [direction-flags] --emit-c --isa <isa>
       VFFT_NO_REGALLOC=1 gen_radix.exe 25 --twiddled --log3 [direction-flags] --emit-c --isa <isa>

2. Compile to assembly with `gcc -O3 <arch-flags> -mfma -S`.

3. Extract function body (label to next function or `.cfi_endproc`)
   and classify each `vmovapd reg, reg` by surrounding context:

   - **PRE_FMA**: dest becomes FMA destination within 3 instructions
   - **POST_DEF**: src defined by arithmetic op within 3 prior instructions
   - **POST_BCAST**: src defined by vbroadcastsd within 3 prior
   - **PRE_SPILL**: dest spilled to stack within 2 instructions
   - **OTHER**: none of the above (mostly mechanism 2 + scheduling
     artifacts)

4. Spill hotness: count stack-relative memory operands by offset,
   report max and top-5 distribution. Heavily-reused offsets =
   high-value variables being evicted.

Analysis scripts and raw measurement output captured in
`/tmp/asm_diag/` (gitignored; regenerable from the steps above).

## 8. Summary

M-project's intent — keep high-value variables in YMM/ZMM, accept
cold-value spills as the cost — is sound and measured to work. The
current implementation pays three classes of overhead for that
benefit:

1. **+85-121 routing moves on AVX2** (mechanism 2, OTHER category)
2. **+46-82 FMA encoding-tax moves** (mechanism 1, PRE_FMA category)
3. **+18-37 spill-staging moves** (PRE_SPILL category)

On AVX-512 these costs disappear into the free-rename budget; the
benefit dominates and M-on wins by 20-50% universally. On AVX2 the
routing budget runs out, mechanism 2 starts *causing* spills instead
of preventing them, and M-on loses to M-off on dense-DAG codelets
(primes, log3).

The fix isn't to abandon pinning — the spill-prevention benefit is
real and measured. The fix is to (C) pin selectively based on value
and budget, (D) eliminate the per-pin FMA encoding tax, and clean up
the PRE_SPILL staging. Together these should close the AVX2 gap
without sacrificing the AVX-512 wins.
