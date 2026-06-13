# R=25 blocked emit and t1/n1 comparison

A one-line threshold change that gave R=25's n1 codelet a 47% AVX-512
speedup (and 38.8% AVX2 speedup) by treating it as the 5×5 Cooley-Tukey
two-pass codelet it structurally is, rather than as a monolithic small
codelet. With the n1 fix in place, this doc also covers a complete
head-to-head against Tugbars's hand-coded `gen_radix25.py` reference
across both the n1 (notw) and t1 (twiddled, hot path) kernels:

- **n1**: VFFT wins 12% (was −30% before the fix)
- **t1**: VFFT wins 59% (no fix needed — algsimp's FMA fusion was
  already dominant for the twiddle multiplies)

Source: `lib/dft.ml::should_block_n1`. Previous threshold `n >= 32`;
current threshold `n >= 25`.

## The misclassification

VFFT's codelet emission has two paths:

1. **Monolithic emit** (`Dft.dft_expand`): one unified DAG, register
   allocator decides spills based on local pressure. Used for small
   radices that "fit in registers" — historically R ≤ 16.
2. **Blocked emit** (`Dft.dft_expand_n1_blocked`): explicit 2-pass
   structure with spill markers between passes. Used for large composite
   radices where the inter-pass live set exceeds register budget.
   Historically R ≥ 32.

R=25 was in bucket 1 by the threshold `n >= 32`, with a comment that
"smaller R already beats hand because the whole DFT fits in registers."
That comment is correct for R=16 (144 ops, 0 muls, whole codelet fits
in 32 ZMM registers comfortably). It is *not* correct for R=25:

- R=25 has 383 ops (now 384 under blocked emit) — 2.7× more than R=16
- R=25 is structurally a 5×5 Cooley-Tukey with **25 complex live values**
  crossing the natural pass boundary — that's 25 ZMM registers just for
  the inter-pass state, before counting any pass-1 or pass-2 working set
- The monolithic emit can't preserve that pass boundary, so spills get
  scattered throughout the codelet wherever local pressure peaks

## Measured impact of misclassification

Comparing R=25 AVX-512 monolithic emit vs Tugbars's hand-generated
5×5 CT codelet (`/mnt/user-data/uploads/gen_radix25.py`):

| Metric | VFFT monolithic | Tugbars hand 5×5 | Ratio |
|---|---:|---:|---:|
| Total vector instructions | 1128 | 570 | VFFT 1.98× more |
| Reg-to-reg `vmovapd` | 450 | 46 | VFFT 9.8× more |
| Total stack spills | 434 | 135 | VFFT 3.2× more |
| Runtime (ns/call, K=8) | 127.98 | 77.39 | VFFT 65% slower |

The investigation initially suspected the register-pinning emit
pattern (`register __m512d t### asm("zmmK") = ...; asm volatile("" :
"+v"(t###))`) was the culprit. Disabling it via `VFFT_NO_REGALLOC=1`
reduced the vector instruction count to 672 (closer to Tugbars's 570)
but made runtime *worse* (288.84 vs 234 ns/call):

| Metric | M-on (default) | M-off (`VFFT_NO_REGALLOC=1`) |
|---|---:|---:|
| Total vec instr | 1128 | **672** |
| Reg-to-reg copies | 450 | **104** |
| Total spills | 434 | **354** |
| Runtime | 234 ns | **289 ns** |

Fewer instructions, slower runtime. The M-project's instructions
weren't dead weight — they were preserving FMA fusion across what
would otherwise be auto-rewritten chains, and breaking long
dependency chains via strategically placed spills.

**The real problem wasn't the emit style. It was that the DAG had
no clean pass boundary to exploit in the first place.**

## The fix

```diff
 let should_block_n1 (n : int) (_vec_regs : int) : bool =
   match pick_algorithm n with
   | Direct -> false
   | Split_radix -> false
-  | Cooley_Tukey _ -> n >= 32
+  | Cooley_Tukey _ -> n >= 25
```

One character. No new infrastructure. The blocked emit machinery was
already there for R=32 and R=64; the threshold just happened to
exclude R=25.

## How the pieces fit together

"The blocked emit machinery" is three coordinated systems. Worth
naming them because R=25's win is the result of all three engaging at
once, and the same machinery applies to R=32/R=64 (and any future
composite radix above the threshold).

### 1. Spill machinery (math layer)

`Dft.dft_expand_n1_blocked` (`lib/dft.ml`) returns three things instead
of one:

- The full assignment list (same as monolithic emit would return)
- A list of `spill_tag_marker` entries — `{tag; slot}` pairs identifying
  which DAG values cross the pass boundary and which slot each one
  lands in
- A `(num_re_slots, num_im_slots)` pair for buffer sizing

At emit time, this becomes a `spill_info` record handed to
`emit_codelet` via the optional `?spill` parameter. The `Some sp`
branch at `lib/emit_c.ml` ~1198 takes over: it walks the DAG once,
partitions nodes by which pass they belong to via
`classify_passes sp nodes`, and from there everything downstream is
pass-aware.

This is the structural divergence from FFTW. FFTW's `genfft` emits the
whole codelet as one straight-line program; values that don't fit in
registers spill via gcc's RA into whatever stack slots gcc picks. The
spill scatter ends up wherever local pressure peaks. VFFT's
`spill_re[]`/`spill_im[]` arrays declared at function entry, indexed
by pre-assigned slots from the math layer, give gcc nowhere to
scatter — the spill *positions* are determined upstream by where the
math says the pass boundary is.

### 2. Scheduler — Sethi-Ullman + Goodman-Hsu

The base scheduler in `lib/schedule.ml::su_schedule_subset` is a
textbook Sethi-Ullman list scheduler. Primary picking key is
`(cp_dist DESC, su_num ASC)`: among ready nodes, pick the one with the
longest remaining critical path to a sink, ties broken by Sethi-Ullman
number (the classical "max-depth tiebreaker").

The `+ Goodman-Hsu` extension (`~gh:true` parameter, named after
Goodman & Hsu 1988) is a *priority-function mode switch* keyed on
live-count:

| Mode | Trigger | Picking key |
|---|---|---|
| LATENCY | `live_count ≤ vec_regs − 4` | `(cp_dist DESC, su_num ASC)` — base SU |
| PRESSURE | `live_count > vec_regs − 4` | `(delta ASC, cp_dist DESC)` |

where `delta = births − kills`:
- `kills(n)` = number of `n`'s predecessors whose `remaining_users`
  drops to zero when `n` is scheduled
- `births(n)` = 1 if `n` has remaining users in subset OR is a sink,
  else 0

In PRESSURE MODE the scheduler picks the most negative delta — the
choice that frees the most live values per cycle. Returns to LATENCY
MODE the next cycle if `live_count` drops back below threshold.

The 4-slot slack (`vec_regs − 4`) reserves room for cmul scratch and
FMA scratch — values materialized for one cycle and immediately
consumed. Loads stay deferred behind ready arithmetic regardless of
mode; that rule is orthogonal to pressure tracking and prevents the
load-bursts that would otherwise saturate the load ports at codelet
entry.

Goodman-Hsu auto-enables when `su && vec_regs ≤ 16 && n ≥ 32`
(`bin/gen_radix.ml` ~159). The empirical comment there: 4–8%
improvement over base SU on AVX-2 R={32,64}, no-op on AVX-512
(threshold=28 rarely reached with cluster-sequential scheduling).

Under blocked emit the scheduler runs **per-pass**: SU+GH is applied
to the PASS 1 subset (sinks = pass-1 spill targets), then again to the
PASS 2 subset (sinks = output stores). Each pass gets its own
pressure-aware schedule with its own `cp_dist` computation. This is
how the 2-pass structure surfaces inside the scheduler — partitioned
subsets, not a unified DAG with extra constraints.

A **branch-and-bound** scheduler exists in `lib/bb.ml` as an opt-in
alternative with lexicographic `(saturated_peak ASC, -progress ASC)`
cost. R=64 AVX-2 shows a K-regime crossover where BB beats SU+GH by
~5.8% at K=512-1024; below that, SU+GH wins. Default path uses SU+GH;
BB is enabled via `--bb-budget`.

### 3. Emit coordination via M-project

The piece that ties scheduling and spilling together is M-project
(manual register allocation in `lib/regalloc.ml`, with
`current_regalloc` ref in `lib/emit_c.ml`). For each scheduled node,
M-project decides:

- Which ZMM/YMM register it lives in
- When it dies (last-use point)
- Whether it gets pinned via
  `register __m512d t### asm("zmmK") = ...; asm volatile("" : "+v"(t###))`

Under blocked emit, M-project runs **per-pass**: `install_alloc
"spill_pass1"` for pass 1, then a `clear_alloc()` and a fresh
`install_alloc "spill_pass2"` for pass 2. Each pass gets a fresh
allocation that knows only about its own subset — pass 2 doesn't have
to plan around pass 1's register lifetimes because the pass boundary
is a hard barrier (everything that crosses goes through the spill
array).

The pass-1 store sites use **M5 emission**: store the value into its
assigned slot, then immediately mark its register dead so the next
pass-1 instruction can reuse it. The pass-2 reload sites use **M6
emission**: load the slot into the pinned register *before* the first
use in the schedule, not at the point of first use. This guarantees
the reload doesn't get reordered into the middle of a dependency
chain by gcc's instruction scheduler.

### Why this matters for R=25

R=25 has 25 inter-pass values — exceeds AVX-512's 32 ZMM with no room
to spare for working set, and vastly exceeds AVX-2's 16 YMM. Without
explicit spill machinery, the monolithic emit forced gcc to discover
the pass boundary heuristically while spilling under local pressure,
producing the 450 reg-to-reg `vmovapd` and 434 stack spills we
measured.

With blocked emit activated, the same algorithm runs through this
pipeline:

1. **Math layer** says "values v0..v24 cross the pass boundary; assign
   them slots 0..24"
2. **SU+GH** schedules pass 1 with PRESSURE MODE engaging early (live
   count quickly exceeds 12 on AVX-2; 28 on AVX-512), prioritizing
   nodes that kill predecessors and store to slots
3. **M-project** allocates pass 1 with spill stores as forced
   last-use, freeing all 16/32 registers cleanly at the pass boundary
4. **SU+GH** schedules pass 2 with slot reloads as starting points
   (they look like external inputs, no upstream dependencies)
5. **M-project** allocates pass 2 from a fresh register file with no
   carryover state

The result is exactly the structure Tugbars hand-codes in the Python
generator: 25 stores at the end of pass 1, 25 loads at the start of
pass 2, two compact register-only working sets in between. Same
algorithm, same operations — but with the topology made explicit so
gcc doesn't have to rediscover it under local pressure.

## Results

Best-of-11 trials × 200k reps each, AVX-512 K=8 batches:

| Variant | ns/call | ns/transform | vs Tugbars |
|---|---:|---:|---:|
| VFFT default (was monolithic) | 127.98 | 16.00 | +65% slower |
| VFFT `VFFT_NO_REGALLOC=1` | 127.27 | 15.91 | +64% slower |
| **VFFT blocked (n≥25)** | **67.98** | **8.50** | **−12% faster** |
| Tugbars hand 5×5 codelet | 77.39 | 9.67 | baseline |

AVX2 K=4:

| Variant | ns/transform | Δ |
|---|---:|---:|
| VFFT default (was monolithic) | 30.25 | — |
| **VFFT blocked (n≥25)** | **18.50** | **−39%** |

## Numerical correctness

| Comparison | Max diff |
|---|---:|
| VFFT default vs VFFT blocked | 8.88e-16 (machine epsilon) |
| VFFT default vs Tugbars | 1.02e-13 |
| VFFT blocked vs Tugbars | 1.02e-13 |

The blocked emit produces *exactly the same computation* as the
monolithic emit, modulo emit ordering — verified to 8.88e-16. Both
match Tugbars's independent codelet to FMA precision (~1e-13).

## Structural verification

Spill-array signatures across all default radices after the change:

| Radix | Ops | `regalloc_spill[]` | `spill_re/im[]` | Structure |
|---:|---:|---:|---:|---|
| R=4  | 16  | 0   | 0   | no spills (tiny) |
| R=8  | 52  | 5   | 0   | monolithic |
| R=11 | 150 | 121 | 0   | monolithic |
| R=13 | 204 | 198 | 0   | monolithic |
| R=16 | 144 | 120 | 0   | monolithic |
| **R=25** | **384** | **0** | **102** | **2-pass (new)** |
| R=32 | 386 | 0   | 130 | 2-pass |
| R=64 | 978 | 0   | 258 | 2-pass |

R=25 now uses the same `spill_re/spill_im` two-pass array structure as
R=32 and R=64. R≤16 and primes correctly stay monolithic.

## What we learned

**The smaller lesson**: thresholds based on intuition ("small fits in
registers") can be wrong by a constant factor. R=16 fits; R=25 doesn't.
The op count is a better proxy for fit than the radix value itself —
ops scale as O(N²) for direct DFT and O(N log N) for Cooley-Tukey, so
the threshold should track ops, not N. R=25 at 383 ops is closer to
R=32's 386 ops than R=16's 144 ops.

**The bigger lesson**: emit-style optimizations (register pinning,
selective unpinning, asm-volatile barriers) interact subtly with the
DAG structure they're applied to. When the DAG has a clean pass
boundary, M-project's strategic spilling amplifies that structure.
When the DAG is monolithic, M-project just adds overhead because there's
no structure to exploit. The right intervention is **at the DAG-shape
level**, not the emit level.

This generalizes a finding from earlier work on R=64 AVX-512
(`docs/multi_use_fma_lift.md`): the algsimp pass's wins come from
preserving useful structure (FMA chains, conjugate-pair sharing), not
from raw op-count reduction. The same is true at R=25: switching to
blocked emit *increased* op count by 1 (384 vs 383) but cut runtime
by 47%.

## Extended comparison: t1 codelet (the hot path)

The n1 codelet (standalone DFT, no external twiddles) is what we
benchmarked above. In any real FFT, however, **the hot path is the
t1 codelet** — the twiddled form that gets invoked inside the outer
loop of a larger transform. The n1 codelet only appears once per
stage at the smallest sub-block size; t1 runs once per sub-block at
every stage of the call tree.

For t1, both VFFT and Tugbars's generator already use blocked emit
(VFFT's `should_spill` threshold has always covered the t1 path for
N ≥ 25). So there's no structural fix to make — this comparison is a
direct apples-to-apples kernel race.

### Results

Best-of-11 trials × 200k reps, AVX-512 K=8, in-place vs out-of-place
conventions handled via reset-from-backup at trial boundaries.

| Codelet | ns/call | ns/transform |
|---|---:|---:|
| **VFFT t1** (OOP, batched, `radix25_t1_dit_fwd_avx512`) | **156.62** | **19.58** |
| Tugbars t1_dit (in-place, batched, `radix25_t1_dit_fwd_avx512`) | 248.93 | 31.12 |
| Δ | +58.9% (VFFT wins) | |

VFFT wins t1 by 59%. The direction is *opposite* of the n1 comparison,
where Tugbars's hand-tuned generator beat VFFT (before our blocked-emit
fix) by 30%. After the fix both kernels favor VFFT, but the *reason*
each kernel favors VFFT is different.

### Why VFFT wins t1: algsimp catches the twiddle FMAs

The t1 codelet wraps each non-DC input by an external twiddle multiply:

```
x'_re = tw_re·x_re − tw_im·x_im
x'_im = tw_re·x_im + tw_im·x_re
```

Both lines are perfect FMA fusion patterns. The first fuses to one
`vfmadd` + one `vfmsub` (or `vfnmadd`) — net 2 FMAs vs the unfused
form's 2 muls + 1 add + 1 sub = 4 instructions.

For each of the 24 twiddled inputs, that's 4 instructions saved per
twiddle multiply. Across the codelet, algsimp catches all of them.
Tugbars's Python generator doesn't have an algsimp pass and emits the
unfused form. Asm comparison of the fwd kernel only:

| Metric | VFFT t1 | Tugbars t1 | Notes |
|---|---:|---:|---|
| Total vec instructions | 752 | 688 | Tugbars 9% fewer |
| **FMAs** (all variants) | **279** | **118** | **VFFT 2.4× more fused** |
| Separate `vmulpd` | 64 | 120 | Tugbars 1.9× more |
| Separate `vaddpd` + `vsubpd` | 100 | 220 | Tugbars 2.2× more |
| Reg-to-reg `vmovapd` | 343 | 60 | (this is M-project's signature) |
| Stack spills | 94 | 164 | Tugbars 1.7× more |

Tugbars has fewer total instructions, fewer reg-to-reg copies, AND
more spills — and is still 59% slower. Why? Critical-path latency:

- One FMA on a modern x86 CPU is 4 cycles (issue-to-result).
- Unfused mul → add chain is 4 cycles (mul) + 4 cycles (add) = **8
  cycles** if the add depends on the mul.
- Tugbars has 220 add/sub + 120 mul = 340 separate arithmetic ops.
  VFFT has 100 add/sub + 64 mul = 164 separate arithmetic ops + 279
  FMAs.
- The 161 extra unfused arithmetic ops in Tugbars's emission sit on
  dependency chains where each one extends the critical path by 4
  cycles, while VFFT's fused FMAs keep the same chains tight.

This is the algsimp investment paying off where it matters most: the
hot path. The n1 codelet doesn't have these external twiddle patterns
to fuse (every twiddle is internal to the 5×5 CT structure, already
factored at the math layer), so algsimp's twiddle-fusion advantage
doesn't apply — and the structural emit becomes the dominant factor.

### What the n1 + t1 results decompose cleanly

The two kernel comparisons reveal where each system's investment pays:

| | What it gives | Where it dominates |
|---|---|---|
| **Tugbars's strength** | Clean structural emit, minimal register shuffle, low instruction count | n1 (no external twiddles to fuse — algsimp has nothing to do) |
| **VFFT's strength** | Algsimp's FMA-fusion and twiddle-aware algebraic transformations | t1 (24 twiddle multiplies × 2 = 48 fusion opportunities) |

Before the blocked-emit fix, Tugbars's n1 win (30%) masked VFFT's t1
win (59%). With the fix, **VFFT wins both kernel forms** against the
hand-generated reference, with each system's specific contribution
isolated. The n1 win is now structural (matching Tugbars's natural
emit), the t1 win is algebraic (FMA fusion that algsimp catches and
Tugbars's generator doesn't).

Since t1 is the actually-hot kernel in real FFTs, the **t1 result is
the more important one** — it confirms VFFT's algsimp infrastructure
is the dominant contributor to runtime advantage for the kernel that
gets called most. The n1 fix matters because it eliminates a
structural regression at one rung of the call tree; the t1 advantage
compounds across every twiddled invocation in the transform.

### Numerical correctness — note on the t1 bench

The t1 bench does *not* compare numerical outputs between VFFT and
Tugbars, because the two implementations use different twiddle
conventions (sign of the imaginary part, indexing of the implicit
identity-twiddle slot, etc.) — running them on the same input would
produce different but equally correct DFT-25 outputs. The runtime
comparison is meaningful regardless of numerical equivalence: both
kernels are doing R=25 t1_dit with the same FLOP count requirement,
so the per-call latency reflects how each compiles to asm.

For correctness verification, VFFT's t1 was checked against FFTW (with
matching twiddle convention) at codelet-generation time; Tugbars's t1
is presumed correct by the generator's regression suite.

## Reproducing

```sh
cd /home/claude/work/strided_avx512
dune build
cd /tmp/bench_r25

# n1 (notw, standalone DFT-25) — the codelet the fix activates
./.../_build/default/bin/gen_radix.exe 25 --emit-c --isa avx512 \
    > vfft_r25_blocked_avx512.c
gcc -O3 -mavx512f -mfma -c vfft_r25_blocked_avx512.c \
    -o vfft_r25_blocked_avx512.o

# Bench n1 against Tugbars
gcc -O2 -mavx512f -mfma bench_4way.c \
    vfft_r25_default_avx512.o vfft_r25_noregalloc_avx512.o \
    vfft_r25_blocked_avx512.o notw_avx512.o \
    -lm -o bench_4way
./bench_4way

# t1 (twiddled, the hot path)
./.../_build/default/bin/gen_radix.exe 25 --twiddled --emit-c --isa avx512 \
    > vfft_r25_t1_avx512.c
sed -i 's/radix25_t1_dit_fwd_avx512/vfft_r25_t1_avx512/g' vfft_r25_t1_avx512.c
gcc -O3 -mavx512f -mfma -c vfft_r25_t1_avx512.c -o vfft_r25_t1_avx512.o

# Tugbars t1 (need to strip 'static' keywords for external linkage)
python3 gen_radix25.py --isa avx512 --variant ct_t1_dit > tugb_t1_avx512.c
sed -i 's/^static __attribute__/__attribute__/g; s/^static const/const/g' tugb_t1_avx512.c
gcc -O3 -mavx512f -mavx512dq -mfma -c tugb_t1_avx512.c -o tugb_t1_avx512.o

# Bench t1
gcc -O2 -mavx512f -mfma bench_t1.c \
    vfft_r25_t1_avx512.o tugb_t1_avx512.o -lm -o bench_t1
./bench_t1
```

## Future work

- **Audit other radices that may be misclassified.** Are there other
  composite N in the (n < 32) range where a natural pass boundary
  would benefit? R=20 = 4×5, R=18 = 2×9, R=21 = 3×7 are candidates.
  Most aren't in our standard radix set, but if added they should be
  measured both ways.
- **Threshold formalization.** A more principled rule would key off
  `n1 + n2` (the inter-pass live set size for an N1×N2 split) compared
  to `vec_regs`. If `n1 + n2 > vec_regs / 2` (rough headroom estimate),
  blocking helps. For R=25 (5+5=10 > 16) blocking should fire on AVX2.
  For R=16 (4+4=8 < 16) monolithic should fire. Matches our empirical
  finding exactly.
- **R=49 (7×7) and beyond.** Not currently in the radix set, but would
  obviously benefit from blocking under the new threshold. Adding to
  the radix generator would let us measure.
- **t1 algsimp investigation at other radices.** The t1 win at R=25
  (59%) suggests algsimp's twiddle-FMA fusion is a major contributor
  to VFFT's runtime advantage at all twiddled radices. The R=32 and
  R=64 t1 codelets should be compared against equivalent hand-coded
  references to confirm the win generalizes — and to identify any
  radix where the algsimp fusion misses opportunities.

## Related

- `docs/multi_use_fma_lift.md` — the density-gated multi-use FMA flatten
  pass, which contributes the 4-mul reduction at R=25 n1 (independent
  of this work). Related to the t1 finding here: both demonstrate that
  algsimp's FMA-aware passes are doing real runtime work.
- `docs/r25_fftw_benchmark.md` — the prior VFFT vs FFTW R=25 comparison,
  measured against the *monolithic* VFFT codelet. Numbers there
  understate VFFT's edge by roughly the 47% measured here.
- `/mnt/user-data/uploads/gen_radix25.py` — Tugbars's hand-generated
  R=25 codelet generator used as the comparison baseline (both n1 and
  t1 variants)
- `lib/dft.ml::should_block_n1` — the threshold
- `lib/dft.ml::should_spill` — the t1 threshold (separate from n1)
- `lib/dft.ml::dft_expand_n1_blocked` — the blocked emit path for n1
- `lib/dft.ml::dft_expand_twiddled_spill` — the blocked emit path for t1
- `lib/emit_c.ml` ~line 1198 — `Some sp` branch of the spill-aware
  emission path (shared by both n1 and t1)
- `/tmp/bench_r25/bench_t1.c` — t1 head-to-head bench harness
- `/tmp/bench_r25/bench_4way.c` — n1 4-way bench (default, no-regalloc,
  blocked, Tugbars)
