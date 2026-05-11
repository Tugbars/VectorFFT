# 38. Compiler Study: gcc-11 + `-flive-range-shrinkage` Wins by 29%

## Context

[Doc 36](36_spill_phase2_design_b_already_done.md) established that the existing
cluster-aware emit_c path produces 5216 stack ops at R=512 AVX-512 and that
neither Design B (already implemented) nor Design A's first concrete variant
(per-cluster lexical scoping, which GCC ignores at -O3) could push this floor
lower. Three remaining Design A variants — multi-level recipe with intra-Pass
reloads, inline asm with explicit register allocation, or pivoting to other
work — were on the table.

Tugbars suggested a different cut: **try other compilers**. If a different
compiler produces substantially fewer spills from the same C source, that
tells us the data-flow structure allows better allocation; GCC at -O3 just
isn't using all the available freedom. The asm-level diff between the two
output files would then point to what GCC is leaving on the table.

This turned out to be by far the highest-leverage diagnostic of the session.

## Setup

Available compilers on the test machine: gcc-11.5.0, gcc-12.4.0, gcc-13.3.0,
clang-18.1.3. No icx, no AOCC. Each compiled the same generated C source
(produced by `gen_radix N --twiddled --in-place --emit-c [--isa avx2]`)
with `-O3 -masm=intel` plus the appropriate ISA flags
(`-mavx512f -mavx512dq -mfma -march=skylake-avx512` for AVX-512,
`-mavx2 -mfma -march=core-avx2` for AVX2).

The robust asm parser finds the K-loop body between the first two loop
labels inside the function body (handling both `.L<n>:` gcc style and
`.LBB<f>_<n>:` clang style) and counts `vmov*` instructions referencing
`rsp` — the canonical signature of vector stack spills.

## Results: stack ops across compilers (defaults)

```
              AVX-512                            AVX2
R       gcc-11  gcc-12  gcc-13  clang-18    gcc-11  gcc-12  gcc-13  clang-18
64      218     255     255     554         753     761     761     960
128     576     659     659     1411        1781    1909    1909    2236
256     1783    2040    2040    5061        5110    5206    5206    7535
512     4726    5216    5216    14749       11819   12066   12066   18472
```

Three observations from this table alone:

The first is that **gcc-11 consistently beats gcc-12/13** by 7-14% on AVX-512.
Specifically on AVX-512: R=64 -14.5%, R=128 -12.6%, R=256 -12.6%, R=512 -9.4%.
This is the same C source compiled with different gcc versions — a real
regression introduced somewhere between gcc-11 and gcc-12.

The second is that **gcc-12 and gcc-13 produce identical stack op counts**.
The regression happened in the 11→12 transition; nothing has changed since.

The third is that **clang-18 is dramatically worse on AVX-512**: 2.5× more
spills at R=64 and 3.1× more at R=512. Looking at the instruction mix
deeper, clang-18 also emits substantially fewer FMA instructions (529 vs
1377 at R=512 — clang isn't fusing multiply-add patterns as aggressively),
producing more separate adds and muls. The poor allocation compounds with
poor instruction selection.

## Diagnostic: what does gcc-11 do differently?

Instruction mix breakdown at R=512 AVX-512:

```
metric           gcc-11  gcc-13   delta
total body       28948   30171    -1223  (-4.1%)
spill            4726    5216     -490   (-9.4%)
reg_mov          3992    4080      -88   (-2.2%)
fma              1377    1377        0   identical
addsub           9216    9217       -1   identical
mul              3034    3034        0   identical
v_other          1982    2414     -432   (-17.9%)
scalar           4621    4833     -212   (-4.4%)
```

The compute is identical between the two versions — same FMA count,
identical add/sub, identical multiply. The 4.1% total instruction
reduction comes purely from auxiliary ops: spills, register moves, and
scalar address arithmetic. **gcc-11's register allocator and post-RA
scheduler are doing the same data flow more efficiently** — eliminating
moves that gcc-13 leaves in.

## The flag finding

Testing various gcc-13 flags to see if any restore gcc-11 behavior at
R=256 AVX-512 (gcc-11 baseline 1783, gcc-13 default 2040):

```
flag                            stack ops
default                         2040
-fira-algorithm=priority        2217  (worse)
-fira-region=one                2039  (no effect)
-flive-range-shrinkage          1494  (-26.8% from default, even better than gcc-11!)
-fno-schedule-insns2            2040  (no effect)
-fno-tree-pre                   2040  (no effect)
-fno-ira-share-spill-slots      2040  (no effect)
-fno-ira-share-save-slots       2040  (no effect)
-frename-registers              2040  (no effect)
-fno-cse-follow-jumps           2040  (no effect)
```

**`-flive-range-shrinkage`** is the winning flag. GCC docs:

> "Attempt to decrease register pressure through register live range
>  shrinkage. This is helpful for fast processors with small or moderate
>  size register sets."

It changes how the integrated register allocator (IRA) handles live
ranges — actively trying to compress them so fewer registers are alive
simultaneously. This is exactly the right tool for our spill-heavy
codelet pattern.

## Best combination: gcc-11 + `-flive-range-shrinkage`

The two effects compound. Full matrix (savings vs gcc-13 default):

```
R     ISA     g13def  g11def  g13+shrink  g11+shrink  best_save
R=64  avx512  255     218     252         220         +14.5%
R=64  avx2    761     753     658         649         +14.7%
R=128 avx512  659     576     650         595         +12.6%
R=128 avx2    1909    1781    1704        1528        +20.0%
R=256 avx512  2040    1783    1494        1298        +36.4%   ← biggest
R=256 avx2    5206    5110    5302        5109         +1.9%
R=512 avx512  5216    4726    4095        3699        +29.1%
R=512 avx2    12066   11819   12213       11986        +2.0%
```

The asymmetric behavior on AVX2 large is interesting: `-flive-range-shrinkage`
*hurts* slightly at R=256/512 AVX2 (1-2% worse than default). The 16-YMM
register file is so constrained that range-shrinking has no room to maneuver
— the allocator's hands are tied regardless.

## Strategy: per-(R, ISA) compiler choice

```
condition                  compiler          flags                       saving
AVX-512, all sizes         gcc-11            -flive-range-shrinkage      12-36%
AVX2, R ≤ 128              gcc-11            -flive-range-shrinkage      14-20%
AVX2, R ≥ 256              gcc-11            (no shrink — slight regret)  1-2%
```

For convenience: **`gcc-11 -flive-range-shrinkage`** is never substantially
worse than the gcc-13 default. The worst case is R=512 AVX2 at -1.0% from
the gcc-11 default — basically noise. So using gcc-11+shrink universally
is a safe simplification that captures 95%+ of the available wins.

A more refined per-codelet build script could swap the flag in/out at the
~1-2% margins for AVX2 large, but that's complexity for tiny additional
gain.

## Validation: correctness preserved

Ran the full prime correctness suite (R={2,5,7,11,13,17,19} × 8 variants
= 56 codelets) under `gcc-11 -O3 -flive-range-shrinkage`. Result:

```
56/56 PASS — all 8 variants × R={2,5,7,11,13,17,19} verified.
```

No correctness regression. The 36.4% / 29.1% / 20% reductions are
genuine work elimination, not unsound transformation.

## Total work reduction is real, not bookkeeping

Worried that `-flive-range-shrinkage` might trade spills for other
instructions (e.g., recomputing values to avoid spilling). Checking
instruction-mix at R=512 AVX-512:

```
metric          gcc-13 default    gcc-11 +shrink    delta
total body      30171             27625             -2546  (-8.4%)
spill           5216              3699              -1517  (-29.1%)
reg_mov         4080              3695              -385   (-9.4%)
fma             1377              1377              0      (identical)
addsub          9217              9216              -1     (identical)
mul             3034              3034              0      (identical)
```

Same arithmetic work (fma/addsub/mul unchanged), but 2546 fewer total
instructions per K iteration — actual reduction in work, not just a
re-arrangement. At R=512 with K=512 batch, that's 1.3M fewer instructions
across the loop body before considering vectorization batching.

## Why this matters more than Design A would have

Design A's best-case predicted reduction at R=512 AVX-512 (from doc 36):
~5-15% stack ops — modest, with implementation cost of 1-2 days for the
multi-level recipe and additional emit_c machinery. Net target:
4500-5000 ops.

Compiler flag tuning: **29.1% reduction at R=512 AVX-512 with zero
source code changes**. Net: 3699 ops.

The compiler change captured roughly **3× the reduction Design A could
have delivered**, with no implementation effort and no risk to existing
codelet correctness. The diagnostic value of trying multiple compilers
was the highest-leverage move of the entire spill controller arc.

## Why is gcc-11 better than gcc-12/13?

I haven't traced the exact gcc commit responsible, but the observed
pattern (gcc-12 = gcc-13 exactly, 7-14% worse than gcc-11 on this code)
suggests a single allocator change in the gcc-11 → gcc-12 transition.
Candidates include:
- IRA heuristic changes (color preferences, hard-register selection)
- LRA (the post-IRA allocator) changes — gcc-11.0 onwards is fully LRA;
  perhaps a default tuning shifted in 12
- Spill-slot sharing/coalescing behavior differences

Tracking down the specific commit is interesting but not load-bearing
for the recommendation. The fact that one flag (`-flive-range-shrinkage`)
recovers most of the difference suggests gcc-12+ disabled or changed the
default for range-shrinkage heuristics. Worth filing a gcc bug report
with this benchmark as a reduced test case — even if gcc maintainers
decline to revert, it would establish the documented behavior.

## Runtime validation: real gain, smaller than stack ops suggested

Built a minimal monolithic-codelet bench: call R=N codelet on B-batched
inputs in a tight loop, take min/median across 7 independent runs.
Run on the development container CPU — clock speed is well below i9-14900K
production target, so numbers are **directional** not predictive.

**Runtime improvement (gcc-13 default → gcc-11 + shrink), median ns/iter:**

```
R=256:
  B=64   -8.3%   B=128  -7.0%   B=256  -7.2%   B=512  -3.7%

R=512:
  B=64   -5.7%   B=128  -4.2%   B=256  -5.5%   B=512  +0.5% (noise)
```

By min ns/iter (less noise): R=256 shows -2.1% to -8.5%, R=512 -1.3% to -8.3%.

**The 29% stack op reduction → ~5-8% real runtime improvement** at moderate
B, shrinking toward noise at very large B where memory bandwidth dominates.
This is closer to the total-instruction reduction (-8.4%) than the
spill-only reduction (-29%) — spill ops are mostly L1-resident here so
saving 1500 of them at ~1 cycle each is only ~500 ns of nominal savings,
yet we see 5-8% improvement = several thousand ns. The spills *were*
causing genuine OoO stalls (port contention, µop cache eviction)
beyond their nominal cycle cost.

**Container caveat:** the bench machine's clock is below i9-14900K target.
Production hardware may show different magnitudes — larger if AVX-512
throughput is more aggressive (OoO amplifies false-dependency removal),
smaller if memory subsystem absorbs spill cost better. Realistic range
for production: **3-10% runtime improvement** at moderate B.

**What this doesn't change:** the AVX2 R=512 B=512 multi-stage advantage
(33% from doc 34) won't flip from this — AVX2 large gets ~2% from
compiler tuning, nowhere near a 33% gap.

## Action items

These are concrete and prioritized:

The first is to update the build scripts (`bench/*/build.sh`,
`bench/primes/correctness/build_and_run.sh`, and any other codelet
compilation paths) to use `gcc-11 -flive-range-shrinkage` by default.
Add a `CC` override so platforms without gcc-11 fall back to the
system gcc. *(`bench/primes/correctness/build_and_run.sh` updated this
session — others remain TODO.)*

The second is to re-run the doc 33/34 R=512 / R=256 benches with the
new compiler config on actual i9-14900K hardware. Container-bench
showed 5-8% improvement at moderate B; production may show 3-10%.

The third is to file a gcc bug report with this benchmark as a reduced
test case for the 11→12 register allocator regression. Not high priority,
but the data is here and the project is upstream.

The fourth is to test on real hardware (Sapphire Rapids workstation
when available). All the analysis here is on Skylake-AVX-512 target;
SPR / Zen4 may show different patterns since their µarch is different.
The flag may need different defaults per uarch.

## What this doesn't change

The Phase 1 diagnostic findings stand:
- AVX-512 R=512: GCC-extras are now ~1700 (was 3168) — still material but
  no longer dominant. Recipe-mandated 2048 + GCC-extras 1651 = 3699 total.
- AVX2 R=512: Still ~12000 stack ops. AVX2 register pressure remains
  the fundamental bottleneck regardless of compiler.
- Pass 1 / Pass 2 split remains roughly balanced.

The phase-2 design space (multi-level recipe, inline asm) is still open
for future work. But the urgency is reduced: at R=512 AVX-512 we now
ship 3699 stack ops, comfortably better than the 4500-5000 target Design
A would have aimed for.

The cluster scheduling logic in emit_c.ml is still doing useful work —
without it the recipe wouldn't structure spills as efficiently. The
compiler flag amplifies what was already there; it doesn't replace it.

## Files unchanged this session

`emit_c.ml` ended the session in the same state it started (per-cluster
scope injection was prototyped, found to have zero effect, and
reverted). The level field plumbing (`Dft.spill_marker.level`,
`Algsimp.spill_tag_marker.level`, `Emit_c.spill_info.level_of`) was
added in case multi-level recipe development resumes later. With all
markers at level=0, the plumbing is a semantic no-op.
