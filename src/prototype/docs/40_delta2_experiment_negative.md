# 40. Δ2 Experiment: Rematerialization Markers — Negative Result

## What was tested

Doc 39's Phase A diagnostic identified `-flive-range-shrinkage`'s win
as concentrated in Pass 2 (74-79% reduction in Pass 2 GCC-scratch usage).
The recipe-level encoding hypothesis: emit reloads at *every use site*
in Pass 2 instead of the current "single named reload + multiple
references" pattern. If GCC treats each inline load as a fresh
independent value, it might allow shorter live ranges — the loaded
value can die between uses, freeing the register.

## Implementation

Added an environment-controlled toggle (`VFFT_RELOAD_EACH_USE=1`) to
`emit_c.ml`. When active, Pass 2 emission changes from:

```c
const __m512d t100 = _mm512_loadu_pd(&spill_re[5]);  // named reload, once
const __m512d t200 = _mm512_add_pd(t100, t50);        // reference t100
const __m512d t201 = _mm512_mul_pd(t100, t60);        // reference t100 again
```

to:

```c
const __m512d t200 = _mm512_add_pd(_mm512_loadu_pd(&spill_re[5]), t50);  // inline load 1
const __m512d t201 = _mm512_mul_pd(_mm512_loadu_pd(&spill_re[5]), t60);  // inline load 2
```

Mechanically: added a `spilled_load_inline: (int -> string option) option`
callback to `render_node_def`. When set, the renderer emits an inline
load expression for spilled tag references instead of `t<tag>`. In
Pass 2 emission, the callback returns `Some("_mm512_loadu_pd(&spill_re[N])")`
for spilled tags, `None` otherwise. The named-reload emission path is
skipped when the toggle is active.

Verified that the generated C source contains the expected pattern:
zero named reloads (`const __m512d tN = _mm512_loadu_pd(&spill_re[...])`),
1018 inline `_mm512_loadu_pd(&spill_re[...])` calls embedded in
expressions for R=256 AVX-512.

## Result

Zero effect across all tested configurations:

```
R=128  AVX-512  gcc-13         baseline 659   Δ2 659    (delta 0)
R=128  AVX-512  gcc-11+shrink  baseline 595   Δ2 595    (delta 0)
R=256  AVX-512  gcc-13         baseline 2040  Δ2 2040   (delta 0)
R=256  AVX-512  gcc-11+shrink  baseline 1298  Δ2 1298   (delta 0)
R=512  AVX-512  gcc-13         baseline 5216  Δ2 5216   (delta 0)
R=512  AVX-512  gcc-11+shrink  baseline 3699  Δ2 3699   (delta 0)
```

The asm output is byte-for-byte identical between the two C inputs.
Same total body length (13406 at R=256), same stack op count, same
memory loads (2014), same FMA count (620).

## Why

GCC's CSE (Common Subexpression Elimination) is too aggressive to
preserve the source-level distinction. Two `_mm512_loadu_pd(&spill_re[5])`
calls read the same memory address, and GCC's optimizer can prove
they yield the same value (the stack array isn't aliased thanks to
`__restrict__` on the function's pointer parameters, and the loads
don't cross intervening stores to that slot). It merges them into a
single load with a single register destination — exactly equivalent to
the named-reload form.

This is in fact good compiler behavior — GCC is correctly identifying
that the source-level "duplication" carries no extra semantic content.
What we wanted was for GCC to *not* prove the equivalence, but it
proves it easily here.

## Why the inverse approach also fails

The complementary idea — *force* GCC to materialize each load by
making the loads non-equivalent — would require:

- `volatile` qualifiers on the spill arrays: forces every load to
  actually go to memory, but this is far worse than the baseline
  because it removes ALL register-resident caching, not just selected.
- Memory clobbers via `asm volatile("" ::: "memory")`: similar issue;
  too coarse.
- Different pointer values for each load: not possible since the slot
  index is fixed.

There is no clean C-level mechanism to say "treat these two equivalent
loads as independent for register allocation purposes but otherwise
optimize freely." GCC's view of the code is the IR after parsing, and
the optimizer's invariants are stronger than what source structure
can encode.

## What this confirms

The gcc-11 + `-flive-range-shrinkage` win is purely an allocator-internal
heuristic decision: which values to keep in registers vs spill to stack
at each program point. This is governed by IRA (Integrated Register
Allocator) and LRA (Local Register Allocator) C++ code inside GCC,
making decisions based on live-range length, register-class preferences,
and pressure heuristics that the source code cannot influence through
ordinary C constructs.

Source-level hints don't survive optimization. The Δ2 hypothesis is
closed.

## Implications

Three implications for the project:

The first is that any source-level encoding of allocator behavior is
unlikely to work. The recipe operates at a layer that GCC's optimizer
sees through. Even sophisticated transformations (Δ3 expression rewriting,
Δ1 micro-clustering) face the same risk: GCC may simply re-optimize
back to its preferred allocation pattern, especially at `-O3`.

The second is that the practical path is to **accept the compiler
dependency**. gcc-11 + `-flive-range-shrinkage` ships a 29% stack op
reduction at R=512 AVX-512 and a 5-8% real runtime improvement (per
doc 38's container-bench, which is directional). gcc-11 is stable
and available; pinning it in CI is straightforward.

The third is that further leverage on the spill controller specifically
requires going below source level — either:
- Inline assembly with explicit register allocation (multi-week project,
  uncertain win, hard to maintain across µarchs)
- PGO / profile-guided compilation to influence allocator heuristics
  (modest gain, infrastructure cost)
- Modify GCC source to expose more knobs (out of scope)

None of these has a compelling cost/benefit ratio relative to other
work on the project.

## Recommendation

Phase A (doc 39) characterized what gcc-11+shrink does. Δ2 confirmed
that the simplest recipe-level encoding doesn't work. Δ1 (Pass 2
micro-clustering) would face the same CSE risk: GCC's optimizer can
collapse micro-cluster spills back into its preferred allocation if
it sees them as redundant.

**Ship the compiler dependency.** Document `gcc-11 -flive-range-shrinkage`
as the production toolchain for AVX-512 codelets. Move on to higher-
leverage work:

- Cross-stage codelet fusion (attacking executor overhead from VTune)
- DPDK / kernel bypass networking
- Tesla founder's RL subagent prep
- ICEEMDAN extension

The spill controller has had its leverage extracted. The remaining
gains require either compiler internals work (multi-week, uncertain)
or inline asm (multi-week, maintenance burden). Neither has the ROI
of work in those other areas.

## Files state at end of experiment

- `lib/emit_c.ml`: reverted to baseline. The Δ2 toggle code was added,
  tested, found ineffective, removed.
- All other files: unchanged.
- Stack op counts: R=64/128/256/512 AVX-512 match baselines exactly
  (255/659/2040/5216).
- Prime correctness: 56/56 PASS.

No code from this session ships. The findings stand as docs 38, 39, 40.
