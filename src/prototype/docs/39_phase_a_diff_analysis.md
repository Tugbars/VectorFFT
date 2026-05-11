# 39. Phase A: Reverse-Engineering gcc-11+shrink

## Setup

Doc 38 found gcc-11 + `-flive-range-shrinkage` produces 29-36% fewer stack ops
than gcc-13 default. Phase A's goal: characterize *what gcc-11+shrink does
differently* by diff-inspecting the two asm outputs. If a clean pattern
emerges, we could encode it in the recipe and become compiler-independent.

The comparison: same generated C source (R=256 and R=512 AVX-512 twiddled
in-place codelets), same `-O3` plus ISA flags, different
compiler/flag combination. The diff isolates pure
allocator/scheduler behavior.

## Structural finding: two separate stack regions

The stack frame in both compilers has two distinct categories of
64-byte ZMM slots:

**Recipe region** — the contiguous block where the C source's `spill_re[256]`
and `spill_im[256]` arrays live. Both compilers use exactly 512 of these
slots (one per recipe-declared spill). Position in stack: one large run of
~496-500 consecutive 64-byte-spaced offsets, then a small ~12-slot tail
that's the second array. Identical layout across both compilers.

**Scratch region** — additional ZMM slots beyond the recipe arrays. These
are GCC-allocated for its own register-pressure relief. Each scratch slot
is typically reused across the body for multiple short-lived spills.

This is the locus of the difference:

```
                  gcc-13           gcc-11+shrink    saving
R=256 AVX-512:    89 slots, 1016    38 slots, 288   -51 slots, -728 events
                  events            events           (-72%)
R=512 AVX-512:    653 slots, 3116   342 slots, 1627 -311 slots, -1489 events
                  events            events           (-48%)
```

The recipe-declared spills are nearly identical between compilers. The
entire win comes from gcc-11+shrink allocating ~half the scratch slots
and triggering far fewer scratch events.

## Pass 1 vs Pass 2 distribution

Where in the body does scratch usage happen?

```
                   Pass 1 scratch events       Pass 2 scratch events
                   ─────────────────────       ─────────────────────
R=256 gcc-13:      161                          855
R=256 gcc-11+shrink: 108                        180   ← -79% Pass 2 reduction
R=512 gcc-13:      1543                         1573
R=512 gcc-11+shrink: 1225                       402   ← -74% Pass 2 reduction
```

`-flive-range-shrinkage` is overwhelmingly a Pass 2 effect. Pass 1
scratch shrinks modestly (R=512: -21%); Pass 2 scratch collapses
dramatically (R=512: -74%). The flag's biggest leverage is exactly
where doc 35's diagnostic showed the highest GCC-extra concentration
at R=256 (and a substantial portion at R=512).

## Scratch slot reuse pattern

Each scratch slot holds *different values at different times*. Average
stores per slot:

```
R=256 gcc-13:      6.6  (each slot reused ~7 times)
R=256 gcc-11+shrink: 4.0
R=512 gcc-13:      2.7
R=512 gcc-11+shrink: 2.4
```

This is *spill churn*: insufficient registers force GCC to constantly
shuffle values through memory. Compute value A → spill A to slot X →
use A → done → compute value B → spill B to slot X → use B → ...

The slot at offset 36744 in gcc-13's R=256 output, for instance, gets
written 7 times across the body, each time with a different
computational intermediate. None of these values relate to the recipe's
declared spill markers — they're purely GCC's runtime allocation choices.

`-flive-range-shrinkage` reduces this churn by making fewer values
contend for the limited register file at any one time. With shorter
live ranges, each register frees up sooner, reducing the need to spill.

## Why the diff is hard to encode in the recipe

The recipe operates at the **IR level** — placing spill markers based
on the symbolic DFT structure (Pass 1 outputs at sub-DFT boundaries).
gcc-11+shrink's behavior happens at the **allocator level** — choosing,
during physical register assignment, which live ranges to materialize
in which registers and which to push to stack.

These layers don't have a clean correspondence. Specifically:

The recipe can't predict which specific intermediate values will
become high-pressure within a Pass 2 sub-DFT-N1 cluster. Pressure
depends on:
- Order in which compute is scheduled (set by `pass2_ordered`,
  which already uses SU within cluster)
- GCC's instruction scheduling and reordering after register allocation
- µarch-specific port pressure (avoidance of port-5 contention etc.)

The recipe sees only the dependency DAG and slot annotations. It
doesn't see physical register assignment outcomes.

## What the recipe *could* do

Three concrete designs emerge from this analysis. None is a slam dunk,
but all are at least plausible.

**Design Δ1: Pass 2 micro-clustering with intra-cluster spills.**

Currently Pass 2's emission for each cluster emits all the cluster's
nodes in SU-ordered sequence inside one `{ }` scope. Peak live within
the cluster can exceed the register file when many intermediate values
overlap.

The change: break each Pass 2 sub-DFT-N1 cluster into smaller "stages"
of ~K nodes each, with explicit recipe-mandated spill markers at stage
boundaries. Each stage's intermediates die at the stage boundary; only
the stage's outputs (a small number of values) survive into the next
stage.

Predicted impact: should mimic the live-range shrinkage effect for the
Pass 2 case where the recipe currently has no visibility. Implementation
cost: 1-2 days. Risk: GCC may still add scratch on top because its
allocator runs independently of the recipe's stage hints.

**Design Δ2: Rematerialization markers.**

For Pass 2 spilled values (those reloaded from spill_re/spill_im),
instead of emitting a single reload at first use, emit reloads at
*each* use site. GCC's CSE may or may not collapse these.

If CSE collapses them: no effect (we lose nothing).
If CSE preserves them: each reload becomes a true L1 load, but the
value doesn't need to stay live in a register between uses. This
trades extra memory ops for reduced register pressure — the inverse
of normal allocation.

Implementation: trivial in emit_c (just remove the "reloaded set"
tracking and always emit a reload before use). Risk: zero gain if
GCC's CSE is aggressive; possible regression on AVX2 where 16 YMM
already needs every reload to count.

**Design Δ3: Pass 2 expression rewriting.**

In algsimp.ml, detect high-fanout intermediate nodes in Pass 2 and
de-duplicate them — turning shared intermediates into per-use
inlined expressions. This trades extra arithmetic for shorter live
ranges. Doc 32 dealt with the opposite direction (de-duplicate to
reduce arithmetic); this would partially undo that.

Implementation: tricky because we'd need to identify *which* CSEs
to break. Easy to over-correct and regress total work. Risk: hard
to bound the search space; could spend significant effort for unclear
gain.

## The honest assessment

Phase A's diagnostic was successful: we know *what* gcc-11+shrink
does (~60% reduction in Pass 2 scratch slot usage), *where* the win
concentrates (Pass 2), and *why* it's hard to replicate (allocator-level
decisions invisible to the IR).

But it also revealed: the win is in compiler-internal allocator
heuristics that the recipe can only approximate. Of the three designs:

- Δ1 (micro-clustering): plausible, 1-2 day investment, uncertain whether
  GCC will still re-spill on top
- Δ2 (rematerialization): trivial to implement, 0-15% expected gain,
  small risk of AVX2 regression  
- Δ3 (rewriting): unclear cost/benefit, hard to bound

Cost-benefit ranking:

The cheapest is Δ2 (rematerialization markers): a few hours to
implement, gives us a data point on whether GCC respects emit-side
hints. If it works on gcc-13, we get a portable improvement. If not,
we know gcc-11+shrink's gain is purely allocator-internal.

If Δ2 doesn't help, the practical answer is **ship the compiler
dependency**. gcc-11+shrink is a stable, available toolchain.
Document it; move on to higher-leverage work.

If Δ2 does help, it confirms emit-side hints have signal — and Δ1
becomes the next move.

## Open questions

The first is what happens with newer/older clang versions. We have
only clang-18. clang-15/16/17 may behave differently — possibly
better (closer to gcc-11) or possibly worse. Worth one test if
available.

The second is whether the gcc-11→gcc-12 register allocator change
is documented anywhere. A bisection across gcc-11.1 / 11.2 / 11.3 /
11.4 / 12.0 would pinpoint the exact patch. With identified patch,
we could understand the regression and possibly file an upstream
issue (or PGO-train a config that recovers it).

The third is whether Sapphire Rapids / Zen 4 microarchitecture
changes the equation. AVX-512 port pressure differs across these
µarchs; live-range shrinkage may help more or less. The container
CPU is Skylake-ish; real-hardware measurement is needed.

## Recommended next step

Implement Δ2 (rematerialization markers) as a one-day prototype.
Measure on R=256/R=512 AVX-512 against gcc-13 baseline. If it
recovers ≥10% of the gcc-11+shrink advantage, encode it; if not,
ship the compiler dependency.

This is a small enough experiment to fit in one session and
produces decisive directional data either way.
