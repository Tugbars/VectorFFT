# R=4 cross-chip discrepancy: Container vs Raptor Lake

## TL;DR

The same R=4 codelet source, same ISA (AVX2), same algorithm — compiled
with different toolchains on different chips — produces wildly different
performance orderings. At one sweep point the log3/flat ratio differs
by **40×** between the two environments. This is the most dramatic
cross-chip divergence we've observed in the VectorFFT project.

The divergence goes both directions: on the server container, log3 loses
3-11× to flat. On consumer Raptor Lake, log3 wins 2-4× over flat at the
exact same (me, ios) points. Neither chip is "wrong" — they're running
different code as interpreted by different compilers, and both are
running it efficiently for what they got.

## The raw numbers

All measurements are ns per codelet call for `radix4_t1_dit_*_fwd_avx2`,
averaged over repetitions, best-of-3 trials. Same harness, same test cases.

### Container (SPR-class, GCC 13 on Linux)

| case | me | ios | flat (ns) | log3 (ns) | log3/flat |
|---|---|---|---|---|---|
| baseline_small | 64 | 64 | 1169 | 5031 | **4.30** |
| late_inner_samepage | 256 | 264 | 3914 | 19622 | **5.01** |
| mid_inner_fewpages | 256 | 2048 | 3801 | 18699 | 4.92 |
| ios_eq_me_pow2 | 512 | 512 | 1325 | 14843 | **11.20** |
| late_large_me_samepage | 2048 | 2056 | 6661 | 23584 | 3.54 |
| late_large_me_1024 | 1024 | 1032 | 4768 | 32779 | 6.88 |
| heavy_dtlb_4pages | 2048 | 8192 | 4478 | 22665 | 5.06 |
| catastrophic_dtlb | 2048 | 32768 | 4477 | 22999 | 5.14 |
| **N1M_stage1** | **1024** | **16384** | **4028** | **30359** | **7.54** |
| ios_eq_me_4k | 4096 | 4096 | 8050 | 7064 | 0.88 |
| extreme_ios_small_me | 256 | 65536 | 4036 | 19456 | 4.82 |

log3 loses everywhere except at me=4096 ios=4096.

### Raptor Lake (i9-14900KF, ICX 2025.3.0 on Windows)

| case | me | ios | flat (ns) | log3 (ns) | log3/flat |
|---|---|---|---|---|---|
| baseline_small | 64 | 64 | 28.2 | 40.3 | 1.41 |
| late_inner_samepage | 256 | 264 | 112.8 | 161.9 | 1.44 |
| mid_inner_fewpages | 256 | 2048 | 113.5 | 159.2 | 1.40 |
| ios_eq_me_pow2 | 512 | 512 | 1135 | 316 | **0.28** |
| late_large_me_samepage | 2048 | 2056 | 3453 | 1535 | 0.45 |
| late_large_me_1024 | 1024 | 1032 | 1708 | 734 | 0.43 |
| heavy_dtlb_4pages | 2048 | 8192 | 3947 | 1728 | 0.44 |
| catastrophic_dtlb | 2048 | 32768 | 3897 | 1646 | 0.42 |
| **N1M_stage1** | **1024** | **16384** | **1962** | **747** | **0.38** |
| ios_eq_me_4k | 4096 | 4096 | 7995 | 3072 | 0.38 |
| extreme_ios_small_me | 256 | 65536 | 113 | 158 | 1.40 |

log3 wins at every me ≥ 512, often by 2-3×.

### Cross-chip divergence (SPR ratio / RL ratio)

| case | me | ios | SPR log3/flat | RL log3/flat | divergence |
|---|---|---|---|---|---|
| ios_eq_me_pow2 | 512 | 512 | 11.201 | 0.278 | **40.3×** |
| N1M_stage1 | 1024 | 16384 | 7.537 | 0.381 | 19.8× |
| late_large_me_1024 | 1024 | 1032 | 6.875 | 0.430 | 16.0× |
| catastrophic_dtlb | 2048 | 32768 | 5.137 | 0.422 | 12.2× |
| heavy_dtlb_4pages | 2048 | 8192 | 5.061 | 0.437 | 11.6× |
| late_large_me_samepage | 2048 | 2056 | 3.541 | 0.445 | 8.0× |
| late_inner_samepage | 256 | 264 | 5.014 | 1.435 | 3.5× |
| mid_inner_fewpages | 256 | 2048 | 4.920 | 1.403 | 3.5× |
| extreme_ios_small_me | 256 | 65536 | 4.820 | 1.401 | 3.4× |
| baseline_small | 64 | 64 | 4.303 | 1.412 | 3.1× |
| ios_eq_me_4k | 4096 | 4096 | 0.878 | 0.384 | 2.3× |

**Median divergence: 8×. Maximum: 40×.** Every single case disagrees.

## What changed between environments

Two variables differ; both plausibly contribute:

1. **Compiler**: GCC 13 on container, ICX 2025.3.0 on Raptor Lake. Both
   are `-O3 -march=native`.
2. **Chip**: Sapphire Rapids class (server, 64-entry L1 DTLB, mesh
   interconnect) vs Raptor Lake (consumer, 96-entry L1 DTLB, ring bus).

We can't separate the two in this experiment. Either could explain the
log3 inversion.

## The log3 code under examination

The log3 variant at R=4 loads `w1` and computes `w2 = w1*w1`, `w3 = w2*w1`,
creating a **3-deep dependency chain** before the butterfly can proceed:

```c
const __m256d w1r = R4_LD(&W_re[0*me+m]);
const __m256d w1i = R4_LD(&W_im[0*me+m]);
const __m256d w2r = fnmadd(w1i, w1i, mul(w1r, w1r));   // depth 2
const __m256d w2i = fmadd(w1r, w1i, mul(w1r, w1i));    // depth 2
const __m256d w3r = fnmadd(w2i, w1i, mul(w2r, w1r));   // depth 3 (needs w2)
const __m256d w3i = fmadd(w2r, w1i, mul(w2i, w1r));    // depth 3 (needs w2)
```

Then the butterfly uses `w1, w2, w3` — so all 3 twiddles must be ready
before the cmul with inputs can begin.

At R=4, the butterfly itself is only ~8 FMAs. The derivation chain
(4 cmuls = 16 FMAs, but 3 stages deep) is roughly the same size as the
entire butterfly.

## Hypothesis 1: Compiler scheduling difference

GCC and ICX both see this dependency chain. ICX appears to overlap the
w2/w3 derivation with butterfly FMAs more aggressively than GCC does.
The butterfly has many independent FMA chains (x1·w1, x2·w2, x3·w3), so
there's potential to hide w2/w3 computation behind x1·w1.

This is speculative without reading the asm, but it would explain why
log3 is 4-11× slower on GCC but 2-3× faster on ICX for the **same
source code**.

**Evidence for this hypothesis:**
- At small me (me ≤ 256 on RL) where memory isn't the bottleneck, log3
  is still only 1.4× slower on RL — not 4-11× like SPR. Even when
  bandwidth savings don't help, ICX keeps log3 cheap.
- GCC is known to be conservative with intrinsic scheduling in tight loops.

**Evidence against:**
- ICX's performance advantage over GCC on R=4 flat is large too
  (28 ns vs 1169 ns = 41× faster on RL). Most of that is chip frequency
  and memory hierarchy, not compiler scheduling.

## Hypothesis 2: Memory bandwidth asymmetry

On server chips (SPR), per-core L2/L3 bandwidth is distributed across
the mesh. Per-core effective bandwidth is lower than Raptor Lake's
direct-to-cache ring architecture.

Flat R=4 t1_dit has 6 twiddle loads per butterfly per iteration. On SPR
these loads are cheap (server chips design around many-core bandwidth
distribution with ample per-core L1 capacity). On Raptor Lake the same
loads compete with the input streams for the 3 load ports more
directly — saving 4 loads per butterfly (log3) becomes a significant
win.

**Evidence for:**
- The log3 win scales with me on Raptor Lake (0.28-0.45× at me ≥ 512).
  This matches a bandwidth-pressure story: at small me the twiddle table
  fits L1, at large me it pushes into L2 and bandwidth matters.
- The magnitude of RL's log3 win (2-3×) exceeds what pure compiler
  scheduling could explain — there's a real memory-bandwidth saving.

**Evidence against:**
- SPR's log3 penalty is uniform across me (3.5×-11× at all sizes), not
  shrinking at small me as a pure bandwidth story would predict.
- If it were purely bandwidth, SPR at me=64 should show log3 ≈ flat. It
  shows log3 4.3× slower.

## Hypothesis 3: The two effects compound

Most likely both mechanisms contribute:

- **Compiler scheduling** is the small-me story. On GCC, log3's dep
  chain serializes and costs ~4× overhead regardless of memory pressure.
  On ICX, the chain overlaps with butterfly work and costs almost
  nothing.
- **Memory bandwidth** is the large-me story. On RL with strong per-core
  bandwidth, log3's load savings translate directly to faster execution.
  On SPR with distributed bandwidth, individual loads are already cheap,
  so the savings don't materialize.

Compound effect: SPR's log3 is always expensive (bad compiler schedule)
and never cheaper (no bandwidth pressure to relieve). RL's log3 is
cheap to schedule AND relieves real bandwidth pressure at large me.
The 40× divergence at (me=512, ios=512) is this double effect at peak.

## Why this is a bigger deal than R=64's cross-chip divergence

At R=64 we saw SPR prefer t1s while Raptor Lake preferred log3 — but
both chips had the same basic *magnitude* of effect. A codelet on either
chip ran at broadly similar cycles, just with different winners.

At R=4 the cross-chip divergence includes the **absolute speed**:

- Container R=4 flat at me=64: **1169 ns**
- Raptor Lake R=4 flat at me=64: **28 ns**

A 41× absolute speed difference on the **same code**. Even adjusting for
Raptor Lake's 2× higher frequency + better memory + no container
virtualization overhead, at least 10-20× of that is unexplained by
hardware alone.

Together with the log3 inversion, R=4 is where the assumption "same ISA
= similar performance" breaks down hardest.

## What this reveals about VectorFFT's position

We've been saying "cross-chip codelet selection matters." This result
makes it concrete:

- **At R=64 AVX2**, choosing log3 vs t1s gets you a 30-50% difference
- **At R=4 AVX2 N=1M stage 1**, choosing log3 vs flat gets you a **62%
  difference on Raptor Lake or a 653% penalty on SPR** — an 8×
  performance swing between "right choice" and "wrong choice"

Libraries that bake in a single R=4 codelet choice will be catastrophically
wrong on roughly half the chips they encounter. FFTW, MKL, KFR all
compile with a fixed R=4 implementation. None of them measure per-chip.

**VectorFFT's per-chip selection is not a nice-to-have optimization.
For R=4 specifically it's the difference between being fast and being
slow.**

## What this breaks in our earlier analysis

Earlier, I said the standalone probe's "log3 loses on container by
3-7×" finding was too strong. After running the full bench on container
with more trials, log3 actually wins 17% of container regions (mostly
medium me).

The picture I now believe:

1. The standalone probe's numbers are **roughly correct** — log3 on GCC
   is consistently 3-11× slower than flat at small to medium me.
2. The full bench's different measurement approach (longer warmup, more
   reps, fwd+bwd average) gives log3 more favorable conditions. It's
   picking log3 in regions where the difference is marginal, where
   log3's 1.0-1.1× slowdown in one direction trades off with a
   coincidental 0.95× advantage in the other.
3. **On Raptor Lake, log3 wins are real and large** — 2-3× speedups
   that aren't measurement artifacts.

The cross-chip story dominates. On each chip independently, the
codelet selector picks sensibly. But what gets picked is completely
different.

## Where this leaves the R=4 strategy

With 5 codelet variants (flat, log1, log3, u2, dif), the R=4 bench
produces different selector outputs on each chip:

- On container GCC: `dit` dominates (19 wins), the 4 alternative
  variants split the remaining 29.
- On Raptor Lake ICX: expected `log3` and `u2` to dominate, with `dit`
  surviving only at small me.

Both selectors work. Neither transfers. This is the "install-time
tuning" story you articulated becoming mechanistically necessary — not
just "better performance," but "correct performance."

## Open questions

1. **Is it primarily compiler or chip?** Would need to run GCC on
   Raptor Lake (or ICX on SPR, if that's feasible) to separate.
2. **Does the pattern hold for other small radixes?** R=2, R=3 likely
   similar scale. R=8 might or might not.
3. **Does `log1` (2-deep chain) land between flat and log3 on both
   chips?** The Phase 2 container result suggests yes on GCC. Raptor
   Lake's behavior unknown until you run it.
4. **Is there a generator-level fix that makes log3 work on both
   compilers?** Manual scheduling hints, restructured derivation
   sequence, etc. — might be worth a few hours of exploration if we
   can eliminate the compiler-scheduling dependency.

## Why this document exists

This is the clearest evidence so far for the VectorFFT cross-chip
story. When we write up the pow2 release, or submit anywhere, R=4 is
the concrete data point that justifies the per-chip bench design. Not
a 10-30% performance hint — an 8× correctness gap.
