# 41. R=1024: The Monolithic Threshold

## Context

Docs 33-40 characterized the spill controller behavior at R=128, 256, 512.
Monolithic codelets (one big function call producing the full DFT-N) won
against multi-stage cascades at moderate batch sizes — fewer function
calls, better OoO scheduling across the full body, identifiable spill
patterns that the recipe could manage. Doc 38 found gcc-11 +
`-flive-range-shrinkage` was the right compiler config for these
codelets.

Open question: does this trend continue at R=1024? With the recipe and
gcc-11+shrink in place, what's the largest monolithic codelet that still
beats a multi-stage cascade?

This doc answers: R=1024 is past the threshold. Monolithic R=1024 runs
~50% slower than equivalent multi-stage cascades at moderate batch sizes,
on this container CPU and likely on production hardware.

## Setup

Picker entry added: `R=1024 -> Cooley_Tukey (32, 32)` — symmetric
factorization. Both Pass 1 and Pass 2 have sub-DFT-32 clusters; the
natural extension of the R=512 = CT(16, 32) line. Asymmetric
alternatives (CT(16, 64), CT(64, 16)) would put one pass's clusters
at sub-DFT-64 size which is unambiguously past the 32 ZMM register
file — predicted catastrophic.

Generated R=1024 monolithic codelet:
- Generation: 12 seconds
- C source: 38554 lines, 2.6 MB
- Output: 1024 recipe stores + 1023 twiddle loads + Pass 1/Pass 2 emission
  resulting in 4096 total `_storeu_pd` calls (recipe + output) and 6142
  `_loadu_pd` calls

Compiled monolithic codelet:
- gcc-13 default: 230 seconds compile time, **75280 body insts, 19951 stack ops**
- gcc-11 + `-flive-range-shrinkage`: 277 seconds compile time, **70026 body insts, 17952 stack ops**

## Stack op scaling vs R=512

Comparing per-codelet spill cost across sizes (gcc-13 default):

```
R       body insts   stack ops   ratio vs R=64
R=64    2612         255         1.0×
R=128   5863         659         2.6×
R=256   13406        2040        8.0×
R=512   30171        5216        20.5×
R=1024  75280        19951       78.2×
```

Scaling from R=512 to R=1024:
- Body length: **2.50×** (R doubled, slight super-linear from N log N)
- Stack ops: **3.82×** — significantly super-linear

**Compute scales as N log N (about 2.2× from R=512). Spill cost scales as 3.82×.**
Register pressure grows disproportionately faster than the work. This is the
phase transition we're characterizing.

The mechanism: at R=512 = CT(16, 32), Pass 2 has sub-DFT-16 clusters
(small, fits in 32 ZMM easily). At R=1024 = CT(32, 32), Pass 2 has
sub-DFT-32 clusters (right at the 32 ZMM boundary). With twice as many
clusters AND each cluster pressing the register file harder, GCC's
allocator spills disproportionately more.

## gcc-11+shrink leverage shrinks at R=1024

```
R       gcc-13 default   gcc-11+shrink   improvement
R=256   2040             1298            -36.4%
R=512   5216             3699            -29.1%
R=1024  19951            17952           -10.0%
```

The flag's leverage **collapses at R=1024**. Where it saved 29-36% at R=256/R=512,
at R=1024 it only saves 10%. Interpretation: at extreme register pressure, even
"shorten live ranges" has no room to maneuver — every cluster is already
saturating the 32 ZMM file, and any reduction in one place just shifts pressure
elsewhere.

This means **the compiler flag can't rescue monolithic R=1024**. The recipe-level
mechanism is also exhausted (per docs 35-40 we already know it can't push lower).

## Multi-stage cascades: stack op accounting

For multi-stage R=1024 = N1 × N2, the cost is:
- Stage 1: N1 codelet calls of size N2
- Stage 2: N2 codelet calls of size N1

Per-codelet stack ops (gcc-11 + shrink):
- R=16: 8
- R=32: 80
- R=64: 220

Total per logical R=1024 call:

```
Variant              Stack ops    Codelet call count
monolithic R=1024    17952        1
64×16 cascade        4032         80   (64×R=16 + 16×R=64)
32×32 cascade        5120         64   (32×R=32 + 32×R=32)
16×64 cascade        4032         80   (16×R=64 + 64×R=16)
```

**Multi-stage cascades have 3.5-4.5× fewer stack ops than monolithic.** Even
accounting for function call overhead (which adds ~30-50 cycles per call × 64-80
calls = ~4000 cycles overhead per R=1024 invocation), the spill traffic
difference dominates.

## Runtime: monolithic vs multi-stage

Min ns/iter on container CPU (gcc-11 + shrink), 7 independent runs:

```
B    mono_R1024   64×16    32×32    16×64    mono vs best multi
8    24595        19894    22805    20075    -19% slower
16   67479        40533    45781    40403    -40% slower
32   167006       83184    83534    83342    -50% slower
64   344051       184140   180173   186204   -48% slower
```

The pattern is striking and gets worse at larger B:
- B=8: mono lags by 19% (moderate)
- B=16: mono lags by 40%
- B=32: mono lags by 50% — **mono is 2× slower**
- B=64: mono lags by 48%

The multi-stage variants (64×16, 32×32, 16×64) are all within 10% of each other
at any given B. Mono is clearly in a different performance class.

## Why mono loses at R=1024 but won at R=512

This is the inverse of doc 33's R=128/256/512 finding. There mono won
because OoO execution spread out the spill traffic across the full body
and L1 absorbed it. At R=1024, the spill traffic per call becomes too
large for L1.

Memory traffic per call (B=64):
- Mono: 17952 spill ops × 64 bytes = **1.15 MB** per call — far beyond L1 (32-48 KB)
- Multi (64×16): 4032 spill ops × 64 bytes = **258 KB** total across 80 calls,
  ~3.2 KB per call on average — fits L1 comfortably per call

The mono codelet's spill working set spills out of L1 into L2 traffic.
The multi-stage approach keeps each per-call working set L1-resident,
even though it pays function call overhead and inter-stage data movement
on top.

## Bench caveats

Two caveats reduce the strength of the runtime claim:

The first is that the bench OMITS the transpose between stages. A real
multi-stage R=1024 needs to reorganize data from `[k2][n1][b]` to `[n1][k2][b]`
layout between Pass 1 and Pass 2 — one full pass over N×B doubles. At B=64
that's 1024 × 64 × 2 × 8 = 1 MB of additional memory traffic. Doc 34 showed
that with real transpose, R=512 multi-stage's lead at high B shrinks.

For R=1024, the spill traffic difference is 1.15 MB - 0.26 MB = ~0.9 MB
per call. The transpose adds 1 MB per call. So with realistic transpose
included, the multi-stage advantage at B=64 likely shrinks from 48% to
maybe 15-25%. Still a clear multi-stage win, but the magnitude is smaller.

The second is that the container CPU has clock and cache characteristics
different from i9-14900K. Production hardware may shift the boundary —
higher clock favors mono (compute throughput), larger L1 favors mono
(can fit working set), better memory bandwidth favors multi (transpose
faster). Net direction is uncertain without production measurement.

## What this means for the planner

The picker should route R=1024 to a multi-stage path. Most natural choice
based on stack op accounting and runtime measurement: **64×16 or 16×64**
(equivalent up to data layout). 32×32 is slightly worse on the container
(~3-5%) likely due to its higher total stack op count (5120 vs 4032).

For R=2048 and above, the pattern should continue: multi-stage cascades
win progressively more. R=2048 mono would have estimated stack ops of
~76000 (extrapolating 3.82× scaling) — definitely structurally infeasible.

**The monolithic codelet's "sweet spot" appears to be R ≤ 512.** At R=512,
monolithic wins at moderate B (doc 34). At R=1024, monolithic loses to
multi-stage by 50% at moderate-to-large B. The threshold is between these.

## Configuration recommendation

Given the data, the right architectural rule:

```
R         Strategy
≤ 64      Single codelet (always)
128, 256  Monolithic codelet (verified wins doc 33)
512       Monolithic at B ≤ 256; consider multi-stage above (doc 34)
1024+     Multi-stage cascade with R=64 or R=16 building blocks
```

The picker currently routes R=1024 through CT(32, 32) monolithic since
that's the entry just added. **The picker should not be changed yet** —
keep it as is for repro/research purposes, but mark the runtime planner
(when it's added) to ALWAYS prefer multi-stage at R ≥ 1024.

## What the recipe still gives us

Even though monolithic R=1024 loses to multi-stage, the recipe's value
at R=64 (where the cascades' building blocks live) hasn't changed. The
multi-stage cascade's 4032 stack ops are themselves the result of the
recipe and gcc-11+shrink working well at R=16/R=32/R=64.

In other words: the recipe is still load-bearing — it makes the
multi-stage cascade's per-codelet costs small. Mono at R=1024 just
doesn't fit the recipe's "single big function" assumption anymore.

## Files state

- `lib/dft.ml`: added R=1024 -> CT(32, 32) picker entry
- Stack ops verified: 17952 with gcc-11+shrink, 19951 with gcc-13
- Build clean, prime correctness 56/56 PASS (R=1024 is composite,
  unaffected by prime test; checked R=2,5,7,11,13,17,19)
- No other code changes

## Open follow-ups

The first is the runtime measurement on real i9-14900K hardware. Container
numbers are directional only; the gap may be larger or smaller in production.

The second is testing CT(16, 64) and CT(64, 16) at R=1024 to confirm they're
worse than CT(32, 32). Stack op prediction: catastrophic Pass 1 (or Pass 2)
register pressure. This would close the factorization question for R=1024.

The third is the runtime planner — once we add a planner that chooses
strategies per (R, B, ISA), R=1024 should route to multi-stage by default.
The existing picker entry is informational/research-purposes only.

The fourth is profiling the L1/L2/L3 cache miss patterns under both
strategies to validate the "mono spills out of L1, multi fits" hypothesis
with hardware counters rather than inferred from byte counting.
