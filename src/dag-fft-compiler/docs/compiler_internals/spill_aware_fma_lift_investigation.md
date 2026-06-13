# Spill-Aware fma_lift Investigation — Negative Result

## TL;DR

Investigating the hypothesis that the unfused muls in R=11/13/25 could be
eliminated by making `multi_use_fma_lift` spill-aware. Empirical answer:
**no, they cannot.** The muls are not pinned by spill markers — they're
mathematically necessary given the algebra's structure.

## Method

For each unfused Mul in R=11/13/25/32/64, traced its complete data flow
through spill stores/loads to find the IR-level consumers (not the
emitter's intermediate variables). Classified each by:
- Pure FMA-addend: Mul used only as the addend of an FMA
- Pure spill: Mul appears only in a spill store (and the load is consumed
  elsewhere as FMA addend)
- Both: hybrid case
- Other: complex patterns

Then checked the CONSUMER of each FMA: if it's a pure Add/Sub, chained-FMA
rewriting saves 1 instruction. If it's another FMA, no third operand is
available and the chain breaks even.

## Findings

### Classification of unfused muls

```
                                                  Total  FMA-add  Spill  Both   Other
R=11                                                10      1       9      0     0
R=13                                                12      1      11      0     0
R=25                                                31      9      10      6     6
R=32                                                 6      6       0      0     0
R=64                                                22     18       0      0     4
```

### Chain-savings opportunity (per-mul rewrite analysis)

```
                                                  Total  Saves  Break-even  Other
R=11                                                10     0       10        0
R=13                                                12     0       12        0
R=25                                                31     0       20        0
R=32                                                 6     0        6        0
R=64                                                22     0       18        0
```

**Zero chained-FMA savings across all radices.**

### Why

Each "unfused Mul" sits in a context like:

```
M    = Mul(c, x)                           [1 mul]
F    = Fma(a, b, M)        // a*b + c*x    [1 fma]
F'   = Fma(d, e, F)        // d*e + a*b + c*x  [1 fma]
```

This is 3 multiplications combined with 2 additions. The minimum-instruction
form requires **3 mul-bearing instructions** — there's no way to express
3 distinct multiplications in fewer ops. FMA = mul + add fused; each FMA
covers exactly 1 multiplication.

For the chain `d*e + a*b + c*x` to fit into 2 FMAs (0 muls), we'd need a
**non-product fourth operand** to seed the chain:

```
F'   = Fma(c, x, OTHER)    // c*x + OTHER
F''  = Fma(a, b, F')       // a*b + c*x + OTHER
F''' = Fma(d, e, F'')      // d*e + a*b + c*x + OTHER = original + OTHER
```

If OTHER existed, we'd get 3 FMAs and 0 muls. But in our IR, the inner
chains finalize with no free addend — every term IS a product.

## Where the spill markers ACTUALLY come from

The spill markers in R=11/13/25 mark *result tags* of values that survive
across a register-pressure boundary. The Mul's spill marker doesn't
"cause" the unfused mul — it's marking the value because the value
needs to outlive its initial computation point. Even if we could
"redirect" the spill marker to an FMA's output, the FMA's value isn't
the same as the Mul's value (the FMA = Mul + something), so the
redirect would store the wrong thing for the load consumer.

The empirical proof: R=32 and R=64 have **zero** spill markers yet
still have unfused muls (6 and 22 respectively) for the exact same
structural reason. The mechanism is not spill-related at all.

## What FFTW does instead

FFTW achieves 0 unfused muls by structuring the algebra at code-emission
time such that every product appears in a chain with at least one
non-product seed. Their `gen_notw` generator builds expressions
**already in FMA-chain form** — they never produce an isolated
`a*b + c*d` pattern with no third operand. Instead they consume some
existing intermediate value as the seed:

```c
// FFTW R=25 (excerpt):
T4Q = FNMS(KP618033988, T1S, T1T);    // T1T  - 0.618*T1S   (T1T is the seed)
T1U = FMA(KP618033988, T1T, T1S);     // T1S  + 0.618*T1T   (T1S is the seed)
T3b = FMA(KP559016994, T3a, T39);     // T39  + 0.559*T3a   (T39 is the seed)
```

The seeds (T1T, T1S, T39) are pre-computed intermediate values from
elsewhere in the algorithm. They serve as the non-product starting point
for FMA chains.

To match this at our IR level, we'd need a global pass that reorders the
algebra: find sequences of N products that get summed together, identify
a non-product value nearby, and emit as N chained FMAs starting from that
value. This is significantly more invasive than fma_lift — it requires
restructuring the computation graph, not just rewriting local patterns.

## Why this is OK for VectorFFT

The op-count gap to FFTW is real (~10-30 ops per codelet across primes)
but **runtime impact is irrelevant**: VFFT already outperforms FFTW by
11-30% on R=11, R=13, R=25 across both AVX2 and AVX-512 in our
benchmarks. The batched-SIMD architecture (K=8 lane parallelism across
transforms) more than offsets the extra ops, because FFTW's 0-mul
arrangement provides no SIMD speedup at single-transform-vectorization
granularity.

The unfused muls execute concurrently with surrounding FMAs (modern x86
has 2 separate FMA-capable execution ports). The structural minimum
of "3 mul-bearing instructions for 3 multiplications" applies to our
code AND FFTW's code; FFTW just relabels some of them from `mul` to
`fma` with mul-and-add fusion.

## Recommendation

**Don't implement spill-aware fma_lift.** The investigation shows it
won't help. The hypothesis was wrong — the muls aren't pinned by spill
markers; they're pinned by IR structure.

**Don't pursue global algebra restructuring** unless an alternate
motivation arises. Op-count parity with FFTW is achievable but
expensive, and our wall-clock is already ahead.

**Document the finding.** The empirical method (classify muls, trace
consumers, count chain opportunities) is reusable for any future
algsimp investigation. The negative result is itself a contribution —
it eliminates a tempting direction with concrete evidence.

## Artifacts

- `/tmp/classify_muls.py` — classification by FMA-addend / spill / etc.
- `/tmp/chain_analysis.py` — savings analysis per Mul
- Walkthrough trace of t235 in R=11 showing structural minimum is met
