# 42. R=512 log3 vs flat: A B-Dependent Crossover

## Context

Tugbars asked to compare R=512 with `--log3` twiddle policy against the default
`--t1` (flat) policy on the hypothesis that log3's reduced twiddle load count
would relieve port load/store pressure.

The two policies trade off:
- **flat (t1)**: load all (R-1) twiddles directly from memory. R=512 → 511
  distinct twiddle loads per K-iter (1022 if counting re+im).
- **log3**: load only the power-of-2 twiddles W^(2^k), derive the rest by
  binary decomposition using chained complex multiplies. R=512 → 9 power-of-2
  twiddles (18 with re+im). Each derived twiddle costs one cmul (4 muls + 2
  adds).

The advertised tradeoff per `dft.ml`:
> "log3 saves twiddle bandwidth at the cost of arith. Whether it wins depends
> on which is the bottleneck."

This doc characterizes when each wins at R=512.

## Static analysis (gcc-11 + flive-range-shrinkage)

Instruction breakdown of the generated codelets:

```
metric                  flat       log3      delta
total body              27625      30510     +10.4%
spill (rsp)             3699       5320      +43.8%  MORE  ← surprising
mem_loads_rsp           1844       2715      +47.2%  MORE
mem_loads_nonrsp        2037       1039      -49.0%  FEWER ← expected
reg_mov                 1224       1483      +21.2%  MORE
fma                     1377       1879      +36.5%  MORE
addsub                  9216       9216       0.0%
mul                     3034       4038      +33.1%  MORE
```

The advertised win is real (49% fewer non-stack memory loads) but log3 also has:
- 47% MORE stack/spill loads — derived twiddles cause register pressure
- 36% MORE FMAs (the cmul derivations)
- 33% MORE muls
- 10.4% MORE total instructions

The mechanism behind the spill increase: derived twiddles like W^7 = W^3·W^4
become themselves live values. At R=512 there are ~500 derived twiddles. They
overflow the 32 ZMM register file and GCC spills them to the stack —
converting "twiddle memory loads" into "stack spill loads." Static analysis
predicts flat wins.

## Static analysis is wrong

Runtime measurement (gcc-11 + shrink, container CPU, min ns/iter from 7 runs):

```
B      flat       log3      log3 vs flat
8      6993       7517      +7.5%   (log3 slower)
16     13852      15499     +11.9%
32     30168      34005     +12.7%
64     67343      68365     +1.5%   (≈ tied)
128    181564     137128    -24.5%  (log3 WINS)
256    515836     396079    -23.2%  (log3 WINS)
```

The crossover is sharp and lives between B=64 and B=128. Above it log3 is
23-25% faster. Below it flat is 7-13% faster.

## Why the static count misses this

The static asm count is "instructions per codelet call" — it doesn't account
for cache hierarchy effects when those instructions execute repeatedly.

At small B (B ≤ 32), the twiddle working set is tiny:
- flat: 511 twiddles × 32 elements × 16 bytes = 256 KB total twiddle data
- log3: 9 twiddles × 32 elements × 16 bytes = 4.5 KB
- Either fits comfortably in L2 (256-512 KB typical) and partially in L1 (32 KB)
- The "fewer loads" advantage of log3 is small in absolute terms because the
  loads are mostly L1 hits anyway

At large B (B ≥ 128), the twiddle working set explodes:
- flat at B=256: 511 × 256 × 16 = **2 MB** of twiddle data per call — exits L2
  cache, accesses L3 (or worse, main memory) on every iteration
- log3 at B=256: 9 × 256 × 16 = 36 KB — fits L1 comfortably

The shift in cache pressure dominates the increased spill+compute cost. log3's
extra 1621 spill ops are L1-hot (recently-written stack data), while flat's
saved 998 twiddle loads from the cmul derivation are L3/memory-cold reads.

**Cache-cold memory accesses cost orders of magnitude more cycles than L1
spill ops.** A single L3 hit is ~30-40 cycles; a main memory access is
200+ cycles. An L1 hit is ~4 cycles. The static asm count weighs all loads
equally, which is wrong.

## Implication for the planner / wisdom harness

This is a clean case for the codelet-level wisdom harness. The static
analysis says flat wins; the runtime says it depends on B in a non-obvious
way; only measurement on actual hardware resolves it.

The wisdom rule for R=512 AVX-512 on this CPU class:

```
if B ≤ 64:  use t1 (flat)
if B ≥ 128: use t1_log3
```

The threshold likely shifts per hardware:
- Larger L2 caches push the crossover to larger B (flat stays good longer)
- Larger L3 caches push it further
- Faster memory subsystems reduce log3's relative advantage (smaller penalty
  for cache miss)
- Different prefetcher behavior changes which loads are absorbed

So per-machine calibration is exactly the right approach. This isn't a
universal rule, it's a per-(R, ISA, B) measurement.

## Interaction with multi-stage cascade (open question)

Doc 41 showed multi-stage cascades win at R=1024. Those cascades use
sub-codelets at R=16/R=32/R=64. The log3 vs flat choice exists for those
too:
- R=64 has 63 twiddles flat, 6 log3 — much smaller absolute difference
- At B=128 with R=64: flat twiddle data = 63 × 128 × 16 = 128 KB (L2 fits)
  log3 = 6 × 128 × 16 = 12 KB (L1 fits)
- The crossover for R=64 likely shifts to higher B

This adds another dimension to the codelet wisdom: for each (R, ISA), the
log3-vs-flat crossover B. With 4-5 R values and 2 ISAs in scope, that's
8-10 calibration measurements. Tractable.

## What this confirms about wisdom-driven design

The discussion about install-time wisdom harness landed on "we can't predict
the winner from static analysis." This experiment is a clean validation:
the static analysis predicted flat would win across the board (it has fewer
total instructions and fewer stack ops), but at large B log3 wins by 23-25%.

Without measurement, you'd ship flat and leave 25% of high-B performance on
the table. With per-codelet wisdom calibration, the harness identifies the
crossover B and routes to the right variant.

## Files state

No code changes this session — gen_radix already supports both `--log3` and
default policies via the existing `TP_Flat` / `TP_Log3` types in dft.ml.
This was purely a measurement.

The bench harness (`/tmp/log3_bench.c` during the session) follows the
same pattern as the R=1024 bench. Worth packaging into a permanent
measurement tool when building the wisdom harness.

## Caveats

The first is container CPU. The crossover B is hardware-dependent; on
i9-14900K's larger L1/L2 caches the crossover shifts. Need per-target
measurement.

The second is that the bench runs the codelet in a tight loop with the
same twiddle data — best-case for cache reuse. Real applications interleave
codelet calls with other work; cache state at codelet entry varies. The
worst-case (cold cache) probably shifts the crossover further toward log3
(it has less data to refetch).

The third is that the spill increase under log3 (+1621 stack ops) is
dependent on register file size. On AVX2 with only 16 YMM registers, log3
would spill even more aggressively — the crossover might shift differently
or log3 might never win. Worth a separate measurement.
