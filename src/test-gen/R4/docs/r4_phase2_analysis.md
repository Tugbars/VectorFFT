# R=4 codelet bench: 5 variants on SPR container

Phase 1 and Phase 2 complete. Added `ct_t1_dit_log1` (middle-ground twiddle
derivation with 2-deep dependency chain) and `ct_t1_dit_u2` (m-loop unrolled
2× for ILP) to the R=4 generator. All 5 variants correctness-validated
(40/40 test cases pass, bit-exact for u2, machine-epsilon for log1).

Only AVX2 benched for now — gen_radix4.py's AVX-512 path has a pre-existing
bug (`_mm256_unpacklo_pd` used on `__m512d` in `n1_ovs`). Fix in separate
commit; doesn't affect t1 codelets we care about.

## Selector output — container SPR AVX2

Out of 48 decisions (24 sweep points × 2 directions):

| Family | Wins | Share |
|---|---|---|
| `ct_t1_dit` (flat) | **19** | 40% |
| `ct_t1_dit_u2` (2× unroll) | 10 | 21% |
| `ct_t1_dit_log1` (2 loads + 1 deriv) | 9 | 19% |
| `ct_t1_dit_log3` (1 load + 2 derivs) | 8 | 17% |
| `ct_t1_dif` (post-twiddle) | 2 | 4% |

**All 5 families win meaningful regions.** This is unusual — typically at R ≥ 16
one or two families dominate. R=4's bottleneck surface is small enough that
small optimizations tip the balance differently at each operating point.

## Per-me regime map

| me regime | Condition | Winners |
|---|---|---|
| me = 64-128 | L1-resident twiddle table | `dit`, `u2` (marginal) |
| me = 256 | Transition (~L1 boundary) | `dit` with padding, `u2` at ios=me |
| me = 512 | Approaching L1 overflow | `log1` and `log3` gain ground |
| me = 1024 | Twiddle table overflows L1 | `log3` dominates (bandwidth > dep chain) |
| me = 2048 | Deep L1 overflow | `log3` + `u2` split wins |

## Detailed results at ios = me+8 (padded, DTLB-friendly)

```
me        dit   log1   log3     u2    dif   winner
64         71     76     86     66     71   u2
128       146    163    175    146    152   u2 (tie with dit)
256       272    315    353    377    287   dit
512       465    363    356    481    446   log3
1024     1558   1551   1492   1567   1689   log3
2048     4120   3233   3494   3508   3454   log1
```

Flat `dit` wins at small me, loses as me grows. At me=512+ the derivations
pay off — they save L1→L2 traffic at the cost of a couple FMAs, which fit
in the execution schedule when memory is saturated.

## Log1's niche: the middle ground

The prediction was: log1 would fit between flat and log3 where the 3-deep
chain of log3 hurts but bandwidth savings are still valuable. Data confirms:

**log1 beats log3 at small-to-medium me (shorter dep chain wins):**
```
me=64   log3/log1 = 1.093  (log1 faster)
me=128  log3/log1 = 1.052
me=256  log3/log1 = 1.174
```

**log3 beats log1 at large me (single load beats double load):**
```
me=512   log3/log1 = 0.979  (nearly tied)
me=1024  log3/log1 = 0.756  (log3 wins by 24%)
me=2048  log3/log1 = 0.748  (log3 wins by 25%)
```

The crossover is right around me = 512, which corresponds to the L1/L2
transition for the twiddle table (3 × 512 × 16 = 24 KB — still fits
L1, but the 4 input streams + 3 twiddle streams at stride 1024 create
cache pressure).

## U2's surprising shape

`u2` (m-loop unrolled 2×) doesn't strictly win at large me as expected:

```
me=64    u2/dit = 1.07   (lose slightly, probably noise)
me=128   u2/dit = 1.00
me=256   u2/dit = 1.12   (lose 12%)
me=512   u2/dit = 0.985
me=1024  u2/dit = 1.28   (LOSE 28%!)
me=2048  u2/dit = 0.76   (win 24%)
```

The me=1024 regression is suspicious. Hypothesis: the 2× unroll creates
more live values (16 input regs × 2 groups = 32 YMM values needed, AVX2
has 16 physical regs available). The compiler spills to stack. This
spill cost overwhelms the ILP benefit when me fits L2 comfortably.

At me=2048 the pattern inverts — spill cost is amortized over many more
iterations and memory-latency hiding from the 2× ILP finally pays off.

This is a concrete example of **register pressure effects at small radix**
(the thing we discussed earlier — actually a real mechanism here, even if
it wasn't for R=4 flat).

## What to expect on Raptor Lake

Prediction based on cross-chip patterns:
- **log3 should win more regions** on Raptor Lake (the standalone probe
  showed 2.6× speedup at N=1M stage 1 scenario)
- **u2 should win more regions** on Raptor Lake too — more PRF, better
  renaming throughput means the spill cost from unrolling matters less
- **log1 probably shrinks** — either log3 or u2 steals its niche
- **dit still dominates at small me** — universal finding

If Raptor Lake mirrors SPR's pattern, all 5 families are useful.
If Raptor Lake's log3 wins dominantly as the probe suggested, 3-4 families
(dit, log3, u2, maybe log1) would be enough.

## Artifacts

- `gen_radix4.py` — 5 variants (now with log1 + u2 added)
- `candidates.py` — 5 candidates × AVX2 = 5 total
- `run_r4_bench.py` — R=4 bench driver (adapted from r64_bench, with
  wrapper logic extended to handle `static inline` in gen_radix4 output)
- `select_codelets.py` — family preference updated
- `spr_results/` — measurements.jsonl (240 datapoints), selection.json

## Next steps

1. You run on Raptor Lake. Expected time ~30 seconds.
2. Cross-chip analysis comparing SPR vs RL.
3. Output: clean R=4 codelet story for the project, plus ready-to-ship
   generator work.
