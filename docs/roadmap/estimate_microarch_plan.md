# The estimate from the machine's parameters

The estimate plans without measuring, from N, the request and the machine. Its rules were fitted on one
machine, the i9-14900KF. The owner, 2026-11-03: "the estimate should have rough estimate that is based on
microarchitecture parameters. this is the essence of this research, so we need to discuss how to find rules
that adjust based on cache size or other microarch parameters without overfitting." A rule that holds on the
14900KF and fails on another machine has the wrong form, not the wrong constant.

## 1. The two machines

| | i9-14900KF, P-core | Ryzen 5 PRO 8640HS (Zen 4) |
|---|---|---|
| build | AVX2: 4 lanes, 16 vector registers | AVX-512: 8 lanes, 32 vector registers |
| L1d | 48 KiB, 12 ways | 32 KiB, 8 ways |
| L2 | 2 MiB | 1 MiB |
| L3 | 36 MiB | 16 MiB |

Every parameter differs between the two, so a difference between their results cannot be pinned on one
parameter from these two alone (section 5).

The formulas read `est_machine_t` (estimate_il.h:40-49): the build's lanes, the L1d, the L2 and the L3; the
core count is reported and no rule reads it. They do not know the register count (it moves with the lanes in
our two builds), the L1's associativity or way size (4 KiB on both: 48 KiB / 12, 32 KiB / 8), the cache line
(64 B on both), whether the L2 is private (a 14900KF E-core cluster shares 4 MiB among four cores), or how
wide the vector unit runs (Zen 4 executes a 512-bit operation as two 256-bit halves).

## 2. The audit of the fixed numbers (2026-11-03)

Every numeric literal in the code of `src/core/il/estimate/*.h` other than 0, 1 and 2, comments and strings
removed: 401 lines. Each sorted into one of four kinds.

- **Code facts.** What the library builds: kernel lists, legal chain shapes, the door's own ladders and lane
  laws (the `VFFT_*` lists, `est_ztt_ok`, the kernel nibbles, `EST_ND_MT_CSK_COLS`, `EST_ND_MT_TURN_N1_DIV`,
  `EST_ND_MT3_CYCLE_LAST`, `EST_ZFSR_TB`, the four-step's threaded 8.64.8 child). They follow the build and
  are not machine rules.
- **Search order and tie-breaks.** The balance of a chain, the order of its stages, the Bluestein length's
  10 % margin, ties. Machine-free by intent.
- **Cache rules** (section 3): a byte count against the L1d, L2 or L3. The right kind of rule; every divisor
  was set on the 14900KF.
- **Fixed sizes** (section 4): a number that stands for a machine limit without naming it. These are the
  suspects.

## 3. The cache rules (18)

| rule | where | form | Zen 4 so far |
|---|---|---|---|
| the one-thread bound, the ZTURN-T split point | estimate_il.h:237, 246 | 16 N <= 2 L1d; log2(L1d / 16) | holds: 1D to 2^18 at 1.03 |
| the four-step's band | estimate_il.h:1008 | 16 w N2 <= L2 | holds within the split it is given |
| the four-step past the L3 | estimate_il.h:1073, 1198 | 16 N > L3 | untested |
| a prime by Rader | estimate_il.h:1310 | N <= L1d / 16 (with p - 1's factors) | untested |
| the batch's way | estimate_il.h:1508-1520 | one transform within L2 / 8, L2 / 4, L2 | untested |
| the narrow row | estimate_nd_il.h:358 | a radix-16 butterfly across the row, 16 x 16 w <= L1d | the radix 16 in it is a register size (section 4, item 12) |
| one column stage | estimate_nd_il.h:371-374 | the plane within L1d, or L1d / 4 | untested |
| the 2D band | estimate_nd_il.h:737-748 | none while the plane fits L2 / 4; the band within L2 / 4 | **fails**: 256 x 4096, 1024 x 4096 1.35-1.49; the 14900KF loses 1.097 at 512 x 8192 |
| the 3D band | estimate_nd_il.h:750-765 | the band within the L2 | untested |
| the 3D strips | estimate_nd_il.h:783-797 | volume past L3, 16 N1 <= L1d / 4, strip within L2 / 2 | untested |
| the 3D flat tier | estimate_nd_il.h:1061-1071 | the plane 40-125 % of the L2 (75 % past the L3) | see item 17 |
| the 2D plan at T > 1 | estimate_nd_il.h:923-931, 1002 | serial within L1d / 2; strips within L1d / 2 or in and out within the L2 | untested |
| the 3D plan at T > 1 | estimate_nd_il.h:1144 | strips within L2 / 4 | untested |
| the lane-major strips | c2c_estimate_lm_il.h:121, 164, 171 | lanes >= (lanes / 2) T; 16 N1 x 8 <= L2; 16 N1 lanes > L1d | untested |
| the real flat DIT's tile | real_estimate_il.h:307-309 | 8 N > L1d; 16 b <= L1d / 2 | untested |
| the real C2R plane of 128 rows | real_estimate_nd_il.h:192 | bytes > 5/8 L2 | untested |
| the real 3D cube | real_estimate_nd_il.h:680, 734 | past L3 (C2R) or 4 L2 (R2C); within L3 / 3 | untested |
| a crowded L1 set | real_estimate_nd_il.h:174-175, 700-701 | R x min(the stride's lowest bit, 4096) > L1d | the 4096 is the L1 way size, equal on both |

On the 2D band: both planners band where the estimate does not, at bands the estimate counts as 2-8 MiB
(14900KF 1024 x 4096 at 32 rows, 2 MiB, 512 x 8192 at 32, 4 MiB; Zen 4 256 x 4096 at 32, 2 MiB, 1024 x 4096 and
512 x 8192, 8 MiB), past the L2 on both. The L2 is not what limits the band, so changing the divisor would not
fix it.

## 4. The fixed sizes (26)

"Cause" is the one the code's comment states, or marked *guess*.

| # | where | number | decides | cause | would follow | Zen 4 so far |
|---|---|---|---|---|---|---|
| 1 | estimate_il.h:1398 | 2^19 | four-step from here | *guess*: the L2/L3 traffic of one pass | L2 or L3 | holds: both planners switch at 2^19 |
| 2 | estimate_il.h:961-963 | N1 = 512 at 2^19, 4096 from 2^20 (threaded N / 2048 to 2^20) | the four-step's column length | none stated | *guess*: the band against a cache level | **fails**: Zen 4's planner takes N1 = 256-2048 to 2^22, 1.10 over 5 lengths. In the band rule's own count (16 w N2, estimate_il.h:1008) the 14900KF's planner bands within its L2 (0.25-1 L2) and Zen 4's past it (2-8 L2) |
| 3 | estimate_il.h:860-862 | row part <= 512, column radix <= 32, banded radix <= 8 | the split past the L3 off powers of two | stated: a row part of a multiple of 256 puts the columns 4 KiB apart | L1 way size; registers for the 32 | untested |
| 4 | estimate_il.h:1399-1402 | pair route at 32-512; R1 = 4 to 128, else 16 | the two-pass split | *guess*: the leaf's register load | lanes, registers | **fails** at 128: 4.32 loses 1.17 in both runs (Zen 4's planner 8.16) |
| 5 | estimate_il.h:522-524 | inner pair 16-512, the same split | a Rader/Bluestein inner | as 4 | lanes, registers | untested |
| 6 | estimate_il.h:1252-1262, 1272 | Rader inner a pair/chain3 to 4096; Bluestein inner a pair below 256 | the inner's engine | none stated | L1d? | untested |
| 7 | estimate_il.h:136; estimate_nd_il.h:154, 366 | odd radices above 16 spill (25, 27 single; Rader inner odd above 13) | which radices are split | stated: AVX2's 16 vector registers hold a butterfly to about radix 16 | registers (32 at AVX-512) | untested |
| 8 | estimate_il.h:760; estimate_nd_il.h:152 | Rader window 32 columns | the Rader stage's window | stated: a scratch row of 512 B; rows of a radix-64 stage crowd one L1 set | L1 way size | untested |
| 9 | estimate_il.h:1310 | p - 1's largest factor <= 23 | Rader or Bluestein for a prime | the kernel list | code fact mostly | untested |
| 10 | estimate_il.h:313-469 | ZTURN-T chains of 8s then 4s; the tile widths | the 1D chain | *guess*: each radix's speed per lane width | lanes | holds: 1D to 2^18 at 1.03 |
| 11 | estimate_il.h:1508 | 16 KiB | a batch below it keeps one plan at T | stated: handing out work costs more than the work | the wake-up cost against the work rate (time, not cache) | untested |
| 12 | estimate_nd_il.h:122-123 | first stage 32 on narrow rows, 16 on wide | a power-of-two column chain | none stated | *guess*: registers (lanes), or the L1's associativity on the widest-stride stage | **fails**: 32 first 2048 x 128 3.0-5.2, 4096 x 64 3.1-4.2; 16 first 4096 x 256 1.9-2.0, 4096 x 1024 1.5-1.6, 8192 x 512 1.2-1.6. The 14900KF's planner leads with 32 itself (1.01, 1.03) |
| 13 | estimate_nd_il.h:306-312 | five chains by N (512, 1024, 4096, 8192) | the scrambled pass | 14900KF races written as a table | nothing (a table cannot carry) | untested |
| 14 | estimate_nd_il.h:440 | stages = log2 N / 4 narrow, / 3 wide | how many column stages | the radix-16 and radix-8 regimes | registers | partly: Zen 4's planner takes more, smaller stages (4.8.8.8 where the estimate takes 32.8.8) |
| 15 | estimate_nd_il.h:137-144 | skew to N1 16 (odd 31 on rows >= 128); turn from N1 128 at N2 <= 8; batched rows to N2 16 and 64; 8 rows a block | the 2D row route | 14900KF's planner rows | *guess*: a row's bytes against the L1d | mixed: 16 x 1024 1.37 in a round whose control read 0.56 (noise) |
| 16 | estimate_nd_il.h:153 | Rader inner radix above 32 | a bad inner chain | as 7 | registers | untested |
| 17 | estimate_nd_il.h:1070 | never on a power-of-two plane | the 3D flat tier | 14900KF: 1 MB 0.95-1.05, 2 MB 1.05-1.11 | the plane against the L2 | **fails**: 4 x 256 x 256 1.92 (a plane the size of Zen 4's L2; its planner takes the flat tier) |
| 18 | estimate_nd_il.h:246, 251-252 | 2 columns per thread; N1 >= 4 N2; first radix <= 32 | the 2D threaded turn and tile | stated: a shape ratio beat a cache rule there (1.236) | cores? | untested |
| 19 | estimate_nd_il.h:300; real_estimate_nd_il.h:714 | 32 KiB per worker | one thread or the team in 3D | none stated | *guess*: the L1d (2/3 of the 14900KF's, all of Zen 4's) | untested |
| 20 | c2c_estimate_lm_il.h:52-53 | N / 128 lanes; 8 lanes a strip | the turned prime pass; the strip | none stated | lanes, the line | untested |
| 21 | real_estimate_il.h:348, 387 | the real pair to 160, ZTT-r 512/16384-524288: 4 lanes only | the real engine | the engines are absent or refuse at 8 lanes (code fact); measured at 4 lanes only | — | untested |
| 22 | real_estimate_il.h:117, 157, 200, 277, 415 | ZTT-r tiles 512-3072; zrb inner below 2048; C2R route from 1024 | the real engines' details | 14900KF's planner rows | — | untested |
| 23 | real_estimate_il.h:489 | 128 KiB | odd Bluestein's own threads in a batch | none stated | *guess*: an L2 fraction (1/16 of the 14900KF's) | untested |
| 24 | real_estimate_lm_il.h:69-77 | 8 lanes per worker; tile from K 128 to 64 lanes; Rader window 8 | the real lane-major batch | stated for the 8: whole cache lines | the line | untested |
| 25 | real_estimate_nd_il.h:129-133 | five chains by N (512-8192) | the real column chain | 14900KF races written as a table | nothing | untested |
| 26 | real_estimate_nd_il.h:192-340, 657-723 | about 15 shape literals (N1 = 128, N1 in 8-64, 16.8 forced, N2 = 64 at 4 lanes, ...) | the real 2D/3D plan | 14900KF's planner rows | *guess*: a row's bytes against the L1d | untested |

## 5. What Zen 4 has tested

The two runs of 2026-11-03 (the power-of-two c2c sets, one thread; the second plugged in) reach items 1, 2,
4, 10, 12, 14 and 17 and the 2D band. Everything else (odd lengths, primes, Rader, the real transforms,
threads, batches) has no Zen 4 timing.

Of what they reach, the failures fall into two groups that the two machines cannot separate: the radix
choices (items 4, 12, 14) may follow the lanes and registers or the L1, and the size choices (item 2, the
band, item 17) may follow the L2 or the L3. One pattern runs through the second group: Zen 4's planner keeps
bands larger than its L2 in both the 2D column pass and the four-step, where the estimate's rules keep them
within it.

**Zen 4 with the AVX2 build (2026-11-03, 17:04, plugged in; 14 of the failed cells, their rows raced first
into the run's copy of `src/wisdom/Zen4/`).** Same lanes as the 14900KF, Zen 4's caches:

| cell | AVX-512 run (lowest round) | AVX2 run (lowest round) | Zen 4 planner at AVX2 |
|---|---|---|---|
| 2048 x 128 (32 first) | 5.18 (1.64) | 7.78 (3.14) | 4.8.8.8 |
| 4096 x 64 (32 first) | 4.16 (3.22) | 5.09 (3.79) | 4.8.8.16 |
| 4096 x 256 (16 first) | 1.86 (1.42) | 2.73 (2.43) | 8.8.8.8 |
| 8192 x 512 (16 first) | 1.56 (0.96) | 2.53 (2.27), control 1.33 | 8.8.16.8 |
| 4096 x 1024 (16 first) | 1.61 (1.11) | 1.40 (1.20) | 8.8.8.8 |
| 256 x 4096 (no band) | 1.35 (1.19) | 1.45 (1.16) | band 128 |
| 1024 x 4096 (no band) | 1.49 (1.39) | 1.48 (1.36) | band 128 |
| 512 x 8192 (no band) | 1.37 (0.69) | 1.29 (1.06) | band 64 |
| 4 x 256 x 256 (s=1) | 1.92 (1.12) | 1.22 (0.98) | s=2 |
| 128 (pair 4.32) | 1.17 (0.82) | 1.01 (0.83) | 16.8 |
| four-step, 4 lengths | 1.10-1.18 | 0.98-1.17, control 0.98-2.20 | 512, 4096, 2048, 2048 rows |

So the lanes do not cause the first-radix loss or the band loss: at 4 lanes on Zen 4 both lose as much or
more, and Zen 4's AVX2 planner again starts with 4 or 8 where the 14900KF's AVX2 planner starts 2048 x 128
with 32. They follow the machine's memory. The pair split at 128 does follow the lanes (fine at 4 lanes). The
four-step and the 3D cell are within this run's noise.

A lead to check at the desk, not a finding: the first-radix losses are on planes past the L2 (the radix-32
first stage held at AVX-512 on 2048 x 4 and 8192 x 8, planes within it), and Zen 4's L2 has half the ways of
the 14900KF's (8 against 16). The estimate does not read the associativity today.

## 6. Next (the owner decides)

1. **The switch points, desk work.** Both stores hold the planner's choices. For each failed rule, read where
   each machine's planner changes its choice and test candidate forms against both, before any timing.
2. **Zen 4 with the AVX2 build**, the same power-of-two sets, plugged in: the caches stay, the lanes drop to
   4. If item 12's radix-32 first stage stops losing, the lanes or registers cause it; if it still loses, the
   cache does.
3. **A 14900KF E-core, one thread:** the same AVX2 binary on a third cache layout (32 KiB L1d, 4 MiB L2 shared
   by four). Another core design, so a weaker control, but a third set of cache sizes at no cost.

A constant is set on one machine and checked unchanged on the other; a rule that fails there changes its
form, never its constant to suit it. Each rule records how many cells it was fitted on and how many checked
it, and the 14900KF's score must not get worse.

The target (the owner, 2026-11-03): within 10 % of the planner either way on both machines, from rules with a
reason in the hardware whose parameters the CPU reports at run time, or that hold on every platform. Parity
(1.000) on one machine is not the goal.

## 7. The first column radix, desk work (2026-11-03)

**The planners' choices.** The first radix of the banked column chain at every power-of-two plane, one thread
(`src/wisdom/14900KF/wisdom2_2d.txt`, 164 planes; `src/wisdom/Zen4_avx512/wisdom2_2d.txt`, 159):

- 14900KF: 2048 rows or more start with 32 (the blocked form) on rows up to 128 wide (256 at 8192 rows) and
  with 16 on wider rows; 1024 rows and fewer start with 8 (4 at 256 and 512 rows on narrow rows).
- Zen 4, AVX-512: 4 on rows up to 128 wide, 8 on wider rows, at every column length; one 16 (8192 x 128).
  Zen 4 at AVX2 (the 8 cells of section 5): 4 or 8.
- Both switch between rows of 128 and 256. The estimate's narrow test (a radix-16 butterfly across the row
  within the L1d) puts that switch at 192 on the 14900KF and 128 on Zen 4, so it fits both and cannot tell
  them apart.

With two machines, any one parameter that differs (L1d, L2, L3, the L1's or the L2's ways) fits "a large
first radix on the 14900KF, a small one on Zen 4" with one threshold. The planners' choices alone cannot
separate them.

**The mechanism the library names.** A wide column stage past the L2 is one sweep of the plane with 2R
streams (R legs read, R written): "a radix-32 wide stage runs 64 streams 16 KB apart, past the prefetchers
and the first-level TLB" (il2d_tier.h, the chain race's note). At a power-of-two row width the legs are also
a power of two apart: (N1 / R) x 16 N2 bytes, 128 KiB at 2048 x 128 and at 4096 x 64 with R = 32, the way size
of the L2 on both machines (2 MiB / 16, 1 MiB / 8), so every leg falls in the same L2 set and the same L1 set.

**The test that separates set conflicts from stream count:** the same column lengths at row widths that
break the alignment. At R = 32, 2048 rows: 128 wide puts every leg in one L1 set and one L2 set; 124 (124 KiB
apart) one L1 set, 32 L2 sets; 126 (126 KiB) two L1 sets, 32 L2 sets; 120 (120 KiB) one L1 set, 16 L2 sets
of two legs. 4096 rows: 64, 62, 60 likewise.

14900KF, one thread, 15 rounds, the rows raced first into `src/wisdom/14900KF` (5 new 2D rows):

| plane | the planner's chain | the estimate's | E/B (range) |
|---|---|---|---|
| 2048 x 120 | 8.16.16 | 32.8.8 | 1.129 (1.115-1.161) |
| 2048 x 124 | 8.8.32 | 32.8.8 | 1.108 (1.079-1.153) |
| 2048 x 126 | 8.8.32 | 32.8.8 | 1.041 (1.026-1.094) |
| 2048 x 128 | 32.8.8 | 32.8.8 | 1.011 (0.875-1.140) |
| 4096 x 60 | 8.8.8.8 | 32.8.16 | 1.129 (1.106-1.158) |
| 4096 x 62 | 8.8.8.8 | 32.8.16 | 1.138 (1.125-1.184) |
| 4096 x 64 | 32.8.16 | 32.8.16 | 1.023 (0.953-1.087) |

On the 14900KF the planner starts with 32 only at the aligned widths. Once the legs spread over the sets it
starts with 8, and the estimate's radix-32 start loses 4-14 %, a gap on this machine too. The aligned planes
run slower in every chain (2048 x 128 742 us against 568 us at 2048 x 120, 6 % less data): with every leg in one
set, fewer sweeps win, which is what the radix-32 start buys there. So the radix-32 start answers the
alignment, not the row's bytes against the L1d.

Zen 4, AVX2 build, one thread, 15 rounds, plugged in (the owner's run 2026-11-03 18:35; the 5 spread rows
raced first into the laptop's store; control B2/B 1.008):

| plane | the planner's chain | the estimate's | E/B (range) |
|---|---|---|---|
| 2048 x 120 | 8.4.4.16 | 32.8.8 | 1.759 (1.267-2.304) |
| 2048 x 124 | 8.16.4.4 | 32.8.8 | 1.352 (1.293-1.411) |
| 2048 x 126 | 4.8.8.8 | 32.8.8 | 1.300 (0.997-1.404) |
| 2048 x 128 | 4.8.8.8 | 32.8.8 | 6.600 (5.042-6.970) |
| 4096 x 60 | 4.8.8.16 | 32.8.16 | 1.927 (1.536-2.027) |
| 4096 x 62 | 8.8.8.8 | 32.8.16 | 1.306 (1.282-1.809) |
| 4096 x 64 | 4.8.8.16 | 32.8.16 | 7.485 (4.590-8.586) |

The same chain at the aligned width against the spread one (2048 x 128 / 2048 x 120, 4096 x 64 / 4096 x 60):

| machine | the estimate's radix-32 start | the planner's own chain |
|---|---|---|
| 14900KF (L2 16-way) | 1.16, 1.26 | 1.31, 1.40 (the banked radix-32 start); 1.09, 1.06 (a radix-4 start, below) |
| Zen 4 (L2 8-way) | 3.51, 4.31 | 0.95, 1.09 (starts with 4) |

- The 5-8x cliff is the legs sharing one L2 set: with the alignment broken, Zen 4's loss drops from 6.6-7.5x to
  1.3-1.9x. The L2 way size (128 KiB on both machines) says where it falls.
- What a shared set costs a radix-32 start differs: 3.5-4.3x on Zen 4, 1.2-1.3x on the 14900KF, whose L2 has
  twice the ways. A radix-4 start sidesteps it on both machines (the aligned planes run within 0.95-1.09 of
  the spread ones).
- At the spread widths the radix-32 start loses on both machines: 4-14 % here, 30-93 % on Zen 4. The ratios
  carry the row pass too (rows of 124 and 62 have the factor 31 and cost more); the extra time itself is about
  the same at 120, 124, 60 and 62 on Zen 4 (460-580 us), so the larger ratios at 120 and 60 come from cheaper
  rows, not from two legs per L2 set.

**Small starts on the 14900KF's aligned planes** (2026-11-03): the banked row's column chain replaced in a
scratch copy of the store and timed as the B arm (never banked), 15 rounds, E = the estimate's radix-32 start
in the same run:

| plane | the chain timed | B (us) | E/B (range) |
|---|---|---|---|
| 2048 x 128 | 8.8.32 (the planner's at 124 and 126) | 729 | 1.122 (1.049-1.441) |
| 2048 x 128 | 4.8.8.8 (Zen 4's) | 619 | 1.182 (1.145-1.267) |
| 4096 x 64 | 8.8.8.8 (the planner's at 60 and 62) | 636 | 1.281 (1.202-1.619) |
| 4096 x 64 | 4.8.8.16 (Zen 4's) | 630 | 1.365 (1.247-1.635) |

Every small start beats the radix-32 start here too, and Zen 4's chains are the fastest of the four: the
banked radix-32 rows at these planes are not the planner's answer today. All 19 radix-32-start rows and 6 of
the 8 radix-16-start rows among the 14900KF's power-of-two planes of 1024 rows or more were banked on
2026-09-24 (151 of its 328 one-thread natural out-of-place c2c rows carry that date); the rows raced on
2026-11-03 start with 8. The estimate's radix-32 / radix-16 start was copied from those rows, and its 14900KF
score on these planes was measured against them.

So with fresh rows the two machines agree: both start with 4 or 8, and the "large on one machine, small on
the other" split that no single parameter could separate was the old rows, not the machines.

**A radix-4 start at all 7 planes on the 14900KF** (2026-11-03): 4.8.8.8 on 2048 rows, 4.8.8.16 on 4096 rows,
wl = 256 (what the estimate emits), in a scratch copy, run alternately with an unchanged copy (the planner's
banked chain) twice, 15 rounds each; "anchored" divides each run by its own E:

| plane | the planner's chain | planner (us) | radix-4 start (us) | radix-4 / planner | anchored |
|---|---|---|---|---|---|
| 2048 x 120 | 8.16.16 | 585 | 594 | 1.014 | 0.987 |
| 2048 x 124 | 8.8.32 | 711 | 739 | 1.039 | 1.022 |
| 2048 x 126 | 8.8.32 | 631 | 641 | 1.014 | 0.994 |
| 2048 x 128 | 32.8.8 (09-24) | 798 | 637 | 0.798 | 0.741 |
| 4096 x 60 | 8.8.8.8 | 609 | 604 | 0.991 | 0.978 |
| 4096 x 62 | 8.8.8.8 | 875 | 917 | 1.048 | 1.045 |
| 4096 x 64 | 32.8.16 (09-24) | 888 | 645 | 0.726 | 0.750 |

Within 5 % of the planner's fresh rows at the five spread widths, 20-27 % faster than the old radix-32 rows at
the aligned ones, and faster than the estimate's radix-32 start at all seven (E/B 1.02-1.37 in the radix-4
runs). Not yet tried: rows of 256 and wider (Zen 4 starts those with 8, the 14900KF's fresh ones with 8 or 16).
The library already reads the L1d's ways (cpu_cache.h:154 CPUID, :197 Windows, :240/:262 Linux); the
estimate's machine record does not carry them.

**Built (2026-11-03, uncommitted):** the estimate reads the L1d's ways (`vfft_cpu_l1d_ways()`, cpu_cache.h, the
same refuse rule as the size; `est_machine_t.l1d_ways`), and the leading radix of a power-of-two column from 2^s1
is `est_col_lead`: the largest power of two R with 2R <= the ways, at least 4, and 4 when no ways are reported
(the owner: assume a small L1d). Both machines get 4. It reaches every caller of `est_col_chain` (2D, the 3D first
axis, real 2D and 3D, lane-major batches), not the 1D four-step, which builds its own chain. The estimate gate:
ALL PASS. The timing is next, on the power-of-two lists; at 2^s1 = 2048 on both machines it changes the 2D cells
2048x2048, 2048x4, 2048x128, 8192x8, 4096x64, 4096x256, 4096x1024 and 8192x512, and the real 2D cells with a
column of 2048 or more. On wide rows the chain after the 4 comes out 4.16.8.8 (4096) and 4.16.16.8 (8192): the
pool's 4-stage limit and the wide-row picker's largest-first order put the large radix second.
