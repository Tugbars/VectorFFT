# Plan: the estimate for lane-major batches (complex)

**For approval.** Stage 3 of the batch estimate (the owner, 2026-11-01: "yes", in this round, as part of the
C2C estimate for 0.7). Stages 1 and 2, contiguous batches on one and on several threads, were built on
2026-10-31. Today `VFFT_FLAG_ESTIMATE` refuses a batch that names `VFFT_BATCH_LANE_MAJOR`. Nothing in the library
changes before the build step, and no new rule is built without the owner's approval. Planning races bank into
`src/wisdom/14900KF/`; forced arms run on a scratch copy of the store. Every run waits for the owner to say the
machine is quiet. Scope: C2C at ranks 1, 2 and 3 (the library has no estimate for real transforms). Tools and
evidence go to `docs/research/estimate_threaded/evidence/lane_major/`.

## 1. What we know

**How the door plans a lane-major batch of K transforms.** Element e of transform t sits at z[2 (e K + t)], so
every axis is a column pass over lanes:
- **1D** (`il/rank1/lm_batch_il.h`): one column pass of N rows over K lanes.
- **2D** (`il/rank2/lm_batch2d_il.h`): one column pass of N1 rows over N2 K lanes, then N1 blocks, each the 1D
  lane-major batch at (N2, K) on one thread, in place.
- **3D** (`il/rank3/lm_batch3d_il.h`): one column pass of N1 rows over N2 N3 K lanes, then N1 blocks, each the 2D
  lane-major batch at (N2, N3, K).

Each create plans on the batch's own row, `t=c2c n=<shape> q=K ord=<order> place=oop lay=lm nthreads=T` (both
placements share it):
- the column builder's verdicts, as for a 2D plane's column pass: the chain, its forms, a Rader stage, Bluestein,
  the turned prime pass and its plan (`tpc_`);
- the serial lane tile `sw`: the door races the widths of its ladder; for a smooth chain the walk is no arm, a
  chain with a Rader stage and a batch of fewer than 8 lanes walk (`vfft_policy_lm_serial_tiles`);
- at T > 1, the threaded verdict: the threaded walk, the strips over each worker's lanes, or the strips at a
  width of the ladder (`cmt`, `mtarm`, `msw`);
- at rank 2 and 3, the block's plan under `rp_` (the lower-rank lane-major batch's own row). At T > 1 the blocks
  run over copies of it when the column pass threads.

The banded walk and the row fusion of the 2D plane are off.

**The store holds one lane-major C2C row** (256 x 4). Every cell races from cold.

**What the estimate already has.** `est_col_plan` plans a column pass at (N, w) in either order, with the Rader
stage, Bluestein and the turned prime pass. `est_mt_2d` decides the 2D plane's threaded column pass: threaded
once the plane passes a share of the L1d, then the strips at `est__msw_2d`'s width (or the tile, the turn and the
skewed pass, which a lane-major batch does not have). The writers `est__put_col` and `est__put_child` use the
door's own bank functions.

## 2. What carries over, and what is new

| Decision | Carried over (no new rule) | New: the owner's after the study |
|---|---|---|
| The chain, forms, Rader stage, Bluestein, turned prime pass | `est_col_plan` at w = the lanes (K; N2 K; N2 N3 K), in the door's order for the first axis (`vfft_policy_lm_axis_nat`) | |
| The blocks' plan (`rp_`), rank 2 and 3 | the lower-rank lane-major estimate, one thread, in place | |
| Threaded or not, and which arm, at T > 1 | `est_mt_2d`'s test and its strips at `est__msw_2d`'s width | only if the study shows the 2D rule losing: the threaded walk, or the strips over each worker's lanes |
| The serial lane tile `sw` | none: the 2D plane's estimate leaves its column pass untiled | a width rule (the door always tiles a smooth chain) |

The build writes `sw=0` (the walk) until the owner approves a width rule; the study times the walk against every
width of the ladder.

## 3. The cells (out of place, natural order)

| Rank | Shapes | K | Threads |
|---|---|---|---|
| 1D | N = 16, 64, 256, 1024, 4096, 16384, 65536; 100, 1000, 12005 (smooth, not powers of two); 131, 1031, 4099 (primes: a Rader stage or Bluestein) | 4, 32, 256 (K = 2 at the primes: the turned prime pass) | 1 and 8 |
| 2D | 8x8, 32x32, 64x64, 16x256, 256x16, 100x100, 131x64 | 2, 8, 32 | 1 and 8 |
| 3D | 8x8x8, 16x16x16, 32x32x32, 8x8x64, 30x30x30 | 2, 4, 8 | 1 and 8 |

A batch past 256 MiB is dropped. About 190 cells, plus 20 in place (the door shares the row between placements)
and 60 unseen cells drawn after the rules are set (20 per rank, fixed seed, none of the shapes above).

## 4. Steps

| # | Step | Time |
|---|---|---|
| 1 | Tools in the scratchpad: the cell list; a create tool that races the planner's lane-major batch into the real store and prints its recipe; a timing tool (one process, the estimate's plan against the planner's, a second planner arm as control, 15 rounds in a seeded order, threaded arms never paused, a check that no arm measured at create); the forced arms for the serial tile (each width of the ladder and the walk); the scoring | about 2 – 3 h |
| 2 | Build without new rules: the estimate writes the lane-major rows (the column pass, `sw=0`, at T > 1 the 2D strips rule, the blocks' plan under `rp_`, `tpc_`); the refusal is lifted; gate sections at ranks 1 – 3, one and eight threads (against the naive DFT and K runs of the one-transform plan; nothing measured; the plan served is the recipe) | about half a day |
| 3 | Pilot: 2 cells per rank | about 15 min |
| 4 | Race the planner's batches into the real store (cold: each create races its chain, forms, tile and threaded verdict) | 45 – 90 min |
| 5 | Time the estimate's plan against the planner's, and the forced serial tiles | 30 – 60 min |
| 6 | Score: the estimate against the planner by region, rank and thread count; the serial tile rule candidates by time lost | about 1 h, no machine |
| 7 | The owner approves the tile rule (and a threaded-arm rule if step 6 shows one is needed) | |
| 8 | Build the approved rules; the unseen cells; a section on `estimate_performance.md` | about half a day |

Quiet machine in total: about 2 – 3 h, in two windows.

## 5. Decisions for the owner

1. The scope is C2C lane-major at ranks 1 – 3, as above.
2. The thread counts are 1 and 8. 4 threads would add about 40 minutes.

The owner approved the plan on 2026-11-01 ("yes please"; the thread counts stay 1 and 8, 4 can be added).

## 6. Built (2026-11-01)

Step 2: `il/estimate/c2c_estimate_lm_il.h` (the recipe `est_plan_lm`, the pre-flight, the row writer, the plan
check `est__check_lm`, the build), wired from `_vfft_create_estimate`; the front door no longer refuses a
lane-major batch under the flag. `estimate_gate` sections LM1, LM2, LM3 (one thread) and LM1T, LM2T, LM3T (8
threads): 540 lane-major batches built with nothing measured, served as the recipe wrote them, and correct against
the one-transform plan lane by lane; 102 of the 270 at 8 threads ran their column pass threaded. The gate passes in
full (15,907 checks).

## 7. The study (2026-11-01, 01:10 – 01:59): verdict and proposals

Steps 4 and 5 ran on a quiet machine. **One thread is complete** (88 cells). **Eight threads is not:** the planner's
own lane-major create crashed at 12005 x 32 (see the defects), so 28 of 88 cells were timed, all 1D. Tools and logs:
`docs/research/estimate_threaded/evidence/lane_major/`.

**Verdict at one thread** (88 cells; the control P2/P 1.003, 5 cells off by more than 5%): the estimate's batch
takes 1.023 times the planner's at the geometric mean and 1.003 at the median; 25 within 5%, 29 more than 5%
faster, 34 more than 5% slower. By rank: 1D 1.093, 2D 0.977, 3D 0.909. By the batch's bytes: up to 48 KiB 0.895,
48 KiB - 2 MiB 1.050, 2 - 36 MiB 1.083, past 36 MiB 0.934. The losses gather in two places:
- **A prime N at small K.** The estimate takes the Rader stage; the planner takes Bluestein with the turned prime
  pass: 1.75 (131 x 2), 1.97 (1031 x 2), 2.12 (4099 x 2), 1.18 - 1.57 at K = 4. At K = 32 and 256 the planner
  takes the Rader stage too (1031 at 8 threads).
- **Some smooth 1D chains.** The estimate leads with large radices where the planner leads with 4 or 8: 256 x 32
  (16.16 against 4.64) 1.62, 1024 x 32 1.29, 4096 x 4 1.25, 65536 x 4 and x 32 1.24, 256x16 x 2 and x 8 1.56 - 1.75.

The estimate is faster at the smallest batches (8x8, 8x8x8, 64 x 32: 0.53 - 0.73), where the planner tiles.

**The serial lane tile:** on the 57 cells whose chain tiles, the walk (the build's `sw=0`) loses 1.061 against the
best width forced per cell (E/P 0.966; the best width would give 0.913). No simple rule beats the walk: the
narrowest width 1.292, the widest 1.111, the widest within the L1d 1.149, L1d / 2 1.161, L2 / 8 1.129, L2 / 4
1.121, the 2D strips' width rule 1.119.

**Verdict at 8 threads** (28 cells, 1D only; the control 1.000): 1.291 at the geometric mean, 1.073 at the median.
The carried-over strips rule fails where each worker gets few lanes: at K = 4 the strips are 3.5 - 5.8 times
slower than the planner's threaded walk (1000, 1024, 4096, 16384, 65536 x 4), and at a long column with K = 32 1.60
(65536 x 32). At K = 32 and 256 on shorter columns the two are close (0.98 - 1.24).

**Defects found** (both fixed on 2026-11-01, below):
1. **A crash in the planner's lane-major create** (12005 x 32 and 4099 x 32 at 8 threads; reproduced in three
   processes; gdb: a pool worker in `_il2d_cmt_body` -> `_il2d_blu_cols_range` -> memset). `_lm_race`
   (`il/rank1/lm_batch_il.h`) added its plain threaded arm for a column through the turned prime pass; once the
   lanes were wide enough to split, the threaded column pass ran the Bluestein columns over tables the turned pass had
   dropped. The 2D door keeps such a column serial. A user who planned these batches by measuring crashed.
2. **The lane-major race picked threaded where serial wins.** At 100 x 4 on 8 threads it banked the threaded walk,
   6.7 times slower than serial (also 16 x 4, 256 x 4).
3. The study tool lost the 131 x 4, 32 and 256 races (its parser; fixed).

The owner's decisions (2026-11-01): keep the serial tile at the walk; fix 1 and 2; give the race the 1D batch race's
protocol; a prime at small K as Bluestein with the turned pass, the Rader stage above; the strips only when each
worker gets enough lanes; try the unbanded chain pick in 1D -- "today I will verify the results".

## 8. The fixes and the rules (2026-11-01, built overnight; commit 3ff2d127)

**The planner (`_lm_race`, `il/rank1/lm_batch_il.h`):**
- A column through the turned prime pass gets no threaded arm and no threaded replay. The four cells that crashed
  build (12005 x 32 and x 256, 4099 x 32 and x 64 at 8 threads).
- The real cause of defect 2 was a missing arm, not the sample size: a c2c batch racing its tile widths has no walk
  arm, and with fewer than 8 lanes there is no width either, so only threaded arms ran. The walk is now the serial
  arm wherever no width exists.
- The 1D batch race's rounds: the minimum of 7 (9 in place), the order reversed every other round. The samples were
  already longer than the 1D race's (about 0.3 ms each, `_il2d_race_reps`) and stay so. The lane-major real batches
  share this race and get the same.
- `lmbatch_gate` passes. The store's lane-major rows of 2026-11-01 (raced before the fix) were removed and raced
  again: 212 study cells and 60 unseen ones, 247 rows in `src/wisdom/14900KF`.

**The estimate (`il/estimate/c2c_estimate_lm_il.h`; each rule fitted on recipe variants of the estimate's own
plan, built by `est_build_lm`, against the best variant of each cell):**
1. **A prime column on few lanes:** a column the chain plan gives a Rader stage runs as Bluestein with the turned
   prime pass while its lanes are at most max(2, N / 128). The switch grows with N: 131 at 2 lanes, 1031 at 4
   (a tie at 8), 4099 up to 32 at 8 threads and at every K on one thread. Loss against the better form over 50
   prime cells: 1.009 (a fixed 4 lanes: 1.039).
2. **The strips at T > 1** only when every worker gets a vector of lanes (lanes at least 2 T on AVX2), on a chain
   with no Rader stage whose 8-lane strip fits the L2; else the threaded walk. Loss against the best of serial, walk
   and strips over 103 cells at 8 threads: 1.019 (the carried-over rule: 1.188).
3. **The 1D chain past the L1d:** the unbanded walk's pick (`est__col_unbanded`). Over the 40 one-thread cells where
   the picks differ: 1.091 -> 1.018 against the better; at 8 threads 1.046 -> 1.042. Below the L1d the chain plan's
   pick stays (the unbanded one lost 1.215 there).
`estimate_gate` passes in full with the rules (the LM sections cover a prime switch, both threaded forms and the 1D
chain).

**The estimate against the planner** (the planner raced with the fixed race; the control 0.997 - 1.002):

| | Before the rules | With the rules |
|---|---|---|
| Study, one thread (106 cells) | 1.046, median 1.018 | 0.985, median 1.001 |
| — 1D / 2D / 3D | 1.126 / 0.957 / 0.894 | 1.026 / 0.932 / 0.910 |
| Study, 8 threads (106 cells) | 1.189, median 1.038 | 1.004, median 1.004 |
| — 1D / 2D / 3D | 1.317 / 1.034 / 0.982 | 1.007 / 1.013 / 0.978 |
| Cells above 1.10 (one thread / 8 threads) | 38 / 41 | 19 / 21 |

**The unseen cells** (60, drawn after the rules with a fixed seed: 20 per rank, none of the study's shapes, K the
study did not use, 32 at one thread and 28 at 8): 0.991 at the geometric mean, 0.994 at the median; 24 within 5%, 18
more than 5% faster, 18 more than 5% slower, 11 above 1.10. One thread 0.962, 8 threads 1.025; 1D 1.004, 2D 0.940,
3D 1.030. By the batch's bytes: up to 48 KiB 0.993, 48 KiB - 2 MiB 0.924, 2 - 36 MiB 1.102 (above the 1.05 bar),
past 36 MiB 0.974.

**What the cells above 1.10 show** (study and unseen):
- **A prime whose p - 1 is smooth** wants the Rader stage sooner: 2003 x 6 at 8 threads, 2.07 (2002 = 2 7 11 13;
  the Rader stage threads, the turned pass runs serial). The study's primes that favour the turned pass have a
  large prime in p - 1 (1030 = 2 5 103, 4098 = 2 3 683); 131 (130 = 2 5 13) switches at 2 lanes. **Built
  (the owner, 2026-11-01; commit 3ff2d127):** the bound max(2, N / 128) only where p - 1 has a prime factor above
  47 (`est__lm_tpc_lanes`), else 2 lanes. It changes no study cell and, of the unseen ones, only 2003 x 6: 2.07 ->
  1.13 (both now the Rader stage; the planner threads it as strips, the estimate as the threaded walk, which the
  strips rule keeps for a Rader column; strips there time at 1.00).
- **3D at 8 threads:** the strips at 256 lanes where the planner takes 16 - 64 or the walk (32x24x60, 24x24x40,
  80x24x60: 1.12 - 1.35). The strips' width rule (`est__msw_2d`) is the 2D plane's; a narrower bound for lane-major
  passes is the next fit.
- **The first axis's chain at rank 2 and 3** (256x16 x 2 and x 8: 1.60 - 1.84; 256x197 x 3: 1.69): the unbanded
  pick applies at rank 1 only, as asked.
- **16384 and 65536 in 1D** (1.12 - 1.47 at both thread counts): the planner pairs a radix of 32 or 64 with small
  ones (8.4.64.8, 16.8.8.64), the unbanded pick balances 16s.
- 30x30x30 (1.19 - 1.55, the planner tiles at 16 lanes; the walk stays as decided) and 6007 x 96 at 8 threads
  (1.72, both the Rader stage: its inner and window).

**Verified 2026-11-01 (12:35 - 13:06), with the p - 1 test in:** the CMake library builds with no new warning;
`lmbatch_gate`, `estimate_gate`, `api_matrix_gate`, `il3d_real_batch_gate`, `vfft_tcbatch_gate` and
`dfti_shim_gate` pass; the store's lane-major rows have no duplicate key, no threaded row on a column through the
turned prime pass, and 100 x 4 at 8 threads holds the serial verdict. The unseen 60 timed again: 0.990 at the
geometric mean, 1.007 at the median (the control 0.999); 25 within 5%, 15 more than 5% faster, 20 more than 5%
slower, 11 above 1.10. Besides 2003 x 6, the cells that moved more than 5% kept their plans (run-to-run spread).
The 11 above 1.10 are the gaps listed above: 6007 x 96 (1.71), the first axis's chain at rank 2 (256x197 x 3,
1.69; 32x256 x 24, 1.19), the 3D strips at 256 lanes (32x24x60 x 6, 24x24x40 x 12, 40x40x64 x 12, 32x80x40 x 12:
1.15 - 1.37), 2003 x 6 (1.13), two 3D cells at one thread with the planner's column chain (8x48x8 x 12, 40x20x32 x
12: 1.18 - 1.30, the difference in the forms or the block child) and 48 x 3 (1.10). The section on
`estimate_performance.md` is written.

Next (the owner's): the 3D strips' width and the first axis's chain at rank 2 and 3 are the largest gaps left.
Tools and logs: `docs/research/estimate_threaded/evidence/lane_major/`.

## 9. A Rader stage among others takes the strips (2026-11-03)

The strips rule of section 8 gave every column with a Rader stage the threaded walk; the study's prime columns were a
Rader stage alone (131, 197, 2003, 6007), where the walk won. The real batch's study found a Rader stage among other
stages walking 3.3 – 5.7 times slower than the planner (`real_estimate_plan.md` section 17), and the C2C batch was
timed for it: 23 shapes whose column is a prime 53 – 1031 times 2 – 12 (rank 1 at K 32 – 1024, rank 2 and 3 at K 4 –
16, three of them in place too), 26 cells at 8 threads, the planner's rows raced into `src/wisdom/14900KF` first
(none was banked; the planner took the strips at every cell but 1018 x 96, where it runs Bluestein with the turned
pass). The estimate's recipe with the threaded form the only change (scratch `fix3/`, `lmst time`):

| Cells | Count | The walk (the rule as it was) | The strips | Serial | Control |
|---|---|---|---|---|---|
| rank 1 | 9 | 2.439 | 0.768 | 4.237 | 0.999 |
| rank 2 and 3 | 17 | 4.844 | 0.935 | 5.564 | 1.011 |
| all | 26 | 3.820 | 0.874 | 5.063 | 1.007 |

The walk ran 4.3 – 5.8 times the planner at rank 2 and 3 (2062x8 x 4: 1.31) and up to 7.6 at rank 1 (212 x 1024);
the strips were within 5% of the planner or faster at all 26 (2062 x 64 and 1018 x 96: 0.29 and 0.33). **Built:** a column
with a Rader stage among two stages or more takes the strips at `est__msw_2d`'s width under the same lanes and L2
bounds (`est__lm_threaded`, `c2c_estimate_lm_il.h`); a Rader stage alone keeps the walk. The real batch's own clause
(`est__lmr_rader_strips`) is the same test and now lives there, so the real batch's recipes do not change. None of
the 2026-11-01 study's or unseen cells has such a column, so their numbers and the performance page's section stand.
`estimate_gate` gained 212 (rank 1), 106x16 and 106x4x4 in its lane-major sections and passes in full (31,642 checks,
2026-11-03 08:34).
Timed again with the rule built (08:39): the estimate serves the strips at all 26 cells, 0.874 against the planner
(rank 1 0.776, rank 2 and 3 0.930; the worst cell 1.045; the control 1.006).
