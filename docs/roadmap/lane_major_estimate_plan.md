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
