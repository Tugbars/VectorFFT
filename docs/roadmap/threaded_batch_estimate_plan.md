# Plan: the batch estimate on several threads (complex, contiguous batches)

**For approval.** Stage 2 of the batch estimate. Stage 1, a contiguous batch on one thread, was built on
2026-10-31. The owner chose MKL's cache test as the starting rule. Nothing in the library changes before the build
step, and no rule is built without the owner's approval of it. Planning races bank into `src/wisdom/14900KF/`. Forced
arms run on a scratch copy of the store. The library's own guards hold the second hyperthread of every core in use
([`pool_sibling_guard.md`](../design/pool_sibling_guard.md)). Every run waits for the owner to say the machine is
quiet. The batch layout is always the caller's choice: this plan only decides how the threads share a contiguous
batch. Tools and evidence go to `docs/research/estimate_threaded/evidence/batch/`.

## 1. What we know

**How the door runs a contiguous batch of K transforms on T threads.** It has two ways, and it races them at create:
- **All threads on each transform in turn.** The plan of one transform at T threads, with its own threaded
  verdict, runs K times. The code calls this the serial loop in 1D (`_tc_mt_decide`, `vfft.c`) and the loop in 2D
  and 3D (`il/rank2/plane_queue.h`).
- **Whole transforms per thread.** Each worker runs its own one-thread copy of the plan.
  - In 1D (the slabs), each worker takes ⌈K/T⌉ consecutive transforms, so K = 9 on 8 threads runs 2, 2, 2, 2, 1
    transforms on 5 workers.
  - In 2D and 3D (the queue), min(T, K) copies pull transforms from a shared counter.
- A third way, K groups of T/K threads each, exists only for 3D real batches whose sides are powers of two.
- In 1D, a one-transform plan that is a four-step cannot be copied for a worker at T > 1. Its batch always takes
  the first way.

**The store holds no verdict for a complex batch.** There are 0 `tcmt` rows for C2C (the 30 that exist are real
batches), and the 2 C2C `pq` rows are scrambled order. The one-transform rows the cells below need are mostly there.
Missing: 6 of the 15 1D sizes at 8 threads, and every 1D size in place at 8 threads; 2 2D shapes at one thread (the
queue's copies use one-thread plans); 5 3D shapes at 8 threads and 2 at one thread; nearly every size at 4 threads.
Each missing row is raced before its batch.

**The planner's own batch races are short.** The 2D and 3D race keeps the minimum of 3 samples per way
(`_pq_mt_race`); the 1D race keeps the minimum of 7 to 9 (`_tc_mt_decide`). The 1D verdict is filed without the
thread count: raced at one T, it is replayed at any T. The 2D and 3D verdict is filed on the row of the one-transform
plan at T, with K.

**MKL's cache test** (`docs/research/mkl_dfti/batch_1d.md`, `md_2d_3d.md`). It compares one transform's bytes (16
per point, counted twice out of place) with the last-level cache divided by T.
- 1D: from K = 3, all threads on each transform once N ≥ L3 / T / 16 (halved out of place); below that, each
  thread takes a contiguous chunk of ⌈K/T⌉ transforms.
- 2D: groups = L3 / bytes per transform, between 1 and K; the threads are split evenly among the groups.
- 3D: whole transforms per thread while one fits L3 / T and K ≥ T, the K mod T left over done by all threads
  together; otherwise the threads split the planes.

On this machine (L3 36 MiB) at 8 threads, a thread's share is 4.5 MiB. MKL's 1D switch is at 294,912 points in
place and 147,456 out of place.

## 2. The cells (8 threads, out of place, natural order)

Regions by one transform's bytes:
- up to 48 KiB: fits one core's L1d;
- 48 KiB – 2 MiB: fits one core's L2;
- 2 – 4.5 MiB: fits one thread's share of the L3. MKL's in-place test keeps whole transforms per thread here;
  its out-of-place test switches from 2.25 MiB;
- 4.5 – 36 MiB;
- past 36 MiB.

| Region | 1D N | 2D | 3D |
|---|---|---|---|
| up to 48 KiB | 64, 1000, 2048 | 8x8, 32x32, 16x128 | 4x4x4, 8x8x8, 16x16x8 |
| 48 KiB – 2 MiB | 4096, 12005, 65536, 65537 (prime), 100000 | 64x64, 100x100, 256x256, 8x8192 | 16x16x16, 30x30x30, 32x32x32, 64x32x32 |
| 2 – 4.5 MiB | 131072, 196608, 262144 | 360x360, 512x512 | 64x64x32, 60x60x60, 64x64x64 |
| 4.5 – 36 MiB | 524288, 1000000, 2097152 | 1000x1000, 1024x1024, 8x131072, 2048x1024 | 128x64x64, 100x100x100, 128x128x128 |
| past 36 MiB | 4194304 | 2048x2048 | 256x128x128 |

Batch sizes: K = 2 and 4 (below T), 8 (equal to T), 9 (one past T, the uneven case) and 32 (four times T). A cell
whose batch passes 640 MiB is dropped: K = 32 at 32 MiB and above, 6 cells. That leaves 209 cells (43 sizes x 5
batch sizes, less those 6).

| Cells | What | Count |
|---|---|---|
| main | the table above: both ways forced, the planner's verdict, a control | 209 |
| in place | 5 sizes per rank in the 2 – 36 MiB regions, K = 4 and 9: MKL's test moves with the placement | 30 |
| 4 threads | 5 sizes per rank around 9 MiB (a thread's share at 4 threads), K = 4 and 5: does the switch move with L3 / T? | 30 |
| unseen | after the rule is frozen: 20 per rank drawn by a script with a fixed seed across the regions, K = 3, 6, 12 and 24 (none of them in the main cells); the estimate's batch plan against the planner's | 60 |

## 3. Steps

| # | Step | Time |
|---|---|---|
| 1 | Tools in the scratchpad (no quiet machine needed): the cell-list script; a create tool that races a batch at T into the real store and prints the verdict, the seconds and what raced; a timing tool (below); the scoring script | about 3 h |
| 2 | Pilot: 2 cells per rank, the largest batches and the sizes without rows | about 20 min |
| 3 | Race the main, in-place and 4-thread cells into the real store. About 30 missing one-transform rows race first (0.5 – 2 min each); each batch verdict takes a few seconds | 45 – 75 min |
| 4 | Time the same cells on a scratch copy of the store | 30 – 45 min |
| 5 | Score each rule of section 4 by the time it loses against the faster way, by region, K and rank; count how often the planner's short race picked the slower way | about 1 h, no machine |
| 6 | The owner approves the rule | |
| 7 | Build: the estimate writes the batch verdict (1D: the `tcmt` row at q = K; 2D and 3D: `pq` and `pqn` on the row at T) and, for the queue, the one-thread rows of its copies. `VFFT_TCMT`, `VFFT_NO_TCMT` and `VFFT_PQ_NO_MT` join the environment guard, since they steer a batch at T > 1. The plan check reads the way and the copies; the refusal at T > 1 is lifted; a gate section at 8 threads; the sketches | about half a day |
| 8 | The unseen cells: race them, then time the estimate's batch plan against the planner's | 45 – 60 min |

Quiet machine in total: about 1.5 – 2.5 h for steps 2 – 4 and about 1 h for step 8, in two windows.

The timing tool holds, in one process, inside one measurement scope (so the library holds its guards):
- both ways forced at create (`VFFT_TCMT=0|1` in 1D, `VFFT_PQ_NO_MT=1|0` in 2D and 3D);
- the planner's verdict, and a second planner arm as control;
- 15 rounds in a seeded order, threaded arms never paused;
- the engagement counters (`vfft_tc_mt_dispatches`, `vfft_pq_mt_passes`);
- a check that no arm measured anything at create.

## 4. Rules to test

- **MKL's test as it is.** Whole transforms per thread while one transform's bytes (counted twice out of place)
  fit L3 / T, in 1D from K = 3 and in 3D only when K ≥ T; otherwise all threads on each transform.
- **The same test with its bound fitted** on the main cells, by time lost.
- **Whole transforms per thread whenever the one-transform plan at T runs on one thread anyway.** The estimate
  already knows this, and the first way then leaves the other threads idle.
- **K below T** (only K workers busy) and **K just past T** (the slowest worker runs ⌈K/T⌉ transforms): a clause
  each, only if the main cells show the plain test losing there.

## 5. Decisions for the owner

1. The 4-thread cells: they test whether the switch moves with the thread count (the test divides the L3 by T).
   They add about 20 – 40 min, mostly races of new rows at 4 threads.
2. The pass bar for the unseen cells. Suggested, as for 3D: geomean at most 1.05 per region, every cell above 1.10
   explained.
3. MKL is not timed in this study. Say if you want MKL's own batch at 8 threads beside ours.

## 6. Door gaps to report, not fill

- Several transforms at once with a few threads each (MKL's 2D groups) exists only for 3D real batches whose sides
  are powers of two.
- MKL's 3D split (whole transforms per thread, then the K mod T left over on all threads) has no counterpart.
- A 1D four-step plan cannot run as a one-thread copy at T > 1, so its batches always take the first way.
- The 1D batch verdict is filed without the thread count, so a verdict raced at 8 threads is replayed at 4.

## 7. Risks

- Memory: 31.8 GB, about 11 GB free on 2026-10-31. The largest cell (9 x 64 MiB out of place, four arms, each with
  its copies' scratch) needs about 3 – 5 GB.
- The two ways are timed with the planner's one-transform plans; the estimate's own plans may move the switch a
  little. The unseen cells check the whole.
- Noise: a control arm in every process, and threaded arms are never paused. The races also wait for a quiet
  machine.
- The races of missing one-transform rows take most of the time; the pilot sets the budget and drops the largest
  cells if needed.
