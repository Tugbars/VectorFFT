# Plan: the estimate for real transforms (R2C and C2R)

**For approval; the owner's answers to its decisions (2026-11-01) are in section 5.** The owner, 2026-11-01: "we will release 0.7 after reflecting the same estimate to R2C/C2R".
"The same estimate" is the C2C one as it stands on 2026-11-01: one transform at ranks 1 – 3, on one thread and
on T threads, contiguous batches and lane-major batches. Today `vfft_create` refuses `VFFT_FLAG_ESTIMATE` on R2C
and C2R (`src/core/vfft_api.h`, "plans VFFT_C2C only").

The owner's standing decisions for this work:
- One library file for both real directions, and the least call overhead (2026-10-28). The estimate replays its
  rows through the door, so an estimated plan is the planner's own plan object with the same execute.
- R2C and C2R are separate pieces. A rule fitted on one direction leaves the other as it is, and no new wisdom
  token is added unasked.
- The thread rulings of 2026-10-29. `nthreads` is a ceiling. Thread rules are simple formulas from the data's
  bytes against the caches and the count of independent pieces, never fitted to how 8 threads scaled on this CPU.
- Natural order only (2026-11-01): scrambled order is a beta feature, not fully fleshed out.
- A cell banked before does not race again; the library races only where the banked cell is missing or the user
  asked for recalibration (2026-11-01).

Nothing in the library changes before a build step, and no new rule is built without the owner's approval.
Planning races bank into `src/wisdom/14900KF/`; forced arms run on a scratch copy of the store. Every run waits
for the owner to say the machine is quiet. Tools and evidence go to `docs/research/estimate_real/` (git-ignored).

## 1. What we know

### How the doors plan a real transform

**1D, one transform** (`il/real/real_create_il.h`; even N `zrp_build.h` `_real_il_build`, odd N `odd_build.h`).
The engine is raced among those a fixed rule admits. Each direction has its own row,
`t=r2c|c2r n=N q=1 ord=nat place=oop|ip [nthreads=T]`, and only the default order is served.

| Engine | Admitted | Its row | Threaded form (T > 1) |
|---|---|---|---|
| `zrm`, the small real kernel | N ≤ 64 with a kernel | `eng=zrm` | none |
| `zrp`, the real pair (4 lanes) | even N, every even pair in 4..64, N/2 not a power of two ≥ 2048 | `pair= leaf=` | none |
| `zttr`, real ZTURN-T (4 lanes) | even N ≥ 64 with a chain at N/2 | `chain= tile= stk= [mt=]` | `mt=` |
| `zr2c`, the complex half | every even N | `route=` plus the half's C2C recipe (`il_*`, `fs_*`) | `fold=mt`; a prime half also `il_mt=` |
| `zfsr`, the real four-step | N/2 a power of two in 2^18 .. 2^24 | `split=` plus the four-step child `fs_*` | the child at T |
| `zrf`, the odd chain | odd N ≥ 9 with a chain of 2 .. 10 stages | `chain= msz= tile= [mt=]` | `mt=` |
| `zrb`, odd Bluestein | odd N > 64 with no `zrf` chain | `m= in= in_sh= in_tw= [mt=]` | `mt=` on a ZTURN-T inner |

**1D batches.** A real batch is **lane-major by default** (`include/vfft.h`, `VFFT_BATCH_DEFAULT`).
- **Lane-major** (`il/rank1/lm_batch_il.h`): even N ≥ 4 (odd N is refused). The batch packs to M = N/2, runs
  one column pass of M rows over K lanes (the 2D tier's column builder), then folds. Both directions and both
  placements share one row, `t=r2c n=N q=K ord=nat place=oop lay=lm [nthreads=T]`, with the column builder's
  tokens, `sw=`, and at T > 1 `cmt mtarm msw` (the lane-major race of 3ff2d127).
- **Contiguous**, when named (`vfft.c`): the plan of one transform, run once per transform; at T > 1 the batch's
  own row per direction, `eng=tcb tcmt=`.

**2D, one plane** (`il/rank2/fft2d_create_il.h`, `il2d_real_plan.h`, `il2d_real_axis.h`, `il2d_real_fuse.h`,
`il2d_real_pitch.h`). N2 is the real axis; the spectrum is N1 × (N2/2 + 1). R2C runs the rows, then the column
pass over N2/2 + 1 columns. C2R runs the columns first, into a private plane, then the rows. Both directions share
one row (`t=r2c`); the C2R verdicts carry `_c2r`. The create races:
- the column pass (the column builder: `chain forms bluforms blu rd rdw tpc`, shared by both directions);
- the row engine `rx` (the row child `door`, the rows kernel `lm`, `zr2c`, `zrm`, `zrp`, `zttr`; at odd N2
  `zrf`, `zrb`), each at four stack states `rxs`;
- the column form `cx` (plain, strided, staged, a leaf) and its stack states `cxs`;
- whole-transform forms: the fused walk `tf`; for R2C the real axis on N1 `raxis` and the skewed plane `csk`;
  for C2R the private plane's pitch `cxp_c2r` and the destroying form `cxd_c2r`;
- children: `rp_` (at even N2 the 1D real contiguous batch of N1 at T; at odd N2 a C2C child) and `rx_`.

At T > 1 the column pass races serial, threaded, strips and strip widths (`cmt mtarm msw`), and the rows run on
clones. The whole-transform forms are one-thread only. With one thread, no threaded columns, `rx=lm` and a leaf
`cx`, the door binds its lean two-kernel executor.

**3D, one volume** (`il/rank3/fftnd_real_il.h`). N3 is the real axis. The plane child is the 2D real plan at
(N2, N3) on one thread (`plane_`). Axis 0 is a column pass of N1 rows over N2 (N3/2 + 1) lanes (the column
builder; pay-once axis 1: `chain1`). The create races the structure `s` (the child, pay-once, or for C2R out of
place the band) and the forms `nf` (in place or strips) with `nsw`; the C2R ones carry `_c2r`. At T > 1 the
plane arm runs plane ranges on clones, and axis 0 runs by blocks, strips or stage-wise (`cmt cmtt cmts cmtf ...`).

**2D and 3D batches.** Contiguous by default: the plane queue (loop, queue, or for 3D groups), whose verdict is
its own row per direction (`pq=`). Lane-major when named (even N2 ≥ 4, or even N3 ≥ 4): the lower-rank real
lane-major batch as blocks, and a column pass over N1 rows; one row for both directions.

### The store (`src/wisdom/14900KF`)
- 1D: 3,067 R2C and 2,069 C2R rows at one thread (R2C: `zr2c` 2,000, `zrb` 666, `zrf` 331, `zttr` 31, `zrm`
  20, `zrp` 15, `zfsr` 4); 33 per direction at 8 threads. The 8 `zfsr` rows (2026-09-30) hold the split but no
  four-step child (`fs_*`). The reader counts such a row as missing, so the first create at each races the whole
  cell and banks a complete row (to-do item of 2026-10-28).
- Batches: 15 contiguous verdicts per direction at 8 threads; one lane-major row; no 2D or 3D real batch row.
- 2D: 148 R2C planes, natural, out of place, one thread (nearly all power-of-two sides 2 .. 8192, raced mostly on
  2026-10-11). One carries C2R verdicts. No natural plane at 8 threads.
- 3D: 328 R2C volumes, one thread (2026-10-11); none carries C2R verdicts; none at 8 threads.

So the C2R side of 2D and 3D, every threaded 2D and 3D real cell and every real batch above one row race from
cold. In the planes the store has, the R2C verdicts are mixed: `rx` door 78 / `lm` 50 / a pair 22; `rxs` any
87, a fixed state 61; `cx` plain 72 / strided 46 / staged 19 / a leaf 10; `tf` on 22, off 43, absent 83. In the
volumes `s=1` 293 of 328 and `nf=1` 282.

### What the estimate has
- **A 1D real prototype, not in the library:** `src/estimate/estimate_1d.h` (`est_real`, both directions, the
  1D sketch's sections 6 and 7), built by `src/estimate/estimate_build.h` through a private store and the
  door's replay. Its `zr2c` half is the library's C2C estimate at N/2 (`est_c2c`), so every later C2C rule
  reaches it. Timed on 2026-10-28 on 13 lengths per direction (`estimate_performance.md`, "1D real transforms"):
  R2C 1.057, C2R 1.061 against the planner. The misses it showed: C2R 1,000 (the half's route, 1.30), 1,031 (the
  Bluestein length: the C2C rule's mistake before 55c75923, 1.09 – 1.12), 1,280 (where real ZTURN-T starts,
  1.10 – 1.13), the pair at 64 and 120, the odd chain's order at 105, the chain at 531,441.
- Its threaded rules were fitted at 8 threads before the 10-29 rulings, and it refuses `zfsr` at T > 1.
- **No 2D or 3D real formulas, and no real batch formulas.**
- **From the C2C estimate, in the library:** `est_col_plan` (every real column pass is the column builder's, as
  in C2C), `est_mt_2d` with the strips at `est__msw_2d`, the 3D threaded rules (the plane arm, the axis-0
  chains), `est_batch_whole` (the contiguous batch's way), and the lane-major recipe (`est_plan_lm`: the walk,
  the strips rule, the prime column's switch).

## 2. What carries over, and what is new

| Piece | Carried over (no new rule) | New: the owner's after a study |
|---|---|---|
| 1D engine and recipe | the prototype's `est_real`, moved into the library | the first sample's misses, one by one |
| 1D `zr2c` half | the C2C estimate at N/2 | |
| 1D at T > 1 | the C2C 1D thread rules at N/2 (the serial bound, the inner's arm) | the engines' own arms (`zttr`, `zrf`, `zrb` `mt=`, `fold=mt`) restated in the 10-29 form; `zfsr` at T |
| 1D lane-major batch | the C2C lane-major recipe at M = N/2 over K lanes | only if the study shows the fold changes it |
| 1D contiguous batch | `est_batch_whole` on the `tcb` row of each direction | |
| 2D column pass | `est_col_plan` at (N1, N2/2 + 1); at T > 1 `est_mt_2d` | |
| 2D row pass (`rp_`) | the 1D real estimate at N2 as a contiguous batch of N1; at odd N2 the C2C one | |
| 2D forms: `rx rxs cx cxs tf raxis csk` and C2R's `cxp_c2r cxd_c2r` | none | a rule per verdict, from forced arms (like the lane-major tile study) |
| 3D | axis 0 by `est_col_plan` at (N1, N2 (N3/2 + 1)); the plane = the 2D real estimate at one thread; at T > 1 the C2C 3D rules | `s nf nsw` and C2R's band arm |
| 2D and 3D contiguous batches | `est_batch_whole` on the `pq` row of each direction | |
| 2D and 3D lane-major batches | the C2C lane-major recipe for the column pass; the 1D real lane-major one for the blocks | |

A create writes only its own direction's verdicts, into its private store.

**Call overhead.** The estimate runs only inside `vfft_create`, as the C2C one does. It writes rows into a
private store, the door's own create replays them, and the handle handed back is the door's. The estimate keeps
no wrapper and adds no call to the execute path. Each stage checks this three ways:
- the build's plan check refuses a plan that is not entered by the lean door's jump (`k1_go` in the transform's
  direction, which every 1D real executor of one transform binds), so the gate's every estimate plan has it;
- the study tool prints, per arm, the engine served and whether its lean entry is bound, and the timing includes
  the small cells where a call costs the most (`zrm` 12, `zrp` 64, planes 4x4 to 16x16), where a leak would show
  as a ratio above 1 on the same recipe;
- at 2D, the door binds its lean two-kernel executor only for some verdicts (one thread, `rx=lm`, a leaf `cx`,
  no threaded columns), so the rules for `rx` and `cx` weigh what they cost per call, not only per point.

## 3. Stages

Each stage repeats the C2C steps:
1. Build with no new rule. The front door admits what the stage covers. Add gate sections that check nothing
   was measured, the plan served is the recipe, and the output matches the naive DFT.
2. Race the planner's cells into the store.
3. Time the estimate against the planner, with a second planner arm as control.
4. Score, and propose rules.
5. The owner approves the rules; build them.
6. Time cells drawn after the rules, which the rules were not fitted on.
7. Write the section on `estimate_performance.md`.

| # | Stage | Build | Quiet machine |
|---|---|---|---|
| 1 | 1D, one transform, 1 and 8 threads, both directions | about 1 day | about 2 – 3 h |
| 2 | 1D batches: lane-major (the default real batch) and contiguous | about half a day | about 1.5 – 2 h |
| 3 | 2D and 3D, one transform, 1 and 8 threads | about 2 days | about 4 – 6 h (a 2D real create races four stack states per arm) |
| 4 | 2D and 3D batches, contiguous and lane-major | about 1 day | about 2 h |

## 4. Cells (out of place, both directions; a share in place; 60 unseen cells per stage after its rules)

| Stage | Lengths or shapes | K | Threads |
|---|---|---|---|
| 1 | powers of two 16 .. 4,194,304; even smooth 12, 60, 120, 1,000, 1,280, 4,050, 6,000, 12,000, 100,000, 1,000,000; even with a prime half 262, 2,062, 8,198; odd composites 105, 945, 1,035, 3,375, 15,625, 59,049, 531,441; odd primes 13, 61, 131, 1,031, 4,099, 50,021 | 1 | 1 and 8 |
| 2, lane-major | 16, 64, 256, 1,024, 4,096, 16,384, 65,536; 100, 1,000, 12,000; 262, 2,062, 8,198 | 4, 32, 256 | 1 and 8 |
| 2, contiguous | 20 lengths from stage 1 | 3, 6, 12 or 24 | 8 |
| 3, 2D | about 30 planes: power-of-two squares and rectangles 8 .. 2,048, odd N2 (64x63, 127x100), smooth (100x100, 1000x1000) | 1 | 1 and 8 |
| 3, 3D | about 20 volumes: cubes 8 .. 128, flat and tall, an odd N3 | 1 | 1 and 8 |
| 4 | about 20 shapes per rank | 2 .. 24 | 1 and 8 |

Of the 36 stage-1 lengths, 28 (R2C) and 27 (C2R) already have a one-thread row; the rest and every 8-thread
cell race first.

## 5. The owner's answers (2026-11-01)

1. **The order:** 1D whole first (stages 1 and 2, because a real batch is lane-major by default), then 2D and 3D.
2. **The file:** the real estimate has its own file, `il/estimate/real_estimate_il.h`, serving both directions
   ("their own file. however we have to be careful with call overhead if a call is leaking into execution"; see
   "Call overhead" in section 2).
3. **The first sample's misses** go into stage 1, each proposed with its evidence for approval one at a time.
4. **The 8 `zfsr` rows are not raced again** on purpose: "if it was created before, it should not re-race. it
   should race only when the banked cell is missing or the user asked for recalibration."
5. **Natural order only**, at every rank: "we only handle natural order now, assume scrambled as a beta feature".
   Both placements.
6. **The thread counts are 1 and 8**, as in the C2C studies.

## 6. Stage 1, step 1: built (2026-11-01)

- **The library:** `il/estimate/real_estimate_il.h` holds the 1D real estimate, moved from `src/estimate/` with no
  rule changed: the recipe `est_real_recipe_t`, the formulas `est_plan_real` / `est_real`, the row writer, the
  pre-flight, the plan check (now also the lean entry) and the build `est_build_real`, reached from
  `_vfft_create_estimate`. The front door serves one 1D R2C or C2R transform under the flag and refuses a real
  transform at rank 2 or 3 or in a batch, each with its own message. The real door's seven steering variables
  (`VFFT_ZRM`, `VFFT_ZRP`, `VFFT_ZTTR`, `VFFT_ZFSR`, `VFFT_ZRF`, `VFFT_ZRB`, `VFFT_ZR2C_ROUTE`) joined the guard.
  `src/estimate/` keeps the tools, calling the library.
- **Same plans as before the move:** the prototype's check over 2..4096 (16,380 cells), the powers of two
  262,144..2^24 at 1 and 8 threads (56) and 300 random lengths to 2,000,000 at 1 and 8 threads (2,400) gives the
  same results before and after: every cell built and correct with nothing measured, except the 20 real four-step
  cells at 8 threads, which the formulas refuse.
- **`estimate_gate`** passes in full (27,130 checks): the new sections RL (4,368 real plans at one thread, both
  placements) and RLT (120 at 8 threads, 20 executes on a threaded arm), the new refusals and the `VFFT_ZRP` guard.
- **The study, ready to run:** 188 cells (section 4's lengths, both directions, out of place at 1 and 8 threads, 12
  lengths in place), the planner's creates into the store, then the timing (P, E and a control in one process, 15
  rounds). It waits for the owner to say the machine is quiet.

## 7. Noticed, not filled (unverified)
- The 1D lane-major real Bluestein path (`_vfft_create_real_lanes`, rows `eng=zrbl`) looks unreachable for an
  interleaved batch: `vfft.c` serves or refuses every lane-major K > 1 before `_vfft_create_real`.
- `fftnd_real_il.h` seeds the 3D plane from the 2D row keyed at `nthreads=T`, but creates the plane child at one
  thread. Whether the child's lookup reconciles the two was not checked.
- A `zfsr` row whose child is missing makes the door race the whole cell, its engine included
  (`il/wisdom/wisdom2_real_il.h`, `vw2_real_il_lookup_zfsr`), where only the child is missing.
