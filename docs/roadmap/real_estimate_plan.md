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

## 7. Stage 1 study: verdict and proposals (2026-11-01, 14:22 – 15:48)

**Run:** the planner's 188 creates into `src/wisdom/14900KF` (126 raced: every 8-thread cell and the one-thread
lengths without a row; the two real four-step rows without their child were raced whole and banked complete), then
the 188 cells timed: the planner's plan (P), the estimate's (E) and P again as the control, one process, 15 rounds.
The control 0.997; no arm measured; every arm bound its lean entry; the outputs agree within 9e-16. Tools and logs:
`docs/research/estimate_real/evidence/stage1/`.

| | Cells | Geomean | Median | Within 5% | More than 5% slower | Above 1.10 |
|---|---|---|---|---|---|---|
| One thread, R2C | 48 | 1.027 | 1.019 | 31 | 15 | 10 |
| One thread, C2R | 48 | 1.046 | 1.039 | 25 | 22 | 9 |
| One thread, in place | 24 | 1.060 | 1.065 | 11 | 13 | 8 |
| **One thread** | **96** | **1.037** | **1.022** | **56** | **37** | **19** |
| 8 threads | 92 | 1.263 | 1.075 | 38 | 50 | 30 |
| — where the planner threads `zttr` or `zrb` and the estimate does not | 14 | 3.551 | 4.779 | 0 | 14 | 14 |
| — the other 78 | 78 | 1.049 | 1.040 | 38 | 36 | 16 |

**Proposals (each the owner's to approve):**
1. **Threaded `zttr` and `zrb`.** The estimate keeps them on one thread: the 1D sketch's section 7 says they have
   no threaded row, which is no longer so. Proposed: the C2C ZTURN-T rule of that section at the real data, N/2:
   above the serial bound (16 N/2 > 2 L1d) TILES (`mt=2`) when tiled with at least 2 tiles, else BLOCKS
   (`mt=1`); `zrb` only on a ZTURN-T inner. It fits all 42 `zttr` and `zrb` rows of the store at 8 threads (`zttr`
   one thread at 1,280 and 4,096, `mt=2` from 16,384; `zrb` one thread up to 4,099, `mt=2` at 50,021). The
   estimate's `zrb` inner at 50,021 is untiled where the planner's has a tile of 512 to 2,048, so it would take
   BLOCKS; a forced arm checks TILES against BLOCKS there.
2. **One thread: the first sample's misses**, each fitted on forced arms of the estimate's own recipe (one change,
   built by the library's `est_build_real`): the Bluestein length (the C2C rule of 55c75923, the shortest length
   whose inner runs ZTURN-T; 1,031 at 1.12 – 1.16, 131 at 1.14); the odd chain's order (the planner leads with a
   large radix: 3,375 9.5.3.5.5, 105 7.3.5, 945 9.3.7.5; 1.11 – 1.23); the pair at 64 (4.16 against 8.8, 1.13 –
   1.15); where real ZTURN-T starts (1,024 and 1,280 in place, 1.10 – 1.13).
3. **The `zr2c` fold at T > 1:** mixed (100,000: the estimate threads the fold, the planner does not, 1.32; 8,198:
   the planner threads the fold and the prime half, 1.22 – 1.25). Forced arms (fold on one thread or threaded)
   before any rule.

**The owner's answers (2026-11-01):** proposal 1 yes, with the check of TILES against BLOCKS at 50,021; forced
arms for the rest.

**The forced arms (2026-11-01, 16:14 – 16:48):** the 188 cells again with the estimate's recipe changed in one place
(built by `est_build_real`, so the door serves it and the plan check holds), then 84 targeted one-thread cells (odd
Bluestein lengths, odd composites with large factors, the lengths around where real ZTURN-T starts, the pairs).
The control 0.997 – 1.003; nothing measured; every arm bound its lean entry.
- **`zttr` at 8 threads:** TILES above the bound beats the planner's speed in reach on all 10 cells (1.55 – 5.2 ->
  0.96 – 1.29); BLOCKS loses to TILES on all 10; below the bound (1,024, 4,096) threads lose 1.6 – 8.3 x. Proposal
  1 holds as written for `zttr` and is **built** (`est_real`: `est_inner_mt` at M = N/2; the sketch's section 7).
- **`zrb` at 8 threads:** threads pay from 4,099 as well (M = 7,168 passes the bound; the planner's row there is
  serial): TILES 0.84 – 0.89, BLOCKS 0.92 – 0.94 against one thread at 1.0; at 50,021 TILES 1.11 – 1.15, BLOCKS
  1.19 – 1.23. TILES needs a tiled inner, and the estimate's odd-band ZTURN-T inner is untiled.
- **The `zr2c` fold:** the rule holds (threading the fold below the bound costs 1.5 – 8 x) except where the half is
  prime: 8,198, the fold and the prime half threaded 0.995 against 1.24 – 1.28.
- **`zrf` at 8 threads:** the threaded arm the estimate takes is right (one thread loses on 5 of 6).
- **One thread:** `zrb`'s three-stage inner (1,009, 1,031 at 1,792 or 1,536) against the next length with a pair or
  ZTURN-T inner (2,048, the planner's): 1.10 – 1.15 -> 1.03 – 1.06. The odd chain with its largest radix up to 9
  first, the rest ascending: 22 cells 1.101 -> 1.041 (13 faster, 1 slower: 1,035); descending loses, the largest
  first loses where it is 23. Real ZTURN-T at lengths that are not powers of two wins from 1,152 (1,920: 1.20 ->
  1.005); R2C powers of two up to 8,192 run faster as the complex half (512: 1.01 -> 0.90, 1,024: 1.07 -> 0.96),
  C2R powers of two faster as real ZTURN-T. The pair: the planner puts 10 second and prefers a second factor of at
  least 10 (48 4.12, 64 4.16, 120 12.10, 160 16.10: 1.07 – 1.28 -> 0.95 – 1.03); that rule matches 32 of the
  store's 34 pair rows (the current one 21).

**Proposals (the owner's to approve), with the cells above at the rules' arms:** the one-thread stage-1 cells 1.033
-> 1.019, the 8-thread ones 1.272 -> 1.070, the targeted cells 1.059 -> 1.015 (fitted on these cells; the unseen
cells come after):
1. `zrb` threads by the C2C prime cell's own reading: the bound at its convolution length M (not N/2), and at
   T > 1 an untiled ZTURN-T inner takes the first legal tile of 1,024, 2,048, 512, so it runs TILES.
2. A `zr2c` prime half reads the bound at its inner's M, as the C2C prime cell: the fold and the half thread there.
3. `zrb`'s length skips a three-stage inner: the smallest candidate whose inner is a pair or ZTURN-T.
4. The odd chain leads with its largest radix up to 9, the rest ascending.
5. Real ZTURN-T at lengths that are not powers of two from N = 1,024 (now 2,048); R2C powers of two up to 8,192 take
   the complex half (C2R unchanged).
6. The pair: 10 second when it divides; else the most balanced pair whose second factor is at least 10; else the
   most balanced, smaller first.

**The six rules, built (the owner: "yes to all"), judged on the arms already timed (2026-11-01, no new timing):** each
cell's recipe under the rules matched to a timed arm: the old estimate's, the planner's, or the forced arm that
makes the same change. The study's one-thread cells 1.037 -> 1.015 (94 of 96 matched), its 8-thread cells 1.263 ->
1.028 (88 of 92), the targeted cells 1.059 -> 1.015 (84 of 84). The 6 unmatched cells are 1,035, whose chain the
rule turns into 9.5.23; the arm timed at one thread out of place was 0.99 -> 1.04. The 8-thread gain is the
threading rules: `zttr` 10 cells 3.0 -> 1.11, `zrb` 6 cells 3.1 -> 1.03.
- **Still behind, untouched by the rules:** `zttr` at 8 threads takes a 2,048 tile on all 10 cells, the planner 1,024
  on 8 (16,384 and 65,536: 1.20 – 1.29); R2C 100,000 at 8 threads threads the fold, the planner does not (1.32; C2R
  100,000 and R2C 1,000,000 want it threaded); C2R 1,000,000's complex half (1.21 – 1.24) and the C2R child route at
  1,000 and 6,000 (1.10 – 1.12) come from the C2C side.
- **The unseen cells** (59 of 60 raced by the planner before the run was stopped; not timed): the estimate's engine
  is the planner's on all 59, the whole recipe on 10; at 8 threads the thread choice agrees on all 23 recipes that
  differ, the fold on 21. The differences are the Bluestein inner's shape and tile, and the complex half's details.
- **The real four-step at 8 threads** (`zfsr`, the C2C four-step's threaded split and child at N/2; the owner: "yes"):
  built; the 20 cells (2^20 – 2^24, both directions, both placements) plan and compute within 1e-15; not timed.

**Real ZTURN-T at 8 threads: the cause (2026-11-01, 23:05 – 23:25; the 10 cells 16,384 – 262,144, 15 rounds, control
1.002 – 1.013):** the estimate as built 1.104 – 1.123. One change at a time: the planner's stage order 1.035 – 1.040
(6 faster, 1 slower), the tile 1,024 1.152 (worse), the planner's stk no change; the planner's whole recipe built by
`est_build_real` 1.013 (the arms are faithful). The order carries the gap; at 16,384 only the order with the 1,024
tile together is fast. Candidate orders: ascending (the 4s first, the 8s last) with the tile 1,024 1.042 (6 faster,
none slower), ascending alone 1.075, one 8 moved to the end 1.104. **Proposal (waiting for the owner's fresh-cell
check):** at T > 1 real ZTURN-T takes its stages ascending and the tile 1,024.

## 8. Noticed, not filled (unverified)
- The 1D lane-major real Bluestein path (`_vfft_create_real_lanes`, rows `eng=zrbl`) looks unreachable for an
  interleaved batch: `vfft.c` serves or refuses every lane-major K > 1 before `_vfft_create_real`.
- `fftnd_real_il.h` seeds the 3D plane from the 2D row keyed at `nthreads=T`, but creates the plane child at one
  thread (confirmed 2026-11-02, `fftnd_real_il.h:1238, 1370, 1386`): at T > 1 the borrowed seed is invisible to the
  child, which races. No 3D real row at T > 1 exists yet to show it in the data.
- The 2D real door's in-place plan looks up a `zr2c` row engine's child at `is_inplace=0` but banks it at the plan's
  placement (`il2d_real_plan.h:495` against `:509`, `:1158`), so an in-place plan with `rx=zr2c_r*` would race its
  child at every create (read from the code, not run).
- A `zfsr` row whose child is missing makes the door race the whole cell, its engine included
  (`il/wisdom/wisdom2_real_il.h`, `vw2_real_il_lookup_zfsr`), where only the child is missing.

## 9. Stage 3, step 1: the 2D and 3D build (2026-11-02, overnight; the owner: "full approval")

**Built, no new rule.** `src/core/il/estimate/real_estimate_nd_il.h`, its own file beside the 1D one, reached from
`_vfft_create_estimate` for R2C and C2R at rank 2 and 3; the front door admits one real transform at rank 1 to 3 in
natural order (`vfft_api.h`; a batch and `VFFT_FLAG_OWN_PITCH` refused by name); the 2D and 3D real doors'
switches joined the steering list (`c2c_estimate_il.h`).
- **2D** (`est_plan_2d_real`): the column pass by `est_col_plan` at (N1, N2/2 + 1), natural, as C2C; the forms as the
  door's real race spells them (R2C's pool adds the half-store twins at the leaf, so its token is always written;
  each stage the full-store default, the planner's leaf in the store 4 times in 5); `wl` 0; at T > 1 `est_mt_2d`'s
  column rule at the real plane's bytes (threaded past half the L1d, the strips at `est__msw_2d`'s width); the row
  engine `lm` where it is offered, else the door's row child; the column form strided on a natural chain, else
  plain; one thread: `tf` past the L2, `raxis` n2, `csk` 0, C2R's `cxp` 0 and `cxd` off; the row child the 1D real
  estimate at N2 (at T > 1 with the batch's verdict by `est_batch_whole`), at odd N2 the C2C estimate at N2. The row
  engine can also be the 1D real estimate's engine at N2 (its child written under `rx_` for `zr2c`).
- **3D** (`est_plan_3d_real`): axis 0 by `est_col_plan` at (N1, N2 (N3/2 + 1)), the scrambled pass; axis 1's pay-once
  chain at (N2, N3/2 + 1) wherever axis 0 is a plain chain; the child in place (`s=1 nf=1`, the store's majority);
  at T > 1 `est_mt_3d`'s bound, the plane arm on the child with the strips where axis 0 is a plain chain (R2C with a
  natural twin cannot run the child in place threaded, `fftnd_real_il.h:1830`), `cmtt` written; the plane child the
  2D estimate at (N2, N3), one thread.
- **The build and checks:** the 1D builds' steps (`est__replay`'s detectors: nothing measured, nothing re-banked)
  and a plan check (the column chain, the row engine by the door's own spelling, the column form, the one-thread
  forms only where asked, the threaded verdict, the children; at 3D the structure, the threaded verdict and the
  plane child as the 2D recipe).
- **Checked** (scratch `ndchk`, long-double DFT on 48 sampled bins for R2C, C2R back from that spectrum): every
  plane of the study grid at 1 and 8 threads out of place, 13 planes and 12 volumes in place, the edge shapes
  (axes of 2 and 3, tiny odd axes) at both ranks, both placements, 1 and 8 threads: all built, nothing measured,
  worst error under 3e-15. The gate gained `RD2 RD2T RD3 RD3T` (built, not run) and its real refusals now name a
  batch and the own pitch.

**The study grid** (scratch `nd/cells2d.py`): 50 planes by layer (the spectrum's bytes against L1d 48 KiB, L2 2 MiB,
L3 36 MiB) and class (powers of two, smooth composites, odd composite real axes, prime real axes, even real axes
with a prime half, prime columns): 16 in L1, 16 in L2, 13 in L3, 5 beyond. The planner's creates raced into
`src/wisdom/14900KF` (R2C, then C2R on the shared row), one thread first, then 8 threads.

**The one-thread study (2026-11-02, 00:41 – 01:30; 100 cells, the 50 planes both directions, out of place, 15 rounds,
control 1.001; tools in scratch `nd/`, `ndtime` with one-change arms).** The build as above: E/P 1.087 (R2C 1.138,
C2R 1.039). By layer: L1 1.064, L2 1.134, L3 1.068, beyond 1.069. The gap was R2C at odd real axes (odd composites
1.17 – 1.82, primes 1.34 – 1.77 by layer): the door's C2C row child where the planner runs a 1D real engine. Six
rules came out of the arms, each built (the owner's approval for the night):
1. **R2C at odd N2: the row engine is the 1D real estimate's odd engine** (`zrf`, `zrb`; the small kernel `zrm` stays
   on the child, 5.3 x slower at 16 x 61): 1.46 -> 1.10 on 8 of 8 `zrf` planes, 1.52 -> 1.24 on 5 of 6 `zrb` planes.
2. **The fused walk:** C2R at every size (1.07 – 1.12 -> 1.03 on 18 planes below the L2, none slower); R2C past the
   L2 only (below it 1.10 -> 1.20 at L1); past the L2 turning it off is slower on 28 of 30.
3. **N1 = 128: the column leaf**, R2C `b448`, C2R `b816` (1.07 – 1.24 -> 0.99 – 1.08).
4. **C2R's column-inverse pitch `cxp=3`** (1.014 -> 0.999 on 17 planes, none slower).
5. **N1 = 64: one column stage** at any width (C2C's formula splits it 8.8 on a wide row): 1.09 – 1.37 -> 1.00 – 1.05
   on 4 of 8, none slower; the planner's chain at every N1 = 64 plane in the store (extended below to N1 = 8 .. 64).
6. **R2C, odd prime N2 up to 2048, even N1 >= 4: the real axis on N1** (`raxis=n1` with its children `rax_`, the C2C
   column chain at N1/2 down N2, and `raxr_`, the C2C plan at N2): 16 x 61, 32 x 127, 256 x 127, 64 x 1031, 1024 x
   1031 1.09 – 1.68 -> 1.00 – 1.07; at 4099 and at the odd composites (but 24 x 45) it is slower.
With rules 1 – 5 the same 100 cells: **E/P 1.033** (R2C 1.054, C2R 1.012; L1 1.021, L2 1.067, L3 1.014, beyond
1.014; control 0.999); rule 6 on its 5 planes as above (fitted on these cells; the unseen ones come after).
Still open: the column chain at N1 >= 1024 (the planner's two radix-32 stages, `32.32`, against C2C's four, run
1.06 – 1.13 faster on 3 of 5 such planes and slower on others: not one rule yet); the Bluestein row engine's inner at
N2 = 127 and 1031 in the row role (the planner's `2p:6.32` and `ztt:8.8.4.8` against the 1D estimate's `12.16` and
`8.8.8.4`).

**The 8-thread study (2026-11-02, 02:16 – 02:45; the same 100 cells at T = 8, control 1.001 – 1.003).** The build
with the one-thread rules: E/P 1.143 (R2C 1.145, C2R 1.141). Three rules came out of the arms, built:
7. **The rows kernel `lm` on one thread only** (at 8 threads 2.2 – 3.8 x slower at 16 x 16 and 32 x 32; the door's row
   child 0.98 – 1.02).
8. **The row batch serial where the plane is** (16 N1 (N2/2 + 1) <= L1d / 2, the column rule's bound): 8 x 256 and
   48 x 48 1.06 – 1.83 -> 0.52 – 0.88, the other serial planes flat; above the bound serial rows are 1.6 – 3.5 x slower.
9. **The N1 = 128 leaf at T > 1 for R2C** (`b448`: 1.15 – 1.33 -> 0.88 – 1.01 on 3 of 4 planes; C2R's mixed, strided
   kept).
With them the same 100 cells: **E/P 1.035** (R2C 1.032, C2R 1.039; L1 0.893, L2 1.126, L3 1.043, beyond 1.246).
Still open at T > 1: the tall power-of-two planes at L2 (1024 x 64, 512 x 128, 256 x 256: 1.4 – 1.6; the planner's
two radix-32 stages with the matched partition, `mtarm=0`; neither alone closes it); the Rader column at 4099 x 2048
beyond the L3 (2.4 – 2.7; the matched partition 1.1 – 1.2: the strips' width 8 on a one-stage Rader pass). The
matched partition is no rule: past the L2 it is 1.2 – 2.6 x slower than the strips.

**Unseen planes, one thread (2026-11-02, 02:20 – 03:10; 32 planes the rules were not fitted on, the same layer x
class grid at other sizes, both directions, raced into the store, then timed; scratch `nd/cellsu2d.py`): E/P 1.027**
(R2C 1.040, C2R 1.014; L1 1.019, L2 1.063, L3 1.008, beyond 0.978; control 0.997), against 1.033 on the fitted cells.
Two misses there: 32 x 2048 (1.24 – 1.38; C2C's formula splits the 32-point column on its wide row, the planner keeps
one stage) -> **rule 5 extended to every power-of-two N1 from 8 to 64** (the planner's chain at all of them in the
store but 8 x 4): 32 x 2048 then 1.02 and 0.98. And R2C 256 x 63 (1.53): the planner's real axis on N1 at an odd
composite N2 (the arm 0.94), where rule 6 takes primes only; the odd composites were mixed on the fitted grid
(24 x 45 faster, 16 x 105, 128 x 225 and the large ones slower) and stay open.

**3D, one thread (2026-11-02, 03:05 – 03:28; 25 volumes by layer x class, scratch `nd/cells3d.py`: powers of two,
smooth, an odd N3, a prime N3, a prime axis 0, a prime N2; both directions; raced into the store, then timed with
`ndtime3`, control 1.000).** The build: E/P 1.058 (R2C 1.082, C2R 1.036; L1 1.012, L2 1.037, L3 1.089, beyond
1.142). One rule came out of the arms, built:
10. **The strips (`nf=2`): R2C past four L2s, C2R past the L3** (C2C's `est_nf_3d` bound), on a plain axis 0 with no
    Rader stage and 16 N1 <= L1d / 4, at the width the door's policy gives (`vfft_policy_ilndr_strip_w`): R2C 9 – 129
    MiB, 6 volumes 1.11 – 1.34 -> 0.94 – 1.02; C2R 56 – 129 MiB 1.08 – 1.15 -> 0.98 – 1.07; below those bounds, and on
    a Rader axis 0 (109, 331), slower.
Open: the pay-once structure (`s=2`: 100 x 100 x 100 1.11 – 1.21 -> 1.00 – 1.03, slower on 17 small volumes) and C2R's
band (`s=3`, the planner's at 128^3 and 256^3; no arm yet).

**3D, 8 threads (2026-11-02, 03:27 – 03:58; the same 50 cells at T = 8, control 0.994).** The build with rule 10:
E/P 1.099 (R2C 1.142, C2R 1.057; L1 1.034, L2 1.201, L3 1.120, beyond 0.995). One rule came out of the arms, built:
11. **A one-stage Rader axis 0 runs the plane arm in place, not in strips** (53, 109, 331, both directions 1.10 –
    1.27 -> 0.97 – 1.03, none slower), as rule 10 keeps the serial strips off it.
Open: the threaded pay-once (`cmts=2`: R2C at L2 1.30 -> 1.23 with 4 faster, 3 slower; at L1, L3 and beyond slower).

**The library as built at the end of the night (2026-11-02, 03:58 – 04:08; no arms, the stores the races left):**
2D grid one thread **1.008** (R2C 1.010, C2R 1.006), unseen planes one thread **1.018** (51 of 64 within 5 %),
2D grid 8 threads **1.040**, 3D grid one thread **1.026**, 3D grid 8 threads **1.088** (controls 0.997 – 1.006);
`estimate_performance.md`, "2D and 3D real transforms". Not done: the in-place cells timed (only built and checked
for correctness), the 2D and 3D batches (stage 4), the owner's review of the eleven rules.

## 10. Stage 3, step 2: the banked planes (2026-11-02, morning)

The owner: "we have incredible amount of saved / banked 2D cells, shall we test the estimate 2D real on those?" Every
banked real 2D row at one thread, natural order, out of place: 211 planes, R2C on all of them, C2R on the 83 whose
row carried its C2R side; the other 128 C2R sides raced into the store first (the owner's yes). The planner's plan
replayed from the store, no race on any cell.

Before the rule below: 422 cells, E/P 1.013 (R2C 1.031, C2R 0.995). The one gap was the tall power-of-two plane: a
column of 4096 or 8192 rows over a row of 256 or fewer at 1.2 – 1.5 (R2C), 2048 and 512 at 1.1 – 1.3. The planner's
column verdicts together (chain, forms, `cx`, `tf`) close it at every length (1.002 – 1.007); the chain alone carries
nearly all of it. C2C's formula leads the column with 32 from 2048 on narrow rows, which the C2C planner banks
there; the real planner leads small. Approved by the owner and built:

12. **The tall real column: a power-of-two N1 from 512 to 8192 takes one chain per length** (`est_col_real_tall`),
    both directions, every row width, one thread: 512 `16.32`, 1024 `8.4.32`, 2048 `8.8.32`, 4096 `8.8.8.8`, 8192
    `8.16.64`, each the best of 5 – 7 chains timed on the banked planes. The 102 cells it changes: R2C 1.124 and C2R
    1.043 -> 0.972 together (90 within 5 %). At 8 threads the table is no gain (13 planes, 1.192 -> 1.18), so it
    stays at one thread; the 8-thread loss there is the open tall-plane threading (1024 x 64, 512 x 128 at 1.7 – 2.6).

After it, all 422 banked cells: **E/P 0.985** (R2C 0.994, C2R 0.976; 362 within 5 %; control 0.999); the planes the
night never timed 0.975. Noticed, not filled: R2C at x32 rows where the planner fuses the walk below the L2
(512 x 32, 1024 x 32 1.15 – 1.18); C2R's N1 = 128 leaf (rule 3) at very wide planes (128 x 8192 1.20, 128 x 4096
1.11, the planner's plain `16.8`); R2C's half-store leaf at 16 x N2 >= 1024 (1.16 – 1.17).

The owner: "calibrate more cells like these to see the pattern more clearly". 35 new planes raced into the store
(non-power-of-two shapes and neighbouring lengths), 109 cells timed with every arm. Each gap has one pattern; three
rules, approved by the owner and built (one thread):

13. **R2C over rows of N2 = 32 from N1 = 256: the fused walk** at every size. 10 of 10 planes (256 – 4096 rows, power
    of two or not): 1.146 -> 1.003. At rows of 4 – 16 the fused walk costs 5 – 40 %, at 24, 48 and 64 it is mixed.
14. **C2R at N1 = 128 past 5/8 of the L2: the plain column `16.8`** (the planner's on every such row) instead of
    rule 3's `b816` leaf. 10 planes 128 x 1536 – 16384: 1.122 -> 0.961. The leaf still wins at 128 x 1024 and below.
15. **R2C at N1 = 16 over N2 a multiple of 512: the leaf's half-store twin `h`**. 6 planes: 1.099 -> 0.990 (16 x 2048
    flat). At 16 x 1000, 3000, 6000, 12000 it is 1 – 3 % slower; the 8, 32 and 64-point leaves gain nothing.

The 109 cells as built: 1.033 -> 0.999; the cells the rules do not touch unchanged.

**The 3D banked volumes** (the owner's yes): every banked 3D real row at one thread, 346 R2C volumes and the 25 with a
C2R side, 371 cells, E/P 1.034 (L1 0.955, L2 1.065, L3 1.060); the 2D rules reach the plane child without harm (rule
13's 6 volumes 0.953). The arms (the planner's axis-0 chain, its axis-1 chain, both, one stage on axis 0) put the gap on
axis 0, where C2C's scrambled pass splits 16, 32 and 64. Approved by the owner and built:

16. **3D axis 0 at N1 = 16, 32, 64: one stage, unless its N1 elements crowd an L1d set** (one thread, no strips). The
    elements lie 16 N2 (N3/2 + 1) bytes apart and fall into 4096 / gcd(stride, 4096) sets of L1d / 4096 lines each, so
    one stage while N1 gcd(stride, 4096) <= L1d. Without crowding: 16 28 volumes 1.263 -> 0.992, 32 38 volumes 1.144
    -> 0.974, 64 27 volumes 1.037 -> 0.906, none slower; with it one stage is slower (16 x 256 x 4 1.22, 64 x 128 x 128
    1.25) and the planner splits on all 45. Open: axis 0 at N1 = 256 (`16.16` at 1.2 – 1.45 on 8 volumes), 128 mixed;
    the C2R side thin (25 volumes).

**The C2R side of 3D and the rules at 8 threads** (the owner: "start with these"). The C2R side raced for the 321
banked 3D volumes that lacked it: 346 C2R volumes at one thread, E/P 1.016 (L1 0.947, L2 1.029, L3 1.076); rule 16
holds for C2R (16, 32, 64: 0.993, 1.000, 1.009); open: axis 0 at 128 and 256, and C2R's band (`s_c2r=3`) on the worst
volumes (256 x 16 x 4 1.38). The rules at 8 threads, each applied as an arm on 49 new 8-thread rows: 12 no gain
(42 planes 1.537 -> 1.528, 4096 x 16 5.9 -> 8.4) and 16 slower (22 volumes 1.358 -> 1.480), so both stay at one
thread; 13 has nothing to apply (the planner banks no fused walk at T > 1); 14 and 15 a small gain on thin evidence
(6 planes 1.105 -> 1.051, 5 planes 1.212 -> 1.164), left at one thread. The 8-thread gap on these planes is the tall
plane's threaded column (1.985 on 64 tall planes); approved by the owner and built:

17. **At T > 1 a tall plane, N1 >= 4 N2, runs the plain threaded column (`mtarm=0`) over the chain with the fewest
    stages** (two stages the most balanced, three or more the largest radix last). The planner takes `mtarm=0` on all
    62 banked 8-thread rows with N1 >= 4 N2. 60 tall planes at 8 threads (256 – 8192 rows, power of two or not, both
    directions): 2.071 -> 1.077 against the planner (its own column verdicts 1.045), 58 faster, one slower (C2R
    3000 x 64, 1.02 -> 1.11). The near-square 2048 x 1024 (the planner keeps the strips) is left out by the bound.
    Open: R2C over rows of 64 at 1.23 – 1.33 (the row side); 1000 x 100 (the planner's radix 25, not in the pool).

**3D at 8 threads** (the owner: "we can attack here then", after the page's 1.169 at 8 threads, L2 and L3 1.23 – 1.26).
56 new 8-thread races (28 volumes of L2 and L3), 128 cells with arms. The planner runs a one-stage axis 0 in place,
and on R2C the pay-once structure dealt by stage; the estimate split axis 0 and ran the strips. Approved by the owner
and built:

18. **At T > 1 axis 0 of 16, 32 or 64 runs one stage, in place (`cmtf=1`)**, both directions, with no L1d-set test
    (crowded volumes gain too). 60 cells: 1.318 -> 1.014 as built, 52 faster, 3 slower (C2R 16 x 16 x 256, 32 x 32 x
    61, 16 x 256 x 256: 0.96 – 1.11 -> 1.02 – 1.22).
19. **At T > 1 R2C over a plain axis 0 of two stages or more, the volume within a third of the L3, runs the stage-wise
    pay-once** (`cmts=2 cmtf=3`): from 0.3 to 9 MiB it won at every size, from 16 MiB it lost. 19 cells: 1.235 ->
    0.993 as built, 17 faster, 2 slower (30 x 30 x 30 1.00 -> 1.17; 128 x 128 x 64 0.92 -> 0.97).

The 128 cells as built: 1.208 -> 1.035 (R2C 1.265 -> 1.011, C2R 1.154 -> 1.058; L2 1.244 -> 1.036, L3 1.252 -> 1.037;
control 1.002). Open: C2R over an axis 0 of two stages or more (about 1.13; the planner's structure and form vary).

**The remaining gaps, one by one** (the owner: "let's go one by one", then "go through all the items, I will review in
the end"). Built, for the owner's review:

20. **3D axis 0 at one thread: a chain with a stage that crowds an L1d set takes the wide rows' chain.** A stage of R
    elements L rows apart crowds when R gcd(16 L N2 (N3/2 + 1), 4096) > L1d (`est__chain_crowds`; the one-stage axis
    0's count, stage by stage). 16.16 at 256 over 16 – 64 x 4 – 16 planes: 12 volumes 1.402 -> 0.977 (4.8.8).
21. **3D axis 0 at one thread on wide rows: the narrow rows' chain where it crowds no set**, C2R at 128 and 256 and R2C
    at 256 (R2C at 128 is mixed). C2R 128 25 volumes 1.081 -> 1.007, 256 5 volumes 1.098 -> 1.007; R2C 256 5 volumes
    1.099 -> 1.003. Rules 20 and 21 as built, the 172 banked volumes of 128 and 256 rows: 1.052 -> 1.007, the 47
    whose chain changed 1.161 -> 0.999 (38 faster, one slower: C2R 128 x 16 x 32, 1.08 -> 1.15). C2R's band and
    pay-once structures alone gained little (253 volumes: the band at its best width 1.061 -> 1.045, the pay-once
    mixed); its gap was axis 0's chain.
22. **R2C at T > 1 on a tall plane (N1 >= 4 N2) over rows of 64: the rows by the real pair 4.16, form a**
    (`zrp_4.16_a`, the planner's on every such banked plane; 4 lanes only, as the 1D estimate's pair engine). 8
    planes at 8 threads, as built: 1.181 -> 1.040. Rows of 32 and 128, and C2R, had no gap.
23. **C2R in place at one thread: the pay-once structure (s=2)**, the planner's on all 12 volumes raced in place.
    12 volumes, as built: 1.146 -> 0.991, none slower. Out of place it is mixed and R2C in place slower.

In place, timed for the first time (64 cells raced in place, one thread): 2D 40 cells 1.012 (R2C 1.030, C2R 0.995);
3D R2C 1.014, C2R 1.146 before rule 23.

24. **At T > 1 a Rader column on a plane past the L3 runs the plain threaded column (`mtarm=0`)**, as the tall plane
    does. One plane: 4099 x 2048 at 8 threads, R2C 2.75 -> 1.17, C2R 2.62 -> 1.14 -- the only Rader plane banked past
    the L3, so the rule rests on that plane alone; on the smaller Rader planes the plain column was slower (109 x 512
    0.96 -> 1.26). Open: the large Rader columns below the L3 (4099 x 512 1.35 – 1.53, 4099 x 1024 1.18 – 1.27, 2053
    x 1024 1.21), which no threaded verdict of the column moves -- the Rader stage itself, studied next:
25. **At T > 1 the Rader stage's window is 8 columns** (not `EST_ND_RD_W` = 32), the planner's window at every prime
    from 1031 up. 14 prime-column planes, 53 x 64 – 4099 x 2048, both directions, 8 threads, as built: 1.111 -> 1.032,
    15 faster, 2 slower (C2R 263 x 2048 0.83 -> 0.93; C2R 521 x 8192 1.05 -> 1.17, where the planner's window is 24);
    4099 x 512 1.38 – 1.54 -> 1.00. At one thread the window of 32 stays (8 was slower at 17 of 28). Bluestein is not
    the lever here: the planner raced it on these planes at both thread counts and banked the Rader chain, and the
    turned prime pass (Bluestein through the 1D prime route) is serial in the door. Open: the Rader inner chain at
    2053 (`4.27.19`, a spilling 27; `est__rd_inner_bad` rejects every candidate of 2052 for its unavoidable 19, so
    `est__rader` falls back to the engine's pick) -- the planner's `9.19.12` / `3.3.19.12` take the 2053 planes from
    1.03 – 1.11 to 0.91 – 0.97 at one thread and from 1.10 – 1.16 to 0.99 – 1.08 at 8; the filter is shared with the C2C
    estimate, the owner's call.
