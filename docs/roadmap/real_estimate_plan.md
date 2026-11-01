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

## 8. Noticed, not filled (unverified)
- The 1D lane-major real Bluestein path (`_vfft_create_real_lanes`, rows `eng=zrbl`) looks unreachable for an
  interleaved batch: `vfft.c` serves or refuses every lane-major K > 1 before `_vfft_create_real`.
- `fftnd_real_il.h` seeds the 3D plane from the 2D row keyed at `nthreads=T`, but creates the plane child at one
  thread. Whether the child's lookup reconciles the two was not checked.
- A `zfsr` row whose child is missing makes the door race the whole cell, its engine included
  (`il/wisdom/wisdom2_real_il.h`, `vw2_real_il_lookup_zfsr`), where only the child is missing.
