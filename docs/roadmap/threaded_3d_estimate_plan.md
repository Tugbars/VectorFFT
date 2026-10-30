# Plan: the threaded 3D estimate (complex, natural order, T > 1)

**For approval.** Nothing in the library changes before the build in step 7, and no estimate rule is built
without the owner's approval of it. Planning races bank into `src/wisdom/14900KF/`; forced or pinned probes run on
scratch copies of the store; every timing holds the hyperthread siblings of the cores in use
([`pool_sibling_guard.md`](../design/pool_sibling_guard.md)). The research behind this plan (six read-only
reports, a draft plan and its critique, the scripts) is in `docs/research/estimate_threaded/evidence/nd/3d_research/`
(`result.json`, `plan.md`, `critique.md`).

## 1. What we know

**The store** (`src/wisdom/14900KF/wisdom2_3d.txt`, 2,210 rows):

| Rows | Threads | Cells | Not a power of two | Dated |
|---|---|---|---|---|
| complex, natural, out of place | 1 | 1,345 | 55 | September 1,288, October 57 |
| complex, natural, out of place | 8 | 534 | 0 | 2026-09-25 (446), 09-26 (88) |
| real (r2c), natural, out of place | 1 | 328 | 5 | October |
| complex, scrambled | 1 | 3 | 1 | October |

The 534 rows at 8 threads keep their threaded decisions when created, but none carries the saved plan of its
planes (`plane_` / `rp_` tokens), so the first create of each cell races that plane plan (2-13 s; checked on a
scratch copy: 5 of 5 raced, the threaded tokens unchanged, 24-39 tokens added). They were raced unguarded,
before the caller pin and the pool fixes. Their `ns=` is the one-thread axis-0 time: the store holds no threaded
timing. Cubes run 128 B - 64 MB with none between 8 KB and 64 KB.

**How the door threads a cube** (`src/core/il/rank3/fftnd_il.h`):
- Axis 0 runs as a column pass over an N1 x (N2·N3) plane on min(N2·N3, T) workers. In the natural class each
  worker walks strips of `nsw` columns through its own dense scratch.
- The planes then run on min(N1, T) workers, or on a half team. Each plane is a one-thread 2D plan (the child,
  s=1) or the flat tier (s=2: its own axis-1 pass plus a 1D row plan). No plane is threaded inside.
- The race at T offers the one-thread arm (cubes up to 512 KB only), and the plane arm with each structure, on
  the full team and on the half team where one exists. The band arm (`cmt=1`) and the cycle form at T are never
  raced. The strip width and the axis-0 chain come from the one-thread race.

**The old verdicts at 8 threads:**
- The plane arm on 531 of 534 rows; one thread only at 2³, 4³ and 8³.
- Strips on all 438 rows where the door allows them. The door forces this, so it is not evidence: in the
  09-25 race, which raced both forms, the cycle form won 13 rows (for example 64x4096x4, by 2-9%) before 09-26
  overwrote them.
- Child on 393 rows, flat on 141. That equals the one-thread structure on 75% of the cells, and two one-thread
  races of the same cell agree on only 83%.
- Half team at N1 = 8: the full team won all 7 cells with a 2 MB plane; the half team won 16 of 17 with a larger
  plane. At N1 = 4 the half team won 0 of 22. All of it is powers of two at 16 MB and above.
- The one-thread boundary is unsettled: the policy says threading lost up to 256 KB, the 09-24 grid ran one
  thread on every cube up to 16 KB and on 39 of 44 at 32 KB, and the 09-25 race chose threads on all 108 cubes
  of 64-512 KB.

**The one-thread 3D estimate today** (76 cells, `docs/performance/estimate_performance.md`): 1.017 against the
planner overall; L2 1.013 (37 cells); L3 1.033 (29 cells, 12 more than 5% slower); only 2 cells within L1d and 8
past L3. The threaded estimate inherits these recipes.

**MKL** (`docs/research/mkl_dfti/md_2d_3d.md`):
- One thread only for equal-sided cubes of side 15 or less (its tiny backend).
- Otherwise planes on min(T, N1) workers in contiguous blocks, then the outer axis on min(T, N2) workers, or
  pencils.
- One cache test: threads per plane = ceil(T·32·N2·N3 / LLC).

Against MKL at 8 threads (old rows, unguarded, 1,033 cells, MKL's time over ours):

| Cube | up to 32 KB | 64-256 KB | 0.5-2 MB | 4-16 MB | 32 MB | 64 MB |
|---|---|---|---|---|---|---|
| MKL / ours | 6.47 | 1.59 | 1.25 | 1.26 | 0.92 | 0.93 |

At 32-64 MB with N1 of 8 or less we stand at 0.64-0.74. The main cause is the pass order (DRAM traffic 2.1-2.6
times MKL's, `docs/roadmap/fftnd_il_mt.md`); the half team recovers 2-14% of it at N1 = 8.

## 2. Cache regions and cell sets (8 threads, out of place, natural order)

Each cell is tagged three ways, because one core's view is the wrong scale at 8 threads:
- the cube's bytes against L1d (48 KB): the one-thread question;
- each worker's share against its L2 (2 MB): cube / T in the axis-0 phase, one plane in the plane phase;
- the out-of-place footprint (32·N1·N2·N3) against the L3 (36 MB).

The regions, by cube bytes: up to 48 KB; 48 KB - 2 MB; 2-16 MB (each worker's eighth within its L2);
16-36 MB (the footprint crosses the L3); above 36 MB.

| Set | What | Cells | When |
|---|---|---|---|
| A | reference: guarded races into the real store, about 24 per region | 120 | step 3 |
| B1 | one-thread boundary, 8-512 KB, several shapes; timed as forced arms (one thread vs threads) on a scratch copy | 30 | step 3 |
| B2 | half team: N1 = 5..15, planes 0.5-8 MB | 36 | step 3 |
| B3 | child vs flat, forced, only on planes where the estimate's two candidates differ (sides not both powers of two) | 30 | step 4 |
| B4 | strip widths, forced, including N1 of 1000 or more | 30 | step 4 |
| S | Rader axis 0, Bluestein axis 0, prime plane sides: 8-10 cells each, half in A, half in C | 30 | steps 3 and 6 |
| C | unseen, picked by script after the rules are frozen; a fixed quota at each rule's boundary | 80 | step 6 |
| D | A + C against MKL, guarded | 200 | step 8 |

Within A and C: about a quarter cubes, a quarter flat (N1 of 16 or less), a fifth tall or few-column, the rest
other shapes; about half not powers of two. C excludes every cell of the 09-24 grid (951 power-of-two cubes), the
534 old rows, A, B and the pilot. Past L3 includes one cube of 256 MB in the pilot; the larger sizes are dropped if
it races longer than about 2 minutes.

## 3. Steps

| # | Step | Tool | Output | Time |
|---|---|---|---|---|
| 0 | Guards for the research (owner, 2026-10-30: integrate the whole fix first): the library's own guards hold the second hyperthread of every core the pool uses, in its races and in its pool ([`pool_sibling_guard.md`](../design/pool_sibling_guard.md), built 2026-10-30); the tools add none of their own and run unconfined or under `vfft_measure_confine(0)`, which keeps the siblings in the set | library | guarded races and timings | done |
| 1 | Tools in the scratchpad: `plan3d_mtg.c` (3D race at T under the library's guards; prints the threaded fields, seconds, RACED or replayed); `ab3d_mtn.c` (several arms in one process, rotated, inside one measurement scope so the library holds the guards, a second planner arm as control, the engagement counter, a check that no arm measured anything, and an MKL arm); a 3D twin of `est_rows_lib.c` (the library's estimate rows at T); `cells3d.py` | scratch | tools | about 3 h |
| 2 | Pilot: 2 cells per region plus the expensive classes (256 MB, tall, Rader, Bluestein) | `plan3d_mtg` | seconds per race, the budget | about 30 min |
| 3 | Race A, B1, B2 and half of S, guarded, into the real store | `plan3d_mtg`, `ab3d_mtn` | fresh rows with their plane plans; verdicts by region and shape; the one-thread boundary and the team by time lost | 1-2 h |
| 4 | Score each draft rule of section 4 on A: once with the planner's recipe and once with the estimate's one-thread recipe, so threading loss and recipe loss separate; run B3 and B4; where a lead rests on fewer than about 10 cells, one region or powers of two only, race more cells before believing it | `ab3d_mtn`, scratch stores | estimate / planner per rule, region and shape | 2-3 h |
| 5 | The owner approves the rules to test; they are written down before C is picked | | frozen rules | |
| 6 | Race C and the other half of S into the real store; score; add in-place and backward arms on 20 cells of C (the row serves both placements and both directions) | `cells3d`, `plan3d_mtg`, `ab3d_mtn` | geomean, median, counts above 1.05 and 1.10 per region, every cell above 1.10 with its cause | about 1.5 h |
| 7 | Build, after approval: `est_mt_3d`; the threaded fields on `est_plan3d_t`; `est_put_rows_3d` keyed at T and writing both structures' plans where needed; the preflight at T (including the `nsw` beside the cycle form refusal); `est__colkey` at T; lift the refusals (`vfft_api.h`, `c2c_estimate_nd_il.h`) and invert the gate's "rank 3 at 4 threads" refusal check; a D3T gate section; `estimate_race_nd.c` at T; the sketch's stale "s=1 always" text; check that the library's rows equal the scored rows and the forward is bitwise equal | library, `estimate_gate` | the threaded 3D estimate | about a day |
| 8 | Against MKL on D, guarded, both engine orders, for the estimate's rows and the planner's | the gauntlet (its confinement keeps the siblings since 2026-10-30) or `ab3d_mtn` | the standing per region against the old one | about 1 h |

## 4. First rules to test (each tied to a fact of the door)

- **R0, a baseline that copies MKL.** One thread for equal-sided cubes of side 15 or less; otherwise the plane
  arm on the full team with the one-thread structure.
- **R1, one thread.** When the cube fits L1d/2 (24 KB), as in 2D; a second candidate at L1d (48 KB). Fact: a
  threaded call costs two fork-joins and T-1 clone executes. B1 decides.
- **R2, partition.** The plane arm everywhere else. The band arm is not written: the planner never produces it.
- **R3, form.** Strips when the axis-0 chain has two stages or more and no Bluestein; otherwise the cycle form.
  This is the door's own rule.
- **R4, team.** The door's own condition for a half team (it keeps at least 2 workers); the half team when
  N1 < 2·min(N1, T) and a plane is larger than L2. The plane-size clause is read from 24 power-of-two rows: B2 and
  C's quota must confirm it, and it needs the owner's ruling as a data-read threshold.
- **R5, structure.** Two candidates: the one-thread estimate's structure, or always the child. Scored by time
  lost on B3, not by matching the planner (two one-thread races agree on only 83%). A flat structure whose row
  plan cannot be cloned cannot thread; it is excluded.
- **R6, strip width.** The 2D width rule with N2 replaced by N2·N3 and the door's 3D width pool. It inherits the
  2D defect (32-64 columns where 8-16 is best at 1000 rows or more), so B4 includes those cells.
- **R7, recipes.** The axis-0 chain and the plane plan stay as the one-thread estimate makes them; T is written as
  the request's ceiling.

## 5. Decisions for the owner

1. Decided (owner, 2026-10-30): the library's own guards, built first, hold the second hyperthreads in every
   race and execute of the research; the tools hold none themselves.
2. Re-race the 534 old rows? Not needed for scoring: only freshly raced cells are scored.
3. The band arm (never raced, an old note had it 10-17% ahead at 8192x8x2 and 8192x32x4) and the cycle form at T
   (won 13 rows in the 09-25 race): report as door gaps only, or race them as forced arms on about 10 cells each?
4. The pass bar for C. Suggested: geomean at most 1.05 per region, every cell above 1.10 explained.
5. The MKL comparison: the confinement helper keeps the siblings now (built with the guards), so the gauntlet
   times both engines under the library's guards. Open: hot input (the gauntlet's default) or cold (it moves
   the lead by 16-28%)?

## 6. Door gaps to report, not fill

- The band arm and the cycle form at T are never raced (section 5, point 3).
- The 32-64 MB loss to MKL with small N1 comes from the pass order; an estimate rule recovers only part of it.
- In place, the race once picked flat where the out-of-place child is faster (`fftnd_il.h:1241-1245`).

## 7. Risks

- Unguarded threaded timings can run 2.4 times slow for a whole process: every arm is guarded and carries a
  second planner arm as control; threaded arms are never paused.
- Race time today is unknown (the October one-thread 3D creates took about 21 s at the median, a 2x2x65536 create
  135 s); the pilot sets the budget and drops the largest cubes if needed.
- Draft rows can race silently when they lack a plane plan; every A/B arm checks that it measured nothing.
- The one-thread recipe's gap in the L3 region (1.033) carries into every threaded result; step 4 separates it.
- Other sessions share the machine: runs go in quiet windows with the control arm watched.

After 3D (owner, 2026-10-30; the order between the two is not set):

- 2D and 3D at 4 and 2 threads.
- The real estimate (R2C and C2R) as its own workstream: the 1D real formulas brought up to the C2C ones and to
  the 10-29 thread rulings, then a 2D and 3D real estimate.
