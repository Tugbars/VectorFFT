# 1D C2C and 1D real: open work

**Date:** 2026-10-04 · **Status:** OPEN, the list to return to after the wisdom work ·
**Scope:** 1D c2c and 1D r2c/c2r, both libraries, K=1 and batches.

Status tags: **OPEN** decided, not built · **HELD** designed, waiting on the owner ·
**RULING** needs a decision · **RUN** a calibration or gauntlet run, not code ·
**PARKED** parked by the owner · **KNOWN** acknowledged by the owner.

The wisdom system (persistence, profiles, machine identity, the door fixes) belongs to
`docs/research/wisdom_portability/REPORT.md`. Its items that touch these doors are named in
§4 so they stay visible here. 2D/3D items are named in §5 only.

## 1. 1D c2c, interleaved

1. **Non-power-of-two N above 262144 refuses** (e.g. 759375). The K=1 planner races
   odd-factored N only up to `VFFT_K1_IL_PLAN_ODD_MAX_N` (`il/planning/policy_il.h:49`),
   ZTURN-T stops at `VFFT_ZTT_MAX_N` (`il/rank1/ztt.h:121`), and the four-step is power of
   two only. Owner, 10-03: "Let's tackle this." No plan written yet. **OPEN**
2. **The radix 37-47 kernels are weak on odd chains.** Planned beside item 1, as its own
   piece. **OPEN**
3. **A missing backward row is served with default backward forms, without a race.**
   Correct, not optimal. Designed: an `ord`-keyed backward sibling row, raced when the
   sibling is missing. Owner, 10-03: "wait before you proceed." **HELD**
4. **Radix 32 and 64 in the interleaved c2c pipeline.** `codelet_radix_sunset.md` (09-16)
   lays out what the banked verdicts use; no ruling is recorded. **RULING**
5. **In-place engine forms below 2048** (`inplace_engine_forms.md`). In place runs exactly
   as fast as out of place, because no engine below 2048 has a form that saves a pass in
   place. Owner: "we can pursue this later." **OPEN**
6. **Bind every plan's executor at create** (`execute_door_binding.md` §2): the batch loop
   calls the inner engine's bound executor, and every tier binds its own. **OPEN**
7. **Arms that never win still race.** A recalibrate at N=240 takes 30 s over 141 races;
   the prime route wins 0 of 859 chain-able cells whose largest prime factor is at most 13,
   and the flat pool takes three quarters of the time for 9% of the wins (calibration
   census, 10-02). Which arms stop racing, and on what evidence, needs a ruling. **RULING**

## 2. 1D c2c, split

8. **In place, the forward output fails the DFT check** at N=256 K=8, 1000 K=8 and
   4096 K=32, with exact roundtrips; identical on mingw gcc, Linux gcc and ICX, and present
   at HEAD 44c380d3. Owner, 10-04: known. **KNOWN**
9. **The out-of-place K=1 door never races.** It serves a structural default and banks
   nothing (`split/rank1/c2c_oop_create_split.h`; wisdom report §2 item 8 and D10, which
   recommends documenting it now and leaving the race to this roadmap). **OPEN**

## 3. 1D real

10. **Real four-step (zfsr), steps 2 and 3.** Step 2 races whole zfsr plans in the real
    role: the child's own races become heats and the final pick is made on whole plans.
    Step 3 re-benches the 8 zfsr cells, which are misses until then. Owner, 10-03: "after
    this is done, let's tackle this one." **OPEN**
11. **No T=8 rows for even N.** The store's T=8 real rows are all odd (zrf 52, zrb 14), so
    every threaded even create (zttr, zr2c, zfsr) races until a T=8 calibration of the even
    cells is merged. Owner, 10-03: "should be solved asap." **RUN**
12. **In place is never calibrated.** The gauntlet's real contract is out of place only,
    and the 18 in-place zr2c rows are stale and race on create. Needs an in-place real
    contract in the gauntlet, then a run. Owner: "we should do it in future yes." **OPEN**
13. **Non-power-of-two N of 262147 and above.** Smooth odd N are served: zrf rows reach
    4782969. What refuses above 262144 has not been censused since the odd-real bridge was
    deleted (10-03). The candidates are odd N with a prime factor above 47, and even N whose
    N/2 child is a length item 1 refuses. Planned with item 1. **OPEN**
14. **c2r at N=2 refuses:** no engine builds it. No ruling yet. **OPEN**
15. **N=512 r2c picks zttr** (232 ns in its race), which benched 17% slower than zr2c did
    on 09-30. Unverified suspect: the stack-alignment lottery, since zr2c's child spills
    without zttr's stack-aligning entry. A re-race with the race log was offered, not
    decided. **OPEN**
16. **zrf's heats still take the old sample:** a 200 µs reps target with a reset before
    every sample (`_zrf_heat`, `il/real/odd_build.h:169`), not the gauntlet sample (at
    least 8 executes and 2 ms back to back) that the c2c planner's heats take since 10-04.
    **OPEN**
17. **Lane-major batches (K>1) cross to the split engines** through
    `bridge/real_bridge.h`, the one crossing left. The native IL engine is decided, not
    started (`il_real_lane_major_batch.md`), and is not to be started unprompted. **OPEN**
18. **Split rfft: `VFFT_RFFT_MAX_RADIX` compiles as 16, not the intended 32**, so the
    registry writes `reg->r2cf[32]` into an array of 17; ICX reports the out-of-bounds
    write (10-04). Parked by the owner, 09-01: "stop touching rfft." **PARKED**

## 4. Both transforms

19. **Re-race the rows banked by races that timed plain-malloc buffers.** The 10-04 sweep
    moved those buffers onto the aligned allocator. The 1D sites among the allocator
    research's nine: `il/rank1/k1_commit.h` (the two-pass order probe and the flat DIT's
    threaded race), `vfft.c` (the batch loop-vs-slabs race) and
    `split/natorder/natorder_calibrate.h`. Their rows carry `src=race date=` stamps.
    **RUN**

**Owned by the wisdom work** (wisdom report D1 and D4):
- A read-only miss races and discards the winner: the 1D c2c door serves its structural
  pair (N=1024: 827 ns against the winner's 718), the real door re-races on every create,
  and a power of two of 8192 or more refuses after its race.
- A banked row this build cannot construct falls through to a pair.
- zr2c's route race keeps a 3% bias toward the structural route
  (`il/real/zr2c_build.h:756`), older than the no-bias rule.
- The lane-major real create returns NULL when wisdom is off, so the request falls to the
  split routes; it should race and not bank.
- Create-time races are not pinned, so a user's create can time on an E-core.

## 5. Not in this file

2D/3D: 2D/3D rows read and write 1D wisdom (deferred, owner 10-01); an IL 2D plan with an
explicit lane-major geometry and K>1 is served contiguous; prime N1 refuses on the IL 2D
tier; the column chains' R16/32/64 spill test; the 2D heats' sample definition.
