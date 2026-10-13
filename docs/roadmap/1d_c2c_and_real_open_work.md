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

1. **Non-power-of-two N above 262144: SERVED (built 2026-10-12, field trial item 1).** The K=1
   four-step admits every N in (262144, 2^24] with a divisor pair whose sides lie in [64, 16384]
   (`il/rank1/k1_fourstep_band.h`; `docs/design/k1_fourstep_design.md`, "The non-power-of-two
   band"): the eight most balanced splits whose column side the 2D tier chains and whose row side
   the K=1 door serves race against the flat DIT (and the prime cell where a factor past 47 leaves
   no chain), in both order classes; the transposes take tail blocks, the twiddle records
   ceil(N2/B) entries. Verified on the scratch store (1D c2c, T=1): 1000000 oop (fs 1250x800,
   11.3 ms) and in place (the flat chain won, 10.8 ms), 759375 (fs 1875x405, 9.9 ms), 531441
   (fs 6561x81, 2.9 ms), 1049760 (fs 972x1080, 4.9 ms), 786432 (fs 3072x256, 3.9 ms), 3145728
   (fs 3072x1024, 16.4 ms), 1000000 scrambled (fs); every forward within 1.2e-15 of FFTW and
   MKL, every roundtrip within 1.6e-15, no unwritten output, no canary touched. vs MKL at one
   thread (docs/performance/v1_0_results.md, "HIGH N"): 1.54x to 3.97x out of place, 1.49x to
   1.93x in place; at T=8 3145728 wins 1.33x, 12582912 ties, 1000000 loses 0.30x (its threaded
   2D child's column pass is serial). Cold creates take 3 to 13 minutes, 35 at 3145728 and 56
   at 12582912 (the flat search over hundreds of orderings and eight 2D children per cell).
   The four-step's row cells at each split's N2 come from the store's banked 1D rows, read only
   (`VFFT_HIGHN_PLAN=borrow`, the default; `=scratch` races them for the plan's own use), and
   reach the 2D child through a create-scoped row source; before 2026-10-13 the warmed rows
   never reached the child, so every replay raced the row cell again (V26's mechanism). The same change lets the factor-of-4 cells in (16384, 262144]
   outside ZTURN-T's odd band race to the flat DIT's ceiling (trial V17: the 16384 cap was the
   planner's initial scratch planes). The size promise above 2^24 stays open: the ladder's reach
   ends at 4096 x 4096 (pow2) and 16384 x 16384 (non-pow2 sides), nothing above has an engine.
   **BUILT**
2. **The radix 37-47 kernels are weak on odd chains.** Planned beside item 1, as its own
   piece. **OPEN**
2b. **A prime factor above 47 inside a composite length has no stage-level engine** (owner,
   2026-10-13: "clearly a missing gap", c2c first, then r2c/c2r). Rader serves a whole prime N
   only, and a composite with such a factor (2,157 of the 4,095 lengths to 4,096; 2101 = 11 x 191,
   2118 = 2 x 3 x 353, 1149 = 3 x 383) runs Bluestein on the whole length: 0.41x to 0.6x against
   FFTW, which runs the prime part as a Rader or generic leaf under a small-radix stage. A prime
   N whose N-1 is not chain-smooth (2131: 2130 = 2 x 3 x 5 x 71) loses Rader for the same
   reason. The candidate form is the four-step's 2D view at N = N1 x N2 with the prime part on
   the column axis, where the 2D tier already races the column Bluestein and the turned
   per-column prime cell (Rader at a prime column), and the smooth part as rows, admitted
   below 262144 for prime cells only; the same inner then serves Rader at a prime N. Not
   designed, not started. **OPEN**
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
9b. **DEFAULT order at K=1 is natural (built 2026-10-13, field trial V31).** The in-place
   door's natural block runs for DEFAULT as for NATURAL (bins natural to 2e-16 at 16..4096,
   360 and 1000), and the out-of-place door maps DEFAULT to NATURAL: where the cell has no
   natural champion (K=1 at 1000 and 360, where only the scrambled kind exists) it refuses
   with the reason and names SCRAMBLED, instead of serving the permuted spectrum under
   DEFAULT. The out-of-place natural champion at K=1 is item 9's race. **BUILT**

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
13. **Real N past the old inner ceiling: SERVED (built 2026-10-12).** The prime cell and the
    real Bluestein take the four-step as an inner (kind `fs`, its split as the shape), so the
    inner ceiling is the four-step's reach, 2^24 (`VFFT_POLICY_INNER_MAX_N`): odd real N with a
    prime factor above 47 is served to (3N-1)/2 <= 2^24, an even N whose half is such a cell
    through the zr2c child, and a composite c2c prime cell to 2N-1 <= 2^24. Verified on the
    scratch store against FFTW and MKL (within 1.3e-15): r2c 174767, 200005, 262202, c2r 174767,
    c2c 131101, 131073, 200005. vs MKL at one thread (canonical bench, both flip orders): r2c
    174767 2.24-2.25x, 200005 2.02-2.04x, 262202 2.24-2.25x, c2r 174767 2.24-2.28x; c2c 131101
    2.34-2.42x, 131073 2.25-2.27x, 200005 1.44-1.46x. **BUILT**
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
tier; the column chains' R16/32/64 spill test; the 2D heats' sample definition; a 2D/3D
cell whose last axis is a length item 1 or 13 refuses (16x131101, 32x262202, 64x200003,
4x4x236196) refuses only after 4-167 s of racing, and names the row child ("in-place C2C
N=131101" for an out-of-place request, a "transform-contiguous batch" at K=1) instead of
the request: until items 1 and 13 are filled, a guard at create entry refuses the row
length before any race and names the request (owner 10-12: a bug, not a gap).
