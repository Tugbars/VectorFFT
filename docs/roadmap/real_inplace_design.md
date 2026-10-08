# In-place real transforms at rank 2 and 3 — the padded-plane contract (design of record)

*Declaration of the in-place R2C and C2R contract of the interleaved real tiers
(`il/rank2/`, [`fft2d_real_il_design.md`](fft2d_real_il_design.md); `il/rank3/fftnd_real_il.h`,
[`fft3d_real_il_design.md`](fft3d_real_il_design.md)). DESIGN 2026-10-08, decided with the
owner. BUILT 2026-10-08: 2D r2c in place through door 1 (§1), one thread, gated by
`src/tools/gates/il2d_real_ip_gate.c` (vs FFTW in place, replay, the door's refusal,
the store reloaded from disk; 71 checks); 3D r2c and c2r in place the same day (§2,
`il3d_real_ip_gate.c`, 100 checks). Door 2, the 2D c2r twin, the pay-once and band
c2r arms in place, and the threaded forms are the next pieces. In place is a feature the library provides whatever it
measures (owner, 2026-10-08): the race decides how it is served, never whether.
The 1D in-place real contract (`vfft.c`, the zr2c route and the odd door) is the
rank-1 case of the same layout and stays as it is.*

## 1. The contract

**The layout is FFTW's and MKL's in-place real layout.** One interleaved plane
holds the real input and the CCE output; the last axis is padded so a row of N
reals and its N/2+1 complex bins occupy the same `2·(N/2+1)` doubles (N+2 at
even N, N+1 at odd N). `in == out`: a distinct output plane is refused at the
execute door, the 1D law (`vfft_execute.h`). `destroy_input` is implied for
C2R (an in-place C2R has no input to preserve). One transform at rank 3; rank 2's
`howmany > 1` rides the plane queue, one padded plane per transform, later.

**The row pitch is plan input.** Two doors:

1. **The caller's own buffer** (`owned_buffers = 0`, the default): the row pitch
   is exactly `2·(N/2+1)` doubles. That pitch is the caller's and cannot move.
2. **The plan's own plane** (`owned_buffers = 1`): create allocates ONE plane at
   the pitch the policy names, zeroed; `vfft_plan_planes()` returns it in both
   roles (`sre == dre`, the `im` roles NULL) and `vfft_plan_stride()` returns the
   row pitch in doubles. The policy (`planning/policy_il.h`) names the smallest
   pad past `2·(N/2+1)` at which no pass of the plan aliases itself: the CCE
   pitch where it is harmless, `hp1 + 8` pairs where `16·hp1` is 16 bytes past
   a 4 KB multiple (`512 | N2`), the staged scratch's own search otherwise
   (`fft2d_create_il.h`: every leg stride and the leaf stride non-zero mod
   4096). `vfft.h`'s "1D and SPLIT only" on `owned_buffers` widens to this plan.

Door 2 is the fast path and costs nothing at execute: the alias of §3 never
exists. Door 1 is served by the raced forms of §3.

## 2. The walks in place

**Placement is per pass, not per plan** (owner, 2026-10-08). A form is a
sequence of kernel calls, each landing where the form says, in or out of
place; the in-place contract binds only the two ends, the first read and the
last write on the caller's plane. The natural column pass is already that
shape (stage 0 out of place onto the pre-leaf plane, the mids in place there,
the leaf out of place back). So every walk shape that can serve in place is a
candidate, and every codelet in either placement stays one: a private landing
(the c2r column-inverse plane, the r2c skewed plane, the pre-leaf plane) can
sit anywhere. Same kernels, same calls; what changes is where a pass lands.

**The race is the cell's own** (owner, 2026-10-08). An in-place cell races
its own candidates, built and timed in place on one padded plane, and banks
the winner on the cell's own `pl=IP` row of the 2D or 3D shard (`rx=`, `cx=`,
`tf=`, `csk=`, `cxp_c2r=`, the rank-3 tokens): the key value exists, no new
token. An in-place create never reads nor writes the cell's `pl=OOP` row, and
the out-of-place cell keeps its verdict. The 2D-borrow law holds within a
placement: a rank-3 in-place create seeds its plane children from the 2D
shard's `pl=IP` rows only. The rejected alternative: the `pl=OOP` plan run
with `out == in` (its row engines are other builds; its verdict was timed on
two planes).

**r2c, rank 2.** The row pass runs in place row by row: each row's N2 reals are
read before its hp1 bins are written, by an engine whose in-place form exists
(the 1D engines' in-place forms, the door's in-place K = 1 plan). The column
pass is unchanged: stage 0 to the pre-leaf plane, the mids there, the leaf
back onto the caller's plane. The fused walk is legal as it stands: it reads
every row before its first write to the plane. The one-kernel pass at the
caller's pitch is §3's case.

**c2r, rank 2.** The column pass lands where today's destroying form lands: a
one-kernel reverse pass in place on the caller's plane, the default here
(no permission to ask: in place destroys); a chain of two stages or more
through its own scratch, the leaf back onto the plane. The private landing
(`cxp_c2r=`) stays a raced form. The backward row pass runs in place row by
row.

**rank 3** (BUILT 2026-10-08, both directions, one thread; gate
`src/tools/gates/il3d_real_ip_gate.c`, 100 checks). r2c: the child arm runs
the rank-2 in-place plan per plane, then axis 0 in place (the pass that
already existed); the pay-once arm moves each plane into its private volume
first, where its row engine runs in place, and keeps its out-of-place leaf
into the caller's volume. c2r: axis 0 as the forward chain in place in the
caller's volume (`destroy` by nature), which leaves position q holding the
CCE plane of real plane (N1 − nat[q]) mod N1; the planes then walk the cycles
of that permutation BACKWARDS through one buffer (the cycle's first plane
copied out, every other plane produced from the position that holds it,
`pos0`, straight into its own place): the child arm through the out-of-place
rank-2 c2r child and a tight plane whose rows land at the padded pitch, the
pay-once arm's axis-1 stages in place on the source position and its rows
landing directly (the backward row set takes the output pitch). One plane
copy per cycle; no rank-2 c2r in-place plan needed. The band is not an
in-place arm: its planes land across bands, and saving every destination's
unprocessed content is the private volume again. The strips form serves both
directions. The cell's own `pl=ip` row of the 3D shard; the
2D borrow reads the 2D shard's `pl=ip` row for an in-place r2c plane child.
The race decides per cell, in place on one volume re-laid before every
sample.

**Threads.** Unchanged in structure: the row pass by row ranges, each worker
in place on its own rows; the column and axis-0 partitions as they are.

## 3. What in place changes, measured

- **Saved:** the write-allocate of a cold output plane in the standard walk's
  row pass. Beyond L2 the row pass moves two planes of traffic instead of
  three (at 2048×2048: 64 MB for 96, about a tenth of the standard r2c
  walk). The column pass gains nothing (its leaf writes cold lines either
  way); the fused walk gains footprint only. On one-kernel c2r cells the
  destroying form's landing is saved: 1.06–1.27 measured 2026-10-06.
  Measured on the gauntlet, 2026-10-08, 2D r2c door 1, one thread, 16
  cells, each contract its own race: in place over out of place 1.09–1.16
  where the standard walk serves past L1 (16×1024, 32×1024, 128×512,
  256×1024, 512×512), 1.00–1.05 at 64×64 to 1024×1024, 0.95 at 2048×2048
  (both contracts the fused walk: nothing to save); FFTW in place over out
  of place 0.92–0.95 at L2 sizes, 1.05–1.20 from 512×512. Against FFTW in
  place: 1.05–1.40 from 64×64 up (16×1024 at parity, 0.99), 0.59–0.81 on the
  tiny cells (64×15, 15×16, 64×30), where 15×16 in place costs 1.7× its own
  out-of-place plan: the rows kernel is out of place only, and a per-row
  engine at N2 = 16 is call overhead. The real axis on N1 (`raxis`) is not
  yet admitted in place: 64×15 is 0.92 of its out-of-place plan for it.
  Rank 3, the same day, 11 cells: r2c in place over out of place 1.15–1.35
  on every cell past L2 (8×128×2048 1.35, 128³ 1.33, 32×256×256 1.25), and
  1.05–1.19 against FFTW in place from 32³ up (64×128×128 0.97; the tiny
  cells 0.68–0.69, the plane child's call overhead at N3 ≤ 32); 128³ goes
  from 0.83 of FFTW out of place to 1.05 in place. c2r in place, with the
  planes walked backwards through one buffer and the pay-once arm in place
  (it wins the in-place race at 9 of 10 gated cells): over its own
  out-of-place plan 1.07–1.38 on 8 cells (128³ 1.38, 32×256×256 1.22, 64³
  1.19, 64×128×128 1.14) and 0.93–0.98 on the three smallest; against FFTW
  in place 1.16–1.49 at 16³, 32³, 64³, 16×256×256, 64×128×128, 32×256×256
  and 128³, parity at 32×8×1000 and 16×64×1024 (0.96–0.97), behind at
  8×128×2048 (0.81) and the tiny 9×16×30 (0.78). The first walk, two
  buffers and the child arm only, had been 0.56–0.76 where the plane
  buffers left L2; the backward walk recovered it (8×128×2048 0.56 → 0.81,
  16×64×1024 0.76 → 0.96, 32×8×1000 0.75 → 0.97).
- **Unchanged:** the shuffle count. The shuffles are the kernels' lane-order
  conversions and the twiddle swap (the fold's 20 on port 5, the IL boundary
  stages); placement changes addresses, not lane order.
- **The hazard, door 1 only:** at `512 | N2` the caller's pitch is 16 bytes
  past a 4 KB multiple, where a one-kernel column pass in place aliases its
  loads against the stores of the pair just written: n1c_16 2.4–2.8× slower
  (`il2d_real_pitch.h`). Three levers break it, each attacking one of the
  three conditions (the pitch, one plane, the pair-by-pair order):
  - the pitch — door 2 (decided);
  - a second plane — the private landings (built, raced): one more plane
    pass, recovering 1.12–1.18 of the loss at N1 = 16;
  - the order — the blocked strided twin of n1c_16 (measured on scratch
    2026-10-07: 2.5–2.7× at the kernel, the whole plan 1.28–1.32 at
    16×512/2048/4096, parity with FFTW; a raced twin, 7–14% slower off the
    pitch; not built; N1 = 16 only, the plain order barely suffers at 64 and
    128). Open, §4.

## 4. Open decisions (one at a time, with the owner)

1. ~~The race.~~ DECIDED 2026-10-08, §2: the cell's own race and wisdom
   cells, every form and codelet in either placement a candidate.
2. **The rows kernel.** r2zr (N2 ≤ 32) declares `__restrict__` in and out and
   serves a lone last row by re-running the row before it, so in place it
   is undefined and wrong. Either a generator variant for in place (no
   restrict, the lone row served alone), gated on speed, or the kernel stays
   out of in-place cells.
3. **The strided twin** for door 1 at N1 = 16 (§3).
4. **The order of the build.** Proposed: 2D r2c, 2D c2r, rank 3, the threaded
   forms; each gated against FFTW's in-place `r2c_2d/c2r_2d/r2c_3d/c2r_3d`
   per cell, MT == ST, replays bitwise; a gauntlet cell per step.

## 5. What the build touched (2026-10-08)

The placement entered the rank-2 real row key (`vw2_ilcol_key_t.ip`, the
`vw2_2d_rl_*` wrappers' last argument; the chain race's fresh record carries it
too). The real plane's row pitch is one helper, `_il2d_rp(h)`: `N2` out of
place, `2 il2d_ipP` in place, threaded through the row-pass bodies, the row-set
helpers and the odd-N2 route. The row engines are built in the plan's placement
(`_il2d_rowx_cfg`); the door route in place runs the K = 1 inner per row. The
rows kernel is never an arm of an in-place race. The three whole-transform
races (rows, the fused walk, the skewed plane) run in place on one plane re-laid
before every timed sample (`_il2d_ip_reset`, the race proto's hook) with reps
capped at 32, so a sample stays finite; the column plan race already ran its
pass in place. The front door admits the request through
`vfft_policy_il2d_ip_ok`; the execute's fast path takes the in-place call and
the door refuses two planes.

## 6. Gates

Every served in-place plan is gated against FFTW in place on the same padded
plane; a one-thread plan against its out-of-place twin (the same numbers at
≤ 1e-12 relative, bitwise where the form is the same); the threaded form
against the serial one; a replay bitwise against the create that banked it.
The gauntlet's in-place real cell (2026-10-08): `bench_1d_vs_mkl --2drealnat
--realfwd --realip` runs ours in place against FFTW or MKL in place on the same
padded plane (the reference stays the comparator's out-of-place spectrum of
the same input; path `nat-ip`), and `gauntlet.py --real r2c --inplace` drives
it, its calibrate stage banking the cells' own `pl=ip` rows into the run's
store copy. The out-of-place run of the same cells is the other half of the
comparison. Rank 3 and c2r join the cell with their pieces; door 2 with its.
