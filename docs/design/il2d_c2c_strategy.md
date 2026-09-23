# The 2D interleaved c2c strategy: two passes, three batched arms, one race

The 2D interleaved complex transform of an N1 x N2 plane (N1 rows of N2 points) is two
passes that commute: the **column pass** along N1 and the **row pass** along N2. Every
verdict about how those passes run is made by the front door: a `vfft_create` on a cell
with no banked verdict races the candidates on the full forward execute, banks the winner
on the cell's wisdom row, and every later create replays it. Nothing below is a rule;
each is an arm the door can pick, per cell.

This document declares the arms, what each one solves, why there are three of them
beside the plain route, and what they did to the numbers.

## The cell

| | |
|---|---|
| contract | c2c, interleaved, natural order, either placement, K=1; the gauntlet measures out of place, one thread, against MKL DFTI 2D out of place |
| column pass | the column chain: stages of radix R over the plane, every kernel call covering `rn` adjacent columns two per vector (`il2d_col`); a natural-order leaf; the Bluestein column axis for prime N1 |
| row pass | every row a length-N2 transform on the destination plane, after the column pass (stage 0 of the chain does the source-to-destination move) |
| verdict | `chain= wl= tf= sw= ro= rbk= turn=` on the cell's `lay=il` row, one race |

The plain route runs the row pass as N1 calls of the in-place K=1 plan at N2 (`ro=0`): one
door walk per row. It is the coverage route: it exists for every N2 the 1D in-place tier
serves, which is every N2 (the tier's last candidate is the prime engine).

## Why the plain route loses

Measured on the same plan with the routes toggled (`il2d_phase_probe`), 2026-09-23:

| cell | column pass | row pass, plain | what the rows cost |
|---|---|---|---|
| 8x8 | 15 ns | 59 ns | 8 door walks into the radix-8 kernel at count 1, its 128-bit one-lane tail |
| 32x32 | 344 ns | 563 ns | 32 door walks into the two-pass child: two prologues and two short lane loops per row |
| 8192x2 | 55.8 us | 34 us plain, 2.7 us batched | the column chain over ONE lane pair: 3.5 ns per point where the 1D engine does 0.55 |

Three different costs, so three arms.

## Arm 1: the batched mono rows (`ro=2`)

**Solves:** the per-row door and the count-1 tail at row lengths the mono kernels cover.

The `n1ccs` kind is the `n1c` leaf with column-stride addressing: lane k is one whole
row at pitch `Gs`, two rows per vector through paired 128-bit loads and stores, in
place, both directions, every mono radix (2..47, 64). One call runs a whole run of rows:

```
fn(rows, NULL, rows, NULL, NULL, NULL, 1, pitch, 1, pitch, nrows)
```

No door, no per-row prologue, full width where the solo ran its tail.

| cell | plain | batched mono | MKL |
|---|---|---|---|
| 8x8 | 86 ns | 36 ns | 36 ns |
| 16x16 | 260 ns | 161 ns | 146 ns |
| 64x8 | 767 ns | 333 ns | |

It loses at 32 and 64 rows: the batched mono at radix 32 is slower than the two-pass
child's arithmetic, and radix 64 spills.

## Arm 2: the batched two-pass rows (`ro=3`, tile `rbk=`)

**Solves:** the two-pass child's per-row prologues and short lane loops at N2 = 32 and 64,
without changing its arithmetic.

The row child's own factorization (R1 x R2, its bound kernel forms including the tangent
leaf) runs through the **row-loop twins** of its four stage kernels (`n1tr`, `t2r`,
`t2tr`, `n1r` and their `tan` forms): the same body with its lane loop wrapped in a loop
over rows. On the frozen 11-argument ABI, `count` = rows x `Ls` lanes (a two-pass stage's
legs are strided by the other factor, which is its lane count), `Gs` the input row pitch,
`OGs` the output row pitch. Per chunk of rows: stage 1 rows into a per-worker scratch at
pitch N2, stage 2 scratch back into the rows; backward `t2t` then `n1` the same way.

**The tile is raced, not ruled.** A 64-row chunk at N2 = 32 spilled L1 and lost to the
plain route. The chunk scratch in KB, {4, 8, 16, 32}, is a ladder of arms beside the
route, banked as `rbk=`, exactly as ZTURN-T races its tile. 4 to 8 KB win at most cells,
16 KB at some.

| cell | plain | two-pass batched | MKL |
|---|---|---|---|
| 32x32 rows | 563 ns | 377 ns | |
| 32x32 whole | 930 ns | 730 ns | 721 ns |
| 64x32 rows | 1151 ns | 757 ns | |
| 4096x32 rows | 76.9 us | 50.3 us | |

Bitwise the plain route where the child's forms are the plain ones (same kernels, same
order of operations). The blocked radix-32 forms have no twin, so N2 = 128 and 512 have
no arm yet.

## Arm 3: the turn route (`turn=1`)

**Solves:** the column chain on narrow planes, where a stage call covers one lane pair
and streams the whole plane once per stage.

No column chain at all. The batched mono row kernel stores its row DFTs **transposed**:
its store addressing is `zout[2*(l*OLs + k*OGs)]`, so an output leg stride of N1 and a
lane pitch of 1 land row k's leg l at column l, row k of an N2 x N1 scratch. The N2
columns of the plane are now rows of that scratch, N1 long and contiguous, and run
through the in-place K=1 natural plan at N1: the door's own banked 1D verdict, ZTT,
four-step or prime. One back-turn (`_il2d_turn_back`: 2 x 2 complex blocks through one
lane permute, destination rows blocked so 256 of them stay in L1) writes the plane.

Natural cells only: the scrambled contract keeps the chain's digit-reversed column
order, and the four-step's 2D child is a scrambled cell whose rows carry a twiddle hook.
One arm in the race, no band, no tile. Serial for now; a turn plan runs no MT race.

| cell | chain + batched rows | turn | MKL |
|---|---|---|---|
| 8192x2 | 58.7 us | 26.5 us | 53 us |
| 8192x4 | 132 us | 53.5 us | 78 us |
| 4096x2 | 35.6 us | 12.7 us | 26 us |
| 4096x8 | 90.8 us | 55.4 us | 68 us |
| 2048x8 | 50.6 us | 26.4 us | 33 us |
| 256x8 | 3.2 us | 2.7 us | |
| 64x8 | 336 ns | 409 ns | |
| 8x8 | 36 ns | 88 ns | |

The door picks it at every N1 >= 256 with N2 <= 8 and at some 16-column cells, and drops
it where the plane is small. At N2 = 8 the turned stores and the back-turn are half of
its time; the 1D plans are the rest.

## Why three arms and not one method

The winner flips with the shape, and the door has logged every arm on the same cells:

| row length N2 | winner | margin over the next |
|---|---|---|
| 2, 4, 8 at N1 <= 64 | batched mono | 1.3 to 4x over the plain route |
| 2, 4, 8 at N1 >= 256 | turn | 1.25 to 2.3x over the batched mono |
| 16 | batched mono and two-pass tie | |
| 32, 64 | two-pass batched | 1.03 to 1.30x over the plain route |
| lengths with no mono kernel and no twin | plain route | the only arm |

A dominant variant becomes the kernel; none of these dominates, so they stay arms and
the race stays. What was deleted is the arm that never won: the out-of-place row child
with a copy back (the old `ro=1`), which lost 1.5 to 5x wherever a batched kernel existed
and sat within noise of the plain route elsewhere; it also had no forced-path case left,
since the in-place tier serves every N2.

## What it did to the grid

The pow2 grid: every 2^a x 2^b with 2..8192 per axis and planes up to 2^22 points, 159
cells, every cell re-raced by the door with the calibrate probe holding the SMT sibling
(an unguarded race is a single cold race in the two-speed lottery, and it re-raced
128x128 onto a column chain 37% slower than a guarded one).

| state | median vs MKL | cells below parity | below 0.8 |
|---|---|---|---|
| the plain route (18-cell group) | 1.09 | 7 of 18 | 4 |
| + batched mono | 1.17 | 31 | 12 |
| + two-pass batched, tile, guard | 1.21 | 23 | 7 |

Cell by cell, plain route to the guarded grid: 8x8 0.39 to 0.95, 32x32 0.79 to 1.01,
1024x32 0.86 to 1.52, 4096x16 0.72 to 0.93, 8192x64 0.91 to 1.12. The turn route's grid
follows below once measured.

## What remains

- 128x16 at 0.52: a three-stage natural column chain on a small plane, its own probe.
- The last nanoseconds of door at 8x8 and 16x16: a bound 2D execute like the K=1 door.
- Row-loop twins for the blocked radix-32 forms (N2 = 128, 512).
- The turn route threaded: N2 independent long transforms are trivially parallel.
- The library's own front door has no sibling guard: a user's create races in the same
  lottery the calibrate probe did.

## Where things are

| what | where |
|---|---|
| the routes' fields and serving | `src/core/vfft_internal.h`, `src/core/transforms/fft2d/il2d_tier.h` (`_il2d_rows_exec`, `_il2d_turn_exec`), `src/core/vfft_execute.h` |
| the race, the bank | `_il2d_axis_race` in `il2d_tier.h`; tokens `ro= rbk= turn=` (`wisdom2_2d_reader.h`) |
| the kernels | `codelets/zil/avx2/pure_il/radix*_z_n1ccs*`, `radix*_z_{n1tr,t2r,t2tr,n1r}*`; the generator's `--cil-n1ccs` and `--cil-rowloop` |
| the pins | `VFFT_IL2D_ROWOOP=2|3|4`, `VFFT_IL2D_RB2_KB`, `VFFT_IL2D_LOG=1` prints every arm |
| the measurements | `gauntlet/results/gauntlet_2d-pow2grid*/report_2d.md`; the phase probe in the session scratchpad |
| the arm list | `docs/design/measurement_arms.md` E1.5, E1.5b, E1.5c |
