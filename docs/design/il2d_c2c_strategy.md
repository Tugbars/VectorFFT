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

The scratch's row pitch is N1 + 8 complex, not N1: at every N1 >= 256 the N2 column
streams would otherwise sit a multiple of 4 KB apart and land in one L1 set (12-way), and
with 8 or 16 streams the turned stores and the back-turn thrashed it. Before the skew the
route's margin shrank from 2.3x at N2 = 2 to nothing at 16; with it, on the same plans:

| cell | best chain route | turn, pitch N1 | turn, pitch N1 + 8 | MKL |
|---|---|---|---|---|
| 2048x16 | 84 us | 87 us | 60 us | 79 us |
| 4096x16 | 178 us | 172 us | 152 us | 163 us |
| 8192x16 | 364 us | 403 us | 334 us | 396 us |
| 256x16 | 7.7 us | 8.6 us | 6.2 us | 7.5 us |
| 8192x8 | 194 us | 148 us | 136 us | 159 us |

The door picks it at every N1 >= 256 with N2 <= 16 and drops it where the plane is small.
At N2 = 8 the turned stores and the back-turn are still a third of its time; the 1D
plans are the rest.

## Arm 4: the skewed column pass (`csk=1`)

**Solves:** the column stage's stores on wide planes. A single column stage of radix R
writes R output streams N2 x 16 bytes apart; at N2 = 1024 and up that is a multiple of
4 KB, so 16 or 32 streams land in one 12-way L1 set and thrash. Measured on the kernel
alone at 4096 lanes:

| radix | user's pitch in and out | output skewed only | input skewed only | both skewed |
|---|---|---|---|---|
| 8 | 0.428 ns/pt | 0.369 | 0.417 | 0.341 |
| 16 | 1.115 | 0.453 | 1.103 | 0.419 |
| 32 | 1.305 | 0.935 | 1.223 | 0.740 |

It is the stores, not the loads, so the fix needs no copy: the route's own single column
stage (the n1c pair at radix N1, natural order by construction) writes a private scratch
at pitch N2 + 8, and the row pass reads the scratch and writes the plane. The batched row
kernels take separate input and output pitches; the per-row route runs the out-of-place
K=1 plan at N2 as the row engine that reads the scratch. Offered wherever N1 is a mono
radix, as its own route beside the banked chain, crossed with the row routes.

| cell | best chain route | skewed pass | MKL |
|---|---|---|---|
| 16x4096 | 133.6 us | 107.3 us | 133 us |
| 16x1024 | 27.7 us | 20.6 us | 31 us |
| 16x256 | 8.0 us | 5.0 us | 7.0 us |
| 32x64 | 2.85 us | 2.50 us | 1.88 us |
| 32x4096 | 266 us | 293 us in the race, 255 in a tight loop | 322 us |

The last row is the open caveat of the race protocol: the race rotates many arms, so at
large planes each arm starts on a cold L2, and a route whose scratch pushes the working
set past L2 pays for that where the bench's tight loop would not.

**The race now runs the cell's own placement, on aligned planes.** It executed every arm
in place while the out-of-place cell is served out of place, and its planes came from
`malloc` (16-byte aligned) while every plane the door serves is 64-byte aligned; under
both, every arm read about 1.3x slower than the same plan in the probe and a 64x32
verdict was a lottery. With a second buffer for out-of-place cells and aligned planes,
the race agrees with the probe and the bench.

## Arm 5: the turned prime column pass (`tpc=1`)

**Solves:** a prime N1 above 47, where no direct column kernel exists and the column
pass was the column Bluestein: a chirp multiply, two length-M column chains over every
lane (M the next power of two at or above 2·N1 - 1) and the pointwise kernel. Against
MKL it read 0.51x at 131x4096 and 0.69x at 257x256, and the rank-3 tier's first axis is
this very pass over N2·N3 lanes, so 131x64x64 read 0.54x and 257x16x16 0.70x. The 1D prime
route at the same lengths wins: 131 at 2.69x, 257 at 3.14x, 509 at 1.18x.

The turned pass runs the axis through that route. The plane's lanes are transposed into
a scratch of N2 rows at pitch N1 + 8 (2x2 complex blocks through one 128-bit lane
permute, the rows blocked so a block's source lines stay in L1), the in-place natural
K=1 plan at N1 runs on every row of the scratch, and the rows are transposed back into
the plane. The row pass follows as on any route. The door races it against the
Bluestein at create, min of three alternated on scratch planes, and banks `tpc=1` on the
column row beside `blu=M`; the loser's pieces are freed. The rank-3 tier reaches it
through the same column descriptor, serially (one plan). Timed through the bench on the
sweep's cells:

| cell | column Bluestein | turned pass |
|---|---|---|
| 131x4096 | 0.51x | 1.49x |
| 257x256 | 0.69x | 1.83x |
| 509x1024 | 0.87x | 0.96x |
| 131x64x64 (3D) | 0.54x | 1.30x |
| 257x16x16 (3D) | 0.70x | 1.81x |
| 97x97x97 (3D) | 1.23x | 1.90x |

At 509 and 1021 the 1D route's own margin is thin (1.18x, 1.14x) and the turned pass
lands at parity; the lever there is the 1D prime route itself.

## Why five arms and not one method

The winner flips with the shape, and the door has logged every arm on the same cells:

| row length N2 | winner | margin over the next |
|---|---|---|
| 2, 4, 8 at N1 <= 64 | batched mono | 1.3 to 4x over the plain route |
| 2, 4, 8 at N1 >= 256 | turn | 1.25 to 2.3x over the batched mono |
| 16 | batched mono and two-pass tie | |
| 32, 64 | two-pass batched | 1.03 to 1.30x over the plain route |
| lengths with no mono kernel and no twin | plain route | the only arm |
| 16 rows and long rows | skewed column pass | 1.25 to 1.6x over the chain |
| a prime N1 above 47 | turned prime column pass | 1.5 to 2.9x over the column Bluestein |

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
| + the turn route | 1.27 | 20 | 2 |
| + the turn route's skewed pitch | 1.34 | 10 | 3 |
| + the skewed column pass, the race out of place on aligned planes | 1.41 | 11 | 2 |

Cell by cell, plain route to the last grid: 8x8 0.39 to 1.06, 32x32 0.79 to 0.98, 64x64
1.21 to 1.44, 1024x32 0.86 to 1.46, 16x4096 1.01 to 1.31, 4096x16 0.72 to 1.04, 8192x64
0.91 to 1.14, 8192x2 0.84 to 1.97, 2048x8 0.65 to 1.28. Every plane-size band sits at or
above a 1.22 median with 82 to 98 percent of its cells at or above parity. The skewed
column pass serves 52 cells, the turn route 27, the batched rows 39. Eleven cells remain
below parity, nine of them between 0.90 and 0.99; the two below 0.8 are 128x16 (0.76, the
turn route at its worst) and 32x64 (0.79). Run-to-run: the guarded grids' control cell
reads 1.08..1.34, 1.23..1.41, 1.25..1.44, and the last two medians (1.45 with malloc'd race
planes, 1.41 aligned) are within that swing; the per-cell wins of each arm were measured on
the same plans, above, and do not depend on it.

## What remains

- 128x16 at 0.78: the turn route is its best arm and still loses; its 2 KB stride never
  aliased, so the cost is elsewhere. Its own probe.
- The race's cold-rotation bias at large planes (above), which hides the skewed pass at
  32x4096 and 8x4096.
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
