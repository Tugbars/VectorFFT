# The tall short-N2 planes at T > 1: a rows-first walk — roadmap item

**Date:** 2026-09-25 · **Status:** OPEN (owner: "we'll stop here and add the tall cell 2D
solution to the roadmap") · **Evidence:** `gauntlet/results/mt8_still5_2026-09-25/`
(the last losers-only pass at T=8), the race logs and bench cells of the same night.

## 1. The finding

After the 2D threaded design of 2026-09-24/25 (`docs/design/il2d_c2c_mt.md`), three
pow2 cells of the T=8 grid still lose to the comparator beyond noise, all tall planes with a
short second axis:

```
 cell        plane    ours (us)   comparator (us)   ratio   the threaded arm the race picked
 8192x128    16 MB    870-1050       663-772         0.65-0.85   strips, 16 columns
 4096x256    16 MB    745-770        712             0.96-1.04   strips, 16 columns
 8192x256    32 MB    2690-3340      2300-2980       0.89-0.93   strips, 8 columns
```

The row phase is fine (194-221 us at 8192x128, the serial rate over 8 workers). The column
phase alone (690-815 us) equals the comparator's whole transform, and no threaded partition
of the column pass changes that: the dense strips at 8 or 16 columns, the block arm, and the
tile arm (stage 0 across the plane, one worker per first-stage sub-problem with the rows fused
into the leaf) all land between 1.1 and 1.4 ms in the race.

## 2. The cause: a traffic count, not an exchange

At 8192x128 the column-first walk moves the plane about five half-sweeps of 16 MB:

```
 read x for the column gather                          16 MB
 scatter the column output into y                      16 MB   + 16 MB read for ownership
 read y for the rows, write y                          32 MB
                                                       ------
                                                       80 MB   (+ a 2 MB strip that spills L2)
```

The comparator's 663 us at the machine's aggregate bandwidth (about 97 GB/s across the
eight cores) is 64 MB: the count of a **rows-first** walk whose column pass runs in place in
the output while it is hot. (The count comes from the order of the passes alone: the streamed-store experiments of
2026-09-25 were refuted, and the order is what remains.)

Measured and refuted as fixes on their own (each through the pinned bench, interleaved
twice): a write-prefetch of the leaf's destination pieces (within noise); a streamed strip
scatter with the row phase still after it (slower: the rows then read y from DRAM); the tile
partition (96 MB: two ownership reads). The tile arm stays raced since it ties or wins by 1%
at planes up to 4 MB.

## 3. The form

A second walk ORDER for the natural out-of-place class at T > 1, raced like every other arm:

1. **Rows first, streamed.** Row slabs across the workers (the clones exist): each row of x
   is transformed into a per-worker row buffer and leaves as one sequential streaming store
   into y (`_il2d_row_stream(nt = 1)`, the staged leaf's helper, already writes a finished
   row that way). y is then in memory, owned by nobody's L2.
2. **Columns in place on y, streamed.** The dense strips (`_il2d_col_pass_nat_strip`) gather
   from y, run the column chain in the worker's L2 block, and land the strip in natural row
   order in a second dense buffer, from which a sequential copy-out streams every row's
   piece back into y (the copy-out of 2026-09-25's second experiment; with no row phase
   after it, its DRAM penalty disappears). Widths from the same ladder (8..256), the same
   dense scratch, one more block per worker.

Expected count at 8192x128: read x 16 + stream y 16 + read y 16 + stream y 16 = 64 MB, the
comparator's; expected gain about 1.2x at the three cells, with the tall in-place class
(`place=ip`) following the same order (the row pass in place, no first copy).

## 4. What exists, what is new

| piece | state |
|---|---|
| per-worker row clones, row slabs, the engagement counter | exist (`_il2d_c2c_mt_phase` mode 2) |
| the dense per-worker strip scratch and the strip pass | exist (`natsscr`, `_il2d_col_pass_nat_strip`) |
| a streaming row store | exists (`_il2d_row_stream`, il2d_cols.h) |
| the streamed strip copy-out through a second dense buffer | written and measured on 2026-09-25, removed (kept in the session record) |
| the rows-first order in `_il2d_c2c_mt` and its race arm (`mtarm=3`, a token beside `msw`) | NEW |
| the backward direction (columns first in place, then rows streamed into the destination) | NEW, the mirror |
| the row buffer per worker for the streamed row pass (route 0 executes the row plan out of place into it) | NEW |

The arm joins the threading race for natural cells whose plane exceeds L2 x T (the planes
where the count matters); the race decides everywhere else. Gate: bitwise the serial plan
(a loop restriction plus a copy), the naive-DFT reference, and the tall cells at T=8 against
the comparator through the gauntlet's cell protocol.
