# The scrambled ZTURN-T class — the plain schedule, in place

Design, 2026-09-13. Measured basis: `probes/ZT/zt_stage_profile_results.md`
(the shipped driver's time by stage) and `probes/ZT/zt_order_cost_results.md`
(the scatter bound, the plane-vs-dest cost). This supersedes the 09-12 draft:
its comb terminator was not in place, and the zero-kernel backward that
depended on it is withdrawn.

The owner's framing is the design: **scrambled is the plain algorithm and
natural is the modified one.** Nothing is added to make the output scrambled.
Two things natural order needs are removed, and one of them is a whole pass
over the array.

## Terms

- **kind** — one generated kernel body for one stage shape (`t0tp`, `tmg`,
  `tlf` today). A kind is a template over radix and direction.
- **driver** — the one generated function per (cell, chain, direction) that
  inlines the kinds with literal trip counts and no calls inside
  (`generated/ztt_drivers_avx2.c`); `vfft_ztt_execute_*` calls it.
- **sweep** — a stage whose groups span the whole array. It streams all of N
  through the core from wherever N lives (L1, L2 or DRAM).
- **block / tile** — a contiguous span of the array that holds whole groups
  of several consecutive stages, so those stages run back to back while the
  span is L1-hot. `tile` is the raced run-time width (16 KB or 32 KB).

## What natural order costs in ZTURN-T

Both costs sit at the ends of the pipeline. The mids do not know the order.

1. **The ingest scatter.** Column c's run is parked at `rb[c]` so that later
   stages merge adjacent runs. Measured against the same kernels with linear
   placement: 0 below 4096, 4-8% at 8192..131072, 22% at 262144.
2. **A second sweep.** The natural terminator's legs are N/R apart (that is
   what natural output means: X[k] and X[k+N/R] leave the same butterfly), so
   it sweeps the whole plane after the tiles are done. It is 17-28% of the
   runtime at every tiled cell. A stage that runs inside the tile costs
   7-15%.

Shipped driver, by stage, share of runtime (2026-09-13, core 2, bitwise-gated):

| N | chain, tile | ingest | scatter | tiled mids | mid sweeps | terminator sweep |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 8.8.8.4, none | 34% | 0 | (untiled) | 22 + 21% | 23% |
| 4096 | 8.8.8.8, 1024 | 32% | -1.3% | 40% (2) | — | 28% |
| 8192 | 8.8.4.4.8, 2048 | 32% | -4.3% | 43% (3) | — | 25% |
| 16384 | 8.8.4.8.8, 2048 | 32% | -6.0% | 44% (3) | — | 24% |
| 32768 | 8.8.8.8.8, 2048 | 31% | -7.3% | 29% (2) | 18% | 23% |
| 65536 | 8.8.4.4.8.8, 2048 | 40% | -6.1% | 28% (3) | 14% | 19% |
| 131072 | 8.8.4.8.8.8, 2048 | 34% | -8.0% | 28% (3) | 14% | 22% |
| 262144 | 4.8.4.8.4.8.8, 2048 | 43% | -22% | 21% (3) | 9 + 10% | 17% |

**One sweep is the floor.** The first stage of any natural-input transform
touches elements N/R0 apart; the last stage of any natural-output transform
writes elements N/R apart. Natural in, natural out has at least two sweeps.
Natural in, scrambled out has at least one. The schedule below reaches that
floor, so within this kind family nothing beats it on traffic shape; what
remains is kernel quality.

## The plain schedule

Sande-Tukey, in place, row-major: contiguous in, butterfly, twiddle applied
after, contiguous out, the output in whatever digit order falls out.

Block ladder, the dual of natural's growing run length `L[]`:

    Len_0 = N,   Len_{s+1} = Len_s / R_s,   so  Len_s = prod_{u >= s} R_u

Stage s in the shipped 11-parameter kind ABI:

    Ls = Len_{s+1}   Gs = N / Len_s   count = Len_{s+1}

Read the `R_s` legs of column b at `base + q*Len_{s+1} + b`, butterfly,
multiply output leg p by `W_{Len_s}^{p*b}`, store back to the slots read.
Every stage is in place; stage 0 reads `zin` and writes `zout` at the same
offsets, so `zin == zout` is the same driver.

- **Stage 0** is the only layout-changing stage: interleaved loads of the
  caller's input at leg stride `Len_1 = N/R0` — the addresses `t0tp` loads
  today — de-interleave, butterfly, post-twiddle, block-split leg-major
  stores at `2*(p*Ls + k)`. R0 sequential read streams, R0 sequential write
  streams. No table, no scatter. It is the one sweep.
- **Stages 1..nf-2** are `tmg`'s combine with the twiddle after the butterfly.
- **Stage nf-1** has `Len_{nf-1} = R`, so `b = 0`: twiddle-free, legs
  adjacent, and it stores the caller's interleaved format IN PLACE. Because it
  is in place there is no plane, no buffer-mode axis, and no output-prefetch
  twin. Its groups are R complexes wide, so it always runs inside the tile.

## Tiling: the mirror of the natural loop

The natural driver runs the mids with `R*L <= tile` per tile (a prefix of the
stages, `R*L` grows with s) and sweeps the rest. The plain driver runs the
stages with `Len_s <= tile` per block (a suffix, `Len_s` shrinks with s) and
sweeps the rest; the terminator is always in the suffix. A block of `tile`
complexes holds whole groups of every suffix stage because the blocks nest.

Legal tile: a power of two, at least `R_{nf-2} * R_{nf-1}` (below it no stage
tiles), below N; 0 = untiled. Every width is bitwise the untiled result, as
today: the tile changes group order only.

Sweep count per cell: natural = 1 (ingest) + the mids with `R*L > tile` + 1
(terminator); plain = 1 (stage 0) + the stages with `Len_s > tile`. On the
reversed chain the two middle terms coincide, so plain is always exactly one
sweep fewer.

## The twiddle stream is group-invariant and the same size

    e(stage s, output leg p, column b) = (p*b) mod Len_s,   root W_{Len_s}

The group index does not appear: after stage 0 the array is `N/Len_s`
independent contiguous sub-transforms and the block index enters only the
base address. Stage s carries `2*(R_s - 1)*Len_{s+1}` doubles as `(R_s - 1)`
records of `[c x4][s x4]` per column quad, `tlf`'s stream shape; stage nf-1
carries none. The totals equal natural's on the reversed chain (1008 doubles
at 512 chain 8.8.8, 32752 at 16384 chain 8.8.4.8.8). `_ztt_fill_stage` fills a
plain stage with `(R_s, Len_{s+1}, Len_s)` where natural fills it with
`(R_s, L_s, R_s*L_s)`.

## Kinds

Forward, three per leaf radix (4 and 8):

| kind | stage | load edge | twiddle | store edge |
| --- | --- | --- | --- | --- |
| `t0d` (new) | 0 | `E_z "Ls"`: interleaved legs at stride Ls, de-interleave | post, `%TWF*k` | `E_planes "Ls"`: block-split leg-major |
| `tmgd` | 1..nf-2 | `tmg`'s | post (`dif = true` at Fwd, as `msd` is to `ms`) | `tmg`'s, in place |
| `tld` (new) | nf-1 | adjacent legs, block-split: TR4 turns column lanes into group lanes | none | interleaved, in place |

Backward, one new kind; the rest is shipped:

| kind | stage | load edge | twiddle | store edge |
| --- | --- | --- | --- | --- |
| `tldb` (new) | inverse of nf-1 | adjacent legs, interleaved | none | block-split, TR4 inverse, in place |
| `tmgb` (shipped) | inverse of s | `tmg`'s | pre, conjugate | in place |
| `tlfb` (shipped) | inverse of 0 | block-split legs at stride Ls | pre, conjugate | `E_z "OLs"`: natural interleaved |

`t0d` is `tlf` mirrored (edges swapped, twiddle side swapped). `tld` and
`tldb` are one template in two directions; their TR4 is the generator's
`E_blocks` transpose. Nothing here touches `rb`, and no kind has a plane
argument.

**Two kinds of generated code, both kept (owner's ruling 2026-09-14).** The
stage kernels above are the product: one stage each, composable, the same
files that a 2^a·odd chain, the 2D/3D pow2 passes and both order classes'
backwards are built from. The **fused codelets** in
`generator/generated/fused_codelets/` (README there) are one whole-transform
function per pow2 cell with those bodies inlined and literal trip counts —
the pow2 solution's executable form, and only that: nothing else may bind
one, and none can be recombined. Measured, the fusion is worth 4-12% below
2048 and 0-3% above; its cost is 7 MB of generated text, split per family
and size so a kernel change rebuilds in about a minute.

Shuffle accounting, measured: `t0d`'s de-interleave is one shuffle per
complex where `t0tp`'s turn lattice does transpose and de-interleave together
in half of one; `tld` carries a TR4 plus the interleaving stores, 1.5 per
complex, where `tlf` carries 0.75. At 16384 `tld` costs 5.5 us against
`tlf`'s 4.4 (1.9 cycles per complex against a tiled mid's 0.94): the
transpose network, not memory, bounds the plain last stage. Below the tile
band the class is ~11% slower than natural in both modes for this reason.

## The output order

Position i, written in the chain's mixed radix as
`i = d_0*Len_1 + d_1*Len_2 + ... + d_{nf-1}`, holds frequency
`k = d_{nf-1}*(R_0...R_{nf-2}) + ... + d_1*R_0 + d_0`, the digit reversal.
Inside each block of four R-groups the terminator
stores whichever lane order costs the fewest shuffles (the unpack-only store
puts groups k and k+2 in one register); that order is fixed at emit time and
is part of the permutation. Scrambled is any fixed bijection, so this is
free.

The plan tabulates the permutation once at create, for the gate and for
introspection. Nothing reads it at run time. Contract consequences, to be
written into `include/vfft.h` when the class ships:

- the order is a property of the plan (chain and terminator), so the
  backward must come from the same plan or its matched inverse;
- two scrambled spectra may be multiplied pointwise only if they came from
  the same plan.

## Worked cells

**N = 512, chain 8.8.8, untiled** — every stage is an L1 pass; nothing to win,
correctness only.

| stage | natural (shipped) | plain |
| --- | --- | --- |
| 0 | `t0tp8` (Ls 64, count 64) + rb, twiddle-free | `t0d8` (Ls 64, Gs 1, count 64), post, 896 doubles at +0 |
| 1 | `tmg8` (8, 8, 8), tw +0, 112 doubles | `tmgd8` (8, 8, 8), tw +896, 112 doubles |
| 2 | `tlf8` (64, 1, 64), tw +112, 896 doubles | `tld8`, 64 groups of 8, twiddle-free |

**N = 16384, chain 8.8.4.8.8, tile 2048** — the spike cell. `Len` = 16384,
2048, 256, 64, 8, 1.

| stage | kind | Ls | Gs | count | twiddle (doubles, offset) | placement |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `t0d8` | 2048 | 1 | 2048 | 28672 at +0 | sweep |
| 1 | `tmgd8` | 256 | 8 | 256 | 3584 at +28672 | per block, 1 group |
| 2 | `tmgd4` | 64 | 64 | 64 | 384 at +32256 | per block, 8 groups |
| 3 | `tmgd8` | 8 | 256 | 8 | 112 at +32640 | per block, 32 groups |
| 4 | `tld8` | — | 2048 | — | none | per block, 256 groups |

The natural driver at this cell is `t0tp8` sweep, per tile `tmg8` x32,
`tmg4` x8, `tmg8` x1, then the `tlf8` sweep. Same tile work, one sweep fewer.

## What it buys, measured (2026-09-14, `probes/ZT/zt_scr_spike_results.md`)

Plain against the banked natural, same tile, percent of the natural:

| N | OOP fwd | OOP bwd | in place fwd | in place bwd |
| --- | --- | --- | --- | --- |
| 512 | +11 | +12 | +10 | +11 |
| 2048 | +25..+30 | +5..+8 | -3..-5 | -3..-6 |
| 4096 | +6..+17 | +5..+22 | -3..-5 | -5..-6 |
| 8192 | +16..+17 | +2..+16 | -2..-5 | -3..-5 |
| 16384 | +13..+18 | -2..+1 | -5..-9 | -7..-10 |
| 32768 | +8..+17 | -2..-3 | -11 | -15 |
| 65536 | +7..+12 | -2..+1 | -17 | -26 |
| 131072 | -2..-6 | -6..-14 | -28..-30 | -21..-30 |
| 262144 | -7..-10 | -16..-17 | -23..-24 | -18..-20 |

**In place the class wins at every cell from 2048 up, both directions**,
because every plain stage is in place and the natural pays its plane path
there. **Out of place the backward is at parity from 16384 up and wins above
L2; the forward loses at L2-resident sizes and wins above L2.** Below 2048
the class loses ~11% in both modes and is served regardless (ruling 1).

Why the earlier projection (-15% at 16384) was wrong: the natural
terminator's sweep is mostly its own work at L2 sizes, not traffic, so
removing it saves little, while the plain last stage's 4x4 lane transpose
plus interleaving stores (1.5 shuffles per complex against `tlf`'s 0.75)
cost 25% more than the terminator they replace. The transpose is the price
of an in-place stage 0 — row-major is the only layout in which stage 0 reads
and writes the same offsets, and it forces adjacent legs at the last stage.
The sweep is real traffic only above L2, and there the class wins in every
mode. The out-of-place forward additionally pays its destination's
write-allocate inside its one memory-bound sweep (stage 0: +15..27% over the
same stage in place), which the backward pays inside an L1-resident block
stage; that asymmetry is the next optimization target (build order 1b).

## Gates

1. **Exact order.** `out[i] == nat[perm[i]]` against the natural ZTURN-T
   result on the same input, to a tolerance of a few ulps times log N (the
   twiddle side differs, so it is not bitwise). Never a magnitude multiset.
2. **Matched roundtrip.** `bwd(fwd(x)) == N*x`, out of place and with
   `zin == zout`, at every chain in the registry and every legal tile.
3. **Tile invariance.** Every legal width bitwise the untiled result, both
   directions.
4. **Cross-alignment invariance.** Bitwise across destination offsets 0, 16,
   32, 48 bytes.
5. **Speed.** At the spike cell the plain forward must beat the natural
   forward on core 2 with the house protocol, by an amount in the
   projection's order. If it does not, the design has not collected the
   prize and the kernels are examined; the contract is not in question.

## Build order

0. **Spike — DONE 2026-09-13/14.** `t0d`, `tmgd`, `tld`, `tldb` for radices
   4 and 8 (corpus rows, corpus law 103/103); both plain drivers per cell in
   `ztt_drivers.ml` (446 drivers, registry fields `fwd_scr`/`bwd_scr`);
   `probes/ZT/zt_scr_spike.c` — gates 1-4 pass at every cell 512..262144,
   both chains tried, both modes; the races are the table above.
0b. **Split the driver TU — DONE 2026-09-14.** One file per (family, N):
   `ztt_drivers_avx2_<N>.c` and `zttp_drivers_avx2_<N>.c`, 30 files; a
   plain-family change rebuilds in 84 s against 47 minutes.
1. **Backward — DONE with the spike.** `tldb` + the shipped `tmgb`/`tlfb`;
   gate 2 passes everywhere; raced (the table).
1b. **The out-of-place forward — BOTH LEVERS REFUTED 2026-09-14.** The
   `tlfi`-style output prefetch on `t0d`: no gain (removed). The packed
   per-column w^1 stream with the squaring tree, `t0dq` (`TP_PowW1`, 224 KB
   to 32 KB at 16384, raced twin `zttpq_*`, registry `fwd_scrq`): correct
   and SLOWER at every L2-resident cell (+12..+48% against `t0d`'s
   +5..+22%), a 3-6% gain only above L2. The stage is bound by neither
   store-miss latency nor its twiddle bytes; ~0.85 us at 16384 is the second
   buffer's write-allocate and ~1.1 us is a fused-driver-only excess with a
   single copy of the loop and the same instruction mix, unexplained. The
   class stands as measured: in place from 2048 up and out of place above
   L2 it wins; out of place at L2-resident sizes its forward loses 5-22%.
   `t0dq` was not dominant and was DELETED 2026-09-14 (owner's ruling); the
   results doc keeps its table so it is not rebuilt.
2. **Create — DONE 2026-09-14.** One plan type, one order class per plan:
   `vfft_ztt_create_chain_ord(N, chain, nf, scr)` (`vfft_ztt_create_chain`
   = the natural wrapper). `scr` builds the plain streams (the natural fill
   with `(Len_{s+1}, R_s, Len_s)`), the conjugate stream, the plain tile law
   (`vfft_ztt_tile_legal_ord`: the last mid's block), the permutation table
   (`vfft_ztt_perm`, gates and introspection only), no `rb`, no plane;
   `vfft_ztt_bind` is a no-op for it and `vfft_ztt_execute_*` dispatches to
   the cell's `fwd_scr` / `bwd_scr` codelets. The fingerprint compares
   order class, tile and placement beside the chain.
3. **Planner, doors, wisdom — DONE 2026-09-14.** `_il_dp_enumerate_ztt_ord`
   pushes plain candidates (`il_scr = 1`) with the plain tile law; the
   scrambled pool at a pow2 band cell is that and nothing else (the
   sub-2048 natural-writer leftover and the cascade chains are gone there);
   the bench binds through `_ord`; `_il_dp_bin_of` carries the plain
   permutation's independent re-derivation; the commit's replay and the OOP
   door's create pass the request's order, the prime inner takes the plain
   plan from its `ord=scr` row; the doors build no cascade at pow2 in any
   order class. Verified cold through the front door: `k1_pow2_gate` and
   `ztt_gate` (the scrambled create at every pow2 N replays the plain engine
   bitwise; one race banks both classes).
4. **Tree gates — DONE 2026-09-14.** `ztt_gate` gained the plain pass over
   every registry cell (permutation, exact order vs natural, in place
   bitwise, roundtrip, tiles bitwise) and the scrambled front-door
   assertion; `il_dp_overflow_gate` restamped (7/18/36/48/63/84/108).
5. **Contract text — DONE 2026-09-14.** `include/vfft.h`'s order paragraph
   names the class and the same-plan rule for combining spectra.
6. **The cascade leaves pow2 — ROLE DONE, CODE PENDING.** No door builds or
   races it at a pow2 band cell in any order class, the planner enumerates
   none of its chains there, and its 24 pow2 rows are purged from the shipped
   store. Its code (`zturn.h`, `zsplit.h`, the `_k1z_*` paths, the
   `stf*/s0t*/ms*` kernels) is shared with the 2^a·odd cells it still serves
   and is deleted with the 2^a·odd ZTURN-T, per the owner's earlier ruling.
7. **Calibration — owner's word.** The shipped store has no `ord=scr`
   ZTURN-T rows yet; a cold scrambled pow2 request races at create (wisdom
   or race). Banking the band on core 2 with pacing is the calibrator's run.

## Rulings (owner, 2026-09-13)

1. **The scrambled pool admits scrambled writers only, at every cell.**
   Natural order and scrambled order are contracts, not optimization angles.
   A scrambled request is served the plain engine below 4096 even where it
   ties or loses slightly. The leftover admission of natural writers into the
   scrambled pool at pow2 below 2048 (`_il_dp_enumerate_natural_engines`)
   ends when the plain engine becomes that pool's writer (step 3).
2. **Wisdom.** The key's order class carries the row: `ord=nat` for
   interleaved natural, `ord=scr` for interleaved scrambled (the
   `wisdom2_scr` shard), with `il_route=ztt il_ztt=<chain> il_tw=<tile>`
   unchanged under either. No new token.
3. **No permutation API.** `include/vfft.h` already declares that a scrambled
   plan produces its own self-consistent permutation and promises no
   particular one; the matched backward is the whole promise.
4. **Kind names** follow the generator's convention: `t0d`, `tmgd`, `tld`,
   `tldb`.
