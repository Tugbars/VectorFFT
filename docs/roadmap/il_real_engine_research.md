# The IL real engine: the research record

The standing record of the interleaved real-transform work: what the engine is
to be, which methods are on the table, and each method's current measured
verdict. A verdict is overwritten when re-measured; the runs it comes from are
in `gauntlet/results/`. The design input is `fftw_real_study.md` beside this
file. This is a declaration of the current state, not a journal.

## 1. The primitive

One batched real-rows-to-CCE-rows transform and its c2r mirror:

- input: K contiguous real rows of N reals at pitch P (out of place P = N; in
  place the padded 2·(N/2+1)); output: K CCE rows of N/2+1 pairs at pitch Q;
  one call for the whole batch; natural order (a real spectrum has no other);
- 1D K>1 is rows = K (transform-contiguous, the default geometry for
  interleaved real once the primitive lands; lane-major stays with split);
  2D is rows = N1 followed by the il2d complex column pass over the N2/2+1
  columns; 3D is rows = N1·N2 followed by the il3d column passes;
- a c2r plan may destroy its input where that buys speed; the plan declares it.

The complex column machinery exists; this record is about the rows and about
how the fold is placed.

## 2. The baseline

Today's library through the front door: even N = the real door's engine race
(zr2c: a c2c(N/2) child and a separate fold pass; the real pair; and, since
2026-09-29, ZTT-r: every {4,8} chain at every tile width swept at create, the
four fastest and the winner's stack states raced against zr2c and the pairs,
the winner banked as `eng=zttr chain= tile= stk=`, in place through a plan
scratch with the ZTT's prefetching in-place terminator), odd N = the
promote-to-complex bridge, K>1 = the split real engines behind the z-doors,
2D = per-row zr2c or ROWSPLIT rows then the il2d columns. The baseline
below predates ZTT-r in the door. Measured by the gauntlet's real contract (`--real r2c|c2r`,
`--k K`, `--cmp fftw|mkl`; the bench cell `--realfwd/--realbwd/--2drealnat`)
against FFTW 3.3.10 (MEASURE, out of place) and MKL on the same banked plans.
Cell sets: 1D K=1 (196 N: pow2, 2^a·odd, smooth even, primes, odd composites;
2..4096 and 8192..65536, the larger N waiting on their in-place child
verdicts), 1D K=8 and K=32 rows (12 N), 2D (67 shapes). Runs:
`real1d_base_2026-09-29`, `realk_base_2026-09-29`, `real2d_base_2026-09-29`.

Verdict (2026-09-29): at parity or behind FFTW wherever FFTW's SIMD real path
runs, ahead only where FFTW is scalar; ahead of MKL nearly everywhere.
Speedups are ours over the comparator, r2c / c2r:

| Cells | vs FFTW | vs MKL |
|---|---|---|
| 1D K=1, 196 N | median 1.04 (85 below parity) / 0.99 (100 below) | 1.19 (43 below) / 1.13 (47 below) |
| 1D K=1 by class | 2^a·odd 0.92 / 0.87 (34 and 41 of 42 below); even composite 1.00 / 0.94; pow2 1.00 / 0.96; odd composite 1.26 / 1.22; prime 2.13 / 2.07; N ≤ 15 at 0.31–0.58 | odd composite 0.93 / 0.88 is the one class behind |
| rows K=8, 12 N | 0.97 (10 below) / 0.90 (11 below); N=16 rows 0.38 / 0.36 | 1.08 / 0.99 |
| rows K=32 | 0.93 (8 below) / 0.86 (10 below) | 1.10 / 0.91 |
| 2D, 67 shapes | 0.96 (38 below) / 0.89 (48 below); tall pow2 0.85 / 0.82, wide pow2 0.95 / 0.82, the 16-column shapes 0.45–0.60, wide mixed 2.34 / 2.24 | 1.05 (29 below) / 1.19 (13 below) |

The losses are the primitive's cells: the even N whose fold is a separate
pass (2^a·odd and pow2 at parity or below), the tiny N and the 16-column
shapes where the child-plus-fold overhead and the promote bridge dominate,
and the batched rows served by the split engines. The FFTW arm of these runs
planned per process (its 1D c2r control spread 0.75–0.98 is that variance);
later runs replay one wisdom file per run.

With ZTT-r in the door (method 2, 2026-09-29/30; the same 196 cells, the
1D real records stripped from a store copy so every cell raced its engines
with the c2c children replayed, runs `zttr_full_r2c_2026-09-29b` and
`zttr_full_c2r_2026-09-29b`, controls 1.14–1.16 and 1.08–1.11), r2c / c2r
medians vs FFTW, before → after, with the count of cells below parity:

| Cells | r2c | c2r |
|---|---|---|
| 1D K=1, 196 N | 1.04 (85 below) → 1.11 (56) | 0.99 (100 below) → 1.08 (77) |
| pow2 (13) | 1.00 (7) → 1.06 (2) | 0.96 (7) → 1.08 (4) |
| 2^a·odd (42) | 0.92 (34) → 1.00 (20) | 0.87 (41) → 0.96 (26) |
| even composite (64) | 1.00 (32) → 1.02 (26) | 0.94 (37) → 0.99 (32) |
| odd composite (37) | 1.26 → 1.34 (no engine change: the session's drift) | 1.22 → 1.24 |
| prime (40) | 2.13 → 2.11 | 2.07 → 2.13 |

The door banked ZTT-r at 24 r2c cells (2^a·odd 320..3840, 16384, 65536)
and 28 c2r cells (2^a·odd 448..3840, pow2 512..65536); the 2^a·odd cells
from 1792 up moved +20..43% (r2c) and +40..59% (c2r) against FFTW. Below
2^7·odd ZTT-r does not apply (the ingest's runs and the terminator's run
length), and the batched rows and the 2D shapes are untouched.

## 3. The methods

Each method is a create-time switch, then a raced arm, so the races see
combinations: a method that loses alone may win inside one. The verdict line
names the run that decided it.

| # | Method | Mechanism | Verdict |
|---|---|---|---|
| 1A | Fold in the last stage, 1D (form A) | the stock `n1t` leaf over the packed view, then `t2h` (real_il.ml): the untangle, the twiddles, the radix-R1 butterfly and the mirror stores in one pass; no fold pass. The pair engine `zrp` (il/real/zrp.h) beside zr2c in the real door, raced and banked per cell (`eng=zrp pair=R1.R2`) | **LOSES at every pair cell** (2026-09-29, the door race): 256: 122 vs zr2c 100 ns; 1024: 555 vs 420; 4096: 3892 vs 2092. The stage probe (gauntlet/zrp_time_probe.c) says why: at L1-resident N the fold PASS costs nothing, its ARITHMETIC does — the fold is 123 ns at 1024, and fused into `t2h(16)` the same untangle costs +29 ns per 8 columns against the fold's 31; the lone column and the DC/Nyquist pass add a block; the monolithic radix-32/64 tops spill (`t2h(32)` 388 vs the blocked `t2b48` 204). A blocked top would reach a wash, not a win. Dead below the ZTT band; the arm stays in the door (it never wins the race). |
| 1B | Real leaf + mirror mids, 1D (form B) | `r2z` (real_il.ml, cx_real.ml): four real columns per vector, the real R2-point DFT with real arithmetic (the radix-2 recursion on real samples, the direct odd form), each column's half spectrum packed ((DC, Nyquist) in one slot) into half of an R2-wide row; `t2m`: the radix-R1 combine of the R1 half spectra with direct loads and mirror stores, in place. No untangle anywhere. The pair engine's form B (`eng=zrp pair=R1.R2 leaf=r2z`), raced beside form A and zr2c | **built and raced 2026-09-29; wins only where zr2c's child is the mono kernel**: 128: 52 vs zr2c 69 ns (banked); 256: 109 vs 100; 512: 241 vs 214; 1024: 594 vs 428; 96: 59 vs 59; 192: 84 vs 80; c2r 256: 111 vs 106. Kernel gate 38/38 pairs at ~1e-16 both directions, both placements. The stage probe: the real leaf costs what the packed leaf costs (r2z(16) 36 vs n1t(16) 39 ns at 256 — its arithmetic is 60% less but the interleaving store edge is two shuffles per store and the kernel is not issue-bound), the top `t2m(16)` is t2(16) + 9%, and the monolithic radix-32/64 kinds spill (`r2z(64)` 304, `t2m(32)` 427). Blocked twins of both would reach a wash at 1024 (~380-410 vs 420), not a win. The untangle the packing trick pays is not where this core's time goes. |
| 2 | ZTT-r (r2c) | the ZTT at M = N/2 on the packed view, ingest and mids as they are, the fold FUSED into the terminator `tlfhc` (il/real/zttr.h, hand-written at radix 4/8): each aligned column quad of every leg against its mirror window (leg R-1-r, columns L-k..L-k-3, assembled reversed by one permute and one blend per plane from the block it straddles and the block carried from the previous window); pass 1 parks both windows' half-size DFTs (the even legs' first, then the odd legs': at most twelve vectors live), pass 2 combines pair (m, m+R/2) of the primary with pair (R/2-1-m, R-1-m) of the partner, untangles in split form (no shuffle) and stores both through the REINT edge, in the destination plane (no scratch, no fold pass). Column 0's partner is the run's own column 0 (the carry starts there); column L/2 is peeled. The parking and the carry sit in a hand-aligned region and the kernel is entered through a stack-aligning trampoline at a chosen rsp residue (`p->stk`): Win64 gives 16-B frames, mingw never realigns, and a spilling kernel otherwise runs at one of four speeds by its caller's rsp (+75% in the worst). Gate 1e-16, both placements | **ahead at 7 of 8 cells** (2026-09-29, gauntlet/zttr_race.c: every {4,8} chain x tile {0,512,1024,2048,3072} calibrated per cell, the door raced paced and alternated, `VFFT_ZRP=0`): door/zttr 1.07 at 512 (4.4.4.4), 0.96 at 1024 (4.8.4.4), 1.01 at 2048 (4.8.8.4/512), 1.01 at 4096 (8.4.4.4.4), 1.01 at 8192 (8.4.8.4.4/2048), 1.06 at 16384 (8.4.4.8.8/2048), 1.02 at 32768 (8.8.8.4.8/2048), 1.02 at 65536 (8.4.4.8.8.4/2048). The calibration prefers a radix-4 last stage (half the parking, no private spills) even at the price of a fifth stage. At a fixed chain the fused terminator costs exactly the last stage plus the fold pass (2048: 525 vs 263 + 253 ns; 8192: 1995 vs 1035 + 972): with sixteen ymm the radix-8 combine of the two mirror windows must park 32 vectors and carry 16 in place, and that parking is the fold's memory pass in another place (64 memory uops per 64 bins either way); the gain is the chain the cheaper terminator admits. Neither form is port-bound (2.6-3.0 IPC): 48 fewer shuffles per window bought nothing. Isolated out-of-place passes overstate the fold above L1 (32768: 9286 ns alone, 1986 in place); only pipelines are judged. Left on the table: the primary's half-DFTs kept in registers at R = 4, the column-major last stage -- single digits on paper. The VTune arm (docs/research/il_real_engine/vtune/) attributes the passes per function. IN THE DOOR (3% hysteresis toward zr2c) the r2c margin is mostly inside the hysteresis: the door banks ZTT-r r2c only where it clears it (2048/4096 out of place, 4096/8192 in place, 2-4%). |
| 2' | ZTT-r (c2r) | the backward fold fused into the ZTT's backward INGEST `t0h` (il/real/zttr.h, from the emitted `t0tp` bwd bodies at radix 4/8): the untangle is pointwise on the pair (X[n], X[M-n]) and sits at the ingest's load edge before any butterfly, so nothing is parked. Each iteration takes the column pair (k, k+1) of every leg and the pair (Ls-k, Ls-k-1) of the mirror leg (one 32-B load each, the mirror's lanes swapped once), untangles both, and runs four butterflies (the two columns and their two mirrors, the mirror legs in reversed order) straight into the plane at rb[c]. Column 0's partners are column 0 of the other legs and X[M], which the mirror load holds by the runs' contiguity; the mirror "column Ls" is not stored; column Ls/2 is peeled; X[0], X[M] are taken real. The backward mids and last stage follow in the output, no scratch plane, no fold pass; entered through the stack-aligning trampoline | **WINS at every cell** (2026-09-29, gauntlet/zttr_race.c --c2r: every {4,8} chain x tile calibrated per cell, the door's c2r raced paced and alternated, gate 6e-16..9e-16): door/zttr 1.04 at 512 (4.8.8), 1.04 at 1024 (4.8.4.4), 1.15 at 2048 (4.8.8.4/512), 1.24 at 4096 (4.4.4.8.4), 1.15 at 8192 (4.4.8.8.4/1024), 1.26 at 16384 (4.8.8.4.8/1024), 1.28 at 32768 (4.4.8.4.4.8/2048), 1.24 at 65536 (4.4.8.4.8.8/1024) -- every cell picks a radix-4 ingest, where the fused untangle parks nothing. The door's c2r pays the fold into a scratch plane and the child's read of it; the fused ingest costs the plain ingest plus the untangle (2048: 478 vs 203 + fold 253, a wash on the pass itself) and the scratch round trip is gone. The lesson of the two halves: fuse the fold where it is POINTWISE at a load edge, not where it pairs the outputs of two combines. THROUGH THE DOOR AGAINST FFTW (gauntlet `zttr2_c2r_2026-09-29`, pow2, the door recalibrating; control 1.088..1.096): the door banks ZTT-r c2r at 1024..65536, and the pow2 c2r cells move from the baseline's 0.89 / 0.92 / 0.96 / 1.13 / 1.13 / 1.12 / 1.09 (1024..65536) to 0.90 / 1.02 / 1.10 / 1.28 / 1.00 (flapped, flips 1.33x apart) / 1.22 / 1.28. The r2c run (`zttr2_r2c_2026-09-29`) kept zr2c at every pow2 cell; its control drifted 0.82..1.16, so it is not a comparable run. THE ODD BAND (2026-09-29, late): the sweep now offers the ZTT's odd mids (3, 5, 7, 9, 15) between the {4,8} ends, so every 2^a*odd N with 2^7 in it (the ingest's runs in whole blocks, the terminator's run length a multiple of 8) is a ZTT-r cell, run staged. First calibrations against the door (gauntlet/zttr_race.c, gates 4e-16..1e-15, quiet machine): 1536 r2c 1.22x (8.8.3.4) / c2r 1.17x (4.4.4.3.4); 3840 r2c 1.31x (8.4.5.3.4) / c2r 1.57x (4.8.3.5.4); 5120 r2c 1.04x (8.4.4.5.4) / c2r 1.16x (4.5.4.4.8). The 2^a*odd class was the baseline's largest loss (0.92 / 0.87 vs FFTW); the gauntlet's mixed group through the rebuilt door is the pending measurement. |
| 2'' | The tiny cells: the call, not the arithmetic | gauntlet/call_overhead.c, per N through the door (ns per call, paced): at N = 3 the door's r2c cost 12.5 where the c2c child it wraps costs 4.2 alone, the promote-and-copy passes 3.5, and the public execute's general signature walk the rest; FFTW's whole r2c is 4 ns there. (a) BOUND REAL DISPATCH (2026-09-30, il/il_execute.h `_vfft_real_bind_exec`, bound at `_real_finish`): a 1D K=1 interleaved real plan reaches the bridge in one indirect call like the c2c plans -- 12.5 -> 9.3 ns at N = 3, 19.9 -> 15.7 at 7, 16.1 -> 15.1 at 16, 31.3 -> 30.6 at 64. (b) THE REAL MONO ENGINE `zrm` (2026-09-30, il/real/zrm.h, codelets/zil/avx2/real/mono/): the `rn1` kind is the n1 body on real input (c2c_il.ml `--cil-rn1`, the real address forms of cx_ir.ml / cx_render.ml): the forward loads N reals as (x, 0), computes the N-point DFT and stores bins 0..N/2 only; the backward loads bins 0..N/2, forms bin N-l as the conjugate of bin l (a sign flip, no load) and stores N real lanes. One kernel call is the whole transform (count 1 runs the VEX-128 tail): no promote, no child, no fold, no table, no scratch. The kind is alias-tolerant (no `__restrict__`; every load of a loop body precedes every store), so in place is the same call. 60 kernels at the n1 radices 3..64; gate gauntlet/rn1_gate.c (solo, in place, two rows per vector: 4e-17..2e-14). The door races it at even N <= 64 beside zr2c, the pair and ZTT-r; the real bridge races it at odd N <= 64 against the odd-real routes (`eng=zrm` or `eng=oddr` in the real shard, a different key from the oddr route record); the bound execute calls the kernel itself (`_k1x_zrm`); `VFFT_ZRM=1` pins it, `VFFT_ZRM=0` keeps it out. The real codelet family is one folder per engine since 2026-09-30: `real/pair/` (r2z, t2h, t2m) and `real/mono/` (rn1). | (a) shipped; (b) SHIPPED 2026-09-30. The door gate: ALL PASS at the 30 mono cells, both directions, both placements (gauntlet/real_door_check.c `tiny`, scratch store, recalibrate then replay). The create-time race (ns per call, 9 rounds paced and alternated, 3% hysteresis toward the incumbent): the mono WINS both directions at every N <= 19 -- r2c 3: 11 -> 2, 4: 9 -> 3, 5: 13 -> 3, 7: 16 -> 5, 8: 9 -> 4, 9: 22 -> 7, 11: 20 -> 8, 12: 12 -> 7, 13: 22 -> 11, 15: 39 -> 14, 16: 13 -> 9, 17: 28 -> 19, 19: 36 -> 24; c2r 3: 10 -> 2, 5: 15 -> 4, 7: 18 -> 10, 9: 15 -> 6, 15: 28 -> 13, 17: 26 -> 19, 19: 33 -> 25 (c2r 16 goes to the pair 4.4A by a wash). In place the even cells to 16 keep it for r2c; c2r in place at 10-16 stays zr2c by 1-4 ns. From 21 the straight-line kernel spills and it is a wash (21, 23, 29, 31 split by direction), the odd routes keep 25, 27, 37, 41, 43, 47, zr2c keeps 22 and 26, the pair keeps 32 and 64 (64: zrm 60 vs zrp4.16B 25). The library body, the probe's own race (door pick vs the incumbent, both through vfft_execute): 2.4x at 3, 2.6x at 4, 2.7x / 3.0x at 5, 2.7x / 2.9x at 7, 2.0x at 8 and 12. AGAINST FFTW (the 2..64 gauntlet, stripped-store protocol; `zrm_tiny_r2c_2026-09-30` control 1.135..1.159, `zrm_tiny_c2r_2026-09-30` control 1.084..1.092; before = `zttr_full_*_2026-09-29b`): over the 33 common cells the median goes 0.83 -> 1.00 in both directions and the cells below parity 21 -> 11 (r2c) / 21 -> 13 (c2r); the primes 3..13 go from 0.33-0.50 to 0.83-1.11, the even composites from 0.81 to 0.98 (r2c) and 0.85 to 0.94 (c2r). N = 3..16 sit at parity (r2c 0.87-1.12, c2r 0.80-1.13); N = 15 stays 0.70 / 0.69 (FFTW's 3x5 is 11 ns, the straight-line 15 is 15). The primes 17..31 are 2.0-2.9x ahead of FFTW's scalar ladder (17: 20 vs 56 ns, 19: 25 vs 69, 23: 36 vs 86, 29: 55 vs 111, 31: 61 vs 129, both directions alike). Left on the table: N = 2 (no rn1 kernel: zr2c 9 vs 4 ns r2c; c2r builds no engine and is refused -- untouched by this work, the 09-29 runs did not list the cell); the 2*odd and 4*odd composites 18..56 at 0.7-0.9 (zr2c with an odd child; no rn1 at 18, 20, 24, 28, 30); 25 c2r at 0.57 (unchanged: FFTW's 5x5); the wash cells 21-31 flap between the mono and the odd routes run to run (27 r2c 1.46 -> 1.12). |
| 3 | Fold radix as a raced axis | the fused stage's radix chosen per cell by the chain race, from a radix-2 fold pass to a fused radix-16 stage | not built |
| 4 | Deferred cross-row fold, 2D/3D | rows as plain c2c(N2/2), the fold as the last column stage's store edge over the 2D mirror pair; the c2c banded walk applies whole; column count N2/2 | not built |
| 5 | Rows as lanes | short rows vectorized across rows (complex mids through the column-stride kinds; the real leaf by gather), raced against along-row | not built |
| 6 | Real odd leaf | the odd blocked emission with real inputs: real s/r pairs, four columns per ymm, half the FMAs; replaces the promote bridge in every rank | not built |
| 7 | One row verdict per cell class | the primitive's wisdom keyed on (N2, row-count class, pitch class), shared by 1D K, 2D and 3D | not built |
| 8 | Rows threaded | the batch split across workers by row range, the TC batch's mechanism | not built |
| 9 | Real DC/Nyquist columns | the two real-valued columns of the 2D/3D half-spectrum transformed as real columns | not built |
| 10 | c2r destroy-input arm | the column pass in place on the caller's spectrum, no scratch plane | not built |

## 4. The protocol

Every measurement is the gauntlet's: core 2 pinned with its SMT sibling held,
one process per cell, both engine orders, best-of-5 in two windows, the
control cell every 100 cells, the same banked plans for both comparators, the
FFTW arm replaying one wisdom file per run (`fftw.wis` beside the store:
MEASURE once per problem, so the comparator holds still across cells,
directions and method races). A method is judged by the class medians and
the count of cells below parity, never by one cell.
