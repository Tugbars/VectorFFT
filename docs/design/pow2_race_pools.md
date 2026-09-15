# Pure pow2 K=1 interleaved: what is raced, what was eliminated, what remains

Scope: the 1D C2C interleaved K=1 cells at a power of two, 16..262144, out of
place and in place, natural (= DEFAULT) and scrambled order, T=1. Everything
odd (chain3, the flat DIT, the 2^a * odd cells, the prime path) and the split
library are out of scope by the owner's instruction ("do not touch odd N
machinery, we'll come to that later. let's only focus on pure pow2"). The
laws behind every cut are in `design_contracts.md` (gitignored, like this
file). Written 2026-09-09, evening.

## 1. What was eliminated, and why

Every cut below is either a ruling on the contract or a retirement by
measurement (the pool-sunset policy). None is a heuristic. Counts are
candidates per cell as the census bench measures them.

| # | what left the race | where | why | evidence |
| --- | --- | --- | --- | --- |
| 1 | ZTURN-T's tile ladder 1 KB..64 KB (8 widths per chain) -> untiled, 16 KB, 32 KB | `dp_planner_il.h` `_il_dp_enumerate_ztt` | a width matters only through the stages it admits into L1; 32 KB is the largest width that leaves room for the twiddle streams; no width below 16 KB admits a stage 16 KB misses | every chain x every width at 2048..32768 on the planner clock: 16/32 KB win the cell at every N >= 4096, untiled wins 2048 (its 32 KB plane is L1-resident); the 1..8 KB "wins" were ties on chains that lose the cell; 64 KB never won a chain |
| 2 | the Bailey pairs at 2048 and 4096 (16 and 4 candidates) | `_il_dp_enumerate_natural_engines` | 2048..262144 is ZTURN-T's alone (contract section 4); "bailey shouldn't be in the search pool" | 4096: 64x64 5293 ns vs ZTURN-T 3550; 2048: 32x64 within 2% of ZTURN-T on the planner clock and behind on the paced verdict |
| 3 | the radix-64 pair slots (64xR, Rx64 arrangements) | same | "drop R64 mid, leaf. it's not needed" | never banked on this host or Zen 4 (Zen 4's 16x64 at 1024 predates ZTURN-T there) |
| 4 | at radix 8 and 16: the classic interior, the blocked 4.4 (`t2b`/`n1tb44`), the tangent M-128 mid (`t2tanm128`) | same, forward pools | the pool-sunset policy (2026-08-11): one shipped interior per slot once a tangent form wins its re-race; "tangent should stay, the rest will be gone" | tangent won every radix-8/16 forward slot in the store (64, 256, 512); it beat blocked 4.4 by 20-25% and the radix-8 classic bit-identically by 3% (`pure_il/tangent/README.md`) |
| 5 | the natural writers (ZTURN-T chains x widths) in the SCRAMBLED pools at 2048 and up, and their 9 promoted rows | `_il_dp_enumerate`, `c2c_oop_create.h` scrambled door, `wisdom2_oop.txt` | order is a contract: "scrambled belongs to only scrambled. when the user wants scrambled, then only then they are raced" (contract section 3); the S2 admission of 2026-09-09 15:55 is reverted | the S2 rows had made a scrambled request at 2048..16384 serve natural output ("natural served" in the K=1 gate) |
| 6 | the legacy zsplit cascade engine (`zroute=0`, chains x `sterm`/`sterm2`) in the pow2 scrambled pools | `_il_dp_push_cascade_chain` | superseded by ZTURN-S in July; "should not be part of the runs, it doesn't win anything" | zero banked rows on any host, ever |
| 7 | the cascade race arm at every door (natural OOP, DEFAULT OOP, in place) for every pow2 cell 16..262144, and the 5 rows it had banked | `vfft_ztt_band` (ztt.h), `c2c_oop_create.h`, `c2c_ip_create.h`, the store | contract section 4: no cascade at any pow2 band; "no cascade race arm please, eliminate" | the natural door had raced the natord cascade from 128 up and BANKED it at 512 (`mode=zcasc`, 400 ns, a door-clock verdict over a 300 ns pair): the front door was serving the cascade at 512 natural until this cut; the K=1 gate's 512 in-place error went from 3.5e-16 to 0 when the pair took the cell back |

Candidate counts before and after (natural / scrambled):

| N | before | after |
| --- | --- | --- |
| 16 | 3 / 3 | 3 / 3 |
| 32 | 7 / 7 | 5 / 5 |
| 64 | 15 / 15 | 7 / 7 |
| 128 | 25 / 25 | 13 / 13 |
| 256 | 36 / 36 | 13 / 13 |
| 512 | 41 / 41 | 13 / 13 |
| 1024 | 65 / 65 (37 after the ladder cut) | 23 / 23 |
| 2048 | 34 / 126 | 18 / 48 |
| 4096 | 40 / 175 | 36 / 77 |
| 8192 | 48 / 255 | 48 / 113 |
| 16384 | 63 / 352 | 63 / 166 |
| 32768 | 84 / 494 | 84 / 246 |
| 65536 | 108 / 660 | 108 / 340 |
| 131072 | - | 123 / 407 |
| 262144 | - | 108 / 346 |

## 2. The remaining race pool, cell by cell

NATURAL (and DEFAULT, which is natural by contract):

- 16: ZTURN-T 4.4, the solo `radix16_z_n1`, the pair 4x4.
- 32: ZTURN-T 4.8 and 8.4, the solo `radix32_z_n1`, the pairs 4x8 and 8x4.
- 64: ZTURN-T 4.4.4 and 8.8, the solos `radix64_z_n1` and the fused 8x8
  (`vfft_k1_mono64_8x8`), the pairs 4x16, 16x4, 8x8.
- 128: ZTURN-T 3 chains; pairs 4x32 and 32x4 with the four radix-32 forms
  in the 32 slot (4 candidates each), 8x16 and 16x8 (1 each).
- 256: ZTURN-T 4 chains; pairs 16x16 (1), 8x32 and 32x8 (4 each).
- 512: ZTURN-T 5 chains; pairs 16x32 and 32x16 (4 each).
- 1024: ZTURN-T 7 chains; pair 32x32 (16 = 4 mid forms x 4 leaf forms).
- 2048..262144: ZTURN-T alone, every registry chain x {untiled, 16 KB,
  32 KB} where the width is legal: 18, 36, 48, 63, 84, 108, 123, 108.

Below 2048 every pair candidate is one arrangement and one form pair; the
radix-4 slot has one form, the radix-8 and radix-16 slots have the tangent
form only, the radix-32 slot has four. The pair's BACKWARD forms are raced on
their own pass (`il_bkv`) and only the radix-32 slot has backward variants
(2.16 and 4.8 turned-store), so that pass exists at 128, 512 and 1024 only.

SCRAMBLED:

- 16..1024: the same candidates as the natural pool (the 2026-09-05 design:
  the natural engines' own race, banked on the ord=scr row). Under contract
  section 3 these cells have no scrambled writer at all; the owner's ruling
  on them is pending.
- 2048..262144: the ZTURN-S cascade alone, every {4,8} chain of 3..7
  stages x the terminator twins `stf`/`stf2` x its own tile ladder (256 B ..
  16 KB, seven widths): 48, 77, 113, 166, 246, 340, 407, 346. This is the
  interim scrambled writer; it is not tuned, it is replaced by the scrambled
  ZTURN-T class and deleted.

The doors: natural and DEFAULT, out of place and in place, serve the K=1
plan directly at every pow2 cell (no cascade arm; a stale in-place
`mode=zcasc` row is treated as unset), racing and banking on a miss.
DEFAULT = natural, in place too at pow2. SCRAMBLED at 2048 and up takes
the cascade, out of place and in place, and never enters the K=1 plan race
or the K=1 pair heuristic. T > 1 in the band is the serial verdict (the
ZTURN-T MT arm does not exist yet).

## 3. The codelet types still utilized at pure pow2

Every kernel below is a live kind with a measured place. Directions are
forward and backward unless stated.

**Solos (16, 32, 64, natural)**: `radix{16,32,64}_z_n1_{fwd,bwd}_avx2` (the
mono kind; `n1` forward exists only for these), `vfft_k1_mono64_8x8_il_{fwd,bwd}`
(the fused 8x8 at 64). In place: the alias-tolerant `n1c` twins.

**Bailey pairs (16..1024)**, forward: leaf `n1t` at R2 (transpose fused into
the stores), mid `t2` at R1. Per radix:

| radix | leaf (forward) | mid (forward) |
| --- | --- | --- |
| 4 | `radix4_z_n1t` | `radix4_z_t2` |
| 8 | `radix8_z_n1ttan` (tangent) | `radix8_z_t2tan` (tangent) |
| 16 | `radix16_z_n1ttan` | `radix16_z_t2tan` |
| 32 | `n1tb48` (4.8), `n1tb` (2.16), `n1tbw32` (wing32, T128 edge), `n1tbw32t256` (wing32, T256 edge) | `t2b48` (4.8), `t2b` (2.16), `t2bw32` (wing32), `t2bw32m128` (wing32, M-128 edge) |

Backward: stage 1 `radix{R1}_z_t2t_bwd` (turned store), stage 2
`radix{R2}_z_n1_bwd`, and since 2026-09-11 their tangent twins at radix 8
and 16 (`t2ttan_bwd`, `n1tan_bwd`, backward variant 3) and the radix-32
leaf (`n1btan216_bwd`, tangent 2.16 without the wing combine), which won every
backward slot they were offered (32, 64, 128, 256, 512); at radix 32 the
raced backward variants `n1b216`/`n1b48` and `t2bt216`/`t2bt48` lose to it. The classic radix-8/16 forward
kernels and the blocked 4.4 stay on disk for the backward side and for the
odd cells' rows; they are not in any pow2 forward pool.

**ZTURN-T (16..262144, every cell; the winner from 1024 up)**: one fused
driver per (cell, chain, direction, buffer mode), `ztt_<N>_<chain>_{fwd,bwd}_{dest,plane}_avx2`
in `generated/ztt_drivers_avx2.c`, inlining four kind bodies at radix 4 and 8:
`t0tp` (the permuting ingest), `tmg` (the pre-twiddle mid), `tlf` (the natural
terminator, dest mode), `tlfi` (the in-place terminator, plane mode, with the
output-stream prefetch) and their backward twins `t0tpb`/`tmgb`/`tlfb`/`tlfib`
(DIT order kept, `dif=true`). Sources: `codelets/zil/avx2/boundary_split/radix{4,8}_z_{t0tp,tmg,tlf,tlfi}[_bwd]_avx2.c`.
Twiddles: the baked quarter-wave `ztt_qw16384.h` up to the 16384 octave, the
two-level product above it. Zero calls inside a driver.

**ZTURN-S cascade (scrambled 2048..262144 only, interim)**: `s0t` ingest,
`msg`/`msd` mids (the post-twiddle DIF block combine `tmg` was derived from),
`stf`/`stf2` terminators (the `t2q` placement twins), radix 4 and 8, both
directions; `zturn_mt.h` for T > 1 where the cascade serves. Its
natural-order class (`stfl`/`stfnl` loaded-stream terminators, the `dts`/`dtsn`/`dtso`/`dtt`
DIT-forward boundary kinds, the `s0tu`/`stfu`/`stf2u` unordered-lane twins)
serves no pow2 cell any more; it lives on only at the odd cells' in-place
natural rows until their turn.

## 3b. How the pools are driven

Two calibrators, one per library, each writing only its own rows:
`build_tuned/benches/calibrate_k1_il.c` (the interleaved pools above, lay=il
rows: natural, dir=bwd, ord=scr) and `calibrate_k1_split.c` (the split race,
the lay=split row). The create's own race (`_k1_il_plan_race`) runs the same
interleaved planner on any cold interleaved cell the planner covers — below
2048, at odd N, and (2026-09-09) in ZTURN-T's band up to the ceiling — banks
the winner and serves it: wisdom or race, never a fallback. The prime
engine is a route for its own N, never a fallback for a power of two.

## 3c. The gates that hold this in place

`core/support/slot_check.h` + `core/planning/il_slot_probe.h`, driven by
`benches/form_slot_gate.c` — the resolver invariant: at 16..1024, every legal
(R1, R2) arrangement x every form the arm pools and backward resolvers offer
must build, run and pass the planner's correctness gate (238 kernels, 0
wrong). `benches/tangent_bwd_gate.c` — each tangent backward twin against the
classic kernel it replaces. `benches/bwd_forms_race.c` — the backward forms
race for the shipped pairs, N stable repeats before banking.
`benches/k1_pow2_gate.c` — the served plan per cell, both orders, both
placements, replay-bitwise. `benches/il_dp_overflow_gate.c` — the candidate
census per cell.

## 4. Open at pure pow2

1. (done 2026-09-11) the tangent backward twins at radix 8, 16 and the
   radix-32 leaf; every backward slot of every shipped pow2 pair runs one.
   The backward wing combine was built, raced and lost to the plain tangent
   2.16 leaf (retired); the emitter keeps both directions.
2. The natural rows at 32768..262144 (no row today; first use hits the
   create-time race).
3. The scrambled ZTURN-T class, after which the cascade and its whole kind
   family leave the pow2 tree.
4. The scrambled cells below 2048 (no scrambled writer under section 3).
5. Above ~65536: Bailey vs ZTURN-T, raced, before the upper band is ruled.
6. ZTURN-T's MT arm.
