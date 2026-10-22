# 2D and 3D estimate: the sketch (interleaved c2c, FP64, one transform, natural order, one thread)

2026-10-22. The design of `VFFT_FLAG_ESTIMATE` at rank 2 and 3, written before the code that follows
it (`src/core/il/estimate/estimate_nd_il.h`, `src/core/il/estimate/c2c_estimate_nd_il.h`). Nothing was
timed. Every match rate below is counted against the banked rows of `src/wisdom/14900KF/` (natural
order, one thread: 159 standalone 2D rows, 1,288 3D rows and their 1,008 `plane_` children), with the
formulas mirrored from the code at this machine's caches (L1d 48 KiB, L2 2 MiB, L3 36 MiB). Code
citations are relative to `src/core/`. Tags: **[V]** read in code or a row, **[M]** counted against
the banked rows, **[I]** interpretation.

---

## 1. Scope (the owner's rulings, 2026-10-22)

| axis | served | refused, with a reason |
|---|---|---|
| rank | 2 and 3 (1 as before) | 4 |
| batch | one transform (howmany 1) | howmany > 1 |
| order | natural (the default) | scrambled at rank 2 and 3 |
| threads | one | nthreads > 1 at rank 2 and 3 |
| placement | both, one recipe: the 2D and 3D rows are placement-blind (`place=oop` in the key, il2d_tier.h:3982-3985, fftnd_il.h:1431-1437) | — |
| lengths | every axis >= 2, powers of two, odd, composites, primes: the column Bluestein (`blu=M`), the turned prime pass (`tpc`), the Rader stage (`rd`/`rdw`) | an axis of length 1; a last axis the row guard refuses (`vfft_policy_c2c_row_refused`, policy_il.h:511); a column with no chain whose Bluestein length M the column pool cannot chain (M above 2^24: N above 2^23, every N whose M passes an int among them, §5); a column whose Rader prime's ten listed candidate inners all lead with radix 64, which a request cannot rebuild (12289, 18433, 23041, 32257, 40961, 61441, 63361, 64513, 65537; §6, §14) |
| rank-3 structure | always `s=1`: the 2D plan per plane, then axis 0 | the flat structure is never planned |

The 1D plans inside a 2D or 3D plan (the children `rp_`, `turn_`, `csk_`, `tpc_`, and the `plane_`
child's own) are the 1D estimate's `est_plan_1d`, unchanged. Estimate never measures at create,
never reads or writes the user's wisdom, and is never a fallback.

## 2. Inputs

| input | used for | source |
|---|---|---|
| N1, N2 (2D); N1, N2, N3 (3D) | everything | the request |
| L1d | the single-stage bound, the narrow-row regime of the column chain; the 3D natural form's short axis 0 | `vfft_cpu_l1d_bytes()` (the P-core's, as at rank 1) |
| L2 | the band widths (2D and 3D); the 3D strip width | `vfft_cpu_l2_bytes()` |
| L3 | the 3D natural form: the strips only on a volume past it (no L3 reported counts as past, as the 1D estimate reads it) | `vfft_cpu_l3_bytes()` |
| lane count | refuse a machine whose lane count is not this build's (as at rank 1); the 1D children | `VFFT_IL_VW` |
| core count | not used: no rank-2/3 verdict at one thread relates to it | — |

## 3. What is written

A 2D request writes ONE row into the build's private store, the column row
`t=c2c n=N1xN2 q=1 ord=nat place=oop lay=il` (`vw2__ilcol_key` with rank 2, axis 0, T = 1), with
every token the door reads at one thread, and its children under their prefixes. A 3D request
writes the rank-3 row `n=N1xN2xN3`, whose `plane_` child is a whole 2D row with its own children.

| row | tokens | children |
|---|---|---|
| 2D | `chain` `forms` (when a stage has rival forms) `blu` (N1 not a power of two) `bluforms` (a Bluestein M chain with rival forms) `tpc` (Bluestein) `rd` `rdw` (a Rader stage) `wl` `tf` `ro` `sw` `rbk` `turn` `csk` | `rp_` always; `turn_` when the door builds the turn plan; `csk_` when it builds the skewed pass's row plan; `tpc_` when `tpc=1` |
| 3D | `chain` `forms` `blu` `bluforms` `tpc` `rd` `rdw` (axis 0) `wl` `tf` (the cycle walk only) `s=1` `nf` `nsw` (the strips only) | `plane_` (the 2D row at N2 x N3, natural, with its children); `tpc_` when axis 0 is `tpc=1` |

Every token the door would race behind is written explicitly: on a complete row nothing is timed
[V]. `ro`, `sw`, `rbk`, `turn`, `csk` are written on a Bluestein row too, though the door reads
only `ro` there (fft2d_create_il.h:494-535), and `rbk` when the row child has row-loop twins
(:620-626); the turn, the skew, the band and `sw` are gated on `!blu` (:544, 588, 652). There
`ro=0`: the route the door serves when the token is absent, since no axis race ever runs on a
Bluestein cell (fft2d_create_il.h:1103). The children the door builds whatever the verdict get recipes: the turn
plan (natural, no Bluestein, N2 with an `n1ccs` kernel; fft2d_create_il.h:544-576) and the skewed
pass's row plan (no Bluestein, N1 with a mono kernel; :584-617). The door is not changed.

`forms` follows the race's own spelling: written only when a stage has rival column forms
(`vfft_il2p_col_forms`, il2p.h:481; at c2c only radix 32: b48 / b84), one name per stage, each the
construction table's default `names[0]` ("b48" at 32, "b88" at 64, "-" else), as the race spells
its incumbent (il2d_tier.h:2483, 2597). [M] b48 is the banked form at a radix-32 stage in 33 of 38
2D rows; the whole `forms` token matches 127 of 159 2D rows.

## 4. The column chain (2D: the natural pass; 3D axis 0: the scrambled pass)

Both tiers build a column pass through `_il2d_col_build` (il2d_tier.h:3425) over N rows of `w`
complex: the 2D plan at (N1, w = N2) on the NATURAL pass, the 3D plan's axis 0 at (N1, w = N2 N3)
on the SCRAMBLED pass (both order classes: `vfft_policy_rankn_axis_nat`, policy_il.h:787). The
races of the two passes differ, so the formula takes the pass as an input.

**The candidates are the door's own pool**: every ordered composition of N over the column radices,
at most 4 stages, a closing-only radix (2, 6, 10, 12, 14, 22, 26) last, a Rader prime first
(`_il2d_enum_rec`, il2d_cols.h:160-253) [V]. The formula is a choice from that pool, so a chain it
names always exists and resolves (the pre-flight checks it with the door's resolver anyway).

**The formula.** Let `narrow(w)` = a radix-16 butterfly across the row fits the L1d,
16 · 16 · w <= L1d (w <= 128 on a 32-63 KiB L1d), and n = ceil(log2 N).

1. **One stage** when N is a pool radix and N < 16, or its plane fits the L1d: 16 N w <= L1d; on
   the scrambled pass every N above 16 needs 16 N w <= L1d / 4 (what the code does; whether the
   bound should cover only the blocked radix-32 and radix-64 kernels is open decision 8). One stage
   also when the pool holds no other chain (a prime up to 47).
2. **A Rader prime** (`vfft_policy_il2d_rader_prime(N)` = p, the engine builds it,
   `_ilrd_col0_ok`): the chain [p] followed by this formula's chain of N / p (at most 3 stages,
   nothing when N = p).
3. **A power of two from 2^s1** (s1 = floor(log2(L1d / 16)) = 11, the 1D estimate's tile span):
   a leading 32 on narrow rows, 16 on wide ones, then this formula's chain of N / 32 or N / 16
   (at most 3 stages).
4. **Otherwise** the pool's chains of two stages or more, the first by:
   - the stage count nearest k* = ceil(n / 4) on narrow rows (the radix-16 regime), ceil(n / 3) on
     wide ones (the radix-8 regime); on a tie the fewer;
   - then the most balanced: the least sum over the stages of (log2 R)^2;
   - then the order: the largest radix last (ascending), except the natural pass on wide rows,
     which closes on its smallest radix (descending); a closing-only radix stays last;
   - then the pool's own order.
5. **The exception table** (the scrambled pass only; the races repeat these):

   | N1 | rows | chain | banked rows it fixes |
   |---|---|---|---|
   | 512 | narrow | 4.8.16 | 14 of 21 (formula: 8.8.8) |
   | 1024 | wide | 8.8.4.4 | 21 of 45 (formula: 4.4.8.8) |
   | 4096 | wide | 16.32.8 | 15 of 24 (formula: 16.4.8.8) |
   | 8192 | narrow | 32.8.32 | 10 of 21 (formula: 32.16.16) |
   | 8192 | wide | 32.8.4.8 | 6 of 15 (formula: 16.8.8.8); the most banked there is 32.8.32, 7 of 15 (open decision 9) |

   (The 3D rows, natural, one thread, at that N1 and row regime: the entry's chain banked on the
   first count of rows, out of the second.)

**What it gives at a power of two** (2D natural pass / 3D scrambled pass; the 3D column shows the
table's entries in brackets):

| N | narrow rows (w <= 128) | wide rows | scrambled narrow | scrambled wide |
|---|---|---|---|---|
| 2..8 | N | N | N | N |
| 16 | 16 | 4.4 | 16 | 4.4 |
| 32 | 32 (w <= 64), then 4.8 | 8.4 | 32 (w <= 16), then 4.8 | 4.8 |
| 64 | 64 (w <= 32), then 8.8 | 8.8 | 64 (w <= 8), then 8.8 | 8.8 |
| 128 | 8.16 | 8.4.4 | 8.16 | 4.4.8 |
| 256 | 16.16 | 8.8.4 | 16.16 | 4.8.8 |
| 512 | 8.8.8 | 8.8.8 | [4.8.16] | 8.8.8 |
| 1024 | 8.8.16 | 8.8.4.4 | 8.8.16 | [8.8.4.4] |
| 2048 | 32.64 (w <= 32), then 32.8.8 | 16.8.4.4 | 32.64 (w <= 8), then 32.8.8 | 16.4.4.8 |
| 4096 | 32.8.16 | 16.8.8.4 | 32.8.16 | [16.32.8] |
| 8192 | 32.16.16 | 16.8.8.8 | [32.8.32] | [32.8.4.8] |

**Against the banked rows** [M] (exact chain):

| set | all cells | N >= 16 | the race's own repeatability |
|---|---|---|---|
| 2D natural, the standalone rows (159) | 93 (58%) | 54 / 120 (45%) | the same plane, `plane_` child vs standalone row: 61% |
| 2D natural, the `plane_` children of the 3D rows (1,008) | 551 (55%) | 259 / 708 (37%) | `plane_` vs `plane_`: 89% |
| 3D axis 0, scrambled pass (1,288) | 1,039 (81%) | 597 / 845 (71%) | T=1 row vs the one-thread part of the T=8 row: 87% |
| the same, the formula without the table | 974 (76%) | 532 / 845 (63%) | |

The 2D natural pass is where the races disagree most: at N1 = 512 to 2048 on narrow rows the
standalone rows take three large stages (4.8.16, 8.8.16, 32.8.8) and the `plane_` children four
small ones (4.4.4.8, 4.4.8.8, 4.8.8.8) for the same column problem. No rule fits both; the formula
takes the standalone rows' shape. The table holds only scrambled-pass entries for that reason.

**Odd lengths** (no evidence beyond the 2D 63x63 = 9.7 on the natural pass, 135x64 on the
scrambled pass = 5.3.9, and the 3D 9x15x27: axis 0 = 9 on the scrambled pass, its plane 15x27 = 15
on the natural pass) [I]: the same formula over the same pool, at L1d 48 KiB. 9 and 15 are one stage
on both passes; 27 is one stage while its plane fits (natural pass: w <= 113; scrambled pass:
w <= 28), else 3.9 (9.3 on the natural pass's wide rows). On narrow rows (either pass) 45 = 5.9,
63 = 7.9 (the banked 63x63, natural and narrow, is 9.7: the other order),
100 = 4.25, 1001 = 7.11.13; on wide rows the scrambled pass gives 5.9, 7.9, 4.5.5, 7.11.13 and the
natural pass, closing on its smallest radix, 9.5, 9.7, 5.5.4, 13.11.7. A chain with an odd radix is
served against Bluestein only by `blu=0` (§6).

## 5. The Bluestein inner chain (`blu=M`)

M = the next power of two >= 2N - 1, at least 16 (`_il2d_blu_build`, il2d_cols.h:928-938). The door
races the M chain UNBANDED on the scrambled pass (`_il2d_blu_m_chain`, il2d_tier.h:2405), so the
formula is the radix-8 regime's at M whatever the row width (§4 step 4 with `narrow` false),
ascending, no lead, no table, within the pool's four stages: ceil(log2 M / 3) stages of 4 and 8 up
to M = 4096 (4.4 at 16 ... 8.8.8.8 at 4096); from M = 8192 four stages that take 16s (8.8.8.16 at
8192 to 16.16.16.16 at 65536), then 32s (16.16.16.32 from 2^17), then 64s (32.32.32.64 from 2^21,
64.64.64.64 at 2^24). `bluforms` = the race's spelling of that chain's forms, written whenever the
chain carries a radix-32 stage (M = 2^17 to 2^23). The pool chains no M above 2^24, four stages of at
most 64: a column with no chain at N above 2^23 is refused. The estimate computes M in 64 bits; the
door's own loop computes it in an int (il2d_cols.h:937-938): for N between 2^29 and 2^30 it never ends, and
from N = 2^30 the int 2N - 1 itself overflows, M stays 16 and the chirp writes out of bounds. The estimate refuses all of these before the door.

[M] Against the six banked Bluestein rows (all from before the Rader stage: 127x64 nat 4.8.8,
101x129 nat 8.32, and the scrambled 127x100 8.4.8, 131x64 4.4.4.8, 509x1024 4.8.4.8, 1021x512
8.8.4.8): the exact chain 1 of 6, the radix multiset 4 of 6.

## 6. Chain, Rader stage, Bluestein, turned pass (the N-arm)

The door's N-arm (il2d_tier.h:3607-3877) races a chain with an odd radix against the column
Bluestein, and at a prime N1 with a Rader chain the turned prime pass as the third arm; `blu=`
and `tpc=` on the row replay the verdict. The rules, from the code and the policies:

| N1 | verdict written | evidence |
|---|---|---|
| a power of two | the chain; no `blu` (the door banks none, il2d_tier.h:4200) | — |
| every prime factor <= 47 | the chain, `blu=0` | [V] 63x63 `blu=0`, 135x64 scrambled `blu=0`, 9x15x27 `blu=0` and `blu1=0`: 3 of 3 |
| one prime p above 47 whose p - 1 chains (`vfft_policy_il2d_rader_prime`, policy_il.h:280) | the Rader chain [p] + the rest (§4 step 2), `blu=0`; `rd` = the engine's own inner pick (`_ilrd_pick_chain`, il_rader_stage.h:135) -- where that pick leads with radix 64 (p - 1 a multiple of 64: 193, 257, 449, ...) the first candidate of the engine's own inner pool (`_ilrd_enum_inners`, :658) that leads with a radix up to 47 or a prime, since a requested inner whose leading radix is above 47 and not prime is refused -- only a prime there is taken as a nested stage (`_ilrd_build_inner`, :355; the library defect of §14) -- so `rd=64.x` cannot replay; where all ten candidates the pool lists lead with 64 (12289, 18433, 23041, 32257, 40961, 61441, 63361, 64513, 65537) none stands in and the column is refused; `rdw` = four times the engine's default window (`_ilrd_default_W`, :375) within the column space (N1 / p) w, even | [M] the phase-A2 race records (10 cells raced with the stage in the pool): the planner picked the Rader chain at 10 of 10, `rd` = the greedy pick at 10 of 10 (13.10, 19.10, 4.13, 53.2), `rdw` = 4 x default at 5 of 10 (all five the 2D 131 cells; the others took the column space, 2 x, 1 x and 1/2 x) |
| a factor above 47 and no Rader chain (two primes above 47, p^2, or p - 1 that does not chain: 643, 2809, 3127) | the column Bluestein, `blu=M` with the M chain (§5) and `tpc=1`: the turned prime pass, the in-place natural 1D plan at N1 per lane, its recipe `tpc_` = `est_plan_1d` at N1 | [V] before the Rader stage the turned pass was the route the library served at the six c2c cells of `docs/roadmap/rader_leaf_design.md` §13 (524x64, 262x64, 131x256, 1048x32, 2101x16, 524x8x8) and is banked at 101x129 (`tpc=1`); 127x64 banked `tpc=0`: 7 of 8 |

The turned pass serves c2c axis 0 only (`_il2d_tpc_admits`, il2d_tier.h:3267): the 2D plan, the
3D plan's axis 0 and the `plane_` child's axis 0, which are every column the estimate plans. When
`tpc=1` the door still builds the Bluestein chain at M before it turns (il2d_tier.h:3643-3677), so
the M chain is written and pre-flighted all the same.

## 7. The 2D band width (`wl`, `tf`)

The candidates are the door's axis-race pool (il2d_tier.h:4009-4046): the ladder {8, 16, 32, 64,
128, 256} under the tcut law (`vfft_policy_il2d_wl_cut`, policy_il.h:575) and the stage spans
L[s], s >= 1, of eight rows or more. The formula:

- `wl=0` when the skewed pass or the turn serves (the race never crosses them with a band,
  il2d_tier.h:4082, 4091), on a one-stage or Bluestein chain, or while the plane fits half the L2:
  16 N1 N2 <= L2 / 2;
- otherwise the widest legal width below N1 whose band fits a quarter of the L2:
  16 wl N2 <= L2 / 4; 0 when none does.
- `tf` = (wl > 0), as the race banks it (il2d_tier.h:4199).

[M] Given the banked chain: the exact width in 125 of 159 standalone rows (79%) and 926 of 1,008
`plane_` children (92%), 1,051 together. Of the alternatives scored (the plane bound at L2/4..2 L2,
the band bound at L2/16..L2) the standalone rows score highest here; the `plane_` children score
higher with the band within L2/2: 941 (942 with the plane bound at L2/4), against 120 (119) of the
standalone rows, 1,061 together (open decision 10). The race's own repeatability for the `plane_`
width: 95%.

## 8. The 2D row route (`ro`, `rbk`, `turn`, `csk`, `sw`)

| token | formula | [M] standalone (159) | [M] `plane_` (1,008) |
|---|---|---|---|
| `csk` | 1 when N1 <= 16 and N1 has a mono kernel (`vfft_k1_mono_ilc_fn`), no Bluestein | 142 (89%) | 941 (93%) |
| `turn` | 1 when N1 >= 128, N2 <= 8, N2 has an `n1ccs` kernel, no Bluestein and no skew | 150 (94%) | 979 (97%) |
| `ro` | the turn: 0. N2 <= 16: 2, the batched mono rows (`n1ccs` at N2). 16 < N2 <= 64: 3, the batched two-pass rows, when the row plan is a pair whose four kernels have row-loop twins (`_il2d_rowloop_twin`, il2d_tier.h:889; fft2d_create_il.h:510-525), else 2 when N2 has an `n1ccs` kernel, else 0. N2 > 64: 0, the per-row child | 142 (89%) | 858 (85%) |
| whole route (`ro`, `turn`, `csk`) | | 126 (79%) | 806 (80%) |
| `rbk` | 8 (KB of the two-pass rows' tile; the door's value when the token is absent) | 142 of 159 | modal at every ro=3 row width |
| `sw` | 0 (the unbanded walk untiled) | 143 (90%) | 873 of 1,009 (87%) |

**N2 = 16 [I].** The banked rows put the two-pass rows (`ro=3`) there (standalone 6 of 13, `plane_`
most). The 1D estimate plans the row child at 16, natural, in place, as ZTURN-T 4.4 (the banked
child is the pair 4.4), which has no row-loop twins, so `ro=3` cannot bind (the door
would re-race the axis). The estimate takes `ro=2` there: the batched mono rows, the other batched
route. The count above includes this: with `ro=3` at 16 it would be 146 / 920.

## 9. The 3D plan

| token | formula | [M] against the 1,288 natural one-thread rows |
|---|---|---|
| `s` | 1: the 2D plan per plane (the owner's ruling) | 1,008 (78%) |
| axis-0 `chain` | §4 on the scrambled pass at (N1, w = N2 N3), with the table | 1,039 (81%) |
| axis-0 `wl`, `tf` | with the cycle walk: 0 on a one-stage or Bluestein axis 0; otherwise the widest legal width below N1 (`_ilnd_wl_pool`, fftnd_il.h:1133) whose band fits the L2, 16 wl N2 N3 <= L2, else the narrowest legal width; `tf` = (wl > 0). With the strips: not written (below) | given the banked chain: 607 of 1,201 rows that carry a width (51%); banded or not: 1,014 (84%). The race's own repeatability of this width: 41% |
| `nf`, `nsw` | the owner's rule (2026-10-23). The strips, `nf=2`, when the volume passes the L3, 16 N1 N2 N3 > L3 (no L3 reported counts as past, as at rank 1), and axis 0 is short, 16 N1 <= L1d / 4 (N1 <= 768 at a 48 KiB L1d: 512 among the powers of two); `nsw` = the widest of the door's strip widths (`_ilnd_sw_pool`, fftnd_il.h:826-836: 8 to 1024 columns, powers of two, the strip scratch 16 N1 w within the L2, w within the plane) whose scratch fits half the L2, 16 N1 nsw <= L2 / 2. The strips serve axis 0 unbanded (the door applies no band beside them, fftnd_il.h:1619), and `wl`, `tf` are not written: the race banks the band only where the cycle form won (fftnd_il.h:1733). Otherwise the cycle walk, `nf=1`, no `nsw` -- also on a one-stage or Bluestein axis 0 (the door's strip pool is empty there) and where no width fits | against the 796 rows that carry the race's `nf`: 739 (92.8%); the cycle walk always: 709 (89.1%). The misses: 43 rows where the strips won at volumes up to 32 MiB (19 at 32 MiB), and 14 at 64 MiB where the cycle walk won. The strip width: exact at 43 of the 87 strip rows (41 without the door's w <= plane bound) |
| `plane_` | the 2D plan at N2 x N3, natural, by §3-§8 | see the 2D rows |

At one thread with `s` and `nf` banked, and `wl` beside the cycle walk or `nsw` beside the strips,
the door has nothing to race: the configuration count is one (fftnd_il.h:1608-1621) and no flat
structure is built (`want2` = 0). The door takes the written `nsw` wherever its strip pool at that
axis 0 is not empty (fftnd_il.h:1566-1575); the pre-flight checks that `nsw` is one of that pool.

## 10. The 1D children

Each is `est_plan_1d` at its own request, unchanged, written with `est_put_rows` into a store of
its own and moved onto the parent row under its prefix with the door's `vfft_child_row_update`
(wisdom2_child.h:179):

| prefix | request | when |
|---|---|---|
| `rp_` | N2, in place, natural, T = 1 | always (the 3D plan's: inside `plane_`, at N3) |
| `turn_` | N1, in place, natural | the turn plan is built (§3) |
| `csk_` | N2, out of place, natural | the skewed pass's row plan is built (§3) |
| `tpc_` | N1, in place, natural | `tpc=1` |
| `plane_` (3D) | the whole 2D row at N2 x N3 with its own `rp_` `turn_` `csk_` `tpc_` | always (`s=1`) |

[M] `est_plan_1d` is exact on 4,260 of the 5,980 banked forward child rows (71%), most of the rest
in ZTURN-T stage order.

## 11. The build (`est_build_nd`)

The 1D build's steps, one rank up (`src/core/il/estimate/c2c_estimate_nd_il.h`):

1. **Pre-flight** what the door would refuse or race behind: every axis >= 2; the row guard on the
   last axis; each column chain multiplies to N (or M, computed in 64 bits), resolves
   (`_il2d_resolve`), takes its forms (`_il2d_apply_forms`), has its natural permutation on the
   natural pass (`_il2d_nat_perm`); a Rader stage builds as written and keeps the written window
   (`_ilrd_col0_build`, read back with `_ilrd_col0_desc`); a Bluestein M chain resolves with its
   forms; `ro=2` has the `n1ccs` kernel at N2; `ro=3` has the row plan's four row-loop twins; the turn
   and the skew have their kernels; the band width is legal (`vfft_policy_il2d_wl_cut`); with the 3D
   strips, `nsw` is one of the door's strip pool at axis 0 and axis 0 is unbanded; every 1D child
   passes the 1D pre-flight (`est__preflight`).
2. **Rows** into a private store with the store's own writers: the column row by
   `vw2_ilcol_chain_bank`, its integer tokens by `vw2_ilcol_tok_seti`, its string tokens (`forms`,
   `bluforms`, `rd`) by `vw2_ilcol_forms_bank_base`; each child by `est_put_rows` into its own store,
   then `vfft_child_row_update`; at rank 3 the 2D row is built the same way in a store of its own and
   moved under `plane_`. No fixed text buffer holds a row: a 3D row with nested children grows in the
   store.
3. **The door's replay** (`est__replay`, unchanged: `wisdom_write = 0`, the detectors).
4. **The plan check**, which catches a silent fallback: the 2D handle's chain (`il2d_col.nst`, `R`),
   `blu`, `tpc`, the Rader engine, the natural leaf, `wl`, `wc`, `il2d_rowb` / `il2d_rowb2` / the
   two-pass tile, `il2d_turn`, `il2d_csk`, and every built child's engine (`est__check_plan`); the 3D
   handle's `arm == 1`, `mt == 0`, the natural form `nf` (and with the strips the width `nsw`), the
   axis-0 descriptor the same way, and its plane plan checked as a 2D plan.

## 12. Refusals (the front door, `_vfft_api_translate`)

| request under the flag | message |
|---|---|
| rank 4 | `vfft_create: VFFT_FLAG_ESTIMATE plans ranks 1 to 3, not rank 4 (drop the flag to plan it by measuring)` |
| rank 2 or 3, scrambled | `vfft_create: VFFT_FLAG_ESTIMATE plans rank 2 and 3 in natural order only, not VFFT_ORDER_SCRAMBLED (drop the flag to plan it by measuring)` |
| rank 2 or 3, nthreads > 1 | `vfft_create: VFFT_FLAG_ESTIMATE plans rank 2 and 3 on one thread only, not nthreads %d (drop the flag to plan it by measuring)` |
| howmany > 1 | `vfft_create: VFFT_FLAG_ESTIMATE plans one transform, not a batch of %zu (drop the flag to plan it by measuring)` (as before) |
| any rank, while an environment variable that steers a plan is set (the owner's ruling of 2026-10-22) | `vfft_create: VFFT_FLAG_ESTIMATE: the developer switch %s is set; estimate refuses while a plan-steering switch is set (unset it: estimate plans from formulas only)` |

The formulas refuse, with the reason, an axis of length 1, a last axis the row guard refuses, a
column with no chain whose Bluestein length the column pool cannot chain (M above 2^24; the length
is computed in 64 bits, so a column whose M passes an int is refused at once, where the door's own
loop would never end: `N = 819724288 has no column chain and its Bluestein length 2147483648 none
either`), and a column whose Rader prime's ten listed candidate inners all lead with radix 64 (§6, §14).

**The steering variables** (`est_env_names`, `src/core/il/estimate/c2c_estimate_il.h`; one `getenv`
sweep at create, never at execute), each read on a path an estimate create runs:
- the 1D c2c doors: `VFFT_NO_IL2P`, `VFFT_NO_ILBLK`, `VFFT_NO_T2B`, `VFFT_IL_KV`, `VFFT_IL_BKV`,
  `VFFT_ILFD_NO_MSZ`, `VFFT_ILFD_NO_GL`, `VFFT_ILFD_NO_GORD`, `VFFT_ILFD_NO_GEN2`, `VFFT_ILFD_MT`,
  `VFFT_ZTT_MT`, `VFFT_K1_FS`, `VFFT_K1_FSSB`, `VFFT_ILPR_METHOD`, `VFFT_RD_ONE_LEVEL`,
  `VFFT_NO_K1PLAN`, `VFFT_NO_NAT_ILP`;
- the 2D door (also inside a 1D four-step): `VFFT_IL2D_CHAIN`, `_FORMS`, `_BLU`, `_TPC`, `_WL`,
  `_TFUSE`, `_WC`, `_ROWOOP`, `_CSK`, `_RB2_KB`, `_STAGED`, `_NO_COLMT`, `_MTARM`;
- the 3D door: `VFFT_ILND_ARM`, `_WL`, `_NF`, `_SW`.

Left out: the logging names (`VFFT_IL2D_LOG`, `VFFT_NAT_LOG`, `VFFT_ILPR_LOG`, `VFFT_IL_DP_VERBOSE`,
`VFFT_RACE_LOG`, `VFFT_VERBOSE`, ...), the execute-time profiling (`VFFT_IL2D_PHASES`,
`VFFT_ILND_PROF`), the stores' names, the real, split and transform-contiguous batch doors' names
(the flag refuses those requests), the names no estimate create or execute reads (`VFFT_PQ_NO_MT`,
the batch queues'; `VFFT_IL2D_DENSE`), the 3D door's threaded pins (`VFFT_ILND_MT`, `VFFT_ILND_PT`:
inert at the one thread estimate plans rank 3 at), the thread pool's (`VFFT_PIN_STRIDE`,
`VFFT_THREAD_BLOCK_MS`), and `VFFT_HIGHN_PLAN`, read only inside a race (k1_commit.h:777, 1651).

## 13. How to validate it later (on a quiet machine; the owner runs it)

`src/estimate/estimate_race_nd.c`: per 2D or 3D power-of-two cell, three plans in one process (the
banked plan from a scratch copy of the store, the estimate plan through the public flag, the banked
plan again as the control), the 1D tool's protocol (`vfft_measure_confine`, the caller on logical
CPU 2, HIGH priority, rounds with the arm order rotated 200 ms apart, medians and per-round ratio
ranges, the geometric mean). The cells that price the open choices: the 2D natural chains where the
standalone and `plane_` races disagree (512x16, 1024x8, 2048x4), N2 = 16 (`ro=2` against the banked
two-pass rows), the 3D band widths at volumes 4-64 MiB, the natural form at volumes around the L3
(the strips rule's misses: strips that won at 32 MiB, the cycle walk at 64 MiB), and the Rader
window at 53 and 107.

## 14. Decisions for the owner

Ten decisions: seven from the design (1-7), of which the owner has ruled one (7, 2026-10-23), and
three open from the review (8-10). One library defect is open below them.

1. **The Rader window `rdw`**: four times the engine's default within the column space (5 of 10
   records), or the engine's own default (1 of 10)?
2. **`tpc=1` for every Bluestein column**: the evidence predates the Rader stage (7 of 8); the
   columns still Bluestein today (two primes above 47, or a prime whose p - 1 does not chain) have
   no row.
3. **The Bluestein M chain in the radix-8 regime** (multiset 4 of 6, exact 1 of 6).
4. **N2 = 16 takes `ro=2`** because the unchanged 1D estimate plans the row child as ZTURN-T there.
5. **A Bluestein row carries `ro=0 wl=0 tf=0 sw=0 rbk=8 turn=0 csk=0`**, written for the "every
   token" rule though the door reads only `ro` on such a cell, and `rbk` when the row child has
   row-loop twins.
6. **The exception table** (5 entries, scrambled pass only; none on the natural pass, whose two
   sets of races disagree).
7. **The 3D natural form: ruled (the owner, 2026-10-23).** The strips rule of §9 (the strips past
   the L3 on a short axis 0, the widest door width within L2 / 2) replaces "`nf=1` always": 739 of
   796 raced rows against 709.
8. **The scrambled pass's one-stage bound** (§4 step 1, open). The code bounds every radix above 16
   by 16 N w <= L1d / 4; the alternative bounds only the blocked radix-32 and radix-64 kernels that way,
   and every other radix by the whole L1d. The two differ where a pool radix above 16 other than 32 and 64
   has another chain, on mid planes at a 48 KiB L1d: axis 0 = 21 (the code 3.7, the alternative
   [21], at N2 N3 = 37..146), 22 (11.2, 35..139), 25 (5.5, 31..122), 26 (13.2, 30..118), 27 (3.9,
   29..113). Every radix above 16, or only 32 and 64?
9. **The 8192 wide exception entry** (open). The table names 32.8.4.8, banked on 6 of the 15 wide
   rows at N1 = 8192; the most banked chain there is 32.8.32 (7 of 15; 32.8.8.4 on 2). Keep
   32.8.4.8, or name 32.8.32?
10. **The 2D band bound** (§7, open). The band within L2 / 4: 125 of 159 standalone rows and 926 of
    1,008 `plane_` children, 1,051 together. Within L2 / 2: 120 and 941, 1,061 together (119 and
    942 with the plane bound at L2 / 4). Keep L2 / 4, or take L2 / 2?

**Open library defect: a requested Rader inner chain that leads with radix 64 is refused.**
`_ilrd_build_inner` (il_rader_stage.h:355) refuses a requested inner whose leading radix is above 47
and not prime, while the engine's own pick (`_ilrd_pick_chain`) and its candidate list
(`_ilrd_enum_inners`) lead with 64 wherever p - 1 is a multiple of 64 (193, 257, 449, ...: 130 of
the 4,627 admitted primes from 49 to 70,000). The measuring door's race drops those arms, so it never
banks the engine's own pick there; where every listed candidate leads with 64 (12289, 18433, 23041,
32257, 40961, 61441, 63361, 64513, 65537) it banks that pick, which cannot replay. The estimate
stands in the first candidate that does not lead with 64 (§6) and refuses the column where none
does; its message there ("no inner chain of p - 1 that the engine rebuilds as written") overstates
it, since reorders such as 3.64.64 at 12289 rebuild. No shipped row is affected: the stores hold no
`rd`, `il_rd` or `rdw` token.
