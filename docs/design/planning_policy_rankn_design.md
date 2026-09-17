# The planning policy, rank >= 2 (design, 2026-09-17)

The K=1 rank-1 migration (`planning_policy_design.md`, steps 1-8) put every
law about a 1D REQUEST in `planning/policy.h`. The 2D and 3D interleaved
tiers -- ~6600 lines -- were left with one contribution from the module
(`vfft_policy_ord_rankn`) and, since 2026-09-17, one constant
(`VFFT_IL2D_MAXCAND`). This design does for them what the first did for 1D:
each law written once, each move behavior-preserving, each gated.

## What rank >= 2 policy IS, and is not

There is no rank >= 2 band map to build. The ROW axis calls `vfft_create`
recursively (`fft2d_create.h`, and `fftnd_il.h`'s flat child), so it inherits
the whole 1D map. The COLUMN axis has two regimes only -- a chain raced over
one radix pool, or the column-axis Bluestein when no chain exists -- and
since 2026-09-17 no heuristic between them.

What IS scattered sits one level down, and it is the class that produced
this week's defects (`nat_req` vs `key->ord`; the greedy's private pool):

- **pools** -- which values may enter a race;
- **legality laws** -- which chains, widths and forms are admissible;
- **the arguments the shared column builder takes from its callers** --
  each one a place three callers can compute one thing three ways.

Raced parameters -- the winners -- are NOT policy and do not move.

## The inventory (every site opened and read, 2026-09-17)

The rank >= 2 census (10 agents, 145 rules claimed, 61 duplications claimed)
is a MAP. Only entries verified by reading are listed; the census's other
claims stay in `docs/roadmap/policy_survey_defects.md` section E as claims.

| # | law | written at | verdict |
| --- | --- | --- | --- |
| R1 | the band-width ladder `{8, 16, 32, 64, 128, 256}` | `il2d_tier.h:1257` (real), `:2237` (c2c), `fftnd_il.h:1056` (3D) | THREE identical literals. Move. |
| R1b | the column radix pool | `il2d_cols.h` (`_il2d_enum_rec_body`), once since the greedy left | single site; move for locality only, not dedup |
| R1c | the strip-width ladders | `il2d_tier.h:1138` `{16..256}`, `fftnd_il.h:776` `{8..1024}` | DIFFERENT lists, deliberately. Stay; document |
| R2 | the bounds around the shared L2 gate | c2c `:2269` `w > N1 \|\| N1 % w \|\| w < 8`; real `:1274` `w2 >= N1` (no floor); 3D `fftnd_il.h:1068` `w < 8` + cut | THREE different bound sets. A RULING, not a merge |
| R3 | the tcut law: wl legal iff `wl \| N1` and some stage span divides it; the cut is the first such stage | helpers `_il2d_real_wl_cut` `:992`, `_ilnd_wl_cut` `fftnd_il.h:1034` (byte-equivalent); inline `:2246` (legality, spelled with `cut = -1`), `:2340`, `:2396` (RECOVERY of an admitted width's cut, `cut = 0` default) | five spellings of TWO laws, not one -- caught by reading, 2026-09-17. The recovery loops default to 0 where the helpers refuse; unreachable in practice (`bwl` is a race winner among admitted widths) but not the same predicate. Move as TWO helpers: `wl_cut` (legality + cut) and `cut_of` (an admitted width's cut), each byte-exact at its sites |
| R4 | a stage has a form axis iff its radix is 32 or 64 | authority `vfft_il2p_col_forms` (`il2p.h:457`); hand copies `il2d_tier.h:1540`, `:1766` | THREE spellings. The copies call the authority; nothing moves to policy.h |
| R5 | form precedence: env pin > banked `forms=` > the per-stage race; a pin never banks; a banked form that does not fit re-races | `_il2d_forms_serve` `:1530` (asks the 2D `(is_real, N1, N2, ord)` key) and `_il2d_forms_serve_key` `:1755` (asks the rank-N `ilcol` key) | one 45-line law twice, differing only in the key. Unify (medium) |
| R6 | the column-chain decision: env pin > banked row > race > Bluestein | `il2d_tier.h` (c2c, inside `_il2d_col_build`) and `fft2d_create.h` (the real tier, inline) | the real tier is a COPY of the c2c block with its own lookup/bank. The 3D tier already calls the builder; the real tier should too. Large; its own design |
| R7 | which PASS an axis runs (natural leaf or scrambled): 2D = the request's class; 3D axis 0 = scrambled for BOTH classes; 3D axis 1 = the request's class | literals at the three `_il2d_col_build` calls (`fft2d_create.h:257`, `fftnd_il.h:1114`, `:1307`) | three spellings of one small law -- and the exact one that drifted on 2026-09-17. Move |
| -- | the c2c door admission | `fft2d_create.h:250` | one site (the census claimed two; the second was not found). Stays |

## Contract

Same as rank 1: `policy.h` is declarative (names, integers, small tables),
sits above every engine, and answers questions about a REQUEST or a CELL --
never builds, never measures, never picks a winner. New section "rank >= 2",
below the rank-1 laws, above L3/L6/L8.

The one new kind of thing it holds: LADDERS as tables (R1), which the tiers
iterate. A ladder is a pool; the race still decides.

## Steps, each behavior-preserving, each gated, in this order

- **R1** -- `VFFT_IL2D_WL_LADDER[]` once; three consumers iterate it. (The
  radix pool moves beside it for locality.) Constant tables: a move cannot
  change a verdict. Gate: the three arrays are gone (`grep`), the sweep.
- **R3** -- `vfft_policy_il2d_wl_cut(N, nst, L, wl)`; the two helpers become
  it, the three inline loops call it. Gate: a `_ref_` twin (the old helper,
  frozen) over every (N1 <= 4096, every chain the enumerator yields, every
  wl in the ladder), asserted equal; the sweep.
- **R4** -- the two hand copies ask `vfft_il2p_col_forms(R, names) > 1`.
  Gate: a `_ref_` twin over every radix in the pool; the sweep.
- **R7** -- `vfft_policy_rankn_axis_nat(rank, axis, ord)`; the three call
  sites pass it. Gate: a `_ref_` table of the nine (rank, axis, ord) cases;
  the sweep; and the control that caught the original drift -- the axis-0
  race log reads `(scr)` for a natural 3D cell.
- **R2** -- STOP for a ruling: is `w == N1` a legal band width, and is 8 the
  floor everywhere? Whatever is ruled becomes one predicate; the candidate
  census (arm lists before/after, per cell) is the gate, because this one
  CHANGES which widths race.
- **R5** -- one forms-serve taking the key as a small union; bitwise replay
  is the gate. Medium.
- **R6** -- the real tier onto `_il2d_col_build`. Its own design.

Never: a rank >= 2 band map by analogy with 1D; unifying R2's bounds without
the ruling; touching a raced parameter.

## Gates

`benches/policy_gate.c` grows a rank >= 2 section, same discipline: the old
spelling frozen as a `_ref_` twin, exhaustive equality, and a NEGATIVE TEST
per law watched to fail before it counts. Plus a rank >= 2 candidate census
(`VFFT_IL2D_LOG` arm lists, cold store, timings stripped) over cells covering
both tiers, both classes, prime and composite N1 -- the check that caught
step 3's drift in rank 1.

## Steps R1, R3, R4, R7 -- the evidence (2026-09-17)

| check | result |
| --- | --- |
| `policy_gate`, the rank >= 2 arms | 131,571,790 checks, ALL PASS: R3's two laws equal to their frozen twins over 11,825 chains to N1 = 4096 x every width 0..N1+1 (23.8M pairs); R7 equal over its eight cases |
| NEGATIVE TESTS, watched to fail | R3a with the divisibility term dropped: `3630683 of 23839850 (chain, wl) pairs differ`; R7 with 3D axis 0 returning the natural pass: `1 of 8 cases differ` -- exactly the 2026-09-17 drift |
| `il2d_onechain_gate`, `ilnd_gate` | ALL PASS on the migrated tree |
| the three `WPOOL[]` literals, the `== 32 \|\| == 64` copies, the inline tcut loops | gone (`grep`) |

## What the first 3D gate found (2026-09-17)

`ilnd_gate` -- the 3D interleaved tier's first gate, written the same day --
failed on its first run at 23x8x8, both classes: a cell whose axis 0 is
BLUESTEIN re-raced axis 1 and its structure verdict on every create. Cause:
"axis 0 creates the row, a later axis updates fields on it", and a Bluestein
axis 0 created none (`[wisdom2] il column axis 1 bank refused (no row)`).
And the fix could not be "just bank the row": the shared builder had NO
replay-rebuild for a `blu` row -- it would have run the length-M chain as
the N chain (the 2D create has had that rebuild since 2026-09-02; the 3D
tier, the builder's only other caller, never did, and never noticed only
because no 3D Bluestein cell could bank). Fixed in `_il2d_col_build`,
`rank >= 3` only: a replayed `blu` row rebuilds its inner through the
provider; a cold Bluestein axis 0 banks the cell's row with the M chain and
`blu = M`. 12/12 after the fix; the failing first run is the negative test.
Pre-existing: R7 touches no banking.

## R2, R5, R6 -- the evidence (2026-09-17)

| step | check | result |
| --- | --- | --- |
| R2 | the real tier's ladder, before -> after, 64x64 real | `wl ladder 64x64: 16 32` -> `16 32 64`: the full-width band is an arm, as ruled; no other line of a six-cell census changed (the floor has no instance at those cells) |
| R2 | `policy_gate`, `band_ok` vs the c2c spelling over stage spans and the 3D spelling over every width | equal; NEGATIVE TEST (floor 4): `R2 band_ok: 9204 disagreements` -- watched to fail |
| R5 | `il2d_onechain_gate`, `il2d_real_gate`, `ilnd_gate` | ALL PASS |
| R6 | `il2d_real_gate` (r2c and c2r, warm bitwise) | ALL PASS |
| R6 | the six-cell real census vs R2's "after" | IDENTICAL arms; one line gone -- the deleted block's own summary log |
| all | `policy_gate` | 131,571,790 checks, ALL PASS |

A correction recorded on the way: the shared builder DID already rebuild a
replayed Bluestein inner at every rank -- inside its N-arm block (E1.7,
`il2d_tier.h` ~:2018), past the range read when the 3D fix was written. The
rank >= 3 rebuild added that morning was redundant and is deleted; the 3D
fix that mattered was the BANK (a). `ilnd_gate` passes on (a) alone.

## Checklist

- [x] 0. This design (inventory verified by reading, 2026-09-17).
- [x] R1. The ladder, once (2026-09-17).
- [x] R3. The tcut law, as TWO helpers, each byte-exact (2026-09-17).
- [x] R4. The form-axis rule asks its authority (2026-09-17).
- [x] R7. The axis-pass law, once (2026-09-17).
- [x] R2. Ruled 2026-09-17 (the real tier follows c2c/3D): `vfft_policy_il2d_band_ok` (floor 8 + the tcut law) at all three cascade loops; the real tier's static ladder admits `w == N1`.
- [x] R5. One forms-serve body (2026-09-17): the 2D-key twin is a 2-line wrapper, because `vw2_2d_forms_lookup/bank` were already wrappers building the same ilcol key.
- [x] R6. The real tier's chain through the shared builder (2026-09-17, `il2d_real_on_shared_builder_design.md`): 298 lines -> 55.
- [x] Records (2026-09-17).
