# ZTURN-T at 2^a·odd — the odd mid, both order classes, staged

Design, 2026-09-14. Basis: `ztt_scrambled_design.md` (the two schedules and
their measured geometry), `design_contracts.md` sections 4, 5, 8 and 10 (the
owner's band law, the deletion order, the stage-kernel/fused-codelet ruling),
`zcascade_sunset_plan.md` section 1 (the cascade's inventory). This is the
engine that lets the ZTURN-S cascade be deleted: the cascade's last cells are
the ones defined here.

The owner's framing: **flat Cooley-Tukey may be the better engine for pure
odd N; mixed pow2·odd is another story.** Pure odd stays with the flat DIT.
This doc is the mixed story only.

## Terms

As in the pow2 doc, plus:

- **odd mid** — a mid stage (`tmg` natural, `tmgd` plain, `tmgb` backward)
  whose radix is 3, 5, 7, 9 or 15.
- **staged executor** — the runtime walk of a plan's stage table that calls
  one stage kernel per (stage, block), with run-time trip counts. It is the
  executable form of every ZTURN-T cell outside pow2 (contracts, section 10:
  the stage kernels are the product; fused codelets are the pow2 solution's
  form only).
- **prefix product** `P_s = R_0 · … · R_s`; **suffix product**
  `Len_s = R_s · … · R_{nf-1}`.

## The cells

    N = 2^a · m,   m a product over {3, 5, 7, 9, 15},   a >= 4,
    2048 <= N <= 262144

This is exactly the set the cascade admits today (`_vfft_zt_msg_pick`'s odd
set, `_il_dp_enumerate_odd_mids`'s decomposition: at most five odd mids),
so its deletion loses no cell. It is 339 sizes, 2160 to 259200. `a >= 4` is
forced by the chain grammar below (a radix-4 or 8 ingest and a radix-4 or 8
terminator with the odd factor between them). Odd-factor cells outside this
set (a prime factor of 11 or more, `a < 4`, below 2048, above the ceiling)
are not this engine's: the flat DIT serves them where it is admitted and the
pairs, chain3 and prime engines where they are; nothing here touches them.

What serves the band today, from the shipped store (2026-09-02/03 rows):

| contract | engine today | example row |
| --- | --- | --- |
| natural, out of place | chain3 (pairs + one mid), raced against the natord cascade on the door | 3072 `il_route=chain3 il_chain=32.16.6` 3.7 us |
| natural, in place | the cascade's natord path | 3072 `mode=zcasc` 12.6 us |
| scrambled, either placement | the cascade alone (comb terminator) | 3072 `eng=zturn chain=4.3.8.8.4 zt_tw=768` |
| threaded, N >= 16384 | the cascade's sectioned walk (`zt_mt`) | — |

The in-place natural cell is the one that is badly served: 3.4x its
out-of-place twin at 3072, because the cascade's natural path is a plane
walk with a comb, not an in-place engine. ZTURN-T brings its plane driver
and its in-place plain class to that cell.

## Placement law: the odd factor is a mid, never an end

Three template facts fix this, none of them negotiable by the planner:

1. **The natural ingest `t0tp` is a closed-form radix-4/8 turn lattice**
   (`emit_s0t_body`, `emit_s0t8_body`), not a DAG-emitted kind. There is no
   radix-R form of it. Its column count `N/R_0` must be a multiple of 4.
2. **The plain terminator `tld` and its backward `tldb` divide the radix by
   the lane width** (`E_blocks`, `E_zcol`, `E_sect_tr4`: `radix/vw`, radix in
   {4, 8}). The natural terminator `tlf`'s section transpose has the same
   gate.
3. **The mids are radix-agnostic.** `tmg`, `tmgd` and `tmgb` use only the
   `E_z` and `E_planes` edges, whose addressing is `2*(leg*Ls + k)` for any
   leg count; the DFT body comes from `Dft_select` (direct conjugate-pair
   for 3, 5, 7; Cooley-Tukey 3x3 and 3x5 for 9 and 15), the same math the
   cascade's `msg`/`msz` odd mids ship today. The only thing that stops
   `gen_radix.exe 3 --zp-tmg` is the kind-name gate at `cascade_z.ml:949`.

The lanes themselves do not forbid an odd end: every kind processes four
columns per iteration and any leg count, and an odd ingest's column count
`N/R_0` would still be a multiple of 4. The ends are barred by their lane
transposes, not by the lane width. That leaves one admissible exception:
the plain class's stage 0, `t0d`, is DAG-emitted on the radix-agnostic
edges (no turn lattice), so an odd radix at stage 0 is possible there in
principle. It is not enumerated: one grammar serves both classes, and the
natural class has no such stage. It stays a stated choice, to be raced if
the odd-mid position ever measures as the limiting factor.

So the chain grammar is

    R_0 in {4, 8}     R_{nf-1} in {4, 8}     R_s in {4, 8, 3, 5, 7, 9, 15} for 0 < s < nf-1
    3 <= nf <= 7      prod R_s = N

with the odd part of N decomposed greedily, largest first, over
{15, 9, 7, 5, 3} (one decomposition per N: 9 is a radix-9 mid, never 3·3;
45 is 15·3) and its mids placed at every interior position, the pow2 slots
walked over ordered {4, 8}. This is `_il_dp_enumerate_odd_mids`'s grammar
today; it moves from the cascade's enumerator to ZTURN-T's, with one
relaxation: the cascade demanded a pow2 mid beside the odd ones
(`nf >= nm + 3`); ZTURN-T needs only the two ends (`nf >= nm + 2`), so
`4.3.4`-shaped chains exist at the band's smallest sizes.

Every 4-column contract in the kinds holds under this grammar, in both
classes:

- natural: the ingest count is `N/R_0` (a multiple of 4 since `a >= 4`);
  a mid's count is `L_s = P_{s-1}`, which contains `R_0`; the terminator's
  is `L_{nf-1}`, likewise.
- plain: stage 0's count is `Len_1 = N/R_0`; a mid's is `Len_{s+1}`, which
  contains `R_{nf-1}`; the terminator's group is `R_{nf-1}` wide.
- the twiddle fill `_ztt_fill_stage` is radix-agnostic in `R` and needs
  its `L` argument to be a multiple of 4 — the same quantities.
- the digit reversal `_ztt_digitrev` is mixed-radix already; `rb[]` and the
  plain permutation (`R_{nf-1} * (col & ~3) + 4p + sig[col & 3]`) use it
  unchanged, because the terminator radix is still 4 or 8.

## Geometry

Unchanged from the pow2 doc; only the products are no longer powers of two.

Natural: `L_1 = R_0`, `L_{s+1} = L_s · R_s`, `Gs_s = N / (R_s · L_s)`; stage
s carries `2 (R_s - 1) L_s` twiddle doubles; the ingest scatters column c's
run to `rb[c]`; the terminator's legs are `N / R_{nf-1}` apart.

Plain: `Len_0 = N`, `Len_{s+1} = Len_s / R_s`; stage s carries
`2 (R_s - 1) Len_{s+1}` doubles; stage `nf-1` none; every stage in place.

**Tile law.** A tile holds whole groups of every tiled stage and the tiles
partition the array. A group of stage s spans `P_s` complexes, so a width
`T` is legal when `T` divides `N` and the first mid's group `P_1 = R_0 R_1`
divides `T`; the tiled prefix is then the mids whose `P_s` divides `T`
(a prefix, since each `P_s` divides the next), the rest are sweeps. Plain
mirrors it with the suffix products `Len_s`. At pow2 every divisibility is
automatic and this is the shipped law (`tile >= R_0 R_1`, `tile < N`).

The ladder is the owner's (2026-09-14): **8, 16, 32 KB and, at the odd
cells, 48 KB** — 512, 1024, 2048 and 3072 complexes — plus untiled; not
every product in a band. At an odd cell a pow2 width holds whole groups only
of the mids before the odd one (`P_s` is a power of two until the odd mid),
so the pow2 widths tile chains whose odd mid comes late; 48 KB, a multiple
of 3, tiles through a radix-3 mid (`8.3.8.8.8` at 12288: `P` = 8, 24, 192,
1536; 3072 holds two groups of the last mid, 2048 holds none). Chains with
a 5, 7, 9 or 15 mid tile only with the pow2 widths before that mid. The
race over chains times widths sorts this out; the tile invariance gate
(bitwise across widths) is unchanged.

**Sweep count** is the pow2 doc's: natural `1 + #{mids with R_s L_s > tile}
+ 1`, plain `1 + #{stages with Len_s > tile}`; plain is one fewer on the
reversed chain.

## The executable form: staged

The stage kernels are called, not inlined. Every ZTURN-T kind is exported
in the 11-parameter ABI with the group loop inside it
(`radixR_z_<kind>_{fwd,bwd}_avx2(zin, zin2, zout, zout2, tw_re, tw_im, Ls,
Gs, OLs, OGs, count)` loops `Gs` groups over one group-invariant stream),
so one call serves one (stage, block) — at 12288 that is 5 calls untiled,
or 1 + 4·3 + 1 with a 3072 tile, not one per group. The fused
drivers' measured worth over this is 0-3% at every cell from 2048 up; the
band has no cell below 2048.

The plan gains a stage table and the executor a walk:

    struct { vfft_ztt_stage_fn fn; long Ls, Gs, count; size_t twoff; long pitch; } st[nf]

`vfft_ztt_execute_*` on a plan with `cell == NULL` runs the table with the
fused driver's exact loop nest: natural = ingest sweep, then per tile the
prefix of mids with `R_s L_s <= tile`, then the remaining mids as sweeps,
then the terminator (dest or plane, as bound); plain = stage 0 sweep, the
stages with `Len_s > tile` as sweeps, then per block the suffix. The plain
backward is `tldb` per block, `tmgb` high-to-low, `tlfb` in place. This is
`ztt_drivers.ml`'s `emit_natural_driver` / `emit_plain_driver` transcribed
once into C with run-time bounds; the emitter is the reference for the
walk and the gate below holds the two to each other at pow2.

What is new in the tree:

| piece | content |
| --- | --- |
| kernels | `radix{3,5,7,9,15}_z_{tmg,tmg_bwd,tmgd}_avx2.c` — 15 files, corpus rows, corpus law restamped |
| generator | `cascade_z.ml:949` admits `tmg`/`tmgd`/`tmgb` at the odd set; nothing else in the emitter changes |
| registry | a per-kind, per-radix function table (`vfft_ztt_kind_fn(kind, R, dir)`), beside the pow2 cell table |
| create | `vfft_ztt_create_chain_ord` accepts the grammar above, fills the stage table when no fused cell matches, and the odd tile law |
| execute | the staged walk, natural and plain, both directions, dest and plane |
| planner | `_il_dp_enumerate_ztt_ord` at an odd-band cell enumerates the grammar times the odd tile widths |

Nothing is emitted per cell. The band's 339 sizes add no generated text
beyond the 15 kernels. What they do add is calibration: the calibrator
banks the list the owner names (today's six banked sizes and the spike cell
at least); every other band cell races at create under the wisdom-or-race
law, as any cold cell does.

## Worked cell: N = 12288 = 2^12 · 3

Natural, chain 8.3.8.8.8, tile 3072 (48 KB, four tiles). `L` = 8, 24, 192,
1536.

| stage | kind | Ls | Gs | count | twiddle (doubles, offset) | placement |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `t0tp8` + `rb` | 1536 | 1 | 1536 | none | sweep |
| 1 | `tmg3` | 8 | 512 | 8 | 32 at +0 | per tile, 128 groups |
| 2 | `tmg8` | 24 | 64 | 24 | 336 at +32 | per tile, 16 groups |
| 3 | `tmg8` | 192 | 8 | 192 | 2688 at +368 | per tile, 2 groups |
| 4 | `tlf8` / `tlfi8` | 1536 | 1 | 1536 | 21504 at +3056 | sweep |

Plain, chain 8.8.8.3.8, tile 3072 (four blocks). `Len` = 12288, 1536, 192,
24, 8, 1.

| stage | kind | Ls | Gs | count | twiddle (doubles, offset) | placement |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `t0d8` | 1536 | 1 | 1536 | 21504 at +0 | sweep |
| 1 | `tmgd8` | 192 | 8 | 192 | 2688 at +21504 | per block, 2 groups |
| 2 | `tmgd8` | 24 | 64 | 24 | 336 at +24192 | per block, 16 groups |
| 3 | `tmgd3` | 8 | 512 | 8 | 32 at +24528 | per block, 128 groups |
| 4 | `tld8` | — | 1536 | — | none | per block, 384 groups |

Same 24560 doubles of twiddle on the reversed chain; the plain has one sweep
fewer, as at pow2. The radix-3 stage is the cheapest stage of either
schedule (two twiddled legs, four multiplies per column); where the odd mid
sits in the chain is a question for the race, not the design, and the
cascade's banked chains (odd mid first, `4.3.8.8.4`, `4.5.8.8.4`,
`4.4.4.4.7.4`) are one data point on it.

## The output order

Natural: natural. Plain: position i in the chain's mixed radix
`i = d_0 Len_1 + … + d_{nf-1}` holds `k = d_{nf-1} P_{nf-2} + … + d_0`, the
mixed-radix digit reversal, with the terminator's lane order inside each
block of four groups — `vfft_ztt_perm` tabulates it at create as at pow2.
The contract text in `include/vfft.h` (the same-plan rule) already covers
it; nothing is added.

## Planner, doors, wisdom

- `vfft_ztt_band(N)` becomes the union of the pow2 band and the odd band
  above. Every door that today says "no cascade inside the band" then says
  it at the odd cells too, and the cascade has no door left.
- Route 9 (`VFFT_K1_IL_ZTT`), tokens `il_ztt=<chain>` (odd digits appear in
  the chain, e.g. `8.3.8.8.8`) and `il_tw=<width>`; `ord=nat` / `ord=scr`
  carry the class. No new token.
- The scrambled pool at an odd-band cell is the plain ZTURN-T grammar times
  its tile widths and nothing else (ruling 1 of the pow2 doc; the cascade
  was that pool's only writer).
- The natural pool at an odd-band cell is the ZTURN-T grammar times its
  widths beside the chain3 and pair arms that enumerate there today. The
  pool sunset policy applies after the band is banked: whichever arm never
  wins a cell leaves the enumerator. (At pow2 the owner ruled ZTURN-T alone
  from 2048; that ruling was measured, and this one will be.)
- The prime engines' inner transforms at a composite M in the band take a
  ZTURN-T plan from the banked row, as the pow2 inner does today.
- Threaded requests (T > 1) at an odd-band cell: the cascade's `zt_mt` arm
  is the only engine today. After the deletion there is none until
  ZTURN-T's MT arm exists — the state contracts section 5 already declares
  for T > 1 at pow2 (refusal allowed for a missing engine, never a fallback).
  The owner accepts that consequence with the deletion or orders the MT arm
  first.

## Gates

1. **Exact result.** Natural: against the flat DIT's natural result at the
   same N, to a few ulps times log N. Plain: `out[i] == nat[perm[i]]`
   against the natural ZTURN-T result on the same chain.
2. **Matched roundtrip**, out of place and `zin == zout`, every chain in the
   grammar at every band size, every legal width.
3. **Tile invariance**, bitwise across widths, both directions.
4. **Staged equals fused.** At every pow2 registry cell the staged walk over
   the same chain and width is bitwise the fused driver, both classes, both
   directions. This is the gate that keeps the C walk and the emitter the
   same program.
5. **Cross-alignment invariance**, destination offsets 0, 16, 32, 48 bytes.
6. **Speed.** At the spike cell (12288, both classes, both placements) the
   staged ZTURN-T must beat the cascade's banked row on core 2 with the
   house protocol, and the in-place natural must close most of its 3.4x gap
   to the out-of-place cell. If it does not, the kernels are examined; the
   band is not in question.

## The cascade deletion (after the band is banked)

Inventory, from the sunset plan's section 1, verified 2026-09-14:

- runtime: `src/core/oop/zturn.h`, `zsplit.h`, `zturn_mt.h`,
  `cascade_calibrate.h`; the `_k1z_*` replay/race paths in `k1_commit.h`;
  the cascade arms in `c2c_oop_create.h` (three doors) and
  `c2c_ip_create.h` (one), `il2p.h`, `il_prime.h`, `oop_plan.h`
  (`VFFT_K1_IL_CASCADE`), `vfft.c`, `vfft_execute.h`, `vfft_internal.h`,
  `wisdom2_oop.h` (the `zt_*`, `zs_*`, `mode=zcasc` tokens),
  `dp_planner_il.h` (`_il_dp_push_cascade_chain`, the {4,8} mask walk;
  `_il_dp_enumerate_odd_mids` moves, it does not die).
- codelets (`codelets/zil/avx2/boundary_split/`), the families referenced
  from `zturn.h`, `zsplit.h` and `zturn_mt.h` only (grep of 2026-09-14):
  `s0s`, `s0t`/`s0tu`, `msg`/`msg_bwd` at every radix (the odd ones too),
  `msd`, `dts`/`dtsn`/`dtso`/`dtt`, `stf`/`stf2`/`stfu`/`stf2u`, `stfn`,
  `stfl`/`stfnl`, `sterm`/`sterm2`, `sink`, the `zp-r0` r8-ingest variants;
  their corpus rows and `gen_main.ml` flags; the corpus law restamped.
  NOT `msz`/`mszt`: those are `il2p.h`'s mids (the pair and chain3
  engines) and stay.
- store: every `eng=zturn`, `eng=stride mode=zcasc`, `eng=classic
  route=modeb` row (the seeded legacy rows included); the six odd-band
  natural in-place signposts and their comp targets.
- gates: `nat_bankloss_gate`, `pool_preserve_gate`, `vfft_ilp_front_gate`,
  `vfft_natural_front_gate` restamped; `zturn_dit_pipe_gate`,
  `zturn_r8_gate` deleted; `il_dp_overflow_gate` and `k1_pow2_gate`
  restamped for the odd cells.
- docs: `zcascade_sunset_plan.md` and `TODO_zcascade.md` close;
  `design_contracts.md` section 4's odd row and section 5's odd bullet
  become past tense.

The deletion is one change after the last banked verdict in the band, not
a series: a half-deleted cascade serves nothing and gates nothing.

## What it measures (2026-09-14/15, `probes/ZT/zt_odd_spike_results.md`)

Staged ZTURN-T against what the front door served before it, ns, medians of
5 paced races on core 2:

| N | cell | ztt natural | ztt plain | door natural (chain3 / cascade) | door scrambled (cascade) |
| --- | --- | --- | --- | --- | --- |
| 3072 | OOP fwd / IP fwd | 2891 / 3279 | 3466 / 2989 | 3192 / 3177 | 2969 / 3045 |
| 12288 | OOP fwd / IP fwd | 13156 / 13875 | 15862 / 14000 | 17488 / 17462 | 15931 / 15744 |
| 245760 | OOP fwd / IP bwd | 1002 us / 1152 us | 832 us / 749 us | 1198 us / 1706 us | 933 us / 991 us |

The natural class beats the door's natural engine at every cell and placement
(9-25% forward, 23-47% backward); the plain class wins the scrambled cell in
place everywhere and out of place above L2, and trails the cascade's comb out
of place at L2 sizes by up to 17% forward — the pow2 class's known forward
weakness, now against a scrambled incumbent. The planner's own race at 3072
(`benches/il_dp_odd_probe.c`, 248 natural candidates of which 52 ZTURN-T, 52
plain) puts `8.4.4.3.8` untiled at 2720 ns natural and `4.3.4.4.4.4` at 16 KB
at 3201 ns plain; chain3's best there is ~4000. Staging costs nothing
measurable: the spread at 12288 is 0.2%.

**Against MKL** (2026-09-15, `bench_1d_vs_mkl --k1noop` / `--k1nat`, one
process per cell, cold store): 1.73-1.94x out of place and 1.47-1.98x in
place at 3072, 6144, 12288, 24576, 61440 and 245760, errors 4-6e-16. The
first 245760 reading (0.24x) was a planner defect, fixed the same day: the
chain3 enumerator pushed kernel-less divisor splits that overflowed the
candidate cap, the cell was refused and Bluestein served; kernels that do
not exist are no longer candidates and ZTURN-T enumerates first.

## Build order

1. **Kernels — DONE 2026-09-14.** The generator gate admits `tmg`/`tmgb`/
   `tmgd` at 3/5/7/9/15; 15 files in `boundary_split/`; corpus 118/118
   byte-identical; `il_registry_avx2.h` lists the odd radices for the three
   kinds.
2. **Create + staged executor — DONE 2026-09-14.** `_ztt_create(N, chain,
   nf, scr, force_staged)` behind the two public creates: the grammar
   (`_ztt_radix_ok`), the stage table (`st_fwd/st_bwd`, `tl_*_plane`,
   `twoff`) resolved from `il_registry_avx2.h`'s lists, the twiddle fill's
   odd-modulus branch (libm, reduced angle), the tile law as divisibility
   (`vfft_ztt_tile_legal[_ord]`), `vfft_ztt_odd_band`, `_ztt_staged_run`.
   Gate 4 holds at every pow2 registry cell: 6544 executions bitwise.
3. **Spike — DONE 2026-09-14.** `benches/ztt_odd_gate.c` (17 odd chains:
   exact to 6e-16, in place / tiles / alignment bitwise, roundtrips 2e-15)
   and `probes/ZT/zt_odd_spike.c` (the table above).
4. **Planner, doors, wisdom — DONE 2026-09-15.** `_il_dp_enumerate_ztt_odd`
   (the grammar times the odd ladder) in the natural pool beside chain3 and
   the pairs and ALONE in the scrambled pool; `_k1_il_plan_race` admits the
   band to the ceiling; the no-row guard covers it; every door's cascade arm
   is fenced by both bands. The stale odd-band cascade rows (six `mode=zcasc`
   in-place signposts, six `eng=zturn` comp targets, four seeded `route=modeb`
   rows) and the pre-ruling `ord=scr il_route=2p` rows at 32..512 and 2048
   with their in-place signposts are purged from the store. Verified cold
   through the front door (`ztt_odd_gate --wisdir`): at 3072 and 12288 every
   (placement, order) cell builds, computes and banks ZTURN-T odd chains
   (the natural 3072 replays its banked chain3 row, as the law says).
5. **Tree gates + contract text — DONE 2026-09-15.** `ztt_odd_gate` in the
   sweep (`run_gates.py`, seeded); `ztt_gate`, `k1_pow2_gate`,
   `il_dp_overflow_gate` unchanged and green; `include/vfft.h` names the
   band.
6. **Calibrate the band** (with the pow2 `ord=scr` rows: one calibrator
   run, owner's word). Until then a cold band cell races at create.
7. **ZTURN-T's threaded arm** (owner, 2026-09-15: "we will make zturn-t
   threaded too") — its own design doc; the cascade's `zt_mt` sectioned walk
   is the last cell shape it serves in either band.
8. **Delete the cascade whole — DONE 2026-09-15** (owner: "you can start
   deleting the old cascade"; the order in 6-7 was set aside for it). Gone:
   `zturn.h`, `zsplit.h`, `zturn_mt.h`, `cascade_calibrate.h`,
   `oop_width_gate.h`; every door arm, replay, race, execute dispatch,
   plan field (`zsplit`, `zroute`, `zturn`, `zt_mt`) and fingerprint token;
   the planner's cascade route (build, exec, joint metric, bin map, chain
   pusher, odd-mid enumerator, banker, top-K diversity); the wisdom entry's
   cascade fields, the kind-4 reader/writer/migration and the `zt_*` tokens
   and env laws; the public `vfft_zt_mt_passes`; the benches and gates that
   were the cascade's; 64 codelets and their corpus rows (the boundary
   corpus is 54/54); `il_registry_avx2.h` regenerated. The house 64-B
   allocator `VFFT_ZS_ALLOC/FREE` moved to `support/zalloc.h`. The full
   gate sweep on the cascade-free tree: 26 of 26 pass. Kept on
   purpose: the retired enum slots `VFFT_K1_IL_CASCADE`,
   `VFFT_OOP_KIND_ZSPLIT`, `VFFT_NAT_ZCASC` and the `zcasc` mode name
   (persisted numbering; a legacy kind-4 line is skipped), the wisdom
   selftest's opaque token fixtures, and the generator's dead OCaml for the
   deleted kinds (`cascade_z.ml` emitters, `codelet.ml` constructors,
   `gen_main.ml` flags) — the last is the one cleanup left. The threading
   method is recorded in `cascade_mt_method.md` for ZTURN-T's MT arm.

## Rulings (owner, 2026-09-15)

0. **The library is clean, not a history.** Superseded engines are deleted
   whole with their codelets and wirings once their replacement serves every
   cell they served; the cascade goes after ZTURN-T's threaded arm exists.
1. **The cascade is not kept for its one remaining edge.** Out of place,
   scrambled, forward, at L2-resident sizes (2048..~65536) the cascade's comb
   beats the plain class by 0-8% while losing the same cell's backward by
   20-24% and every other cell outright. The pow2 band retired the cascade
   with the identical weakness; the odd band does the same. The fix, if
   any, is in the plain class's out-of-place forward (the pow2 doc's open
   item), never a cascade arm.

## Assumptions stated for the owner's ruling

- The band's odd set is the cascade's {3, 5, 7, 9, 15}; no new odd radix.
- (Ruled 2026-09-14.) The odd-cell ladder is 8, 16, 32 and 48 KB plus
  untiled, each width admitted where it divides N and holds the first mid's
  group; the pow2 ladder is untouched.
- The natural pool at odd-band cells keeps chain3 and the pairs until the
  banked verdicts sunset them.
- The threaded arm at odd-band cells goes with the cascade; its ZTURN-T
  replacement is separate work.
