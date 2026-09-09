# Retiring the ZTURN-S cascade behind ZTURN-T — the sunset plan

**Owner direction (2026-09-09):** "the next step is to strip away old zcascade
method, and its own specific codelets carefully. zt-t is clearly a win here."
The verdict it rests on: `docs/performance/v1_0_results.md` §1 (K=1
INTERLEAVED natural order, 2026-09-09) — ZTURN-T beats the natord cascade at
4096 / 8192 / 16384 (tiled 3728 / 8115 / 18463 ns vs 4041 / 8731 / 18693) and
MKL (1.02 / 1.05 / 1.02), and at 1024 / 2048 below.

**The law of this sunset:** the cascade is stripped PER CONTRACT, after
ZTURN-T has a banked verdict for that contract — the pool sunset policy
(superseded arms deleted per slot after the re-race), never the engine at
once. The cells the cascade still serves alone are FEATURES, not levers
(`TODO_zcascade.md` item 6): stripping them without a ZTURN-T twin is a
feature regression, not a cleanup.

## 1. What the cascade owns today (inventory, 2026-09-09)

Runtime: `src/core/oop/zturn.h` (73 KB, the ZTURN-S engine: ingest s0s/s0t,
mids msg, terminators stf/stfl/stf2/stfn/stfnl/stfu, r0 = 8 twins, tcut
tiling `zt_tw`, natord, tform/t2q axes), `zturn_mt.h` (17 KB, the threaded
arm), `zsplit.h` (18 KB, the LEGACY block-split engine — still an arm in the
scrambled pool; serves NO cell in the shipped store: 0 rows vs 29 for zturn).
Callers outside its own files: `c2c_oop_create.h`, `c2c_ip_create.h`,
`k1_commit.h`, `il2p.h`, `il_prime.h`, `oop_plan.h`, `cascade_calibrate.h`,
`dp_planner_il.h`, `vfft.c`, `vfft_execute.h`, `vfft_internal.h`,
`wisdom2_oop.h`. The 2D / 3D / real / trig tiers call none of it directly.
Codelets (corpus `zp-` rows): s0s/s0t(+u), msg, stf/stfl/stf2(+u)/stfn/stfnl/
stfbn, sterm/sterm2, dts/dtsn/dtso/dtt, sink, msd, the 41 `zp-r0` r8-ingest
variants, and the ODD mids msz/mszt/mszb at radices 3 5 7 9 15 (the
2^a·odd cascade). ZTURN-T shares the emitter (`cascade_z.ml`) but only its
own kinds t0tp/tmg/tlf. Gates naming the cascade: il_dp_overflow_gate,
k1_pow2_gate, nat_bankloss_gate, pool_preserve_gate, vfft_ilp_front_gate,
vfft_natural_front_gate, zturn_dit_pipe_gate, zturn_r8_gate. Wisdom tokens:
`zt_tw zt_l1 zs_t2q zt_t2q zt_tf zt_ntf zt_mt ...`, `mode=zcasc` door rows.

Contracts the cascade serves ALONE today, and what ZTURN-T needs first:

| contract | cascade today | ZTURN-T twin needed |
| --- | --- | --- |
| K=1 natural, T=1, 2048..16384, both placements | LOST to ZTURN-T (banked) | none — served |
| K=1 natural, 32768..262144 | serves | the two-level create (`TODO_zcascade.md` item 2) + registry cells |
| K=1 SCRAMBLED, >= 2048 | serves (its comb skips the ordering) | admission to the ord=scr pool: natural output is a legal scrambled answer — a race, no new class |
| K=1 threaded (T > 1), >= 16384 | serves (raced `zt_mt`) | an MT arm (sections over groups per stage, `zturn_mt.h`'s shape) |
| K=1 2^a·odd | serves via the odd mids | odd radices in ZTURN-T's chain — a new axis; else the odd cascade STAYS |

## 2. Stages

S1 — **Promote** the 2026-09-09 verdict rows (4096 / 8192 / 16384: il_ztt +
il_tw, the split re-race, the scrambled cascade re-race, `mode=free` door
rows) after the full gate sweep. ZTURN-T then serves natural T=1 there.

S2 — **Scrambled pool admission at >= 2048.** `dp_planner_il.h`
`_il_dp_enumerate` admits the natural engines into the scrambled pool only
below 2048; admit them at every N (the pool's own law: "every engine that
legally answers a scrambled request competes"). Recount
`il_dp_overflow_gate`'s rows; calibrate ord=scr at 4096..16384; where
ZTURN-T wins, the scrambled cell serves natural output and says so.

S3 — **Strip the dead arm first: `zsplit.h`.** It wins nowhere (measured:
0 banked rows). Move `VFFT_ZS_ALLOC/FREE` (the tree's 64-B allocator, used by
the planner and both doors) and `_vfft_zs_brev/_vfft_zs_base` (used by
zturn.h) to their survivors, delete the engine, its pool arm, its dispatch
branch, its wisdom `eng=` value, its gate references. Full sweep.

S4 — **32768 and above:** the two-level create, cells to 262144, the tile
ladder, race natural + scrambled per cell. Then the cascade's natural path
has no cell left: strip it (s0s/s0t, msg, stf* natord terminators, r0
variants, tcut/tform/t2q axes, `mode=zcasc` doors, the natord machinery)
per slot after each cell's re-race.

S5 — **MT arm for ZTURN-T**, raced at T; then `zturn_mt.h` and `zt_mt` go.

S6 — **The odd cascade** (msz/mszt/mszb, `K=1 ODD cascade — N = 2^a·odd`):
owner's call — either odd radices for ZTURN-T or the odd cascade stays as
its own engine. Nothing of it is stripped by S1–S5.

Each stage: change, gate (fused == unfused / tiled == untiled bitwise where
it applies, the full `run_gates.py` sweep), race on a scratch store,
promote, then delete. No stage deletes what a banked row still serves.
