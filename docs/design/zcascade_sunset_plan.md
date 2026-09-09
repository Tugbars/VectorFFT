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

## 1b. Where ZTURN-T goes in

Owner, 2026-09-09: "wire zt-t to 2D, 3D, basically everywhere where zcascade
was." Every consumer of the cascade, from the grep of 2026-09-09 (code and
comments; the 2D / 3D / real / trig tiers mention it only in comments):

| consumer | reaches the cascade how | ZTURN-T there |
| --- | --- | --- |
| K=1 natural OOP door (`c2c_oop_create.h` `[natorder]`) | races the K=1 plan vs the natord cascade | DONE: ZTURN-T is the K=1 plan and wins 2048..16384 (S1 promotes) |
| K=1 natural IN-PLACE door (`c2c_ip_create.h`, `@nat` writer) | races the K=1 plan (ZTURN-T bound to its `plane` drivers) vs the in-place natord cascade; the store's `place=ip` door rows at 2048..16384 still say `mode=zcasc` from an older restamp | S1b: restamp the in-place door 2048..16384 on aligned buffers and take the paced in-place verdict; expected to follow OOP but it is its own cell |
| K=1 SCRAMBLED (`_il_dp_enumerate`, `_exec_zcascade`) | the cascade's comb; natural engines are excluded from the pool at >= 2048 | S2: admit them; the race decides |
| K=1 threaded (`zturn_mt.h`, `zt_mt`) | the cascade's sectioned walk at >= 16384 | S5: an MT arm for ZTURN-T |
| Bluestein / Rader inner transform (`il_prime.h` `_ilprime_inner_make`) | `vfft_zturn2_create(M)` for pow2 M > 4096 — the comb is enough for a convolution (matched roundtrip) | S2b: an `il_prime` inner provider that fills a ZTURN-T plan from the banked `il_ztt=` row at M (natural fwd/bwd is a matched roundtrip too); the cascade inner stays only for M above ZTURN-T's ceiling until S4 |
| 2D / 3D IL tiers (`fft2d_create.h`, `il2d_tier.h`, `fftnd_il.h`) | NO direct use — their row / column / plane passes are their own engines; where they create an inner K=1 1D plan through `vfft_create`, that plan comes from the K=1 doors | nothing to wire: they inherit ZTURN-T through the doors once S1 / S1b / S2 serve it; verify with the 2D / 3D gates in the sweep |
| K > 1 batch IL tier (`il_kv`, `tcmt`; packed IL <= 2048, split >= 4096) | never used the cascade (K=1 only) | S7, a race not a replacement: a "ZTURN-T x K" candidate (the fused driver looped over the batch) in the K > 1 pool at >= 4096, against the split batch |
| real / trig (`rfft.h` "cascade", `c2r.h` "packed cascade") | a DIFFERENT cascade (the real transforms' own stage chain), not ZTURN-S | nothing |
| 2^a·odd K=1 (the odd mids) | the cascade's odd path | S6, owner's call |

## 2. Stages

S1 — **Promote** the 2026-09-09 verdict rows (4096 / 8192 / 16384: il_ztt +
il_tw, the split re-race, the scrambled cascade re-race, `mode=free` door
rows) after the full gate sweep. ZTURN-T then serves natural T=1 there.

S1b — **The in-place natural door** 2048..16384: restamp on aligned buffers
(the `@nat` in-place writer; ZTURN-T bound to its `plane` drivers vs the
in-place natord cascade), then the paced in-place verdict with MKL
in-process. Its own cell, its own row. Found while promoting S1 (the
`ztt_gate` cold front-door pass): `c2c_ip_create.h` builds the K=1 IL
candidate only when the cell's banked in-place door row is not `mode=zcasc`
(or N < 2048) — so with the legacy `place=ip mode=zcasc` rows at 2048..16384
the in-place create never even constructs ZTURN-T, and under
`VFFT_NO_NAT_ZCASC` it fails outright ("no interleaved engine"). The door
restamp is the fix; until the cascade goes, that guard also decides whether
ZTURN-T is a candidate in place at all.

S2 — **Scrambled pool admission at >= 2048.** `dp_planner_il.h`
`_il_dp_enumerate` admits the natural engines into the scrambled pool only
below 2048; admit them at every N (the pool's own law: "every engine that
legally answers a scrambled request competes"). Recount
`il_dp_overflow_gate`'s rows; calibrate ord=scr at 4096..16384; where
ZTURN-T wins, the scrambled cell serves natural output and says so.

S2b — **The prime path's inner transform** (`il_prime.h`): an inner
provider that fills a ZTURN-T plan from the banked `il_ztt=` (+ `il_tw=`)
row at the padded length M; the cascade inner stays only above ZTURN-T's
ceiling until S4. Gate: the prime gates bitwise-unchanged in output class
(natural roundtrip), and the prime cells re-raced.

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

S7 — **K > 1 at >= 4096** (not a cascade cell; the owner's "everywhere"):
a "ZTURN-T x K" candidate — the fused driver looped over the batch — in the
K > 1 IL pool beside the split batch, raced per (N, K). A race, never a
replacement.

Each stage: change, gate (fused == unfused / tiled == untiled bitwise where
it applies, the full `run_gates.py` sweep), race on a scratch store,
promote, then delete. No stage deletes what a banked row still serves.
