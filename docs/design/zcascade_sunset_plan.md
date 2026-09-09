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
ZTURN-T is a candidate in place at all. The in-place door's verdict on the
plain plane driver is the CASCADE at 2048..16384, and it is right: ZTURN-T
in place ran 25..31% over its out-of-place time. The cause and the remedy —
the in-place terminator kind `tlfi` (the terminator with its output stream
prefetched) plus the plane's per-call page offset — are
`zturn_t_ship_plan.md` §9. **S1b MEASURED 2026-09-09 with `tlfi`:** in place
ZTURN-T 1836 / 3931 / 8527 / 19562 ns vs the in-place cascade 1999 / 4069 /
8602 / 18548 vs MKL in place 2119 / 4016 / 8818 / 18986 at 2048 / 4096 /
8192 / 16384; the re-raced in-place door banked the ENGINE (`mode=ilp`) at
all four cells (16384 a tie cell on the door's clock). Promote after the
sweep. The cascade's in-place path leaves the served set at 2048..16384.

S2 — **Scrambled pool admission at >= 2048.** `dp_planner_il.h`
`_il_dp_enumerate` admits the natural engines into the scrambled pool only
below 2048; admit them at every N (the pool's own law: "every engine that
legally answers a scrambled request competes"). Recount
`il_dp_overflow_gate`'s rows; calibrate ord=scr at 4096..16384; where
ZTURN-T wins, the scrambled cell serves natural output and says so.
IN THE TREE 2026-09-09 (evening), with two findings from its first verdict:
(a) the pool admission alone changed nothing served — an explicit SCRAMBLED
pow2 request at N >= 2048 attached the kind-4 cascade by fiat in
`c2c_oop_create.h` before the K=1 admission ran, so the ord=scr K=1 row
(ZTURN-T at 2048..16384) was banked and never read; the scrambled door (the
DEFAULT-order one) now also runs for explicit SCRAMBLED when that row names
a K=1 engine, with the request's own kind-4 cascade as the arm, banking
mode=free|zcasc on the ord=scr mode row; (b) the planner's clock reads the
cascade candidates at ~2x their bench time in the scrambled pool (3960 /
8313 / 17748 / 38144 vs ~1900 / 4050 / 8600 / 18500 at 2048..16384) while
ZTURN-T's read true — uniform across chains, so the chain pick stands and
the door's own race decides engine vs engine, but the pool's cross-engine
ranking is not a verdict until this bias is found (probe queued). Census
rows: 2048 / 4096 / 8192 / 16384 = 126 / 175 / 255 / 352.

**S2 MEASURED (2026-09-09, 15:55, `probes/ZT/phaseE4_scr.final.txt`):** an
explicit SCRAMBLED request, the K=1 arm (ZTURN-T, natural output) vs the
cascade's comb, 7 paced runs each, arms verified distinct, MKL natural OOP
in every process; the scrambled door's own race on the same store agreed
at all four cells ("engine"):

| N | ZTURN-T natural | cascade comb | MKL | ZTURN-T ahead |
| --- | --- | --- | --- | --- |
| 2048 | 1657 ns | 1991 | 2148 | 20% |
| 4096 | 3661 | 3971 | 3817 | 8.5% |
| 8192 | 8109 | 8335 | 8395 | 2.8% |
| 16384 | 17243 | 17848 | 18516 | 3.5% |

The comb's 4..15% over the cascade's OWN natural terminator (same-day
passes: 1900 / 3915 / 8350 / 18000 vs 2237 / 4041..4100 / 8731 / 18693)
does not carry against ZTURN-T. **The bound on a scrambled ZTURN-T class**
(owner's question, "scrambled zt-t vs scrambled zcascade"): the same plans
with an IDENTITY run-base table — the ingest storing runs sequentially,
everything else unchanged, output invalid, timing exact — gain 0..4%
(2048 1597 vs 1593; 4096 3554 vs 3518; 8192 7826 vs 7598; 16384 17502 vs
17299; two repeats). That is the most such a class could recover at the
ingest, and it would need strided-run mids and a re-derived tile law to
exist. Not worth building for speed; the natural-writing ZTURN-T already
beats the comb. **The cascade wins no cell in 2048..16384** (natural OOP,
in place 2048..8192, scrambled); it keeps 32768+ (ZTURN-T's octave), the
threaded arm, the odd cascade, and the 16384 in-place tie.

S2b — **The prime path's inner transform** (`il_prime.h`): an inner
provider that fills a ZTURN-T plan from the banked `il_ztt=` (+ `il_tw=`)
row at the padded length M; the cascade inner stays only above ZTURN-T's
ceiling until S4. Gate: the prime gates bitwise-unchanged in output class
(natural roundtrip), and the prime cells re-raced.

**Order change (2026-09-09, 16:25):** S3 folds into the FINAL deletion —
the legacy arm's plumbing (the `zroute` axis, the `eng=` engine value, the
dispatcher, 13 files) is the cascade's plumbing, and removing it twice is
waste. The coverage stages come first, since they are what keeps the
cascade alive: S4 (32768..262144), S5 (the MT arm), S6 (the flat DIT into
the odd cells), then one deletion of the whole family.

**S4 IN FLIGHT (16:30):** `ztt.h` ceiling 262144 with the TWO-LEVEL create
(above the 16384 octave: table(a) x a per-stage fine table of <= 16 cos/sin
entries, one complex product per record); `ztt_drivers.ml` `max_n = 262144`
-> 223 cells / 892 drivers (driver TU 2.5 MB); plane 512 KB..4 MB, twiddle
streams up to 4 MB per direction at 262144 (the cascade's own footprint
there). Gates, calibration and the paced verdict follow; the natural door
rows at 32768+ (`mode=zcasc` at 32768, none above) re-race on the restamp.

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
its own engine. Nothing of it is stripped by S1–S5. **Owner 2026-09-09
(16:00): the cascade WILL be deleted**, so its odd cells need an engine: the
flat DIT (route 8) already serves every odd-factor N below 2048 and every
N % 4 != 0 above it; only a gate in `_il_dp_enumerate_natural_engines`
(`N < 2048 || (N & 3)`) keeps it out of the 2^a·odd, N % 4 == 0 cells at
>= 2048 that the odd cascade holds. Admit it there and race — no new
kernels; the odd cascade goes when the flat DIT's rows are banked.

**S2 REVERTED 2026-09-09 (evening), under the order law
(`design_contracts.md` section 3): a scrambled request races scrambled
writers only, so the natural engines leave the pow2 scrambled pools at
2048 and above (`_il_dp_enumerate`), the explicit-SCRAMBLED door takes the
cascade again (`c2c_oop_create.h`), and the promoted ord=scr K=1 rows plus
the scrambled mode rows at 2048..32768 are out of the store. The pow2
scrambled cell is the ZTURN-S cascade's comb until the scrambled ZTURN-T
class exists; the legacy zsplit engine is out of those pools for good
(never banked on any host). S2b (the prime inner) keys on the NATURAL row
at M — section 7 of the contracts. The record below stands as history.**

**S2 PROMOTED 2026-09-09 16:20** (selective: the ord=scr K=1 rows naming
ZTURN-T and the scrambled door rows `mode=free` at 2048..16384;
`probes/ZT/promote_scr_rows.py` — the same calibration's re-raced ord=nat
siblings were NOT taken, the paced verdict's rows stand). Sweep
`gates_after_s2.txt` 25/26: `flatdit_gate` failed once under the sweep and
passed on a direct run and on a runner re-run — a flap, recorded, not
attributed; the ZTURN-T gate's cold front-door pass on the promoted store is
green. `include/vfft.h` now states the SCRAMBLED contract (order-agnostic,
any self-consistent permutation, natural included, matched roundtrip the
only decode). S2b (the prime inner on the banked ord=scr verdict) shipped
with it, `vfft_ilp_front_gate` PASS.

**Decision record, 2026-09-09 (15:55..16:00).** The owner floated keeping
the cascade as the explicit scrambled engine; the pressure test above (the
comb loses to natural-writing ZTURN-T at every cell, a scrambled ZTURN-T
class bounded at 0..4%) settled it: the sunset stays on, the cascade will be
deleted, S2 completes as built (the scrambled cell is a race).

S7 — **K > 1 at >= 4096** (not a cascade cell; the owner's "everywhere"):
a "ZTURN-T x K" candidate — the fused driver looped over the batch — in the
K > 1 IL pool beside the split batch, raced per (N, K). A race, never a
replacement.

Each stage: change, gate (fused == unfused / tiled == untiled bitwise where
it applies, the full `run_gates.py` sweep), race on a scratch store,
promote, then delete. No stage deletes what a banked row still serves.
