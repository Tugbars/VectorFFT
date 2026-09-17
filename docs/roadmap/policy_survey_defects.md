# Open defects found by the planning-policy surveys (2026-09-16)

The policy migration's surveys read far more of the tree than the migration
touched. What they found that is NOT migration work is listed here, with the
owner's ruling attached. Every entry was produced by a survey and then
attacked by an independent checker; the state column says how far each one
got, and nothing here is presented as verified unless it says so.

The two migrations these surveys were costing — unifying the `recalibrate`
check on one shape, and moving the banking side onto the order policy — were
both **declined by the owner on 2026-09-16** as not worth their touch
surface. See `planning_policy_design.md` for why. This file is the part that
survived that decision.

## A. `recalibrate` is not honored at all — owner ruled: FIX

The caller sets `cfg.recalibrate = 1` meaning "ignore what is banked, race it
again". These paths replayed anyway.

A guard is only a fix where the miss-path RE-DERIVES AND OVERWRITES. Two of
the four sites first listed here do neither: skipping the row hands them a
heuristic, not a measurement, so they are moved to section A2.

### FIXED 2026-09-16

**rank-3 SPLIT c2c** — `transforms/fftnd/fftnd_create.h:189` now hides the
wisdom2 row under the flag, and `wisdom2/wisdom2_fftnd.h`'s
`vfft_fft3d_plan_create_wisdom` takes a `recalib` argument that hides the
SECOND replay, the legacy in-process table. Guarding one and not the other
would have left the stale plan serving. The greedy path below both re-derives
and `vfft_fft3d_wisdom_put` replaces on `(N1,N2,N3)`, so the flag re-derives
AND overwrites.

**rank-3 / rank-4 SPLIT r2c and c2r** — `transforms/fftnd/fftnd_r2c.h:538`,
the last-dim row engine's adopt verdict, which is a MEASURED A/B (eight timed
reps per arm, 5% hysteresis). `stride_plan_nd_r2c` and its `_il` twin now take
`recalib`; the three call sites pass `cfg->recalibrate`.

**the adopt table could never be updated at all** —
`planning/adopt_wisdom.h:113` ended its tmp+rename with a bare `rename()`,
which does not replace an existing file on Windows. Every update after the
first write was silently lost: the measured verdict landed in
`strided_adopt.wis.tmp` and the stale row kept serving. PRE-EXISTING and
independent of the flag; found because the r2c fix above is inert without it.
The store's own writer has always done this properly
(`vw2__replace_file`, `MOVEFILE_REPLACE_EXISTING`). Now: atomic rename first
so POSIX keeps it, remove+rename as the fallback, no orphan `.tmp`.

Evidence, `benches/recal_nd_probe.exe` (warm store, same cell, flag off then
on):

| cell | flag off | flag on |
| --- | --- | --- |
| 48x48x48 split c2c | 1.02 ms (replay) | 3073.34 ms (re-derive) |
| 128x128x128 split r2c | 9.11 ms, poisoned verdict KEPT | 93.14 ms, poisoned verdict OVERWRITTEN with the measured one |

The r2c row was deliberately poisoned to `nd 16384 128 4 0 0` before each
run, so "overwritten" is the proof and the latency is only corroboration.

### A2. Not a guard — the miss-path is a heuristic, not a race

These two were mis-filed as one-line guards. Adding a guard would make
`recalibrate` DEGRADE the plan instead of re-measuring it, which is worse
than the bug.

| where | what a naive guard would do | what it needs |
| --- | --- | --- |
| ~~`transforms/real/c2r_dispatch.h:350`~~ | **VACUOUS, closed 2026-09-16**, and the survey's framing was wrong twice. (1) The race EXISTS: `_c2r_race_arms` in `real_route_race.h`, an alternating-order median-of-9 A/B with hysteresis, is step 4 of `_c2r_route_decide`. (2) `vfft_c2r_disp_create_auto` has exactly ONE caller — `vfft.c:591`, step 3, the "cannot race" fallback — and `vw2_real_route_bank` runs only behind `may_race`, so on that path no wisdom2 route row can exist for the flag to hide. Both arms are SPLIT-library engines (`VFFT_C2R_NATURAL` names the stage-0 natural initiator feeding split re/im through the packed cascade, NOT output order and NOT the interleaved library), so the bakeoff never crossed the library line. | closed |
| `transforms/real/c2r_dispatch.h:307` | RESIDUE, not a recalibrate bug: `c2r_path.txt` is a SECOND, offline-seeded authority for a choice the bakeoff also makes, consulted only where the bakeoff is not allowed to run. Nothing refreshes it. Worth a sunset ruling | open, low |
| ~~`oop/k1_commit.h:209`~~ | **FIXED 2026-09-18** by the owner's ruling ("prime cells should have their own inner race", `ilprime_inner_race_design.md`): the borrowed inner (M's row, the `M > 4096` cliff, the structural pair) is deleted; the prime cell races every (method, inner) pair on the whole convolution, banks `in= in_sh= in_tw=` on its own row, and `recalibrate` re-races it -- no nested planner call, so the lock is never touched. Gate `ilprime_inner_gate`, watched to fail | closed |

(`vfft.c:506` and its c2r twin were filed here too, and are now CLOSED as
vacuous — see the table below.)

A note for anyone writing a probe: `VFFT_MEASURE` is the CHEAPEST rigor tier,
production uses `VFFT_PATIENT`, and several races are gated on
`rigor != VFFT_MEASURE`. A probe that sets MEASURE has those windows SHUT and
will not reproduce what a real caller sees.

**the flag reached the wrong children, in both directions** — two one-line
fixes, 2026-09-16:

- `transforms/fftnd/fftnd_il.h` — `_ilnd_build_flat` copied rigor, wisdom and
  wisdom_write into the axis-2 child config but NOT `recalibrate`, while its
  twin `_ilnd_build_child` had always set it. So a recalibrate 3D IL create
  re-raced the parent and replayed the child.
- `transforms/fft2d/fft2d_create.h` — the opposite. `ic = *cfg` carries the
  flag into the plane-queue's child config and `:147` cleared `wisdom_write`
  but not `recalibrate`, so each of the T clones re-raced the cell the
  PRIMARY had just banked. The primary at `:98` still keeps the flag, which
  is correct; only the clones are cleared.

Evidence, warm store, same cell, flag off then on:

| cell | flag off | flag on |
| --- | --- | --- |
| 32x32x32 IL 3D c2c | 0 child race lines, 0.11 ms | 2 child race lines, 901.40 ms |

and for the clones, a CONTROL build with the one line reverted, 512x512 K=4
T=4 with the flag on, counting race lines:

| | axis race | chain race | colmt race | pq race |
| --- | --- | --- | --- | --- |
| clones inherit the flag (the bug) | 5 | 5 | 1 | 1 |
| clones cleared (the fix) | 1 | 1 | 1 | 1 |

5 = the primary plus four clones. The control was necessary: "each line
appears once" on its own is equally consistent with the clones never having
raced, which would have made the fix a no-op.

### Still open in this section
| ~~`vfft.c:506`, c2r twin `:585`~~ | **VACUOUS, closed 2026-09-16.** The claim was that the flag hides the row and the next step returns the threshold default. It cannot bite. `may_race` is `rigor != VFFT_MEASURE && (N % 2) == 0 && bK > 1 && bK <= 64` (c2r: `bK <= 128`), and in each failing case there is no row to lose: the race is the only writer, so odd N and K above the window have none, and K=1 is explicitly unbankable ("nowhere legal to bank" — q=1 real cells are the interleaved zr2c verdicts'). The rigor term never blocks in practice because production uses PATIENT (owner, 2026-09-16), and it is not inverted: VFFT_MEASURE is the CHEAP tier, as every other rigor test in the tree confirms (`budget 0.02 vs 0.05`, `trials 2 vs 3`, `RR 31 vs 81`). | closed |
| `transforms/fft2d/fft2d_r2c.h:988` | the rank-2 twin of the adopt verdict | **FIXED 2026-09-17.** `stride_plan_2d_r2c_from` took a `recalib` argument. Only the FRONT DOOR forwards it (`vfft.c`, where `_build_2d` already had the flag in its signature); the two calibrators and the two plan-from-entry rebuilds pass 0 deliberately -- they build candidate plans for an OUTER race, and re-measuring a sub-decision per candidate would both cost an A/B per arm and make that race unfair. Proof (poisoned row `2d 128 128 4 0 0`, warm): flag off 0.20 ms, poison KEPT; flag on 74.0 s, poison OVERWRITTEN with the measured `1 1` |

## B. Order-class mismatches — a reader and a writer that disagree

Order is a contract the caller chooses. Where a writer banks under one class
and the reader that should serve it asks for the other, the verdict lands on
a row nobody reads (a calibration that never sticks) or the wrong class is
served (a contract violation).

**Verified in the source, not merely surveyed:**

- **`transforms/fft2d/plane_queue.h:157` — FIXED 2026-09-17** by the owner's
  ruling that both order classes SHARE the plane-queue verdict: the key
  walker now tries the IL row under `scr` then `nat`, in that fixed order
  for both the replay and the bank, so exactly one verdict is ever consulted
  or written, and a natural-only store banks it instead of re-racing forever.
- ~~**`transforms/fft2d/plane_queue.h:157`**~~ — `_pq_row_key` computes
  `nat` on the line above and then hardcodes `VW2_ORD_SCR` for the IL row
  anyway, while the 2D IL tier keys its rows by `vfft_policy_ord_rankn`. For
  a natural cell the plane-queue's MT verdict has nowhere to live: case 0
  looks at a row the tier never banks, and case 1 wants a `lay=ANY` split row
  an IL caller never writes. The stale comment beside it ("the IL rows
  (lay=il, ord=scr)") is what it was written from.
- **`oop/k1_commit.h:1053` — FIXED 2026-09-17**: the mono candidate takes
  `cfg` and reads the row of the request's order class, the same law
  `_k1_il_candidate` asks with. The refusal it could cause -- a scrambled
  in-place cell whose own race banked MONO finds no MONO on the natural row,
  builds nothing, is refused -- was NOT reproduced: at N = 16/32/64 the
  scrambled race picked ZTURN-T (the owner: "scrambled wins in-place
  usually"), and the shipped store has zero scrambled rows (of 288) naming
  mono. The misread was real; its trigger is latent.
- ~~**`oop/k1_commit.h:1053`**~~ — `_k1_il_mono_candidate` takes no `cfg` and
  reads the ord=nat row unconditionally; `oop/c2c_ip_create.h:231` calls it
  for every in-place request, including an explicit SCRAMBLED one, three
  lines after the writer-band refusal fires. Same shape as the 2026-07-29 /
  2026-09-07 defects. OPEN QUESTION before this is called a contract
  violation: whether the solo kernel's output order makes the served
  spectrum actually wrong, or only the row choice.
- **`transforms/fftnd/fftnd_il.h:1302` — FIXED 2026-09-17.** The one-token
  change is in `il2d_tier.h`: the chain race now takes `nat_req` (which pass
  this build will RUN) instead of `key->ord == VW2_ORD_NAT` (which order cell
  the ROW belongs to). The two are equal at the 2D tier and at the 3D tier's
  axis 1, and differ only at the 3D tier's axis 0, which is the scrambled
  class for both order classes by design. Proof by CONTROL BUILD, same cold
  48x48x48 natural cell: with the bug the axis-0 race logs
  `chain race 48x2304 (nat)`, with the fix `chain race 48x2304 (scr)` -- the
  pass the tier actually runs.
  WITHDRAWN: the prediction that this would also enlarge the candidate pool
  (chains with no natural leaf are excluded when `nat` is set). Both builds
  raced 5 arms at this cell, so the exclusion path did not fire here. It
  exists in the code; it was not observed.
  The banked axis-0 chains for natural 3D cells were chosen under the old
  measurement and are worth re-racing.

- ~~**`transforms/fftnd/fftnd_il.h:1302`**~~ — the comment three lines above
  states the axis-0 pass is the SCRAMBLED class for both order classes ("the
  natural class orders planes in its plane pass, never in the column pass"),
  and `nat_req = 0` honors that. But the same call hands `key0.ord`
  (= natural, correctly, for the ROW KEY) to `_il2d_col_build`, which uses it
  at `il2d_tier.h:1872` to choose **which pass the chain race times**. For a
  natural 3D cell the race therefore times a pass the tier never runs. One
  field is answering two different questions and the 3D tier needs them to
  differ.

## C. Needs a ruling, not a fix

**FIXED 2026-09-17** (owner: "Default should be natural"): `_tc_mt_decide`'s
one `ord` local -- it feeds both the lookup and the bank -- now comes from
`vfft_policy_ord_k1(cfg, N, inplace)`, the rank-1 law. DEFAULT batch cells
re-measure their threading verdict once, on the natural row.

~~`vfft.c:1278`~~ — the K>1 transform-contiguous batch derives its order class
with the rank>=2 spelling on a cell that is keyed rank 1. At DEFAULT order
and a power of two the two laws disagree: the rank-1 law says natural, this
line says scrambled. Pointing it at `vfft_policy_ord_k1` would send every
DEFAULT-order batch cell to a different wisdom row and change its plans;
pointing it at `vfft_policy_ord_rankn` preserves today's behaviour exactly
but names the wrong rank. Nobody has measured which is right.

## D. Claimed, then OPENED AND RULED (2026-09-18)

Every claim was read in source on 2026-09-18 and ruled by the owner the same
day. Five were real and are FIXED; one is real and benign.

| # | claim | verdict |
| --- | --- | --- |
| D1 | the 2D Bluestein inner's M-chain hardcodes `VW2_ORD_SCR` and banks on the (M, N2) row | **REAL, FIXED.** The order is right (a convolution is a matched roundtrip in any order, the prime cell's own reasoning); the ROW was not. The bank passed a positive time with every axis verdict at -1 -- the REPLACE path -- so a user's own scrambled M x N2 cell lost its width, fusion, row-route and column-MT verdicts and re-raced them on its next create. Now the inner's chain and `blu=M` bank on THE CELL'S OWN row (`_il2d_blu_ctx.key`), its per-stage forms under a separate token `bluforms=` on that row (so they can never overwrite the cell's own `forms=`), and a replay is handed the chain the builder already read rather than re-reading a row of its own. A speculative N-arm arm banks nothing (`commit`). Gate `il2d_blu_row_gate`, watched to fail |
| D2 | `vw2__oop_find_k1_bwd` never filters on `key.ord` | **REAL, BENIGN.** One writer, and it stamps `VW2_ORD_NAT` (`wisdom2_oop_reader.h:634`); the row is a backward kernel-variant verdict keyed by direction and layout, so only one can exist per N. It becomes a defect the day a scrambled backward verdict is banked. Left as is, recorded here |
| D3 | `_bank_natoop_1d`, `_bank_scrmode_oop_1d`, `_bank_nat_raced` have zero callers | **REAL, FIXED.** Confirmed by grep over `src/` and `benches/`; deleted whole (69 lines) with the header comment that described them. `_bank_nat_1d` is live (the in-place create). Leftover: `vw2_stride_bank_scrmode_oop` is now an orphan WRITER -- no caller -- but its reader still serves rows a shipped store carries, so it stays until a sunset ruling |
| D4 | `transpose.h` bakes `TP_L1_BYTES` / `TP_L2_BYTES` while `cpu_cache.h` reports 1.5-2x more | **REAL, FIXED.** 32 KB / 1 MB baked against 48 KB / 2 MB measured, so the recursion body was picked from the wrong regime; and in the `#ifndef default` spelling this tree has been bitten by before. `cpu_cache.h` is now the authority; `-DTP_L*_BYTES` still overrides for a named cross-target. Split-library only (the four-step has its own transpose) |
| D5 | the four-step is invisible to the fingerprint | **REAL, FIXED.** No `FP__P(k1fs)` bit and no detail line, so two four-step plans differing in their raced split hashed the same and a regression there was invisible to the refactor harness. Both added. NOTE: the `have=` bitmap is one digit wider and a new `k1fs=[...]` line exists, so baselines captured before 2026-09-18 do not compare -- re-capture with `capture_baseline.py` |
| D6 | MONO validated at form 0, resolved at the banked form | **REAL, FIXED.** Form 1 exists only at N = 64, so a row carrying `il_route=MONO il_kv=1` at any other N passed the form-0 check and resolved to NULL pointers that `vfft_execute.h:1092` calls unguarded. The validator now asks for the form it will resolve, in the resolution's own expression, both directions. Latent (the planner never banks that pair), so this guards a hand-edited or foreign store |

Evidence: `il2d_blu_row_gate` ALL PASS on the fix and, with the pre-2026-09-18
foreign-row bank injected, failing exactly as described -- the 64 x 64
scrambled owner row went from `chain=8.8 wl=32 tf=1 ro=0 sw=0 ns=3956.1` to
`chain=8.8 ns=2300.0`, and the 23 x 64 cell banked no row of its own. Fourteen
targeted gates green (`il2d_blu_row`, `il2d_onechain`, `ilnd`, `il2d_real`,
`ilprime_inner`, `il_solo`, `k1_fourstep`, `policy` = 131,571,790 checks,
`blocked_tail`, `zr2c_fd`, `odd_ct`, `k1_pow2`, `vfft_k1scr`, `api_matrix`).

## E. From the rank>=2 census (2026-09-17) -- verified by hand, and what became of them

The census of the 2D/3D interleaved tiers (145 policy-shaped rules, 61
claimed duplications) is a MAP, not evidence; only entries opened and read
are listed.

| finding | verified | state |
| --- | --- | --- |
| the column radix pool typed TWICE in `il2d_cols.h` (`_il2d_enum_rec` :129, `_il2d_build_chain` :372), and the two copies DISAGREE: depth 4 vs 8, and the greedy adds a remainder rule (`L/r == 1 \|\| L/r >= 4`) the race does not have | yes, both sites read | **CLOSED BY DELETION.** The owner ruled the greedy unacceptable; it is gone with its pool, so there is nothing left to unify. One-candidate cells now race (one arm) and bank; `il2d_onechain_gate` holds it |
| the no-silent-caps law: the enumerator's cap is logged at `il2d_tier.h:1417/:1864` and DISCARDED at `fft2d_create.h:573` (the real tier) and `k1_fourstep.h:561` (the four-step super-band) -- both declare `dropped`, pass it, never read it | yes, all four sites read | **FIXED 2026-09-17**: `_il2d_enum_rec` is now an entry that warns ONCE when its recursive body dropped candidates; the two caller-side warnings are gone and the two silent callers are covered by construction |
| `k1_fourstep.h:560` sizes `cand[24][8], cl[24]` with a literal while the enumerator fills up to `VFFT_IL2D_MAXCAND` (= 24 today) | yes | **FIXED 2026-09-17**: the macro moved to `planning/policy.h` (a pool cap is policy, and it is the one place ahead of every consumer -- `k1_fourstep.h` is included long before `il2d_cols.h`); the four-step sizes by it. `ilfd_probe.c`, the one standalone includer of `il2d_cols.h`, carries `policy_gate.c`'s include recipe |
| the Bluestein M-chain provider hook installed ONLY by the 2D create (`fft2d_create.h:64`); the 3D tier set the ctx and never the hook, so its prime-axis inner was greedy -- or raced, depending on whether a 2D create had run earlier in the process | yes | **FIXED 2026-09-17** (`fftnd_il.h` installs it) |
| the census's other ~57 claimed duplications (three width-ladder literals, the tcut law in six spellings, the form-axis rule in three, the real tier as a second copy of much of the c2c tier) | NOT verified | recorded as claims only |
| a 3D cell whose axis 0 is Bluestein never banks its own row; axis 1 and the structure verdict are refused and re-raced on every create -- and the shared builder had no replay-rebuild for a `blu` row, so banking alone would have served the M chain as the N chain | yes: found by `ilnd_gate`'s first run, traced in the tap (`bank refused (no row)`), the missing rebuild read in `_il2d_col_build` | **FIXED 2026-09-17**, `rank >= 3` only (the 2D create keeps its own rebuild) |

## F. Race quality, from the band-map check (2026-09-18)

Not a policy defect: every region banks the right method, and the shipped
verdicts replay at their recorded cost on today's tree. But a recalibration
is a single cold race, and at two cells it banked a verdict far worse than
the recorded one:

| where | re-raced verdict | recorded | cost |
| --- | --- | --- | --- |
| 15625, the flat DIT's forms | `t.m.t.t.o @ 625` | `t.t.t.t.o @ 3125` | 0.71 vs 1.06 vs MKL |
| 512x4096, the four-step's child at 2097152 | `chain=64.8 wl=0 sw=64` (strip form) | `chain=8.8.8 wl=8` (band) | 14.4 M ns vs 8.6 M |

Open: (1) whether the race body takes more than one cold sample per arm
before banking; (2) the child race under the four-step's recalibrating
create -- the strip arm won a race the DRAM-roof work says it loses, so its
timing there is the first suspect. `build_tuned/band_recal_check.sh`
reproduces the check; the outlier re-run is
`scratchpad/outliers.sh`'s shape (re-race + bench twice).
