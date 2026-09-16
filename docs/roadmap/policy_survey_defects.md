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

These are bugs independent of any refactor. The caller sets
`cfg.recalibrate = 1` meaning "ignore what is banked, race it again", and
these paths replay anyway.

| where | what | state |
| --- | --- | --- |
| `transforms/fftnd/fftnd_create.h:191` | rank-3 SPLIT c2c: `vw2_3d_lookup` replays the banked row with no guard, and the second-level fallback at `:196` is unguarded too | checker CONFIRMED end to end, including that a guard there genuinely re-measures AND overwrites (`vw2_3d_bank_entry` stamps `LAY_ANY`, `vw2__bank_pinned` replaces on key equality) |
| `transforms/fftnd/fftnd_r2c.h:538` | 3D and 4D SPLIT r2c/c2r: `vfft_adopt_lookup("nd", ...)` replays a MEASURED A/B verdict (timed arms, 5% hysteresis) with no guard | survey; checker did not refute |
| `transforms/real/c2r_dispatch.h:350` | the per-cell measured NATURAL/STRIDE verdict is read from `c2r_path.txt` with no guard | survey; checker did not refute |
| `vfft.c:506`, and its c2r twin at `:585` | the banked route row IS correctly invisible under the flag, but the next step then fires on `!may_race` and returns the STRUCTURAL THRESHOLD DEFAULT, racing nothing | confirmed for the MEASURE cell; one of the survey's two named cells was refuted as vacuous (`vw2_real_route_bank` is only reachable with `bK <= 64`, so no r2c K=128 route row can exist) |
| `oop/k1_commit.h:209` | the prime engine's ZTURN-T inner reads the ord=scr K=1 row at M with no recalibrate term, while the same function honors the flag two hundred lines later | survey; checker did not refute |
| `transforms/fftnd/fftnd_il.h:1128` | `_ilnd_build_flat` copies rigor, wisdom and wisdom_write into the child config but NOT `recalibrate`; its twin `_ilnd_build_child` sets it 28 lines earlier | checker UPGRADED from the survey's reading: reachable on EVERY recalibrate=1 3D IL create, not only when the flat arm wins |
| `transforms/fft2d/fft2d_create.h:92` | `ic = *cfg` propagates the flag into the plane-queue's child config; `:147` clears `wisdom_write` but not `recalibrate`, so each of the T clones re-races everything instead of replaying what the primary just banked | found by the checker, missed by the survey; contradicts the design comment at `:127-131` |

## B. Order-class mismatches — a reader and a writer that disagree

Order is a contract the caller chooses. Where a writer banks under one class
and the reader that should serve it asks for the other, the verdict lands on
a row nobody reads (a calibration that never sticks) or the wrong class is
served (a contract violation).

**Verified in the source, not merely surveyed:**

- **`transforms/fft2d/plane_queue.h:157`** — `_pq_row_key` computes
  `nat` on the line above and then hardcodes `VW2_ORD_SCR` for the IL row
  anyway, while the 2D IL tier keys its rows by `vfft_policy_ord_rankn`. For
  a natural cell the plane-queue's MT verdict has nowhere to live: case 0
  looks at a row the tier never banks, and case 1 wants a `lay=ANY` split row
  an IL caller never writes. The stale comment beside it ("the IL rows
  (lay=il, ord=scr)") is what it was written from.
- **`oop/k1_commit.h:1053`** — `_k1_il_mono_candidate` takes no `cfg` and
  reads the ord=nat row unconditionally; `oop/c2c_ip_create.h:231` calls it
  for every in-place request, including an explicit SCRAMBLED one, three
  lines after the writer-band refusal fires. Same shape as the 2026-07-29 /
  2026-09-07 defects. OPEN QUESTION before this is called a contract
  violation: whether the solo kernel's output order makes the served
  spectrum actually wrong, or only the row choice.
- **`transforms/fftnd/fftnd_il.h:1302`** — the comment three lines above
  states the axis-0 pass is the SCRAMBLED class for both order classes ("the
  natural class orders planes in its plane pass, never in the column pass"),
  and `nat_req = 0` honors that. But the same call hands `key0.ord`
  (= natural, correctly, for the ROW KEY) to `_il2d_col_build`, which uses it
  at `il2d_tier.h:1872` to choose **which pass the chain race times**. For a
  natural 3D cell the race therefore times a pass the tier never runs. One
  field is answering two different questions and the 3D tier needs them to
  differ.

## C. Needs a ruling, not a fix

`vfft.c:1278` — the K>1 transform-contiguous batch derives its order class
with the rank>=2 spelling on a cell that is keyed rank 1. At DEFAULT order
and a power of two the two laws disagree: the rank-1 law says natural, this
line says scrambled. Pointing it at `vfft_policy_ord_k1` would send every
DEFAULT-order batch cell to a different wisdom row and change its plans;
pointing it at `vfft_policy_ord_rankn` preserves today's behaviour exactly
but names the wrong rank. Nobody has measured which is right.

## D. Claimed but not yet verified

Worth a look; recorded so they are not re-derived, not acted on.

- `transforms/fft2d/il2d_tier.h:1404` / `:1429` — the Bluestein inner
  M-chain hardcodes `VW2_ORD_SCR` for both its lookup and its bank, and races
  with `nat=0`, from a natural request. The checker confirmed the path is
  reachable but downgraded the severity, on the grounds that the row
  describes the length-M inner pass rather than the user's cell. Separately,
  its bank at `:1429` (and the second half at `:2418`) passes `cmt=-1
  cmtt=-1` with a positive time, which replaces the record — so it can wipe
  the column-MT verdict of an unrelated user cell that happens to be (M, N2).
- `wisdom2/wisdom2_oop_reader.h:302` — `vw2__oop_find_k1_bwd` filters on t,
  rank, n[0], dir, role, lay and eng, and never on `key.ord`, while its
  writer stamps `VW2_ORD_NAT`. The asymmetry is live; the consequence is not
  established.
- `oop/k1_commit.h:1111`, `:1138`, `:1094` — `_bank_natoop_1d`,
  `_bank_scrmode_oop_1d` and `_bank_nat_raced` have zero callers anywhere in
  `src/` or `benches/`. Two stride row families therefore have neither a live
  writer nor a live reader.
- `transforms/fft2d/transpose.h:73-77` — bakes `TP_L1_BYTES` / `TP_L2_BYTES`
  and picks its recursion body from them, while `cpu_cache.h` reports 1.5-2x
  more. Two disagreeing cache authorities.
- `vfft.c:2164` — the four-step has no `FP__P(k1fs)` in the fingerprint's
  subplan presence bitmap and no detail line: it is invisible to the
  fingerprint.
- `oop/c2c_oop_create.h:508` vs `:560` — the MONO route is validated with
  form 0 while the handle resolves the row's BANKED form; a row carrying
  `il_route=MONO il_kv=1` at N != 64 would build NULL pointers that
  `vfft_execute.h:1092` calls unguarded. Latent: the planner refuses to bank
  that form.
