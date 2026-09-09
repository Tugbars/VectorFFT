# To-do: the N >= 2048 cascade (ZTURN-S), after ZTURN-T shipped

**Status:** decided by the owner 2026-09-09 ("we'll fix that and also bring
some of the zt-t optimizations to zcascade"); none started. Each item names
what is measured and what is not. Everything here is the cascade's — ZTURN-T
(`oop/ztt.h`, 16..2048) is shipped and separately parked for its own
improvements.

## 1. The backward terminator's spills — the loaded-stream backward twin

**Measured.** Spill ops per innermost loop, served kernels (2026-09-09
census, `probes/ZT/census/`): `stf_r4` r8 fwd 7 / **bwd 33**; `stfn_r4` r8
**fwd 49 / bwd 30**; radix-4 forms 0. ZTURN-T's `tlf` 10/10, `tmg` 6/6.

**Mechanism.** Forward `stf` is pre-twiddle: the w¹ squaring tree runs with
two vectors live and each power dies at the lane it multiplies. Backward
`stf` is post-twiddle (the transposed pipeline: IDFT then conj-w¹): all 16
outputs are live when the 14-vector tree starts. Reordering inside that set
cannot win — store sinking (B1, 2026-07-27: 33 → 6 spills, **washed** in
time) and narrow stores (refuted twice) proved it.

**Fix.** Remove the tree: the loaded-stream form (TP_Flat, every power a
record consumed as a memory operand) already exists forward as `stfl` /
`stfnl`, raced per cell on the terminator-form axis (`zt_tf` / `zt_ntf`), and
measured **+25–33% at the L1-resident cells 512..4096** (`zturn.h`, the tform
comment). The backward twins are missing:

- `cascade_z.ml`: kinds `stflb` = `{ stfl with bwd = true }` and `stfnlb`
  (the natural class), two corpus rows (`--zp-stflb`, `--zp-stfnlb`).
- `zturn.h`: the create builds `tzlb` beside `tzl` (the conjugate loaded
  stream, sin negated, like `twzb`); `_vfft_zt_term_bwd` /
  `_vfft_zt_term8_bwd` pick by `tform` / `ntform` exactly as the forward
  does. 14N bytes more per plan.
- No new race machinery: the backward joins the existing tform race as an
  arm; the race decides per cell (32768+ will likely keep the tree form, as
  the forward does).

Expected: 33/30 spill ops → the forward's 6–10. Whether that is *time* is
the race's verdict — but this remedy has the measured precedent, the other
two have refutations.

## 2. Bring ZTURN-T's create to the cascade — the baked quarter-wave

**Measured (probe).** Build 1.7–4.5 µs vs 6–11 µs, 4–6× closer to the exact
twiddle than `cos`/`sin`. **Bound:** an index shift into a quarter-wave at
M = 2048 resolves only RL <= 2048; the cascade's terminator angles are modulo
N up to 262144. So for the cascade it is a two-level product — the baked
coarse table times a 128-entry per-N fine table (128 `cos`/`sin` at create
in place of N), one complex multiply per twiddle, ~1 ulp. Create stays
cheap at large N; the accuracy edge does not survive the product
(`docs/design/CORRECTIONS_sub2048_2026-09-09.md`).

## 3. Pace the calibrator between cells

**Learned the hard way** (sub-2048 campaign): back-to-back runs manufactured
a −2.4% effect and a 17 ns variance difference that vanished under 15 s
cooldowns. `calibrate_k1.exe` races all cells in ONE process, back to back.
Add a cooldown between cells (and report the across-run spread where a
verdict is close) before the next store restamp.

## 4. Twiddle-policy diversity on the mids — a question, not a plan

Asked 2026-09-09 ("our split codelets are diversity: t1 / t1s / log3").

- **t1s (scalar broadcast): structurally inapplicable.** The split family's
  four lanes are four transforms (the lane-batch), so one scalar serves all
  lanes; the cascade's four lanes are four adjacent columns of one
  transform and the twiddle differs per lane (`zturn.h:817`). Nothing to
  broadcast. Same for ZTURN-T's `[c(k..k+3)][s(k..k+3)]` records.
- **t1 (vector twiddles per position): already the mids' form** (`msg` is
  TP_Flat loaded records).
- **log3 (derive powers by multiplication): the cascade owns the verdict
  already** — the terminator's PowW1 squaring tree IS that idea, and loaded
  beats it at every L1-resident cell (+25–33%); the IL side agrees (log3
  raced 2026-09-04: −42% at r32, spills). A derived-twiddle mid could only
  pay where the mid stream streams from L2 (32768+), which tiling (`zt_tw`)
  already targets. Cheap test if ever wanted: one kind entry, raced on a
  tform-style axis at 32768+ only. Not expected to pay below that.

## 5. Fused calls for the cascade

Owner decision 2026-09-09: "zcascade should get fused call treatment too."
Per-stage call cost measured < 0.5% at >= 2048 (`cascade_stage_fusion.md`), so
the value there is not the calls: it is what fusion enables — literal trip
counts and cross-stage register scheduling in one body — and that is
unmeasured for the cascade. The mechanism exists (ZTURN-T's `body_only`
emission + a per-cell driver TU with a registry, `ztt_drivers.ml`); the
cascade's driver would carry the tcut tiling and the tform/t2q picks as
literal variants per cell, which is a larger corpus than ZTURN-T's 33 cells.
Gate it as ZTURN-T was: fused == unfused bitwise, then the race.

## 6. ZTURN-T above 2048 — a TILED ZTURN-T as the cascade's challenger (owner's
question, 2026-09-09: "isn't tiling the only advantage zcascade has?")

**What ZTURN-S has that ZTURN-T does not:** (a) TILING — structural above L1
(plane = 16N B: 32 KB at 2048, 64 KB at 4096); (b) the SCRAMBLED class — the
comb skips the ordering work and is the whole scrambled column's lead;
(c) an MT arm; (d) inventory: the raced terminator/placement twins, r0 = 8.
**What ZTURN-T holds:** twiddle footprint ~half the cascade's (one stream per
stage serves every group), a pre-twiddle backward that does not spill
(vs 33/49), natural order paid on the ingest instead of the terminator.

**Tiling maps onto ZTURN-T cleanly** — runs are contiguous, so a tile of T
complexes holds WHOLE groups of every stage with RL <= T. The tiled driver is
MKL's large path (the RE'd spec: "a 16 KB tile loop that runs stages up to
L=1024 per tile before the cross-tile stages"): per tile, ingest the tile
CONTIGUOUSLY from its strided columns (the load-permuted ingest — the probe's
`t0tl`, refuted sub-1024 for a reason that only holds while the plane is
L1-resident), run every stage with RL <= T while the tile is hot, then the
cross-tile stages sweep the plane. Kernels unchanged (they take Ls/Gs/count);
new: the `t0tl` kind emitted properly, the tiled driver shape in
`ztt_drivers.ml`, the two-level twiddle create (item 2), registry cells
above 2048, and the race against the tiled cascade per cell.

**Evidence today:** one datum — same chain at 2048, ZTURN-T and ZTURN-S TIE
(the 2048 win was the chain). Tiled vs tiled above L1 is unmeasured; the
footprint and backward edges are reasons to expect competitiveness, not a
guarantee. A scrambled ZTURN-T class (`t0ts`: store runs in column order, no
rb[] — the cheapest possible ingest — output digit-reversed) would be what
lets it challenge the scrambled column too; without it the cascade keeps
that column regardless of the natural result.

## 7. Not transferring (measured)

- **Carried twiddle cursor / contiguous stream:** timing-null everywhere.
- **Chain-shape preference (radix-4 ingest):** contradicted at 2048; the
  chain is a searched axis above 2048.
- **Buffer rule (run in dst):** blocked by the transposed backward's
  orientation.
- **Narrow stores / permute savings:** refuted twice. Do not re-propose.
