# ROWSPLIT + row-mode — the row-major boundary design for the IL 2D real tier

**STATUS: SHIPPED + A/B-RACED 2026-08-26.** Companion to
`docs/roadmap/fft2d_real_il_design.md` (§1/§2.4 name this route; this
file is its execution design of record). Owner directives carried:
"IL at the boundary, split inside" (the cascade pattern), the
thin-driver rule, measured serving, and the no-cross-layout law.

## 1. The problem this design solves

The native IL 2D real tier's row pass transforms N1 independent real
rows (transform-contiguous: row `t` at `x + t*N2`, output row `t` at
`z + t*2*hp1`, hp1 = N2/2+1, interleaved CCE). At tiny N2 the per-row
door pays a fixed per-row toll (~30 ns of dispatch around ~15 ns of
math at N2=16), which sank the many-tiny-rows cells (4096×16 measured
0.02× vs MKL before this design; parity after).

The fast engines for batched tiny FFTs are the **split lane-batch
engines** — SIMD across K lanes — but their data geometry is
**lane-major** (`x[e*K + t]`, split re/im planes), the transpose of
what the tier holds. ROWSPLIT is the bridge: run the split engine on
bands of W rows with the layout conversion at the boundary — never as
a convert *wrapper* (banned), but as fused boundary movement in the
cascade's s0/terminator style.

## 2. The three layers (each degrades safely to the next)

| layer | gate | on failure |
|---|---|---|
| **row-mode doors** (in-engine fusion) | split engine on the STRIDE path (`K=W ≥ decouple_min_k=32`) AND the even-N half-complex plan | door returns −1, NOTHING done → staged route |
| **ROWSPLIT route** | `W % 8 == 0`, `W \| N1`, `N2 % 4 == 0`, AND the raced `rw=` verdict picked it | per-row zr2c door serves (a raced outcome, not a failure) |
| **the native tier** | even N2, chainable N1, OOP, non-NATURAL | LOUD refusal (owner law — no split fallback for IL callers) |

The route is decided per cell by the create-time row race (`rw=` in
the direction-shared `lay=il` real cell; env `VFFT_IL2D_ROWSPLIT`
pins, 0 = per-row; env never banks). In practice rowsplit wins only
the tiny-N2 class — everywhere N2 ≥ 32-ish the per-row zr2c door wins
the race, honestly.

## 3. The staged route (v1 — now the fallback)

Per band of W rows:

1. `_vfft_k1_transpose` — rows → lane-major real scratch.
2. One split r2c/c2r execute at (N2, K=W).
3. `_il2d_transpose_zip` / `_il2d_unzip_transpose`
   (`src/core/transforms/fft2d/fft2d_real_il.h`) — fused 4×4 AVX
   transpose ⊗ (re,im) interleave in ONE register pass per side
   (replaced three passes each when first shipped).

Kernel contracts: store tails are EXACT (destinations are caller
memory); the c2r unzip reads full 4-wide blocks legally only because
its source is the tier-owned rscr plane, over-allocated +8 doubles at
create. For every legal rowsplit cell (N2 % 4 == 0) hp1 is ODD, so
the store tail always runs 1–3 bins.

## 4. Row-mode: the boundaries folded into the engine (v2)

Scout-established facts that made this a pure C change (no OCaml):
the STRIDE engine's stage-0 "fused pack" is a memcpy-gather shim —
the ingest never reaches a codelet — and every postprocess store goes
through the `_r2c_st1/4/8` helper lattice. So the conversion moves
one level down, into `src/core/transforms/real/r2c.h`:

- **Plan fields** (the `zo`/`zi` set-around-execute idiom):
  `rowx/rowxp` (fwd real rows in), `rowz/rowzp` (fwd CCE rows out),
  `rowxo/rowxop` (bwd real rows out), plus lazy scratch
  `rowscr_re/im` ((halfN+1)·K) and `rowwork` (N·K).
- **Movement helpers** (SIMD 4×4, exact tails BOTH axes,
  pitch-parametric): `_r2c_row_pack` (rows → pack scratch, fused
  even/odd split + lane transpose — and it doubles as the bwd
  (re,im)-pair unzip: same movement), `_r2c_row_zip` (postprocess
  scratch → interleaved rows), `_r2c_row_trans` (bwd lane plane →
  rows).
- **Worker hooks**: fwd row-mode ingest branch (`_r2c_row_pack` +
  the WHOLE inner from stage 0 — row-mode yields the plain decoupled
  layout, so stage-0 load fusion is deliberately given up); per-block
  L1-hot `_r2c_row_zip` after the postprocess; per-block L1-hot
  `_r2c_row_trans` after the bwd unpack.
- **Doors**: `vfft_r2c_execute_fwd_rowz` (r2c_dispatch.h) and
  `vfft_c2r_disp_execute_rowz` (c2r_dispatch.h) — guard, set fields,
  execute, clear; return −1 untouched when the plan can't serve
  (non-STRIDE path, odd plan). The bwd door also deletes
  `stride_execute_c2r`'s two full-plane input memcpys (it unzips
  straight into `rowwork`/`c2r_im_buf`).

Pass accounting per direction: 4 boundary passes → 2.

## 5. Measured verdicts (same-run A/B, pinned; the create-time
`VFFT_IL2D_NO_ROWZ` knob keeps both routes alive in one process —
the permanent A/B hook for these doors)

- **c2r: fused WINS** — 1.064× (4096×16), 1.113× (512×16) of the
  WHOLE plane. Mechanism: the deleted double-memcpy + hot transpose.
- **r2c: a wash** (0.994×/1.010×) — the lane-major staged path's
  stage-0 load fusion is worth exactly what row-mode's pass savings
  buy. Both doors serve (fwd is not a loss; strictly fewer passes).
- Outputs bitwise-identical between routes (pure data movement);
  gated in `benches/il2d_real_gate.c` (rs64/rs32 passes + the A/B's
  bitwise cross-check).

## 6. Refuted alternatives (do not reopen without new evidence)

- **Count-padding the column pass** and the **VTW column twin**: the
  column kernels are FLAT across counts 9..513 (~15% tail penalty,
  not 2×; `benches/il2d_colcount_probe.c`) and padding loses per real
  point. (Column-side, adjacent to this design — recorded here
  because the same "tiny hp1" analysis motivated both.)
- **The OCaml UL-store × il_out lattice** (a true in-kernel row-major
  codelet edge): explicitly `failwith`-refused in `c2c_split.ml`
  today; it only becomes the right tool if a PATIENT-tier route
  verdict flips these cells to the RFFT path (whose terminator IS
  codelet-shaped) or the stride stage-0 becomes a true OOP codelet.
  Until then the shim-level C fusion is strictly cheaper.

## 7. Contracts and hazards

- Row-mode fields live on SHARED plan data, set around one execute —
  single-executor-at-a-time (same as `zo`/`zi`). The MT-over-bands
  arc must clone plans or externalize these fields.
- The doors assume nothing about pitch: `rowxp`/`rowzp` are explicit,
  so padded planes (the in-place §2.7 contract, 4KB-skewed pitches)
  ride for free.
- Exact-tail discipline: any helper touching caller memory neither
  over-reads nor over-writes; padded-read shortcuts are legal ONLY
  against tier-owned over-allocated planes and say so at the call
  site.
