# fft2d r2c v2 — movement-free composition via block-transposing codelet IO

Status: DESIGN (spiked, staged, not started) · Origin: §6a33 checkpoint + §6a34 spike
Owner queue: after 1D debt items or on demand.

## 1. Thesis

§6a29–§6a33 took the v1 tiled-transpose composition to its architectural
floor: every phase sits on its best machinery, and the decomposition at 256²
is unambiguous — **compute (inner-r2c + col-c2c ≈ 173 µs) already beats
MKL-real's entire 205 µs; the ~101 µs of mandatory data movement
(transpose-in, transpose-out+pad, pack) IS the remaining gap.** v2 deletes
the movement by fusing it into the codelets' own load/store slots.

## 2. Why not within-transform SIMD

MKL/FFTW vectorize inside a single transform for large N. Adopting that
style would abandon VectorFFT's entire lane-batched codelet corpus (the DAG
compiler, every emitted family, the jit, the wisdom). Rejected. The
VectorFFT-native move keeps the lane-batched DAG untouched and changes only
the IO shape: **lanes enter and leave through in-register block transposes.**

## 3. Spike evidence (§6a34, this container)

| shape | engineered stride_transpose | 8×4 register-block | memcpy bound |
|---|---|---|---|
| 256×8 (L1) | 16.99 µs/plane | **11.10 (−35%)** | 4.58 (overheads 3.71× / 2.42×) |
| 512×8 | 71.21 | 79.95 (+12%) | 47.63 (1.50× / 1.68×) |

Readings: (a) at the tile shapes the register-block sweep already beats the
engineered kernel standalone — shipped immediately as the v1 skinny fast
path; (b) even the register version pays 2.4× copy cost **as a separate
pass** — fusion targets that multiplier by making the shuffles ride load/
store slots the codelet already spends; (c) the 512 loss is a **write-side
blocking** lesson — and fusion structurally sidesteps it on the input side
(the fused leaf does streaming row-major reads and has no transpose-write
at all).

## 4. ~~The bt codelet ABI family~~ — CORRECTED: extend the strided quadrant

**[CORRECTION, same day as v1 of this doc — stage-1 recon + prior art.]**
The original §4 specified block-transposing load/store preludes on the
CASCADE leaf and terminator. Recon against the emitted leaf killed it: the
leaf reads the DIT fold {g + n·S} — **S-strided columns** — and the
terminator writes {k + s·m} — m-strided rows. The 8×4 block kernel's
economy requires contiguous column runs; on fold-strided access it degrades
to scalar gathers. The mechanism as written does not survive contact with
the codelets' actual access patterns.

**The prior art that governs instead** (found after the fact — the same
process failure as the §6a29 v1.0 incident, recorded as such):
`docs/performance/strided_rows_case_study.md` and the **strided codelet
quadrant** (`codelets/strided/`, "Design C, 2D rows") — mono n1 codelets
that bt-load consecutive row-major rows, run the FULL N-point FFT, and
bt-store transposed. Mono is *why* it works: a single-radix chain consumes
every column exactly once, so the emitter sweeps contiguous column blocks —
the fold-stride problem is a cascade artifact that mono bodies do not have.
The family is wired (strided_rows.h; fft3d/fftnd/2D), gated, tail-handled,
and **naturally ordered for free** (one-digit chain ⇒ digit reversal is
identity). Measured 1.72×/1.40× (AVX-512/AVX2) on isolated row passes.

**What v2 therefore actually is — the quadrant's own named growth
directions, applied to the 2D r2c row pass:**

1. **Strided r2c mono emission** (small N2): r2c editions of the existing
   c2c strided monos — the case study §5 names this gap verbatim. Covers
   rows up to the mono ceiling (N=64 today).
2. **Strided twiddle-stage codelets** (large N2 — our 256/512 cells): the
   quadrant doc calls this "the real growth direction." Design hypothesis
   to validate first: **DIF-front stages** — DIF's first stages pair
   columns at contiguous half-span runs ({c, c+N/2}, then quarters), unlike
   DIT's strided fold, so a strided-DIF-front codelet can bt-load streaming
   row-major blocks, butterfly+twiddle, and hand a lane-major intermediate
   to the existing cascade. The r2c wrinkle (real-input first stage +
   where the hc terminator lands in a DIF-front composition) is the first
   design question of stage 1.
3. The 2D r2c tile path today rides neither: the quadrant is c2c-only and
   our cells exceed mono coverage — both reasons dissolve under (1)+(2).

## 5. Budget projection (256², current-regime µs)

| phase | v1 | v2 projected | mechanism |
|---|---|---|---|
| transpose-in | ~31 | ~8–12 | pass deleted; shuffles ride leaf loads |
| inner-r2c | ~113–117 | +5–10 | shuffle cost absorbed into leaf/terminator |
| transpose-out+pad | ~38 | ~10–14 | pass deleted; blocked store epilogue |
| col-c2c | ~60 | unchanged | already jit-bound, lane-contiguous on pads |
| pack | ~30 | unchanged (split) / already fused (z) | perm retained |
| **total** | **~262** | **~215–225** | **→ ~0.92–0.95× MKL-real** |

Sanity: the projection claims fusion recovers ~60–70% of the standalone
transpose cost, consistent with the spike's copy-overhead multipliers.
Beyond this sits only the pack/perm (natural-order col output via the
existing NATURAL tapes is the follow-on investigation) and MKL's remaining
kernel-level edge.

## 6. Staged plan

1. **Design study: DIF-front feasibility for strided r2c** — on paper +
   one hand-written N=16 or N=32 strided r2c mono as the concrete probe
   (extending Design C's lattice to real input), BIT/roundtrip-gated
   against transpose+native.
2. **Generator: strided r2c mono emission** (gen_set family extension) for
   the covered small-N set; wire into the 2D row pass behind the measured-
   adoption gate **with the §6a34 hysteresis margin (>5%)**.
3. **Strided twiddle-stage (DIF-front) emission** for N2 ∈ {128…512} — the
   quadrant's growth direction; this is the piece that reaches our
   campaign cells.
4. **Bwd mirrors** (c2r strided editions).
5. Re-run the campaign table; revisit pack/natural-order-col (note: strided
   output being naturally ordered may retire part of the pack's perm too).

Each stage lands independently valuable and independently revertible; the
v1 path remains the fallback forever (the adoption gates guarantee no
regression by construction).

## 7. Risks and open questions

- Store-side blocking for the terminator-bt at large shapes (the 512
  lesson) — the epilogue design must be measured early, not assumed.
- avx512 8×8 editions double the shuffle vocabulary — stage after avx2
  proves the economics.
- Register pressure: the bt prelude holds 8 in-flight rows + transposed
  outputs; interaction with the leaf DAG's own live set needs the
  generator's spill accounting (the term_ls 32-zmm precedent says the
  machinery exists).
- Create-time adoption gates flip winners across container weather regimes
  when deltas are small (§6a34 observation) — the hysteresis margin is
  mandatory for every v2 gate; consider persisting the decision into the
  2D wisdom tables so it's calibrated once under controlled conditions.
- howmany==1 only (2D design constraint, unchanged).
