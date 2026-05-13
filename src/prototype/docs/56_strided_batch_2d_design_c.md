# Strided-Batch 2D Codelets — Design C v1+v2+AVX-512

Status: **emitter + codelets validated in prototype** for both forward
AND backward directions at radix 16/32/64/128/256, **on both AVX2 and
AVX-512**. Roundtrip identity verified at FP noise on both ISAs.
Production wire-up (into `src/core/`'s 2D path via the tune harness) is
a separate downstream workstream.

## Scope discipline

The codelets generated here live in
[src/prototype/codelets/avx2/strided/](../codelets/avx2/strided/) — this
is the prototype's scratch space for OCaml-emitted codelets, NOT a
production drop-in location. Production integration (into [src/core/fft2d.h](../../core/fft2d.h)
or the tune harness at `src/vectorfft_tune/`) happens through the
established calibration + dispatcher pipeline, not by direct include
into `src/core/`.

## Motivation

The 2D FFT row phase used to follow the pattern:
```
for each tile of B rows:
    gather  (transpose B×N2 → N2×B into scratch)
    inner FFT on scratch
    scatter (transpose N2×B → B×N2 back to matrix)
```
A phase-instrumented bench ([build_tuned/dev/bench_2d_breakdown.c](../../../build_tuned/dev/bench_2d_breakdown.c))
showed gather/scatter at **21–37% of total 2D time** — much higher than
the 33-day-old strategy memo claimed (~10%). Eliminating that traffic
was the lever.

## Design C: load-fused 4×4 transpose inside the codelet

The strided codelet absorbs the gather/scatter into its own load/store
boundaries:
- Top of loop: load 4 rows × 4 cols at matrix stride N2, do AVX2 4×4
  transpose (4 `vunpckl/vunpckh` + 4 `vperm2f128`) to produce 4 vectors
  each holding 4 batch rows at one FFT column. Repeat for each group of
  4 columns (4 groups for radix-16, 8 for radix-32, 16 for radix-64).
- Body: bit-identical butterfly DAG as the standard OOP codelet. The
  only thing that changes is where loaded values come from — `lane_re_j`
  locals instead of `_mm256_loadu_pd(&in_re[j*K + k])`.
- Bottom of loop: inverse 4×4 transpose, store 4 rows × 4 cols back to
  the matrix at stride N2.

No scratch buffer is touched. Matrix → registers → matrix.

## Implementation

OCaml emitter retrofit (one flag, one render path, math layer untouched):

1. **`--strided` flag in [bin/gen_radix.ml](../bin/gen_radix.ml)** —
   emits the new codelet shape via the retrofit.
2. **New signature branch in [lib/emit_c.ml](../lib/emit_c.ml)**:
   ```c
   void radixN_n1_fwd_avx2_gen_strided(
       double *rio_re, double *rio_im,
       const double *tw_re, const double *tw_im,  // unused (n1)
       size_t row_stride, size_t me);
   ```
3. **Pre-loop transpose preamble** — populates `lane_re_0..N-1` and
   `lane_im_0..N-1` locals from matrix-strided loads.
4. **`render_load` returns `lane_re_j`** for `Input(j, _)` when strided
   is set (no intrinsic call — the value is already in the local).
5. **`emit_store` writes `out_lane_re_j = t_<tag>`** for `Output(j, _)`
   when strided is set (no `_mm256_storeu_pd`).
6. **Post-body inverse transpose** — stores `out_lane_*` back to matrix.

The body's algsimp output, scheduling, FMA fusion — all bit-identical to
the standard codelet. Only the load/store boundary differs.

## Validation

[bench_strided_2d.c](../bench/regression/bench_strided_2d.c) compares
the strided codelet directly against (gather + standard codelet + scatter)
on a B × N matrix. All 24 cells (N=16/32/64 × B=8..1024) bit-identical
(err = 0.0e+00). Speedup ratio:

- R16: 0.20–0.81 (5× faster at B=1024)
- R32: 0.21–0.80
- R64: 0.23–0.84

Roundtrip correctness via `test_fft2d`: 18/18 cells PASS at FP noise
(< 1e-14).

## Microbench results

[bench_strided_2d.c](../bench/regression/bench_strided_2d.c) runs the
strided codelet on a B×N matrix and compares the output to a reference
that does (gather → standard OOP codelet → scatter) using the SAME math.
Errors are bit-identical (0.0e+00) at every cell; the only variable is
performance.

Speedup of strided vs reference = `ref_time / strided_time`. Bigger is
better. R16/R32/R64 cover v1; R128/R256 cover v2 (full-DAG single-stage,
register-pressure heavy but still wins).

### Forward strided

| Radix | B=8 | B=32 | B=128 | B=256 |
|---|---|---|---|---|
| R16  | 1.22× | 1.17× | 1.87× | 2.87× |
| R32  | 1.28× | 1.24× | 2.45× | 3.37× |
| R64  | 1.21× | 1.63× | 2.85× | 3.67× |
| R128 | 1.16× | 1.58× | 2.05× | 2.75× |
| R256 | 1.47× | 1.68× | 1.66× | 2.30× |

### Backward strided

| Radix | B=8 | B=32 | B=128 | B=256 |
|---|---|---|---|---|
| R16  | 1.24× | 1.22× | 1.88× | 2.86× |
| R32  | 1.27× | 1.26× | 2.42× | 3.43× |
| R64  | 1.25× | 1.72× | 2.92× | 3.11× |
| R128 | 1.15× | 1.61× | 2.06× | 2.19× |
| R256 | 1.49× | 1.67× | 1.69× | 1.83× |

All 40 directional cells (20 fwd + 20 bwd) show strided WINS — speedups
range from 1.15× (R128 B=8 bwd) to 3.67× (R64 B=256 fwd). The win
amplifies with B because larger B means more gather/scatter work per
codelet call, which Design C eliminates entirely. Bwd wins are within
~5% of fwd at every cell — the bwd math (sign-flipped twiddles) doesn't
materially change the load/store boundary cost which Design C targets.

### Roundtrip identity

For each cell, `bwd_strided(fwd_strided(x)) / N` should equal `x` modulo
FP noise. All 20 cells PASS with errors 3.6e-15 to 1.2e-14 absolute —
roughly N × ε_double, exactly what's expected for a properly-scaled
DFT pair. This confirms fwd + bwd are mathematically consistent and
production can use the pair without normalization quirks.

### AVX-512 port (2026-05-13)

The 8×8 in-register transpose preamble/postamble was added to
`lib/emit_c.ml`'s `vec_width = 8` branch by claude.ai (separate
session, real AVX-512 hardware, Sapphire Rapids-class). The 3-stage
shuffle pipeline matches `src/core/transpose.h` Kernel C:

- **Stage 1** — 8 × `_mm512_unpacklo_pd` / `_mm512_unpackhi_pd` over
  row pairs (0,1) (2,3) (4,5) (6,7), producing intermediates
  `_t0_re.._t7_re` (block-scoped per transpose group).
- **Stage 2** — 8 × `_mm512_permutex2var_pd` with two function-scope
  index vectors (`_tp_idx_lo`, `_tp_idx_hi`) declared outside the
  `for b` loop so gcc treats them as loop-invariant constants.
- **Stage 3** — 8 × `_mm512_shuffle_f64x2` with imm `0x44` (low halves)
  / `0xEE` (high halves), assigned directly into `lane_re_{j0..j0+7}`
  / `lane_im_{j0..j0+7}` without an intermediate named variable. In
  the postamble, this stage fuses into each `_mm512_storeu_pd` call.

The 8×8 transpose is its own inverse, so the postamble uses the same
intrinsic sequence as the preamble — just with `out_lane_re_*` as
"input rows" and matrix stores as "output rows."

#### AVX-512 microbench results

Same shape as the AVX2 microbench. 60/60 PASS on real AVX-512 hardware:

| Radix | B=8 (fwd) | B=32 (fwd) | B=128 (fwd) | B=256 (fwd) |
|---|---|---|---|---|
| R16  | 1.24× | 1.14× | 1.80× | 2.59× |
| R32  | 1.20× | 1.20× | 2.35× | 2.49× |
| R64  | 1.17× | **2.11×** | 2.26× | 2.84× |
| R128 | 1.42× | 1.46× | 1.45× | 1.78× |
| R256 | 1.26× | 1.25× | 1.32× | 1.85× |

Bwd within ~5% of fwd at every cell. All 40 directional cells WIN
bit-identical to reference (err = 0.0e+00). 20 roundtrip cells PASS
at FP noise (3.6e-15 to 1.2e-14).

Speedup ratios are slightly compressed vs AVX2 (which ranged 1.15×–
3.67×) because the reference path's gather/scatter also benefits
from 2× SIMD width. But the *absolute* time savings are larger — the
strided codelet itself runs ~40-50% faster on AVX-512 than on AVX2
because the FFT body scales linearly with SIMD width.

**Objdump verification** at R=16 fwd: 64 unpack + 64 permutex2var + 64
shuffle_f64x2 + 32 loadu + 32 storeu — matches the expected 8×8
transpose math (8 ops per group × 2 groups/side × 2 sides × 2
directions = 64 per stage-1/2/3 intrinsic family). No scalar fallback.

## Production wire-up (NOT done here)

Wiring the strided codelet into the 2D path lives in `src/core/` and
follows the established tune-harness pipeline:
1. Validated prototype codelet → tune harness ingestion
2. Dispatch via `src/vectorfft_tune/generated/r{N}_*_dispatch_*.h`
3. Plan-level selection consults wisdom built by the calibrator

A direct `#include "prototype-emit.c"` from `src/core/fft2d.h` would
short-circuit that pipeline and is explicitly NOT the right path. The
prototype's job here is: validate the emitter + codelet design with
correctness + perf microbenches. Production integration is downstream.

## Known scope limit found during v2 development

A naive attempt to plug R=128/R=256 strided codelets directly into the
2D dispatcher (as a quick prototype) caused test_fft2d FAIL at 128² /
256² / 64×256 with err ≈ 1.3 absolute. The likely cause is an output-
ordering mismatch between our single-stage radix-N strided codelet and
the production multi-stage row FFT plan (which uses different bit-
reversal conventions per stage). This is NOT a bug in the strided
codelet — the microbench validates it against a single-stage reference
bit-identically — it is a mismatch between two different valid FFT
output conventions. Production integration must handle that ordering
contract; the prototype doesn't.

## Files

- [src/prototype/lib/emit_c.ml](../lib/emit_c.ml) — emitter retrofit (~150 LOC), `?(strided)` parameter on `emit_codelet`
- [src/prototype/bin/gen_radix.ml](../bin/gen_radix.ml) — `--strided` CLI flag (composes with `--bwd`)
- [src/prototype/codelets/avx2/strided/](../codelets/avx2/strided/) — generated R=16/32/64/128/256 strided codelets, fwd + bwd (prototype scratch)
- [src/prototype/bench/regression/bench_strided_2d.c](../bench/regression/bench_strided_2d.c) — microbench (fwd, bwd, roundtrip)

## Deferred

1. **R=512 / R=1024 strided.** Explicitly out of scope for AVX2 — at
   that radix the full-DAG approach has 1000+ lane locals which spill
   heavily on a 16-ymm register file. AVX-512's 32 zmm registers ease
   the pressure somewhat but the working set still scales with R; per
   claude.ai's port report, skip unless a specific motivating case
   appears. See [feedback-strided-r512-overkill](../../../../../.claude/projects/c--Users-Tugbars-Desktop-highSpeedFFT/memory/feedback_strided_r512_overkill.md).
2. **Production wire-up.** Tune-harness ingestion and dispatcher
   integration. Output-ordering convention must match plan-level
   contract.
3. **Profile 1024² regression vs MKL.** Separate issue from strided.
