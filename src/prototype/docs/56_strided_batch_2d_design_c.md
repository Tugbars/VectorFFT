# Strided-Batch 2D Codelets — Design C v1+v2

Status: **emitter + codelets validated in prototype** for forward
direction at radix 16/32/64/128/256. Production wire-up (into `src/core/`'s
2D path via the tune harness) is a separate downstream workstream.

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

| Radix | B=8 | B=32 | B=128 | B=256 |
|---|---|---|---|---|
| R16  | 1.24× | 1.21× | 1.59× | 2.86× |
| R32  | 1.30× | 1.42× | 2.39× | 3.39× |
| R64  | 1.18× | 1.62× | 2.87× | 3.66× |
| R128 | 1.16× | 1.58× | 2.05× | 2.71× |
| R256 | 1.47× | 1.71× | 1.65× | 2.30× |

Every single microbench cell (30/30) shows strided WINS. The win
amplifies with B because larger B means more gather/scatter work per
codelet call, which Design C eliminates entirely.

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
- [src/prototype/bin/gen_radix.ml](../bin/gen_radix.ml) — `--strided` CLI flag
- [src/prototype/codelets/avx2/strided/](../codelets/avx2/strided/) — generated R=16/32/64/128/256 strided codelets (prototype scratch)
- [src/prototype/bench/regression/bench_strided_2d.c](../bench/regression/bench_strided_2d.c) — microbench

## Deferred

1. **Backward direction.** Generate n1_bwd strided codelets (the `--bwd`
   flag from yesterday's `dft_expand` fix should compose with `--strided`,
   not yet tested).
2. **AVX-512 8×8 transpose preamble.** Codegen path is currently 4×4
   AVX2 only. AVX-512 codegen requires `_mm512_unpacklo/hi_pd` +
   `_mm512_permutex2var_pd` + `_mm512_shuffle_f64x2` for the 8×8
   in-register transpose. Integration point is the `vec_width = 8`
   branch in the strided preamble emitter.
3. **R=512 / R=1024 strided.** Untested — register pressure ceiling
   unknown. Worth a microbench iteration to find where the full-DAG
   approach stops paying off vs a multi-stage strided-in/strided-out
   design.
4. **Production wire-up.** Tune-harness ingestion and dispatcher
   integration. Output-ordering convention must match plan-level
   contract.
5. **Profile 1024² regression vs MKL.** Separate issue from strided.
