# Strided-Batch 2D Codelets — Design C v1

Status: **landed** for single-stage row FFTs (N2 ∈ {16, 32, 64}), forward
direction. Multi-stage (N2 > 64) and backward direction deferred to v2.

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

## Full 2D bench results (vs MKL)

Single-threaded, i9-14900KF, AVX2, ICX -O2 -march=native, batched [N1][N2]
layout. Strided cells (N2 ∈ {16, 32, 64}) below. "Speedup vs MKL" =
`mkl_time / vfft_time` — bigger is better, >1 means we beat MKL.

| Cell | vfft (µs) | MKL (µs) | Speedup vs MKL |
|---|---|---|---|
| 16²       | 0.2  | 0.4  | **1.70×** |
| 32²       | 1.1  | 1.5  | 1.41× |
| 64²       | 5.1  | 6.5  | 1.28× |
| 32×16     | 0.6  | 0.7  | 1.27× |
| 64×16     | 1.1  | 1.4  | 1.32× |
| 128×16    | 2.4  | 3.6  | 1.50× |
| 256×16    | 5.9  | 7.3  | 1.25× |
| 64×32     | 2.1  | 2.9  | 1.43× |
| 128×32    | 4.9  | 7.5  | **1.53×** |
| 256×32    | 11.7 | 15.6 | 1.33× |
| 512×32    | 26.4 | 32.1 | 1.22× |
| 128×64    | 10.7 | 15.6 | 1.46× |
| 256×64    | 24.1 | 33.4 | 1.39× |
| 512×64    | 51.6 | 67.9 | 1.32× |
| 1024×64   | 111  | 147  | 1.32× |

15/15 strided cells beat MKL — speedups range 1.22×–1.70×. The 64²
case previously was 0.82× (lost by 22%) and is now 1.28× (wins by 28%) —
a 50pt swing.

Non-strided cells (N2 > 64) are unchanged (fallback path unmodified).
The 128² cell still slightly loses to MKL (0.94× speedup) — that's the
next target for v2 multi-stage extension.

## Files

- [src/prototype/lib/emit_c.ml](../lib/emit_c.ml) — emitter retrofit (~150 LOC)
- [src/prototype/bin/gen_radix.ml](../bin/gen_radix.ml) — `--strided` CLI flag
- [src/core/strided/strided.h](../../core/strided/strided.h) — codelet table + dispatch
- [src/core/strided/r{16,32,64}_n1_fwd_strided.h](../../core/strided/) — generated codelets
- [src/core/fft2d.h](../../core/fft2d.h) — dispatch wiring in `_fft2d_tiled_range`
- [src/prototype/bench/regression/bench_strided_2d.c](../bench/regression/bench_strided_2d.c) — microbench
- [build_tuned/dev/bench_2d_breakdown.c](../../../build_tuned/dev/bench_2d_breakdown.c) — phase breakdown

## v2 todo (deferred)

1. **Strided-in / strided-out boundary codelets for multi-stage N2 > 64.**
   First stage reads from matrix (strided gather absorbed), middle stages
   run on scratch (unchanged), last stage writes to matrix (strided scatter
   absorbed). Would extend coverage to N2 ∈ {128, 256, 512, 1024}.
2. **Backward direction.** Generate n1_bwd strided codelets, fill
   `strided_bwd` in fft2d.h plan.
3. **AVX-512 8×8 transpose preamble.** Codegen path is currently 4×4
   AVX2 only. AVX-512 codegen requires `_mm512_unpacklo/hi_pd` +
   `_mm512_shuffle_f64x2` for the 8×8 in-register transpose.
4. **Profile 1024² regression vs MKL.** Separate issue from strided.
