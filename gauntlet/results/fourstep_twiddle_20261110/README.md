# The four-step's inter-pass twiddle by hand (2026-11-10)

`_il2d_fs_twiddle` (src/core/il/rank2/il2d_tier.h) is written by hand at both precisions, as (x F[b]) C[a] with the
fine entry's duplicated forms laid out once per call:

| precision | points per ymm | per zmm (AVX-512 build) | commit |
| --- | ---: | ---: | --- |
| FP32 | 4 | 8 | 6eba715a |
| FP64 | 2 | 4 | 1ee4bdb5 |

i9-14900KF (AVX2), one thread unless said, the library's measurement scope (pinned P-core, sibling held, HIGH
priority), the banked 14900KF plans.

## FP32

The plain C form computed in double. At FP32 the compiler ran it by converting every sample to double and back: 0.80 ns
per point, more than the whole 4096-point row transform (0.48).

MKL / ours, before -> after, both builds in one run (`bench_ab.md`: 7 rounds, arm order alternating; forward):

| N | one thread, out of place | one thread, in place | 8 threads, out of place |
| ---: | ---: | ---: | ---: |
| 2^21 | 0.91 -> 1.28 | 0.67 -> 0.86 | 1.37 -> 1.74 |
| 2^22 | 1.02 -> 1.37 | 0.83 -> 1.09 | 1.47 -> 1.85 |
| 2^23 | 1.18 -> 1.51 | 0.90 -> 1.13 | 1.01 -> 1.20 |
| 2^24 | 1.21 -> 1.53 | 0.98 -> 1.21 | 1.00 -> 1.05 |

- Our time is 0.71 to 0.80 of the time before at one thread out of place, 0.76 to 0.80 in place, 0.77 to 0.92 at
  8 threads.
- The twiddle alone on a hot row (`twiddle_hot.md`): 0.80 -> 0.107 ns per point forward and 0.85 -> 0.106 backward at
  4096-point rows, 0.127 at 2048-point rows.
- Two hand forms were measured. The first (w = C[a] F[b], then x w) runs 0.15 ns per point; the committed one 0.107,
  and is equal or faster in the bench at 11 of the 12 cells.
- The function on its own (`twiddle_check_avx2.md`, `twiddle_check_avx512.md`): both directions, 29 pairs of fine
  period and row length (whole rows, a partial last coarse entry, fine periods that are not a multiple of 4 or exceed
  the entries laid out at a time, a row off a vector boundary, guard floats behind the row). Largest error 3.4 units of
  2^-24, rms 0.71 (the double-arithmetic form: 1.0 and 0.41). The AVX-512 build under Intel SDE gives the same figures
  on every row.
- The whole transform's forward error in the bench is what it was before (`bench_after.md`): 2.7e-06, 7.6e-07, 1.0e-06
  and 1.2e-06 at 2^21..2^24 out of place. `passes_after.md`: the four-step pass by pass after the change.
- The banked times (`ns=`) of the FP32 out-of-place and 8-thread rows at 2^21..2^24 predate the change; the plans are
  the same.

## FP64

The plain C form was already vector code in double; the hand form is faster mainly backward.

| what | before | after | file |
| --- | ---: | ---: | --- |
| twiddle alone, hot row of 1024..4096 points, forward, ns per point | 0.28 | 0.22 | `fp64_twiddle_hot.md` |
| the same, backward | 0.40 | 0.22 | |
| twiddle alone, hot row of 256 or 512 points, forward / backward | 0.30 / 0.42 | 0.26 to 0.27 | |

Whole transforms:

- **Forward**, the canonical bench, both builds in one run (`fp64_bench_ab.md`, 9 rounds), our time after / before:
  1.003 at 2^20 out of place, 0.993 at 2^20 in place, 0.989 at 2^22 out of place, 0.994 at 2^22 in place. A gain of
  0 to 1%.
- **Backward** (`fp64_inplace_check_before_after.md`: the out-of-place backward timed inside one process, two runs of
  each build, alternated): 4% faster at 2^20, 2.5% at 2^21, 1% at 2^22, 2.6% at 1000000, no change at 2^19.

Checks:

- The function on its own (`fp64_twiddle_check_avx2.md`, `fp64_twiddle_check_avx512.md`): both directions, 27 pairs of
  fine period and row length, the same kinds as at FP32 and the partial rows of non-power-of-two sizes. Largest error
  3.2 units of 2^-53, rms 0.71 (the plain C form: 2.5 and 0.65). The AVX-512 build under Intel SDE gives the same
  figures on every row.
- Whole transforms through the public door at 2^19..2^22 and 1000000, in place and out of place, forward and roundtrip:
  the errors are what they were (1e-15 to 2e-15).
- The FP32 core unit's preprocessed source is the same before and after at both ISAs.
- The FP64 object, function by function: the twiddle, the functions that call it (the 2D row pass among them: the
  compiler re-assigned its registers) and three create-side functions changed. `fp64_2d_bench_ab.md`: six 2D cells
  before and after, 0.991 to 1.016 of the time before, every range across 1; the two cells that leaned above 1
  repeated with 16 rounds read 1.004 and 1.008.
- Both precisions compile at AVX2 and AVX-512. AVX-512 speed is not measured (this machine has no AVX-512).

The gates were not run for either change.
