# The FP32 four-step's inter-pass twiddle in float (2026-11-10)

`_il2d_fs_twiddle` (src/core/il/rank2/il2d_tier.h) has an FP32 form written by hand in float: 4 points per ymm, 8 per
zmm on an AVX-512 build. The FP64 function is unchanged. Commit 6eba715a.

The plain C form computes in double. At FP32 the compiler ran it by converting every sample to double and back: 0.80 ns
per point, more than the whole 4096-point row transform (0.48).

i9-14900KF (AVX2), the library's measurement scope (pinned P-core, sibling held, HIGH priority), forward transforms,
the banked 14900KF plans (four-step at 2^21..2^24).

## Speed

MKL / ours, before -> after, both builds in one run (`bench_ab.md`: 7 rounds, arm order alternating):

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
- Two hand forms were measured. The first (w = C[a] F[b], then x w) runs 0.15 ns per point; the committed one
  ((x F[b]) C[a], the fine entry's duplicated forms laid out once per call) 0.107, and is equal or faster in the bench
  at 11 of the 12 cells.

## Checks

| file | what |
| --- | --- |
| `twiddle_check_avx2.md` | the library's function against a long-double reference, AVX2 build |
| `twiddle_check_avx512.md` | the same, the AVX-512 build under Intel SDE |
| `bench_after.md` | the canonical bench after the change: ratios and the forward error of the whole transform |
| `passes_after.md` | the four-step pass by pass after the change |

- **The function on its own**, both directions, 29 pairs of fine period and row length: whole rows, a partial last
  coarse entry, fine periods that are not a multiple of 4, fine periods above the 128 entries laid out at a time, a row
  that starts off a vector boundary, guard floats behind the row. Largest error 3.4 units of 2^-24, rms 0.71 (the
  double-arithmetic form: 1.0 and 0.41). The AVX-512 build gives the same figures on every row.
- **The whole transform's forward error** in the bench is what it was before: 2.7e-06, 7.6e-07, 1.0e-06 and 1.2e-06 at
  2^21..2^24 out of place.
- **FP64**: the preprocessed source of the FP64 core unit is the same before and after at both ISAs, except the
  build-stamp string.
- **AVX-512**: the CMake build with both precisions compiles with 0 errors. Its speed is not measured (this machine
  has no AVX-512).
- The gates were not run for this change.

The banked times (`ns=`) of the twelve FP32 rows at 2^21..2^24 predate the change; the plans are the same.
