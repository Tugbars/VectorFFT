# The natural-order four-step, pass by pass (2026-11-10)

Measurement only: nothing in the library changed for it. i9-14900KF (AVX2), one thread, the library's measurement
scope (pinned P-core, sibling held, HIGH priority), forward transforms, the banked 14900KF plans.

## How

A probe that holds its own private copy of the core in one translation unit, so it can call the four-step's parts
directly: the 2D child (the column passes, the inter-pass twiddle and the row transforms: `_vfft_exec6(p->c2d, ...)`),
the natural-order transpose (`_k1fs_transpose`) and the in-place copy back. Each variant: the input copied once, 2
untimed runs, then back-to-back runs with a clock read around every pass; the median of each pass. The split passes add
up to the unsplit `vfft_k1fs_execute` within about 1%.

The twiddle and the row transforms were also timed alone on one hot row, once per plane position (no memory traffic:
what the arithmetic costs).

| file | what |
| --- | --- |
| `fp32_passes.md` | FP32, 2^21..2^24: every variant |
| `fp64_passes.md` | FP64, 2^20..2^23 (the same plane bytes) |

## FP32, ms

| N | plane | out of place | in place | in-place child | in-place transpose | in-place copy back | inter-pass twiddle alone | row transforms alone |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^21 | 16 MB | 5.16 | 6.09 | 4.27 | 0.76 | 1.07 | 1.67 | 1.04 |
| 2^22 | 32 MB | 11.77 | 13.03 | 9.41 | 1.62 | 1.99 | 3.35 | 2.08 |
| 2^23 | 64 MB | 25.92 | 27.81 | 20.63 | 3.53 | 3.66 | 6.74 | 4.16 |
| 2^24 | 128 MB | 54.10 | 56.10 | 41.32 | 7.49 | 7.30 | 13.47 | 8.00 |

- **The inter-pass twiddle costs 0.80 ns per point, the whole 4096-point row transform 0.48 to 0.50.** It is 24 to 27%
  of the in-place transform. `_il2d_fs_twiddle` (il2d_tier.h) is plain C in double arithmetic; at FP32 the compiler
  vectorises it by converting the float data to double and back (8 `vcvtps2pd` and 2 `vcvtpd2ps` per 4 points).
- **The same twiddle written by hand in float** (a probe-only experiment, the same two-level record, 4 points per
  register): **0.14 ns per point**, within 1.2e-7 of the library's result on the same rows. That is 1.39, 2.78, 5.59
  and 11.18 ms less at 2^21..2^24.
- **The in-place copy back** is 13 to 18% of the in-place transform. The in-place child is 4 to 11% faster than the
  out-of-place one (its first column stage writes the lines it read), so without the copy in place would be the
  faster placement.

## FP64, ms

| N | plane | out of place | in place | in-place child | in-place transpose | in-place copy back | inter-pass twiddle alone | row transforms alone |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^20 | 16 MB | 3.24 | 4.26 | 2.61 | 0.61 | 1.05 | 0.31 | 0.56 |
| 2^21 | 32 MB | 8.23 | 9.25 | 5.89 | 1.37 | 1.99 | 0.57 | 2.00 |
| 2^23 | 128 MB | 42.27 | 46.08 | 31.58 | 7.37 | 7.13 | 2.38 | 7.30 |

- The twiddle is 0.27 to 0.30 ns per point at FP64 (5 to 7% of the transform), and by hand 0.28: the compiler's form is
  already as fast there. The cost above is FP32's.
- The in-place copy back is 15 to 25% of the FP64 in-place transform; in place is 9 to 31% slower than out of place.
- 2^22: the in-place cell raced on the scratch store and took the no-copy form, which this probe does not split.

## Variants that did not pay (both files)

- **Regular instead of streaming stores in the transpose**: the sweep costs 1.4 to 2.6 times as much. In place, with
  the copy back in regular stores as well, the total is lower only at a 16 MB plane, where the whole transform then
  stays inside L3: FP64 2^20 3.63 against 4.26 ms, FP32 2^21 5.84 against 6.09. From a 32 MB plane up it is 10 to 26%
  higher.
- **The existing no-copy form** pinned at the banked in-place split (FP32): 5.96 ms against 6.03 at 2^21, then 6 to 29%
  slower at 2^22..2^24.
