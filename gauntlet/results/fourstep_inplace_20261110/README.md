# The natural four-step in place, through the plane (2026-11-10)

In place, the natural-order four-step ran its 2D child on the caller's array, transposed into the plan's scratch
plane and copied the plane back. It now runs as out of place does, for either placement: the child out of place from
the caller's array into the plane, the transpose from the plane into the destination. In place the destination is the
array the child read; the child has read it whole before the transpose's first store. The backward mirrors it.
Commit f6e4a970 (`src/core/il/rank1/k1_fourstep.h`), both precisions.

i9-14900KF (AVX2), one thread, the library's measurement scope, forward transforms.

## Speed, FP32

Both builds in one run (`bench_ab_fp32.md`: 5 rounds, arm order alternating):

| N | in place before, ms | in place after, ms | MKL / ours in place, before -> after | out of place after, ms |
| ---: | ---: | ---: | ---: | ---: |
| 2^21 | 4.39 | 3.69 | 0.83 -> 1.00 | 3.43 |
| 2^22 | 9.95 | 8.47 | 1.02 -> 1.20 | 8.46 |
| 2^23 | 21.76 | 19.79 | 1.11 -> 1.22 | 19.72 |
| 2^24 | 44.03 | 42.19 | 1.17 -> 1.22 | 42.16 |

- In place is 4 to 17% faster than before, and from 2^22 up costs what out of place costs.
- Out of place is unchanged: 1.000 to 1.004 of its time before.
- At 2^21 in place stays about 7% behind out of place.

## FP64

The seven FP64 rows were re-raced with the new form; their banked in-place times are 9 to 30% lower
(`reraced_rows.md`). A before/after bench was not run at FP64.

## The rows

`reraced_rows.md`: the eleven in-place natural four-step rows of the 14900KF store (4 at FP32, 7 at FP64) re-raced with
a clean f6e4a970 build, and the five scrambled in-place rows the same recalibrations re-banked. The four FP64 rows of
2026-09-30 now carry their child, so their plans no longer race it at every create.

## Checks

| file | what |
| --- | --- |
| `ip_check_fp32.md` | in place against out of place through the public door, FP32, 2^21..2^24 |
| `ip_check_fp64.md` | the same at FP64: 2^19..2^22 and 1000000, one thread and eight |

- The in-place forward is the out-of-place plan's forward (bitwise where the two rows hold the same split), and the
  in-place roundtrip returns the input: 9e-07 at FP32, 2e-15 at FP64.
- An out-of-place plan called with one buffer gives bitwise its two-buffer result.
- The FP64 object, function by function: the four-step's create and execute changed, and one wisdom banking function
  whose stack frame the compiler laid out differently. No execute path outside the four-step changed.
- Both precisions compile at AVX2 and AVX-512.
- The gates were not run for this change.
