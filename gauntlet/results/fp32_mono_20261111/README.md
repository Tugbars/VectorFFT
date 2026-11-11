# FP32 fused mono kernels, N = 16 .. 1024 (2026-11-11)

The FP32 library planned its small power-of-two sizes as two passes or as ZTURN-T and was behind FFTW and MKL
from 64 to 1024. It now has one-call kernels for N = 16 .. 1024 (`codelets/zil/avx2_f32/mono/`, 36 files from the
OCaml generator), raced as forms of the MONO route. Uncommitted at the time of writing.

i9-14900KF (AVX2), one thread, single precision, out of place, forward; the library's measurement scope.

## Against FFTW and MKL

`vectorfft_vs_fftw_pow2_fp32` and `vectorfft_vs_mkl_pow2_fp32`, one full run each (17 sizes, 64 .. 4194304). Each
ratio is the other library's time over ours, the worse of the two engine orders.

| N | ours, ns | FFTW, ns | FFTW / ours | before | ours, ns | MKL, ns | MKL / ours | before |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 14 | 21 | 1.47 | 0.93 | 14 | 19 | 1.35 | 0.90 |
| 128 | 33 | 39 | 1.18 | 0.95 | 33 | 41 | 1.25 | 1.05 |
| 256 | 67 | 74 | 1.10 | 0.87 | 68 | 79 | 1.17 | 0.96 |
| 512 | 146 | 157 | 1.07 | 0.90 | 146 | 166 | 1.13 | 0.92 |
| 1024 | 325 | 353 | 1.09 | 0.90 | 331 | 334 | 1.01 | 0.86 |

- Both programs: faster on 17 of 17 sizes. Geometric mean 1.49x against FFTW (was 1.45x, 12 of 17), 1.18x against
  MKL.
- A second full run of each (`vs_*_run2_*.csv`): MKL 17 of 17 again (1.21x). FFTW 16 of 17: at 512 one of the two
  orders read ours at 169 ns instead of 146, so the cell printed 0.97x; the other order read 1.11x.
- **1024 against MKL is a thin lead: 2 to 3% in a typical reading.** Five more runs of that cell
  (`repeat_1024_vs_mkl.csv`): ours 325 or 326 ns in 8 of 10 readings (330 and 338 in the others), MKL 333 to 337.
  The program prints the worse of its two orders: 1.03, 1.02, 0.99, 1.03, 1.02.
- The slow readings (169 at 512, 338 at 1024) are not the kernels' own spread: called directly, the library's
  kernels stay within 2% over every stack position and over the buffers' positions in their pages
  (`stack_position_scan.txt`, `buffer_position_scan.txt`: 512 = 146 to 149 ns, 1024 = 324 to 330 ns).
- Below the programs' range (`--cells 2,4,8,16,32`): 16 points 3.9x MKL and 4.7x FFTW, 32 points 2.2x and 2.1x.
- "before", FFTW: the same program run on 2026-11-11 00:03, before the kernels. "before", MKL: the gauntlet record
  of 2026-11-10 (`../fp32_night_20261110/`); this program had not been run against MKL before.
- Our time at 32768 read 30.3 us in the FFTW run and 20.0 us in the MKL run (the same plan). Not looked into.

Files: `vs_fftw_summary.csv`, `vs_mkl_summary.csv` (and the per-order `*_results_1d.csv`), `small_vs_*_summary.csv`.

## The rows

`reraced_rows.diff`: the 11 rows of `src/wisdom/14900KF/wisdom2f_oop.txt` re-raced (recalibrate, through the front
door). The race's own times, ns:

| N | out of place before | after | in place before | after |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 11.3 (mono, the solo kind) | 3.9 | 11.8 (2p 4.4) | 7.2 |
| 32 | 13.4 (2p 4.8) | 7.0 | 14.3 (2p 4.8) | 10.6 |
| 64 | 21.2 (2p 8.8) | 14.9 | 22.1 (2p 8.8) | 17.1 |
| 128 | 41.0 (2p 8.16) | 33.8 | 41.7 (2p 8.16) | 34.5 |
| 256 | 84.4 (2p 8.32) | 68.5 | not re-raced | |
| 512 | 179.5 (ztt 8.4.4.4) | 147.8 | not re-raced | |
| 1024 | 384.9 (ztt 8.4.8.4) | 326.7 | not re-raced | |

Every re-raced cell banked the MONO route: form 1 everywhere, form 3 (8x16) in place at 128. In place from 256 up
the kernels have no in-place form (they use the output as their work plane), so those rows keep their plans.

## The kernels alone

`kernels_alone_vs_prototype.txt`: every emitted shape against the Python prototype's kernels of the FP32 study, called
through a function pointer, the better of two stack positions per kernel. Best generated shape over the
prototype's best: 16: 0.99, 32: 1.04, 64: 0.98, 128: 1.00, 256: 1.01, 512: 1.00, 1024: 1.01.
(`c0` / `c50` = FMA-port setting 0 / 50, `s0` = the generator's usual schedule instead of construction order.)

## Checks

| file | what |
| --- | --- |
| `forms_through_the_door.txt` | each of the 18 forms forced by a one-row scratch store, through the public door |
| `vfftf_gate_to_2048.txt` | the FP32 gate on a cold scratch store, N = 2 .. 2048: all pass |

- Each form's plan runs that form's kernel (bitwise equal to a direct call), forward and backward within 1.3e-07
  of a long-double DFT.
- An out-of-place plan called with one buffer gives bitwise its two-buffer result, at every form. In place to 128
  points is bitwise the out-of-place result.
- An in-place row naming a form with no in-place kernel (256 and up) is not served by it: the create plans the cell
  again (2p or ZTURN-T), and the result is right.
- The generator: the 1954 FP64 codelets and the 236 existing FP32 codelets are byte-identical after the change;
  the FP64 registries are unchanged.
- The FP64 solo gate (`il_solo_gate`) passes. Both precisions compile at AVX2 and AVX-512.
- No other gate was run.

## Not done

- AVX-512: no fused kernels yet (`avx512_f32` keeps its plans).
- In place at 256, 512, 1024: the plans of before.
- N = 2, 4, 8 still run the solo kind (2.7 to 5 ns; 8 points is slower than 16).
