# FP32 1D C2C power of two: the night of 2026-11-10

Branch `dev/FP32Experiments`. i9-14900KF (AVX2), one process per cell pinned to core 2 at HIGH
priority (the canonical bench's protocol), MKL 2025.3 at one thread unless T = 8 is stated. Our FP32
plans come from the banked store `src/wisdom/14900KF/wisdom2f_oop.txt`.

## Files

| file | what |
| --- | --- |
| `fp64_ab_summary.md`, `fp64_ab_rounds.csv` | FP64 regression A/B: main (fe0012c9) against the branch, 15 rounds, 27 cells, alternating arms (`gauntlet/ab_fp64.py`) |
| `fp32_vs_mkl_final_k1.csv`, `fp32_vs_mkl_final_mt_batch.csv` | FP32 against MKL FP32 (DFTI_SINGLE) on the final build: median of 3 processes (flips 0, 1, 0), the canonical bench's `--f32` arm |
| `fp32_vs_mkl_pass1.csv` | the same bench before the ceiling change and the FMA-port setting |
| `fp32_port_ab_all_kinds.md` | the FMA-port A/B: every FP32 cil kind at `--cil-port` 0, 50 and 100, 9 rounds |
| `fp32_port_ab_adopted.md` | the adopted setting (the pair family at 50, every other kind at 0) against all-0, 9 rounds |
| `fp32_accuracy_oop.csv`, `fp32_accuracy_ip.csv` | forward and roundtrip error against a long-double DFT, ours and MKL FP32, 2..262144 (`k1_fwd_ref_probe --f32`) |
| `vfftf_gate.md` | the FP32 gate (`src/tools/gates/vfftf_gate.c`) on a cold scratch store |
| `experiment_ceiling_2p20_race.md`, `experiment_ceiling_2p20_bench.csv` | NOT SHIPPED: the FP32 ZTURN-T ceiling at 2^20 instead of 2^19, raced and benched on a scratch store |

## FP64

- The generator's FP64 output is byte-identical: 1954 corpus files, 4 registries, 60 fused drivers.
- Every FP64 function body of the core object matches main but the build stamp (`_bundle_load`), one new
  function (`_vfft_err_shared`) and two bodies that differ in addresses or register allocation only.
- Timing A/B: median new/old 0.996..1.003 on every cell, every route matched.

## FP32 against MKL FP32, one thread (MKL / ours: above 1 = we are faster)

| N | oop | ip | ours oop ns | MKL oop ns | route |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 4.16 | 3.53 | 3 | 11 | mono |
| 4 | 3.71 | 3.52 | 3 | 11 | mono |
| 8 | 2.44 | 2.21 | 5 | 11 | mono |
| 16 | 1.05 | 1.02 | 11 | 12 | mono |
| 32 | 1.03 | 0.96 | 14 | 14 | 2p |
| 64 | 0.90 | 0.88 | 21 | 19 | 2p |
| 128 | 1.05 | 1.04 | 39 | 41 | 2p |
| 256 | 0.96 | 0.95 | 82 | 79 | 2p |
| 512 | 0.92 | 0.92 | 180 | 166 | ztt |
| 1024 | 0.86 | 0.86 | 386 | 334 | ztt |
| 2048 | 1.11 | 1.04 | 839 | 935 | ztt |
| 4096 | 1.23 | 1.23 | 1799 | 2209 | ztt |
| 8192 | 1.21 | 1.22 | 3930 | 4765 | ztt |
| 16384 | 1.29 | 1.33 | 8383 | 10813 | ztt |
| 32768 | 1.14 | 1.15 | 19816 | 22561 | ztt |
| 65536 | 1.17 | 1.21 | 41783 | 49067 | ztt |
| 131072 | 1.21 | 1.13 | 99060 | 119747 | ztt |
| 262144 | 1.15 | 1.15 | 273325 | 302287 | ztt |
| 2^19 | 1.23 | 1.24 | 592612 | 727025 | ztt |
| 2^20 | 0.79 | 0.62 | 2186675 | 1733125 | fs |
| 2^21 | 0.92 | 0.63 | 4794750 | 4406200 | fs |
| 2^22 | 1.02 | 0.79 | 11276225 | 11539950 | fs |
| 2^23 | 1.18 | 0.90 | 25265425 | 29774850 | fs |
| 2^24 | 1.20 | 0.96 | 52813750 | 63799050 | fs |

T = 8 (both libraries at 8 threads), out of place: 2048 1.14 (serial plan), 16384 1.71, 65536 1.61,
262144 1.76, 2^19 1.69, 2^20 1.04, 2^22 1.48, 2^24 0.94.
Contiguous batches, one thread: N = 64 x 1024 0.65, 1024 x 64 0.97, 4096 x 16 1.21 (as at FP64, a
contiguous batch runs its K one-transform plans in turn).

## What changed between the passes

- 2^19: ZTURN-T at FP32 now reaches 524288, the same 4 MB plane as FP64's 262144; the planner picks it
  over the four-step (0.70 / 0.59 -> 1.23 / 1.24).
- 128 and 256: the pair family's kernels at `--cil-port 50`, our time 0.951 at 128 and 0.965 / 0.976 at
  256 (out of / in place) of the all-0 build; the controls at 16, 32, 64, 512 and 4096 stay at 1.00.

## Accuracy

- Against a long-double DFT (`k1_fwd_ref_probe --f32`, 2..262144, the banked plans, out of and in place): all
  36 cells pass; the largest forward error 5.32e-7 (N = 262144; MKL FP32 5.18e-7), the largest roundtrip
  error 7.15e-7; ours is at most 1.40x MKL FP32's forward error (N = 16). The probe's reference is O(N^2),
  so it stops at 262144.
- The gate (cold scratch store, 2..2^20, K = 1 and 3, T = 1 and 4, out of and in place, forward and
  backward against a double FFT, each placement on its own): ALL PASS, 80 served cells and 5 refusals, the
  largest error 2.01e-7 forward and 2.02e-7 backward (limit 1e-5); `vfftf_gate.md`.

## The 2^20 experiment (not shipped)

With the FP32 ZTURN-T ceiling at 2^20 the planner picks ZTURN-T at 2^20 (chain 8.8.4.8.8.8.8, tile 2048:
1400 us out of place, 1536 us in place in the race; the four-step's plan runs 2187 / 2752 us), and the
bench gives 1.28 out of place and 1.13 in place against MKL FP32, where the shipped four-step gives 0.79
and 0.62. The change is four lines and four generated files; 2^21 would also need the twiddle builders'
fine table at 128 entries (it holds 64).
