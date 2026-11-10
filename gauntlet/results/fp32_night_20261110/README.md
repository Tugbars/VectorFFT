# FP32 1D C2C power of two: the night of 2026-11-10

Branch `dev/FP32Experiments`. i9-14900KF (AVX2), one process per cell pinned to core 2 at HIGH
priority (the canonical bench's protocol), MKL 2025.3 at one thread unless T = 8 is stated. Our FP32
plans come from the banked store `src/wisdom/14900KF/wisdom2f_oop.txt`.

## Files

| file | what |
| --- | --- |
| `fp64_ab_summary.md`, `fp64_ab_rounds.csv` | FP64 regression A/B: main (fe0012c9) against the branch, 15 rounds, 27 cells, alternating arms (`gauntlet/ab_fp64.py`) |
| `fp32_vs_mkl_final_k1.csv`, `fp32_vs_mkl_final_mt_batch.csv` | FP32 against MKL FP32 (DFTI_SINGLE) on the night's final build: median of 3 processes (flips 0, 1, 0), the canonical bench's `--f32` arm |
| `fp32_vs_mkl_ceiling_2p20.csv` | the same bench the next morning, after ZTURN-T's FP32 ceiling moved to 2^20: 2^20 out of place, in place and at T = 8, with 2^19 and 2^21 (and 2^19 at T = 8) as controls |
| `fp32_vs_mkl_pass1.csv` | the same bench before the 2^19 ceiling and the FMA-port setting |
| `fp32_port_ab_all_kinds.md` | the FMA-port A/B: every FP32 cil kind at `--cil-port` 0, 50 and 100, 9 rounds |
| `fp32_port_ab_adopted.md` | the adopted setting (the pair family at 50, every other kind at 0) against all-0, 9 rounds |
| `fp32_accuracy_oop.csv`, `fp32_accuracy_ip.csv` | forward and roundtrip error against a long-double DFT, ours and MKL FP32, 2..262144 (`k1_fwd_ref_probe --f32`) |
| `vfftf_gate.md`, `vfftf_gate_ceiling_2p20.md` | the FP32 gate (`src/tools/gates/vfftf_gate.c`) on a cold scratch store: at night, and after the ceiling moved to 2^20 |
| `experiment_ceiling_2p20_race.md`, `experiment_ceiling_2p20_bench.csv` | the night's experiment behind the 2^20 ceiling, raced and benched on a scratch store |

## FP64

- The generator's FP64 output is byte-identical: 1954 corpus files, 4 registries, 60 fused drivers.
- Every FP64 function body of the core object matches main but the build stamp (`_bundle_load`), one new
  function (`_vfft_err_shared`) and two bodies that differ in addresses or register allocation only.
- Timing A/B: median new/old 0.996..1.003 on every cell, every route matched.
- The 2^20 ceiling (the next morning) changes no FP64 output: the generator's FP64 files byte-identical again,
  the core object's bodies as above.

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
| 2^20 | 1.31 | 1.19 | 1329837 | 1741475 | ztt |
| 2^21 | 0.92 | 0.63 | 4794750 | 4406200 | fs |
| 2^22 | 1.02 | 0.79 | 11276225 | 11539950 | fs |
| 2^23 | 1.18 | 0.90 | 25265425 | 29774850 | fs |
| 2^24 | 1.20 | 0.96 | 52813750 | 63799050 | fs |

2^20 is the next morning's run, after the ceiling moved (`fp32_vs_mkl_ceiling_2p20.csv`); at night the
four-step gave 0.79 / 0.62 there (ours 2186675 ns out of place).

T = 8 (both libraries at 8 threads), out of place: 2048 1.14 (serial plan), 16384 1.71, 65536 1.61,
262144 1.76, 2^19 1.69, 2^20 1.70 (the morning run; the four-step's 1.04 at night), 2^22 1.48, 2^24 0.94.
Contiguous batches, one thread: N = 64 x 1024 0.65, 1024 x 64 0.97, 4096 x 16 1.21 (as at FP64, a
contiguous batch runs its K one-transform plans in turn).

## What changed between the passes

- 2^19: ZTURN-T at FP32 reached 524288 at night (2^20 the next morning, below); the planner picks it
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
  largest error 2.01e-7 forward and 2.02e-7 backward (limit 1e-5); `vfftf_gate.md`. After the ceiling
  moved: ALL PASS again, ZTURN-T at 2^20 in both placements (1.77e-7 forward, 1.69e-7 backward there), the
  largest error 1.86e-7 forward (N = 65536) and 2.02e-7 backward (N = 524288); `vfftf_gate_ceiling_2p20.md`.

## The 2^20 ceiling (the owner's ruling, the next morning)

The night's experiment: with ZTURN-T's FP32 ceiling at 2^20 the planner picked ZTURN-T at 2^20 (chain
8.8.4.8.8.8.8, tile 2048: 1400 us out of place, 1536 us in place in the race; the four-step's plan runs
2187 / 2752 us) and the bench gave 1.28 out of place and 1.13 in place (`experiment_ceiling_2p20_*`).

Shipped: `max_n_f32`, `VFFT_ZTT_MAX_N` and the FP32 four-step band's start at 1048576; the four 2^20 driver
files (186 cells in each FP32 registry); the three 2^20 rows of the store (one thread out of and in place,
T = 8 out of place) taken out and raced on a miss. The race took ZTURN-T at all three: 8.8.4.8.8.8.8 out of
place, 8.8.8.4.8.8.8 in place, both at tile 2048, and T = 8 on the one-thread route.

| cell | MKL / ours | spread of 3 | at night |
| --- | ---: | --- | ---: |
| 2^20 out of place | 1.31 | 1.31..1.33 | 0.79 (four-step) |
| 2^20 in place | 1.19 | 1.18..1.21 | 0.62 (four-step) |
| 2^20 at T = 8 | 1.70 | 1.70..1.88 | 1.04 (four-step) |
| 2^19 out of / in place (control) | 1.23 / 1.25 | 1.22..1.23 / 1.24..1.29 | 1.23 / 1.24 |
| 2^21 out of / in place (control) | 0.92 / 0.63 | 0.91..0.92 / 0.62..0.64 | 0.92 / 0.63 |
| 2^19 at T = 8 (control) | 2.06 | 1.67..2.12 | 1.69 |

The banked out-of-place row records 2189538 ns, where the bench runs its plan at 1329837 ns and the night's
race timed the same chain and tile at 1396750 ns: that one race measurement was disturbed, and the plan is
the experiment's. 2^20 is the twiddle builders' end: 2^21 would need their fine table at 128 entries (it
holds 64).
