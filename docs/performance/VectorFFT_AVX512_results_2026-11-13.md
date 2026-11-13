# VectorFFT on real AVX-512: results for the 2830a053+ task list

Tree: `main` at **2497429** (il_prime.h is in, so change 6 was tested and B7 was run). Machine: the claude.ai cloud VM, a 2-vCPU Intel Xeon @ 2.10 GHz with AVX-512 (2 cores, 1 thread per core). Compiler: GCC 16.2, -march=native. MKL 2026.0.1, capped with `MKL_ENABLE_INSTRUCTIONS=AVX2` for the AVX2 column.
Times are in microseconds and are medians: B1/B2 use 8 timings (4 launcher runs × 2 flip orders), B4-estimate/B5/B6/B7 use 4 (2 runs × 2 flips), and the planned rows (B3, B4-planned) use 2 (1 run × 2 flips). "512/AVX2" is AVX2 time divided by AVX-512 time, so **below 1.00x (⚠) means our AVX-512 build is slower than our AVX2 build**. "vs MKL" is MKL time divided by ours.

## Part A

| Item | Result |
|---|---|
| A1: FP64 and FP32, at AVX-512 and AVX2 | **PASS**, all four build (1,212 codelets, 0 errors). |
| A2: gates | All 8 gates built at AVX-512 (FP32 `vfftf_gate` with `--float`). `real_door_gate` and `zttr_mt_gate` did not compile on Linux (`_aligned_malloc`/`_putenv` are Windows only), so I added a POSIX shim. Gates were stopped early on request: **`real_door_gate tiny` PASS (62 cells, rc=0); `big` 12 cells in, all PASS**. None failed. The rest were not run. twin_gate was skipped as instructed. |
| A3: accuracy vs MKL | R2C error is ours vs MKL's forward; C2R error is vs N·x. Estimate 1D 64..4M: max 6.9e-16 (R2C) and 1.9e-15 (C2R). Planned 1D 32..512K: same level. 2D real 64x64..1024x1024 (planned and estimate): max 4.5e-16 (R2C) and 1.3e-15 (C2R). 3D: ≤1.6e-15. Odd/prime real: ≤1.9e-15. C2C primes: ≤3.4e-15 (65537). **Same level as before.** Planned 1M–4M was not run. |

## Where our AVX-512 is slower than our AVX2

| Table | Cells (512/AVX2) | Why (from the store rows) |
|---|---|---|
| B1 R2C oop | 64 (0.95), 256K (0.96), 512K (0.94) | AVX2 estimate picks `zrp` at 64 and `zttr` at 32K–512K; AVX-512 estimate picks `zr2c` (it can't use zrp/zttr yet) |
| B1 C2R oop | 64 (0.81), 32K (0.96), 64K (0.90), 128K (0.76), **256K (0.74)**, 512K (0.80) | same: AVX2 `zttr chain=4.8.8.8.8.x`, AVX-512 `zr2c il_route=ztt` |
| B2 R2C ip | 64 (0.98), 512K (0.97) | same |
| B2 C2R ip | 64 (0.88), 128 (0.89), 256K (0.98) | same |
| B4 2D estimate, short rows | **all 12 cells, both directions: 0.54–0.83x** | the AVX-512 estimate uses `rx=door`, the AVX2 estimate `rx=lm`. Planned AVX-512 banks `rx=lm` and is 1.04–1.69x faster than AVX2 |
| B4 2D estimate, others | R2C 1024x64 (0.79), 1024x1024 (1.00), 64x13 (0.95), 256x17 (0.85), 128x21 (0.96), 64x63 (0.93); C2R 512x512 (0.98), 1024x1024 (0.85), 64x13 (0.96), 256x17 (1.00) | |
| B4 2D planned | R2C 128x21 (0.96), 64x63 (0.91); C2R 256x256 (0.93), 1024x1024 (0.98), 1024x64 (0.84), 256x17 (0.98) | one planned pass each; at C2R 256x256 and 1024x64 the planned plan is slower than the same build's estimate, so the race looks noisy |
| B5 3D estimate | R2C 256x256x16 (0.97); C2R 256x256x16 (0.89), 32x32x1024 (0.97) | 64³/128³ row kernel: AVX-512 `zr2c`, AVX2 `zrp` |
| B6 1D odd estimate | R2C 10125 (0.94) | both use `zrf` |
| B7 C2C primes | none (1.03–1.42x) | |

## What B3 says for the AVX-512 estimate rules (1D real)

- **N ≤ 160:** the planner banks `eng=zrp` in most cells (`pair=4.8`, `4.12`, `4.16`, `10.10`, `12.10`, `16.10`, leaf `r2z`, sometimes `n1t`). It is up to 1.88x faster than the estimate's `zr2c 2p`.
- **N = 128:** all four modes bank `zr2c il_route=mono`, and that is no better than the estimate (R2C oop 0.63x of it). This looks like a bad race at one cell.
- **1536–256K:** `eng=zttr` almost everywhere.
  - C2R oop gains 1.14–1.58x over the estimate. That covers all of B1's C2R AVX-512<AVX2 cells.
  - R2C oop: 0.87–1.55x (64K is the one cell where the zttr pick is slower than the estimate).
  - In place it's mostly a wash (0.86–1.14x). The exceptions are 3840 (1.44–1.62x), C2R 1536 (1.35x) and R2C 6144 (1.23x).
- **512K:** `eng=zfsr` (`split=1024x256`, `256x1024`, `512x512`), 1.05–1.26x over the estimate. So four-step should start one octave lower at AVX-512.
- **512–1024:** `zr2c il_route=ztt` (`8.4.8`, `8.8.8`).
- **Losses to MKL, planned:**
  - 128 (0.54–0.89x), 96 (0.77–0.89x), 1920 (0.76–0.86x, three of four modes) and R2C ip 1536 (0.76x).
  - At 4K in place: C2R 0.94x, R2C 0.96x.

## Against MKL: cells where we still lose (estimate, AVX-512)
- **1D (B1/B2):**
  - Out of place: R2C 128 (0.87x), C2R 64/128 (0.87x/0.88x), C2R 4K (0.95x).
  - In place: R2C 4K (0.94x), C2R 4K (0.96x), C2R 128 (0.71x).
  - The watched cells now win: R2C 2K is 1.38x and 4K is 1.13x (were 0.93x and 0.78x). C2R 1K/2K/8K are 1.13x/1.14x/1.24x; only 4K still loses.
- **2D (B4):**
  - The pow2 planes 128²–512² lose to MKL in R2C (0.77–0.96x estimate; 0.86–0.94x planned).
  - 1024x64 R2C estimate is 0.66x.
  - C2R 64x13 is 0.49x and 128x21 is 0.87x.
  - Short rows, estimate: 0.79–0.98x at the wider rows (because of `rx=door`). Planned they are 1.18–3.9x.
- **3D (B5):** R2C 64³ (0.80x), 128³ (0.90x), 32x32x1024 (0.96x). C2R wins (1.01–1.20x).
- **Odd/prime (B6/B7):** all win: 1.48–3.31x real, 1.24–4.46x C2C.


## B8 (partial, stopped on request)
**Not estimate.** The estimate store has no `nthreads=2` rows, so each first create raced a threaded plan and banked it, for example `n=262144 nthreads=2 | eng=zttr chain=8.8.4.8.8.8 tile=2048 mt=2`. The VM has only 2 cores. Only out of place was run: R2C both ISAs, C2R AVX-512 up to 2M.


### B8 1D R2C out of place, estimate store, T=2 (us, median of 4)
| N | ours AVX-512 | ours AVX2 | MKL AVX-512 (T=2) | MKL AVX2 (T=2) | 512/AVX2 | vs MKL-512 | threaded executes (512 / AVX2) |
|---|---|---|---|---|---|---|---|
| 65536 | 69.8 | 81.6 | 135.9 | 167.9 | 1.17x | 1.95x | 757 / 732 |
| 131072 | 142.7 | 173.3 | 279.3 | 389.8 | 1.21x | 1.96x | 365 / 363 |
| 262144 | 311.0 | 414.7 | 581.1 | 727.8 | 1.33x | 1.87x | 185 / 185 |
| 524288 | 710.8 | 783.2 | 1398.7 | 1363.8 | 1.10x | 1.97x | 104 / 101 |
| 1048576 | 1658.7 | 1915.5 | 2813.9 | 2621.5 | 1.15x | 1.70x | 192 / 192 |
| 2097152 | 3430.6 | 3689.3 | 6460.3 | 5491.7 | 1.08x | 1.88x | 186 / 184 |
| 4194304 | 6808.9 | 8455.0 | 14459.0 | 19110.8 | 1.24x | 2.12x | 182 / 182 |

### B8 1D C2R out of place, estimate store, T=2 (us, median of 4)
| N | ours AVX-512 | ours AVX2 | MKL AVX-512 (T=2) | MKL AVX2 (T=2) | 512/AVX2 | vs MKL-512 | threaded executes (512 / AVX2) |
|---|---|---|---|---|---|---|---|
| 65536 | 78.1 | — | 163.2 | — | — | 2.09x | 752 / — |
| 131072 | 153.0 | — | 384.3 | — | — | 2.51x | 348 / — |
| 262144 | 360.8 | — | 860.7 | — | — | 2.39x | 184 / — |
| 524288 | 763.9 | — | 1944.3 | — | — | 2.55x | 102 / — |
| 1048576 | 1900.1 | — | 3979.4 | — | — | 2.09x | 190 / — |
| 2097152 | 4327.6 | — | 8542.0 | — | — | 1.97x | 184 / — |


## Tooling changes in the build clone (not committed yet)
- `vectorfft_vs_mkl_*`: new `--r2c` / `--c2r` options covering 1D/2D/3D, out of place and in place. The worker writes estimate rows to `wisdom2_real.txt`, and the planned path passes `--r2c/--c2r` to `recal_1d_probe`.
- `bench_1d_vs_mkl --realip` now also runs 1D: one padded row of 2(N/2+1).
- `real_door_gate.c` and `zttr_mt_gate.c` get a POSIX shim for `_aligned_malloc` / `_aligned_free` / `_putenv`.

Attached store rows: `b3_wisdom2_real_avx512.txt` (96 rows of B3), `b4_wisdom2_2d_avx512_planned.txt` and `b4_wisdom2_2d_avx2_planned.txt` (26 rows each; `rx=` is the R2C row token, `rx_c2r=` the C2R one).

# Tables

### B1 R2C out of place  (us, median of 8 timings over 4 runs)
| N | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 speedup | vs MKL-512 | max err |
|---|---|---|---|---|---|---|---|
| 64 | 0.054 | 0.051 | 0.056 | 0.046 | 0.95x ⚠ | 1.04x | 1.9e-16 |
| 128 | 0.084 | 0.098 | 0.073 | 0.078 | 1.17x | 0.87x | 3.1e-16 |
| 256 | 0.153 | 0.215 | 0.246 | 0.178 | 1.41x | 1.61x | 3.7e-16 |
| 512 | 0.303 | 0.406 | 0.379 | 0.373 | 1.34x | 1.25x | 2.5e-16 |
| 1024 | 0.639 | 0.887 | 0.738 | 0.733 | 1.39x | 1.15x | 2.5e-16 |
| 2048 | 1.365 | 2.167 | 1.881 | 1.653 | 1.59x | 1.38x | 3.9e-16 |
| 4096 | 3.219 | 4.712 | 3.648 | 3.611 | 1.46x | 1.13x | 4.3e-16 |
| 8192 | 7.680 | 10.213 | 9.450 | 9.509 | 1.33x | 1.23x | 2.9e-16 |
| 16384 | 20.634 | 22.512 | 25.139 | 25.017 | 1.09x | 1.22x | 4.1e-16 |
| 32768 | 40.850 | 47.538 | 49.505 | 48.517 | 1.16x | 1.21x | 4.4e-16 |
| 65536 | 95.377 | 107.584 | 129.626 | 127.347 | 1.13x | 1.36x | 3.6e-16 |
| 131072 | 237.238 | 264.022 | 394.130 | 378.117 | 1.11x | 1.66x | 5.2e-16 |
| 262144 | 693.950 | 663.962 | 1188.637 | 1158.444 | 0.96x ⚠ | 1.71x | 8.0e-16 |
| 524288 | 1652.468 | 1554.722 | 2643.713 | 2612.518 | 0.94x ⚠ | 1.60x | 5.4e-16 |
| 1048576 | 3196.084 | 3438.552 | 5879.016 | 5870.498 | 1.08x | 1.84x | 6.0e-16 |
| 2097152 | 6863.541 | 7407.758 | 14260.266 | 13602.531 | 1.08x | 2.08x | 6.0e-16 |
| 4194304 | 14889.859 | 15867.164 | 34274.658 | 32649.946 | 1.07x | 2.30x | 6.9e-16 |

### B1 C2R out of place  (us, median of 8 timings over 4 runs)
| N | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 speedup | vs MKL-512 | max err |
|---|---|---|---|---|---|---|---|
| 64 | 0.063 | 0.051 | 0.055 | 0.055 | 0.81x ⚠ | 0.87x | 3.4e-16 |
| 128 | 0.085 | 0.101 | 0.075 | 0.080 | 1.18x | 0.88x | 4.4e-16 |
| 256 | 0.151 | 0.236 | 0.177 | 0.174 | 1.56x | 1.17x | 4.7e-16 |
| 512 | 0.302 | 0.464 | 0.365 | 0.372 | 1.53x | 1.21x | 5.6e-16 |
| 1024 | 0.688 | 1.019 | 0.777 | 0.761 | 1.48x | 1.13x | 6.7e-16 |
| 2048 | 1.591 | 2.196 | 1.810 | 1.867 | 1.38x | 1.14x | 5.6e-16 |
| 4096 | 4.538 | 4.699 | 4.325 | 4.348 | 1.04x | 0.95x | 8.9e-16 |
| 8192 | 9.740 | 12.331 | 12.030 | 12.869 | 1.27x | 1.24x | 8.9e-16 |
| 16384 | 21.032 | 24.799 | 28.835 | 30.556 | 1.18x | 1.37x | 8.9e-16 |
| 32768 | 46.895 | 45.062 | 56.521 | 61.996 | 0.96x ⚠ | 1.21x | 1.1e-15 |
| 65536 | 125.217 | 112.557 | 144.345 | 155.343 | 0.90x ⚠ | 1.15x | 1.1e-15 |
| 131072 | 383.385 | 290.348 | 439.111 | 497.440 | 0.76x ⚠ | 1.15x | 1.1e-15 |
| 262144 | 924.702 | 685.857 | 1654.679 | 1729.579 | 0.74x ⚠ | 1.79x | 1.4e-15 |
| 524288 | 2080.222 | 1655.398 | 3641.255 | 3681.758 | 0.80x ⚠ | 1.75x | 1.7e-15 |
| 1048576 | 3441.259 | 4322.294 | 7844.949 | 8116.438 | 1.26x | 2.28x | 1.6e-15 |
| 2097152 | 7954.233 | 11696.148 | 17724.271 | 17636.896 | 1.47x | 2.23x | 1.8e-15 |
| 4194304 | 19589.829 | 27660.641 | 55542.380 | 62787.493 | 1.41x | 2.84x | 1.9e-15 |

### B2 R2C in place  (us, median of 8 timings over 4 runs)
| N | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 speedup | vs MKL-512 | max err |
|---|---|---|---|---|---|---|---|
| 64 | 0.063 | 0.062 | 0.062 | 0.062 | 0.98x ⚠ | 0.99x | 1.9e-16 |
| 128 | 0.090 | 0.122 | 0.082 | 0.077 | 1.35x | 0.91x | 3.1e-16 |
| 256 | 0.150 | 0.223 | 0.190 | 0.202 | 1.49x | 1.26x | 3.7e-16 |
| 512 | 0.300 | 0.428 | 0.372 | 0.373 | 1.43x | 1.24x | 2.5e-16 |
| 1024 | 0.685 | 0.910 | 0.739 | 0.752 | 1.33x | 1.08x | 2.5e-16 |
| 2048 | 1.419 | 2.208 | 1.716 | 1.698 | 1.56x | 1.21x | 3.9e-16 |
| 4096 | 3.868 | 5.061 | 3.651 | 3.617 | 1.31x | 0.94x | 4.3e-16 |
| 8192 | 8.559 | 10.982 | 9.494 | 9.471 | 1.28x | 1.11x | 2.9e-16 |
| 16384 | 18.601 | 23.599 | 25.320 | 24.749 | 1.27x | 1.36x | 4.1e-16 |
| 32768 | 49.096 | 50.224 | 49.326 | 48.472 | 1.02x | 1.00x | 3.5e-16 |
| 65536 | 97.171 | 112.799 | 124.377 | 119.715 | 1.16x | 1.28x | 3.6e-16 |
| 131072 | 300.620 | 305.332 | 417.160 | 362.202 | 1.02x | 1.39x | 4.3e-16 |
| 262144 | 777.429 | 779.442 | 1616.328 | 1570.150 | 1.00x | 2.08x | 6.7e-16 |
| 524288 | 1836.334 | 1784.956 | 3702.880 | 3496.812 | 0.97x ⚠ | 2.02x | 5.4e-16 |
| 1048576 | 3338.160 | 3649.896 | 8020.655 | 7828.258 | 1.09x | 2.40x | 6.0e-16 |
| 2097152 | 7623.166 | 7994.501 | 18858.251 | 17576.739 | 1.05x | 2.47x | 6.0e-16 |
| 4194304 | 15690.499 | 16984.015 | 54058.302 | 48714.559 | 1.08x | 3.45x | 6.9e-16 |

### B2 C2R in place  (us, median of 8 timings over 4 runs)
| N | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 speedup | vs MKL-512 | max err |
|---|---|---|---|---|---|---|---|
| 64 | 0.065 | 0.057 | 0.058 | 0.056 | 0.88x ⚠ | 0.90x | 3.4e-16 |
| 128 | 0.114 | 0.102 | 0.081 | 0.074 | 0.89x ⚠ | 0.71x | 4.4e-16 |
| 256 | 0.151 | 0.235 | 0.171 | 0.168 | 1.56x | 1.13x | 4.7e-16 |
| 512 | 0.309 | 0.462 | 0.446 | 0.406 | 1.49x | 1.44x | 5.6e-16 |
| 1024 | 0.685 | 1.019 | 0.759 | 0.765 | 1.49x | 1.11x | 6.7e-16 |
| 2048 | 1.450 | 2.168 | 2.337 | 1.770 | 1.50x | 1.61x | 5.6e-16 |
| 4096 | 3.917 | 5.115 | 3.767 | 4.341 | 1.31x | 0.96x | 8.9e-16 |
| 8192 | 8.560 | 10.686 | 10.553 | 9.784 | 1.25x | 1.23x | 8.9e-16 |
| 16384 | 18.355 | 22.199 | 24.265 | 24.363 | 1.21x | 1.32x | 8.9e-16 |
| 32768 | 41.643 | 46.770 | 48.639 | 54.948 | 1.12x | 1.17x | 1.1e-15 |
| 65536 | 101.008 | 114.546 | 114.805 | 111.644 | 1.13x | 1.14x | 1.1e-15 |
| 131072 | 306.731 | 328.628 | 366.231 | 362.378 | 1.07x | 1.19x | 1.1e-15 |
| 262144 | 780.406 | 764.876 | 1560.329 | 1551.885 | 0.98x ⚠ | 2.00x | 1.4e-15 |
| 524288 | 1834.058 | 1841.038 | 3581.148 | 3595.939 | 1.00x | 1.95x | 1.7e-15 |
| 1048576 | 3500.761 | 4308.202 | 7879.359 | 7784.887 | 1.23x | 2.25x | 1.6e-15 |
| 2097152 | 7713.038 | 9625.354 | 17493.823 | 17341.538 | 1.25x | 2.27x | 1.8e-15 |
| 4194304 | 16180.749 | 18970.018 | 51364.592 | 47397.939 | 1.17x | 3.17x | 1.9e-15 |


### B3 R2C out of place, AVX-512 (us)
| N | banked engine (planned) | estimate's engine | planned | estimate | MKL | planned vs MKL | planned vs estimate |
|---|---|---|---|---|---|---|---|
| 32 | zrp pair=4.8 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=4.4 | 0.044 | 0.045 | 0.036 | 0.82x | 1.03x |
| 48 | zrp pair=4.12 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=4.6 | 0.040 | 0.069 | 0.048 | 1.22x | 1.75x |
| 64 | zrp pair=4.16 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=8.4 | 0.048 | 0.075 | 0.048 | 1.01x | 1.58x |
| 96 | zr2c route=child_oop_il il_route=2p il_pair=12.4 | zr2c route=child_oop_il il_route=2p il_pair=6.8 | 0.072 | 0.086 | 0.078 | 1.08x | 1.20x |
| 100 | zrp pair=10.10 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=5.10 | 0.086 | 0.120 | 0.134 | 1.55x | 1.39x |
| 120 | zrp pair=12.10 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=6.10 | 0.121 | 0.201 | 0.103 | 0.85x | 1.66x |
| 128 | zr2c route=child_oop_il il_route=mono | zr2c route=child_oop_il il_route=2p il_pair=8.8 | 0.136 | 0.086 | 0.073 | 0.54x | 0.63x |
| 160 | zr2c route=child_oop_il il_route=2p il_pair=4.20 | zr2c route=child_oop_il il_route=2p il_pair=8.10 | 0.114 | 0.145 | 0.148 | 1.30x | 1.27x |
| 256 | zr2c route=child_oop_il il_route=2p il_pair=8.16 | zr2c route=child_oop_il il_route=2p il_pair=8.16 | 0.163 | 0.192 | 0.181 | 1.12x | 1.18x |
| 512 | zr2c route=child_oop_il il_route=2p il_pair=16.16 | zr2c route=child_oop_il il_route=2p il_pair=16.16 | 0.322 | 0.419 | 0.418 | 1.30x | 1.30x |
| 1024 | zr2c route=child_oop_il il_route=ztt il_ztt=8.8.8 | zr2c route=child_oop_il il_route=2p il_pair=16.32 | 0.634 | 0.635 | 0.743 | 1.17x | 1.00x |
| 1536 | zttr chain=8.8.3.4 | zr2c route=child_oop_il il_route=2p il_pair=12.64 | 1.004 | 1.479 | 1.250 | 1.25x | 1.47x |
| 1920 | zr2c route=child_oop_il il_route=chain3 il_pair=60.16 | zr2c route=child_oop_il il_route=2p il_pair=15.64 | 2.013 | 2.184 | 1.736 | 0.86x | 1.09x |
| 2048 | zttr chain=8.4.4.8 | zr2c route=child_oop_il il_route=ztt il_ztt=8.4.4.8 | 1.427 | 2.001 | 1.839 | 1.29x | 1.40x |
| 3840 | zttr chain=8.3.4.5.4 | zr2c route=child_oop_il il_route=chain3 il_pair=120.16 | 3.100 | 4.791 | 4.350 | 1.40x | 1.55x |
| 4096 | zr2c route=child_oop_il il_route=ztt il_ztt=8.8.8.4 | zr2c route=child_oop_il il_route=ztt il_ztt=8.4.8.8 | 3.153 | 3.736 | 3.614 | 1.15x | 1.19x |
| 6144 | zttr chain=8.4.4.3.8 | zr2c route=child_oop_il il_route=ztt il_ztt=8.3.8.4.4 | 5.979 | 6.223 | 8.320 | 1.39x | 1.04x |
| 8192 | zr2c route=child_oop_il il_route=ztt il_ztt=8.8.8.8 | zr2c route=child_oop_il il_route=ztt il_ztt=8.8.8.8 | 8.197 | 8.149 | 11.040 | 1.35x | 0.99x |
| 16384 | zttr chain=8.8.4.8.4 | zr2c route=child_oop_il il_route=ztt il_ztt=8.4.8.4.8 | 16.951 | 18.193 | 25.398 | 1.50x | 1.07x |
| 32768 | zttr chain=8.8.8.4.8 | zr2c route=child_oop_il il_route=ztt il_ztt=8.4.4.4.4.8 | 40.756 | 46.072 | 52.133 | 1.28x | 1.13x |
| 65536 | zttr chain=8.8.4.4.8.4 | zr2c route=child_oop_il il_route=ztt il_ztt=8.8.8.8.8 | 108.365 | 94.329 | 144.341 | 1.33x | 0.87x |
| 131072 | zttr chain=8.8.4.4.8.8 | zr2c route=child_oop_il il_route=ztt il_ztt=8.4.8.4.8.8 | 234.398 | 244.274 | 380.017 | 1.62x | 1.04x |
| 262144 | zttr chain=8.8.8.4.8.8 | zr2c route=child_oop_il il_route=ztt il_ztt=8.4.4.4.4.8.8 | 655.195 | 704.611 | 1355.643 | 2.07x | 1.08x |
| 524288 | zfsr split=1024x256 | zr2c route=child_oop_il il_route=ztt il_ztt=8.8.8.8.8.8 | 1372.976 | 1683.955 | 2694.540 | 1.96x | 1.23x |

### B3 R2C in place, AVX-512 (us)
| N | banked engine (planned) | estimate's engine | planned | estimate | MKL | planned vs MKL | planned vs estimate |
|---|---|---|---|---|---|---|---|
| 32 | zrp pair=4.8 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=4.4 | 0.049 | 0.071 | 0.049 | 0.99x | 1.45x |
| 48 | zrp pair=4.12 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=4.6 | 0.061 | 0.076 | 0.053 | 0.87x | 1.25x |
| 64 | zrp pair=4.16 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=8.4 | 0.070 | 0.064 | 0.065 | 0.93x | 0.91x |
| 96 | zr2c route=child_oop_il il_route=2p il_pair=4.12 | zr2c route=child_nat_ip il_route=2p il_pair=6.8 | 0.103 | 0.099 | 0.080 | 0.77x | 0.96x |
| 100 | zrp pair=10.10 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=5.10 | 0.093 | 0.121 | 0.114 | 1.22x | 1.30x |
| 120 | zrp pair=12.10 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=6.10 | 0.114 | 0.210 | 0.099 | 0.87x | 1.84x |
| 128 | zr2c route=child_oop_il il_route=mono | zr2c route=child_nat_ip il_route=2p il_pair=8.8 | 0.105 | 0.141 | 0.077 | 0.74x | 1.34x |
| 160 | zrp pair=16.10 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=8.10 | 0.158 | 0.126 | 0.171 | 1.09x | 0.80x |
| 256 | zttr chain=8.4.4 | zr2c route=child_nat_ip il_route=2p il_pair=8.16 | 0.174 | 0.147 | 0.181 | 1.04x | 0.85x |
| 512 | zr2c route=child_nat_ip il_route=2p il_pair=16.16 | zr2c route=child_nat_ip il_route=2p il_pair=16.16 | 0.298 | 0.315 | 0.401 | 1.34x | 1.05x |
| 1024 | zttr chain=8.4.4.4 | zr2c route=child_nat_ip il_route=2p il_pair=16.32 | 0.627 | 0.630 | 0.747 | 1.19x | 1.00x |
| 1536 | zttr chain=8.8.3.4 | zr2c route=child_nat_ip il_route=2p il_pair=12.64 | 1.615 | 1.381 | 1.222 | 0.76x | 0.86x |
| 1920 | zr2c route=child_nat_ip il_route=chain3 il_pair=60.16 | zr2c route=child_nat_ip il_route=2p il_pair=15.64 | 2.200 | 3.256 | 1.673 | 0.76x | 1.48x |
| 2048 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8 | 1.401 | 1.443 | 2.432 | 1.74x | 1.03x |
| 3840 | zttr chain=8.5.4.3.4 | zr2c route=child_nat_ip il_route=chain3 il_pair=120.16 | 3.713 | 6.010 | 4.654 | 1.25x | 1.62x |
| 4096 | zttr chain=8.4.4.4.4 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8 | 3.937 | 3.916 | 3.769 | 0.96x | 0.99x |
| 6144 | zttr chain=8.3.4.8.4 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.3.8.4.4 | 6.959 | 8.591 | 7.426 | 1.07x | 1.23x |
| 8192 | zttr chain=8.4.4.8.4 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8 | 8.598 | 8.537 | 9.300 | 1.08x | 0.99x |
| 16384 | zttr chain=8.8.8.4.4 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8.8 | 21.898 | 20.772 | 26.797 | 1.22x | 0.95x |
| 32768 | zttr chain=8.8.8.8.4 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8.8 | 46.414 | 43.452 | 49.615 | 1.07x | 0.94x |
| 65536 | zttr chain=8.4.4.8.8.4 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8.8 | 115.305 | 106.540 | 118.764 | 1.03x | 0.92x |
| 131072 | zttr chain=8.8.4.4.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8.8.8 | 302.890 | 303.726 | 366.465 | 1.21x | 1.00x |
| 262144 | zttr chain=8.8.4.8.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8.8.8 | 773.529 | 805.247 | 1823.043 | 2.36x | 1.04x |
| 524288 | zfsr split=512x512 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8.8.8 | 1591.201 | 1968.108 | 3866.822 | 2.43x | 1.24x |

### B3 C2R out of place, AVX-512 (us)
| N | banked engine (planned) | estimate's engine | planned | estimate | MKL | planned vs MKL | planned vs estimate |
|---|---|---|---|---|---|---|---|
| 32 | zrp pair=4.8 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=4.4 | 0.029 | 0.056 | 0.050 | 1.69x | 1.88x |
| 48 | zrp pair=4.12 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=6.4 | 0.041 | 0.054 | 0.060 | 1.48x | 1.33x |
| 64 | zrp pair=4.16 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=4.8 | 0.062 | 0.056 | 0.060 | 0.96x | 0.90x |
| 96 | zrp pair=8.12 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=8.6 | 0.090 | 0.089 | 0.070 | 0.77x | 0.98x |
| 100 | zrp pair=10.10 leaf=n1t | zr2c route=child_oop_il il_route=2p il_pair=10.5 | 0.096 | 0.134 | 0.123 | 1.28x | 1.40x |
| 120 | zr2c route=child_nat_ip il_route=2p il_pair=4.15 | zr2c route=child_oop_il il_route=2p il_pair=10.6 | 0.104 | 0.158 | 0.102 | 0.98x | 1.51x |
| 128 | zr2c route=child_oop_il il_route=mono | zr2c route=child_oop_il il_route=2p il_pair=8.8 | 0.105 | 0.108 | 0.094 | 0.89x | 1.02x |
| 160 | zrp pair=16.10 leaf=r2z | zr2c route=child_oop_il il_route=2p il_pair=10.8 | 0.140 | 0.143 | 0.195 | 1.39x | 1.02x |
| 256 | zrp pair=8.32 leaf=n1t | zr2c route=child_oop_il il_route=2p il_pair=16.8 | 0.185 | 0.150 | 0.196 | 1.06x | 0.81x |
| 512 | zr2c route=child_oop_il il_route=ztt il_ztt=8.4.8 | zr2c route=child_oop_il il_route=2p il_pair=16.16 | 0.333 | 0.419 | 0.442 | 1.33x | 1.26x |
| 1024 | zr2c route=child_oop_il il_route=ztt il_ztt=8.8.8 | zr2c route=child_nat_ip il_route=2p il_pair=32.16 | 0.673 | 0.716 | 0.946 | 1.40x | 1.06x |
| 1536 | zttr chain=8.4.3.8 | zr2c route=child_nat_ip il_route=2p il_pair=64.12 | 1.142 | 1.332 | 1.766 | 1.55x | 1.17x |
| 1920 | zr2c route=child_nat_ip il_route=chain3 il_pair=60.16 | zr2c route=child_nat_ip il_route=2p il_pair=64.15 | 2.759 | 2.209 | 2.107 | 0.76x | 0.80x |
| 2048 | zttr chain=8.8.4.4 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8 | 1.546 | 2.150 | 2.534 | 1.64x | 1.39x |
| 3840 | zttr chain=8.15.4.4 | zr2c route=child_nat_ip il_route=chain3 il_pair=120.16 | 3.351 | 5.228 | 5.725 | 1.71x | 1.56x |
| 4096 | zttr chain=8.8.4.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8 | 3.228 | 5.101 | 5.261 | 1.63x | 1.58x |
| 6144 | zttr chain=8.4.4.3.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.3.8.4.4 | 5.585 | 7.954 | 8.832 | 1.58x | 1.42x |
| 8192 | zttr chain=8.8.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8 | 8.101 | 10.802 | 11.889 | 1.47x | 1.33x |
| 16384 | zttr chain=8.8.4.4.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8.8 | 18.517 | 21.061 | 29.932 | 1.62x | 1.14x |
| 32768 | zttr chain=8.8.4.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8.8 | 35.416 | 47.257 | 60.218 | 1.70x | 1.33x |
| 65536 | zttr chain=8.4.4.8.4.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8.8 | 110.802 | 129.483 | 179.976 | 1.62x | 1.17x |
| 131072 | zttr chain=8.4.4.8.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8.8.8 | 284.880 | 420.576 | 467.699 | 1.64x | 1.48x |
| 262144 | zttr chain=8.8.4.8.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8.8.8 | 702.157 | 1026.828 | 1963.465 | 2.80x | 1.46x |
| 524288 | zfsr split=256x1024 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8.8.8 | 1826.178 | 2295.017 | 4224.125 | 2.31x | 1.26x |

### B3 C2R in place, AVX-512 (us)
| N | banked engine (planned) | estimate's engine | planned | estimate | MKL | planned vs MKL | planned vs estimate |
|---|---|---|---|---|---|---|---|
| 32 | zrp pair=4.8 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=4.4 | 0.043 | 0.054 | 0.046 | 1.07x | 1.26x |
| 48 | zrp pair=4.12 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=6.4 | 0.052 | 0.058 | 0.047 | 0.91x | 1.12x |
| 64 | zrp pair=4.16 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=4.8 | 0.067 | 0.064 | 0.077 | 1.15x | 0.96x |
| 96 | zr2c route=child_nat_ip il_route=2p il_pair=12.4 | zr2c route=child_nat_ip il_route=2p il_pair=8.6 | 0.081 | 0.093 | 0.072 | 0.89x | 1.16x |
| 100 | zr2c route=child_nat_ip il_route=2p il_pair=10.5 | zr2c route=child_nat_ip il_route=2p il_pair=10.5 | 0.123 | 0.106 | 0.131 | 1.06x | 0.86x |
| 120 | zr2c route=child_nat_ip il_route=2p il_pair=4.15 | zr2c route=child_nat_ip il_route=2p il_pair=10.6 | 0.097 | 0.156 | 0.111 | 1.15x | 1.62x |
| 128 | zr2c route=child_oop_il il_route=mono | zr2c route=child_nat_ip il_route=2p il_pair=8.8 | 0.091 | 0.092 | 0.076 | 0.84x | 1.01x |
| 160 | zrp pair=8.20 leaf=n1t | zr2c route=child_nat_ip il_route=2p il_pair=10.8 | 0.137 | 0.124 | 0.131 | 0.95x | 0.91x |
| 256 | zrp pair=16.16 leaf=r2z | zr2c route=child_nat_ip il_route=2p il_pair=16.8 | 0.213 | 0.149 | 0.173 | 0.81x | 0.70x |
| 512 | zr2c route=child_oop_il il_route=ztt il_ztt=8.4.8 | zr2c route=child_nat_ip il_route=2p il_pair=16.16 | 0.315 | 0.349 | 0.424 | 1.35x | 1.11x |
| 1024 | zr2c route=child_oop_il il_route=ztt il_ztt=8.8.8 | zr2c route=child_nat_ip il_route=2p il_pair=32.16 | 0.642 | 0.788 | 0.737 | 1.15x | 1.23x |
| 1536 | zttr chain=8.8.3.4 | zr2c route=child_nat_ip il_route=2p il_pair=64.12 | 1.156 | 1.555 | 1.232 | 1.07x | 1.35x |
| 1920 | zr2c route=child_nat_ip il_route=chain3 il_pair=60.16 | zr2c route=child_nat_ip il_route=2p il_pair=64.15 | 2.091 | 2.381 | 2.164 | 1.04x | 1.14x |
| 2048 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8 | 1.429 | 1.460 | 1.730 | 1.21x | 1.02x |
| 3840 | zttr chain=8.3.5.4.4 | zr2c route=child_nat_ip il_route=chain3 il_pair=120.16 | 3.798 | 5.463 | 3.974 | 1.05x | 1.44x |
| 4096 | zttr chain=8.4.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8 | 3.949 | 3.840 | 3.729 | 0.94x | 0.97x |
| 6144 | zttr chain=8.4.4.3.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.3.8.4.4 | 7.798 | 7.200 | 7.931 | 1.02x | 0.92x |
| 8192 | zttr chain=8.8.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8 | 8.427 | 8.544 | 10.591 | 1.26x | 1.01x |
| 16384 | zttr chain=8.8.4.4.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8.8 | 17.671 | 18.671 | 24.310 | 1.38x | 1.06x |
| 32768 | zttr chain=8.8.4.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8.8 | 36.451 | 41.437 | 52.657 | 1.44x | 1.14x |
| 65536 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8.8 | 108.471 | 98.001 | 114.986 | 1.06x | 0.90x |
| 131072 | zttr chain=8.8.4.8.8.4 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.4.8.8.8 | 293.640 | 291.209 | 373.910 | 1.27x | 0.99x |
| 262144 | zttr chain=8.8.4.8.8.8 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.4.8.8.8.8 | 766.878 | 736.351 | 1597.511 | 2.08x | 0.96x |
| 524288 | zfsr split=1024x256 | zr2c route=child_nat_ip il_route=ztt il_ztt=8.8.8.8.8.8 | 1697.921 | 1791.204 | 3822.258 | 2.25x | 1.05x |


### B4 2D R2C out of place (us)
| N1xN2 | planned 512 | planned AVX2 | 512/AVX2 | est 512 | est AVX2 | 512/AVX2 | MKL 512 | planned-512 vs MKL | est-512 vs MKL | rx= planned 512 / AVX2 | rx= est 512 / AVX2 | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64x4 | 0.27 | 0.32 | 1.19x | 0.58 | 0.32 | 0.55x ⚠ | 0.93 | 3.45x | 1.61x | lm / lm | door / lm | 3.5e-16 |
| 64x8 | 0.44 | 0.51 | 1.16x | 0.83 | 0.51 | 0.61x ⚠ | 1.16 | 2.65x | 1.40x | lm / lm | door / lm | 4.3e-16 |
| 64x16 | 0.89 | 1.09 | 1.23x | 1.53 | 1.21 | 0.79x ⚠ | 1.65 | 1.86x | 1.08x | lm / lm | door / lm | 3.2e-16 |
| 64x32 | 1.58 | 2.38 | 1.51x | 3.57 | 2.37 | 0.66x ⚠ | 3.10 | 1.97x | 0.87x | lm / lm | door / lm | 3.4e-16 |
| 256x4 | 1.29 | 2.07 | 1.60x | 2.44 | 1.78 | 0.73x ⚠ | 3.75 | 2.91x | 1.54x | lm / lm | door / lm | 3.0e-16 |
| 256x8 | 2.28 | 3.02 | 1.33x | 3.74 | 2.95 | 0.79x ⚠ | 5.02 | 2.21x | 1.34x | lm / lm | door / lm | 3.8e-16 |
| 256x16 | 5.24 | 7.11 | 1.36x | 7.75 | 6.40 | 0.83x ⚠ | 7.61 | 1.45x | 0.98x | lm / lm | door / lm | 2.9e-16 |
| 256x32 | 12.91 | 13.66 | 1.06x | 18.76 | 14.06 | 0.75x ⚠ | 15.22 | 1.18x | 0.81x | lm / lm | door / lm | 3.8e-16 |
| 1024x4 | 10.28 | 10.67 | 1.04x | 17.76 | 11.19 | 0.63x ⚠ | 17.33 | 1.69x | 0.98x | lm / lm | door / lm | 4.0e-16 |
| 1024x8 | 15.84 | 17.09 | 1.08x | 22.74 | 17.19 | 0.76x ⚠ | 22.40 | 1.41x | 0.99x | lm / lm | door / lm | 3.0e-16 |
| 1024x16 | 25.95 | 29.72 | 1.15x | 39.16 | 32.09 | 0.82x ⚠ | 37.53 | 1.45x | 0.96x | lm / lm | door / lm | 3.2e-16 |
| 1024x32 | 59.08 | 61.80 | 1.05x | 91.66 | 63.81 | 0.70x ⚠ | 72.67 | 1.23x | 0.79x | lm / lm | door / lm | 3.6e-16 |
| 64x64 | 4.79 | 5.76 | 1.20x | 5.29 | 5.31 | 1.00x | 5.02 | 1.05x | 0.95x | door / door | door / door | 4.5e-16 |
| 128x128 | 19.42 | 23.45 | 1.21x | 21.50 | 24.21 | 1.13x | 17.44 | 0.90x | 0.81x | door / door | door / door | 3.8e-16 |
| 256x256 | 112.53 | 132.33 | 1.18x | 125.75 | 128.97 | 1.03x | 96.92 | 0.86x | 0.77x | zttr_8.4.4_t0_s0 / zr2c_r0 | door / door | 3.6e-16 |
| 512x512 | 622.92 | 782.38 | 1.26x | 607.73 | 626.65 | 1.03x | 583.42 | 0.94x | 0.96x | zttr_8.8.4_t0_s0 / door | door / door | 3.7e-16 |
| 1024x1024 | 2748.65 | 3069.25 | 1.12x | 2896.34 | 2890.41 | 1.00x ⚠ | 2907.85 | 1.06x | 1.00x | zr2c_r1 / door | door / door | 4.5e-16 |
| 1024x64 | 128.24 | 151.62 | 1.18x | 182.20 | 144.28 | 0.79x ⚠ | 120.55 | 0.94x | 0.66x | zrp_4.16_b / door | door / door | 4.3e-16 |
| 64x1024 | 90.83 | 104.11 | 1.15x | 94.72 | 105.06 | 1.11x | 96.05 | 1.06x | 1.01x | door / zr2c_r0 | door / door | 4.4e-16 |
| 64x13 | 1.55 | 2.48 | 1.60x | 2.00 | 1.90 | 0.95x ⚠ | 1.93 | 1.25x | 0.97x | zrm / zrm | door / door | 5.6e-16 |
| 256x17 | 12.11 | 12.69 | 1.05x | 14.80 | 12.63 | 0.85x ⚠ | 22.70 | 1.88x | 1.53x | zrm / zrm | door / door | 5.4e-16 |
| 128x21 | 6.52 | 6.25 | 0.96x ⚠ | 8.50 | 8.12 | 0.96x ⚠ | 9.13 | 1.40x | 1.07x | zrf_7.3_w0 / zrf_7.3_w0 | zrf_7.3_w0 / zrf_7.3_w0 | 4.4e-16 |
| 64x63 | 9.30 | 8.43 | 0.91x ⚠ | 8.82 | 8.19 | 0.93x ⚠ | 10.07 | 1.08x | 1.14x | door / door | zrf_9.7_w0 / zrf_9.7_w0 | 4.5e-16 |
| 61x64 | 9.30 | 11.09 | 1.19x | 10.09 | 11.07 | 1.10x | 43.94 | 4.72x | 4.35x | zrp_4.16_b / door | door / door | 7.2e-16 |
| 131x64 | 24.41 | 29.97 | 1.23x | 25.62 | 29.53 | 1.15x | 56.69 | 2.32x | 2.21x | door / zrp_4.16_b | door / door | 7.6e-16 |
| 257x256 | 160.83 | 207.30 | 1.29x | 159.87 | 210.22 | 1.31x | 495.09 | 3.08x | 3.10x | door / zr2c_r0 | door / door | 6.5e-16 |

### B4 2D C2R out of place (us)
| N1xN2 | planned 512 | planned AVX2 | 512/AVX2 | est 512 | est AVX2 | 512/AVX2 | MKL 512 | planned-512 vs MKL | est-512 vs MKL | rx= planned 512 / AVX2 | rx= est 512 / AVX2 | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64x4 | 0.25 | 0.31 | 1.26x | 0.57 | 0.31 | 0.54x ⚠ | 0.97 | 3.92x | 1.70x | lm / lm | door / lm | 5.6e-16 |
| 64x8 | 0.44 | 0.49 | 1.13x | 0.85 | 0.50 | 0.59x ⚠ | 1.17 | 2.69x | 1.38x | lm / lm | door / lm | 5.6e-16 |
| 64x16 | 1.00 | 1.06 | 1.05x | 1.64 | 1.07 | 0.65x ⚠ | 1.57 | 1.57x | 0.96x | lm / lm | door / lm | 6.7e-16 |
| 64x32 | 2.02 | 2.90 | 1.44x | 3.71 | 2.45 | 0.66x ⚠ | 3.34 | 1.65x | 0.90x | lm / lm | door / lm | 7.8e-16 |
| 256x4 | 1.22 | 1.98 | 1.62x | 2.62 | 2.00 | 0.76x ⚠ | 3.58 | 2.93x | 1.37x | lm / lm | door / lm | 4.4e-16 |
| 256x8 | 2.60 | 4.41 | 1.69x | 4.14 | 3.11 | 0.75x ⚠ | 4.96 | 1.90x | 1.20x | lm / lm | door / lm | 5.6e-16 |
| 256x16 | 5.28 | 7.11 | 1.34x | 7.91 | 6.27 | 0.79x ⚠ | 8.09 | 1.53x | 1.02x | lm / lm | door / lm | 6.7e-16 |
| 256x32 | 12.42 | 13.68 | 1.10x | 17.20 | 13.83 | 0.80x ⚠ | 16.53 | 1.33x | 0.96x | lm / lm | door / lm | 7.8e-16 |
| 1024x4 | 8.27 | 11.73 | 1.42x | 16.49 | 12.42 | 0.75x ⚠ | 16.07 | 1.94x | 0.97x | lm / lm | door / lm | 7.8e-16 |
| 1024x8 | 13.59 | 17.20 | 1.27x | 20.99 | 15.67 | 0.75x ⚠ | 25.06 | 1.84x | 1.19x | lm / lm | door / lm | 7.8e-16 |
| 1024x16 | 25.91 | 30.39 | 1.17x | 38.35 | 28.72 | 0.75x ⚠ | 36.25 | 1.40x | 0.95x | lm / lm | door / lm | 8.9e-16 |
| 1024x32 | 49.40 | 61.03 | 1.24x | 80.44 | 60.89 | 0.76x ⚠ | 73.10 | 1.48x | 0.91x | lm / lm | door / lm | 8.9e-16 |
| 64x64 | 5.71 | 5.76 | 1.01x | 5.31 | 5.53 | 1.04x | 5.89 | 1.03x | 1.11x | door / door | door / door | 7.8e-16 |
| 128x128 | 22.03 | 25.79 | 1.17x | 19.44 | 25.21 | 1.30x | 23.66 | 1.07x | 1.22x | door / door | door / door | 8.9e-16 |
| 256x256 | 132.49 | 123.76 | 0.93x ⚠ | 106.18 | 124.15 | 1.17x | 103.23 | 0.78x | 0.97x | zr2c_r0 / zttr_4.8.4_t0_s2 | door / door | 1.1e-15 |
| 512x512 | 596.45 | 688.31 | 1.15x | 596.49 | 587.47 | 0.98x ⚠ | 674.83 | 1.13x | 1.13x | door / door | door / door | 1.2e-15 |
| 1024x1024 | 2650.40 | 2600.13 | 0.98x ⚠ | 3290.03 | 2794.15 | 0.85x ⚠ | 3038.23 | 1.15x | 0.92x | zr2c_r0 / zttr_4.8.4.4_t0_s0 | door / door | 1.3e-15 |
| 1024x64 | 151.49 | 127.14 | 0.84x ⚠ | 129.21 | 130.60 | 1.01x | 129.13 | 0.85x | 1.00x | door / door | door / door | 1.1e-15 |
| 64x1024 | 75.30 | 116.25 | 1.54x | 93.47 | 103.85 | 1.11x | 99.77 | 1.33x | 1.07x | door / door | door / door | 1.2e-15 |
| 64x13 | 2.74 | 3.19 | 1.16x | 3.39 | 3.24 | 0.96x ⚠ | 1.65 | 0.60x | 0.49x | zrm / zrm | - / - | 1.1e-15 |
| 256x17 | 18.28 | 17.84 | 0.98x ⚠ | 18.96 | 18.91 | 1.00x ⚠ | 21.42 | 1.17x | 1.13x | zrm / zrm | - / - | 1.1e-15 |
| 128x21 | 11.80 | 13.12 | 1.11x | 10.25 | 11.43 | 1.12x | 8.96 | 0.76x | 0.87x | zrf_7.3_w0 / zrf_7.3_w0 | - / - | 9.3e-16 |
| 64x63 | 10.79 | 11.24 | 1.04x | 10.72 | 11.21 | 1.05x | 12.33 | 1.14x | 1.15x | door / door | - / - | 9.4e-16 |
| 61x64 | 9.79 | 10.81 | 1.10x | 9.86 | 10.54 | 1.07x | 44.99 | 4.59x | 4.56x | door / door | door / door | 1.4e-15 |
| 131x64 | 25.30 | 29.44 | 1.16x | 25.68 | 28.50 | 1.11x | 58.73 | 2.32x | 2.29x | door / door | door / door | 1.8e-15 |
| 257x256 | 163.38 | 219.85 | 1.35x | 166.24 | 209.48 | 1.26x | 511.81 | 3.13x | 3.08x | zr2c_r0 / zttr_4.8.4_t0_s0 | door / door | 1.4e-15 |


### B5 3D R2C estimate, out of place (us, median of 4 timings over 2 runs)
| cell | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 | vs MKL-512 | engine 512 / AVX2 (3D: row kernel) | err |
|---|---|---|---|---|---|---|---|---|
| 64x64x64 | 710.82 | 791.78 | 571.12 | 583.09 | 1.11x | 0.80x ⚠ | zr2c / zrp | 4.1e-16 |
| 128x128x128 | 6008.35 | 6400.65 | 5420.96 | 5134.36 | 1.07x | 0.90x ⚠ | zr2c / zrp | 5.0e-16 |
| 256x256x16 | 3948.01 | 3845.35 | 4155.42 | 4158.76 | 0.97x ⚠ | 1.05x | zrm / zrm | 4.5e-16 |
| 32x32x1024 | 3049.70 | 3191.75 | 2930.25 | 2961.75 | 1.05x | 0.96x ⚠ | zr2c / zr2c | 4.3e-16 |

### B5 3D C2R estimate, out of place (us, median of 4 timings over 2 runs)
| cell | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 | vs MKL-512 | engine 512 / AVX2 (3D: row kernel) | err |
|---|---|---|---|---|---|---|---|---|
| 64x64x64 | 610.94 | 632.05 | 694.83 | 694.57 | 1.03x | 1.14x | zr2c / zrp | 1.0e-15 |
| 128x128x128 | 5063.42 | 5701.39 | 6068.62 | 5994.13 | 1.13x | 1.20x | zr2c / zrp | 1.6e-15 |
| 256x256x16 | 3762.30 | 3335.52 | 3812.86 | 3792.36 | 0.89x ⚠ | 1.01x | zrm / zrm | 1.3e-15 |
| 32x32x1024 | 2386.75 | 2320.34 | 2438.35 | 2431.03 | 0.97x ⚠ | 1.02x | zr2c / zttr | 1.3e-15 |

### B6 1D R2C odd lengths, estimate, out of place (us, median of 4 timings over 2 runs)
| cell | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 | vs MKL-512 | engine 512 / AVX2 (3D: row kernel) | err |
|---|---|---|---|---|---|---|---|---|
| 1009 | 7.38 | 10.88 | 10.99 | 10.67 | 1.48x | 1.49x | zrb / zrb | 7.1e-16 |
| 2039 | 13.46 | 17.34 | 26.86 | 26.88 | 1.29x | 2.00x | zrb / zrb | 7.2e-16 |
| 10007 | 88.09 | 116.48 | 285.66 | 287.80 | 1.32x | 3.24x | zrb / zrb | 9.3e-16 |
| 65537 | 1286.72 | 1372.05 | 3697.08 | 3729.35 | 1.07x | 2.87x | zrb / zrb | 7.9e-16 |
| 99991 | 2062.70 | 2316.51 | 3864.31 | 3818.93 | 1.12x | 1.87x | zrb / zrb | 6.4e-16 |
| 3645 | 7.35 | 7.80 | 10.85 | 10.60 | 1.06x | 1.48x | zrf / zrf | 5.7e-16 |
| 10125 | 25.25 | 23.77 | 37.70 | 37.36 | 0.94x ⚠ | 1.49x | zrf / zrf | 5.1e-16 |

### B6 1D C2R odd lengths, estimate, out of place (us, median of 4 timings over 2 runs)
| cell | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 | vs MKL-512 | engine 512 / AVX2 (3D: row kernel) | err |
|---|---|---|---|---|---|---|---|---|
| 1009 | 7.49 | 10.81 | 11.50 | 11.21 | 1.44x | 1.54x | zrb / zrb | 1.5e-15 |
| 2039 | 14.35 | 17.74 | 27.81 | 27.70 | 1.24x | 1.94x | zrb / zrb | 1.4e-15 |
| 10007 | 89.46 | 114.18 | 296.17 | 287.54 | 1.28x | 3.31x | zrb / zrb | 1.8e-15 |
| 65537 | 1325.38 | 1387.13 | 3753.88 | 3753.51 | 1.05x | 2.83x | zrb / zrb | 1.6e-15 |
| 99991 | 2057.70 | 2356.38 | 3864.72 | 3898.99 | 1.15x | 1.88x | zrb / zrb | 1.9e-15 |
| 3645 | 6.40 | 7.14 | 13.63 | 13.56 | 1.11x | 2.13x | zrf / zrf | 1.3e-15 |
| 10125 | 21.79 | 22.07 | 56.01 | 57.52 | 1.01x | 2.57x | zrf / zrf | 1.2e-15 |

### B7 1D C2C primes, estimate, out of place (us, median of 4 timings over 2 runs)
| cell | ours AVX-512 | ours AVX2 | MKL AVX-512 | MKL AVX2 | 512/AVX2 | vs MKL-512 | engine 512 / AVX2 (3D: row kernel) | err |
|---|---|---|---|---|---|---|---|---|
| 1009 | 8.40 | 10.78 | 10.41 | 10.73 | 1.28x | 1.24x | k1 / k1 | 7.2e-16 |
| 2039 | 17.20 | 24.42 | 26.01 | 26.34 | 1.42x | 1.51x | k1 / k1 | 5.3e-16 |
| 10007 | 129.09 | 173.66 | 278.97 | 286.08 | 1.35x | 2.16x | k1 / k1 | 8.7e-16 |
| 65537 | 833.25 | 857.60 | 3716.43 | 3705.20 | 1.03x | 4.46x | k1 / k1 | 3.4e-15 |
