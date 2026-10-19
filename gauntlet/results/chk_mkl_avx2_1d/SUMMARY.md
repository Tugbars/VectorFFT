# chk_mkl_avx2_1d

Written by benchmark_scripts/run_suite.py, 2026-10-20 00:11.

- machine: Intel(R) Core(TM) i9-14900KF, Windows-11-10.0.26200-SP0, 32 logical CPUs
- VectorFFT: codelets for avx2, compiler gcc, tools built 2026-10-20 00:07 (build_tools.py), commit e3aa92ed
- comparator: Intel(R) oneAPI Math Kernel Library Version 2025.3-Product Build 20251007 for Intel(R) 64 architecture applications (C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\bin\mkl_rt.2.dll)

## 1d

1D c2c, every length N from 2 to 4,096, throughput in GFLOPS.

- command: `python gauntlet/gauntlet.py run --cells 64,4096 --cmp mkl --name chk_mkl_avx2_1d --yes`
- cells: 64,4096 (a quick check, not the README's set)
- csv: gauntlet/results/chk_mkl_avx2_1d/gauntlet.csv
- report: gauntlet/results/chk_mkl_avx2_1d/report.md
- picture: gauntlet/results/chk_mkl_avx2_1d/vectorfft-mkl-gflops.svg
- plot: `python src/tools/plots/gen_gflops.py gauntlet/results/chk_mkl_avx2_1d/gauntlet.csv --out gauntlet/results/chk_mkl_avx2_1d/vectorfft-mkl-gflops.svg --cmp MKL --logy`
- the plot script says: ok gauntlet/results/chk_mkl_avx2_1d/vectorfft-mkl-gflops.svg 2 sizes N = 64..4096 median speedup x1.06; pow2 (2 cells) VF 66.9, MKL 62.5 GFLOPS, x1.06

## accuracy

The forward 1D c2c transform, every N from 2 to 4,096, against a long-double reference.

- command: `python gauntlet/gauntlet.py verify --cells 64,4096 --cmp mkl --name chk_mkl_avx2_1d --yes`
- cells: 64,4096 (a quick check, not the README's set)
- csv: gauntlet/results/chk_mkl_avx2_1d/verify.csv
- picture: gauntlet/results/chk_mkl_avx2_1d/vectorfft-mkl-precision.svg
- plot: `python src/tools/plots/gen_precision.py gauntlet/results/chk_mkl_avx2_1d/verify.csv --out gauntlet/results/chk_mkl_avx2_1d/vectorfft-mkl-precision.svg`
- the plot script says: ok gauntlet/results/chk_mkl_avx2_1d/vectorfft-mkl-precision.svg {'VectorFFT': 2, 'MKL': 2} median x1e-16: VectorFFT 2.26, MKL 2.36
