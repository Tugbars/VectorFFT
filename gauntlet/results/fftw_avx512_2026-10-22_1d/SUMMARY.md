# fftw_avx512_2026-10-22_1d

Written by benchmark_scripts/run_suite.py, 2026-10-23 06:57.

- machine: AMD Ryzen 5 PRO 8640HS w/ Radeon 760M Graphics, Windows-11-10.0.26200-SP0, 12 logical CPUs
- VectorFFT: codelets for avx512, compiler gcc, tools built 2026-10-22 15:01 (build_tools.py)
- comparator: FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512 (C:\Users\TUH\Desktop\VectorFFT-main\gauntlet\fftw\libfftw3-3.dll)

## 1d

1D c2c, every length N from 2 to 4,096, throughput in GFLOPS.

- command: `python gauntlet/gauntlet.py run --cells 2..4096 --cmp fftw --name fftw_avx512_2026-10-22_1d`
- csv: gauntlet/results/fftw_avx512_2026-10-22_1d/gauntlet_fftw.csv
- report: gauntlet/results/fftw_avx512_2026-10-22_1d/report_fftw.md
- picture: gauntlet/results/fftw_avx512_2026-10-22_1d/vectorfft-fftw-gflops.svg
- plot: `python src/tools/plots/gen_gflops.py gauntlet/results/fftw_avx512_2026-10-22_1d/gauntlet_fftw.csv --out gauntlet/results/fftw_avx512_2026-10-22_1d/vectorfft-fftw-gflops.svg --cmp FFTW --logy`
- the plot script says: ok gauntlet/results/fftw_avx512_2026-10-22_1d/vectorfft-fftw-gflops.svg 4095 sizes N = 2..4096 median speedup x1.87; pow2 (12 cells) VF 42.8, FFTW 35.5 GFLOPS, x1.06
