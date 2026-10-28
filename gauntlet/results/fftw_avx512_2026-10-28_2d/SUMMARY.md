# fftw_avx512_2026-10-28_2d

Written by benchmark_scripts/run_suite.py, 2026-10-28 10:12.

- machine: AMD Ryzen 5 PRO 8640HS w/ Radeon 760M Graphics, Windows-11-10.0.26200-SP0, 12 logical CPUs
- VectorFFT: codelets for avx512, compiler gcc, tools built 2026-10-28 09:24 (build_tools.py)
- comparator: FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512 (C:\Users\TUH\Desktop\VectorFFT-main\gauntlet\fftw\libfftw3-3.dll)

## 2d

2D c2c, every power-of-two plane up to 4M points (159 planes), the speedup per plane.

- command: `python gauntlet/gauntlet.py run --group 2d-pow2 --cmp fftw --name fftw_avx512_2026-10-28_2d`
- csv: gauntlet/results/fftw_avx512_2026-10-28_2d/gauntlet_2d_fftw.csv
- report: gauntlet/results/fftw_avx512_2026-10-28_2d/report_2d_fftw.md
- picture: gauntlet/results/fftw_avx512_2026-10-28_2d/vectorfft-fftw-2d-pow2.svg + .png
- plot: `python src/tools/plots/gen_2d_pow2.py gauntlet/results/fftw_avx512_2026-10-28_2d/report_2d_fftw.md gauntlet/results/fftw_avx512_2026-10-28_2d/vectorfft-fftw-2d-pow2 --cmp FFTW`
- the plot script says: wrote gauntlet/results/fftw_avx512_2026-10-28_2d/vectorfft-fftw-2d-pow2.svg and gauntlet/results/fftw_avx512_2026-10-28_2d/vectorfft-fftw-2d-pow2.png
