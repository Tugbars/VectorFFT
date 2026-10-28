# fftw_avx512_2026-10-22_pow2

Written by benchmark_scripts/run_suite.py, 2026-10-22 16:01.

- machine: AMD Ryzen 5 PRO 8640HS w/ Radeon 760M Graphics, Windows-11-10.0.26200-SP0, 12 logical CPUs
- VectorFFT: codelets for avx512, compiler gcc, tools built 2026-10-22 15:01 (build_tools.py)
- comparator: FFTW fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512 (C:\Users\TUH\Desktop\VectorFFT-main\gauntlet\fftw\libfftw3-3.dll)

## pow2

1D c2c, every power of two from 2 to 2^23 (23 lengths), throughput in GFLOPS: the quick check.

- command: `python gauntlet/gauntlet.py run --group pow2 --cmp fftw --name fftw_avx512_2026-10-22_pow2`
- csv: gauntlet/results/fftw_avx512_2026-10-22_pow2/gauntlet_fftw.csv
- report: gauntlet/results/fftw_avx512_2026-10-22_pow2/report_fftw.md
- picture: gauntlet/results/fftw_avx512_2026-10-22_pow2/vectorfft-fftw-pow2-gflops.svg
- plot: `python src/tools/plots/gen_gflops.py gauntlet/results/fftw_avx512_2026-10-22_pow2/gauntlet_fftw.csv --out gauntlet/results/fftw_avx512_2026-10-22_pow2/vectorfft-fftw-pow2-gflops.svg --cmp FFTW --logy`
- the plot script says: ok gauntlet/results/fftw_avx512_2026-10-22_pow2/vectorfft-fftw-pow2-gflops.svg 23 sizes N = 2..8388608 median speedup x1.33; pow2 (23 cells) VF 21.4, FFTW 10.6 GFLOPS, x1.33
