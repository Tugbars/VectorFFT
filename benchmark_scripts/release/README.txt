VectorFFT benchmark programs
============================

These programs time VectorFFT, a fast Fourier transform library, against FFTW on your own
machine. Nothing to install on Windows; on Linux, install FFTW (Debian / Ubuntu:
sudo apt install libfftw3-3 libfftw3-single3; Fedora: sudo dnf install fftw-libs).

Run it
------
Windows: double-click run_benchmark.bat and press Enter for the first choice.
Linux:   ./run.sh

Each program prints a table: for every transform VectorFFT's time, FFTW's time and the
speedup (FFTW's time divided by VectorFFT's; above 1, VectorFFT is faster). It writes the
table and the raw measurements to a folder under vectorfft_results/ beside the programs.

The programs
------------
  vectorfft_vs_fftw_selected          18 commonly used transforms: 1D, 2D (including the
                                      1920 x 1080 and 1280 x 720 frames) and 3D; about 2 minutes
  vectorfft_vs_fftw_pow2              the 1D transform at every power of two from 64 to
                                      4,194,304, the lengths FFTW is tuned for most; a minute
  vectorfft_vs_fftw_selected_planned  the selected transforms with the plans of VectorFFT's
                                      planner, which measures (see "Plans" below)
  vectorfft_vs_fftw_pow2_fp32         the powers of two in single precision (FP32), planned

Options (every program)
-----------------------
  --cells LIST       your own transforms instead of the program's, separated by commas:
                     N (1D), N1xN2 (2D: N1 rows of N2 points), N1xN2xN3 (3D)
                     example: --cells 1000,4096,1080x1920,64x64x64
  --avx2, --avx512   the instruction set; by default AVX-512 when the CPU runs it, else AVX2
  --out DIR          where the results go (default: vectorfft_results)
  --inplace          every transform in place on both libraries: one buffer, the input
                     overwritten by the output
  --check-only       plan and check the transforms, time nothing
Run a program from a command prompt to pass options, or run the menu with them:
run_benchmark.bat --cells 1000,4096 (Linux: ./run.sh 1 --cells 1000,4096).

How it measures
---------------
Complex to complex, double precision (the FP32 program: single), interleaved, natural
order, out of place, one transform per call, one thread. Every transform runs in its own
process, VectorFFT and FFTW in the same process, in both orders (VectorFFT first, then FFTW
first); the speedup shown is the worse of the two orders. The timed thread is pinned to one
core at high priority, with pauses between timing blocks. A control transform is timed
before and after each group; its spread is the machine's noise during the run. Close other
programs while it runs.

FFTW plans its transforms by measuring them (FFTW_MEASURE) the first time and keeps the plans
in vectorfft_results/fftw.wis (fftwf.wis for single precision): the first run takes longer
than the next ones. VectorFFT's errors against FFTW are in the results' csv files (rt_err).

Plans
-----
"Estimate plans": VectorFFT plans from formulas (the lengths and this machine's caches),
nothing measured, the same plans every run.
"The planner's plans": VectorFFT's default planning, which measures. A transform planned
on this machine before is read back at once from wisdom/ beside the programs (it holds the
plans of the CPUs the project has measured); a new one is measured first and saved there,
so the next run is quick.

Requirements
------------
An x86-64 CPU with AVX2 and FMA (Intel from Haswell, 2013; AMD from Zen, 2017, or later);
the programs say so and stop on an older CPU. 64-bit Windows 10 or 11, or Linux.
The largest transform (256 x 256 x 256) needs about 1 GB of memory.

What is in this folder
----------------------
  vectorfft_vs_*      the programs: each checks the CPU, then runs avx2/ or avx512/
  avx2/, avx512/      the benchmark itself for each instruction set
  fftw/               (Windows) FFTW 3.3.10, double and single precision, with its license
                      (GNU GPL, version 2 or later: fftw/COPYING). VectorFFT contains no FFTW
                      code; FFTW is here only to compare against.
  wisdom/             measured plans (see "Plans")
  BUILD.txt           the source commit these programs were built from, and every file's
                      SHA-256
  LICENSE             VectorFFT's license (MIT)

The source, and how to build these programs yourself (including the comparison with Intel
MKL): the project's repository, benchmark_scripts/README.md.
