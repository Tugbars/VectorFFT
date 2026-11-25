VectorFFT 0.9 beta

The library, ready built. Unpack it, compile your program with one command, run it. There is nothing else to build.

**Start here**

1. Download `vectorfft-@VERSION@-windows-x64.zip` or `vectorfft-@VERSION@-linux-x64.tar.gz` from the files below and unpack it. You get one folder, `vectorfft`.
2. Check that it works on your machine: double-click `check.bat` (Linux: `sh check.sh`). It ends with `ALL PASS`.
3. Save your program in that folder and compile it:
   - Windows: `gcc app.c -I. -L. -lvfft -o app.exe`
   - Linux: `gcc app.c -I. -L. -lvfft -lm -Wl,-rpath,'$ORIGIN' -o app`

Chapter 1 of the [guide](https://github.com/Tugbars/VectorFFT/blob/main/how-to-use/VectorFFT-Guide.pdf) has a first program to copy.

**What you need**

- A CPU with AVX2 and FMA: an Intel Core since 2013, any AMD Ryzen. On a CPU with AVX-512 the library uses it by itself.
- GCC; on Windows, its MinGW-w64 build.
- On Linux: Ubuntu 24.04 or a system as new. On an older Linux, build the library from its sources (the guide's Appendix C).

**This is a beta**

When you replace the library with a newer version, compile your program again.

**Also in this release**

`vectorfft-benchmarks-*`: VectorFFT against FFTW on your own machine, no compiler needed. One file per test: Windows: unzip, double-click `run_selected.bat`. Linux: extract, `./run_selected.sh` (install libfftw3 first).
