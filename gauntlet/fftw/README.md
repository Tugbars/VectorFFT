# FFTW for the gauntlet

`libfftw3-3.dll` is [FFTW](https://www.fftw.org/) 3.3.10, double precision, built for
Windows x64, and `libfftw3f-3.dll` the same release in single precision. The gauntlet loads
them at run time as its FFTW comparator (`--cmp fftw`; the single-precision one for the
FP32 cell, `--f32 --cmp fftw`), so a Windows user can benchmark VectorFFT against FFTW
without installing anything. The VectorFFT library does not use them and does not link them.

| | `libfftw3-3.dll` (double) | `libfftw3f-3.dll` (single) |
|---|---|---|
| Version string | `fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512` | `fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512` |
| SIMD | SSE2, AVX, AVX2 and AVX-512 codelets | SSE, SSE2, AVX, AVX2 and AVX-512 codelets |
| Threads | combined: `fftw_init_threads` and `fftw_plan_with_nthreads` are in the DLL | combined: `fftwf_init_threads` and `fftwf_plan_with_nthreads` |
| Depends on | Windows' own system libraries only (the Universal C Runtime of Windows 10 and 11) | the same |
| SHA-256 | `74df0224b723008123fee8cc1f43b67403fc4339d0060258b7212cf57bf24bdb` | `c531ff679b2e08d5fc865f6a0a2bfd760de1e34e40679fe483de7d8f6d3ad678` |

FFTW checks the CPU at run time and uses only the instruction sets the CPU has: on an AVX2
machine the AVX-512 codelets stay unused.

## How the gauntlet finds them

The bench binds FFTW by absolute path, never through a `PATH` search:

1. `VFFT_FFTW_DLL` (single precision: `VFFT_FFTWF_DLL`), when set: that file and no other.
   Use it to compare against your own FFTW build.
2. Otherwise the file here. `gauntlet/build.py` and `gauntlet/CMakeLists.txt` compile its
   absolute path into the tools (`VFFT_FFTW_BUNDLED_DLL`, `VFFT_FFTWF_BUNDLED_DLL`).
3. Otherwise vcpkg's `C:\vcpkg\installed\x64-windows\bin\fftw3.dll` (`fftw3f.dll`).

The bench prints the version string and the path of the FFTW it loaded at the start of
every cell. It refuses any library whose version string is not FFTW's own, such as MKL's
FFTW wrappers. FFTW's wisdom for a run is the file named by `VFFT_FFTW_WIS` (single
precision: `VFFT_FFTWF_WIS`, its own file: the two precisions' wisdom never share one).

On Linux there is no bundled library: the bench loads `libfftw3.so.3` (`libfftw3f.so.3`),
or the file named by `VFFT_FFTW_DLL` (`VFFT_FFTWF_DLL`).

## License

FFTW is free software, copyright (c) 2003, 2007-14 Matteo Frigo and the Massachusetts
Institute of Technology, distributed under the GNU General Public License, version 2 or
later. The license is in [`COPYING`](COPYING), the notice in [`COPYRIGHT`](COPYRIGHT).
VectorFFT is MIT-licensed and contains no FFTW code; these DLLs are separate programs that
ship beside it for comparison.

The source is the FFTW 3.3.10 release, <https://www.fftw.org/fftw-3.3.10.tar.gz>. No
library source file is changed. Two build files are, by
[`patch_fftw_cmake.py`](patch_fftw_cmake.py):

- `CMakeLists.txt` sets the version to 3.3.10 (the release's CMake file says 3.3.9) and
  gains the AVX-512 option that FFTW's autotools build has (`--enable-avx512`) and its
  CMake build lacks. The `-mavx512f` flag goes on the AVX-512 codelet files only.
- `cmake.config.h.in` passes `HAVE_AVX512` through to `config.h`.

## Rebuilding them

With MinGW-w64 GCC and CMake (Ninja or make), from a copy of the extracted release:

```
python gauntlet/fftw/patch_fftw_cmake.py <fftw-3.3.10 copy>
cmake -S <fftw-3.3.10 copy> -B <build dir> -G Ninja -DCMAKE_C_COMPILER=gcc ^
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
      -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DDISABLE_FORTRAN=ON ^
      -DENABLE_SSE2=ON -DENABLE_AVX=ON -DENABLE_AVX2=ON -DENABLE_AVX512=ON ^
      -DENABLE_THREADS=ON -DWITH_COMBINED_THREADS=ON ^
      "-DCMAKE_C_FLAGS=-DWITH_OUR_MALLOC" "-DCMAKE_SHARED_LINKER_FLAGS=-static -static-libgcc"
cmake --build <build dir>
```

The build writes `libfftw3.dll`; copy it here as `libfftw3-3.dll`. For single precision add
`-DENABLE_FLOAT=ON -DENABLE_SSE=ON` to the configure line (a build directory of its own); the
build writes `libfftw3f.dll`, copied here as `libfftw3f-3.dll` (built 2026-11-10 with GCC 16.2,
UCRT64, and make). `WITH_OUR_MALLOC` is FFTW's own aligned allocator, which MinGW needs. The
static linker flags put the GCC and threads runtimes inside the DLL, so it loads on a machine
without MinGW.
