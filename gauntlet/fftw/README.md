# FFTW for the gauntlet

`libfftw3-3.dll` is [FFTW](https://www.fftw.org/) 3.3.10, double precision, built for
Windows x64. The gauntlet loads it at run time as its FFTW comparator (`--cmp fftw`), so
a Windows user can benchmark VectorFFT against FFTW without installing anything. The
VectorFFT library does not use it and does not link it.

| | |
|---|---|
| Version string | `fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512` |
| SIMD | SSE2, AVX, AVX2 and AVX-512 codelets. FFTW checks the CPU at run time and uses only the sets the CPU has: on an AVX2 machine the AVX-512 codelets stay unused. |
| Threads | combined: `fftw_init_threads` and `fftw_plan_with_nthreads` are in this DLL, so the threaded cells work |
| Depends on | Windows' own system libraries only (the Universal C Runtime of Windows 10 and 11) |
| SHA-256 | `74df0224b723008123fee8cc1f43b67403fc4339d0060258b7212cf57bf24bdb` |

## How the gauntlet finds it

The bench binds FFTW by absolute path, never through a `PATH` search:

1. `VFFT_FFTW_DLL`, when set: that file and no other. Use it to compare against your own
   FFTW build.
2. Otherwise this file. `gauntlet/build.py` and `gauntlet/CMakeLists.txt` compile its
   absolute path into the tools (`VFFT_FFTW_BUNDLED_DLL`).
3. Otherwise vcpkg's `C:\vcpkg\installed\x64-windows\bin\fftw3.dll`.

The bench prints the version string and the path of the FFTW it loaded at the start of
every cell. It refuses any library whose version string is not FFTW's own, such as MKL's
FFTW wrappers.

On Linux there is no bundled library: the bench loads `libfftw3.so.3`, or the file named
by `VFFT_FFTW_DLL`.

## License

FFTW is free software, copyright (c) 2003, 2007-14 Matteo Frigo and the Massachusetts
Institute of Technology, distributed under the GNU General Public License, version 2 or
later. The license is in [`COPYING`](COPYING), the notice in [`COPYRIGHT`](COPYRIGHT).
VectorFFT is MIT-licensed and contains no FFTW code; this DLL is a separate program that
ships beside it for comparison.

The source is the FFTW 3.3.10 release, <https://www.fftw.org/fftw-3.3.10.tar.gz>. No
library source file is changed. Two build files are, by
[`patch_fftw_cmake.py`](patch_fftw_cmake.py):

- `CMakeLists.txt` sets the version to 3.3.10 (the release's CMake file says 3.3.9) and
  gains the AVX-512 option that FFTW's autotools build has (`--enable-avx512`) and its
  CMake build lacks. The `-mavx512f` flag goes on the AVX-512 codelet files only.
- `cmake.config.h.in` passes `HAVE_AVX512` through to `config.h`.

## Rebuilding it

With MinGW-w64 GCC, CMake and Ninja, from a copy of the extracted release:

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

The build writes `libfftw3.dll`; copy it here as `libfftw3-3.dll`. `WITH_OUR_MALLOC` is
FFTW's own aligned allocator, which MinGW needs. The static linker flags put the GCC and
threads runtimes inside the DLL, so it loads on a machine without MinGW.
