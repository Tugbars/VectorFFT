"""Patch a COPY of the FFTW 3.3.10 source tree's CMake build (never the original tree):
  1. the version string: the 3.3.10 release's CMakeLists.txt still sets 3.3.9 (configure.ac and NEWS say 3.3.10);
  2. AVX-512, which FFTW's autotools build offers (--enable-avx512) and its CMake build lacks: an ENABLE_AVX512
     option, the dft/rdft avx512 codelet sources, HAVE_AVX512 in config.h. The -mavx512f flag goes on the
     AVX-512 codelet files ONLY: the CMake build puts its other SIMD flags on the whole library, and a
     library-wide -mavx512f would run AVX-512 instructions in generic code on a CPU without them. FFTW checks
     the CPU at run time (have_simd_avx512) before it registers those codelets, as it does for AVX and AVX2.
No library source file is changed. Usage: python patch_fftw_cmake.py <copy of fftw-3.3.10>"""
import io, sys

root = sys.argv[1].rstrip("/\\") + "/"

def patch(rel, pairs):
    p = root + rel
    t = io.open(p, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert t.count(old) == 1, (rel, old[:60], t.count(old))
        t = t.replace(old, new)
    io.open(p, "w", encoding="utf-8", newline="").write(t)
    print("patched", rel)

patch("CMakeLists.txt", [
    ("set (FFTW_VERSION 3.3.9)", "set (FFTW_VERSION 3.3.10)"),
    ('option (ENABLE_AVX2 "Compile with AVX2 instruction set support" OFF)\n',
     'option (ENABLE_AVX2 "Compile with AVX2 instruction set support" OFF)\n'
     'option (ENABLE_AVX512 "Compile with AVX-512 instruction set support" OFF)\n'),
    ("file(GLOB           fftw_kernel_SOURCE              kernel/*.c          kernel/*.h)\n",
     "file(GLOB           fftw_dft_simd_avx512_SOURCE     dft/simd/avx512/*.c dft/simd/avx512/*.h)\n"
     "file(GLOB           fftw_rdft_simd_avx512_SOURCE    rdft/simd/avx512/*.c rdft/simd/avx512/*.h)\n"
     "file(GLOB           fftw_kernel_SOURCE              kernel/*.c          kernel/*.h)\n"),
    ("if (HAVE_AVX2)\n  list (APPEND SOURCEFILES ${fftw_dft_simd_avx2_SOURCE} ${fftw_rdft_simd_avx2_SOURCE})\nendif ()\n",
     "if (HAVE_AVX2)\n  list (APPEND SOURCEFILES ${fftw_dft_simd_avx2_SOURCE} ${fftw_rdft_simd_avx2_SOURCE})\nendif ()\n"
     "\n"
     "if (ENABLE_AVX512)\n"
     "  check_c_compiler_flag (\"-mavx512f\" HAVE_AVX512)\n"
     "endif ()\n"
     "if (HAVE_AVX512)\n"
     "  list (APPEND SOURCEFILES ${fftw_dft_simd_avx512_SOURCE} ${fftw_rdft_simd_avx512_SOURCE})\n"
     "  set_source_files_properties (${fftw_dft_simd_avx512_SOURCE} ${fftw_rdft_simd_avx512_SOURCE}\n"
     "                               PROPERTIES COMPILE_OPTIONS \"-mavx512f\")\n"
     "endif ()\n"),
])
patch("cmake.config.h.in", [
    ("/* #undef HAVE_AVX512 */", "#cmakedefine HAVE_AVX512 1"),
])
