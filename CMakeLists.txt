# ============================================================================
# VectorFFT — dag-fft-compiler build (CMake mirror of build_tuned/build.py)
#
# Builds the bench_1d_vs_mkl benchmark against the dag runtime:
#   - the dag SIMD codelets (codelets/{inplace,rfft,c2r,oop}/<isa>/*.c) -> a
#     static lib (the only .c the header-only core needs linked in),
#   - the header-only core in src/core/ (bare cross-includes -> recursive -I walk),
#   - Intel MKL for the vs-MKL comparison.
#
# Only bench_1d_vs_mkl is wired (all other benches/tests are intentionally out).
#
# Toolchains: GCC/Clang and Intel ICX (IntelLLVM), on both Linux and Windows.
#   GNU-style front end (gcc, clang, icx gcc-driver)  -> LP64, link mkl_rt.
#   MSVC-style front end (cl, clang-cl, icx-cl)        -> ILP64, three MKL libs.
#
# Configure examples:
#   GCC/mingw  : cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=gcc
#   ICX        : cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=icx
#   (on Windows + ICX, configure from an oneAPI command prompt so LIB is set)
#   cmake --build build
# Run (Windows): put MKL's bin (mkl_rt*.dll / the three dlls) and, for mingw,
#   libwinpthread-1.dll on PATH, then ./build/bench_1d_vs_mkl[.exe] [--mt|--oop].
#
# ISA override: -DVFFT_ISA=avx2|avx512  (default avx2).
# The previous (retired) production-lib root CMake is preserved as CMakeLists.old.txt.
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(VectorFFT_dag VERSION 2.0.0 LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(VFFT_ISA "avx2" CACHE STRING "Codelet ISA (avx2/avx512)")

# ── Front-end classification ────────────────────────────────────────────────
# MSVC is true for cl, clang-cl and icx-cl (MSVC-ABI drivers). Everything else
# (gcc, clang, icx gcc-driver) is GNU-style. ICX = IntelLLVM either way.
set(VFFT_MSVC_STYLE ${MSVC})
set(VFFT_IS_ICX FALSE)
if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_C_COMPILER_ID STREQUAL "Intel")
    set(VFFT_IS_ICX TRUE)
endif()

# ── Tree layout (mirrors build.py's DAG/DAG_CORE/DAG_GEN) ───────────────────
set(DAG          "${CMAKE_SOURCE_DIR}/src/dag-fft-compiler")
set(DAG_CORE     "${CMAKE_SOURCE_DIR}/src/core")
set(DAG_GEN      "${DAG}/generator/generated")
set(DAG_JIT      "${DAG}/jit")
set(DAG_CODELETS "${DAG}/codelets")

# ── Recursive core include walk ─────────────────────────────────────────────
# Core headers cross-include BARE (#include "executor.h"), so every dir that
# holds a header must be on the -I path. Collect them from the header tree so a
# future core reorg needs no edit (same intent as build.py build_includes()).
file(GLOB_RECURSE _core_headers "${DAG_CORE}/*.h")
set(_core_inc_dirs "${DAG_CORE}")
foreach(_h ${_core_headers})
    get_filename_component(_d "${_h}" DIRECTORY)
    list(APPEND _core_inc_dirs "${_d}")
endforeach()
list(REMOVE_DUPLICATES _core_inc_dirs)

set(VFFT_INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/include"
    "${DAG}" "${DAG_GEN}" "${DAG_JIT}"
    ${_core_inc_dirs})

# ── ICX runtime lib dir (libircmt / svml / libmmt) — mirror the old root CMake ─
if(VFFT_IS_ICX)
    get_filename_component(_icx_bin "${CMAKE_C_COMPILER}" DIRECTORY)
    get_filename_component(_icx_root "${_icx_bin}" DIRECTORY)
    foreach(_cand "${_icx_root}/lib" "${_icx_root}/compiler/lib")
        if(EXISTS "${_cand}")
            link_directories("${_cand}")
        endif()
    endforeach()
endif()

# ── MKL discovery ───────────────────────────────────────────────────────────
set(_mkl_inc_hints
    "$ENV{MKLROOT}/include"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include"
    "C:/Program Files/Intel/oneAPI/mkl/latest/include"
    "/opt/intel/oneapi/mkl/latest/include")
set(_mkl_lib_hints
    "$ENV{MKLROOT}/lib" "$ENV{MKLROOT}/lib/intel64"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64"
    "C:/Program Files/Intel/oneAPI/mkl/latest/lib"
    "C:/Program Files/Intel/oneAPI/mkl/latest/lib/intel64"
    "/opt/intel/oneapi/mkl/latest/lib" "/opt/intel/oneapi/mkl/latest/lib/intel64")

find_path(MKL_INCLUDE_DIR mkl_dfti.h HINTS ${_mkl_inc_hints})
if(NOT MKL_INCLUDE_DIR)
    message(FATAL_ERROR "MKL not found (mkl_dfti.h). Set MKLROOT or install Intel oneAPI MKL.")
endif()

if(VFFT_MSVC_STYLE)
    # ILP64 three-lib (matches build.py MSVC path). Needs -DMKL_ILP64.
    find_library(MKL_ILP64 NAMES mkl_intel_ilp64 HINTS ${_mkl_lib_hints})
    find_library(MKL_SEQ   NAMES mkl_sequential  HINTS ${_mkl_lib_hints})
    find_library(MKL_CORE  NAMES mkl_core        HINTS ${_mkl_lib_hints})
    if(NOT MKL_ILP64 OR NOT MKL_SEQ OR NOT MKL_CORE)
        message(FATAL_ERROR "MKL ILP64 libs not found (mkl_intel_ilp64/mkl_sequential/mkl_core).")
    endif()
    set(VFFT_MKL_LIBS ${MKL_ILP64} ${MKL_SEQ} ${MKL_CORE})
    set(VFFT_MKL_DEFS VFFT_HAS_MKL MKL_ILP64)
else()
    # LP64 single dynamic lib (mkl_rt). ILP64's 8-byte MKL_LONG corrupts the
    # DFTI strides array under the GNU front ends -> LP64 only here (build.py).
    find_library(MKL_RT NAMES mkl_rt mkl_rt.2 libmkl_rt HINTS ${_mkl_lib_hints})
    if(NOT MKL_RT)
        message(FATAL_ERROR "MKL LP64 lib not found (mkl_rt).")
    endif()
    set(VFFT_MKL_LIBS ${MKL_RT})
    set(VFFT_MKL_DEFS VFFT_HAS_MKL)
endif()
message(STATUS "MKL include: ${MKL_INCLUDE_DIR}")
message(STATUS "MKL libs:    ${VFFT_MKL_LIBS}")

# ── Per-front-end compile flags ─────────────────────────────────────────────
if(VFFT_MSVC_STYLE)
    if(VFFT_ISA STREQUAL "avx512")
        set(_arch /arch:AVX512)
    else()
        set(_arch /arch:AVX2)
    endif()
    set(VFFT_C_FLAGS    /O2 ${_arch} /fp:fast /wd4244 /wd4267)
    set(VFFT_CODE_FLAGS ${VFFT_C_FLAGS} /w)
else()
    if(VFFT_ISA STREQUAL "avx512")
        set(_arch -mavx512f -mavx512dq -mfma)
    else()
        set(_arch -mavx2 -mfma)
    endif()
    set(VFFT_C_FLAGS -O3 ${_arch} -march=native -fpermissive
        -Wno-overflow -Wno-implicit-function-declaration -Wno-unused-function
        -Wno-unknown-argument -Wno-incompatible-pointer-types -Wno-deprecated-declarations)
    set(VFFT_CODE_FLAGS -O3 ${_arch} -march=native -fpermissive -w)
endif()

# ── Codelet static lib (inplace=c2c, rfft/c2r=real-FFT, oop=LEAF/BAILEY2) ────
foreach(_fam inplace rfft c2r oop)
    file(GLOB _g "${DAG_CODELETS}/${_fam}/${VFFT_ISA}/*.c")
    list(APPEND VFFT_CODELET_SRCS ${_g})
endforeach()
list(LENGTH VFFT_CODELET_SRCS _ncodelets)
if(_ncodelets EQUAL 0)
    message(FATAL_ERROR "no codelet .c found under ${DAG_CODELETS}/*/${VFFT_ISA}/")
endif()
message(STATUS "codelets:    ${_ncodelets} .c (${VFFT_ISA})")

add_library(dagcodelets STATIC ${VFFT_CODELET_SRCS})
target_include_directories(dagcodelets PRIVATE ${VFFT_INCLUDE_DIRS})
target_compile_options(dagcodelets PRIVATE ${VFFT_CODE_FLAGS})

# ── The benchmark (the ONLY driver wired) ───────────────────────────────────
add_executable(bench_1d_vs_mkl "${CMAKE_SOURCE_DIR}/build_tuned/benches/bench_1d_vs_mkl.c")
target_include_directories(bench_1d_vs_mkl PRIVATE ${VFFT_INCLUDE_DIRS} "${MKL_INCLUDE_DIR}")
target_compile_definitions(bench_1d_vs_mkl PRIVATE ${VFFT_MKL_DEFS} _CRT_SECURE_NO_WARNINGS)
target_compile_options(bench_1d_vs_mkl PRIVATE ${VFFT_C_FLAGS})
target_link_libraries(bench_1d_vs_mkl PRIVATE dagcodelets ${VFFT_MKL_LIBS})

# libm: GNU front ends need it (mingw libm on Windows, libm on Linux); MSVC CRT
# and the ICX-on-Windows MSVC CRT supply it, so skip there (build.py parity).
if(NOT VFFT_MSVC_STYLE AND NOT (WIN32 AND VFFT_IS_ICX))
    target_link_libraries(bench_1d_vs_mkl PRIVATE m)
endif()
# Linux LP64 mkl_rt also pulls pthread + dl.
if(UNIX AND NOT VFFT_MSVC_STYLE)
    target_link_libraries(bench_1d_vs_mkl PRIVATE pthread dl)
endif()
# ICX on Windows links through lld (build.py: -fuse-ld=lld).
if(WIN32 AND VFFT_IS_ICX)
    target_link_options(bench_1d_vs_mkl PRIVATE -fuse-ld=lld)
endif()

message(STATUS "")
message(STATUS "  VectorFFT (dag) — bench_1d_vs_mkl")
message(STATUS "  Build:    ${CMAKE_BUILD_TYPE}   Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Front-end: ${CMAKE_C_COMPILER} (MSVC-style=${VFFT_MSVC_STYLE}, ICX=${VFFT_IS_ICX})")
message(STATUS "  ISA:      ${VFFT_ISA}   codelets: ${_ncodelets}")
message(STATUS "  Run:      put MKL bin (+ mingw bin for gcc) on PATH, then ./bench_1d_vs_mkl [--mt|--oop]")
message(STATUS "")
