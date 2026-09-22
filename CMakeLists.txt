# ============================================================================
# VectorFFT — dag-fft-compiler build (CMake mirror of build_tuned/build.py)
#
# Builds the library's codelet corpus and the gauntlet's tools (gauntlet/) against the dag runtime:
#   - the dag SIMD codelets (codelets/{inplace,rfft,c2r,oop}/<isa>/*.c) -> a
#     static lib (the only .c the header-only core needs linked in),
#   - the header-only core in src/core/ (bare cross-includes -> recursive -I walk),
#   - Intel MKL for the vs-MKL comparison (OPTIONAL since 2026-09-22).
#
# The gauntlet's binaries are the only executables wired (gauntlet/CMakeLists.txt).
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

set(VFFT_ISA "avx2" CACHE STRING "Codelet ISA (avx2/avx512/auto)")

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

# Resolve VFFT_ISA to a CONCRETE codelet ISA. Codelets are pre-generated per ISA
# (codelets/*/avx2, codelets/*/avx512) — there is no "auto" tree.
#   - explicit avx2 / avx512  -> trust the user.
#   - auto (the default, and what the CMake Tools extension may hold cached from the
#     old root CMake) -> DETECT the host: pick avx512 only if the avx512 codelets
#     exist AND this CPU actually runs AVX-512 (try_run compiles an _mm512 op and
#     executes it — it SIGILLs / returns non-zero on a non-AVX-512 CPU). Else avx2.
if(VFFT_ISA STREQUAL "avx2" OR VFFT_ISA STREQUAL "avx512")
    # explicit override — leave as-is
elseif(VFFT_ISA STREQUAL "auto")
    set(VFFT_ISA "avx2")   # safe fallback (also covers cross-compiling)
    file(GLOB _a512_probe "${DAG_CODELETS}/inplace/avx512/*.c")
    if(_a512_probe AND NOT CMAKE_CROSSCOMPILING)
        if(VFFT_MSVC_STYLE)
            set(_av512_flag "/arch:AVX512")
        else()
            set(_av512_flag "-mavx512f -mavx512dq -mfma")
        endif()
        try_run(_av512_run _av512_compile
            "${CMAKE_BINARY_DIR}/_avx512_probe"
            "${CMAKE_SOURCE_DIR}/cmake/check_avx512.c"
            CMAKE_FLAGS "-DCMAKE_C_FLAGS=${_av512_flag}")
        if(_av512_compile AND _av512_run EQUAL 0)
            set(VFFT_ISA "avx512")
        endif()
    endif()
    message(STATUS "VFFT_ISA=auto -> detected ${VFFT_ISA} (host CPU)")
else()
    message(WARNING "unknown VFFT_ISA='${VFFT_ISA}', using avx2")
    set(VFFT_ISA "avx2")
endif()

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
# MKL is OPTIONAL since 2026-09-22 (the gauntlet is usable without a comparator:
# absolute numbers + correctness). VFFT_MKL_FOUND gates every MKL use below.
set(VFFT_MKL_FOUND FALSE)
set(VFFT_MKL_LIBS)
set(VFFT_MKL_DEFS)
if(NOT MKL_INCLUDE_DIR)
    message(WARNING "MKL not found (mkl_dfti.h): building WITHOUT the MKL comparator. Set MKLROOT or install Intel oneAPI MKL for the vs-MKL columns.")
elseif(VFFT_MSVC_STYLE)
    find_library(MKL_ILP64 NAMES mkl_intel_ilp64 HINTS ${_mkl_lib_hints})
    find_library(MKL_SEQ   NAMES mkl_sequential  HINTS ${_mkl_lib_hints})
    find_library(MKL_CORE  NAMES mkl_core        HINTS ${_mkl_lib_hints})
    if(NOT MKL_ILP64 OR NOT MKL_SEQ OR NOT MKL_CORE)
        message(WARNING "MKL ILP64 libs not found (mkl_intel_ilp64/mkl_sequential/mkl_core): building WITHOUT the MKL comparator.")
    else()
        set(VFFT_MKL_FOUND TRUE)
        set(VFFT_MKL_LIBS ${MKL_ILP64} ${MKL_SEQ} ${MKL_CORE})
        set(VFFT_MKL_DEFS VFFT_HAS_MKL MKL_ILP64)
    endif()
else()
    find_library(MKL_RT NAMES mkl_rt mkl_rt.2 libmkl_rt HINTS ${_mkl_lib_hints})
    if(NOT MKL_RT)
        message(WARNING "MKL LP64 lib not found (mkl_rt): building WITHOUT the MKL comparator.")
    else()
        set(VFFT_MKL_FOUND TRUE)
        set(VFFT_MKL_LIBS ${MKL_RT})
        set(VFFT_MKL_DEFS VFFT_HAS_MKL)
    endif()
endif()
if(VFFT_MKL_FOUND)
    message(STATUS "MKL include: ${MKL_INCLUDE_DIR}")
    message(STATUS "MKL libs:    ${VFFT_MKL_LIBS}")
endif()

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

# ── Codelet static lib (inplace=c2c, rfft/c2r=real-FFT, oop=LEAF/BAILEY2,
#     strided=6a35-45 row engines, zil=interleaved-complex (pure IL + the
#     N>=2048 boundary-split cascade), trig=DCT/DST) ──
# NOTE `il` below is RETIRED (its derived population was deleted 2026-07-24;
# the dir is kept only as the once-home) and globs to nothing. It stays in the
# list purely for parity with build.py's dag_codelet_srcs(). The LIVE
# interleaved family is `zil`, handled by the second loop.
# *_emit.c are generator bit-gate twins of the hand codelets (same symbol names);
# linking both would collide — the gates compile them standalone. Mirrors
# build_tuned/build.py dag_codelet_srcs().
# Families are discovered ONE AT A TIME and each REPORTS ITS OWN COUNT.
# Rationale, learned the hard way on 2026-08-15: this file globbed 7 families
# and printed a single plausible-looking "codelets: 598", while the three zil
# dirs (265 files) were absent for weeks -- one aggregate number cannot show
# you a family that is missing. So: per-family counts, and a family that globs
# to ZERO is a FATAL ERROR unless it is declared retired below.
set(VFFT_RETIRED_FAMILIES il)  # dir kept as the once-home; population deleted 2026-07-24

# Families that legitimately have NO tree for a given ISA, DECLARED per ISA.
# This is deliberately a declaration and not a discovery: an absence nobody
# wrote down is a hard error below. zil (interleaved-complex) is avx2-only
# today and AVX-512 support is planned -- when those kernels land, delete the
# zil entries from VFFT_ABSENT_avx512 and the guard immediately starts
# protecting them, with no other edit needed.
set(VFFT_ABSENT_avx2)                      # avx2 carries the full corpus
set(VFFT_ABSENT_avx512
    zil/pure_il zil/pure_il/tangent zil/boundary_split)

set(VFFT_CODELET_SRCS)

# label, dir. Appends to VFFT_CODELET_SRCS in the caller's scope (macro, not
# function -- a function would need PARENT_SCOPE gymnastics for every append).
macro(vfft_add_codelet_dir _label _dir)
    file(GLOB _g "${_dir}/*.c")
    # *_emit.c are generator bit-gate twins of the hand codelets (same symbol
    # names); linking both would collide, so the gates compile them standalone.
    list(FILTER _g EXCLUDE REGEX "_emit[.]c$")
    list(LENGTH _g _n)

    list(FIND VFFT_RETIRED_FAMILIES "${_label}" _retired_idx)
    set(_note "")
    if(_retired_idx GREATER -1)
        set(_note "   (RETIRED -- kept for build.py parity)")
    elseif(_n EQUAL 0)
        if(EXISTS "${_dir}")
            # The directory is THERE and empty: the tree layout moved, or a
            # population was dropped without updating this glob. That is a
            # regression, not a configuration -- fail loudly.
            message(FATAL_ERROR
                "codelet family '${_label}' exists but matched 0 .c files:\n"
                "    ${_dir}\n"
                "  The directory is present and EMPTY -- the tree moved or a population\n"
                "  was dropped. Re-sync this glob with build_tuned/build.py\n"
                "  dag_codelet_srcs(), or add '${_label}' to VFFT_RETIRED_FAMILIES if it\n"
                "  is deliberately gone. (A family silently contributing zero is what hid\n"
                "  the CMake-598 vs build.py-863 gap: 265 zil files absent behind a\n"
                "  plausible-looking total.)")
        else()
            # No such directory for this ISA. Legitimate ONLY if declared.
            list(FIND VFFT_ABSENT_${VFFT_ISA} "${_label}" _absent_idx)
            if(_absent_idx EQUAL -1)
                message(FATAL_ERROR
                    "codelet family '${_label}' has no ${VFFT_ISA} tree:\n"
                    "    ${_dir}\n"
                    "  and it is NOT declared in VFFT_ABSENT_${VFFT_ISA}. Either the family\n"
                    "  really is unavailable for this ISA -- in which case DECLARE it there,\n"
                    "  so the omission is written down instead of merely happening -- or the\n"
                    "  glob is wrong. An undeclared absence is how avx2 shipped 598 of 863\n"
                    "  codelets for weeks behind a plausible-looking total.")
            endif()
            set(_note "   (declared absent for ${VFFT_ISA})")
        endif()
    endif()

    # pad the label so the counts line up in the configure log
    string(LENGTH "${_label}" _ll)
    math(EXPR _padn "24 - ${_ll}")
    if(_padn LESS 1)
        set(_padn 1)
    endif()
    string(SUBSTRING "                          " 0 ${_padn} _pad)
    message(STATUS "    ${_label}${_pad}${_n}${_note}")

    list(APPEND VFFT_CODELET_SRCS ${_g})
endmacro()

message(STATUS "codelet families (${VFFT_ISA}):")
foreach(_fam inplace rfft c2r oop strided il trig)
    vfft_add_codelet_dir("${_fam}" "${DAG_CODELETS}/${_fam}/${VFFT_ISA}")
endforeach()
# The zil family sits one level DEEPER (zil/<isa>/{pure_il, pure_il/tangent,
# boundary_split}), so it never matched the <fam>/<isa> pattern above and was
# simply ABSENT -- THE 598-vs-863 gap, exactly 265 files. Globs are
# non-recursive, so pure_il and pure_il/tangent are listed separately
# (build.py does the same).
foreach(_zdir pure_il pure_il/tangent boundary_split)
    vfft_add_codelet_dir("zil/${_zdir}" "${DAG_CODELETS}/zil/${VFFT_ISA}/${_zdir}")
endforeach()

list(LENGTH VFFT_CODELET_SRCS _ncodelets)
if(_ncodelets EQUAL 0)
    message(FATAL_ERROR "no codelet .c found under ${DAG_CODELETS}/*/${VFFT_ISA}/")
endif()
message(STATUS "    ----------------------------")
message(STATUS "    TOTAL                   ${_ncodelets}")
message(STATUS "  (must equal build_tuned/build.py's '[codelets] cached lib (N codelets)';")
message(STATUS "   the two build systems share one corpus -- a mismatch is a real defect)")
message(STATUS "  runtime source:  ${DAG_CORE}/vfft.c  (the ONE non-header .c in the core)")

add_library(dagcodelets STATIC ${VFFT_CODELET_SRCS})
target_include_directories(dagcodelets PRIVATE ${VFFT_INCLUDE_DIRS})
target_compile_options(dagcodelets PRIVATE ${VFFT_CODE_FLAGS})

# ── The benchmark (the ONLY driver wired) ───────────────────────────────────
# src/core/vfft.c is the unified public API (vfft_create/execute/destroy/
# wisdom_load). The core is otherwise header-only, so this is the ONE runtime
# .c that must be compiled in; without it the bench fails to link with
# undefined vfft_* symbols. build.py does the same behind its --vfft flag
# (the retired src/vfft.c this replaced is what the orphaned src/CMakeLists.txt
# still names).
# The bench and the tools that drive it live in gauntlet/ (2026-09-22): the
# user-facing measurement path (calibrate / bench / report / merge), see
# gauntlet/README.md. build_tuned/ is the development scratch space, retiring.
add_subdirectory(gauntlet)

message(STATUS "")
message(STATUS "  VectorFFT (dag) — library + gauntlet")
message(STATUS "  Build:    ${CMAKE_BUILD_TYPE}   Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Front-end: ${CMAKE_C_COMPILER} (MSVC-style=${VFFT_MSVC_STYLE}, ICX=${VFFT_IS_ICX})")
message(STATUS "  ISA:      ${VFFT_ISA}   codelets: ${_ncodelets}")
message(STATUS "  Run:      cmake --build <build> --target vfft-gauntlet, then python gauntlet/gauntlet.py run --group pow2 --bin-dir <build>/gauntlet")
message(STATUS "")
