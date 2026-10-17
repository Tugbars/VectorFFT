# ============================================================================
# VectorFFT — the library, the examples and the gauntlet
#
#   - VectorFFT::vfft  : the library, ONE static archive (libvfft.a): the dag SIMD
#     codelets (codelets/<family>/<isa>/*.c, an OBJECT library compiled once) plus
#     the header-only core (src/core/, bare cross-includes -> recursive -I walk)
#     compiled through its one .c, src/core/vfft.c. Consumers get include/ only.
#   - how-to-use/examples/ (VFFT_BUILD_EXAMPLES) and the gauntlet's tools, gauntlet/
#     (VFFT_BUILD_GAUNTLET), with Intel MKL as the gauntlet's OPTIONAL comparator.
#   Built as a subproject (add_subdirectory, FetchContent) it is the library alone:
#   no gauntlet, no examples, no MKL search, no install rules, no build type set.
#   The DFTI interface (src/shims/dfti/) ships as two source files: a program
#   compiles vfft_dfti.c with its own files and puts src/shims/dfti/ ahead of
#   MKL's include folder (src/shims/dfti/mkl_dfti.h).
#
# Configure, build, install (how-to-use/):
#   cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=<prefix> -DVFFT_BUILD_GAUNTLET=OFF
#   cmake --build build
#   cmake --install build
#   (-DVFFT_NATIVE=OFF for programs that run on other machines: -march=native is the default)
# The gauntlet: cmake --build build --target vfft-gauntlet (gauntlet/README.md).
# Run (Windows): for mingw, put libwinpthread-1.dll on PATH (or link with -static);
#   for the gauntlet's MKL arm, MKL's bin (mkl_rt*.dll) as well.
#
# Toolchains: GCC and Clang (GNU-style front ends; icx in its gcc driver mode too).
#   The MSVC-style branch below is kept for the gauntlet; MSVC is not supported.
# ISA override: -DVFFT_ISA=avx2|avx512|auto (default avx2). -DVFFT_NATIVE=ON|OFF.
# ============================================================================
cmake_minimum_required(VERSION 3.16)
# A parent's ORDINARY variable (set(VFFT_WISDOM ...) before add_subdirectory) is
# honoured on the first configure too: without CMP0126 NEW, set(CACHE) below would
# drop it until the second. CMake 3.21+ knows the policy; older ones keep the old way.
if(POLICY CMP0126)
    cmake_policy(SET CMP0126 NEW)
endif()
project(VectorFFT_dag VERSION 2.0.0 LANGUAGES C)

# ── Top level or a subproject (PROJECT_IS_TOP_LEVEL needs CMake 3.21) ────────
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(VFFT_IS_TOP_LEVEL ON)
else()
    set(VFFT_IS_TOP_LEVEL OFF)
endif()
option(VFFT_BUILD_GAUNTLET "Build the gauntlet's tools (gauntlet/), MKL discovery included" ${VFFT_IS_TOP_LEVEL})
option(VFFT_BUILD_EXAMPLES "Build the examples (how-to-use/examples/)" ${VFFT_IS_TOP_LEVEL})
option(VFFT_INSTALL        "Install rules, the CMake package and the pkg-config file" ${VFFT_IS_TOP_LEVEL})
option(VFFT_NATIVE         "Compile for this machine's CPU (-march=native): the binaries may not run on another" ON)

# A subproject never sets its parent's (cache-wide) build type.
if(VFFT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)

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
# PROJECT_SOURCE_DIR, never CMAKE_SOURCE_DIR: under add_subdirectory/FetchContent
# CMAKE_SOURCE_DIR is the consumer's tree.
set(DAG          "${PROJECT_SOURCE_DIR}/src/dag-fft-compiler")
set(DAG_CORE     "${PROJECT_SOURCE_DIR}/src/core")
set(DAG_GEN      "${DAG}/generator/generated")
set(VFFT_WISDOM  "${PROJECT_SOURCE_DIR}/src/wisdom" CACHE PATH
    "The build tree's wisdom store root, one folder per CPU: the gauntlet's tools serve from and save into this CPU's folder under it, and so does VectorFFT::vfft when VFFT_INSTALL is OFF (an installed library uses its installed store)")
if(VFFT_WISDOM STREQUAL "")
    message(FATAL_ERROR "VFFT_WISDOM is empty: name a directory. An empty root would compile the current drive's root as the store.")
endif()
# Forward slashes: the path becomes a C string, where a backslash starts an escape
# (a :STRING value or a parent's variable keeps its backslashes; a PATH value does not).
file(TO_CMAKE_PATH "${VFFT_WISDOM}" VFFT_WISDOM)

# A path compiled into a library as a C string (VFFT_WISDOM_DIR_DEFAULT,
# VFFT_FROZEN_WISDOM_DIR) is opened as it stands: it must be absolute, or each
# program would resolve it against its own working directory. On Windows the store
# is opened with the ANSI file calls, which do not find a path outside ASCII.
function(vfft_check_compiled_path _path)
    if(NOT IS_ABSOLUTE "${_path}")
        message(FATAL_ERROR "'${_path}' would be compiled into the library as a store or bundle directory, and it is "
                            "not absolute: every program would resolve it against its own working directory. Name an "
                            "absolute directory (VFFT_WISDOM, CMAKE_INSTALL_PREFIX).")
    endif()
    if(WIN32 AND _path MATCHES "[^ -~]")
        message(WARNING "'${_path}' (compiled into the library as a store or bundle directory) holds a character "
                        "outside ASCII: the library opens it with the Windows ANSI file calls, which do not find it, "
                        "so its programs neither read nor save wisdom there. Use an ASCII path.")
    endif()
endfunction()

# The gauntlet's store and frozen bundle: the clone's, as build.py compiles them.
# The frozen bundle (spike_wisdom.txt, bluestein_wisdom.txt, c2r_path.txt) stays
# in generator/generated/ and is read from there whatever store is given.
set(VFFT_WISDOM_DEFS
    "VFFT_WISDOM_DIR_DEFAULT=\"${VFFT_WISDOM}\""
    "VFFT_FROZEN_WISDOM_DIR=\"${DAG_GEN}\"")
# The build stamp of banked wisdom rows, bld=<version>-<commit> (src/core/vfft.c):
# the last commit that touched the library's sources, -dirty when they carry
# uncommitted changes. Read at CONFIGURE time; without git the library stamps
# its version alone.
set(VFFT_STAMP_DEFS)
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" log -1 --format=%h --abbrev=8 -- src/core include
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE VFFT_BUILD_COMMIT
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain -- src/core include
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE VFFT_BUILD_DIRTY
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(VFFT_BUILD_COMMIT)
        if(VFFT_BUILD_DIRTY)
            string(APPEND VFFT_BUILD_COMMIT "-dirty")
        endif()
        list(APPEND VFFT_STAMP_DEFS "VFFT_BUILD_COMMIT=\"${VFFT_BUILD_COMMIT}\"")
    endif()
endif()
list(APPEND VFFT_WISDOM_DEFS ${VFFT_STAMP_DEFS})
set(DAG_JIT      "${DAG}/jit")
set(DAG_CODELETS "${DAG}/codelets")

# Resolve VFFT_ISA to a CONCRETE codelet ISA. Codelets are pre-generated per ISA
# (codelets/*/avx2, codelets/*/avx512) — there is no "auto" tree.
#   - explicit avx2 / avx512  -> trust the user.
#   - auto (what the CMake Tools extension may hold cached from the old root
#     CMake) -> DETECT the host: pick avx512 only if the avx512 codelets
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
            "${PROJECT_BINARY_DIR}/_avx512_probe"
            "${PROJECT_SOURCE_DIR}/cmake/check_avx512.c"
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
# PRIVATE to the library: no consumer ever sees these directories.
file(GLOB_RECURSE _core_headers "${DAG_CORE}/*.h")
set(_core_inc_dirs "${DAG_CORE}")
foreach(_h ${_core_headers})
    get_filename_component(_d "${_h}" DIRECTORY)
    list(APPEND _core_inc_dirs "${_d}")
endforeach()
list(REMOVE_DUPLICATES _core_inc_dirs)

set(VFFT_INCLUDE_DIRS
    "${PROJECT_SOURCE_DIR}/include"
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

# ── MKL discovery: the gauntlet's comparator only (the library has no MKL code) ─
if(VFFT_BUILD_GAUNTLET)
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
        message(WARNING "MKL not found (mkl_dfti.h): building the gauntlet WITHOUT the MKL comparator. Set MKLROOT or install Intel oneAPI MKL for the vs-MKL columns.")
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
endif()

# ── Per-front-end compile flags (PRIVATE to the library and the gauntlet) ───
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
    # ISA CLAMP (build.py's, mirrored 2026-09-26): on a CPU whose -march=native
    # implies AVX-512 (Zen 4+, Sapphire Rapids+) an avx2 build would otherwise
    # define __AVX512F__ -- the core then dispatches to avx512 registries the
    # avx2 codelet library does not carry (undefined radix*_avx512 at link) --
    # and the avx2 codelets themselves would be compiled with EVEX encodings
    # and ymm16-31: not an AVX2 binary, and not the 16-register code the avx2
    # kernels were scheduled for. The selected ISA is the source of truth; it
    # must follow -march=native to override it. Native tuning is kept. The clamp
    # stays without VFFT_NATIVE: a parent project's -march=native (CMAKE_C_FLAGS,
    # add_compile_options) comes before these options on the command line.
    set(_isa_clamp)
    if(NOT VFFT_ISA STREQUAL "avx512")
        set(_isa_clamp -mno-avx512f)
    endif()
    set(_native)
    if(VFFT_NATIVE)
        set(_native -march=native)
    endif()
    set(VFFT_C_FLAGS -O3 ${_arch} ${_native} ${_isa_clamp} -fpermissive
        -Wno-overflow -Wno-implicit-function-declaration -Wno-unused-function
        -Wno-unknown-argument -Wno-incompatible-pointer-types -Wno-deprecated-declarations)
    set(VFFT_CODE_FLAGS -O3 ${_arch} ${_native} ${_isa_clamp} -fpermissive -w)
endif()

# ── Codelet families (inplace=c2c, rfft/c2r=real-FFT, oop=LEAF/BAILEY2,
#     strided=6a35-45 row engines, zil=interleaved-complex (pure IL + the
#     N>=2048 boundary-split cascade), trig=DCT/DST) ──
# NOTE `il` below is RETIRED (its derived population was deleted 2026-07-24;
# the dir is kept only as the once-home) and globs to nothing. It stays in the
# list purely for parity with build.py's dag_codelet_srcs(). The LIVE
# interleaved family is `zil`, handled by the second loop.
# *_emit.c are generator bit-gate twins of the hand codelets (same symbol names);
# linking both would collide — the gates compile them standalone. Mirrors
# gauntlet/build.py dag_codelet_srcs().
# Families are discovered ONE AT A TIME and each REPORTS ITS OWN COUNT.
# Rationale, learned the hard way on 2026-08-15: this file globbed 7 families
# and printed a single plausible-looking "codelets: 598", while the three zil
# dirs (265 files) were absent for weeks -- one aggregate number cannot show
# you a family that is missing. So: per-family counts, and a family that globs
# to ZERO is a FATAL ERROR unless it is declared retired below.
set(VFFT_RETIRED_FAMILIES il)  # dir kept as the once-home; population deleted 2026-07-24

# Families that legitimately have NO tree for a given ISA, DECLARED per ISA.
# This is deliberately a declaration and not a discovery: an absence nobody
# wrote down is a hard error below. The zil avx512 tree landed 2026-09-26
# (716 corpus cells); only pair2p/tangent has no avx512 folder -- its 14
# kernels exist only as env-knob/sed recipes outside the corpus
# (docs/roadmap/zil_avx512_design.md, decision D3).
set(VFFT_ABSENT_avx2)                      # avx2 carries the full corpus
set(VFFT_ABSENT_avx512 zil/shared/col/half zil/pair2p/tangent zil/real/pair zil/real/mono zil/real/flat zil/real/flat/herm zil/real/rows)

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
                "  was dropped. Re-sync this glob with gauntlet/build.py\n"
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
# The zil family sits one level DEEPER (zil/<isa>/<folder>, one folder per engine
# since 2026-09-24: docs/roadmap/il_codelet_reorg.md), so it never matched the <fam>/<isa> pattern above and was
# simply ABSENT -- THE 598-vs-863 gap, exactly 265 files. Globs are
# non-recursive, so pure_il and pure_il/tangent are listed separately
# (build.py does the same).
foreach(_zdir shared shared/col shared/col/blocked shared/col/half mono pair2p pair2p/blocked pair2p/tangent rows chain3 flat flat/odd_mid ztt real/pair real/mono real/flat real/flat/herm real/rows)
    vfft_add_codelet_dir("zil/${_zdir}" "${DAG_CODELETS}/zil/${VFFT_ISA}/${_zdir}")
endforeach()
# ZTURN-T's FUSED CODELETS (2026-09-09): one whole-transform driver per pow2 cell,
# emitted under generated/ (build.py's dag_codelet_srcs carries the same dir);
# missing from this list until 2026-09-22, when every link of vfft.c failed on
# ztt_*/zttp_* symbols. The avx512 drivers sit in their own subfolder (design
# decision D5), which the non-recursive glob keeps out of the avx2 build.
if(VFFT_ISA STREQUAL "avx2")
    vfft_add_codelet_dir("fused_codelets" "${DAG_GEN}/fused_codelets")
elseif(VFFT_ISA STREQUAL "avx512")
    vfft_add_codelet_dir("fused_codelets/avx512" "${DAG_GEN}/fused_codelets/avx512")
endif()

list(LENGTH VFFT_CODELET_SRCS _ncodelets)
if(_ncodelets EQUAL 0)
    message(FATAL_ERROR "no codelet .c found under ${DAG_CODELETS}/*/${VFFT_ISA}/")
endif()
message(STATUS "    ----------------------------")
message(STATUS "    TOTAL                   ${_ncodelets}")
message(STATUS "  (must equal gauntlet/build.py's '[codelets] cached lib (N codelets)';")
message(STATUS "   the two build systems share one corpus -- a mismatch is a real defect)")
message(STATUS "  runtime source:  ${DAG_CORE}/vfft.c  (the ONE non-header .c in the core)")

# ── The codelet corpus: an OBJECT library, compiled ONCE and archived into
#    VectorFFT::vfft (below) and the gauntlet's vfft_core alike ──
add_library(dagcodelets OBJECT ${VFFT_CODELET_SRCS})
target_include_directories(dagcodelets PRIVATE ${VFFT_INCLUDE_DIRS})
target_compile_options(dagcodelets PRIVATE ${VFFT_CODE_FLAGS})

find_package(Threads REQUIRED)

# ── THE LIBRARY: VectorFFT::vfft = libvfft.a (src/core/vfft.c + every codelet) ─
# Its store root and frozen bundle are compiled in (src/core/vfft.c reads them):
#   installed (VFFT_INSTALL=ON): <prefix>/<datadir>/vectorfft/{wisdom,frozen},
#     the absolute path of the prefix configured now (the install refuses another);
#   build tree (VFFT_INSTALL=OFF): VFFT_WISDOM and the clone's generator/generated.
# Everything but include/ is PRIVATE: no ISA flag, no internal directory, no path
# reaches a consumer.
if(VFFT_INSTALL)
    file(TO_CMAKE_PATH "${CMAKE_INSTALL_FULL_DATADIR}/vectorfft" VFFT_LIB_DATA_DIR)
    set(VFFT_LIB_STORE    "${VFFT_LIB_DATA_DIR}/wisdom")
    set(VFFT_LIB_FROZEN   "${VFFT_LIB_DATA_DIR}/frozen")
else()
    set(VFFT_LIB_DATA_DIR "")
    set(VFFT_LIB_STORE    "${VFFT_WISDOM}")
    set(VFFT_LIB_FROZEN   "${DAG_GEN}")
endif()
set(_vfft_compiled "${VFFT_LIB_STORE}" "${VFFT_LIB_FROZEN}")
if(VFFT_BUILD_GAUNTLET)
    list(APPEND _vfft_compiled "${VFFT_WISDOM}" "${DAG_GEN}")   # vfft_core's (VFFT_WISDOM_DEFS)
endif()
list(REMOVE_DUPLICATES _vfft_compiled)
foreach(_p IN LISTS _vfft_compiled)
    vfft_check_compiled_path("${_p}")
endforeach()

# An installed store inside this source tree would be written by every program
# that uses the installed library. Configure and build go on (the gauntlet is
# untouched); the install refuses such a prefix (its first rule, below).
set(VFFT_PREFIX_IN_SOURCE FALSE)
if(VFFT_INSTALL)
    file(RELATIVE_PATH _vfft_rel "${PROJECT_SOURCE_DIR}" "${VFFT_LIB_DATA_DIR}")
    if(NOT IS_ABSOLUTE "${_vfft_rel}" AND NOT _vfft_rel MATCHES "^[.][.]")
        set(VFFT_PREFIX_IN_SOURCE TRUE)
        message(WARNING "CMAKE_INSTALL_PREFIX (${CMAKE_INSTALL_PREFIX}) lies inside the VectorFFT source tree "
                        "(${PROJECT_SOURCE_DIR}): every program using the installed library would save wisdom under "
                        "${VFFT_LIB_STORE}, so cmake --install refuses this prefix. The build goes on. To install, "
                        "configure with a prefix outside the source tree.")
    endif()
endif()

add_library(vectorfft_vfft STATIC "${DAG_CORE}/vfft.c" $<TARGET_OBJECTS:dagcodelets>)
add_library(VectorFFT::vfft ALIAS vectorfft_vfft)
# a target name of its own (a consumer may own a target called vfft); the file is
# libvfft.a and the exported name VectorFFT::vfft
set_target_properties(vectorfft_vfft PROPERTIES OUTPUT_NAME vfft EXPORT_NAME vfft)
target_include_directories(vectorfft_vfft
    PUBLIC  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE ${VFFT_INCLUDE_DIRS})
target_compile_options(vectorfft_vfft PRIVATE ${VFFT_C_FLAGS})
target_compile_definitions(vectorfft_vfft PRIVATE ${VFFT_STAMP_DEFS} _CRT_SECURE_NO_WARNINGS
    "VFFT_WISDOM_DIR_DEFAULT=\"${VFFT_LIB_STORE}\""
    "VFFT_FROZEN_WISDOM_DIR=\"${VFFT_LIB_FROZEN}\"")
target_link_libraries(vectorfft_vfft PRIVATE Threads::Threads)   # a static archive exports these as $<LINK_ONLY:>
if(NOT VFFT_MSVC_STYLE)
    target_link_libraries(vectorfft_vfft PRIVATE m)
endif()
# A parent project's flags reach these targets as well (CMAKE_C_FLAGS,
# add_compile_options, CMAKE_COMPILE_WARNING_AS_ERROR): its -Werror would stop
# vfft.c on three default warnings, its -ffast-math would change the library's
# rounding. Both are undone after them. Nothing here applies at the top level.
if(NOT VFFT_IS_TOP_LEVEL AND NOT VFFT_MSVC_STYLE)
    target_compile_options(vectorfft_vfft PRIVATE -Wno-error -fno-fast-math)
    target_compile_options(dagcodelets PRIVATE -fno-fast-math)
    set_target_properties(vectorfft_vfft PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
endif()

# ── Install: the archive, the headers, the package, pkg-config, the store ────
# Before every add_subdirectory below: under CMP0082 a subdirectory's install
# rules run where it is added, and the guards must run first.
if(VFFT_INSTALL)
    # FIRST RULES: libvfft.a carries this prefix's data path (VFFT_LIB_STORE). A
    # prefix inside the source tree is refused, and so is an install to another
    # prefix (cmake --install --prefix), which would leave that path empty. DESTDIR
    # staging keeps the prefix and passes.
    if(VFFT_PREFIX_IN_SOURCE)
        install(CODE "message(FATAL_ERROR \"VectorFFT was configured for the prefix ${CMAKE_INSTALL_PREFIX}, inside its source tree ${PROJECT_SOURCE_DIR}: every program using the installed library would save wisdom under ${VFFT_LIB_STORE}. Reconfigure with a prefix outside the source tree, then install.\")")
    endif()
    install(CODE "
      get_filename_component(_vfft_to   \"\${CMAKE_INSTALL_PREFIX}\" ABSOLUTE)
      get_filename_component(_vfft_conf \"${CMAKE_INSTALL_PREFIX}\" ABSOLUTE)
      if(WIN32)
        string(TOLOWER \"\${_vfft_to}\" _vfft_to)
        string(TOLOWER \"\${_vfft_conf}\" _vfft_conf)
      endif()
      if(NOT _vfft_to STREQUAL _vfft_conf)
        message(FATAL_ERROR \"VectorFFT was configured for the prefix ${CMAKE_INSTALL_PREFIX}: its library opens its wisdom store under ${VFFT_LIB_STORE}. Installing it to \${CMAKE_INSTALL_PREFIX} would leave that path empty. Reconfigure with -DCMAKE_INSTALL_PREFIX=\${CMAKE_INSTALL_PREFIX}, then install.\")
      endif()")

    set(VFFT_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/VectorFFT")
    set(VFFT_DATA_DIR  "${CMAKE_INSTALL_DATADIR}/vectorfft")   # relative: VectorFFT_DATA_DIR follows the files

    install(TARGETS vectorfft_vfft EXPORT VectorFFTTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES "${PROJECT_SOURCE_DIR}/include/vfft.h" "${PROJECT_SOURCE_DIR}/include/vfft_diagnostics.h"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # pkg-config: the prefix relative to the .pc file's own folder; the system
    # libraries go in Libs, as the library is a static archive only
    file(RELATIVE_PATH VFFT_PC_RELPREFIX "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
    string(REGEX REPLACE "/$" "" VFFT_PC_RELPREFIX "${VFFT_PC_RELPREFIX}")
    string(STRIP "${CMAKE_THREAD_LIBS_INIT} -lm" VFFT_PC_SYSLIBS)
    configure_file("${PROJECT_SOURCE_DIR}/cmake/vectorfft.pc.in" "${PROJECT_BINARY_DIR}/vectorfft.pc" @ONLY)
    install(FILES "${PROJECT_BINARY_DIR}/vectorfft.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

    install(EXPORT VectorFFTTargets NAMESPACE VectorFFT:: DESTINATION ${VFFT_CMAKE_DIR})
    include(CMakePackageConfigHelpers)
    configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/VectorFFTConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/VectorFFTConfig.cmake"
        INSTALL_DESTINATION ${VFFT_CMAKE_DIR}
        PATH_VARS VFFT_DATA_DIR)
    # PROJECT_VERSION, as cmake/vectorfft.pc.in reports it today; the release's
    # number for the package files is the owner's call
    write_basic_package_version_file("${PROJECT_BINARY_DIR}/VectorFFTConfigVersion.cmake"
        VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
    install(FILES "${PROJECT_BINARY_DIR}/VectorFFTConfig.cmake" "${PROJECT_BINARY_DIR}/VectorFFTConfigVersion.cmake"
            DESTINATION ${VFFT_CMAKE_DIR})

    # runtime data: the shipped store (the live shards of every CPU folder, new/
    # included; no backup, no lock, no README) and the frozen bundle, the three
    # files src/core/vfft.c reads from it (c2r_path.txt when it exists)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/wisdom/" DESTINATION ${VFFT_DATA_DIR}/wisdom
            FILES_MATCHING PATTERN "wisdom2_*.txt")
    install(FILES "${DAG_GEN}/spike_wisdom.txt" "${DAG_GEN}/bluestein_wisdom.txt"
            DESTINATION ${VFFT_DATA_DIR}/frozen)
    install(FILES "${DAG_GEN}/c2r_path.txt" DESTINATION ${VFFT_DATA_DIR}/frozen OPTIONAL)
endif()

# ── The gauntlet's tools (the test tool; top-level builds) ───────────────────
# They live in gauntlet/ (2026-09-22): the user-facing measurement path
# (calibrate / bench / report / merge), see gauntlet/README.md. Their library,
# vfft_core, is the gauntlet's own: the clone's store, the internal include walk
# PUBLIC for the two tools that compile engine internals; never installed.
# build_tuned/ was retired 2026-10-07; the gates live in src/tools/gates/.
if(VFFT_BUILD_GAUNTLET)
    add_subdirectory(gauntlet)
endif()

# ── The examples of the guide ────────────────────────────────────────────────
if(VFFT_BUILD_EXAMPLES)
    add_subdirectory(how-to-use/examples)
endif()

message(STATUS "")
message(STATUS "  VectorFFT (dag) — library + examples + gauntlet")
message(STATUS "  Build:    ${CMAKE_BUILD_TYPE}   Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Front-end: ${CMAKE_C_COMPILER} (MSVC-style=${VFFT_MSVC_STYLE}, ICX=${VFFT_IS_ICX})")
message(STATUS "  ISA:      ${VFFT_ISA}   native: ${VFFT_NATIVE}   codelets: ${_ncodelets}")
message(STATUS "  Store:    ${VFFT_LIB_STORE}   (VectorFFT::vfft's compiled root)")
message(STATUS "  Parts:    examples=${VFFT_BUILD_EXAMPLES} gauntlet=${VFFT_BUILD_GAUNTLET} install=${VFFT_INSTALL}")
if(VFFT_BUILD_GAUNTLET)
    message(STATUS "  Gauntlet: cmake --build <build> --target vfft-gauntlet, then python gauntlet/gauntlet.py run --group pow2 --bin-dir <build>/gauntlet")
endif()
message(STATUS "")
