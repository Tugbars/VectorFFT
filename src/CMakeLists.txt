cmake_minimum_required(VERSION 3.16)

# ═══════════════════════════════════════════════════════════════
# VectorFFT v1.0 production library (new core)
#
# Compiles src/vfft.c against the new core in src/core/ and the
# orchestrator-generated codelets in src/vectorfft_tune/generated/r{N}/.
#
# Public surface: include/vfft.h.  Everything else is implementation
# detail (planner, executor, codelets, wisdom).
#
# Options:
#   VFFT_ISA  - Force ISA: "avx512", "avx2", "scalar", "auto"  (default: auto)
# ═══════════════════════════════════════════════════════════════

set(VFFT_ISA "auto" CACHE STRING "Codelet ISA (auto/avx512/avx2/scalar)")

# ── ISA detection ───────────────────────────────────────────────
if(VFFT_ISA STREQUAL "auto")
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "")

    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_avx512_flag "/arch:AVX512")
    else()
        set(_avx512_flag "-mavx512f -mavx512dq -mfma")
    endif()
    set(VFFT_HAS_AVX512 FALSE)
    if(NOT CMAKE_CROSSCOMPILING)
        try_run(_avx512_run _avx512_compile
            "${CMAKE_BINARY_DIR}/avx512_check"
            "${CMAKE_SOURCE_DIR}/cmake/check_avx512.c"
            CMAKE_FLAGS "-DCMAKE_C_FLAGS=${_avx512_flag}"
        )
        if(_avx512_compile AND _avx512_run EQUAL 0)
            set(VFFT_HAS_AVX512 TRUE)
        endif()
    endif()

    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_avx2_flag "/arch:AVX2")
    else()
        set(_avx2_flag "-mavx2 -mfma")
    endif()
    set(CMAKE_REQUIRED_FLAGS "${_avx2_flag}")
    check_c_source_compiles("
        #include <immintrin.h>
        int main() { __m256d a = _mm256_setzero_pd(); (void)a; return 0; }
    " VFFT_HAS_AVX2)
    set(CMAKE_REQUIRED_FLAGS "")

    if(VFFT_HAS_AVX512)
        set(_isa "avx512")
    elseif(VFFT_HAS_AVX2)
        set(_isa "avx2")
    else()
        set(_isa "scalar")
    endif()
else()
    set(_isa "${VFFT_ISA}")
endif()

set(VFFT_STRIDE_ISA "${_isa}" PARENT_SCOPE)
message(STATUS "VectorFFT ISA: ${_isa}")

# ── ISA compile flags + ISA selection define for registry.h ──────
set(_isa_defs)
if(_isa STREQUAL "avx512")
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_isa_flags "/arch:AVX512")
    else()
        set(_isa_flags "-mavx512f;-mavx512dq;-mfma")
    endif()
    set(_isa_defs VFFT_ISA_AVX512)
elseif(_isa STREQUAL "avx2")
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_isa_flags "/arch:AVX2")
    else()
        set(_isa_flags "-mavx2;-mfma")
    endif()
    set(_isa_defs VFFT_ISA_AVX2)
else()
    set(_isa_flags "")
endif()

# ── Codelet include paths ────────────────────────────────────────
# All SIMD codelets live under src/vectorfft_tune/generated/r{N}/.
# R=2 is the bootstrap; R=3..64 are the tuned set; dct8 holds the
# specialized straight-line DCT-II/III N=8 codelets.
set(_tuned_radixes 2 3 4 5 6 7 8 10 11 12 13 16 17 19 20 25 32 64)
set(_codelet_includes)
foreach(_r IN LISTS _tuned_radixes)
    list(APPEND _codelet_includes
        "${CMAKE_SOURCE_DIR}/src/vectorfft_tune/generated/r${_r}")
endforeach()
list(APPEND _codelet_includes
    "${CMAKE_SOURCE_DIR}/src/vectorfft_tune/generated/dct8")

# Scalar fallback path still pulls headers from the legacy tree.
if(_isa STREQUAL "scalar")
    list(APPEND _codelet_includes
        "${CMAKE_SOURCE_DIR}/src/stride-fft/codelets/scalar")
endif()

# ── Compiled library: vfft (opaque C API) ───────────────────────
add_library(vfft STATIC vfft.c)

target_include_directories(vfft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core
        ${_codelet_includes}
)

target_compile_options(vfft PRIVATE ${_isa_flags})

# Always emit debug info even in Release builds. Performance impact is
# zero (PDB is generated alongside the .obj, not affecting codegen) and
# it makes VTune / profiler output show real codelet function names
# (`radix4_t1_dit_kernel_fwd_avx2`, etc.) instead of `func@0x1402ad480`.
if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
    target_compile_options(vfft PRIVATE /Zi)
    set_target_properties(vfft PROPERTIES
        COMPILE_PDB_NAME vfft
        COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
else()
    target_compile_options(vfft PRIVATE -g)
endif()

target_compile_definitions(vfft PRIVATE
    ${_isa_defs}
    _CRT_SECURE_NO_WARNINGS
    $<$<PLATFORM_ID:Windows>:_USE_MATH_DEFINES>
    # MSVC enables FMA3 implicitly under /arch:AVX2 but doesn't define
    # the GCC-style __FMA__ predicate that registry.h's ISA gate checks.
    $<$<AND:$<C_COMPILER_ID:MSVC>,$<STREQUAL:${_isa},avx2>>:__FMA__>
    $<$<AND:$<C_COMPILER_ID:MSVC>,$<STREQUAL:${_isa},avx512>>:__FMA__>
)

set_target_properties(vfft PROPERTIES
    C_STANDARD 11
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    EXPORT_NAME vfft
    OUTPUT_NAME vfft
)
