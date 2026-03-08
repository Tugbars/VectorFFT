# ============================================================================
# test/CMakeLists.txt
#
# Full-radix benchmark: VectorFFT vs FFTW
# Wires all radixes into the planner: R=2-8,16,32 (fused DIT+DIF),
# R=11,13,17,19,23 (genfft notw), R=64,128 (N1 innermost).
# ============================================================================

# ── Find FFTW ───────────────────────────────────────────────────────────────
find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND OR TARGET FFTW3::fftw3)
    set(FFTW3_FOUND TRUE)
    if(TARGET FFTW3::fftw3)
        set(FFTW3_LIBRARIES FFTW3::fftw3)
    endif()
    message(STATUS "test: FFTW3 found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
        if(FFTW3_FOUND)
            message(STATUS "test: FFTW3 found via pkg-config")
        endif()
    endif()
endif()

if(NOT FFTW3_FOUND)
    find_library(FFTW3_LIB NAMES fftw3 libfftw3-3)
    find_path(FFTW3_INC NAMES fftw3.h)
    if(FFTW3_LIB AND FFTW3_INC)
        set(FFTW3_FOUND TRUE)
        set(FFTW3_LIBRARIES ${FFTW3_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INC})
        message(STATUS "test: FFTW3 found via manual search")
    endif()
endif()

if(NOT FFTW3_FOUND)
    message(WARNING "test: FFTW3 not found — bench_full_fft disabled")
    return()
endif()

# ── Highest available ISA flags ──────────────────────────────────────────────
if(VFFT_HAS_AVX512)
    set(BENCH_ISA_FLAGS ${VFFT_FLAGS_AVX512})
elseif(VFFT_HAS_AVX2)
    set(BENCH_ISA_FLAGS ${VFFT_FLAGS_AVX2})
elseif(VFFT_HAS_SSE2)
    set(BENCH_ISA_FLAGS ${VFFT_FLAGS_SSE2})
else()
    set(BENCH_ISA_FLAGS "")
endif()

# ── Include paths for all codelet headers ────────────────────────────────────
set(BENCH_CODELET_INCLUDES
    # DIT/DIF dispatch modules
    ${PROJECT_SOURCE_DIR}/src/radix2
    # radix3: flat layout — dispatch.h includes "avx2/...", "scalar/...", "avx512/..."
    # so we add both the root dir AND the ISA subdirs (shim headers forward to root)
    ${PROJECT_SOURCE_DIR}/src/radix3
    ${PROJECT_SOURCE_DIR}/src/radix3/avx512
    ${PROJECT_SOURCE_DIR}/src/radix3/avx2
    ${PROJECT_SOURCE_DIR}/src/radix3/scalar
    ${PROJECT_SOURCE_DIR}/src/radix4
    ${PROJECT_SOURCE_DIR}/src/radix5
    ${PROJECT_SOURCE_DIR}/src/radix7
    ${PROJECT_SOURCE_DIR}/src/radix8
    ${PROJECT_SOURCE_DIR}/src/radix10
    ${PROJECT_SOURCE_DIR}/src/radix16
    ${PROJECT_SOURCE_DIR}/src/radix25
    ${PROJECT_SOURCE_DIR}/src/radix32
    # Genfft prime modules
    ${PROJECT_SOURCE_DIR}/src/radix11
    ${PROJECT_SOURCE_DIR}/src/radix13
    ${PROJECT_SOURCE_DIR}/src/radix17
    ${PROJECT_SOURCE_DIR}/src/radix19
    ${PROJECT_SOURCE_DIR}/src/radix23
    # N1 large codelets — driver headers include ISA files flat (no subdir prefix),
    # so the ISA subdirs must be on the search path
    ${PROJECT_SOURCE_DIR}/src/radix64
    ${PROJECT_SOURCE_DIR}/src/radix64/scalar
    ${PROJECT_SOURCE_DIR}/src/radix64/avx2
    ${PROJECT_SOURCE_DIR}/src/radix64/avx512
    ${PROJECT_SOURCE_DIR}/src/radix128
    ${PROJECT_SOURCE_DIR}/src/radix128/scalar
    ${PROJECT_SOURCE_DIR}/src/radix128/avx2
    ${PROJECT_SOURCE_DIR}/src/radix128/avx512
    # Planner + registry
    ${PROJECT_SOURCE_DIR}/src/3-execute
    # Common utilities
    ${PROJECT_SOURCE_DIR}/src/common
)

# ── bench_full_fft ───────────────────────────────────────────────────────────
add_executable(bench_full_fft bench_full_fft.c)

target_compile_options(bench_full_fft PRIVATE
    ${VFFT_BASE_FLAGS}
    ${BENCH_ISA_FLAGS}
)
set_target_properties(bench_full_fft PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

if(VFFT_BASE_DEFINITIONS)
    target_compile_definitions(bench_full_fft PRIVATE ${VFFT_BASE_DEFINITIONS})
endif()

target_include_directories(bench_full_fft PRIVATE
    ${BENCH_CODELET_INCLUDES}
)

if(TARGET FFTW3::fftw3)
    target_link_libraries(bench_full_fft PRIVATE FFTW3::fftw3)
else()
    target_include_directories(bench_full_fft PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(bench_full_fft PRIVATE ${FFTW3_LIBRARIES})
endif()

if(NOT WIN32)
    target_link_libraries(bench_full_fft PRIVATE m)
endif()

message(STATUS "test: bench_full_fft [ISA: ${BENCH_ISA_FLAGS}]")

# ── diag_plan (no FFTW dep) ──────────────────────────────────────────────────
add_executable(diag_plan diag_plan.c)
target_compile_options(diag_plan PRIVATE ${VFFT_BASE_FLAGS} ${BENCH_ISA_FLAGS})
set_target_properties(diag_plan PROPERTIES C_STANDARD ${VFFT_C_STANDARD})
if(VFFT_BASE_DEFINITIONS)
    target_compile_definitions(diag_plan PRIVATE ${VFFT_BASE_DEFINITIONS})
endif()
target_include_directories(diag_plan PRIVATE ${BENCH_CODELET_INCLUDES})
