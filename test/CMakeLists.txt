# ============================================================================
# test/CMakeLists.txt
#
# Full-radix benchmark: VectorFFT vs FFTW
# Wires all radixes into the planner: R=2-8,10,16,20,25,32 (fused DIT+DIF),
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
    # radix16: dispatch.h includes avx2/..., avx512/..., scalar/...
    ${PROJECT_SOURCE_DIR}/src/radix16
    ${PROJECT_SOURCE_DIR}/src/radix16/avx512
    ${PROJECT_SOURCE_DIR}/src/radix16/avx2
    ${PROJECT_SOURCE_DIR}/src/radix16/scalar
    # radix20: same folder structure
    ${PROJECT_SOURCE_DIR}/src/radix20
    ${PROJECT_SOURCE_DIR}/src/radix20/avx512
    ${PROJECT_SOURCE_DIR}/src/radix20/avx2
    ${PROJECT_SOURCE_DIR}/src/radix20/scalar
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

# ── Helper: create a bench target with standard flags + FFTW ─────────────────
function(add_bench_target TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_compile_options(${TARGET_NAME} PRIVATE
        ${VFFT_BASE_FLAGS}
        ${BENCH_ISA_FLAGS}
    )
    set_target_properties(${TARGET_NAME} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})
    if(VFFT_BASE_DEFINITIONS)
        target_compile_definitions(${TARGET_NAME} PRIVATE ${VFFT_BASE_DEFINITIONS})
    endif()
    target_include_directories(${TARGET_NAME} PRIVATE ${BENCH_CODELET_INCLUDES})
    if(TARGET FFTW3::fftw3)
        target_link_libraries(${TARGET_NAME} PRIVATE FFTW3::fftw3)
    else()
        target_include_directories(${TARGET_NAME} PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(${TARGET_NAME} PRIVATE ${FFTW3_LIBRARIES})
    endif()
    if(NOT WIN32 OR VFFT_MINGW)
        target_link_libraries(${TARGET_NAME} PRIVATE m)
    endif()
endfunction()

# ── bench_full_fft ───────────────────────────────────────────────────────────
add_bench_target(bench_full_fft bench_full_fft.c)
message(STATUS "test: bench_full_fft [ISA: ${BENCH_ISA_FLAGS}]")

# ── bench_factorize — brute-force factorization search ───────────────────────
add_bench_target(bench_factorize bench_factorize.c)
message(STATUS "test: bench_factorize [ISA: ${BENCH_ISA_FLAGS}]")

# ── bench_r20_codelets — R=20 codelet correctness + DAG/CT crossover ─────────
add_executable(bench_r20_codelets bench_r20_codelets.c)
target_compile_options(bench_r20_codelets PRIVATE ${VFFT_BASE_FLAGS} ${BENCH_ISA_FLAGS})
set_target_properties(bench_r20_codelets PROPERTIES C_STANDARD ${VFFT_C_STANDARD})
if(VFFT_BASE_DEFINITIONS)
    target_compile_definitions(bench_r20_codelets PRIVATE ${VFFT_BASE_DEFINITIONS})
endif()
target_include_directories(bench_r20_codelets PRIVATE ${BENCH_CODELET_INCLUDES})
if(NOT WIN32 OR VFFT_MINGW)
    target_link_libraries(bench_r20_codelets PRIVATE m)
endif()
message(STATUS "test: bench_r20_codelets [ISA: ${BENCH_ISA_FLAGS}]")

# ── diag_plan (no FFTW dep) ──────────────────────────────────────────────────
#add_executable(diag_plan diag_plan.c)
#target_compile_options(diag_plan PRIVATE ${VFFT_BASE_FLAGS} ${BENCH_ISA_FLAGS})
#set_target_properties(diag_plan PROPERTIES C_STANDARD ${VFFT_C_STANDARD})
#if(VFFT_BASE_DEFINITIONS)
#    target_compile_definitions(diag_plan PRIVATE ${VFFT_BASE_DEFINITIONS})
#endif()
#target_include_directories(diag_plan PRIVATE ${BENCH_CODELET_INCLUDES})

# ── Walk threshold benchmark ──
add_executable(bench_walk bench_walk.c)
target_include_directories(bench_walk PRIVATE
    ${CMAKE_SOURCE_DIR}/src/3-execute
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/radix2
    ${CMAKE_SOURCE_DIR}/src/radix3
    ${CMAKE_SOURCE_DIR}/src/radix4
    ${CMAKE_SOURCE_DIR}/src/radix5
    ${CMAKE_SOURCE_DIR}/src/radix7
    ${CMAKE_SOURCE_DIR}/src/radix8
    ${CMAKE_SOURCE_DIR}/src/radix10
    ${CMAKE_SOURCE_DIR}/src/radix11
    ${CMAKE_SOURCE_DIR}/src/radix13
    ${CMAKE_SOURCE_DIR}/src/radix16
    ${CMAKE_SOURCE_DIR}/src/radix16/avx512
    ${CMAKE_SOURCE_DIR}/src/radix16/avx2
    ${CMAKE_SOURCE_DIR}/src/radix16/scalar
    ${CMAKE_SOURCE_DIR}/src/radix17
    ${CMAKE_SOURCE_DIR}/src/radix19
    ${CMAKE_SOURCE_DIR}/src/radix20
    ${CMAKE_SOURCE_DIR}/src/radix20/avx512
    ${CMAKE_SOURCE_DIR}/src/radix20/avx2
    ${CMAKE_SOURCE_DIR}/src/radix20/scalar
    ${CMAKE_SOURCE_DIR}/src/radix23
    ${CMAKE_SOURCE_DIR}/src/radix25
    ${CMAKE_SOURCE_DIR}/src/radix32
    ${CMAKE_SOURCE_DIR}/src/radix64
    ${CMAKE_SOURCE_DIR}/src/radix128
)
target_link_libraries(bench_walk PRIVATE ${FFTW3_LIBRARIES})
if(NOT WIN32 OR VFFT_MINGW)
    target_link_libraries(bench_walk PRIVATE m)
endif()

# ── IL crossover benchmark ──
add_executable(bench_il bench_il.c)
target_include_directories(bench_il PRIVATE
    ${CMAKE_SOURCE_DIR}/src/3-execute
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/radix2
    ${CMAKE_SOURCE_DIR}/src/radix3
    ${CMAKE_SOURCE_DIR}/src/radix4
    ${CMAKE_SOURCE_DIR}/src/radix5
    ${CMAKE_SOURCE_DIR}/src/radix7
    ${CMAKE_SOURCE_DIR}/src/radix8
    ${CMAKE_SOURCE_DIR}/src/radix10
    ${CMAKE_SOURCE_DIR}/src/radix11
    ${CMAKE_SOURCE_DIR}/src/radix13
    ${CMAKE_SOURCE_DIR}/src/radix16
    ${CMAKE_SOURCE_DIR}/src/radix16/avx512
    ${CMAKE_SOURCE_DIR}/src/radix16/avx2
    ${CMAKE_SOURCE_DIR}/src/radix16/scalar
    ${CMAKE_SOURCE_DIR}/src/radix17
    ${CMAKE_SOURCE_DIR}/src/radix19
    ${CMAKE_SOURCE_DIR}/src/radix20
    ${CMAKE_SOURCE_DIR}/src/radix20/avx512
    ${CMAKE_SOURCE_DIR}/src/radix20/avx2
    ${CMAKE_SOURCE_DIR}/src/radix20/scalar
    ${CMAKE_SOURCE_DIR}/src/radix23
    ${CMAKE_SOURCE_DIR}/src/radix25
    ${CMAKE_SOURCE_DIR}/src/radix32
    ${CMAKE_SOURCE_DIR}/src/radix64
    ${CMAKE_SOURCE_DIR}/src/radix128
)
target_link_libraries(bench_il PRIVATE ${FFTW3_LIBRARIES})
if(NOT WIN32 OR VFFT_MINGW)
    target_link_libraries(bench_il PRIVATE m)
endif()

# Frame-pointer + ISA flags: /Oy- is MSVC/clang-cl, -fno-omit-frame-pointer is GCC/Clang
if(VFFT_WINDOWS_CLANGCL)
    set(_BENCH_FRAMEPTR /Oy-)
else()
    set(_BENCH_FRAMEPTR -fno-omit-frame-pointer)
endif()
target_compile_options(bench_walk PRIVATE ${_BENCH_FRAMEPTR} -mavx2 -mfma -mno-avx512f)
target_compile_options(bench_il   PRIVATE ${_BENCH_FRAMEPTR} -mavx2 -mfma -mno-avx512f)