# ============================================================================
# src/radix32/CMakeLists.txt
# VectorFFT -- Radix-32 FFT stage (4x8 decomposition)
#
# Layout:
#   radix32/
#     fft_radix32_fv.c         Forward dispatch (AVX-512 -> AVX2 -> scalar)
#     fft_radix32_bv.c         Backward dispatch
#     fft_radix32_uniform.h    ISA detection + backend includes
#     fft_radix32_platform.h   Compiler abstraction (TARGET_AVX512, etc.)
#     fft_radix32_dispatch.h   Dispatch logic
#     avx512/                  AVX-512 kernels (header-only)
#     avx2/                    AVX2 kernels (header-only)
#     scalar/                  Scalar kernels (header-only)
#     tests/                   Unit tests + benchmarks
#
# The backends (avx512/, avx2/, scalar/) are header-only. All code is
# inlined into the two dispatch TUs (fv.c, bv.c) which compile with the
# highest available ISA flags. Runtime CPUID gates actual code paths.
# ============================================================================

# =========================================================================
# LIBRARY: radix-32 dispatch drivers
# =========================================================================
#
# Uses the highest available ISA. The dispatch TUs include ALL backend
# headers via fft_radix32_uniform.h. Per-function TARGET_AVX512 /
# TARGET_AVX2_FMA attributes control codegen; the file-level ISA flag
# just needs to cover the highest tier so the compiler accepts the
# intrinsic headers.
#
# Preference: AVX-512 > AVX2 > skip

if(VFFT_HAS_AVX512)
    vfft_add_isa_library(vfft_radix32 AVX512
        fft_radix32_fv.c
        fft_radix32_bv.c
    )
    set(_R32_ISA AVX512)
elseif(VFFT_HAS_AVX2)
    # Without AVX-512 compiler support, the AVX-512 backend #ifdef's out.
    # Dispatch falls through to AVX2 -> scalar at runtime.
    vfft_add_isa_library(vfft_radix32 AVX2
        fft_radix32_fv.c
        fft_radix32_bv.c
    )
    set(_R32_ISA AVX2)
else()
    message(WARNING "[VectorFFT] radix32: no SIMD ISA available -- skipping")
    return()
endif()

message(STATUS "[VectorFFT] radix32: dispatch at ${_R32_ISA}")


# =========================================================================
# TESTS
# =========================================================================
if(VFFT_BUILD_TESTS)

    # -- Enable C++ (needed for GTest roundtrip suites) --------------------
    # The root project may be C-only; enable CXX here so .cpp sources and
    # GTest can compile without forcing LANGUAGES CXX on the whole tree.
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # -- GTest (for C++ roundtrip suites) ---------------------------------
    if(NOT TARGET GTest::gtest_main)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        # ICX (clang-cl) on Windows: GTest builds with -WX and triggers
        # -Wcharacter-conversion inside gtest-printers.h (char16_t->char32_t).
        # Suppress for GTest TUs only.
        if(VFFT_COMPILER STREQUAL "ICX" AND WIN32)
            set(_r32_saved_cxx "${CMAKE_CXX_FLAGS}")
            string(APPEND CMAKE_CXX_FLAGS " -Wno-character-conversion")
        endif()

        FetchContent_MakeAvailable(googletest)

        if(VFFT_COMPILER STREQUAL "ICX" AND WIN32)
            set(CMAKE_CXX_FLAGS "${_r32_saved_cxx}")
        endif()
    endif()

    # -- Helper: C test ---------------------------------------------------
    #
    # All radix32 tests compile at the driver's ISA level because
    # fft_radix32_uniform.h (included by most tests) pulls in all backends.
    #
    # NEEDS_DISPATCH  link against dispatch objects (radix32_forward() etc.)

    function(_r32_add_c_test NAME SOURCE)
        cmake_parse_arguments(ARG "NEEDS_DISPATCH" "" "" ${ARGN})

        add_executable(${NAME} tests/${SOURCE})
        _vfft_setup_target(${NAME} ${_R32_ISA})

        if(ARG_NEEDS_DISPATCH)
            target_link_libraries(${NAME} PRIVATE
                $<TARGET_OBJECTS:vfft_radix32>)
        endif()

        if(NOT WIN32)
            target_link_libraries(${NAME} PRIVATE m)
        endif()

        add_test(NAME ${NAME} COMMAND ${NAME})
        set_tests_properties(${NAME} PROPERTIES LABELS "radix32")
        message(STATUS "[VectorFFT] Test: ${NAME}")
    endfunction()

    # -- Helper: C++ GTest ------------------------------------------------

    function(_r32_add_gtest NAME SOURCE)
        cmake_parse_arguments(ARG "NEEDS_DISPATCH" "" "" ${ARGN})

        add_executable(${NAME} tests/${SOURCE})
        _vfft_setup_target(${NAME} ${_R32_ISA})

        # _vfft_setup_target sets C_STANDARD; GTest needs CXX_STANDARD too
        set_target_properties(${NAME} PROPERTIES CXX_STANDARD 17)

        target_link_libraries(${NAME} PRIVATE GTest::gtest_main)

        if(ARG_NEEDS_DISPATCH)
            target_link_libraries(${NAME} PRIVATE
                $<TARGET_OBJECTS:vfft_radix32>)
        endif()

        if(NOT WIN32)
            target_link_libraries(${NAME} PRIVATE m)
        endif()

        # Suppress C-header pedantic warnings in C++ (int literal in
        # _mm_prefetch, compound literals in twiddle macros)
        target_compile_options(${NAME} PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fpermissive -Wno-pedantic>
            $<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-Wno-pedantic>
            $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-pedantic>
        )

        include(GoogleTest)
        gtest_discover_tests(${NAME} TEST_PREFIX "${NAME}/")
        message(STATUS "[VectorFFT] GTest: ${NAME}")
    endfunction()

    # -- C unit tests -----------------------------------------------------

    # Scalar codelet (BLOCKED8 + BLOCKED4 + RECURRENCE)
    _r32_add_c_test(vfft_radix32_test_scalar  test_radix32_scalar.c)

    # AVX2 codelet
    _r32_add_c_test(vfft_radix32_test_avx2    test_radix32_avx2.c)

    # AVX-512 core butterflies + full driver (pass1 U=2, pass2 U=2)
    if(VFFT_HAS_AVX512)
        _r32_add_c_test(vfft_radix32_test_avx512_core
            test_radix32_avx512_core.c)
        _r32_add_c_test(vfft_radix32_test_avx512_driver
            test_radix32_avx512_driver.c)
    endif()

    # N=1 codelet (scalar + AVX-512 cross-ISA agreement, Parseval)
    _r32_add_c_test(vfft_radix32_test_n1  test_radix32_n1.c)

    # Dispatch (three-way ISA agreement, constraint downgrades)
    _r32_add_c_test(vfft_radix32_test_dispatch
        test_radix32_dispatch.c  NEEDS_DISPATCH)

    # -- C++ GTest roundtrip suites ---------------------------------------

    # N=1: backward(forward(x)) = 32*x
    _r32_add_gtest(vfft_radix32_test_n1_roundtrip
        test_radix32_n1_roundtrip.cpp)

    # Twiddle: 130+ parameterized (K x mode x ISA)
    _r32_add_gtest(vfft_radix32_test_twiddle_roundtrip
        test_radix32_twiddle_roundtrip.cpp)

endif()  # VFFT_BUILD_TESTS


# =========================================================================
# BENCHMARKS
# =========================================================================
if(VFFT_BUILD_BENCHMARKS)

    # -- FFTW discovery (for comparison benchmark) ------------------------
    #
    # Prefer FFTW3::fftw3 from parent. If unavailable, try pkg-config,
    # then raw find_library as final fallback.
    if(NOT TARGET FFTW3::fftw3)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(_r32_fftw3 QUIET IMPORTED_TARGET fftw3)
            if(_r32_fftw3_FOUND)
                add_library(FFTW3::fftw3 ALIAS PkgConfig::_r32_fftw3)
            endif()
        endif()
    endif()

    if(NOT TARGET FFTW3::fftw3)
        find_library(_R32_FFTW3_LIB fftw3)
        find_path(_R32_FFTW3_INC fftw3.h)
        if(_R32_FFTW3_LIB AND _R32_FFTW3_INC)
            add_library(FFTW3::fftw3 IMPORTED INTERFACE)
            set_target_properties(FFTW3::fftw3 PROPERTIES
                INTERFACE_LINK_LIBRARIES "${_R32_FFTW3_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${_R32_FFTW3_INC}"
            )
        endif()
    endif()

    # -- Helper: benchmark ------------------------------------------------

    function(_r32_add_bench NAME SOURCE)
        cmake_parse_arguments(ARG "NEEDS_DISPATCH;NEEDS_FFTW" "" "" ${ARGN})

        if(ARG_NEEDS_FFTW AND NOT TARGET FFTW3::fftw3)
            message(STATUS "[VectorFFT] Skipping ${NAME} -- FFTW3 not found")
            return()
        endif()

        add_executable(${NAME} tests/${SOURCE})
        _vfft_setup_target(${NAME} ${_R32_ISA})

        if(ARG_NEEDS_DISPATCH)
            target_link_libraries(${NAME} PRIVATE
                $<TARGET_OBJECTS:vfft_radix32>)
        endif()

        if(ARG_NEEDS_FFTW)
            target_link_libraries(${NAME} PRIVATE FFTW3::fftw3)
            target_compile_definitions(${NAME} PRIVATE VFFT_HAS_FFTW3=1)
        endif()

        if(NOT WIN32)
            target_link_libraries(${NAME} PRIVATE m)
        endif()

        message(STATUS "[VectorFFT] Bench: ${NAME}")
    endfunction()

    # -- Internal micro-benchmarks ----------------------------------------
    _r32_add_bench(vfft_radix32_bench_avx2   bench_radix32_avx2.c)
    _r32_add_bench(vfft_radix32_bench_pass2  bench_radix32_pass2.c)

    # -- FFTW comparison --------------------------------------------------
    _r32_add_bench(vfft_radix32_bench_vs_fftw  bench_radix32_vs_fftw.c
        NEEDS_DISPATCH  NEEDS_FFTW)

endif()  # VFFT_BUILD_BENCHMARKS