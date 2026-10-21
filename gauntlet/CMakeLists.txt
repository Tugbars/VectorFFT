# ============================================================================
# gauntlet/CMakeLists.txt — the gauntlet's binaries, wired by the root
# CMakeLists.txt (add_subdirectory(gauntlet), VFFT_BUILD_GAUNTLET). Mirrors
# build.py's --vfft path: each tool is its own driver TU + src/core/vfft.c,
# compiled with the root's VFFT_C_FLAGS and archived with the root's
# dagcodelets objects.
#
#   cmake --build <build> --target vfft-gauntlet
#   python gauntlet/gauntlet.py run --cells 2..64 --bin-dir <build>/gauntlet
#
# The driver (gauntlet.py) runs recal_1d_probe (calibrate), bench_1d_vs_mkl
# (bench) and k1_fwd_ref_probe (verify). bench_1d_vs_fftw is the FFTW
# comparator: it binds FFTW at run time (dlopen / LoadLibrary, ref_fftw.h), so
# it needs no FFTW at build time — point VFFT_FFTW_DLL at the libfftw3 to race.
#
# MKL is optional (the root's discovery: VFFT_MKL_FOUND). Without it the bench
# reports absolute numbers and correctness only.
#
# This file was missing from the repository until 2026-09-26: the root
# .gitignore's `*.txt` rule swallowed it (now excepted there).
# ============================================================================

# vfft.c is the ONE runtime .c of the header-only core. Its only build-time
# switch is VFFT_USE_JIT (build.py --jit), which this build does not set, so
# every tool shares one compiled copy. vfft_core is the gauntlet's own
# library, never installed: the clone's store (VFFT_WISDOM_DEFS) and the
# internal include walk PUBLIC, which the bench tools need; the user's
# library is VectorFFT::vfft.
add_library(vfft_core STATIC "${DAG_CORE}/vfft.c" $<TARGET_OBJECTS:dagcodelets>)
target_include_directories(vfft_core PUBLIC ${VFFT_INCLUDE_DIRS})
target_compile_options(vfft_core PRIVATE ${VFFT_C_FLAGS})
target_compile_definitions(vfft_core PUBLIC ${VFFT_WISDOM_DEFS} _CRT_SECURE_NO_WARNINGS)
if(WIN32)   # the repository's own FFTW, the comparator's default (ref_fftw.h, fftw/README.md)
    target_compile_definitions(vfft_core PUBLIC
        VFFT_FFTW_BUNDLED_DLL="${CMAKE_CURRENT_SOURCE_DIR}/fftw/libfftw3-3.dll")
endif()
if(NOT VFFT_MSVC_STYLE)
    find_package(Threads REQUIRED)
    target_link_libraries(vfft_core PUBLIC Threads::Threads m ${CMAKE_DL_LIBS})
endif()

set(VFFT_GAUNTLET_TOOLS bench_1d_vs_mkl recal_1d_probe k1_fwd_ref_probe bench_1d_vs_fftw)
foreach(_tool ${VFFT_GAUNTLET_TOOLS})
    add_executable(${_tool} "${CMAKE_CURRENT_SOURCE_DIR}/${_tool}.c")
    target_include_directories(${_tool} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${_tool} PRIVATE ${VFFT_C_FLAGS})
    target_link_libraries(${_tool} PRIVATE vfft_core)
endforeach()

# the MKL comparator arms (the bench and the forward reference check)
if(VFFT_MKL_FOUND)
    foreach(_tool bench_1d_vs_mkl k1_fwd_ref_probe)
        target_include_directories(${_tool} PRIVATE "${MKL_INCLUDE_DIR}")
        target_compile_definitions(${_tool} PRIVATE ${VFFT_MKL_DEFS})
        target_link_libraries(${_tool} PRIVATE ${VFFT_MKL_LIBS})
    endforeach()
endif()

# each tool's last word when it is built: the codelet ISA, the CPU, its comparator
foreach(_tool ${VFFT_GAUNTLET_TOOLS})
    if(_tool STREQUAL "bench_1d_vs_mkl" OR _tool STREQUAL "k1_fwd_ref_probe")
        if(VFFT_MKL_FOUND)
            set(_cmp "MKL arm built in; FFTW loaded at run time")
        else()
            set(_cmp "no MKL arm (MKL not found); FFTW loaded at run time")
        endif()
    elseif(_tool STREQUAL "bench_1d_vs_fftw")
        set(_cmp "FFTW loaded at run time")
    else()
        set(_cmp "no comparator")
    endif()
    add_custom_command(TARGET ${_tool} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "VectorFFT gauntlet: built $<TARGET_FILE_NAME:${_tool}> -- codelet ISA ${VFFT_ISA}, for ${VFFT_TARGET_TXT}; ${_cmp}"
        VERBATIM)
endforeach()

add_custom_target(vfft-gauntlet DEPENDS ${VFFT_GAUNTLET_TOOLS})
