# ============================================================================
# gauntlet/CMakeLists.txt — the gauntlet's binaries, wired by the root
# CMakeLists.txt (add_subdirectory(gauntlet)). Mirrors build.py's --vfft path:
# each tool is its own driver TU + src/core/vfft.c, compiled with the root's
# VFFT_C_FLAGS and linked against the root's dagcodelets library.
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
# every tool shares one compiled copy.
add_library(vfft_core STATIC "${DAG_CORE}/vfft.c")
target_include_directories(vfft_core PUBLIC ${VFFT_INCLUDE_DIRS})
target_compile_options(vfft_core PRIVATE ${VFFT_C_FLAGS})
target_compile_definitions(vfft_core PUBLIC ${VFFT_WISDOM_DEFS} _CRT_SECURE_NO_WARNINGS)
target_link_libraries(vfft_core PUBLIC dagcodelets)
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

add_custom_target(vfft-gauntlet DEPENDS ${VFFT_GAUNTLET_TOOLS})
