# ============================================================================
# benchmark_scripts/CMakeLists.txt -- the vectorfft_vs_mkl_* programs: VectorFFT against Intel MKL on a fixed
# list of transforms, one build target and one executable each, for anyone who wants the comparison without
# Python or a shell (README.md, "Build targets"). Wired by the root CMakeLists.txt beside the gauntlet
# (VFFT_BUILD_GAUNTLET): each program drives the gauntlet's bench (bench_1d_vs_mkl), so building it builds the
# bench too.
#
#   cmake --build <build> --target vectorfft_vs_mkl_selected
#   <build>/benchmark_scripts/vectorfft_vs_mkl_selected
#
# Each program compiles the library into itself (vectorfft_vs_mkl.h includes src/core/vfft.c with the
# gauntlet's flags and store definitions, as vfft_core is built): it writes the estimate's plans into a store
# and checks them before the bench times them. It times nothing itself.
# ============================================================================

if(NOT VFFT_MKL_FOUND)
    message(STATUS "VectorFFT: the vectorfft_vs_mkl_* programs are not built: Intel MKL was not found (set MKLROOT)")
    return()
endif()

# the directories the bench needs on PATH at run time on Windows: MKL's runtime (mkl_rt.2.dll), the Intel
# OpenMP runtime MKL's default threading layer loads (libiomp5md.dll, the oneAPI compiler's folder; without it
# the bench dies at MKL's first call, 0xC06D007E) and the compiler's runtime. The programs put them in front of
# PATH for the bench (and try the usual oneAPI folders as well).
get_filename_component(_vs_cc_bin "${CMAKE_C_COMPILER}" DIRECTORY)
set(_vs_mkl_bin "")
set(_vs_omp_bin "")
if(WIN32)
    set(_vs_hints "$ENV{MKLROOT}/bin")
    foreach(_lib ${VFFT_MKL_LIBS})
        get_filename_component(_d "${_lib}" DIRECTORY)
        list(APPEND _vs_hints "${_d}/../bin" "${_d}/../redist/intel64")
    endforeach()
    list(APPEND _vs_hints "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/bin" "C:/Program Files/Intel/oneAPI/mkl/latest/bin")
    find_file(VFFT_VS_MKL_DLL NAMES mkl_rt.2.dll mkl_rt.dll HINTS ${_vs_hints})
    if(VFFT_VS_MKL_DLL)
        get_filename_component(_vs_mkl_bin "${VFFT_VS_MKL_DLL}" DIRECTORY)
    endif()
    find_file(VFFT_VS_OMP_DLL NAMES libiomp5md.dll
              HINTS "${_vs_mkl_bin}/../../compiler/latest/bin" "$ENV{MKLROOT}/../../compiler/latest/bin"
                    "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin" "C:/Program Files/Intel/oneAPI/compiler/latest/bin")
    if(VFFT_VS_OMP_DLL)
        get_filename_component(_vs_omp_bin "${VFFT_VS_OMP_DLL}" DIRECTORY)
    endif()
    message(STATUS "VectorFFT: vectorfft_vs_mkl_* run the bench with MKL from '${_vs_mkl_bin}', OpenMP from '${_vs_omp_bin}'")
endif()

set(VFFT_VS_MKL_PROGRAMS vectorfft_vs_mkl_selected)
foreach(_p ${VFFT_VS_MKL_PROGRAMS})
    add_executable(${_p} "${CMAKE_CURRENT_SOURCE_DIR}/${_p}.c" $<TARGET_OBJECTS:dagcodelets>)
    target_include_directories(${_p} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${VFFT_INCLUDE_DIRS})
    target_compile_options(${_p} PRIVATE ${VFFT_C_FLAGS})
    target_compile_definitions(${_p} PRIVATE ${VFFT_WISDOM_DEFS} _CRT_SECURE_NO_WARNINGS
        "VS_BENCH_EXE=\"$<TARGET_FILE:bench_1d_vs_mkl>\""
        "VS_TOOLCHAIN_BIN=\"${_vs_cc_bin}\"")
    if(_vs_mkl_bin)
        target_compile_definitions(${_p} PRIVATE "VS_MKL_BIN=\"${_vs_mkl_bin}\"")
    endif()
    if(_vs_omp_bin)
        target_compile_definitions(${_p} PRIVATE "VS_OMP_BIN=\"${_vs_omp_bin}\"")
    endif()
    if(NOT VFFT_MSVC_STYLE)
        target_link_libraries(${_p} PRIVATE Threads::Threads m ${CMAKE_DL_LIBS})
        if(WIN32)
            target_link_options(${_p} PRIVATE -static)   # starts without the compiler's DLLs on PATH
        endif()
    endif()
    add_dependencies(${_p} bench_1d_vs_mkl)
    add_custom_command(TARGET ${_p} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "VectorFFT: built $<TARGET_FILE_NAME:${_p}> -- codelet ISA ${VFFT_ISA}; run it: $<TARGET_FILE:${_p}>"
        VERBATIM)
endforeach()
