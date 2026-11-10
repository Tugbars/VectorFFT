# ============================================================================
# benchmark_scripts/CMakeLists.txt -- the vectorfft_vs_* programs: VectorFFT against Intel MKL or FFTW on a fixed
# list of transforms, one build target and one executable each, for anyone who wants the comparison without
# Python or a shell (README.md, "Build targets"). Wired by the root CMakeLists.txt beside the gauntlet
# (VFFT_BUILD_GAUNTLET): the programs drive the gauntlet's bench (bench_1d_vs_mkl) and planning tool
# (recal_1d_probe), so building one builds those too.
#
#   cmake --build <build> --target vectorfft_vs_fftw_selected
#   <build>/benchmark_scripts/vectorfft_vs_fftw_selected [--cells LIST] [--out DIR] [--avx2 | --avx512]
#
#   vectorfft_vs_{mkl,fftw}_selected          18 commonly used transforms, 1D / 2D / 3D, the estimate's plans
#   vectorfft_vs_{mkl,fftw}_selected_planned  the same 18, the planner's plans (into this machine's wisdom folder)
#   vectorfft_vs_{mkl,fftw}_pow2              the 1D powers of two from 64 to 4M, the estimate's plans
#   vectorfft_vs_{mkl,fftw}_pow2_fp32         the 1D powers of two in single precision, the planner's plans
#
# The MKL programs exist only where MKL is found; the FFTW programs always (the bench loads FFTW at run time:
# gauntlet/fftw/libfftw3-3.dll and libfftw3f-3.dll on Windows, the system's libfftw3 elsewhere); the FP32 ones with
# VFFT_ENABLE_FLOAT.
#
# Two kinds of executables:
#   vectorfft_bench_worker   the work (vectorfft_bench.h): it compiles the library into itself with the gauntlet's
#                            flags and store definitions, as vfft_core is built, writes and checks the estimate's
#                            plans or plans through recal_1d_probe, and runs the bench. One build per instruction
#                            set: this build's, and with VFFT_BENCH_BOTH_ISAS the other one, made by a second build
#                            of this same source tree in <build>/<isa>/ (the library is compiled for one
#                            instruction set per build).
#   the programs             launchers built for plain x86-64 (vectorfft_launcher.h): they check the CPU first,
#                            so a CPU without AVX2 gets a message instead of an illegal instruction, pick the
#                            worker (AVX-512 when the CPU runs it, else AVX2; --avx2 / --avx512 force one) and run it.
# ============================================================================

set(VFFT_BENCH_BOTH_ISAS "AUTO" CACHE STRING
    "The vectorfft_vs_* programs' second instruction set: AUTO = built when this CPU runs AVX-512 (the only CPU that runs both), ON = always (packaging), OFF = never")
set_property(CACHE VFFT_BENCH_BOTH_ISAS PROPERTY STRINGS AUTO ON OFF)
option(VFFT_BENCH_SUBBUILD "internal: this build is the second instruction set's benchmark worker" OFF)
mark_as_advanced(VFFT_BENCH_SUBBUILD)

# the directories the bench needs on PATH at run time on Windows: MKL's runtime (mkl_rt.2.dll), the Intel
# OpenMP runtime MKL's default threading layer loads (libiomp5md.dll, the oneAPI compiler's folder; without it
# the bench dies at MKL's first call, 0xC06D007E) and the compiler's runtime. The worker puts them in front of
# PATH for the bench (and tries the usual oneAPI folders as well).
get_filename_component(_vs_cc_bin "${CMAKE_C_COMPILER}" DIRECTORY)
set(_vs_mkl_bin "")
set(_vs_omp_bin "")
if(WIN32 AND VFFT_MKL_FOUND)
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
    if(NOT VFFT_BENCH_SUBBUILD)
        message(STATUS "VectorFFT: the vectorfft_vs_mkl_* programs run the bench with MKL from '${_vs_mkl_bin}', OpenMP from '${_vs_omp_bin}'")
    endif()
endif()

# ── the worker of this build's instruction set ──────────────────────────────
add_executable(vectorfft_bench_worker "${CMAKE_CURRENT_SOURCE_DIR}/vectorfft_bench_worker.c" $<TARGET_OBJECTS:dagcodelets>)
target_include_directories(vectorfft_bench_worker PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${VFFT_INCLUDE_DIRS})
target_compile_options(vectorfft_bench_worker PRIVATE ${VFFT_C_FLAGS})
target_compile_definitions(vectorfft_bench_worker PRIVATE ${VFFT_WISDOM_DEFS} _CRT_SECURE_NO_WARNINGS
    "VS_BENCH_EXE=\"$<TARGET_FILE:bench_1d_vs_mkl>\""
    "VS_RECAL_EXE=\"$<TARGET_FILE:recal_1d_probe>\""
    "VS_TOOLCHAIN_BIN=\"${_vs_cc_bin}\"")
if(_vs_mkl_bin)
    target_compile_definitions(vectorfft_bench_worker PRIVATE "VS_MKL_BIN=\"${_vs_mkl_bin}\"")
endif()
if(_vs_omp_bin)
    target_compile_definitions(vectorfft_bench_worker PRIVATE "VS_OMP_BIN=\"${_vs_omp_bin}\"")
endif()
if(VFFT_MKL_FOUND)
    target_compile_definitions(vectorfft_bench_worker PRIVATE VS_BENCH_HAS_MKL)   # the bench has its MKL arm
endif()
if(VFFT_ENABLE_FLOAT)
    target_compile_definitions(vectorfft_bench_worker PRIVATE VS_HAS_FLOAT)       # the bench and the planning tool have their FP32 arms
endif()
if(NOT VFFT_MSVC_STYLE)
    target_link_libraries(vectorfft_bench_worker PRIVATE Threads::Threads m ${CMAKE_DL_LIBS})
    if(WIN32)
        target_link_options(vectorfft_bench_worker PRIVATE -static)   # starts without the compiler's DLLs on PATH
    endif()
endif()
add_dependencies(vectorfft_bench_worker bench_1d_vs_mkl recal_1d_probe)

if(VFFT_BENCH_SUBBUILD)
    return()   # the second instruction set's build makes its worker (and the bench and planning tool) alone
endif()

# ── the other instruction set's worker: this source tree built again, for that instruction set ─────
string(TOUPPER "${VFFT_ISA}" _vs_isa)
if(VFFT_ISA STREQUAL "avx512")
    set(_vs_other avx2)
else()
    set(_vs_other avx512)
endif()
string(TOUPPER "${_vs_other}" _vs_OTHER)
set(_vs_both FALSE)
if(VFFT_BENCH_BOTH_ISAS STREQUAL "ON")
    set(_vs_both TRUE)
elseif(VFFT_BENCH_BOTH_ISAS STREQUAL "AUTO" AND VFFT_HOST_RUNS_AVX512 AND NOT CMAKE_CROSSCOMPILING)
    set(_vs_both TRUE)
endif()
set(_vs_workers "VL_WORKER_${_vs_isa}=\"$<TARGET_FILE:vectorfft_bench_worker>\"")
set(_vs_worker_deps vectorfft_bench_worker)
if(_vs_both)
    include(ExternalProject)
    # short names: a codelet's object path inside a build is up to ~130 characters, and make on Windows stops at
    # 260 for the whole path (build folder + this folder + the object)
    set(_vs_dir "${CMAKE_BINARY_DIR}/${_vs_other}")
    set(_vs_args -DVFFT_ISA=${_vs_other} -DVFFT_BENCH_SUBBUILD=ON -DVFFT_BUILD_EXAMPLES=OFF -DVFFT_INSTALL=OFF
                 -DVFFT_NATIVE=${VFFT_NATIVE} -DVFFT_ENABLE_FLOAT=${VFFT_ENABLE_FLOAT} -DVFFT_WISDOM=${VFFT_WISDOM}
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
    foreach(_v CMAKE_BUILD_TYPE CMAKE_MAKE_PROGRAM Python3_EXECUTABLE MKL_INCLUDE_DIR MKL_RT MKL_ILP64 MKL_SEQ MKL_CORE)
        if(${_v})
            list(APPEND _vs_args "-D${_v}=${${_v}}")
        endif()
    endforeach()
    # the search paths, so the second build finds (or does not find) what this one did -- MKL above all: a bench
    # linked to MKL in one build and not the other would make the two versions differ. Lists travel with "|"
    # (LIST_SEPARATOR below), never ";", which ExternalProject splits on.
    foreach(_v CMAKE_PREFIX_PATH CMAKE_IGNORE_PATH CMAKE_IGNORE_PREFIX_PATH)
        if(${_v})
            string(REPLACE ";" "|" _vs_list "${${_v}}")
            list(APPEND _vs_args "-D${_v}=${_vs_list}")
        endif()
    endforeach()
    ExternalProject_Add(bench_${_vs_other}
        LIST_SEPARATOR |
        PREFIX "${CMAKE_BINARY_DIR}/ep"
        SOURCE_DIR "${PROJECT_SOURCE_DIR}"
        BINARY_DIR "${_vs_dir}"
        CMAKE_ARGS ${_vs_args}
        BUILD_COMMAND ${CMAKE_COMMAND} --build "${_vs_dir}" --target vectorfft_bench_worker
        INSTALL_COMMAND ""
        BUILD_ALWAYS TRUE          # the second build follows every change of the sources (its own make decides what to redo)
        USES_TERMINAL_BUILD TRUE)
    list(APPEND _vs_workers
        "VL_WORKER_${_vs_OTHER}=\"${_vs_dir}/benchmark_scripts/vectorfft_bench_worker${CMAKE_EXECUTABLE_SUFFIX}\"")
    list(APPEND _vs_worker_deps bench_${_vs_other})
    message(STATUS "VectorFFT: the vectorfft_vs_* programs carry ${_vs_isa} and ${_vs_OTHER} (the second built in ${_vs_dir})")
else()
    message(STATUS "VectorFFT: the vectorfft_vs_* programs carry ${_vs_isa} only (VFFT_BENCH_BOTH_ISAS=${VFFT_BENCH_BOTH_ISAS})")
endif()

# ── the programs ─────────────────────────────────────────────────────────────
set(_vs_programs vectorfft_vs_fftw_selected vectorfft_vs_fftw_selected_planned vectorfft_vs_fftw_pow2)
if(VFFT_ENABLE_FLOAT)
    list(APPEND _vs_programs vectorfft_vs_fftw_pow2_fp32)
endif()
if(VFFT_MKL_FOUND)
    list(APPEND _vs_programs vectorfft_vs_mkl_selected vectorfft_vs_mkl_selected_planned vectorfft_vs_mkl_pow2)
    if(VFFT_ENABLE_FLOAT)
        list(APPEND _vs_programs vectorfft_vs_mkl_pow2_fp32)
    endif()
else()
    message(STATUS "VectorFFT: the vectorfft_vs_mkl_* programs are not built: Intel MKL was not found (set MKLROOT)")
endif()
if(NOT VFFT_ENABLE_FLOAT)
    message(STATUS "VectorFFT: the single-precision programs (vectorfft_vs_{mkl,fftw}_pow2_fp32) need -DVFFT_ENABLE_FLOAT=ON")
endif()
foreach(_p ${_vs_programs})
    add_executable(${_p} "${CMAKE_CURRENT_SOURCE_DIR}/${_p}.c")
    target_include_directories(${_p} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(${_p} PRIVATE ${_vs_workers} _CRT_SECURE_NO_WARNINGS)
    if(NOT VFFT_MSVC_STYLE)
        # any x86-64 CPU: the launcher checks the CPU before anything that needs AVX runs
        target_compile_options(${_p} PRIVATE -O2 -march=x86-64 -mtune=generic)
        if(WIN32)
            target_link_options(${_p} PRIVATE -static)
        endif()
    endif()
    add_dependencies(${_p} ${_vs_worker_deps})
    add_custom_command(TARGET ${_p} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "VectorFFT: built $<TARGET_FILE_NAME:${_p}>; run it: $<TARGET_FILE:${_p}>"
        VERBATIM)
endforeach()
