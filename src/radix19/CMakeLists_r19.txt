cmake_minimum_required(VERSION 3.16)
project(VectorFFT_Radix19 C)
set(CMAKE_C_STANDARD 11)

include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx512f" HAS_AVX512F)
check_c_compiler_flag("-mavx2" HAS_AVX2)
check_c_compiler_flag("-mfma" HAS_FMA)

set(RADIX19_C_FLAGS "")
if(HAS_AVX512F) list(APPEND RADIX19_C_FLAGS -mavx512f -mavx512dq) endif()
if(HAS_AVX2) list(APPEND RADIX19_C_FLAGS -mavx2) endif()
if(HAS_FMA) list(APPEND RADIX19_C_FLAGS -mfma) endif()

add_library(radix19 INTERFACE)
target_include_directories(radix19 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(radix19 INTERFACE ${RADIX19_C_FLAGS})

enable_testing()
add_executable(test_radix19 bench_final_r19.c)
target_link_libraries(test_radix19 radix19 m)
target_compile_options(test_radix19 PRIVATE -O2 -Wno-unused-function)
find_library(FFTW3_LIB fftw3)
if(FFTW3_LIB)
    target_link_libraries(test_radix19 ${FFTW3_LIB})
endif()
add_test(NAME radix19_correctness COMMAND test_radix19)

install(FILES fft_radix19_genfft.h fft_radix19_dispatch.h DESTINATION include/vectorfft)
