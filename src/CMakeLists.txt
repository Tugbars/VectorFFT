cmake_minimum_required(VERSION 3.10)

project(highspeedFFT C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SOURCES
    ${SRC_DIR}/highspeedFFT.c
    ${SRC_DIR}/convolve.c
    ${SRC_DIR}/real.c
)

set(HEADERS
    ${SRC_DIR}/highspeedFFT.h
    ${SRC_DIR}/real.h
)

add_library(highspeedFFT STATIC ${SOURCES} ${HEADERS})

target_include_directories(highspeedFFT PUBLIC ${SRC_DIR})

# Find math library for non-MSVC compilers
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(highspeedFFT PRIVATE ${MATH_LIBRARY})
endif()

# Check for SSE2 support (required for radix-4)
include(CheckCCompilerFlag)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    check_c_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
    if(COMPILER_SUPPORTS_SSE2)
        target_compile_options(highspeedFFT PRIVATE -msse2)
        message(STATUS "SSE2 support enabled with -msse2")
    else()
        message(FATAL_ERROR "Compiler does not support -msse2, required for SSE2 intrinsics in radix-4")
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    message(STATUS "SSE2 support enabled by default in MSVC")
else()
    message(WARNING "Unknown compiler; SSE2 support not guaranteed")
endif()

# Check for AVX2 support (required for radices 2, 7, 8)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(highspeedFFT PRIVATE -O3 -mavx2)
    else()
        message(FATAL_ERROR "Compiler does not support -mavx2, required for AVX2 intrinsics")
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    # MSVC uses /arch:AVX2; no flag check needed, as it's supported in modern versions
    target_compile_options(highspeedFFT PRIVATE /O2 /arch:AVX2)
    if(MSVC_VERSION LESS 1800) # VS 2013 Update 4
        message(WARNING "MSVC version may lack full AVX2 support; recommend VS 2015 or later")
    endif()
else()
    message(WARNING "Unknown compiler; AVX2 support not guaranteed")
endif()

# Check for FMA support
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    if(COMPILER_SUPPORTS_FMA)
        target_compile_options(highspeedFFT PRIVATE -mfma)
        target_compile_definitions(highspeedFFT PRIVATE USE_FMA)
        message(STATUS "FMA support enabled with -mfma and USE_FMA defined")
    else()
        message(STATUS "FMA support not detected; building with FMA emulation")
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    target_compile_options(highspeedFFT PRIVATE /fp:fast)
    target_compile_definitions(highspeedFFT PRIVATE USE_FMA)
    message(STATUS "FMA support enabled with /fp:fast and USE_FMA defined")
endif()

# Note mixed SIMD usage
message(STATUS "Building with mixed SIMD: SSE2 for radix-4, AVX2 for radices 2, 7, 8")

# TODO: Add checks for aligned allocation (e.g., _aligned_malloc, posix_memalign) for future pre-allocation