# VectorFFT — radix10 codelet module (header-only)
add_library(vfft_radix10 INTERFACE)
target_include_directories(vfft_radix10 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar
)
message(STATUS "[VectorFFT] Module: radix10 (header-only)")
if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    endif()
endif()
