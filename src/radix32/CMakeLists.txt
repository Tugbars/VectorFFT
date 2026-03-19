add_library(vfft_radix32 INTERFACE)
target_include_directories(vfft_radix32 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar)
message(STATUS "[VectorFFT] Module: radix32 (flat CT + ladder + pack-walk + IL)")
if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        add_subdirectory(tests)
    endif()
endif()
