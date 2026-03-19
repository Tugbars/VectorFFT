add_library(vfft_radix8 INTERFACE)
target_include_directories(vfft_radix8 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar)
message(STATUS "[VectorFFT] Module: radix8 (log3 + packed-IL + walk + DAG fallback)")
if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        add_subdirectory(tests)
    endif()
endif()
