add_library(vfft_radix16 INTERFACE)
target_include_directories(vfft_radix16 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar)
