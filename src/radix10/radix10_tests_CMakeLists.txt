# VectorFFT — radix10 tests
add_executable(test_radix10_tw test_radix10_tw.c)
target_link_libraries(test_radix10_tw PRIVATE vfft_radix10 m)
target_compile_options(test_radix10_tw PRIVATE ${VFFT_ISA_FLAGS})
add_test(NAME radix10_tw COMMAND test_radix10_tw)
