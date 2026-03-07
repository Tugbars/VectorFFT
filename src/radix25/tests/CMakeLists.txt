# ============================================================================
# VectorFFT — radix25 tests
# ============================================================================

# ── N1 (notw) correctness test ──
add_executable(test_radix25_n1 test_radix25_n1.c)
target_link_libraries(test_radix25_n1 PRIVATE vfft_radix25 m)
target_compile_options(test_radix25_n1 PRIVATE ${VFFT_ISA_FLAGS})
add_test(NAME radix25_n1 COMMAND test_radix25_n1)

# ── Twiddled (DIT + DIF) correctness test ──
add_executable(test_radix25_tw test_radix25_tw.c)
target_link_libraries(test_radix25_tw PRIVATE vfft_radix25 m)
target_compile_options(test_radix25_tw PRIVATE ${VFFT_ISA_FLAGS})
add_test(NAME radix25_tw COMMAND test_radix25_tw)
