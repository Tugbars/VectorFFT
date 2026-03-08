# ============================================================================
# VectorFFT — radix25 tests
# ============================================================================

# ── Twiddled (DIT + DIF) correctness test ──
add_executable(test_radix25_tw test_radix25_tw.c)
target_compile_options(test_radix25_tw PRIVATE ${VFFT_ISA_FLAGS})
if(NOT WIN32)
    target_link_libraries(test_radix25_tw PRIVATE vfft_radix25 m)
else()
    target_link_libraries(test_radix25_tw PRIVATE vfft_radix25)
endif()
target_compile_options(test_radix25_tw PRIVATE ${VFFT_ISA_FLAGS})
add_test(NAME radix25_tw COMMAND test_radix25_tw)
