# ============================================================================
# src/CMakeLists.txt
# VectorFFT source tree — add radix subdirectories here as they become ready
# ============================================================================

# ── Ready ────────────────────────────────────────────────────────────────
add_subdirectory(radix2)
add_subdirectory(radix3)
add_subdirectory(radix4)
add_subdirectory(radix5)
add_subdirectory(radix7)
add_subdirectory(radix8)
add_subdirectory(radix10)
add_subdirectory(radix11)
add_subdirectory(radix13)
add_subdirectory(radix16)
add_subdirectory(radix17)
add_subdirectory(radix19)
add_subdirectory(radix20)
add_subdirectory(radix23)
add_subdirectory(radix25)
add_subdirectory(radix32)
add_subdirectory(radix64)
add_subdirectory(radix128)

add_subdirectory(3-execute)

# ── Not ready yet (uncomment when each radix is migrated) ────────────────





