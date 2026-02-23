# ============================================================================
# src/CMakeLists.txt
# VectorFFT source tree — add radix subdirectories here as they become ready
# ============================================================================

# ── Ready ────────────────────────────────────────────────────────────────
add_subdirectory(radix4)
add_subdirectory(radix8)

# ── Not ready yet (uncomment when each radix is migrated) ────────────────
# add_subdirectory(radix5)
# add_subdirectory(radix7)
# add_subdirectory(radix8)
# add_subdirectory(radix11)
# add_subdirectory(radix13)
# add_subdirectory(radix16)
# add_subdirectory(radix32)
# add_subdirectory(radix64)
