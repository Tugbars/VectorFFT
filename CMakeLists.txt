# ============================================================================
# VectorFFT — High-Performance Mixed-Radix FFT Library
# Root CMakeLists.txt
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# Auto-detect vcpkg if present in project root
if(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(VectorFFT
    VERSION   2.0.0
    LANGUAGES C
)

# ── Default build type ──────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# ── Intel compiler runtime library path (ICX needs libircmt.lib) ──────
if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    get_filename_component(_icx_bin "${CMAKE_C_COMPILER}" DIRECTORY)
    get_filename_component(_icx_root "${_icx_bin}" DIRECTORY)
    if(EXISTS "${_icx_root}/lib")
        link_directories("${_icx_root}/lib")
    elseif(EXISTS "${_icx_root}/compiler/lib")
        link_directories("${_icx_root}/compiler/lib")
    endif()
endif()

# ── Stride-FFT module ──────────────────────────────────────────────────
add_subdirectory(src/stride-fft)

# ── Install rules ──────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the static library
install(TARGETS vfft
    EXPORT VectorFFTTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install the public header
install(FILES include/vfft.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets for find_package
install(EXPORT VectorFFTTargets
    FILE VectorFFTTargets.cmake
    NAMESPACE VectorFFT::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VectorFFT
)

# Generate config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VectorFFTConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/VectorFFTConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VectorFFT
)

# Generate version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VectorFFTConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/VectorFFTConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/VectorFFTConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VectorFFT
)

# ── pkg-config ────────────────────────────────────────────────────────
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vectorfft.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/vectorfft.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vectorfft.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# ── Summary ─────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "  VectorFFT ${PROJECT_VERSION}")
message(STATUS "  ────────────────────────────────")
message(STATUS "  Build:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:    ${CMAKE_C_COMPILER_ID} (${CMAKE_C_COMPILER_VERSION})")
message(STATUS "  Stride ISA:  ${VFFT_STRIDE_ISA}")
message(STATUS "  Install:     ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
