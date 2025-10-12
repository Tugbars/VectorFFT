cmake_minimum_required(VERSION 3.10)

project(highspeedFFT_project LANGUAGES C CXX)

# Set language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#==============================================================================
# SIMD OPTIONS (Define at top level so subdirectories can access)
#==============================================================================
option(ENABLE_GTEST "Enable GoogleTest-based unit tests" OFF)
option(ENABLE_AVX512 "Enable AVX-512 instructions..." ON)

# Fetch dependencies conditionally
if(ENABLE_GTEST)
    include(FetchContent)

    # GoogleTest
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)

    # FFF (Fake Function Framework)
    FetchContent_Declare(
        fff
        GIT_REPOSITORY https://github.com/meekrosoft/fff.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(fff)
endif()

# Add project subdirectories
add_subdirectory(src)
add_subdirectory(test)

# Enable CTest
enable_testing()

#==============================================================================
# BUILD SUMMARY
#==============================================================================
message(STATUS "============================================")
message(STATUS "Build Configuration:")
message(STATUS "  GoogleTest: ${ENABLE_GTEST}")
message(STATUS "  AVX-512:    ${ENABLE_AVX512}")
message(STATUS "============================================")