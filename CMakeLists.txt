cmake_minimum_required(VERSION 3.10)
project(highspeedFFT_project LANGUAGES C CXX)

# Set language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#==============================================================================
# BUILD OPTIONS
#==============================================================================
option(ENABLE_GTEST "Enable GoogleTest-based unit tests" OFF)
option(ENABLE_AVX512 "Enable AVX-512 instructions" ON)
option(ENABLE_PREFETCH "Enable advanced prefetch system" OFF)
option(DEBUG_PREFETCH "Enable prefetch debugging output" OFF)

#==============================================================================
# FETCH DEPENDENCIES (only if GoogleTest is enabled)
#==============================================================================
if(ENABLE_GTEST)
    message(STATUS "GoogleTest enabled - fetching dependencies")
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
else()
    message(STATUS "GoogleTest disabled - using main.c for tests")
endif()

#==============================================================================
# ADD SUBDIRECTORIES
#==============================================================================
add_subdirectory(src)
add_subdirectory(test)

# Enable CTest (even for main.c tests)
enable_testing()

#==============================================================================
# BUILD SUMMARY
#==============================================================================
message(STATUS "============================================")
message(STATUS "Build Configuration:")
message(STATUS "  Test Framework: ${ENABLE_GTEST}")
if(ENABLE_GTEST)
    message(STATUS "    → Using GoogleTest (test_mixedRadixFFT.cpp)")
else()
    message(STATUS "    → Using standalone tests (main.c)")
endif()
message(STATUS "  AVX-512:        ${ENABLE_AVX512}")
message(STATUS "  Prefetch:       ${ENABLE_PREFETCH}")
if(ENABLE_PREFETCH AND DEBUG_PREFETCH)
    message(STATUS "    → Prefetch Debug: ENABLED")
endif()
message(STATUS "============================================")