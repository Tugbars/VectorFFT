# Top-level CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(highspeedFFT_project LANGUAGES C CXX)

# Set language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable GoogleTest/FFF tests (can be changed manually)
set(ENABLE_GTEST OFF CACHE BOOL "Enable GoogleTest-based unit tests")

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