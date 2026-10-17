# how-to-use/examples/CMakeLists.txt — the guide's four examples.
#
# In the repository the root CMakeLists.txt adds this folder (VFFT_BUILD_EXAMPLES,
# on by default at the top level) and the examples link the library built beside
# them. On its own, this folder builds against an installed VectorFFT:
#
#   cmake -S how-to-use/examples -B build-examples -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build build-examples
cmake_minimum_required(VERSION 3.16)
project(vfft_examples LANGUAGES C)

# Built on its own, an unnamed build type is Release (in the repository the root sets it).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT TARGET VectorFFT::vfft)
    find_package(VectorFFT CONFIG REQUIRED)
endif()

foreach(_ex convolve3d gradient3d poisson3d wavepacket3d)
    add_executable(${_ex} ${_ex}.c)
    target_link_libraries(${_ex} PRIVATE VectorFFT::vfft m)   # m: the examples' own sin, cos, exp
endforeach()
