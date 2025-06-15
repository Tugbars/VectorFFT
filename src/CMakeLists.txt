# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Define the project name and specify the language
project(highspeedFFT C)

# Set the C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Define the source directory
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add all source files from the src directory
set(SOURCES
    ${SRC_DIR}/highspeedFFT.c
    ${SRC_DIR}/convolve.c
    ${SRC_DIR}/real.c
)

# Add all header files (optional, but good practice for IDEs)
set(HEADERS
    ${SRC_DIR}/highspeedFFT.h
    ${SRC_DIR}/real.h
)

# Create a static library instead of an executable
add_library(highspeedFFT STATIC ${SOURCES} ${HEADERS})

# Add the src directory to the include path so headers can be found
target_include_directories(highspeedFFT PUBLIC ${SRC_DIR})

# Link math library
target_link_libraries(highspeedFFT PRIVATE m)

# Enable AVX and FMA SIMD optimizations
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(highspeedFFT PRIVATE -O3 -mavx -mfma)
elseif(MSVC)
    target_compile_options(highspeedFFT PRIVATE /O2 /arch:AVX2)
endif()