#==============================================================================
# LIBRARY: highspeedFFT
#==============================================================================
add_library(highspeedFFT STATIC
    # Core dispatcher and initialization
    highSpeedFFT.c
    
    # Utility modules
    real.c
    convolve.c
    
    # Radix implementations
    fft_radix2.c
    fft_radix3.c    # Uncomment as you extract them
    fft_radix4.c
    fft_radix5.c
    fft_radix7.c
    fft_radix8.c
    fft_radix11.c
    fft_radix13.c
    fft_radix16.c
    fft_radix32.c
)

target_include_directories(highspeedFFT PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

#==============================================================================
# COMPILER FLAGS (Platform-Specific)
#==============================================================================
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(highspeedFFT PRIVATE
        -O3
        -march=native
        -mtune=native
        -mavx2
        -mfma
        -msse4.2
        
        # Link-Time Optimization for cross-file inlining
        -flto
        
        # Safer fast-math components
        -fno-math-errno
        -ffinite-math-only

        -funroll-loops
        -Wall
        -Wextra
    )
    
    # Enable LTO at link time
    set_target_properties(highspeedFFT PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
    
    # AVX-512 (optional)
    if(ENABLE_AVX512)
        message(STATUS "Library: AVX-512 enabled")
        target_compile_options(highspeedFFT PRIVATE -mavx512f -mavx512dq)
        target_compile_definitions(highspeedFFT PUBLIC HAS_AVX512)
    endif()
    
elseif(MSVC)
    target_compile_options(highspeedFFT PRIVATE
        /O2
        /arch:AVX2
        /fp:fast
        /W4
        /GL              # Whole Program Optimization (MSVC's LTO)
    )
    
    # Enable LTO for MSVC
    set_target_properties(highspeedFFT PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
    
    if(ENABLE_AVX512)
        message(STATUS "Library: AVX-512 enabled (MSVC)")
        target_compile_options(highspeedFFT PRIVATE /arch:AVX512)
        target_compile_definitions(highspeedFFT PUBLIC HAS_AVX512)
    endif()
endif()

#==============================================================================
# FEATURE FLAGS
#==============================================================================
# Prefetch system
if(ENABLE_PREFETCH)
    message(STATUS "Library: Prefetch system ENABLED")
    target_compile_definitions(highspeedFFT PUBLIC FFT_ENABLE_PREFETCH)
    
    if(DEBUG_PREFETCH)
        message(STATUS "Library: Prefetch debugging ENABLED")
        target_compile_definitions(highspeedFFT PUBLIC FFT_DEBUG_PREFETCH)
    endif()
else()
    message(STATUS "Library: Prefetch system DISABLED")
endif()

# Alignment debugging
if(DEBUG_ALIGNMENT)
    message(STATUS "Library: Alignment debugging ENABLED")
    target_compile_definitions(highspeedFFT PUBLIC FFT_DEBUG_ALIGNMENT)
endif()

if(STRICT_ALIGNMENT)
    message(STATUS "Library: Strict alignment checks ENABLED")
    target_compile_definitions(highspeedFFT PUBLIC FFT_STRICT_ALIGNMENT)
endif()

#==============================================================================
# LINK LIBRARIES
#==============================================================================
if(NOT MSVC)
    target_link_libraries(highspeedFFT PUBLIC m pthread)
endif()

#==============================================================================
# INSTALLATION (Optional)
#==============================================================================
install(TARGETS highspeedFFT
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES
    highSpeedFFT.h
    real.h
    convolve.h
    simd_utils.h
    simd_math.h
    simd_math_impl.h
    fft_radix2.h
    fft_radix3.h
    fft_radix4.h
    fft_radix5.h
    fft_radix7.h
    fft_radix8.h
    fft_radix11.h
    fft_radix13.h
    fft_radix16.h
    fft_radix32.h
    # Add other radix headers as extracted
    DESTINATION include
)
