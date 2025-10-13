#==============================================================================
# LIBRARY: highspeedFFT
#==============================================================================
add_library(highspeedFFT STATIC
    highspeedFFT.c
    prefetch_strategy.c
    real.c
    convolve.c
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
        -ffast-math
        -funroll-loops
        -Wall
        -Wextra
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
    )
    
    if(ENABLE_AVX512)
        message(STATUS "Library: AVX-512 enabled (MSVC)")
        target_compile_options(highspeedFFT PRIVATE /arch:AVX512)
        target_compile_definitions(highspeedFFT PUBLIC HAS_AVX512)
    endif()
endif()

#==============================================================================
# PREFETCH SYSTEM (Feature Flags)
#==============================================================================
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
    highspeedFFT.h
    prefetch_strategy.h
    real.h
    convolve.h
    DESTINATION include
)