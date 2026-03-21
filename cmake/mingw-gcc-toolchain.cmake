set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

# Find GCC from PATH — no hardcoded paths
find_program(_MINGW_GCC gcc)
find_program(_MINGW_GXX g++)
find_program(_MINGW_RC windres)

if(_MINGW_GCC)
    set(CMAKE_C_COMPILER "${_MINGW_GCC}")
endif()
if(_MINGW_GXX)
    set(CMAKE_CXX_COMPILER "${_MINGW_GXX}")
endif()
if(_MINGW_RC)
    set(CMAKE_RC_COMPILER "${_MINGW_RC}")
endif()
