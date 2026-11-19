# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuğbars Heptaşkın
#
# src/cuda -- the CUDA FFT project. A project of its own: the root CMakeLists.txt
# does not include it, and it does not use the CPU library.
#
#   cmake -S src/cuda -B <build dir> -G "Visual Studio 17 2022" -A x64 ^
#         -DVKFFT_ROOT=<VkFFT checkout>
#   cmake --build <build dir> --config Release
cmake_minimum_required(VERSION 3.20)
project(vfft_cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# ---- baseline: VkFFT's CUDA backend, timed next to a plain copy kernel --------
set(VKFFT_ROOT "" CACHE PATH "VkFFT checkout (the folder that holds vkFFT/vkFFT.h)")
if(VKFFT_ROOT AND EXISTS "${VKFFT_ROOT}/vkFFT/vkFFT.h")
  add_executable(vkfft_baseline baseline/vkfft_baseline.cpp)
  target_include_directories(vkfft_baseline PRIVATE "${VKFFT_ROOT}/vkFFT")
  target_compile_definitions(vkfft_baseline PRIVATE VKFFT_BACKEND=1)
  target_link_libraries(vkfft_baseline PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
  message(STATUS "vkfft_baseline: VkFFT from ${VKFFT_ROOT}")
else()
  message(STATUS "vkfft_baseline: not built (pass -DVKFFT_ROOT=<VkFFT checkout>)")
endif()
