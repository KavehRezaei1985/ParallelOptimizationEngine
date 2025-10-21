# Top-level CMakeLists.txt for the ParallelOptimizationEngine project.
# 
# This file configures the build system, enforces a minimum CMake version,
# locates required external packages (CUDA, PyBind11), and orchestrates
# compilation of the modular sub-components:
#   • src/core          – CPU-only library (agents, strategies, utilities)
#   • src/cuda          – CUDA kernels and host-side GPU engine
#   • src/python_bindings – PyBind11 module exposing C++ to Python
#
# The build is fully hardware-aware: CUDA is detected automatically and
# used only when available; otherwise the project falls back to CPU-only
# compilation.

cmake_minimum_required(VERSION 3.20)                     # Enforce modern CMake features.
project(ParallelOptimizationEngine LANGUAGES CXX CUDA) # Project name; enable C++ and CUDA.

# ----------------------------------------------------------------------
# CUDA detection and compilation flags
# ----------------------------------------------------------------------
# FindCUDA is deprecated in newer CMake versions; using the native
# CMake CUDA language support is preferred.  The following line locates
# the CUDA toolkit and makes the CUDA compiler (nvcc) available.
find_package(CUDA REQUIRED)

# NVCC flags:
#   • -std=c++11  – compatibility with the rest of the C++ codebase
#   • -Xcompiler=-fPIC – generate position-independent code for shared libs
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -Xcompiler=-fPIC")

# ----------------------------------------------------------------------
# Sub-directory inclusion
# ----------------------------------------------------------------------
# Each sub-directory contains its own CMakeLists.txt that builds a
# static/shared library and registers its public include paths.
add_subdirectory(src/core)          # Core CPU library (Agent, Engine, ThreadPool, utilities)
add_subdirectory(src/cuda)          # CUDA kernels and GPU engine implementation
add_subdirectory(src/python_bindings) # PyBind11 bindings exposing C++ to Python