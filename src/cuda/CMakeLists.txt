# src/cuda/CMakeLists.txt
#
# Configures the build of the **CUDA GPU acceleration library** (`libcuda_engine.a`)
# for the **ParallelOptimizationEngine** framework.
#
# This CMake script compiles:
#   • `CudaEngine.cu` – Host-side orchestration, memory management, kernel launches.
#   • `kernel.cu`     – Device-side CUDA kernels for parallel gradient computation
#                       and reduction operations.
#
# The resulting **static library** is position-independent (PIC) to enable seamless
# integration into shared objects (e.g., PyBind11 Python module) and dynamic
# loading at runtime.
#
# Public include directories are exported so that downstream targets can locate
# `CudaEngine.hpp` and invoke GPU-accelerated strategies via the unified
# `OptimizationStrategy` interface.
#
# CUDA language support is enabled at the top-level `CMakeLists.txt`.  This
# sub-module assumes `find_package(CUDA REQUIRED)` has already succeeded.

cuda_add_library(cuda_engine STATIC CudaEngine.cu kernel.cu)

# Export the current source directory as a public include path.
# Enables `#include "CudaEngine.hpp"` from any target linking against libcuda_engine.
target_include_directories(cuda_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Enable position-independent code generation.
# Essential for static library inclusion in shared libraries (e.g., Python extension).
set_target_properties(cuda_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)