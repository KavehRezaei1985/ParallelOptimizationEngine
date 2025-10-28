# src/cuda/CMakeLists.txt
#
# Configures the build of the CUDA static library (cuda_engine) containing
# GPU-accelerated components of the **ParallelOptimizationEngine** framework:
# • CudaEngine.cu – host-side CUDA code for naive and collaborative strategies
# • kernel.cu – device-side CUDA kernels for gradient computation and reduction
#
# The library is position-independent (PIC) to support linking into shared
# objects (e.g., PyBind11 module for Python bindings). Public include paths are
# exported to allow downstream targets to locate CudaEngine.hpp for invoking
# GPU-accelerated strategies.
cuda_add_library(cuda_engine STATIC CudaEngine.cu kernel.cu)

# Export the current source directory as a public include path.
# Enables `#include "CudaEngine.hpp"` from any target linking against libcuda_engine.
target_include_directories(cuda_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Enable position-independent code generation.
# Essential for static library inclusion in shared libraries (e.g., Python extension module).
set_target_properties(cuda_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Added: Link against the core library using plain signature to match FindCUDA.
# Reason: To ensure cuda_engine can access core library components (e.g., Agent class,
# utilities like computeGlobalCost, computeClosedForm), supporting GPU strategy
# implementation and correctness testing in test_correctness.cpp, which requires
# both libraries.
target_link_libraries(cuda_engine core)