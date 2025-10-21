# src/python_bindings/CMakeLists.txt
#
# Configures the build of the **PyBind11 Python extension module** (`poe_bindings.so`)
# for the **ParallelOptimizationEngine** framework.
#
# This CMake script:
#   • Locates the PyBind11 package (required for Python/C++ interoperability).
#   • Compiles `binding.cpp` into a loadable shared library using `pybind11_add_module`.
#   • Links against the static core and CUDA libraries to resolve all symbols.
#   • Sets explicit output properties for consistent naming and versioning.
#
# The resulting module exposes the full C++ optimization engine hierarchy
# (CPU, GPU, and strategy interfaces) to Python, enabling seamless integration
# with `run_simulation.py` and ML components.
#
# Build requirements:
#   • PyBind11 (found via `find_package`)
#   • Pre-built `core` and `cuda_engine` static libraries (from src/core and src/cuda)
#   • CUDA runtime libraries (if GPU support is enabled)
#
# The module is position-independent and compatible with CPython 3.8+.

find_package(pybind11 REQUIRED)

# Create the Python extension module from binding.cpp
pybind11_add_module(poe_bindings binding.cpp)

# Link static libraries and CUDA runtime
#   • core: CPU strategies, agents, utilities
#   • cuda_engine: GPU kernels and strategies
#   • ${CUDA_LIBRARIES}: CUDA runtime (libcudart) for device API calls
target_link_libraries(poe_bindings PRIVATE core cuda_engine ${CUDA_LIBRARIES})

# Configure output properties for consistent deployment
set_target_properties(poe_bindings PROPERTIES
    LIBRARY_OUTPUT_NAME "poe_bindings"   # Module name: import poe_bindings
    SUFFIX ".so"                         # Force .so extension (Linux/macOS)
    VERSION "1.0.0"                      # Semantic version for tracking
)