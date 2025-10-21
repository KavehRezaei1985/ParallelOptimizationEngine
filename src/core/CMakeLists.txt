# src/core/CMakeLists.txt
#
# Configures the build of the core static library (libcore.a) containing
# all CPU-only components of the ParallelOptimizationEngine:
#   • Agent.cpp      – implementation of the quadratic-cost agent model
#   • Engine.cpp     – CPU-side optimization strategies (naive & collaborative)
#   • util.cpp       – utility functions (agent generation, global cost, etc.)
#
# The resulting library is position-independent (PIC) to support static
# linking into shared objects (e.g., PyBind11 module) and dynamic loading.
# Public include paths are exported so downstream targets can locate
# headers such as Agent.hpp, Engine.hpp, and threadpool.hpp.

add_library(core STATIC Agent.cpp Engine.cpp util.cpp)

# Export the current source directory as a public include path.
# This allows #include "Agent.hpp" style includes from any target linking
# against libcore.
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Enable position-independent code generation.
# Required for the static library to be safely embedded in shared libraries
# (e.g., the Python extension module).
set_target_properties(core PROPERTIES POSITION_INDEPENDENT_CODE ON)