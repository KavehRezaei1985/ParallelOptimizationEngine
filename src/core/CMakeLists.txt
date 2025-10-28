# src/core/CMakeLists.txt
#
# Configures the build of the core static library (libcore.a) containing
# all CPU-only components of the ParallelOptimizationEngine:
# • Agent.cpp – implementation of the quadratic-cost agent model
# • Engine.cpp – CPU-side optimization strategies (naive & collaborative)
# • util.cpp – utility functions (agent generation, global cost, etc.)
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
# Added: Find and link OpenMP for parallel strategies
# Reason: To enable OpenMP support for new parallel reduction-based strategies (e.g., NaiveOpenMPStrategy and CollaborativeOpenMPStrategy), as per the requirement to implement and debug OpenMP variants for CPU parallelism (parallel gradient sum with reduction). This ensures the library can use #pragma omp directives without linking errors.
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(core PUBLIC OpenMP::OpenMP_CXX)
endif()
# Added: Add CTest target for correctness testing
# Reason: To implement a unit test harness that verifies solver outputs against the closed-form solution across modes and random seeds, as per the requirement for a correctness harness exposed as a CTest target. This adds automated testing to the build system, runnable via 'ctest'.
add_executable(test_correctness test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE core cuda_engine ${CUDA_LIBRARIES})
add_test(NAME CorrectnessTest COMMAND test_correctness)