// src/python_bindings/binding.cpp
//
// PyBind11 bindings for the **ParallelOptimizationEngine** framework.
// Exposes C++ classes (`Agent`, `OptimizationEngine`, `CudaOptimizationEngine`)
// and utility functions to Python, enabling seamless integration with the
// Python orchestration layer (`run_simulation.py`).
//
// The bindings leverage PyBind11's zero-copy capabilities to pass `std::vector<Agent>`
// directly to C++ without copying, ensuring high performance. The Factory Pattern
// is implemented via `create_strategy` to instantiate appropriate engine and
// strategy combinations based on user-specified method and mode.
//
// Key bindings:
// • `Agent`: Constructor, cost, gradient, and local minimum access.
// • `OptimizationEngine`: CPU-based engine with strategy selection.
// • `CudaOptimizationEngine`: GPU-based engine with strategy selection.
// • `generateAgents`: Utility to create random agents in C++ for testing.
// • `computeClosedForm`: Computes exact global minimum for verification.
//
// Modified: Added explicit binding list; included CudaOptimizationEngine
// and computeClosedForm.
// Reason: To align with project's concise documentation style (e.g., util.hpp, CudaEngine.hpp),
// reflect new bindings for GPU engine and correctness testing, and support updated functionality
// in CudaEngine.cu and util.cpp.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Modified: Updated include paths to local directory.
// Reason: To simplify build configuration, assuming a flattened include structure or updated
// CMake setup, ensuring headers are resolved correctly.
#include "Agent.hpp"
#include "Engine.hpp"
#include "CudaEngine.hpp"
#include "util.hpp"
namespace py = pybind11;

// Modified: Renamed from createStrategy/createEngine to create_strategy; added method/mode
// parameters; added support for threadpool, openmp, ml modes.
// Reason: To align with run_simulation.py's two-argument structure, support new execution
// modes (threadpool, openmp, ml) per performance addendum, and streamline factory pattern
// for clarity and extensibility.
/**
 * @brief Factory function to create an optimization engine based on method and mode.
 *
 * @param method Optimization method ("naive" or "collaborative").
 * @param mode Execution mode ("cpu", "threadpool", "openmp", "gpu", "ml").
 * @return OptimizationEngine* Pointer to the created engine (Python manages ownership).
 *
 * This implements the **Factory Pattern**, abstracting strategy and engine creation.
 * Throws if method/mode combination is invalid. Supports auto-detection fallbacks
 * (e.g., CPU if GPU unavailable) in the Python layer.
 */
OptimizationEngine* create_strategy(const std::string& method, const std::string& mode) {
    OptimizationStrategy* strategy = nullptr;
    // Select naive strategy based on mode
    if (method == "naive") {
        if (mode == "cpu") {
            strategy = new NaiveSequentialStrategy();
        } else if (mode == "threadpool") {
            strategy = new NaiveParallelCPUStrategy();
        // Added: Support for OpenMP mode.
        // Reason: To enable OpenMP-based parallel reduction for naive strategy, per performance addendum.
        } else if (mode == "openmp") {
            strategy = new NaiveOpenMPStrategy();
        } else if (mode == "gpu") {
            strategy = new NaiveCudaStrategy();
        } else if (mode == "ml") {
            strategy = new NaiveMLStrategy();
        }
    // Select collaborative strategy based on mode
    } else if (method == "collaborative") {
        if (mode == "cpu") {
            strategy = new CollaborativeSequentialStrategy();
        } else if (mode == "threadpool") {
            strategy = new CollaborativeParallelStrategy();
        // Added: Support for OpenMP mode.
        // Reason: To enable OpenMP-based parallel gradient descent, per performance addendum.
        } else if (mode == "openmp") {
            strategy = new CollaborativeOpenMPStrategy();
        } else if (mode == "gpu") {
            strategy = new CollaborativeCudaStrategy();
        } else if (mode == "ml") {
            strategy = new CollaborativeMLStrategy();
        }
    }
    // Modified: Updated error message to include method and mode.
    // Reason: To improve debuggability for invalid inputs.
    if (strategy == nullptr) {
        throw std::runtime_error("Unknown method or mode: " + method + ", " + mode);
    }
    // Select engine based on mode (GPU or CPU)
    if (mode == "gpu") {
        return new CudaOptimizationEngine(strategy);
    }
    return new OptimizationEngine(strategy);
}

// Modified: Simplified module docstring; removed version attribute; updated bindings for
// Agent, OptimizationEngine, CudaOptimizationEngine; added computeClosedForm.
// Reason: To align with concise style, reflect new GPU engine binding, and support
// correctness harness with computeClosedForm.
/**
 * @brief PyBind11 module definition: `poe_bindings`
 *
 * Exposes C++ classes and utilities to Python for optimization and testing.
 */
PYBIND11_MODULE(poe_bindings, m) {
    m.doc() = "Python bindings for ParallelOptimizationEngine";
    // Bind Agent class
    // Modified: Simplified binding; used def_readwrite instead of def_readonly; removed docstrings.
    // Reason: To reduce verbosity and allow modification of a, b in Python for testing flexibility,
    // though convexity (a > 0) must be enforced externally.
    py::class_<Agent>(m, "Agent")
        .def(py::init<double, double>())
        .def_readwrite("a", &Agent::a)
        .def_readwrite("b", &Agent::b)
        .def("computeCost", &Agent::computeCost)
        .def("computeGradient", &Agent::computeGradient)
        .def("getLocalMin", &Agent::getLocalMin);
    // Bind OptimizationEngine class
    // Modified: Simplified binding; removed docstring; renamed return variable to result.
    // Reason: To align with concise style and improve clarity.
    py::class_<OptimizationEngine>(m, "OptimizationEngine")
        .def("run", [](OptimizationEngine& self, const std::vector<Agent>& agents) {
            double iterations = 0.0, time_taken = 0.0;
            double result = self.run(agents, iterations, time_taken);
            return py::make_tuple(result, iterations, time_taken);
        });
    // Added: Binding for CudaOptimizationEngine as a subclass of OptimizationEngine.
    // Reason: To explicitly expose GPU engine for testing and customization, aligning with
    // updates in CudaEngine.hpp/cu.
    py::class_<CudaOptimizationEngine, OptimizationEngine>(m, "CudaOptimizationEngine")
        .def(py::init<OptimizationStrategy*>())
        .def("run", [](CudaOptimizationEngine& self, const std::vector<Agent>& agents) {
            double iterations = 0.0, time_taken = 0.0;
            double result = self.run(agents, iterations, time_taken);
            return py::make_tuple(result, iterations, time_taken);
        });
    // Bind factory function
    // Modified: Updated to create_strategy with method/mode arguments.
    // Reason: To support new modes (threadpool, openmp, ml) and align with run_simulation.py.
    m.def("create_strategy", &create_strategy, "Create an optimization engine with specified method and mode");
    // Bind utility functions
    // Modified: Renamed generate_agents to generateAgents for consistency.
    // Reason: To align with C++ naming conventions in util.hpp.
    m.def("generateAgents", &generateAgents, "Generate N agents with random coefficients");
    // Added: Binding for computeClosedForm.
    // Reason: To support correctness harness in test_correctness.py, enabling verification
    // of solver outputs against the analytical solution.
    m.def("computeClosedForm", &computeClosedForm, "Compute closed-form solution for verification");
}