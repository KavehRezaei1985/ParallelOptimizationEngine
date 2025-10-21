// src/python_bindings/binding.cpp
//
// PyBind11 bindings for the **ParallelOptimizationEngine** framework.
//
// This file exposes the complete C++ optimization engine hierarchy to Python,
// enabling seamless integration with `run_simulation.py`, ML components,
// and visualization tools.  It implements:
//
//   • **Strategy Factory**: Runtime selection of optimization algorithm
//     (naive, collaborative) and execution backend (CPU, GPU, ML placeholder).
//   • **Engine Factory**: Instantiates the appropriate `OptimizationEngine`
//     facade (CPU or CUDA) based on user mode, preserving RAII ownership.
//   • **Pythonic Interface**: Clean, idiomatic bindings for `Agent`,
//     `OptimizationEngine`, and utility functions.
//
// The module follows **zero-copy** principles where possible and uses
// `py::return_value_policy::take_ownership` to transfer C++ object lifetime
// management to Python.  All exposed types are fully documented and versioned.
//
// Design goals:
//   • **Hardware transparency**: Users select mode via string; backend is auto-detected.
//   • **Extensibility**: New strategies are added by extending `createStrategy`.
//   • **Performance**: Minimal Python overhead; core logic remains in compiled C++/CUDA.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../core/Agent.hpp"
#include "../core/Engine.hpp"
#include "../core/util.hpp"
#include "../cuda/CudaEngine.hpp"

namespace py = pybind11;

/**
 * @brief Factory function to instantiate optimization strategies.
 *
 * @param mode String identifier for the desired algorithm and backend.
 * @return OptimizationStrategy* Heap-allocated strategy (ownership transferred).
 *
 * Supported modes:
 *   - `"naive"`: Sequential unweighted averaging
 *   - `"naive_parallel_cpu"`: ThreadPool-accelerated averaging
 *   - `"naive_gpu"`: CUDA parallel reduction
 *   - `"naive_ml"`: ML placeholder (mirrors sequential)
 *   - `"collaborative_cpu"`: ThreadPool consensus GD
 *   - `"collaborative_gpu"`: CUDA consensus GD
 *
 * Throws `std::runtime_error` on unknown mode for early failure detection.
 */
OptimizationStrategy* createStrategy(const std::string& mode) {
    if (mode == "naive") return new NaiveSequentialStrategy();
    if (mode == "naive_parallel_cpu") return new NaiveParallelCPUStrategy();
    if (mode == "naive_gpu") return new NaiveCudaStrategy();
    if (mode == "naive_ml") return new NaiveMLStrategy();
    if (mode == "collaborative_cpu") return new CollaborativeParallelStrategy();
    if (mode == "collaborative_gpu") return new CollaborativeCudaStrategy();
    throw std::runtime_error("Unknown mode: " + mode);
}

/**
 * @brief Factory function to instantiate the appropriate optimization engine.
 *
 * @param mode Execution mode string (passed to `createStrategy`).
 * @return OptimizationEngine* Heap-allocated engine (ownership transferred).
 *
 * Automatically selects:
 *   - `CudaOptimizationEngine` if mode contains `"gpu"`
 *   - `OptimizationEngine` otherwise (CPU backend)
 *
 * Ensures correct facade is used for GPU resource management.
 */
OptimizationEngine* createEngine(const std::string& mode) {
    auto strat = createStrategy(mode);
    if (mode.find("gpu") != std::string::npos) {
        return new CudaOptimizationEngine(strat);
    }
    return new OptimizationEngine(strat);
}

/**
 * @brief PyBind11 module definition: `poe_bindings`
 *
 * Exposes C++ types and functions to Python under the module name
 * `poe_bindings`.  Includes version metadata and comprehensive docstrings.
 */
PYBIND11_MODULE(poe_bindings, m) {
    m.doc() = "High-performance C++ bindings for ParallelOptimizationEngine";

    // Expose version for debugging and reproducibility
    m.attr("__version__") = "1.0.0";

    // ===================================================================
    // Agent class binding
    // ===================================================================
    py::class_<Agent>(m, "Agent",
        "Convex quadratic agent with cost f(x) = a*(x-b)^2")
        .def(py::init<double, double>(),
             "Initialize agent with coefficients a (convexity) and b (target)",
             py::arg("a"), py::arg("b"))
        .def("compute_cost", &Agent::computeCost,
             "Evaluate local cost at x", py::arg("x"))
        .def("compute_gradient", &Agent::computeGradient,
             "Compute local gradient df/dx at x", py::arg("x"))
        .def("get_local_min", &Agent::getLocalMin,
             "Return local minimum x* = b")
        .def_readonly("a", &Agent::a, "Scaling coefficient a > 0")
        .def_readonly("b", &Agent::b, "Target point b");

    // ===================================================================
    // OptimizationEngine base class
    // ===================================================================
    py::class_<OptimizationEngine>(m, "OptimizationEngine",
        "Facade for running optimization with timing")
        .def("run",
             [](OptimizationEngine& self, const std::vector<Agent>& agents) {
                 double iterations = 0, time_taken = 0;
                 double x = self.run(agents, iterations, time_taken);
                 return py::make_tuple(x, iterations, time_taken);
             },
             "Execute optimization and return (final_x, iterations, time_seconds)");

    // ===================================================================
    // Utility functions
    // ===================================================================
    m.def("generate_agents", &generateAgents,
          "Generate N random convex agents",
          py::arg("N"));

    m.def("compute_global_cost", &computeGlobalCost,
          "Compute sum of all agent costs at x",
          py::arg("agents"), py::arg("x"));

    // ===================================================================
    // Factory function
    // ===================================================================
    m.def("create_engine", &createEngine,
          "Create optimization engine for given mode",
          py::arg("mode"),
          py::return_value_policy::take_ownership);
}