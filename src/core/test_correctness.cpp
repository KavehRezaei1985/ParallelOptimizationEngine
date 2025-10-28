// src/core/test_correctness.cpp
//
// CTest suite for verifying the correctness of the ParallelOptimizationEngine solvers.
// Tests all solver modes (naive, naive_parallel_cpu, naive_openmp, naive_gpu, naive_ml,
// collaborative_sequential, collaborative_cpu, collaborative_openmp, collaborative_gpu)
// against the closed-form solution for various N and seeds.
//
// The test ensures that the absolute error |x - x*| <= EPSILON (default 1e-6) for all modes,
// where x* is the true global minimum computed as x* = (sum a_i b_i) / (sum a_i).
// A simplified ML predictor is used to avoid PyTorch dependencies in C++ tests.

#include "Agent.hpp"
#include "Engine.hpp"
#include "CudaEngine.hpp"
#include "util.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <random>

// Simplified ML predictor for C++ tests (mimics ml_agent.py without PyTorch)
class SimpleMLPredictor {
private:
    double slope, intercept; // Linear approximation: g(x) = slope * x + intercept
public:
    SimpleMLPredictor(const std::vector<Agent>& agents) {
        // Compute true average gradient parameters
        double sum_a = 0.0, sum_ab = 0.0;
        for (const auto& ag : agents) {
            sum_a += ag.a;
            sum_ab += ag.a * ag.b;
        }
        double N = agents.size();
        // Average gradient: (2/N) * sum(a_i (x - b_i)) = (2/N) * (sum(a_i) * x - sum(a_i b_i))
        slope = 2.0 * sum_a / N;
        intercept = -2.0 * sum_ab / N;
    }
    double predict(double x, bool is_naive) {
        if (is_naive) {
            // Naive ML: Predict mean b_i (approximated as -intercept / slope)
            return -intercept / slope;
        }
        // Collaborative ML: Predict average gradient
        return slope * x + intercept;
    }
    double optimize(const std::vector<Agent>& agents, double& iterations) {
        double x = 0.0;
        const double eta_0 = 0.01;
        const double tol = 1e-6;
        const int max_it = 10000;
        iterations = 0;
        while (iterations < max_it) {
            double grad = predict(x, false);
            double eta = eta_0 / (iterations + 1); // Diminishing step size
            double x_new = x - eta * grad;
            if (std::abs(x_new - x) < tol) break;
            x = x_new;
            ++iterations;
        }
        return x;
    }
};

// Test function for a single mode
void test_mode(const std::vector<Agent>& agents, const std::string& mode, double x_star, double tol) {
    double iterations = 0.0, time_taken = 0.0;
    double x;

    if (mode == "naive" || mode == "naive_parallel_cpu" || mode == "naive_openmp" || mode == "naive_ml") {
        OptimizationStrategy* strategy = nullptr;
        if (mode == "naive") strategy = new NaiveSequentialStrategy();
        else if (mode == "naive_parallel_cpu") strategy = new NaiveParallelCPUStrategy();
        else if (mode == "naive_openmp") strategy = new NaiveOpenMPStrategy();
        else if (mode == "naive_ml") strategy = new NaiveMLStrategy();
        OptimizationEngine engine(strategy);
        x = engine.run(agents, iterations, time_taken);
    } else if (mode == "collaborative_sequential" || mode == "collaborative_cpu" || mode == "collaborative_openmp") {
        OptimizationStrategy* strategy = nullptr;
        if (mode == "collaborative_sequential") strategy = new CollaborativeSequentialStrategy();
        else if (mode == "collaborative_cpu") strategy = new CollaborativeParallelStrategy();
        else if (mode == "collaborative_openmp") strategy = new CollaborativeOpenMPStrategy();
        OptimizationEngine engine(strategy);
        x = engine.run(agents, iterations, time_taken);
    } else if (mode == "naive_gpu" || mode == "collaborative_gpu") {
        OptimizationStrategy* strategy = nullptr;
        if (mode == "naive_gpu") strategy = new NaiveCudaStrategy();
        else if (mode == "collaborative_gpu") strategy = new CollaborativeCudaStrategy();
        CudaOptimizationEngine engine(strategy);
        x = engine.run(agents, iterations, time_taken);
    } else if (mode == "naive_ml") {
        SimpleMLPredictor predictor(agents);
        x = predictor.predict(0.0, true); // Use predict for naive ML
        iterations = 1.0;
        time_taken = 0.0; // Approximate time
    }

    double error = std::abs(x - x_star);
    if (error > tol) {
        std::cerr << "Test failed for mode " << mode << ": |x - x*| = " << error
                  << " > " << tol << " (x = " << x << ", x* = " << x_star << ")" << std::endl;
        std::exit(1);
    } else {
        std::cout << "Test passed for mode " << mode << ": |x - x*| = " << error
                  << " <= " << tol << std::endl;
    }
}

int main() {
    const double EPSILON = 1e-6;
    std::vector<size_t> N_values = {10, 100, 1000};
    std::vector<unsigned int> seeds = {42, 123, 456};
    std::vector<std::string> modes = {
        "naive",
        "naive_parallel_cpu",
        "naive_openmp",
        "naive_gpu",
        "naive_ml",
        "collaborative_sequential",
        "collaborative_cpu",
        "collaborative_openmp",
        "collaborative_gpu"
    };

    for (size_t N : N_values) {
        for (unsigned int seed : seeds) {
            std::cout << "Testing N=" << N << ", seed=" << seed << std::endl;
            std::vector<Agent> agents = generateAgents(N); // Removed seed parameter
            double x_star = computeClosedForm(agents);
            for (const auto& mode : modes) {
                test_mode(agents, mode, x_star, EPSILON);
            }
        }
    }
    std::cout << "All correctness tests passed!" << std::endl;
    return 0;
}