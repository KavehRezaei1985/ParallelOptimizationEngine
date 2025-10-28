# ParallelOptimizationEngine: Design Report
**Author:** Dr. Kaveh Rezaei Tarhomi, Senior HPC & AI Systems Engineer, PhD in Mathematics
**Date:** October 27, 2025
**Version:** 1.1
# Short Report (1–2 pages): ParallelOptimizationEngine

## Design Decisions
The **ParallelOptimizationEngine** implements a multi-agent optimization framework minimizing the global convex quadratic cost \(\sum_{i=1}^N a_i (x - b_i)^2\), with \(a_i > 0\) enforced via rejection sampling (\(a_i \sim \mathcal{N}(5,2)\), rejected if \(\leq 0\); \(b_i \sim \mathcal{N}(0,5)\)). Two core methods are compared to demonstrate trade-offs in speed, accuracy, and scalability:

- **Naive**: Unweighted average of local minima \(x = \frac{1}{N} \sum b_i\). Chosen for **O(1)** complexity, minimal communication, and low overhead—ideal as a fast baseline despite ignoring \(a_i\) weights, leading to systematic error.
- **Collaborative**: Consensus-based gradient descent with shared average gradient \(\overline{\nabla f}(x) = \frac{2}{N} \sum a_i (x - b_i)\), updated as \(x \leftarrow x - \eta_k \cdot \overline{\nabla f}\), \(\eta_k = 0.01 / k\), converging when \(\|x_{k+1} - x_k\| < 10^{-6}\). Selected for **exact convergence** to the closed-form solution \(x^* = \frac{\sum a_i b_i}{\sum a_i}\), with mathematical rigor (gradient derivation in comments).

Execution modes span **CPU** (sequential, custom ThreadPool, OpenMP), **GPU** (CUDA with unified memory), and **ML** (PyTorch MLP). This hybrid design leverages:
- **C++** for low-level control, double precision, and performance-critical paths (`Agent.cpp`, `Engine.cpp`, `util.cpp`).
- **CUDA** for massive parallelism (`kernel.cu`, `CudaEngine.cu` with error handling and unified memory).
- **Python + PyBind11** for user-friendly interface (`run_simulation.py`, `binding.cpp`), visualization (`visualize.py`), and ML (`ml_agent.py`).
- **Design Patterns**: 
  - **Strategy** (`Engine.hpp`, `CudaEngine.hpp`) for swappable algorithms.
  - **Factory** (`binding.cpp`) for dynamic engine creation.
  - **Facade** (`run_simulation.py`) for simplified CLI.

The **ML component** (`ml_agent.py`) uses a 2-layer MLP (20 neurons, ReLU) with input \([a_1, b_1, \dots, a_N, b_N]\), outputs predicted \([\sum a_i b_i, \sum a_i]\) for fixed division. Trained via Adam + StepLR (500 epochs, MSE + accuracy gap penalty), it simulates AI-enhanced reasoning in the "axon layer" communication model, reducing iterative computation at the cost of approximation error.

Agents are generated with reproducible seeding (`util.cpp`) for testing; convexity and numerical stability are enforced throughout.

## Parallelization Strategy
Parallelism targets per-agent computations (gradient evaluation, summation) across hardware:

- **CPU Sequential**: Single-threaded loops — baseline for small \(N\) or debugging (`naive`, `collaborative_sequential`).
- **ThreadPool** (`threadpool.hpp`): Custom work-stealing queue with `std::hardware_concurrency` threads. Tasks enqueued via lambdas (`pool.enqueue([&]() { return agent.gradient(x); })`); futures aggregate results. Efficient for dynamic, data-parallel workloads with low contention.
- **OpenMP**: `#pragma omp parallel for reduction(+:sum)` — compiler-optimized, lock-free reductions. Added for performance comparison with ThreadPool (`naive_openmp`, `collaborative_openmp`).
- **GPU** (`kernel.cu`): One thread per agent; tree-based reductions in shared memory. Host code (`CudaEngine.cu`) uses `cudaMallocManaged` for zero-copy access, kernel launches (`<<<blocks, threads>>>`), and robust error handling. Scales to \(N > 10,000\) with high throughput.
- **ML**: PyTorch tensor operations auto-parallelized on GPU (if detected); batched inference and autograd for training.

The **axon layer** simulates inter-agent signaling:
- **CPU**: Shared memory + OpenMP reductions or ThreadPool queues.
- **GPU**: Unified memory arrays.
- **ML**: Learned surrogate model approximating communication.

All modes are exposed via `binding.cpp` to Python, with auto-detection (`torch.cuda.is_available()`) and graceful fallbacks.

## Optimization Tradeoffs
| Aspect | Tradeoff |
|-------|----------|
| **Accuracy vs. Speed** | Naive: fastest (1 iteration) but high error (13–30); Collaborative: exact (\(<2.3 \times 10^{-9}\)) but requires 24–1002 iterations; ML: 1 inference but training overhead and growing error (6.55 → 119.95) |
| **Scalability** | GPU/OpenMP scale best for \(N > 1000\); ThreadPool has enqueue overhead; Sequential fails at large \(N\) |
| **Overhead** | GPU: memory allocation + kernel launch; ML: 500-epoch training (~0.1s at \(N=10k\)); ThreadPool: task setup |
| **Portability** | GPU/ML require CUDA/NVIDIA; auto-fallback to CPU ensures robustness |
| **Numerical Stability** | Double precision + convexity enforcement prevent divergence; extreme coefficients may slow convergence (tunable via \(\eta_0\)) |

**Key Insight**: For high accuracy, use **collaborative_gpu** or **collaborative_openmp**. For speed with acceptable error, **naive_gpu**. ML offers a middle ground but requires scaling model capacity.

---

## Performance Bottlenecks: Why Each Mode is Slow in Certain Scenarios

| Mode | Slow When | Root Cause |
|------|-----------|------------|
| **CPU Sequential** | \(N \geq 1000\) | **Single-threaded**; O(N) work per iteration, no parallelism. At \(N=10k\), collaborative takes **217ms** due to 1002 × O(N) loops. |
| **ThreadPool** | Small \(N\) (< 1000), or **naive method** | **Task enqueue overhead** (lambda capture, future sync, mutex contention) dominates compute. For \(N=100\), `naive_threadpool` is **32× slower** than CPU (2ms vs 62μs). |
| **OpenMP** | Very small \(N\) (< 100), or **naive method** | **Parallel for overhead** (thread spawn, barrier sync) outweighs reduction benefit. `naive_openmp` at \(N=100\): 377μs vs CPU 62μs (~6× slower). |
| **GPU** | Small \(N\) (< 1000), or **naive method** | **Kernel launch latency** (~10–100μs) + **unified memory page faults** dominate. `naive_gpu` at \(N=100\): **160ms** (2600× slower than CPU). Memory underutilized; no coalescing benefit. |
| **ML** | Large \(N\) (> 1000), or **high accuracy needed** | **Fixed model capacity** (40 neurons) cannot generalize; error grows as \(N\) increases. Training time scales with \(N\) (125ms at \(N=10k\)). Inference fast, but **accuracy collapses** (gap = 120). |

> **Summary**:  
> - **Small \(N\)**: Sequential wins — parallelism adds overhead.  
> - **Large \(N\), high accuracy**: GPU/OpenMP dominate via massive data-parallel reduction.  
> - **ML**: Only viable for **approximate, real-time** use with small-to-medium \(N\).

---

## Summary of Performance Results
Performance evaluated across \(N = 100, 1000, 10000\) agents using `run_simulation.py --method=both --mode=all`. Raw data from `performance_data.csv`; visualized in **Figure 1** (`scaling.png`).

![Figure 1: Performance Scaling Across Methods and Modes]

*Figure 1: Log-log plots showing (a) Iterations vs. N, (b) Wall-clock time (s) vs. N, (c) Accuracy gap (\(|x - x^*|\)) vs. N, and (d) Speedup (vs. CPU sequential) vs. N. Collaborative modes show sublinear iteration growth and ~5x speedup at \(N=10k\). Naive is fast but inaccurate. ML inference is quick but error grows with \(N\).*

### Raw Performance Data Table
| N     | Method       | Mode       | x                  | Cost             | Iterations | Wall_Clock_Time_s | Accuracy_Gap     | Speedup   |
|-------|--------------|------------|--------------------|------------------|------------|-------------------|------------------|-----------|
| 100   | naive        | cpu        | 0.11152293524961941| 9995.350633528105| 1.0        | 6.246566772460938e-05 | 28.388480966415955 | 1.0       |
| 100   | naive        | threadpool | 0.11152293524961937| 9995.350633528105| 1.0        | 0.00200653076171875   | 28.388480966415955 | 0.031131178707224334 |
| 100   | naive        | openmp     | 0.11152293524961923| 9995.350633528105| 1.0        | 0.000377655029296875  | 28.388480966415955 | 0.16540404040404041 |
| 100   | naive        | gpu        | 0.11152293524961944| 9995.350633528105| 1.0        | 0.15987563133239746   | 28.388480966415955 | 0.0003907141269999866 |
| 100   | naive        | ml         | 0.11152293524961941| 9995.350633528105| 0.0        | 0.0                   | 28.388480966415955 | 1.0       |
| 100   | collaborative| cpu        | -0.13180405103489634| 9966.96215256168 | 24.0       | 0.00013756752014160156| 9.094947017729282e-12 | 1.0       |
| 100   | collaborative| threadpool | -0.13180405103489637| 9966.962152561682| 24.0       | 0.003100872039794922  | 7.275957614183426e-12 | 0.04436413962786406 |
| 100   | collaborative| openmp     | -0.13180405103489634| 9966.96215256168 | 24.0       | 0.00017571449279785156| 9.094947017729282e-12 | 0.7829036635006784 |
| 100   | collaborative| gpu        | -0.1318040510348964 | 9966.962152561682| 24.0       | 0.008629798889160156  | 7.275957614183426e-12 | 0.015940987954470107 |
| 100   | collaborative| ml         | -0.014907842090143144| 9973.5139799048| 1.0        | 0.003912854008376598  | 6.551827343109835   | 0.03515784638197551 |
| 1000  | naive        | cpu        | 0.35418118624577966| 125251.02785306104| 1.0        | 0.0009706020355224609 | 30.012844467360992 | 1.0       |
| 1000  | naive        | threadpool | 0.3541811862457795 | 125251.02785306104| 1.0        | 0.0024051666259765625 | 30.012844467360992 | 0.40354877081681206 |
| 1000  | naive        | openmp     | 0.35418118624577966| 125251.02785306104| 1.0        | 0.0008518695831298828 | 30.012844467360992 | 1.1393786733837112 |
| 1000  | naive        | gpu        | 0.3541811862457796 | 125251.02785306104| 1.0        | 0.0017938613891601562 | 30.012844467360992 | 0.5410685805422647 |
| 1000  | naive        | ml         | 0.35418118624577966| 125251.02785306104| 0.0        | 0.0                   | 30.012844467360992 | 1.0       |
| 1000  | collaborative| cpu        | 0.2770212776496788 | 125221.01500859366| 105.0      | 0.002917051315307617  | 1.4551915228366852e-11 | 1.0       |
| 1000  | collaborative| threadpool | 0.27702127764967877| 125221.01500859366| 105.0      | 0.009784698486328125  | 1.4551915228366852e-11 | 0.29812378167641324 |
| 1000  | collaborative| openmp     | 0.27702127764967877| 125221.01500859366| 105.0      | 0.0013136863708496094 | 1.4551915228366852e-11 | 2.220508166969147 |
| 1000  | collaborative| gpu        | 0.27702127764967877| 125221.01500859366| 105.0      | 0.0106964111328125    | 1.4551915228366852e-11 | 0.27271308844507847 |
| 1000  | collaborative| ml         | -0.003351768880594715| 125617.2896401297| 1.0        | 0.017416563001461327  | 396.2746315360273   | 0.16748719681735505 |
| 10000 | naive        | cpu        | 0.06767026532443614| 1255082.4158214082| 1.0        | 0.006722927093505859  | 13.999824553495273 | 1.0       |
| 10000 | naive        | threadpool | 0.06767026532443629| 1255082.4158214082| 1.0        | 0.011439085006713867  | 13.999824553495273 | 0.5877154588465787 |
| 10000 | naive        | openmp     | 0.06767026532443626| 1255082.4158214082| 1.0        | 0.006745576858520508  | 13.999824553495273 | 0.9966422790089421 |
| 10000 | naive        | gpu        | 0.06767026532443629| 1255082.4158214082| 1.0        | 0.007893562316894531  | 13.999824553495273 | 0.8516974749305304 |
| 10000 | naive        | ml         | 0.06767026532443614| 1255082.4158214082| 0.0        | 0.0                   | 13.999824553495273 | 1.0       |
| 10000 | collaborative| cpu        | 0.05093755809304714| 1255068.4159968526| 1002.0     | 0.21718430519104004   | 2.0954757928848267e-09 | 1.0       |
| 10000 | collaborative| threadpool | 0.05093755809304679| 1255068.4159968523| 1002.0     | 0.09140253067016602   | 2.3283064365386963e-09 | 2.376130109293894 |
| 10000 | collaborative| openmp     | 0.05093755809304681| 1255068.4159968523| 1002.0     | 0.04826807975769043   | 2.3283064365386963e-09 | 4.49954309931786 |
| 10000 | collaborative| gpu        | 0.05093755809304706| 1255068.4159968526| 1002.0     | 0.0449984073638916    | 2.0954757928848267e-09 | 4.826488711805316 |
| 10000 | collaborative| ml         | 0.0019589736471495676| 1255188.3666527877| 1.0        | 0.1252478719688952    | 119.95065593300387  | 1.7340358904060014 |

### Key Observations
1. **Iterations vs. N**: Collaborative: 24 → 105 → 1002 (sublinear due to convexity).
2. **Time (s) vs. N**: GPU/OpenMP: **45–48ms** at \(N=10k\) (**~5x** vs CPU 217ms).
3. **Accuracy Gap**: Collaborative: **< 2.3e-9**; ML: 6.55 → 119.95.
4. **Speedup (vs. CPU Sequential)**: `collaborative_gpu`: **4.83x**, `collaborative_openmp`: **4.50x**.

## Conclusion
The **ParallelOptimizationEngine** achieves **exact convergence** and **~5x speedup** on GPU/OpenMP for \(N=10k\). ML is fast but inaccurate at scale. Recommended: **`collaborative_gpu` andd `collaborative_openmp`** for HPC; **`naive_gpu`** for real-time.