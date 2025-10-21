// src/core/ThreadPool.hpp
//
// High-performance, fixed-size thread pool implementation with work-stealing
// avoidance and minimal synchronization overhead.  Designed for use in
// CPU-parallel optimization strategies of the **ParallelOptimizationEngine**.
//
// Key features:
//   • **Fixed thread count** based on `std::thread::hardware_concurrency()`
//   • **Lock-based task queue** with `std::condition_variable` for efficient
//     worker wakeup
//   • **RAII-based lifecycle management** – automatic join on destruction
//   • **Zero-overhead task submission** after pool initialization
//   • **Thread-safe enqueue** supporting arbitrary callable objects via
//     perfect forwarding
//
// This implementation prioritizes **simplicity**, **predictability**, and
// **low latency** over advanced work-stealing, as the workload (per-agent
// gradient computation) is highly uniform and benefits from a centralized
// queue with minimal contention.
//
// The pool is reused across all iterations of collaborative strategies to
// eliminate thread creation/teardown costs.  All public methods are exception-
// safe and follow C++17 standards.

#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>

/**
 * @brief Fixed-size thread pool for efficient parallel task execution.
 *
 * Workers continuously wait on a condition variable and execute tasks
 * from a shared queue.  The pool is initialized with a number of threads
 * equal to hardware concurrency by default.
 *
 * @note This class is non-copyable and intended for long-lived use within
 *       optimization loops.
 */
class ThreadPool {
public:
    /**
     * @brief Constructs the thread pool with a specified number of worker threads.
     *
     * @param threads Number of worker threads.  If zero or omitted, defaults
     *                to `std::thread::hardware_concurrency()`.
     */
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency());

    /**
     * @brief Destructor: signals termination and joins all worker threads.
     *
     * Ensures graceful shutdown and resource cleanup via RAII.
     */
    ~ThreadPool();

    /**
     * @brief Enqueues a callable task for asynchronous execution.
     *
     * @tparam F Callable type (function, lambda, functor).
     * @param f  Task to execute (perfect-forwarded).
     *
     * The task is pushed to the internal queue and one waiting worker is
     * notified.  This method is thread-safe and has minimal overhead.
     */
    template<class F>
    void enqueue(F&& f);

private:
    std::vector<std::thread> workers;                    ///< Worker thread container.
    std::queue<std::function<void()>> tasks;             ///< Task queue (FIFO).
    std::mutex mtx;                                      ///< Protects queue access.
    std::condition_variable cv;                          ///< Worker wakeup notification.
    std::atomic<bool> stop{false};                       ///< Termination flag.
};

/**
 * @brief Worker thread loop: waits for tasks and executes them.
 *
 * Each worker runs this lambda indefinitely until `stop` is set.
 * Uses condition variable to block efficiently when idle.
 */
ThreadPool::ThreadPool(size_t threads) {
    if (threads == 0) threads = 1;  // Ensure at least one thread.

    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [this] { return stop || !tasks.empty(); });
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        });
    }
}

/**
 * @brief Destructor implementation: signals shutdown and joins workers.
 *
 * Acquires the mutex to set `stop = true`, notifies all workers,
 * and joins each thread to ensure clean termination.
 */
ThreadPool::~ThreadPool() {
    { 
        std::unique_lock<std::mutex> lock(mtx); 
        stop = true; 
    }
    cv.notify_all();
    for (auto& w : workers) w.join();
}

/**
 * @brief Enqueue implementation with perfect forwarding.
 *
 * Locks the mutex, emplaces the task, and notifies one waiting worker.
 * The use of `std::forward` preserves value category and enables
 * efficient move semantics for temporary objects.
 */
template<class F>
void ThreadPool::enqueue(F&& f) {
    { 
        std::unique_lock<std::mutex> lock(mtx);
        tasks.emplace(std::forward<F>(f));
    }
    cv.notify_one();
}

#endif // THREAD_POOL_HPP