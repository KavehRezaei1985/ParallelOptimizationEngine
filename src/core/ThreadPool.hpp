// src/core/ThreadPool.hpp
//
// High-performance, fixed-size thread pool implementation with work-stealing
// avoidance and minimal synchronization overhead. Designed for use in
// CPU-parallel optimization strategies of the **ParallelOptimizationEngine**.
//
// Key features:
// • **Fixed thread count** based on `std::thread::hardware_concurrency()`
// • **Lock-based task queue** with `std::condition_variable` for efficient
// worker wakeup
// • **RAII-based lifecycle management** – automatic join on destruction
// • **Zero-overhead task submission** after pool initialization
// • **Thread-safe enqueue** supporting arbitrary callable objects via
// perfect forwarding
//
// This implementation prioritizes **simplicity**, **predictability**, and
// **low latency** over advanced work-stealing, as the workload (per-agent
// gradient computation) is highly uniform and benefits from a centralized
// queue with minimal contention.
//
// The pool is reused across all iterations of collaborative strategies to
// eliminate thread creation/teardown costs. All public methods are exception-
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
// Modified: Added <future> include
// Reason: To support std::future and std::packaged_task for task result retrieval in the updated enqueue method, enabling chunk-based parallel strategies in Engine.cpp for scalability.
#include <future>
/**
 * @brief Fixed-size thread pool for efficient parallel task execution.
 *
 * Workers continuously wait on a condition variable and execute tasks
 * from a shared queue. The pool is initialized with a number of threads
 * equal to hardware concurrency by default.
 *
 * @note This class is non-copyable and intended for long-lived use within
 * optimization loops.
 */
class ThreadPool {
public:
    /**
     * @brief Constructs the thread pool with a specified number of worker threads.
     *
     * @param threads Number of worker threads. If zero or omitted, defaults
     * to `std::thread::hardware_concurrency()`.
     */
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency());
    /**
     * @brief Destructor: signals termination and joins all worker threads.
     *
     * Ensures graceful shutdown and resource cleanup via RAII.
     */
    ~ThreadPool();
    // Modified: Updated enqueue to return std::future<R> and added template parameter R
    // Reason: To allow retrieval of task results (e.g., partial sums in NaiveParallelCPUStrategy and CollaborativeParallelStrategy), supporting chunk-based parallelism for scalability as per the performance addendum requirement.
    /**
     * @brief Enqueues a callable task for asynchronous execution.
     *
     * @tparam F Callable type (function, lambda, functor).
     * @tparam R Return type of the callable (deduced).
     * @param f Task to execute (perfect-forwarded).
     * @return std::future<R> Future object to retrieve the result.
     *
     * The task is wrapped in a packaged_task, pushed to the internal queue,
     * and one waiting worker is notified. This method is thread-safe and
     * returns a future for result retrieval.
     */
    template<class F, class R = std::invoke_result_t<std::decay_t<F>>>
    std::future<R> enqueue(F&& f);
    // Added: New method to return the number of worker threads
    // Reason: To support chunk size calculation in Engine.cpp for efficient task distribution in parallel strategies, improving scalability for large N.
    /**
     * @brief Returns the number of worker threads in the pool.
     *
     * @return size_t The number of threads.
     */
    size_t thread_count() const { return workers.size(); }
    // Added: New method to wait for task completion
    // Reason: To enable explicit synchronization in Engine.cpp (e.g., in CollaborativeParallelStrategy), allowing iterative parallel loops to wait for chunked tasks to finish before aggregating results.
    /**
     * @brief Waits for all enqueued tasks to complete.
     *
     * Blocks the calling thread until all currently enqueued tasks are finished.
     * Uses the active_tasks counter to track task completion.
     */
    void wait();
private:
    std::vector<std::thread> workers; ///< Worker thread container.
    std::queue<std::function<void()>> tasks; ///< Task queue (FIFO).
    std::mutex mtx; ///< Protects queue access.
    std::condition_variable cv; ///< Worker wakeup notification.
    std::atomic<bool> stop{false}; ///< Termination flag.
    // Added: New atomic counter for active tasks
    // Reason: To track enqueued tasks for the wait method, ensuring accurate synchronization in iterative parallel strategies.
    std::atomic<size_t> active_tasks{0}; ///< Number of active tasks.
};
/**
 * @brief Worker thread loop: waits for tasks and executes them.
 *
 * Each worker runs this lambda indefinitely until `stop` is set.
 * Uses condition variable to block efficiently when idle.
 * Decrements active_tasks when a task is completed.
 */
inline ThreadPool::ThreadPool(size_t threads) {
    if (threads == 0) threads = 1; // Ensure at least one thread.
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
                // Added: Decrement active_tasks after task execution
                // Reason: To track task completion for the wait method, enabling synchronization in parallel strategies.
                --active_tasks;
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
inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    for (auto& w : workers) w.join();
}
/**
 * @brief Enqueue implementation with perfect forwarding and future return.
 *
 * Locks the mutex, wraps the task in a packaged_task, emplaces it,
 * and notifies one waiting worker. Returns a future for the result.
 */
template<class F, class R>
inline std::future<R> ThreadPool::enqueue(F&& f) {
    // Modified: Wrapped task in packaged_task with shared_ptr for future support
    // Reason: To enable result retrieval via std::future<R>, supporting chunk-based parallelism in Engine.cpp for scalability.
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    std::future<R> future = task->get_future();
    {
        std::unique_lock<std::mutex> lock(mtx);
        tasks.emplace([task = std::move(task)]() { (*task)(); });
    }
    // Added: Increment active_tasks for each enqueued task
    // Reason: To track tasks for the wait method, ensuring proper synchronization in iterative loops.
    ++active_tasks;
    cv.notify_one();
    return future;
}
/**
 * @brief Wait implementation: blocks until all active tasks are completed.
 *
 * Uses the active_tasks counter to determine when all enqueued tasks
 * have been executed by the worker threads.
 */
inline void ThreadPool::wait() {
    while (active_tasks.load() > 0) {
        std::this_thread::yield();
    }
}
#endif // THREAD_POOL_HPP