/*
 * OpenMP Threading Backend Implementation
 * 
 * This file provides the threading abstraction layer using OpenMP.
 * 
 * Purpose: Implements spawn_parallel_loop() using OpenMP pragmas
 * 
 * Why OpenMP?
 * - Simple: Just add #pragma omp directives
 * - Portable: Works on GCC, Clang, Intel, MSVC
 * - Efficient: Compiler handles thread pool
 * - Automatic: No manual thread creation/destruction
 * 
 * This is one of three possible backends:
 * 1. OpenMP (this file) - Simplest, recommended
 * 2. pthreads - More control, more complex
 * 3. Custom - User provides their own implementation
 */

#include "fft_threading.h"

/* Verify OpenMP is available */
#if !defined(_OPENMP)
#error OpenMP threading enabled but compiler does not support OpenMP. Use -fopenmp flag.
#endif

#include <omp.h>

/* ============================================================================
 * THREADING INITIALIZATION AND CLEANUP
 * ============================================================================ */

/**
 * Initialize OpenMP threading system
 * 
 * For OpenMP, there's nothing to initialize - the runtime handles everything.
 * This function exists for API consistency with other backends (pthreads).
 * 
 * Returns: 0 on success (always succeeds for OpenMP)
 */
int initialize_threading(void)
{
    // OpenMP requires no initialization
    // Thread pool is managed automatically by the runtime
    return 0;  // Success
}

/**
 * Cleanup OpenMP threading system
 * 
 * For OpenMP, there's nothing to clean up.
 * This function exists for API consistency with other backends.
 */
void cleanup_threading(void)
{
    // OpenMP requires no cleanup
    // Thread pool is managed automatically by the runtime
}

/**
 * Register thread-safe planner hooks
 * 
 * For OpenMP, thread-safety is automatically handled by the runtime.
 * This function exists for API consistency with other backends.
 * 
 * Note: If you need to call the FFT planner from within parallel regions,
 * additional synchronization may be needed. For now, this is a placeholder.
 */
void register_threadsafe_planner_hooks(void)
{
    // FIXME: What does "thread-safe planning" mean for OpenMP?
    // OpenMP parallel regions are already thread-safe for data access,
    // but planning is typically done serially before execution anyway.
    
    // For now: do nothing
}

/* ============================================================================
 * MAIN THREADING PRIMITIVE
 * ============================================================================ */

/**
 * Spawn parallel loop using OpenMP
 * 
 * This is the core threading function that distributes work across threads.
 * It blocks until all threads complete.
 * 
 * Work Distribution:
 *   - Divides total_iterations into equal-sized blocks
 *   - Assigns one block per thread
 *   - Adjusts thread count if fewer threads needed
 * 
 * Example: total_iterations=1000, num_threads=4
 *   block_size = ceil(1000/4) = 250
 *   Thread 0: iterations [0, 250)
 *   Thread 1: iterations [250, 500)
 *   Thread 2: iterations [500, 750)
 *   Thread 3: iterations [750, 1000)
 * 
 * Example: total_iterations=10, num_threads=4
 *   block_size = ceil(10/4) = 3
 *   actual_threads = ceil(10/3) = 4
 *   Thread 0: iterations [0, 3)
 *   Thread 1: iterations [3, 6)
 *   Thread 2: iterations [6, 9)
 *   Thread 3: iterations [9, 10)  <- only 1 iteration
 * 
 * Parameters:
 *   @param total_iterations  Total number of iterations (0 to N-1)
 *   @param num_threads       Number of threads to use
 *   @param worker_fn         Function to call on each thread
 *   @param shared_data       Data pointer passed to all workers
 */
void spawn_parallel_loop(int total_iterations,
                         int num_threads,
                         WorkerFunction worker_fn,
                         void *shared_data)
{
    int block_size;
    ThreadSpawnData thread_data;
    int i;
    
    /* -----------------------------------------------------------------------
     * INPUT VALIDATION
     * 
     * Ensure parameters are valid before proceeding.
     * ----------------------------------------------------------------------- */
    
    // Assertions (in production, these would be runtime checks)
    assert(total_iterations >= 0);  // Can't have negative iterations
    assert(num_threads > 0);        // Need at least one thread
    assert(worker_fn != NULL);      // Must have a worker function
    
    // Early exit: nothing to do
    if (total_iterations == 0)
        return;
    
    /* -----------------------------------------------------------------------
     * WORK DISTRIBUTION CALCULATION
     * 
     * Goal: Distribute iterations evenly across threads while minimizing
     *       thread overhead.
     * 
     * Strategy:
     *   1. Calculate block_size = ceil(total_iterations / num_threads)
     *   2. Recalculate actual_threads = ceil(total_iterations / block_size)
     * 
     * Why recalculate threads?
     *   If we have 10 iterations and 100 threads requested, we only need
     *   10 threads. Using 100 would waste resources and add overhead.
     * 
     * Examples:
     *   total=1000, requested=4 → block=250, actual=4
     *   total=100,  requested=4 → block=25,  actual=4
     *   total=10,   requested=4 → block=3,   actual=4
     *   total=5,    requested=8 → block=1,   actual=5  (only need 5!)
     * ----------------------------------------------------------------------- */
    
    block_size = (total_iterations + num_threads - 1) / num_threads;
    num_threads = (total_iterations + block_size - 1) / block_size;
    
    /* -----------------------------------------------------------------------
     * CUSTOM BACKEND SUPPORT
     * 
     * If user provided a custom threading backend, use it instead of OpenMP.
     * This allows integration with existing threading systems.
     * ----------------------------------------------------------------------- */
    
    if (custom_spawn_loop_callback) {
        // User wants to use their own threading implementation
        
        // Allocate array of thread data structures (one per thread)
        ThreadSpawnData *thread_data_array;
        STACK_MALLOC(ThreadSpawnData *, thread_data_array,
                     sizeof(ThreadSpawnData) * num_threads);
        
        // Initialize thread data for each thread
        for (i = 0; i < num_threads; ++i) {
            ThreadSpawnData *data = &thread_data_array[i];
            
            // Calculate this thread's iteration range
            data->min_iteration = i * block_size;
            data->max_iteration = data->min_iteration + block_size;
            
            // Clamp to total_iterations (last thread may have fewer iterations)
            if (data->max_iteration > total_iterations)
                data->max_iteration = total_iterations;
            
            // Set thread identifier and shared data
            data->thread_id = i;
            data->shared_data = shared_data;
        }
        
        // Call user's custom spawn function
        custom_spawn_loop_callback(worker_fn,
                                   thread_data_array,
                                   sizeof(ThreadSpawnData),
                                   num_threads,
                                   custom_spawn_loop_data);
        
        // Free temporary array
        STACK_FREE(thread_data_array);
        return;
    }
    
    /* -----------------------------------------------------------------------
     * OPENMP PARALLEL EXECUTION
     * 
     * Use OpenMP to spawn threads and execute worker function in parallel.
     * 
     * OpenMP pragma explanation:
     *   #pragma omp parallel for
     *     - parallel: Create team of threads
     *     - for: Distribute loop iterations across threads
     *   
     *   private(thread_data)
     *     - Each thread gets its own copy of thread_data variable
     *     - Prevents race conditions on this variable
     * 
     * How it works:
     *   1. OpenMP creates/reuses a thread pool
     *   2. Each thread executes one iteration of the for loop
     *   3. Each iteration calls worker_fn with different thread_data
     *   4. Barrier at end ensures all threads complete
     * ----------------------------------------------------------------------- */
    
#pragma omp parallel for private(thread_data)
    for (i = 0; i < num_threads; ++i) {
        // Calculate this thread's iteration range
        thread_data.min_iteration = i * block_size;
        thread_data.max_iteration = thread_data.min_iteration + block_size;
        
        // Clamp maximum to total_iterations
        if (thread_data.max_iteration > total_iterations)
            thread_data.max_iteration = total_iterations;
        
        // Set thread identifier
        thread_data.thread_id = i;
        
        // Set shared data pointer
        thread_data.shared_data = shared_data;
        
        // Call worker function on this thread
        worker_fn(&thread_data);
    }
    
    // Implicit barrier here - all threads have completed when we return
}

/* ============================================================================
 * GLOBAL CALLBACK VARIABLES
 * ============================================================================ */

/**
 * Custom spawn loop callback (NULL = use OpenMP)
 * 
 * If set, this function is called instead of using OpenMP.
 * Allows users to provide their own threading implementation.
 */
CustomSpawnLoopFunction custom_spawn_loop_callback = NULL;

/**
 * Custom data passed to spawn loop callback
 * 
 * User-defined data that gets passed to the custom callback.
 */
void *custom_spawn_loop_data = NULL;

/* ============================================================================
 * USAGE NOTES
 * ============================================================================ */

/*
 * Compiling with OpenMP:
 * 
 * GCC/Clang:
 *   gcc -fopenmp -O3 your_code.c openmp_backend.c -o fft_program
 * 
 * Intel Compiler:
 *   icc -qopenmp -O3 your_code.c openmp_backend.c -o fft_program
 * 
 * MSVC:
 *   cl /openmp /O2 your_code.c openmp_backend.c /Fe:fft_program.exe
 * 
 * Without -fopenmp flag, you'll get the #error at the top of this file.
 */

/*
 * Performance Tips:
 * 
 * 1. Set number of threads:
 *    export OMP_NUM_THREADS=4
 *    Or: omp_set_num_threads(4);
 * 
 * 2. Thread affinity (bind threads to cores):
 *    export OMP_PROC_BIND=true
 * 
 * 3. Scheduling policy:
 *    We use default (static) scheduling which is best for balanced workloads.
 *    For imbalanced work, you could add schedule(dynamic) to the pragma.
 * 
 * 4. Nested parallelism:
 *    export OMP_NESTED=true
 *    Allows parallel regions within parallel regions (use with caution!)
 */

/*
 * OpenMP vs. pthreads Backend:
 * 
 * OpenMP Advantages:
 *   + Simpler code (~100 lines vs ~400 lines)
 *   + Compiler handles thread pool
 *   + Portable across compilers
 *   + Good performance
 *   + Easy to maintain
 * 
 * OpenMP Disadvantages:
 *   - Requires compiler support (-fopenmp flag)
 *   - Less fine-grained control
 *   - Harder to debug threading issues
 * 
 * pthreads Advantages:
 *   + Available on all POSIX systems
 *   + Fine-grained control
 *   + Better for persistent thread pools
 *   + Easier to debug
 * 
 * pthreads Disadvantages:
 *   - More complex code
 *   - Manual thread management
 *   - Platform-specific (Windows needs wrapper)
 * 
 * Recommendation: Use OpenMP unless you have specific requirements for pthreads.
 */

/*
 * Example: Using this backend
 * 
 * void *my_worker(ThreadSpawnData *info) {
 *     printf("Thread %d processing iterations %d to %d\n",
 *            info->thread_id, info->min_iteration, info->max_iteration);
 *     
 *     // Do work...
 *     
 *     return NULL;
 * }
 * 
 * int main() {
 *     initialize_threading();
 *     
 *     // Process 1000 iterations using 4 threads
 *     spawn_parallel_loop(1000, 4, my_worker, NULL);
 *     
 *     cleanup_threading();
 *     return 0;
 * }
 */

/*
 * Advanced: Custom Backend Example
 * 
 * void my_custom_spawn(WorkerFunction worker,
 *                     ThreadSpawnData *data_array,
 *                     size_t data_size,
 *                     int num_threads,
 *                     void *custom_data) {
 *     // Your threading implementation
 *     for (int i = 0; i < num_threads; i++) {
 *         // Spawn thread, call worker(&data_array[i])
 *     }
 *     // Wait for all threads
 * }
 * 
 * int main() {
 *     // Use custom backend
 *     custom_spawn_loop_callback = my_custom_spawn;
 *     custom_spawn_loop_data = my_context;
 *     
 *     // Now all calls use your backend
 *     spawn_parallel_loop(1000, 4, my_worker, NULL);
 * }
 */