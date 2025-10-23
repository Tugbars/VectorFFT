/*
 * FFT Threading Interface Header
 * 
 * This header defines the threading abstraction layer for parallel FFT computation.
 * It provides a backend-agnostic interface that works with OpenMP, pthreads, or
 * any custom threading implementation.
 */

#ifndef FFT_THREADING_H
#define FFT_THREADING_H

#include <stddef.h>
#include <stdbool.h>

/* ============================================================================
 * CORE THREADING DATA STRUCTURES
 * ============================================================================ */

/**
 * Information passed to each worker thread
 * 
 * When parallel work is spawned, each thread receives this structure containing:
 * - Its thread identifier
 * - Iteration range to process
 * - Shared data pointer
 */
typedef struct {
    int min_iteration;      // Start of iteration range (inclusive)
    int max_iteration;      // End of iteration range (exclusive)
    int thread_id;          // Thread identifier (0 to num_threads-1)
    void *shared_data;      // Pointer to shared data structure
} ThreadSpawnData;

/**
 * Function pointer type for worker functions
 * 
 * Worker functions are called by each thread to process their assigned work.
 * They receive a ThreadSpawnData structure with thread-specific information.
 * 
 * Return value: Typically NULL (unused in most implementations)
 */
typedef void *(*WorkerFunction)(ThreadSpawnData *thread_info);

/* ============================================================================
 * THREADING BACKEND INTERFACE
 * ============================================================================ */

/**
 * Spawn parallel loop across multiple threads
 * 
 * This is the main threading primitive that distributes work across threads.
 * It blocks until all threads complete their work.
 * 
 * Parameters:
 *   @param total_iterations   Total number of iterations (0 to total_iterations-1)
 *   @param num_threads        Number of threads to use
 *   @param worker_fn          Function to call on each thread
 *   @param shared_data        Data pointer passed to all worker threads
 * 
 * Example:
 *   spawn_parallel_loop(1000, 4, my_worker, &data);
 *   // Spawns 4 threads, each processing ~250 iterations
 */
void spawn_parallel_loop(int total_iterations,
                         int num_threads,
                         WorkerFunction worker_fn,
                         void *shared_data);

/**
 * Initialize threading system
 * 
 * Must be called before any parallel operations.
 * Returns: 0 on success, non-zero on error
 */
int initialize_threading(void);

/**
 * Cleanup threading resources
 * 
 * Should be called when done with all parallel operations.
 * Releases any resources allocated by the threading system.
 */
void cleanup_threading(void);

/* ============================================================================
 * CUSTOM THREADING BACKEND SUPPORT
 * ============================================================================ */

/**
 * Callback function type for custom threading backends
 * 
 * Advanced users can provide their own threading implementation by
 * setting a custom spawn loop callback.
 */
typedef void (*CustomSpawnLoopFunction)(
    WorkerFunction worker_fn,
    ThreadSpawnData *thread_data_array,
    size_t data_size,
    int num_threads,
    void *custom_data
);

/**
 * Global callback for custom threading backend
 * 
 * If set (non-NULL), this callback is used instead of the default
 * threading backend (OpenMP/pthreads).
 */
extern CustomSpawnLoopFunction custom_spawn_loop_callback;

/**
 * Custom data passed to the spawn loop callback
 */
extern void *custom_spawn_loop_data;

/* ============================================================================
 * FFT SOLVER REGISTRATION
 * ============================================================================ */

// Forward declarations for FFT infrastructure types
typedef struct fft_planner FFTPlanner;
typedef struct ct_solver CTSolver;
typedef struct hc2hc_solver HC2HCSolver;

/**
 * Cooley-Tukey solver function pointer types
 */
typedef CTSolver *(*CTInferiorPlanMaker)(/* parameters */);
typedef bool (*CTForceVectorRecursion)(/* parameters */);

typedef HC2HCSolver *(*HC2HCInferiorPlanMaker)(/* parameters */);

/**
 * Register complex DFT batch parallelization solvers
 * 
 * Registers solvers that parallelize batches of complex FFTs across
 * vector dimensions (data parallelism).
 */
void register_complex_dft_batch_solvers(FFTPlanner *planner);

/**
 * Register real DFT batch parallelization solvers
 * 
 * Registers solvers that parallelize batches of real FFTs.
 */
void register_real_dft_batch_solvers(FFTPlanner *planner);

/**
 * Register RDFT2 batch parallelization solvers
 * 
 * Registers solvers for transforms between pairs of real arrays
 * and complex arrays (e.g., stereo audio processing).
 */
void register_rdft2_batch_solvers(FFTPlanner *planner);

/**
 * Create Cooley-Tukey threaded solver for complex DFT
 * 
 * Creates a solver that uses Cooley-Tukey decomposition with threading.
 * This parallelizes the FFT algorithm itself (not just batches).
 * 
 * Parameters:
 *   @param solver_size            Size of solver structure
 *   @param radix                  Preferred radix (2, 3, 4, 5, ...)
 *   @param decomposition_type     DIT, DIF, or DIF+TRANSPOSE
 *   @param make_inferior_plan     Function to create worker plans
 *   @param force_vector_recursion Function to decide on recursion strategy
 * 
 * Returns: Pointer to created solver
 */
CTSolver *create_cooley_tukey_threaded_solver(
    size_t solver_size,
    int radix,
    int decomposition_type,
    CTInferiorPlanMaker make_inferior_plan,
    CTForceVectorRecursion force_vector_recursion
);

/**
 * Create HC2HC threaded solver for real FFTs
 * 
 * Creates a solver for halfcomplex-to-halfcomplex transforms with threading.
 * Used for real FFTs (R2HC and HC2R).
 * 
 * Parameters:
 *   @param solver_size         Size of solver structure
 *   @param radix              Preferred radix
 *   @param make_inferior_plan Function to create worker plans
 * 
 * Returns: Pointer to created solver
 */
HC2HCSolver *create_hc2hc_threaded_solver(
    size_t solver_size,
    int radix,
    HC2HCInferiorPlanMaker make_inferior_plan
);

/* ============================================================================
 * PLANNER CONFIGURATION
 * ============================================================================ */

/**
 * Register all standard threaded solvers with planner
 * 
 * This is a convenience function that registers all available
 * threaded solvers (batch and Cooley-Tukey) with the planner.
 * 
 * Equivalent to calling:
 *   - register_complex_dft_batch_solvers()
 *   - register_real_dft_batch_solvers()
 *   - register_rdft2_batch_solvers()
 *   - Plus any Cooley-Tukey solvers
 */
void register_all_threaded_solvers(FFTPlanner *planner);

/**
 * Register planner hooks for thread-safe planning
 * 
 * Registers mutex locks to make the planner thread-safe.
 * Call this if you plan to create plans from multiple threads.
 */
void register_threadsafe_planner_hooks(void);

/**
 * Unregister threading hooks
 * 
 * Removes thread-safety hooks from planner.
 */
void unregister_threadsafe_planner_hooks(void);

/* ============================================================================
 * DECOMPOSITION TYPE CONSTANTS
 * ============================================================================ */

/**
 * Cooley-Tukey decomposition strategies
 */
enum {
    DECOMPOSITION_DIT = 0,          // Decimation in Time
    DECOMPOSITION_DIF = 1,          // Decimation in Frequency
    DECOMPOSITION_DIF_TRANSPOSE = 2 // DIF with transpose optimization
};

/* ============================================================================
 * COMPATIBILITY MACROS
 * ============================================================================ */

#ifdef FFTW_COMPATIBILITY_MODE
// Map old FFTW names to new names for backward compatibility
#define spawn_data          ThreadSpawnData
#define spawn_function      WorkerFunction
#define X(name)             fftw_##name
#define spawn_loop          spawn_parallel_loop
#define ithreads_init       initialize_threading
#define threads_cleanup     cleanup_threading
#define spawnloop_callback  custom_spawn_loop_callback
#define spawnloop_callback_data custom_spawn_loop_data
#endif

/* ============================================================================
 * USAGE EXAMPLES
 * ============================================================================ */

#ifdef INCLUDE_USAGE_EXAMPLES

/**
 * Example 1: Basic parallel loop
 */
void example_basic_parallel_loop(void)
{
    // Worker function processes one iteration
    void *worker(ThreadSpawnData *info) {
        int my_id = info->thread_id;
        int *array = (int *)info->shared_data;
        
        // Each thread processes its assigned range
        for (int i = info->min_iteration; i < info->max_iteration; i++) {
            array[i] = process(array[i]);
        }
        return NULL;
    }
    
    int data[1000];
    // ... initialize data ...
    
    // Process array in parallel using 4 threads
    initialize_threading();
    spawn_parallel_loop(1000, 4, worker, data);
    cleanup_threading();
}

/**
 * Example 2: Registering solvers
 */
void example_register_solvers(void)
{
    FFTPlanner *planner = create_fft_planner();
    
    // Register all threading solvers
    register_all_threaded_solvers(planner);
    
    // Or register specific types:
    // register_complex_dft_batch_solvers(planner);
    // register_real_dft_batch_solvers(planner);
    
    // Now create plans - planner will automatically use
    // threaded solvers when beneficial
    // ...
}

/**
 * Example 3: Custom threading backend
 */
void example_custom_backend(void)
{
    // Define custom spawn function
    void my_custom_spawn(WorkerFunction worker,
                        ThreadSpawnData *data_array,
                        size_t data_size,
                        int num_threads,
                        void *custom_data) {
        // Your custom threading implementation
        // ...
    }
    
    // Set custom callback
    custom_spawn_loop_callback = my_custom_spawn;
    custom_spawn_loop_data = NULL;
    
    // Now all spawn_parallel_loop() calls use your backend
    spawn_parallel_loop(1000, 4, worker_fn, data);
}

#endif // INCLUDE_USAGE_EXAMPLES

/* ============================================================================
 * IMPLEMENTATION NOTES
 * ============================================================================ */

/*
 * Threading Backend Selection:
 * 
 * The implementation of spawn_parallel_loop() depends on which backend
 * is compiled in:
 * 
 * 1. OpenMP (openmp.c):
 *    - Uses #pragma omp parallel for
 *    - Compiler handles thread creation/management
 *    - Very simple implementation (~60 lines)
 * 
 * 2. pthreads (threads.c):
 *    - Uses pthread_create/pthread_join
 *    - Maintains worker thread pool for efficiency
 *    - More complex but portable (~400 lines)
 * 
 * 3. Custom (via callback):
 *    - User provides their own implementation
 *    - Useful for integration with existing threading systems
 * 
 * All backends provide the same interface, so algorithm code
 * (batch parallelization, Cooley-Tukey) works with any backend.
 */

/*
 * Work Distribution Strategy:
 * 
 * spawn_parallel_loop() divides iterations into blocks:
 * 
 * Example: 1000 iterations, 4 threads
 *   block_size = ceil(1000/4) = 250
 *   Thread 0: iterations [0, 250)
 *   Thread 1: iterations [250, 500)
 *   Thread 2: iterations [500, 750)
 *   Thread 3: iterations [750, 1000)
 * 
 * Example: 10 iterations, 4 threads
 *   block_size = ceil(10/4) = 3
 *   Thread 0: iterations [0, 3)
 *   Thread 1: iterations [3, 6)
 *   Thread 2: iterations [6, 9)
 *   Thread 3: iterations [9, 10)  <- only 1 iteration!
 * 
 * This ensures load balancing while minimizing overhead.
 */

#endif // FFT_THREADING_H