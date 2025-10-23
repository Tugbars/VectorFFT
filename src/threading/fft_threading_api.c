/*
 * High-Level Threading API for FFT Library
 *
 * This file provides the user-facing API for multi-threaded FFT operations.
 *
 * Purpose: Initialize threading, configure thread count, cleanup resources
 *
 * User Flow:
 *   1. Call initialize_fft_threading() once at startup
 *   2. Call set_fft_thread_count(N) to use N threads
 *   3. Create and execute FFT plans (they automatically use threading)
 *   4. Call cleanup_fft_threading() at shutdown
 *
 * Example:
 *   initialize_fft_threading();        // Init once
 *   set_fft_thread_count(4);           // Use 4 threads
 *   plan = create_fft_plan(...);       // Plan uses 4 threads
 *   execute_fft(plan, ...);            // Executes in parallel
 *   cleanup_fft_threading();           // Cleanup at end
 */

#include "fft_api.h"
#include "fft_threading.h"

/* ============================================================================
 * GLOBAL STATE
 * ============================================================================ */

/**
 * Track whether threading has been initialized
 *
 * This prevents double-initialization and ensures proper setup order.
 */
static bool threading_initialized = false;

/* ============================================================================
 * SOLVER HOOK MANAGEMENT
 * ============================================================================ */

/**
 * Register threaded solver creation hooks
 *
 * This tells the FFT planner to use threaded versions of solvers
 * instead of single-threaded versions.
 *
 * Hooks registered:
 * - Cooley-Tukey threaded solver hook
 * - HC2HC (halfcomplex) threaded solver hook
 */
static void register_threaded_solver_hooks(void)
{
    // Hook for Cooley-Tukey threaded solver
    // When planner needs a CT solver, it will create threaded version
    fft_cooley_tukey_solver_hook = create_cooley_tukey_threaded_solver;

    // Hook for HC2HC threaded solver
    // Used for real FFT transforms
    fft_hc2hc_solver_hook = create_hc2hc_threaded_solver;
}

/**
 * Unregister threaded solver hooks
 *
 * Restores single-threaded solver creation.
 * Called during cleanup or when disabling threading.
 */
static void unregister_threaded_solver_hooks(void)
{
    // Clear hooks - planner will use single-threaded solvers
    fft_cooley_tukey_solver_hook = NULL;
    fft_hc2hc_solver_hook = NULL;
}

/* ============================================================================
 * INITIALIZATION AND CLEANUP
 * ============================================================================ */

/**
 * Initialize FFT threading system
 *
 * This must be called before any other FFT functions if you want to use
 * multi-threaded FFT operations.
 *
 * What this does:
 * 1. Initializes the threading backend (OpenMP/pthreads)
 * 2. Registers threaded solver hooks with the planner
 * 3. Configures the planner for threaded operation
 * 4. Sets up default thread count
 *
 * Safe to call multiple times (subsequent calls are ignored).
 *
 * Returns:
 *   true (1) on success
 *   false (0) on failure
 *
 * Example:
 *   if (!initialize_fft_threading()) {
 *       fprintf(stderr, "Failed to initialize threading\n");
 *       exit(1);
 *   }
 */
int initialize_fft_threading(void)
{
    // Prevent double initialization
    if (threading_initialized)
    {
        return true; // Already initialized, success
    }

    /* -----------------------------------------------------------------------
     * STEP 1: Initialize threading backend
     *
     * This sets up OpenMP or pthreads infrastructure.
     * For OpenMP, this is a no-op. For pthreads, this creates thread pool.
     * ----------------------------------------------------------------------- */

    if (initialize_threading() != 0)
    {
        // Threading backend initialization failed
        return false;
    }

    /* -----------------------------------------------------------------------
     * STEP 2: Register threaded solver hooks
     *
     * This tells the planner: "When you need a Cooley-Tukey solver or
     * HC2HC solver, create the threaded version instead of single-threaded."
     * ----------------------------------------------------------------------- */

    register_threaded_solver_hooks();

    /* -----------------------------------------------------------------------
     * STEP 3: Get the global planner and configure it
     *
     * IMPORTANT: This should be the first time get_fft_planner() is called,
     * so this is when the planner gets configured for threading.
     * ----------------------------------------------------------------------- */

    FFTPlanner *planner = get_fft_planner();

    // Register all standard threaded solvers with the planner
    register_all_threaded_solvers(planner);

    /* -----------------------------------------------------------------------
     * STEP 4: Mark as initialized
     * ----------------------------------------------------------------------- */

    threading_initialized = true;

    return true; // Success
}

/**
 * Cleanup FFT threading system
 *
 * Call this when you're done with all FFT operations.
 * Frees threading resources and resets to single-threaded mode.
 *
 * This calls the general FFT cleanup, then specifically cleans up
 * threading resources.
 *
 * Safe to call multiple times.
 *
 * Example:
 *   cleanup_fft_threading();  // At program exit
 */
void cleanup_fft_threading(void)
{
    // First, do general FFT cleanup (plans, wisdom, etc.)
    cleanup_fft_library();

    // If threading was initialized, clean it up
    if (threading_initialized)
    {
        // Cleanup threading backend (close thread pool, etc.)
        cleanup_threading();

        // Unregister threaded solver hooks
        unregister_threaded_solver_hooks();

        // Mark as no longer initialized
        threading_initialized = false;
    }
}

/* ============================================================================
 * THREAD COUNT CONFIGURATION
 * ============================================================================ */

/**
 * Set the number of threads to use for FFT operations
 *
 * All subsequent FFT plan creation will use this many threads.
 * Plans already created are not affected.
 *
 * If threading is not initialized, this will automatically initialize it.
 *
 * Thread count is clamped to minimum of 1 (negative values become 1).
 *
 * Parameters:
 *   @param num_threads  Number of threads to use (minimum 1)
 *
 * Examples:
 *   set_fft_thread_count(4);   // Use 4 threads
 *   set_fft_thread_count(8);   // Change to 8 threads
 *   set_fft_thread_count(1);   // Back to single-threaded
 *   set_fft_thread_count(-1);  // Becomes 1 (minimum)
 *
 * Typical usage:
 *   // At startup:
 *   int num_cores = omp_get_num_procs();
 *   set_fft_thread_count(num_cores);
 *
 *   // Or set explicitly:
 *   set_fft_thread_count(4);
 */
void set_fft_thread_count(int num_threads)
{
    /* -----------------------------------------------------------------------
     * AUTO-INITIALIZATION
     *
     * If threading hasn't been initialized yet, do it now.
     * This is convenient for users who just want to set thread count.
     * ----------------------------------------------------------------------- */

    if (!threading_initialized)
    {
        // Clean up any non-threaded FFT state
        cleanup_fft_library();

        // Initialize threading
        initialize_fft_threading();
    }

    // Assert that threading is now initialized
    assert(threading_initialized);

    /* -----------------------------------------------------------------------
     * SET THREAD COUNT IN PLANNER
     *
     * The planner stores the thread count. All new plans will use this value.
     * ----------------------------------------------------------------------- */

    FFTPlanner *planner = get_fft_planner();

    // Clamp to minimum of 1 thread
    planner->num_threads = (num_threads < 1) ? 1 : num_threads;
}

/**
 * Get the current thread count setting
 *
 * Returns the number of threads that will be used for new FFT plans.
 *
 * Returns:
 *   Number of threads configured (minimum 1)
 *
 * Example:
 *   int threads = get_fft_thread_count();
 *   printf("Using %d threads\n", threads);
 */
int get_fft_thread_count(void)
{
    FFTPlanner *planner = get_fft_planner();
    return planner->num_threads;
}

/* ============================================================================
 * THREAD-SAFE PLANNING
 * ============================================================================ */

/**
 * Make the FFT planner thread-safe
 *
 * By default, the planner is not thread-safe. If you need to create
 * FFT plans from multiple threads simultaneously, call this function.
 *
 * This adds mutex locks around plan creation to prevent race conditions.
 *
 * Note: Plan EXECUTION is always thread-safe. This is only needed if
 * you're creating plans from multiple threads at the same time.
 *
 * Example use case:
 *   // In a multi-threaded application where each thread creates plans:
 *
 *   make_fft_planner_thread_safe();  // Call once at startup
 *
 *   // Now safe to do this from multiple threads:
 *   #pragma omp parallel
 *   {
 *       FFTPlan *my_plan = create_fft_plan(...);  // Thread-safe
 *   }
 *
 * Performance note:
 *   Thread-safe planning adds a small overhead. Only use if you actually
 *   need to create plans from multiple threads. Most applications create
 *   plans once at startup, then execute them many times - for these,
 *   thread-safe planning is unnecessary.
 */
void make_fft_planner_thread_safe(void)
{
    register_threadsafe_planner_hooks();
}

/* ============================================================================
 * CUSTOM THREADING BACKEND
 * ============================================================================ */

/**
 * Set a custom threading callback
 *
 * Advanced users can provide their own threading implementation instead
 * of using the default OpenMP or pthreads backend.
 *
 * Use case:
 * - Integration with existing threading system
 * - Custom work distribution
 * - Debugging or profiling
 *
 * Parameters:
 *   @param callback_fn   Function that spawns threads and executes work
 *   @param user_data     Custom data passed to callback
 *
 * The callback function signature:
 *   void callback(WorkerFunction worker,
 *                ThreadSpawnData *thread_data_array,
 *                size_t data_size,
 *                int num_threads,
 *                void *user_data)
 *
 * Example:
 *   void my_threading(WorkerFunction worker,
 *                    ThreadSpawnData *data,
 *                    size_t data_size,
 *                    int num_threads,
 *                    void *user_data) {
 *       // Your threading implementation
 *       for (int i = 0; i < num_threads; i++) {
 *           spawn_my_thread(worker, &data[i]);
 *       }
 *       wait_for_my_threads();
 *   }
 *
 *   // Register it:
 *   set_custom_threading_callback(my_threading, my_context);
 *
 * Note: Set to NULL to restore default threading backend.
 */
void set_custom_threading_callback(
    void (*callback_fn)(void *(*work)(char *), char *, size_t, int, void *),
    void *user_data)
{
    // Store callback function pointer
    // (Cast to correct type - void* for flexibility)
    custom_spawn_loop_callback = (CustomSpawnLoopFunction)callback_fn;

    // Store user data that will be passed to callback
    custom_spawn_loop_data = user_data;
}

/* ============================================================================
 * CONVENIENCE FUNCTIONS
 * ============================================================================ */

/**
 * Check if threading is initialized
 *
 * Returns:
 *   true if threading system is initialized
 *   false otherwise
 */
bool is_fft_threading_initialized(void)
{
    return threading_initialized;
}

/**
 * Get recommended thread count for current system
 *
 * Returns the number of CPU cores available, which is usually
 * a good default for thread count.
 *
 * Returns:
 *   Number of CPU cores detected
 *
 * Example:
 *   int recommended = get_recommended_thread_count();
 *   set_fft_thread_count(recommended);
 */
int get_recommended_thread_count(void)
{
#ifdef _OPENMP
    return omp_get_num_procs();
#else
// Fallback: try to detect CPU count
// This is platform-specific
#if defined(_SC_NPROCESSORS_ONLN)
    return sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    return 4; // Reasonable default
#endif
#endif
}

/**
 * Initialize threading with recommended thread count
 *
 * Convenience function that initializes threading and sets
 * thread count to the number of CPU cores.
 *
 * Returns:
 *   true on success, false on failure
 *
 * Example:
 *   if (!initialize_fft_threading_auto()) {
 *       fprintf(stderr, "Threading init failed\n");
 *   }
 */
bool initialize_fft_threading_auto(void)
{
    if (!initialize_fft_threading())
    {
        return false;
    }

    int recommended = get_recommended_thread_count();
    set_fft_thread_count(recommended);

    return true;
}

/* ============================================================================
 * USAGE EXAMPLES
 * ============================================================================ */

#ifdef EXAMPLE_USAGE

/**
 * Example 1: Basic threading setup
 */
void example_basic_setup(void)
{
    // Initialize threading system
    if (!initialize_fft_threading())
    {
        fprintf(stderr, "Failed to initialize threading\n");
        exit(1);
    }

    // Set to use 4 threads
    set_fft_thread_count(4);

    // Create plans - they will automatically use 4 threads
    FFTPlan *plan = create_fft_plan(1024, ...);

    // Execute - runs in parallel
    execute_fft(plan, ...);

    // Cleanup
    destroy_fft_plan(plan);
    cleanup_fft_threading();
}

/**
 * Example 2: Automatic thread count
 */
void example_auto_threads(void)
{
    // Initialize with system's CPU count
    initialize_fft_threading_auto();

    printf("Using %d threads\n", get_fft_thread_count());

    // Use FFT functions normally...

    cleanup_fft_threading();
}

/**
 * Example 3: Adjusting thread count dynamically
 */
void example_dynamic_threads(void)
{
    initialize_fft_threading();

    // Start with 2 threads for small FFTs
    set_fft_thread_count(2);
    FFTPlan *small_plan = create_fft_plan(256, ...);

    // Switch to 8 threads for large FFTs
    set_fft_thread_count(8);
    FFTPlan *large_plan = create_fft_plan(16384, ...);

    // Execute both plans
    execute_fft(small_plan, ...); // Uses 2 threads (from plan creation)
    execute_fft(large_plan, ...); // Uses 8 threads (from plan creation)

    cleanup_fft_threading();
}

/**
 * Example 4: Thread-safe plan creation
 */
void example_threadsafe_planning(void)
{
    initialize_fft_threading();
    set_fft_thread_count(4);

    // Enable thread-safe planning
    make_fft_planner_thread_safe();

// Now safe to create plans from multiple threads
#pragma omp parallel for
    for (int i = 0; i < 10; i++)
    {
        FFTPlan *plan = create_fft_plan(1024, ...); // Thread-safe!
        // Use plan...
        destroy_fft_plan(plan);
    }

    cleanup_fft_threading();
}

/**
 * Example 5: Custom threading backend
 */
void my_custom_threading(void *(*worker)(char *),
                         char *data,
                         size_t data_size,
                         int num_threads,
                         void *user_data)
{
    printf("Custom backend: spawning %d threads\n", num_threads);
    // Your threading implementation...
}

void example_custom_backend(void)
{
    initialize_fft_threading();

    // Register custom backend
    set_custom_threading_callback(my_custom_threading, NULL);

    // All FFT operations now use your threading backend
    set_fft_thread_count(4);
    FFTPlan *plan = create_fft_plan(1024, ...);
    execute_fft(plan, ...);

    cleanup_fft_threading();
}

#endif // EXAMPLE_USAGE

/* ============================================================================
 * NOTES FOR USERS
 * ============================================================================ */

/*
 * Threading Initialization Order:
 *
 * CORRECT:
 *   initialize_fft_threading();    // First
 *   set_fft_thread_count(4);       // Second
 *   plan = create_fft_plan(...);   // Third
 *
 * ALSO CORRECT (auto-init):
 *   set_fft_thread_count(4);       // Automatically initializes
 *   plan = create_fft_plan(...);   // Uses 4 threads
 *
 * INCORRECT:
 *   plan = create_fft_plan(...);   // No threading set up!
 *   set_fft_thread_count(4);       // Too late, plan already created
 */

/*
 * Thread Count Guidelines:
 *
 * Small FFTs (< 256):
 *   Use 1-2 threads (overhead dominates)
 *
 * Medium FFTs (256-4096):
 *   Use 2-4 threads (good balance)
 *
 * Large FFTs (4096+):
 *   Use 4-8+ threads (scales well)
 *
 * Batch FFTs:
 *   Use as many threads as you have cores
 *   Batch parallelization scales very well
 *
 * General rule:
 *   threads = number of CPU cores
 *   But benchmark to find optimal!
 */

/*
 * Common Pitfalls:
 *
 * 1. Forgetting to initialize:
 *    Solution: Call initialize_fft_threading() at startup
 *
 * 2. Changing thread count after plan creation:
 *    Solution: Plans lock in thread count at creation time
 *
 * 3. Using too many threads for small FFTs:
 *    Solution: More threads != always faster, benchmark!
 *
 * 4. Not cleaning up:
 *    Solution: Call cleanup_fft_threading() at shutdown
 *
 * 5. Creating plans from multiple threads without thread-safety:
 *    Solution: Call make_fft_planner_thread_safe()
 */