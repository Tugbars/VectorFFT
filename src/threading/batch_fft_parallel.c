/*
 * Batch FFT Parallelization Strategy
 * 
 * Purpose: Parallelize computation of multiple independent FFTs
 * Example: Computing 1000 FFTs of size 256 using 4 threads
 * 
 * Strategy: Pure data parallelism - each thread computes complete FFTs
 *           Thread 0: FFTs 0-249
 *           Thread 1: FFTs 250-499
 *           Thread 2: FFTs 500-749
 *           Thread 3: FFTs 750-999
 */

#include "fft_threading.h"

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * Solver configuration
 * This determines which dimension to parallelize across
 */
typedef struct {
    int dimension_to_parallelize;  // Which dimension to split across threads
    const int *preferred_dims;     // Preference order for dimensions
    size_t num_preferred_dims;     // Number of dimension preferences
} BatchFFTSolver;

/**
 * Execution plan for batch FFT
 * Contains all the information needed to execute parallel FFTs
 */
typedef struct {
    FFTPlan **thread_plans;        // Array of plans, one per thread
    int num_threads;               // How many threads to use
    
    size_t input_stride_per_thread;   // Bytes between thread data in input
    size_t output_stride_per_thread;  // Bytes between thread data in output
    
    const BatchFFTSolver *solver;  // Pointer to solver that created this plan
} BatchFFTExecutionPlan;

/**
 * Data package passed to worker threads
 * Each thread receives this structure with pointers to its data slice
 */
typedef struct {
    size_t input_stride;
    size_t output_stride;
    
    double *real_input;        // Pointer to real part of input
    double *imag_input;        // Pointer to imaginary part of input
    double *real_output;       // Pointer to real part of output
    double *imag_output;       // Pointer to imaginary part of output
    
    FFTPlan **thread_plans;    // Array of plans for each thread
} WorkerThreadData;

/* ============================================================================
 * WORKER THREAD FUNCTION
 * ============================================================================ */

/**
 * Function executed by each worker thread
 * 
 * This is called by the threading backend (OpenMP/pthreads) and does the
 * actual FFT computation for this thread's data slice.
 */
static void *execute_fft_worker(ThreadSpawnData *thread_info)
{
    // Extract the data package for all threads
    WorkerThreadData *worker_data = (WorkerThreadData *) thread_info->shared_data;
    
    // Get this thread's ID (0, 1, 2, ...)
    int thread_id = thread_info->thread_id;
    
    // Get this thread's FFT plan
    FFTPlan *my_plan = worker_data->thread_plans[thread_id];
    
    // Calculate pointers to this thread's data slice
    // Each thread works on offset data based on thread_id
    double *my_real_input = worker_data->real_input + 
                            (thread_id * worker_data->input_stride);
    double *my_imag_input = worker_data->imag_input + 
                            (thread_id * worker_data->input_stride);
    double *my_real_output = worker_data->real_output + 
                             (thread_id * worker_data->output_stride);
    double *my_imag_output = worker_data->imag_output + 
                             (thread_id * worker_data->output_stride);
    
    // Execute the FFT plan on this thread's data
    fft_execute_complex(my_plan, 
                        my_real_input, my_imag_input,
                        my_real_output, my_imag_output);
    
    return NULL;
}

/* ============================================================================
 * MAIN EXECUTION FUNCTION
 * ============================================================================ */

/**
 * Main function to execute batch FFT in parallel
 * 
 * This is called by the user and coordinates all threads
 */
static void execute_batch_fft_parallel(const BatchFFTExecutionPlan *plan,
                                       double *real_input,
                                       double *imag_input,
                                       double *real_output,
                                       double *imag_output)
{
    // Package up all the data that threads will need
    WorkerThreadData worker_data;
    worker_data.input_stride = plan->input_stride_per_thread;
    worker_data.output_stride = plan->output_stride_per_thread;
    worker_data.thread_plans = plan->thread_plans;
    worker_data.real_input = real_input;
    worker_data.imag_input = imag_input;
    worker_data.real_output = real_output;
    worker_data.imag_output = imag_output;
    
    // Spawn threads and execute FFTs in parallel
    // This will call execute_fft_worker() on each thread
    spawn_parallel_loop(plan->num_threads,      // How many threads
                        plan->num_threads,      // How many iterations
                        execute_fft_worker,     // Function to call
                        &worker_data);          // Data to pass
}

/* ============================================================================
 * PLAN LIFECYCLE FUNCTIONS
 * ============================================================================ */

/**
 * Wake up plan (prepare for execution)
 * Called before plan is executed
 */
static void wake_batch_plan(BatchFFTExecutionPlan *plan)
{
    for (int i = 0; i < plan->num_threads; i++) {
        fft_plan_wake(plan->thread_plans[i]);
    }
}

/**
 * Destroy plan and free resources
 */
static void destroy_batch_plan(BatchFFTExecutionPlan *plan)
{
    // Destroy each thread's plan
    for (int i = 0; i < plan->num_threads; i++) {
        fft_plan_destroy(plan->thread_plans[i]);
    }
    
    // Free the array of plan pointers
    free(plan->thread_plans);
}

/**
 * Print plan information (for debugging)
 */
static void print_batch_plan(const BatchFFTExecutionPlan *plan)
{
    printf("Batch FFT Plan:\n");
    printf("  Threads: %d\n", plan->num_threads);
    printf("  Dimension parallelized: %d\n", 
           plan->solver->dimension_to_parallelize);
    
    // Print unique thread plans (some threads may share plans)
    for (int i = 0; i < plan->num_threads; i++) {
        // Only print if this plan is different from previous ones
        if (i == 0 || 
            (plan->thread_plans[i] != plan->thread_plans[i-1] &&
             (i <= 1 || plan->thread_plans[i] != plan->thread_plans[i-2]))) {
            printf("  Thread %d plan: %p\n", i, plan->thread_plans[i]);
        }
    }
}

/* ============================================================================
 * DIMENSION SELECTION
 * ============================================================================ */

/**
 * Choose which dimension to parallelize
 * 
 * For a 3D array, we might parallelize across the first dimension,
 * meaning each thread processes different "slices" of the 3D volume.
 */
static int choose_parallel_dimension(const BatchFFTSolver *solver,
                                     const FFTDimensions *dimensions,
                                     bool is_out_of_place,
                                     int *chosen_dim)
{
    // Use helper function to pick best dimension based on preferences
    return pick_dimension(solver->dimension_to_parallelize,
                         solver->preferred_dims,
                         solver->num_preferred_dims,
                         dimensions,
                         is_out_of_place,
                         chosen_dim);
}

/* ============================================================================
 * PLAN APPLICABILITY CHECK
 * ============================================================================ */

/**
 * Check if this solver can handle the given problem
 * 
 * Returns true if:
 * - Multiple threads are available
 * - Problem has multiple FFTs to compute (vector rank > 0)
 * - We can find a good dimension to parallelize
 */
static bool is_solver_applicable(const BatchFFTSolver *solver,
                                 const FFTProblem *problem,
                                 const FFTPlanner *planner,
                                 int *chosen_dim)
{
    // Need at least 2 threads to parallelize
    if (planner->num_threads <= 1)
        return false;
    
    // Need multiple FFTs (vector rank > 0)
    if (!problem->has_multiple_transforms)
        return false;
    
    if (problem->num_vector_dimensions <= 0)
        return false;
    
    // Check if input and output are different arrays
    bool is_out_of_place = (problem->real_input != problem->real_output);
    
    // Try to find a good dimension to parallelize
    if (!choose_parallel_dimension(solver, problem->vector_dimensions,
                                   is_out_of_place, chosen_dim))
        return false;
    
    return true;
}

/* ============================================================================
 * PLAN CREATION
 * ============================================================================ */

/**
 * Create execution plan for batch FFT parallelization
 * 
 * This is the core planning function that:
 * 1. Decides how to split work across threads
 * 2. Creates sub-plans for each thread
 * 3. Calculates memory strides
 */
static BatchFFTExecutionPlan* create_batch_parallel_plan(
    const BatchFFTSolver *solver,
    const FFTProblem *problem,
    FFTPlanner *planner)
{
    // Check if this solver can handle this problem
    int parallel_dimension;
    if (!is_solver_applicable(solver, problem, planner, &parallel_dimension))
        return NULL;
    
    // Get information about the dimension we're parallelizing
    const DimensionInfo *dim = &problem->vector_dimensions->dims[parallel_dimension];
    size_t num_ffts = dim->length;  // Total number of FFTs to compute
    
    /* -----------------------------------------------------------------------
     * WORK DISTRIBUTION CALCULATION
     * 
     * Goal: Divide num_ffts across threads as evenly as possible
     * 
     * Example: 1000 FFTs, 4 threads requested
     *   block_size = ceil(1000/4) = 250
     *   actual_threads = ceil(1000/250) = 4
     *   Thread 0: 250 FFTs
     *   Thread 1: 250 FFTs
     *   Thread 2: 250 FFTs
     *   Thread 3: 250 FFTs
     * 
     * Example: 100 FFTs, 4 threads requested
     *   block_size = ceil(100/4) = 25
     *   actual_threads = ceil(100/25) = 4
     * 
     * Example: 10 FFTs, 4 threads requested
     *   block_size = ceil(10/4) = 3
     *   actual_threads = ceil(10/3) = 4 (but last thread gets only 1 FFT)
     * ----------------------------------------------------------------------- */
    
    size_t ffts_per_thread = (num_ffts + planner->num_threads - 1) / 
                             planner->num_threads;
    
    int actual_threads = (int)((num_ffts + ffts_per_thread - 1) / 
                               ffts_per_thread);
    
    // Save original thread count and adjust for nested parallelism
    int original_thread_count = planner->num_threads;
    planner->num_threads = (planner->num_threads + actual_threads - 1) / 
                           actual_threads;
    
    /* -----------------------------------------------------------------------
     * MEMORY STRIDE CALCULATION
     * 
     * Strides determine how far apart each thread's data is in memory.
     * 
     * Example: 1000 FFTs of size 256, 4 threads
     *   Each thread processes 250 FFTs
     *   If each FFT is 256 doubles, each thread's data is 250*256 = 64000 doubles apart
     * ----------------------------------------------------------------------- */
    
    size_t input_stride = dim->input_stride * ffts_per_thread;
    size_t output_stride = dim->output_stride * ffts_per_thread;
    
    /* -----------------------------------------------------------------------
     * CREATE THREAD PLANS
     * 
     * Each thread needs its own FFT plan that knows how many FFTs it will compute
     * ----------------------------------------------------------------------- */
    
    FFTPlan **thread_plans = (FFTPlan **)malloc(sizeof(FFTPlan*) * actual_threads);
    if (!thread_plans)
        return NULL;
    
    // Initialize all to NULL for safe cleanup
    for (int i = 0; i < actual_threads; i++) {
        thread_plans[i] = NULL;
    }
    
    // Create a copy of the vector dimensions to modify for each thread
    FFTDimensions *thread_vector_dims = copy_fft_dimensions(problem->vector_dimensions);
    if (!thread_vector_dims)
        goto cleanup_fail;
    
    // Create plan for each thread
    for (int i = 0; i < actual_threads; i++) {
        // Calculate how many FFTs this thread will process
        size_t thread_fft_count;
        if (i == actual_threads - 1) {
            // Last thread gets the remainder
            thread_fft_count = num_ffts - (i * ffts_per_thread);
        } else {
            // Other threads get full block
            thread_fft_count = ffts_per_thread;
        }
        
        // Update the dimension info for this thread
        thread_vector_dims->dims[parallel_dimension].length = thread_fft_count;
        
        // Create FFT problem for this thread with offset pointers
        FFTProblem *thread_problem = create_fft_problem(
            problem->fft_size,
            thread_vector_dims,
            problem->real_input + (i * input_stride),
            problem->imag_input + (i * input_stride),
            problem->real_output + (i * output_stride),
            problem->imag_output + (i * output_stride),
            problem->transform_type
        );
        
        if (!thread_problem)
            goto cleanup_fail;
        
        // Create plan for this thread
        thread_plans[i] = fft_create_plan(planner, thread_problem);
        
        if (!thread_plans[i])
            goto cleanup_fail;
    }
    
    // Restore original thread count
    planner->num_threads = original_thread_count;
    
    // Cleanup temporary vector dimensions
    destroy_fft_dimensions(thread_vector_dims);
    
    /* -----------------------------------------------------------------------
     * CREATE FINAL PLAN
     * ----------------------------------------------------------------------- */
    
    BatchFFTExecutionPlan *plan = (BatchFFTExecutionPlan *)malloc(
        sizeof(BatchFFTExecutionPlan));
    
    if (!plan)
        goto cleanup_fail;
    
    plan->thread_plans = thread_plans;
    plan->num_threads = actual_threads;
    plan->input_stride_per_thread = input_stride;
    plan->output_stride_per_thread = output_stride;
    plan->solver = solver;
    
    // Calculate operation count (for performance estimation)
    plan->operation_count = 0;
    plan->planning_cost = 0;
    for (int i = 0; i < actual_threads; i++) {
        plan->operation_count += thread_plans[i]->operation_count;
        plan->planning_cost += thread_plans[i]->planning_cost;
    }
    
    return plan;

cleanup_fail:
    if (thread_plans) {
        for (int i = 0; i < actual_threads; i++) {
            if (thread_plans[i])
                fft_plan_destroy(thread_plans[i]);
        }
        free(thread_plans);
    }
    if (thread_vector_dims)
        destroy_fft_dimensions(thread_vector_dims);
    
    return NULL;
}

/* ============================================================================
 * SOLVER CREATION AND REGISTRATION
 * ============================================================================ */

/**
 * Create a new batch FFT solver
 */
static BatchFFTSolver* create_batch_solver(int dimension_to_parallelize,
                                           const int *preferred_dims,
                                           size_t num_preferred)
{
    BatchFFTSolver *solver = (BatchFFTSolver *)malloc(sizeof(BatchFFTSolver));
    if (!solver)
        return NULL;
    
    solver->dimension_to_parallelize = dimension_to_parallelize;
    solver->preferred_dims = preferred_dims;
    solver->num_preferred_dims = num_preferred;
    
    return solver;
}

/**
 * Register batch FFT solvers with the planner
 * 
 * We register multiple solvers with different dimension preferences.
 * The planner will try all of them and pick the fastest.
 */
void register_batch_fft_solvers(FFTPlanner *planner)
{
    // Prefer dimension 1 first, then last dimension (-1)
    static const int dimension_preferences[] = { 1, -1 };
    
    // Register a solver for each preference
    for (size_t i = 0; i < sizeof(dimension_preferences) / sizeof(int); i++) {
        BatchFFTSolver *solver = create_batch_solver(
            dimension_preferences[i],
            dimension_preferences,
            sizeof(dimension_preferences) / sizeof(int)
        );
        
        if (solver) {
            fft_planner_register_solver(planner, solver);
        }
    }
}

/* ============================================================================
 * USAGE EXAMPLE
 * ============================================================================ */

#ifdef EXAMPLE_USAGE

void example_batch_fft_usage(void)
{
    // Example: Compute 1000 FFTs of size 256 using 4 threads
    
    const int num_ffts = 1000;
    const int fft_size = 256;
    const int num_threads = 4;
    
    // Allocate input/output arrays
    double *real_input = (double*)malloc(sizeof(double) * num_ffts * fft_size);
    double *imag_input = (double*)malloc(sizeof(double) * num_ffts * fft_size);
    double *real_output = (double*)malloc(sizeof(double) * num_ffts * fft_size);
    double *imag_output = (double*)malloc(sizeof(double) * num_ffts * fft_size);
    
    // Initialize input data
    for (int i = 0; i < num_ffts * fft_size; i++) {
        real_input[i] = /* your data */;
        imag_input[i] = /* your data */;
    }
    
    // Create planner
    FFTPlanner *planner = fft_create_planner();
    fft_planner_set_threads(planner, num_threads);
    
    // Register batch FFT solvers
    register_batch_fft_solvers(planner);
    
    // Create problem description
    FFTProblem *problem = create_batch_fft_problem(
        fft_size,      // Size of each FFT
        num_ffts,      // How many FFTs
        real_input, imag_input,
        real_output, imag_output
    );
    
    // Create plan (planner will benchmark different strategies)
    BatchFFTExecutionPlan *plan = fft_create_plan(planner, problem);
    
    // Execute FFTs in parallel
    execute_batch_fft_parallel(plan, 
                               real_input, imag_input,
                               real_output, imag_output);
    
    // Cleanup
    destroy_batch_plan(plan);
    fft_destroy_problem(problem);
    fft_destroy_planner(planner);
    
    free(real_input);
    free(imag_input);
    free(real_output);
    free(imag_output);
}

#endif // EXAMPLE_USAGE