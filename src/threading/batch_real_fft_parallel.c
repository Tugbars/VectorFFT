/*
 * Batch Real FFT Parallelization
 * 
 * Purpose: Parallelize computation of multiple independent REAL FFTs
 * Example: Computing 1000 real FFTs of size 256 using 4 threads
 * 
 * Real FFT: Input is real-valued array, output is real-valued array
 * Common use case: Spectral analysis of real signals
 */

#include "fft_threading.h"

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * Solver for batch real FFT parallelization
 */
typedef struct {
    int dimension_to_parallelize;
    const int *preferred_dimensions;
    size_t num_preferred;
} BatchRealFFTSolver;

/**
 * Execution plan for batch real FFT
 * Real FFTs work with single real-valued arrays (no separate real/imaginary)
 */
typedef struct {
    FFTPlan **thread_plans;
    int num_threads;
    
    size_t input_stride_per_thread;   // Elements between threads in input
    size_t output_stride_per_thread;  // Elements between threads in output
    
    const BatchRealFFTSolver *solver;
} BatchRealFFTExecutionPlan;

/**
 * Data passed to worker threads
 * For real FFTs, we only have real-valued input and output
 */
typedef struct {
    size_t input_stride;
    size_t output_stride;
    
    double *real_input;    // Real-valued input array
    double *real_output;   // Real-valued output array
    
    FFTPlan **thread_plans;
} RealFFTWorkerData;

/* ============================================================================
 * WORKER THREAD FUNCTION
 * ============================================================================ */

/**
 * Worker function executed by each thread
 * Computes real FFTs for this thread's data slice
 */
static void *execute_real_fft_worker(ThreadSpawnData *thread_info)
{
    RealFFTWorkerData *worker_data = (RealFFTWorkerData *) thread_info->shared_data;
    int thread_id = thread_info->thread_id;
    
    // Get this thread's FFT plan
    FFTPlan *my_plan = worker_data->thread_plans[thread_id];
    
    // Calculate pointers to this thread's data
    double *my_input = worker_data->real_input + 
                       (thread_id * worker_data->input_stride);
    double *my_output = worker_data->real_output + 
                        (thread_id * worker_data->output_stride);
    
    // Execute real FFT on this thread's data
    fft_execute_real(my_plan, my_input, my_output);
    
    return NULL;
}

/* ============================================================================
 * MAIN EXECUTION FUNCTION
 * ============================================================================ */

/**
 * Execute batch real FFT in parallel
 */
static void execute_batch_real_fft(const BatchRealFFTExecutionPlan *plan,
                                   double *input,
                                   double *output)
{
    RealFFTWorkerData worker_data;
    worker_data.input_stride = plan->input_stride_per_thread;
    worker_data.output_stride = plan->output_stride_per_thread;
    worker_data.thread_plans = plan->thread_plans;
    worker_data.real_input = input;
    worker_data.real_output = output;
    
    // Spawn parallel workers
    spawn_parallel_loop(plan->num_threads,
                        plan->num_threads,
                        execute_real_fft_worker,
                        &worker_data);
}

/* ============================================================================
 * PLAN LIFECYCLE
 * ============================================================================ */

static void wake_batch_real_plan(BatchRealFFTExecutionPlan *plan)
{
    for (int i = 0; i < plan->num_threads; i++) {
        fft_plan_wake(plan->thread_plans[i]);
    }
}

static void destroy_batch_real_plan(BatchRealFFTExecutionPlan *plan)
{
    for (int i = 0; i < plan->num_threads; i++) {
        fft_plan_destroy(plan->thread_plans[i]);
    }
    free(plan->thread_plans);
}

static void print_batch_real_plan(const BatchRealFFTExecutionPlan *plan)
{
    printf("Batch Real FFT (rdft) - Threads: %d, Dimension: %d\n",
           plan->num_threads,
           plan->solver->dimension_to_parallelize);
    
    for (int i = 0; i < plan->num_threads; i++) {
        // Print unique plans only
        if (i == 0 || 
            (plan->thread_plans[i] != plan->thread_plans[i-1] &&
             (i <= 1 || plan->thread_plans[i] != plan->thread_plans[i-2]))) {
            printf("  Plan %d: %p\n", i, (void*)plan->thread_plans[i]);
        }
    }
}

/* ============================================================================
 * APPLICABILITY CHECK
 * ============================================================================ */

static bool is_real_solver_applicable(const BatchRealFFTSolver *solver,
                                      const FFTProblem *problem,
                                      const FFTPlanner *planner,
                                      int *chosen_dim)
{
    // Need multiple threads
    if (planner->num_threads <= 1)
        return false;
    
    // Need vector rank > 0 (multiple FFTs)
    if (!problem->has_vector_dimensions || problem->num_vector_dimensions <= 0)
        return false;
    
    // Check if out-of-place
    bool is_out_of_place = (problem->real_input != problem->real_output);
    
    // Pick dimension to parallelize
    return pick_dimension(solver->dimension_to_parallelize,
                         solver->preferred_dimensions,
                         solver->num_preferred,
                         problem->vector_dimensions,
                         is_out_of_place,
                         chosen_dim);
}

/* ============================================================================
 * PLAN CREATION
 * ============================================================================ */

static BatchRealFFTExecutionPlan* create_batch_real_fft_plan(
    const BatchRealFFTSolver *solver,
    const FFTProblem *problem,
    FFTPlanner *planner)
{
    int parallel_dimension;
    if (!is_real_solver_applicable(solver, problem, planner, &parallel_dimension))
        return NULL;
    
    const DimensionInfo *dim = &problem->vector_dimensions->dims[parallel_dimension];
    size_t num_ffts = dim->length;
    
    // Calculate work distribution
    size_t ffts_per_thread = (num_ffts + planner->num_threads - 1) / 
                             planner->num_threads;
    int actual_threads = (int)((num_ffts + ffts_per_thread - 1) / 
                               ffts_per_thread);
    
    // Save and adjust thread count for nested parallelism
    int original_threads = planner->num_threads;
    planner->num_threads = (planner->num_threads + actual_threads - 1) / 
                           actual_threads;
    
    // Calculate memory strides
    // For real FFTs: stride is in units of real numbers
    size_t input_stride = dim->input_stride * ffts_per_thread;
    size_t output_stride = dim->output_stride * ffts_per_thread;
    
    // Allocate thread plan array
    FFTPlan **thread_plans = (FFTPlan **)malloc(sizeof(FFTPlan*) * actual_threads);
    if (!thread_plans)
        return NULL;
    
    for (int i = 0; i < actual_threads; i++)
        thread_plans[i] = NULL;
    
    // Create modified vector dimensions for thread plans
    FFTDimensions *thread_vector_dims = copy_fft_dimensions(problem->vector_dimensions);
    if (!thread_vector_dims)
        goto cleanup_fail;
    
    // Create plan for each thread
    for (int i = 0; i < actual_threads; i++) {
        // Calculate this thread's workload
        size_t thread_fft_count = (i == actual_threads - 1) ?
                                  (num_ffts - i * ffts_per_thread) :
                                  ffts_per_thread;
        
        // Update dimension for this thread
        thread_vector_dims->dims[parallel_dimension].length = thread_fft_count;
        
        // Create problem with offset pointers
        FFTProblem *thread_problem = create_real_fft_problem(
            problem->fft_size,
            thread_vector_dims,
            problem->real_input + (i * input_stride),
            problem->real_output + (i * output_stride),
            problem->transform_type
        );
        
        if (!thread_problem)
            goto cleanup_fail;
        
        thread_plans[i] = fft_create_plan(planner, thread_problem);
        
        if (!thread_plans[i])
            goto cleanup_fail;
    }
    
    // Restore original thread count
    planner->num_threads = original_threads;
    destroy_fft_dimensions(thread_vector_dims);
    
    // Create final plan
    BatchRealFFTExecutionPlan *plan = (BatchRealFFTExecutionPlan *)malloc(
        sizeof(BatchRealFFTExecutionPlan));
    
    if (!plan)
        goto cleanup_fail;
    
    plan->thread_plans = thread_plans;
    plan->num_threads = actual_threads;
    plan->input_stride_per_thread = input_stride;
    plan->output_stride_per_thread = output_stride;
    plan->solver = solver;
    
    // Accumulate operation counts
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
 * SOLVER REGISTRATION
 * ============================================================================ */

static BatchRealFFTSolver* create_real_fft_solver(int dimension,
                                                   const int *prefs,
                                                   size_t num_prefs)
{
    BatchRealFFTSolver *solver = (BatchRealFFTSolver *)malloc(
        sizeof(BatchRealFFTSolver));
    if (!solver)
        return NULL;
    
    solver->dimension_to_parallelize = dimension;
    solver->preferred_dimensions = prefs;
    solver->num_preferred = num_prefs;
    
    return solver;
}

void register_batch_real_fft_solvers(FFTPlanner *planner)
{
    // Try dimension 1, then -1 (last dimension)
    static const int dimension_preferences[] = { 1, -1 };
    
    for (size_t i = 0; i < sizeof(dimension_preferences) / sizeof(int); i++) {
        BatchRealFFTSolver *solver = create_real_fft_solver(
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

void example_batch_real_fft(void)
{
    // Example: Compute 1000 real FFTs of size 512
    const int num_signals = 1000;
    const int signal_length = 512;
    const int num_threads = 4;
    
    // Allocate real-valued arrays
    double *input_signals = (double*)malloc(sizeof(double) * num_signals * signal_length);
    double *output_spectra = (double*)malloc(sizeof(double) * num_signals * signal_length);
    
    // Initialize with real data (e.g., audio samples, sensor readings)
    for (int i = 0; i < num_signals * signal_length; i++) {
        input_signals[i] = /* your real-valued data */;
    }
    
    // Setup planner
    FFTPlanner *planner = fft_create_planner();
    fft_planner_set_threads(planner, num_threads);
    register_batch_real_fft_solvers(planner);
    
    // Create problem
    FFTProblem *problem = create_batch_real_fft_problem(
        signal_length,
        num_signals,
        input_signals,
        output_spectra
    );
    
    // Create and execute plan
    BatchRealFFTExecutionPlan *plan = fft_create_plan(planner, problem);
    execute_batch_real_fft(plan, input_signals, output_spectra);
    
    // Cleanup
    destroy_batch_real_plan(plan);
    fft_destroy_problem(problem);
    fft_destroy_planner(planner);
    free(input_signals);
    free(output_spectra);
}

#endif // EXAMPLE_USAGE