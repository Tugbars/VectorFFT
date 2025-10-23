/*
 * Cooley-Tukey Threaded FFT Parallelization
 * 
 * Purpose: Parallelize a SINGLE large FFT using Cooley-Tukey decomposition
 * 
 * Strategy: Algorithm parallelism (not data parallelism)
 * 
 * Example: One FFT of size 1024 = 4 × 256
 *   - Factor into radix-4 × 256-point FFTs
 *   - Compute 256-point FFTs in parallel (4 threads)
 *   - Then apply twiddle factors in parallel
 * 
 * This is DIFFERENT from batch parallelization:
 *   Batch: 1000 small FFTs → each thread does complete FFTs
 *   Cooley-Tukey: 1 large FFT → threads cooperate on stages
 * 
 * Cooley-Tukey Decomposition:
 *   FFT(n) = FFT(r) × FFT(m) where n = r × m
 *   
 *   DIT (Decimation in Time):
 *     1. Compute m-point FFTs (parallelized)
 *     2. Apply twiddle factors (parallelized)
 *     3. Compute r-point FFTs
 *   
 *   DIF (Decimation in Frequency):
 *     1. Compute r-point FFTs
 *     2. Apply twiddle factors (parallelized)
 *     3. Compute m-point FFTs (parallelized)
 */

#include "fft_threading.h"

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * Cooley-Tukey threaded execution plan
 * 
 * This plan decomposes an FFT into stages:
 * - One stage of r-point FFTs (serial or parallel depending on DIT/DIF)
 * - One stage of m-point FFTs with twiddle factors (parallelized)
 */
typedef struct {
    FFTPlan *main_fft_plan;         // Plan for the r-point or m-point FFT stage
    FFTPlan **worker_plans;         // Plans for parallel twiddle+FFT stage
    int num_threads;                // Number of threads
    int radix;                      // Radix of decomposition (r)
} CooleyTukeyThreadedPlan;

/**
 * Data passed to worker threads
 * 
 * For Cooley-Tukey, workers apply twiddle factors and small FFTs
 */
typedef struct {
    FFTPlan **worker_plans;         // Array of worker plans
    double *real_data;              // Real part of data
    double *imag_data;              // Imaginary part of data
} CooleyTukeyWorkerData;

/* ============================================================================
 * WORKER THREAD FUNCTION
 * ============================================================================ */

/**
 * Worker function for Cooley-Tukey parallel stage
 * 
 * Each thread applies twiddle factors and computes FFTs for its portion
 * of the decomposition.
 */
static void *execute_cooley_tukey_worker(ThreadSpawnData *thread_info)
{
    CooleyTukeyWorkerData *worker_data = 
        (CooleyTukeyWorkerData *) thread_info->shared_data;
    
    int thread_id = thread_info->thread_id;
    
    // Get this thread's worker plan
    // Worker plans know how to apply twiddle factors + small FFTs
    FFTPlan *my_worker_plan = worker_data->worker_plans[thread_id];
    
    // Execute the worker plan on the shared data
    // Worker plans operate on the same data but different parts
    fft_execute_worker_plan(my_worker_plan,
                            worker_data->real_data,
                            worker_data->imag_data);
    
    return NULL;
}

/* ============================================================================
 * EXECUTION FUNCTIONS - DIT and DIF
 * ============================================================================ */

/**
 * Execute Cooley-Tukey FFT using DIT (Decimation in Time) decomposition
 * 
 * DIT Order:
 *   1. Compute m-point FFTs on time-decimated data (main plan, serial)
 *   2. Apply twiddle factors + r-point FFTs in parallel (worker plans)
 * 
 * Example: FFT(1024) = FFT(4) × FFT(256)
 *   1. Compute 4 FFTs of size 256 (serial, could be another CT step)
 *   2. Apply twiddles and 256 FFTs of size 4 in parallel
 */
static void execute_cooley_tukey_dit(const CooleyTukeyThreadedPlan *plan,
                                     double *real_input,
                                     double *imag_input,
                                     double *real_output,
                                     double *imag_output)
{
    // Step 1: Execute main FFT stage (m-point FFTs)
    // This reorganizes data in time-decimated order
    FFTPlan *main_plan = plan->main_fft_plan;
    fft_execute_complex(main_plan,
                        real_input, imag_input,
                        real_output, imag_output);
    
    // Step 2: Execute parallel twiddle + FFT stage
    // Workers apply twiddle factors and compute r-point FFTs
    CooleyTukeyWorkerData worker_data;
    worker_data.real_data = real_output;
    worker_data.imag_data = imag_output;
    worker_data.worker_plans = plan->worker_plans;
    
    spawn_parallel_loop(plan->num_threads,
                        plan->num_threads,
                        execute_cooley_tukey_worker,
                        &worker_data);
}

/**
 * Execute Cooley-Tukey FFT using DIF (Decimation in Frequency) decomposition
 * 
 * DIF Order:
 *   1. Apply twiddle factors + m-point FFTs in parallel (worker plans)
 *   2. Compute r-point FFTs on frequency-decimated data (main plan, serial)
 * 
 * Example: FFT(1024) = FFT(256) × FFT(4)
 *   1. Apply twiddles and 256 FFTs of size 4 in parallel
 *   2. Compute 4 FFTs of size 256 (serial, could be another CT step)
 */
static void execute_cooley_tukey_dif(const CooleyTukeyThreadedPlan *plan,
                                     double *real_input,
                                     double *imag_input,
                                     double *real_output,
                                     double *imag_output)
{
    // Step 1: Execute parallel twiddle + FFT stage
    // Workers apply twiddle factors and compute m-point FFTs
    CooleyTukeyWorkerData worker_data;
    worker_data.real_data = real_input;
    worker_data.imag_data = imag_input;
    worker_data.worker_plans = plan->worker_plans;
    
    spawn_parallel_loop(plan->num_threads,
                        plan->num_threads,
                        execute_cooley_tukey_worker,
                        &worker_data);
    
    // Step 2: Execute main FFT stage (r-point FFTs)
    // This reorganizes data in frequency-decimated order
    FFTPlan *main_plan = plan->main_fft_plan;
    fft_execute_complex(main_plan,
                        real_input, imag_input,
                        real_output, imag_output);
}

/* ============================================================================
 * PLAN LIFECYCLE
 * ============================================================================ */

/**
 * Wake up Cooley-Tukey plan before execution
 */
static void wake_cooley_tukey_plan(CooleyTukeyThreadedPlan *plan)
{
    // Wake main plan
    fft_plan_wake(plan->main_fft_plan);
    
    // Wake all worker plans
    for (int i = 0; i < plan->num_threads; i++) {
        fft_plan_wake(plan->worker_plans[i]);
    }
}

/**
 * Destroy Cooley-Tukey plan and free resources
 */
static void destroy_cooley_tukey_plan(CooleyTukeyThreadedPlan *plan)
{
    // Destroy main plan
    fft_plan_destroy(plan->main_fft_plan);
    
    // Destroy worker plans
    for (int i = 0; i < plan->num_threads; i++) {
        fft_plan_destroy(plan->worker_plans[i]);
    }
    
    free(plan->worker_plans);
}

/**
 * Print Cooley-Tukey plan information
 */
static void print_cooley_tukey_plan(const CooleyTukeyThreadedPlan *plan,
                                    bool is_dit)
{
    printf("Cooley-Tukey Threaded FFT (%s):\n", is_dit ? "DIT" : "DIF");
    printf("  Threads: %d\n", plan->num_threads);
    printf("  Radix: %d\n", plan->radix);
    printf("  Main plan: %p\n", (void*)plan->main_fft_plan);
    
    // Print unique worker plans
    for (int i = 0; i < plan->num_threads; i++) {
        if (i == 0 ||
            (plan->worker_plans[i] != plan->worker_plans[i-1] &&
             (i <= 1 || plan->worker_plans[i] != plan->worker_plans[i-2]))) {
            printf("  Worker plan %d: %p\n", i, (void*)plan->worker_plans[i]);
        }
    }
}

/* ============================================================================
 * PLAN CREATION
 * ============================================================================ */

/**
 * Create Cooley-Tukey threaded plan
 * 
 * This is complex because we need to:
 * 1. Choose radix r such that n = r × m
 * 2. Decide on DIT vs DIF decomposition
 * 3. Create main plan for one stage
 * 4. Create worker plans for parallel stage
 * 5. Distribute work across threads
 * 
 * Parameters:
 *   @param solver          Cooley-Tukey solver configuration
 *   @param problem         FFT problem to solve
 *   @param planner         FFT planner
 *   @param decomposition   DIT or DIF
 * 
 * Returns: Execution plan or NULL on failure
 */
static CooleyTukeyThreadedPlan* create_cooley_tukey_plan(
    const CTSolver *solver,
    const FFTProblem *problem,
    FFTPlanner *planner,
    DecompositionType decomposition)
{
    // Need multiple threads
    if (planner->num_threads <= 1)
        return NULL;
    
    // Check if Cooley-Tukey decomposition is applicable
    if (!is_cooley_tukey_applicable(solver, problem, planner))
        return NULL;
    
    /* -----------------------------------------------------------------------
     * RADIX SELECTION AND FACTORIZATION
     * 
     * For FFT of size n, choose radix r such that:
     *   n = r × m
     * 
     * Example: n = 1024
     *   radix = 4 → m = 256
     *   radix = 8 → m = 128
     * 
     * The choice affects:
     * - How much parallelism is available
     * - Memory access patterns
     * - Twiddle factor complexity
     * ----------------------------------------------------------------------- */
    
    const DimensionInfo *dim = &problem->fft_size->dims[0];
    size_t fft_length = dim->length;  // n
    
    // Choose radix based on solver preferences and FFT size
    int radix = choose_radix(solver->preferred_radix, fft_length);
    size_t smaller_fft_size = fft_length / radix;  // m = n/r
    
    /* -----------------------------------------------------------------------
     * EXTRACT VECTOR DIMENSIONS
     * 
     * Vector dimensions describe any batch processing or multi-dimensional
     * structure in the problem.
     * ----------------------------------------------------------------------- */
    
    size_t vector_length, input_vector_stride, output_vector_stride;
    extract_vector_info(problem->vector_dimensions,
                       &vector_length,
                       &input_vector_stride,
                       &output_vector_stride);
    
    /* -----------------------------------------------------------------------
     * WORK DISTRIBUTION
     * 
     * Distribute the m smaller FFTs across threads.
     * 
     * Example: m = 256, threads = 4
     *   block_size = ceil(256/4) = 64
     *   Thread 0: FFTs 0-63
     *   Thread 1: FFTs 64-127
     *   Thread 2: FFTs 128-191
     *   Thread 3: FFTs 192-255
     * ----------------------------------------------------------------------- */
    
    size_t block_size = (smaller_fft_size + planner->num_threads - 1) / 
                        planner->num_threads;
    int actual_threads = (int)((smaller_fft_size + block_size - 1) / 
                               block_size);
    
    // Save and adjust thread count for nested planning
    int saved_thread_count = planner->num_threads;
    planner->num_threads = (planner->num_threads + actual_threads - 1) / 
                           actual_threads;
    
    /* -----------------------------------------------------------------------
     * CREATE WORKER PLANS
     * 
     * Worker plans apply twiddle factors and compute small FFTs.
     * Each worker handles a block of the decomposed FFT.
     * ----------------------------------------------------------------------- */
    
    FFTPlan **worker_plans = (FFTPlan **)malloc(sizeof(FFTPlan*) * actual_threads);
    if (!worker_plans)
        return NULL;
    
    for (int i = 0; i < actual_threads; i++)
        worker_plans[i] = NULL;
    
    // Create plans based on decomposition type
    switch (decomposition) {
        case DECOMPOSITION_DIT:
        {
            // DIT: Workers do twiddle + r-point FFTs after main m-point FFTs
            for (int i = 0; i < actual_threads; i++) {
                size_t start_index = i * block_size;
                size_t this_block_size = (i == actual_threads - 1) ?
                                        (smaller_fft_size - start_index) :
                                        block_size;
                
                // Create worker plan for this thread's portion
                worker_plans[i] = create_dit_worker_plan(
                    solver,
                    radix,                          // r
                    smaller_fft_size * dim->output_stride,  // Total output stride
                    smaller_fft_size * dim->output_stride,  // Worker output stride
                    smaller_fft_size,               // m
                    dim->output_stride,             // Output stride within
                    vector_length,                  // Vector dimensions
                    output_vector_stride,
                    output_vector_stride,
                    start_index,                    // Starting position
                    this_block_size,                // How many to process
                    problem->real_output,
                    problem->imag_output,
                    planner
                );
                
                if (!worker_plans[i])
                    goto cleanup_fail;
            }
            
            // Restore thread count
            planner->num_threads = saved_thread_count;
            
            /* Create main plan for m-point FFTs
             * Input: time-decimated by radix r
             * Output: ready for twiddle + r-point FFTs */
            FFTPlan *main_plan = create_dit_main_plan(
                planner,
                smaller_fft_size,
                radix,
                dim,
                vector_length,
                input_vector_stride,
                output_vector_stride,
                problem
            );
            
            if (!main_plan)
                goto cleanup_fail;
            
            // Create final plan structure
            CooleyTukeyThreadedPlan *plan = 
                (CooleyTukeyThreadedPlan *)malloc(sizeof(CooleyTukeyThreadedPlan));
            if (!plan) {
                fft_plan_destroy(main_plan);
                goto cleanup_fail;
            }
            
            plan->main_fft_plan = main_plan;
            plan->worker_plans = worker_plans;
            plan->num_threads = actual_threads;
            plan->radix = radix;
            
            // Set execution function
            plan->execute = execute_cooley_tukey_dit;
            
            // Calculate operation counts
            calculate_operation_counts(plan);
            
            return plan;
        }
        
        case DECOMPOSITION_DIF:
        case DECOMPOSITION_DIF_TRANSPOSE:
        {
            // DIF: Workers do twiddle + m-point FFTs before main r-point FFTs
            
            size_t worker_output_stride_real, worker_output_stride_vector;
            
            if (decomposition == DECOMPOSITION_DIF_TRANSPOSE) {
                // Transpose optimization: change data layout
                worker_output_stride_real = input_vector_stride;
                worker_output_stride_vector = smaller_fft_size * dim->input_stride;
                
                // Validate transpose constraints
                if (!validate_transpose_constraints(
                        radix, vector_length,
                        dim, input_vector_stride, output_vector_stride,
                        problem))
                    goto cleanup_fail;
            } else {
                // Standard DIF
                worker_output_stride_real = smaller_fft_size * dim->input_stride;
                worker_output_stride_vector = input_vector_stride;
            }
            
            // Create worker plans
            for (int i = 0; i < actual_threads; i++) {
                size_t start_index = i * block_size;
                size_t this_block_size = (i == actual_threads - 1) ?
                                        (smaller_fft_size - start_index) :
                                        block_size;
                
                worker_plans[i] = create_dif_worker_plan(
                    solver,
                    radix,
                    smaller_fft_size * dim->input_stride,
                    worker_output_stride_real,
                    smaller_fft_size,
                    dim->input_stride,
                    vector_length,
                    input_vector_stride,
                    worker_output_stride_vector,
                    start_index,
                    this_block_size,
                    problem->real_input,
                    problem->imag_input,
                    planner
                );
                
                if (!worker_plans[i])
                    goto cleanup_fail;
            }
            
            // Restore thread count
            planner->num_threads = saved_thread_count;
            
            // Create main plan
            FFTPlan *main_plan = create_dif_main_plan(
                planner,
                smaller_fft_size,
                radix,
                dim,
                vector_length,
                worker_output_stride_real,
                output_vector_stride,
                problem
            );
            
            if (!main_plan)
                goto cleanup_fail;
            
            // Create final plan
            CooleyTukeyThreadedPlan *plan = 
                (CooleyTukeyThreadedPlan *)malloc(sizeof(CooleyTukeyThreadedPlan));
            if (!plan) {
                fft_plan_destroy(main_plan);
                goto cleanup_fail;
            }
            
            plan->main_fft_plan = main_plan;
            plan->worker_plans = worker_plans;
            plan->num_threads = actual_threads;
            plan->radix = radix;
            plan->execute = execute_cooley_tukey_dif;
            
            calculate_operation_counts(plan);
            
            return plan;
        }
        
        default:
            // Invalid decomposition type
            goto cleanup_fail;
    }

cleanup_fail:
    if (worker_plans) {
        for (int i = 0; i < actual_threads; i++) {
            if (worker_plans[i])
                fft_plan_destroy(worker_plans[i]);
        }
        free(worker_plans);
    }
    
    return NULL;
}

/* ============================================================================
 * SOLVER CREATION
 * ============================================================================ */

/**
 * Create Cooley-Tukey threaded solver
 * 
 * Parameters:
 *   @param solver_size             Size of solver structure
 *   @param preferred_radix         Preferred radix (0 = any)
 *   @param decomposition_type      DIT, DIF, or DIF+TRANSPOSE
 *   @param worker_plan_creator     Function to create worker plans
 *   @param force_vector_recursion  Strategy for vector recursion
 * 
 * Returns: Solver that can create Cooley-Tukey threaded plans
 */
CTSolver *create_cooley_tukey_threaded_solver(
    size_t solver_size,
    int preferred_radix,
    DecompositionType decomposition_type,
    WorkerPlanCreator worker_plan_creator,
    VectorRecursionStrategy force_vector_recursion)
{
    CTSolver *solver = (CTSolver *)malloc(solver_size);
    if (!solver)
        return NULL;
    
    solver->preferred_radix = preferred_radix;
    solver->decomposition_type = decomposition_type;
    solver->create_worker_plan = worker_plan_creator;
    solver->vector_recursion_strategy = force_vector_recursion;
    
    return solver;
}

/* ============================================================================
 * USAGE EXAMPLE
 * ============================================================================ */

#ifdef EXAMPLE_USAGE

void example_cooley_tukey_usage(void)
{
    // Example: Compute one large FFT using Cooley-Tukey with threading
    
    const int fft_size = 4096;  // Large FFT
    const int num_threads = 4;
    
    // Allocate arrays
    double *real_input = (double*)malloc(sizeof(double) * fft_size);
    double *imag_input = (double*)malloc(sizeof(double) * fft_size);
    double *real_output = (double*)malloc(sizeof(double) * fft_size);
    double *imag_output = (double*)malloc(sizeof(double) * fft_size);
    
    // Initialize input
    for (int i = 0; i < fft_size; i++) {
        real_input[i] = /* data */;
        imag_input[i] = /* data */;
    }
    
    // Setup planner
    FFTPlanner *planner = fft_create_planner();
    fft_planner_set_threads(planner, num_threads);
    
    // Register Cooley-Tukey solvers
    register_cooley_tukey_solvers(planner);
    
    // Create problem for single large FFT
    FFTProblem *problem = create_fft_problem(
        fft_size,
        real_input, imag_input,
        real_output, imag_output
    );
    
    // Create plan (planner chooses best decomposition)
    CooleyTukeyThreadedPlan *plan = fft_create_plan(planner, problem);
    
    // Execute
    // Internally: decomposes FFT and uses threads for parallel stages
    plan->execute(plan, real_input, imag_input, real_output, imag_output);
    
    // Cleanup
    destroy_cooley_tukey_plan(plan);
    fft_destroy_problem(problem);
    fft_destroy_planner(planner);
    free(real_input); free(imag_input);
    free(real_output); free(imag_output);
}

#endif // EXAMPLE_USAGE