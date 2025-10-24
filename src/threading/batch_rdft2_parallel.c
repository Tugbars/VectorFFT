/*
 * Batch RDFT2 Parallelization
 * 
 * Purpose: Parallelize RDFT2 transforms (2 real arrays ↔ complex)
 * 
 * RDFT2 Transform Types:
 * - R2HC: Two real arrays → One complex array (halfcomplex format)
 * - HC2R: One complex array → Two real arrays
 * 
 * Use case: Multi-channel real signal processing where you have
 *           separate left/right channels or I/Q signals
 */

#include "fft_threading.h"

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * Solver for batch RDFT2 parallelization
 */
typedef struct {
    int dimension_to_parallelize;
    const int *preferred_dimensions;
    size_t num_preferred;
} BatchRDFT2Solver;

/**
 * Execution plan for batch RDFT2
 * Handles transforms between pairs of real arrays and complex arrays
 */
typedef struct {
    FFTPlan **thread_plans;
    int num_threads;
    
    size_t input_stride_per_thread;   // Stride in input arrays
    size_t output_stride_per_thread;  // Stride in output arrays
    
    const BatchRDFT2Solver *solver;
} BatchRDFT2ExecutionPlan;

/**
 * Data passed to worker threads
 * RDFT2 works with:
 * - Input: Two real arrays (r0, r1) OR complex array (cr, ci)
 * - Output: Complex array (cr, ci) OR two real arrays (r0, r1)
 */
typedef struct {
    size_t input_stride;
    size_t output_stride;
    
    // For R2HC (real to halfcomplex):
    //   r0, r1 are inputs (two real channels)
    //   cr, ci are outputs (complex result)
    // For HC2R (halfcomplex to real):
    //   cr, ci are inputs (complex)
    //   r0, r1 are outputs (two real channels)
    double *real_array_0;      // First real array
    double *real_array_1;      // Second real array
    double *complex_real;      // Complex array - real part
    double *complex_imag;      // Complex array - imaginary part
    
    FFTPlan **thread_plans;
} RDFT2WorkerData;

/* ============================================================================
 * WORKER THREAD FUNCTION
 * ============================================================================ */

/**
 * Worker function for RDFT2 transforms
 * Each thread processes its portion of the batch
 */
static void *execute_rdft2_worker(ThreadSpawnData *thread_info)
{
    RDFT2WorkerData *worker_data = (RDFT2WorkerData *) thread_info->shared_data;
    int thread_id = thread_info->thread_id;
    
    // Get this thread's plan
    FFTPlan *my_plan = worker_data->thread_plans[thread_id];
    
    // Calculate this thread's data pointers
    double *my_r0 = worker_data->real_array_0 + 
                    (thread_id * worker_data->input_stride);
    double *my_r1 = worker_data->real_array_1 + 
                    (thread_id * worker_data->input_stride);
    double *my_cr = worker_data->complex_real + 
                    (thread_id * worker_data->output_stride);
    double *my_ci = worker_data->complex_imag + 
                    (thread_id * worker_data->output_stride);
    
    // Execute RDFT2 transform
    // The plan knows whether this is R2HC or HC2R
    fft_execute_rdft2(my_plan, my_r0, my_r1, my_cr, my_ci);
    
    return NULL;
}

/* ============================================================================
 * MAIN EXECUTION FUNCTION
 * ============================================================================ */

/**
 * Execute batch RDFT2 in parallel
 * 
 * Parameters depend on transform direction:
 * - R2HC: r0, r1 are inputs; cr, ci are outputs
 * - HC2R: cr, ci are inputs; r0, r1 are outputs
 */
static void execute_batch_rdft2(const BatchRDFT2ExecutionPlan *plan,
                                double *real_0,
                                double *real_1,
                                double *complex_real,
                                double *complex_imag)
{
    RDFT2WorkerData worker_data;
    worker_data.input_stride = plan->input_stride_per_thread;
    worker_data.output_stride = plan->output_stride_per_thread;
    worker_data.thread_plans = plan->thread_plans;
    worker_data.real_array_0 = real_0;
    worker_data.real_array_1 = real_1;
    worker_data.complex_real = complex_real;
    worker_data.complex_imag = complex_imag;
    
    // Spawn parallel workers
    spawn_parallel_loop(plan->num_threads,
                        plan->num_threads,
                        execute_rdft2_worker,
                        &worker_data);
}

/* ============================================================================
 * PLAN LIFECYCLE
 * ============================================================================ */

static void wake_batch_rdft2_plan(BatchRDFT2ExecutionPlan *plan)
{
    for (int i = 0; i < plan->num_threads; i++) {
        fft_plan_wake(plan->thread_plans[i]);
    }
}

static void destroy_batch_rdft2_plan(BatchRDFT2ExecutionPlan *plan)
{
    for (int i = 0; i < plan->num_threads; i++) {
        fft_plan_destroy(plan->thread_plans[i]);
    }
    free(plan->thread_plans);
}

static void print_batch_rdft2_plan(const BatchRDFT2ExecutionPlan *plan)
{
    printf("Batch RDFT2 (2 real ↔ complex) - Threads: %d, Dimension: %d\n",
           plan->num_threads,
           plan->solver->dimension_to_parallelize);
    
    for (int i = 0; i < plan->num_threads; i++) {
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

static bool is_rdft2_solver_applicable(const BatchRDFT2Solver *solver,
                                       const FFTProblem *problem,
                                       const FFTPlanner *planner,
                                       int *chosen_dim)
{
    // Need multiple threads
    if (planner->num_threads <= 1)
        return false;
    
    // Need vector dimensions
    if (!problem->has_vector_dimensions || problem->num_vector_dimensions <= 0)
        return false;
    
    // Check if transform is out-of-place
    // For RDFT2: check if real_array_0 != complex_real
    bool is_out_of_place = (problem->real_array_0 != problem->complex_real);
    
    // Pick best dimension
    if (!pick_dimension(solver->dimension_to_parallelize,
                       solver->preferred_dimensions,
                       solver->num_preferred,
                       problem->vector_dimensions,
                       is_out_of_place,
                       chosen_dim))
        return false;
    
    // Additional check for RDFT2-specific constraints
    if (is_out_of_place) {
        return true;  // Can always operate out-of-place
    }
    
    // For in-place, check stride compatibility
    return check_rdft2_inplace_strides(problem, *chosen_dim);
}

/* ============================================================================
 * PLAN CREATION
 * ============================================================================ */

static BatchRDFT2ExecutionPlan* create_batch_rdft2_plan(
    const BatchRDFT2Solver *solver,
    const FFTProblem *problem,
    FFTPlanner *planner)
{
    int parallel_dimension;
    if (!is_rdft2_solver_applicable(solver, problem, planner, &parallel_dimension))
        return NULL;
    
    const DimensionInfo *dim = &problem->vector_dimensions->dims[parallel_dimension];
    size_t num_transforms = dim->length;
    
    // Calculate work distribution
    size_t transforms_per_thread = (num_transforms + planner->num_threads - 1) / 
                                   planner->num_threads;
    int actual_threads = (int)((num_transforms + transforms_per_thread - 1) / 
                               transforms_per_thread);
    
    // Thread count management
    int original_threads = planner->num_threads;
    planner->num_threads = (planner->num_threads + actual_threads - 1) / 
                           actual_threads;
    
    // Calculate strides using RDFT2-specific stride calculation
    size_t input_stride, output_stride;
    calculate_rdft2_strides(problem->transform_type, dim, 
                           &input_stride, &output_stride);
    input_stride *= transforms_per_thread;
    output_stride *= transforms_per_thread;
    
    // Allocate thread plans
    FFTPlan **thread_plans = (FFTPlan **)malloc(sizeof(FFTPlan*) * actual_threads);
    if (!thread_plans)
        return NULL;
    
    for (int i = 0; i < actual_threads; i++)
        thread_plans[i] = NULL;
    
    // Copy vector dimensions for modification
    FFTDimensions *thread_vector_dims = copy_fft_dimensions(problem->vector_dimensions);
    if (!thread_vector_dims)
        goto cleanup_fail;
    
    // Create plan for each thread
    for (int i = 0; i < actual_threads; i++) {
        // Calculate workload for this thread
        size_t thread_transform_count = (i == actual_threads - 1) ?
                                        (num_transforms - i * transforms_per_thread) :
                                        transforms_per_thread;
        
        // Update dimension
        thread_vector_dims->dims[parallel_dimension].length = thread_transform_count;
        
        // Create problem with offset pointers
        FFTProblem *thread_problem = create_rdft2_problem(
            problem->fft_size,
            thread_vector_dims,
            problem->real_array_0 + (i * input_stride),
            problem->real_array_1 + (i * input_stride),
            problem->complex_real + (i * output_stride),
            problem->complex_imag + (i * output_stride),
            problem->transform_type
        );
        
        if (!thread_problem)
            goto cleanup_fail;
        
        thread_plans[i] = fft_create_plan(planner, thread_problem);
        
        if (!thread_plans[i])
            goto cleanup_fail;
    }
    
    // Restore thread count
    planner->num_threads = original_threads;
    destroy_fft_dimensions(thread_vector_dims);
    
    // Create final plan
    BatchRDFT2ExecutionPlan *plan = (BatchRDFT2ExecutionPlan *)malloc(
        sizeof(BatchRDFT2ExecutionPlan));
    
    if (!plan)
        goto cleanup_fail;
    
    plan->thread_plans = thread_plans;
    plan->num_threads = actual_threads;
    plan->input_stride_per_thread = input_stride;
    plan->output_stride_per_thread = output_stride;
    plan->solver = solver;
    
    // Accumulate costs
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

static BatchRDFT2Solver* create_rdft2_solver(int dimension,
                                             const int *prefs,
                                             size_t num_prefs)
{
    BatchRDFT2Solver *solver = (BatchRDFT2Solver *)malloc(
        sizeof(BatchRDFT2Solver));
    if (!solver)
        return NULL;
    
    solver->dimension_to_parallelize = dimension;
    solver->preferred_dimensions = prefs;
    solver->num_preferred = num_prefs;
    
    return solver;
}

void register_batch_rdft2_solvers(FFTPlanner *planner)
{
    // Try dimension 1, then -1
    static const int dimension_preferences[] = { 1, -1 };
    
    for (size_t i = 0; i < sizeof(dimension_preferences) / sizeof(int); i++) {
        BatchRDFT2Solver *solver = create_rdft2_solver(
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
 * USAGE EXAMPLES
 * ============================================================================ */

#ifdef EXAMPLE_USAGE

void example_stereo_to_spectrum(void)
{
    // Example: Convert 1000 stereo audio frames to spectrum
    // Each frame has left and right channels (2 real arrays)
    // Output is complex spectrum
    
    const int num_frames = 1000;
    const int frame_size = 1024;
    const int num_threads = 4;
    
    // Allocate stereo input (left and right channels)
    double *left_channel = (double*)malloc(sizeof(double) * num_frames * frame_size);
    double *right_channel = (double*)malloc(sizeof(double) * num_frames * frame_size);
    
    // Allocate complex spectrum output
    double *spectrum_real = (double*)malloc(sizeof(double) * num_frames * frame_size);
    double *spectrum_imag = (double*)malloc(sizeof(double) * num_frames * frame_size);
    
    // Initialize with stereo audio data
    for (int i = 0; i < num_frames * frame_size; i++) {
        left_channel[i] = /* left audio sample */;
        right_channel[i] = /* right audio sample */;
    }
    
    // Setup
    FFTPlanner *planner = fft_create_planner();
    fft_planner_set_threads(planner, num_threads);
    register_batch_rdft2_solvers(planner);
    
    // Create R2HC (Real to HalfComplex) problem
    FFTProblem *problem = create_rdft2_r2hc_problem(
        frame_size,
        num_frames,
        left_channel, right_channel,    // Input: two real arrays
        spectrum_real, spectrum_imag    // Output: complex
    );
    
    // Execute
    BatchRDFT2ExecutionPlan *plan = fft_create_plan(planner, problem);
    execute_batch_rdft2(plan, 
                       left_channel, right_channel,
                       spectrum_real, spectrum_imag);
    
    // Cleanup
    destroy_batch_rdft2_plan(plan);
    fft_destroy_problem(problem);
    fft_destroy_planner(planner);
    free(left_channel);
    free(right_channel);
    free(spectrum_real);
    free(spectrum_imag);
}

void example_spectrum_to_stereo(void)
{
    // Example: Convert 1000 complex spectra back to stereo audio
    // Input is complex spectrum
    // Output is left/right channels
    
    const int num_frames = 1000;
    const int frame_size = 1024;
    const int num_threads = 4;
    
    // Input: complex spectrum
    double *spectrum_real = (double*)malloc(sizeof(double) * num_frames * frame_size);
    double *spectrum_imag = (double*)malloc(sizeof(double) * num_frames * frame_size);
    
    // Output: stereo audio
    double *left_channel = (double*)malloc(sizeof(double) * num_frames * frame_size);
    double *right_channel = (double*)malloc(sizeof(double) * num_frames * frame_size);
    
    // Setup
    FFTPlanner *planner = fft_create_planner();
    fft_planner_set_threads(planner, num_threads);
    register_batch_rdft2_solvers(planner);
    
    // Create HC2R (HalfComplex to Real) problem
    FFTProblem *problem = create_rdft2_hc2r_problem(
        frame_size,
        num_frames,
        spectrum_real, spectrum_imag,   // Input: complex
        left_channel, right_channel     // Output: two real arrays
    );
    
    // Execute
    BatchRDFT2ExecutionPlan *plan = fft_create_plan(planner, problem);
    execute_batch_rdft2(plan,
                       left_channel, right_channel,
                       spectrum_real, spectrum_imag);
    
    // Cleanup
    destroy_batch_rdft2_plan(plan);
    fft_destroy_problem(problem);
    fft_destroy_planner(planner);
    free(spectrum_real);
    free(spectrum_imag);
    free(left_channel);
    free(right_channel);
}

#endif // EXAMPLE_USAGE