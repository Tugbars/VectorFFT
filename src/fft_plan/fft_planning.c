//==============================================================================
// IMPLEMENTATION
//==============================================================================

#include "fft_twiddles.h"     // Twiddle Manager
#include "fft_rader_plans.h"  // Rader Manager
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
    #include <malloc.h>
    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #define aligned_free(ptr) free(ptr)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif

//==============================================================================
// LOGGING (Configurable)
//==============================================================================

// ✅ Allow user to override logging
#ifndef FFT_LOG_ERROR
#define FFT_LOG_ERROR(fmt, ...) fprintf(stderr, "[FFT ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

#ifndef FFT_LOG_DEBUG
#ifdef FFT_DEBUG_PLANNING
#define FFT_LOG_DEBUG(fmt, ...) fprintf(stderr, "[FFT DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define FFT_LOG_DEBUG(fmt, ...) ((void)0)
#endif
#endif

//==============================================================================
// HELPER: Next power of 2
//==============================================================================

static int next_pow2(int n)
{
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

//==============================================================================
// FACTORIZATION
//==============================================================================

/**
 * @brief Factorize N into radices (FFTW-style: prefer large radices)
 * 
 * Strategy: Try radices in order: 32, 16, 13, 11, 9, 8, 7, 5, 4, 3, 2
 * 
 * ✅ Extended to support more primes (17, 19, 23, etc.) from Rader Manager
 */
static int factorize(int N, int *factors)
{
    int num_factors = 0;
    int n = N;
    
    // ✅ Extended radix priority (largest first for efficiency)
    const int radix_order[] = {32, 16, 13, 11, 9, 8, 7, 5, 4, 3, 2, 
                               17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67};
    const int num_radices = sizeof(radix_order) / sizeof(radix_order[0]);
    
    while (n > 1) {
        int found = 0;
        
        for (int i = 0; i < num_radices; i++) {
            int r = radix_order[i];
            if (n % r == 0) {
                factors[num_factors++] = r;
                n /= r;
                found = 1;
                break;
            }
        }
        
        if (!found) {
            // Can't factorize - will use Bluestein
            return -1;
        }
        
        if (num_factors >= MAX_FFT_STAGES) {
            FFT_LOG_ERROR("Too many factors (>%d) for N=%d", MAX_FFT_STAGES, N);
            return -1;
        }
    }
    
    return num_factors;
}

//==============================================================================
// BLUESTEIN PLANNING
//==============================================================================

/**
 * @brief Plan Bluestein's algorithm for arbitrary N
 * 
 * Strategy: Pad to M = next_pow2(2*N-1), compute chirp twiddles
 */
static int plan_bluestein(fft_plan *plan, int N, fft_direction_t direction)
{
    // ✅ Compute padded size
    int M = next_pow2(2 * N - 1);
    
    FFT_LOG_DEBUG("Using Bluestein: N=%d → M=%d (power-of-2)", N, M);
    
    plan->use_bluestein = 1;
    plan->n_input = N;
    plan->n_fft = M;
    
    // ✅ Allocate chirp twiddles: b[k] = exp(±πi * k^2 / N)
    plan->bluestein_tw = (fft_data*)aligned_alloc(32, M * sizeof(fft_data));
    if (!plan->bluestein_tw) {
        FFT_LOG_ERROR("Failed to allocate Bluestein chirp twiddles");
        return -1;
    }
    
    // ✅ Compute chirp: b[k] = exp(sign * πi * k^2 / N)
    const double sign = (direction == FFT_FORWARD) ? -1.0 : +1.0;
    
    for (int k = 0; k < N; k++) {
        double angle = sign * M_PI * (double)(k * k) / (double)N;
        plan->bluestein_tw[k].re = cos(angle);
        plan->bluestein_tw[k].im = sin(angle);
    }
    
    // Zero-pad the rest
    for (int k = N; k < M; k++) {
        plan->bluestein_tw[k].re = 0.0;
        plan->bluestein_tw[k].im = 0.0;
    }
    
    // ✅ Create internal FFT plans for M (recursive)
    plan->bluestein_plan_fwd = fft_init(M, FFT_FORWARD);
    plan->bluestein_plan_inv = fft_init(M, FFT_INVERSE);
    
    if (!plan->bluestein_plan_fwd || !plan->bluestein_plan_inv) {
        FFT_LOG_ERROR("Failed to create internal Bluestein plans for M=%d", M);
        return -1;
    }
    
    FFT_LOG_DEBUG("Bluestein planning complete: N=%d, M=%d", N, M);
    
    return 0;
}

//==============================================================================
// MAIN PLANNING (THE UMBRELLA)
//==============================================================================

fft_object fft_init(int N, fft_direction_t direction)
{
    //==========================================================================
    // VALIDATION
    //==========================================================================
    if (N <= 0) {
        FFT_LOG_ERROR("Invalid size N=%d", N);
        return NULL;
    }
    
    if (direction != FFT_FORWARD && direction != FFT_INVERSE) {
        FFT_LOG_ERROR("Invalid direction %d", direction);
        return NULL;
    }
    
    //==========================================================================
    // ALLOCATE PLAN
    //==========================================================================
    fft_plan *plan = (fft_plan*)calloc(1, sizeof(fft_plan));
    if (!plan) return NULL;
    
    plan->n_input = N;
    plan->n_fft = N;
    plan->direction = direction;
    plan->use_bluestein = 0;
    
    //==========================================================================
    // STEP 1: FACTORIZE
    //==========================================================================
    int num_stages = factorize(N, plan->factors);
    
    if (num_stages < 0) {
        // ✅ Factorization failed - use Bluestein
        FFT_LOG_DEBUG("Cannot factorize N=%d, using Bluestein", N);
        
        if (plan_bluestein(plan, N, direction) < 0) {
            free_fft(plan);
            return NULL;
        }
        
        return plan;  // Bluestein plan complete
    }
    
    plan->num_stages = num_stages;
    
    FFT_LOG_DEBUG("Planning N=%d, direction=%s", 
           N, direction == FFT_FORWARD ? "FORWARD" : "INVERSE");
    
#ifdef FFT_DEBUG_PLANNING
    fprintf(stderr, "[FFT] Factorization: ");
    for (int i = 0; i < num_stages; i++) {
        fprintf(stderr, "%d%s", plan->factors[i], 
               i < num_stages-1 ? "×" : "\n");
    }
#endif
    
    //==========================================================================
    // STEP 2: BUILD STAGES (Call Managers)
    //==========================================================================
    int N_stage = N;
    
    for (int i = 0; i < num_stages; i++) {
        int radix = plan->factors[i];
        int sub_len = N_stage / radix;
        
        stage_descriptor *stage = &plan->stages[i];
        stage->radix = radix;
        stage->N_stage = N_stage;
        stage->sub_len = sub_len;
        
        //======================================================================
        // TWIDDLE MANAGER: Compute stage twiddles
        //======================================================================
        stage->stage_tw = compute_stage_twiddles(N_stage, radix, direction);
        
        if (!stage->stage_tw) {
            FFT_LOG_ERROR("Failed to compute twiddles for stage %d", i);
            free_fft(plan);
            return NULL;
        }
        
        FFT_LOG_DEBUG("Stage %d: radix=%d, N=%d, sub_len=%d, twiddles=%d",
               i, radix, N_stage, sub_len, (radix - 1) * sub_len);
        
        //======================================================================
        // RADER MANAGER: Get Rader twiddles (if prime radix)
        //======================================================================
        if (radix >= 7 && radix <= 67) {  // ✅ Extended prime support
            // Check if actually prime (simple check for our known set)
            int is_prime = (radix == 7 || radix == 11 || radix == 13 || 
                           radix == 17 || radix == 19 || radix == 23 ||
                           radix == 29 || radix == 31 || radix == 37 ||
                           radix == 41 || radix == 43 || radix == 47 ||
                           radix == 53 || radix == 59 || radix == 61 || radix == 67);
            
            if (is_prime) {
                stage->rader_tw = (fft_data*)get_rader_twiddles(radix, direction);
                
                if (!stage->rader_tw) {
                    FFT_LOG_ERROR("Failed to get Rader plan for prime %d", radix);
                    free_fft(plan);
                    return NULL;
                }
                
                FFT_LOG_DEBUG("  → Rader: prime=%d, conv_twiddles=%d", radix, radix - 1);
            }
        } else {
            stage->rader_tw = NULL;
        }
        
        N_stage = sub_len;
    }
    
    //==========================================================================
    // STEP 3: ALLOCATE SCRATCH BUFFER
    //==========================================================================
    // ✅ FIXED: Use MAX per stage (not sum) + safety margin
    size_t scratch_max = 0;
    N_stage = N;
    
    for (int i = 0; i < num_stages; i++) {
        int radix = plan->factors[i];
        int sub_len = N_stage / radix;
        
        // Each stage needs at most radix * sub_len for temp storage
        size_t stage_need = radix * sub_len;
        if (stage_need > scratch_max) {
            scratch_max = stage_need;
        }
        
        N_stage = sub_len;
    }
    
    // ✅ Add margin for Rader convolutions and safety
    size_t scratch_needed = scratch_max + 4 * N;
    
    plan->scratch_size = scratch_needed;
    plan->scratch = (fft_data*)aligned_alloc(32, scratch_needed * sizeof(fft_data));
    
    if (!plan->scratch) {
        FFT_LOG_ERROR("Failed to allocate scratch buffer (%zu elements)", scratch_needed);
        free_fft(plan);
        return NULL;
    }
    
    FFT_LOG_DEBUG("Scratch buffer: %zu elements (%.2f KB)",
           scratch_needed, scratch_needed * sizeof(fft_data) / 1024.0);
    FFT_LOG_DEBUG("Planning complete!");
    
    return plan;
}

//==============================================================================
// FREE PLAN
//==============================================================================

void free_fft(fft_object plan)
{
    if (!plan) return;
    
    // Free stage twiddles
    for (int i = 0; i < plan->num_stages; i++) {
        if (plan->stages[i].stage_tw) {
            free_stage_twiddles(plan->stages[i].stage_tw);
        }
        // Note: rader_tw is NOT freed here (it's from global cache)
    }
    
    // Free scratch
    if (plan->scratch) {
        aligned_free(plan->scratch);
    }
    
    // Free Bluestein resources (if used)
    if (plan->use_bluestein) {
        if (plan->bluestein_tw) {
            aligned_free(plan->bluestein_tw);
        }
        
        // Free internal Bluestein plans (recursive)
        if (plan->bluestein_plan_fwd) {
            free_fft((fft_plan*)plan->bluestein_plan_fwd);
        }
        if (plan->bluestein_plan_inv) {
            free_fft((fft_plan*)plan->bluestein_plan_inv);
        }
    }
    
    free(plan);
}