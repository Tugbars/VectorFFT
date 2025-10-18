// fft_planning.c
// MAIN PLANNING ORCHESTRATOR - The "fft_init" umbrella
// Coordinates: Factorization → Twiddle Manager → Rader Manager → Plan

#ifndef FFT_PLANNING_H
#define FFT_PLANNING_H

#include "fft_planning_types.h"

//==============================================================================
// MAIN API
//==============================================================================

/**
 * @brief Create FFT plan (FFTW-style planning)
 * 
 * This is the UMBRELLA function that orchestrates:
 *   1. Factorization
 *   2. Twiddle Manager (compute stage twiddles)
 *   3. Rader Manager (get Rader plans)
 *   4. Scratch allocation
 * 
 * @param N Signal length
 * @param direction FORWARD or INVERSE
 * @return Complete FFT plan (or NULL on error)
 */
fft_object fft_init(int N, fft_direction_t direction);

/**
 * @brief Free FFT plan
 */
void free_fft(fft_object plan);

#endif // FFT_PLANNING_H

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

//==============================================================================
// FACTORIZATION
//==============================================================================

/**
 * @brief Factorize N into radices (FFTW-style: prefer large radices)
 * 
 * Strategy: Try radices in order: 32, 16, 13, 11, 9, 8, 7, 5, 4, 3, 2
 */
static int factorize(int N, int *factors)
{
    int num_factors = 0;
    int n = N;
    
    // Radix priority (largest first for efficiency)
    const int radix_order[] = {32, 16, 13, 11, 9, 8, 7, 5, 4, 3, 2};
    const int num_radices = 11;
    
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
            // Can't factorize - need Bluestein
            fprintf(stderr, "[FFT] Cannot factorize N=%d into supported radices\n", N);
            return -1;
        }
        
        if (num_factors >= 32) {
            fprintf(stderr, "[FFT] Too many factors (>32) for N=%d\n", N);
            return -1;
        }
    }
    
    return num_factors;
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
        fprintf(stderr, "[FFT] Invalid size N=%d\n", N);
        return NULL;
    }
    
    if (direction != FFT_FORWARD && direction != FFT_INVERSE) {
        fprintf(stderr, "[FFT] Invalid direction %d\n", direction);
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
        // Factorization failed - TODO: use Bluestein
        fprintf(stderr, "[FFT] Bluestein not yet implemented\n");
        free(plan);
        return NULL;
    }
    
    plan->num_stages = num_stages;
    
#ifdef FFT_DEBUG_PLANNING
    fprintf(stderr, "[FFT] Planning N=%d, direction=%s\n", 
           N, direction == FFT_FORWARD ? "FORWARD" : "INVERSE");
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
            fprintf(stderr, "[FFT] Failed to compute twiddles for stage %d\n", i);
            free_fft(plan);
            return NULL;
        }
        
#ifdef FFT_DEBUG_PLANNING
        int num_tw = (radix - 1) * sub_len;
        fprintf(stderr, "[FFT] Stage %d: radix=%d, N=%d, sub_len=%d, twiddles=%d\n",
               i, radix, N_stage, sub_len, num_tw);
#endif
        
        //======================================================================
        // RADER MANAGER: Get Rader twiddles (if prime radix)
        //======================================================================
        if (radix == 7 || radix == 11 || radix == 13) {
            stage->rader_tw = (fft_data*)get_rader_twiddles(radix, direction);
            
            if (!stage->rader_tw) {
                fprintf(stderr, "[FFT] Failed to get Rader plan for prime %d\n", radix);
                free_fft(plan);
                return NULL;
            }
            
#ifdef FFT_DEBUG_PLANNING
            fprintf(stderr, "[FFT]   → Rader: prime=%d, conv_twiddles=%d\n",
                   radix, radix - 1);
#endif
        } else {
            stage->rader_tw = NULL;
        }
        
        N_stage = sub_len;
    }
    
    //==========================================================================
    // STEP 3: ALLOCATE SCRATCH BUFFER
    //==========================================================================
    // Compute required scratch size (worst case: radix * sub_len per stage)
    size_t scratch_needed = 0;
    N_stage = N;
    
    for (int i = 0; i < num_stages; i++) {
        int radix = plan->factors[i];
        int sub_len = N_stage / radix;
        scratch_needed += radix * sub_len;
        N_stage = sub_len;
    }
    
    // Add safety margin
    if (scratch_needed < 4 * N) {
        scratch_needed = 4 * N;
    }
    
    plan->scratch_size = scratch_needed;
    plan->scratch = (fft_data*)aligned_alloc(32, scratch_needed * sizeof(fft_data));
    
    if (!plan->scratch) {
        fprintf(stderr, "[FFT] Failed to allocate scratch buffer (%zu elements)\n",
               scratch_needed);
        free_fft(plan);
        return NULL;
    }
    
#ifdef FFT_DEBUG_PLANNING
    fprintf(stderr, "[FFT] Scratch buffer: %zu elements (%.2f KB)\n",
           scratch_needed, scratch_needed * sizeof(fft_data) / 1024.0);
    fprintf(stderr, "[FFT] Planning complete!\n\n");
#endif
    
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
        if (plan->bluestein_tw) aligned_free(plan->bluestein_tw);
        // TODO: Free internal Bluestein plans
    }
    
    free(plan);
}