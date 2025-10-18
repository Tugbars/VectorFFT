// fft_planning_types.h
// Core types for FFTW-style FFT planning system

#ifndef FFT_PLANNING_TYPES_H
#define FFT_PLANNING_TYPES_H

#include <stddef.h>
#include <stdint.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#define MAX_FFT_STAGES 32  // ✅ Configurable max stages (increased from hardcoded 32)

//==============================================================================
// BASIC TYPES
//==============================================================================

typedef struct { double re, im; } fft_data;

typedef enum { 
    FFT_FORWARD = 1, 
    FFT_INVERSE = -1 
} fft_direction_t;

//==============================================================================
// STAGE DESCRIPTOR - What each stage needs
//==============================================================================

typedef struct {
    // Stage geometry
    int radix;           // 2, 3, 5, 7, 11, 13, etc.
    int N_stage;         // Size at this stage (N / product of previous radices)
    int sub_len;         // N_stage / radix
    
    // ✅ Pre-computed Cooley-Tukey stage twiddles
    // Layout: stage_tw[k*(radix-1) + (r-1)] = W^(r*k)
    // Size: (radix-1) * sub_len complex numbers
    fft_data *stage_tw;
    
    // ✅ Pre-computed Rader convolution twiddles (NULL if not Rader radix)
    // Points to global Rader cache (shared across stages with same prime)
    // NOT owned by this stage (do not free)
    fft_data *rader_tw;
    
} stage_descriptor;

//==============================================================================
// FFT PLAN - Everything pre-computed at planning time
//==============================================================================

typedef struct fft_plan_struct {
    // Input/output sizes
    int n_input;         // User's requested size
    int n_fft;           // Actual FFT size (padded for Bluestein if needed)
    
    // Transform direction
    fft_direction_t direction;
    
    // Stage decomposition (for mixed-radix)
    int num_stages;                          // Number of mixed-radix stages
    int factors[MAX_FFT_STAGES];             // Radix sequence [r0, r1, r2, ...]
    stage_descriptor stages[MAX_FFT_STAGES]; // Pre-computed stage info
    
    // Algorithm type
    int use_bluestein;   // 0 = mixed-radix, 1 = Bluestein
    
    // Scratch buffer (shared by all stages)
    fft_data *scratch;
    size_t scratch_size;
    
    // Bluestein-specific (if use_bluestein = 1)
    fft_data *bluestein_tw;      // Chirp twiddles b[k] = exp(πi k²/N)
    void *bluestein_plan_fwd;    // Internal FFT plan for padded size M
    void *bluestein_plan_inv;    // Internal inverse FFT plan
    
} fft_plan;

typedef fft_plan* fft_object;  // For compatibility with your existing code

//==============================================================================
// RADER PLAN CACHE ENTRY
//==============================================================================

typedef struct {
    int prime;                   // 7, 11, 13, 17, 19, 23, etc.
    
    // Direction-specific convolution twiddles (OWNED by cache)
    fft_data *conv_tw_fwd;       // exp(-2πi * out_perm[q] / prime)
    fft_data *conv_tw_inv;       // exp(+2πi * out_perm[q] / prime)
    
    // ✅ Permutation arrays (shared by both directions, OWNED by cache)
    // These are useful for:
    // - Dynamic butterfly generation
    // - Verification/testing
    // - Debugging
    // Butterfly implementations may hardcode these for performance
    int *perm_in;                // Input permutation: [g^0, g^1, ..., g^(p-2)] mod p
    int *perm_out;               // Output permutation: inverse of perm_in
    
    int primitive_root;          // Generator (e.g., 3 for prime=7)
    
} rader_plan_cache_entry;

//==============================================================================
// RADIX FUNCTION POINTER TYPES
//==============================================================================

// Forward radix butterfly signature
typedef void (*radix_fv_func)(
    fft_data *restrict output,
    fft_data *restrict input,
    const fft_data *restrict stage_tw,
    const fft_data *restrict rader_tw,
    int sub_len
);

// Inverse radix butterfly signature
typedef void (*radix_bv_func)(
    fft_data *restrict output,
    fft_data *restrict input,
    const fft_data *restrict stage_tw,
    const fft_data *restrict rader_tw,
    int sub_len
);

#endif // FFT_PLANNING_TYPES_H