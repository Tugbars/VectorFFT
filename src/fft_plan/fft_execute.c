//==============================================================================
// fft_execute.c - Execution engine
//==============================================================================

#include "fft_planning_types.h"
#include "fft_planning.h"
#include "../bluestein/bluestein.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #include <malloc.h>
    #define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define aligned_free(ptr) _aligned_free(ptr)
#else
    #define aligned_free(ptr) free(ptr)
#endif

//==============================================================================
// BIT REVERSAL
//==============================================================================

static inline unsigned int bit_reverse(unsigned int x, int bits)
{
    unsigned int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

static void bit_reverse_permutation(fft_data *data, int N)
{
    int bits = __builtin_ctz(N);  // log2(N) for power-of-2
    
    for (unsigned int i = 0; i < (unsigned int)N; i++) {
        unsigned int j = bit_reverse(i, bits);
        
        if (i < j) {
            fft_data temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
}

//==============================================================================
// IN-PLACE RADIX-2 BUTTERFLY
//==============================================================================

static inline void butterfly_radix2_inplace(
    fft_data *data,
    int idx,
    int distance,
    fft_data twiddle)
{
    fft_data a = data[idx];
    fft_data b = data[idx + distance];
    
    // Complex multiply: b * twiddle
    fft_data temp;
    temp.re = b.re * twiddle.re - b.im * twiddle.im;
    temp.im = b.re * twiddle.im + b.im * twiddle.re;
    
    // Write results
    data[idx].re = a.re + temp.re;
    data[idx].im = a.im + temp.im;
    
    data[idx + distance].re = a.re - temp.re;
    data[idx + distance].im = a.im - temp.im;
}

//==============================================================================
// IN-PLACE EXECUTION (Power-of-2 only)
//==============================================================================

static int fft_exec_inplace_bitrev_internal(fft_object plan, fft_data *data)
{
    const int N = plan->n_fft;
    
    // Step 1: Bit-reverse permutation
    bit_reverse_permutation(data, N);
    
    // Step 2: Cooley-Tukey stages (each radix-2)
    int distance = 1;
    
    for (int stage = 0; stage < plan->num_stages; stage++) {
        stage_descriptor *s = &plan->stages[stage];
        
        if (s->radix != 2) {
            FFT_LOG_ERROR("Non-radix-2 in power-of-2 plan! (bug)");
            return -1;
        }
        
        const fft_data *twiddles = s->stage_tw;
        const int num_groups = N / (2 * distance);
        
        for (int group = 0; group < num_groups; group++) {
            for (int k = 0; k < distance; k++) {
                int idx = group * 2 * distance + k;
                
                // Get twiddle: W_N^(k * num_groups)
                fft_data W = twiddles[k * num_groups];
                
                butterfly_radix2_inplace(data, idx, distance, W);
            }
        }
        
        distance *= 2;
    }
    
    return 0;
}

//==============================================================================
// STOCKHAM EXECUTION (Mixed-radix with temp buffer)
//==============================================================================

// Forward declaration of radix butterflies (you'll implement these)
extern void radix_2_kernel(const fft_data *in, fft_data *out, 
                           const fft_data *tw, int stride_in, int stride_out, int count);
extern void radix_3_kernel(const fft_data *in, fft_data *out,
                           const fft_data *tw, int stride_in, int stride_out, int count);
extern void radix_4_kernel(const fft_data *in, fft_data *out,
                           const fft_data *tw, int stride_in, int stride_out, int count);
// ... more radices

static int fft_exec_stockham_internal(
    fft_object plan,
    const fft_data *input,
    fft_data *output,
    fft_data *temp)
{
    const int N = plan->n_fft;
    
    // Ping-pong between buffers
    const fft_data *in_buf = input;
    fft_data *out_buf = temp;
    
    for (int stage = 0; stage < plan->num_stages; stage++) {
        stage_descriptor *s = &plan->stages[stage];
        const int radix = s->radix;
        const int sub_len = s->sub_len;
        const int num_groups = N / s->N_stage;
        
        // Compute stride for this stage
        int stride_in = 1;
        int stride_out = sub_len;
        
        // Call appropriate radix kernel
        switch (radix) {
            case 2:
                radix_2_kernel(in_buf, out_buf, s->stage_tw, 
                              stride_in, stride_out, num_groups);
                break;
            case 3:
                radix_3_kernel(in_buf, out_buf, s->stage_tw,
                              stride_in, stride_out, num_groups);
                break;
            case 4:
                radix_4_kernel(in_buf, out_buf, s->stage_tw,
                              stride_in, stride_out, num_groups);
                break;
            // ... more radices
            
            default:
                FFT_LOG_ERROR("Unsupported radix %d in Stockham", radix);
                return -1;
        }
        
        // Swap buffers
        const fft_data *swap = in_buf;
        in_buf = out_buf;
        out_buf = (fft_data*)swap;
    }
    
    // Final result might be in temp buffer
    if (in_buf != output) {
        memcpy(output, in_buf, N * sizeof(fft_data));
    }
    
    return 0;
}

//==============================================================================
// PUBLIC EXECUTION FUNCTIONS
//==============================================================================

int fft_exec_inplace(fft_object plan, fft_data *data)
{
    if (!plan || !data) {
        FFT_LOG_ERROR("Invalid parameters to fft_exec_inplace");
        return -1;
    }
    
    if (plan->strategy != FFT_EXEC_INPLACE_BITREV) {
        FFT_LOG_ERROR("Plan does not support true in-place execution");
        FFT_LOG_ERROR("This plan requires workspace (use fft_exec or fft_exec_dft)");
        return -1;
    }
    
    return fft_exec_inplace_bitrev_internal(plan, data);
}

int fft_exec(fft_object plan, const fft_data *input, fft_data *output)
{
    if (!plan || !input || !output) {
        FFT_LOG_ERROR("Invalid parameters to fft_exec");
        return -1;
    }
    
    switch (plan->strategy) {
        case FFT_EXEC_INPLACE_BITREV:
            // Can do in-place: copy input to output, then transform
            if (input != output) {
                memcpy(output, input, plan->n_fft * sizeof(fft_data));
            }
            return fft_exec_inplace_bitrev_internal(plan, output);
        
        case FFT_EXEC_STOCKHAM:
        case FFT_EXEC_BLUESTEIN:
            FFT_LOG_ERROR("This plan requires workspace buffer");
            FFT_LOG_ERROR("Use fft_exec_dft(plan, input, output, workspace)");
            return -1;
        
        default:
            FFT_LOG_ERROR("Unknown execution strategy");
            return -1;
    }
}

int fft_exec_dft(
    fft_object plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace)
{
    if (!plan || !input || !output) {
        FFT_LOG_ERROR("Invalid parameters to fft_exec_dft");
        return -1;
    }
    
    switch (plan->strategy) {
        case FFT_EXEC_INPLACE_BITREV:
            if (input != output) {
                memcpy(output, input, plan->n_fft * sizeof(fft_data));
            }
            return fft_exec_inplace_bitrev_internal(plan, output);
        
        case FFT_EXEC_STOCKHAM:
            if (!workspace) {
                FFT_LOG_ERROR("Workspace required for Stockham algorithm");
                return -1;
            }
            return fft_exec_stockham_internal(plan, input, output, workspace);
        
        case FFT_EXEC_BLUESTEIN:
            if (!workspace) {
                FFT_LOG_ERROR("Workspace required for Bluestein algorithm");
                return -1;
            }
            
            size_t scratch_size = bluestein_get_scratch_size(plan->n_input);
            
            if (plan->direction == FFT_FORWARD) {
                return bluestein_exec_forward(
                    plan->bluestein_fwd,
                    input,
                    output,
                    workspace,
                    scratch_size
                );
            } else {
                return bluestein_exec_inverse(
                    plan->bluestein_inv,
                    input,
                    output,
                    workspace,
                    scratch_size
                );
            }
        
        default:
            FFT_LOG_ERROR("Unknown execution strategy");
            return -1;
    }
    
    // ❌ REMOVED: No normalization here!
    // User applies 1/N manually if needed
}


int fft_exec_normalized(
    fft_object plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace)
{
    // Execute unnormalized transform
    int result = fft_exec_dft(plan, input, output, workspace);
    if (result != 0) return result;
    
    // Apply 1/N normalization
    const double scale = 1.0 / (double)plan->n_fft;
    for (int i = 0; i < plan->n_fft; i++) {
        output[i].re *= scale;
        output[i].im *= scale;
    }
    
    return 0;
}

int fft_roundtrip_normalized(
    fft_object fwd_plan,
    fft_object inv_plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace)
{
    if (!fwd_plan || !inv_plan || !input || !output) {
        return -1;
    }
    
    if (fwd_plan->n_fft != inv_plan->n_fft) {
        FFT_LOG_ERROR("Plan size mismatch: forward=%d, inverse=%d",
                      fwd_plan->n_fft, inv_plan->n_fft);
        return -1;
    }
    
    const int N = fwd_plan->n_fft;
    
    // Allocate temporary buffer for frequency domain
    fft_data *freq = (fft_data*)malloc(N * sizeof(fft_data));
    if (!freq) return -1;
    
    // Forward transform
    int result = fft_exec_dft(fwd_plan, input, freq, workspace);
    if (result != 0) {
        free(freq);
        return result;
    }
    
    // Inverse transform
    result = fft_exec_dft(inv_plan, freq, output, workspace);
    if (result != 0) {
        free(freq);
        return result;
    }
    
    // Normalize by 1/N
    const double scale = 1.0 / (double)N;
    for (int i = 0; i < N; i++) {
        output[i].re *= scale;
        output[i].im *= scale;
    }
    
    free(freq);
    return 0;
}