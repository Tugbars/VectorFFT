/**
 * @file fft_butterfly_dispatch.c
 * @brief Butterfly kernel registry for both n1 and twiddle butterflies
 */

#include "fft_butterfly_dispatch.h"

// Include all butterfly headers
#include "fft_radix2.h"
#include "fft_radix3.h"
#include "fft_radix4.h"
#include "fft_radix5.h"
#include "fft_radix7.h"
#include "fft_radix8.h"
#include "fft_radix11.h"
#include "fft_radix13.h"
#include "fft_radix16.h"
#include "fft_radix32.h"

//==============================================================================
// WRAPPERS FOR NON-STANDARD SIGNATURES
//==============================================================================

// Radix-7 needs Rader twiddles (for twiddle version)
static void radix7_fv_wrapper(
    fft_data *output,
    const fft_data *input,
    const fft_twiddles_soa_view *twiddles,
    int sub_len)
{
    fft_radix7_fv(output, input, twiddles, NULL, sub_len);
}

static void radix7_bv_wrapper(
    fft_data *output,
    const fft_data *input,
    const fft_twiddles_soa_view *twiddles,
    int sub_len)
{
    fft_radix7_bv(output, input, twiddles, NULL, sub_len);
}

// Radix-8 uses sign parameter
static void radix8_fv_wrapper(
    fft_data *output,
    const fft_data *input,
    const fft_twiddles_soa_view *twiddles,
    int sub_len)
{
    fft_radix8_butterfly(output, input, twiddles, sub_len, -1);
}

static void radix8_bv_wrapper(
    fft_data *output,
    const fft_data *input,
    const fft_twiddles_soa_view *twiddles,
    int sub_len)
{
    fft_radix8_butterfly(output, input, twiddles, sub_len, +1);
}

// Radix-8 n1 versions (if you have them)
static void radix8_fn1_wrapper(
    fft_data *output,
    const fft_data *input,
    int sub_len)
{
    fft_radix8_butterfly(output, input, NULL, sub_len, -1);
}

static void radix8_bn1_wrapper(
    fft_data *output,
    const fft_data *input,
    int sub_len)
{
    fft_radix8_butterfly(output, input, NULL, sub_len, +1);
}

//==============================================================================
// DISPATCH TABLE
//==============================================================================

/**
 * @brief Butterfly function pair lookup table
 * 
 * Table structure: [radix_index][direction][function_type]
 * - radix_index: 0=radix2, 1=radix3, ..., 9=radix32
 * - direction: 0=inverse, 1=forward
 * - Contains both n1 and twiddle versions
 */
typedef struct {
    butterfly_n1_func_t n1;
    butterfly_twiddle_func_t twiddle;
} butterfly_entry_t;

static const butterfly_entry_t BUTTERFLY_TABLE[10][2] = {
    // Radix 2
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix2_bn1,      // Assuming you have these
            .twiddle = (butterfly_twiddle_func_t)fft_radix2_bv
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix2_fn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix2_fv
        }
    },
    
    // Radix 3
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix3_bn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix3_bv
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix3_fn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix3_fv
        }
    },
    
    // Radix 4
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix4_bn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix4_bv
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix4_fn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix4_fv
        }
    },
    
    // Radix 5
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix5_bn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix5_bv
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix5_fn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix5_fv
        }
    },
    
    // Radix 7
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix7_bn1,
            .twiddle = radix7_bv_wrapper
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix7_fn1,
            .twiddle = radix7_fv_wrapper
        }
    },
    
    // Radix 8
    {
        // Inverse
        { 
            .n1 = radix8_bn1_wrapper,
            .twiddle = radix8_bv_wrapper
        },
        // Forward
        { 
            .n1 = radix8_fn1_wrapper,
            .twiddle = radix8_fv_wrapper
        }
    },
    
    // Radix 11
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix11_bn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix11_bv
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix11_fn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix11_fv
        }
    },
    
    // Radix 13
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix13_bn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix13_bv
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix13_fn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix13_fv
        }
    },
    
    // Radix 16
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix16_bn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix16_bv
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix16_fn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix16_fv
        }
    },
    
    // Radix 32
    {
        // Inverse
        { 
            .n1 = (butterfly_n1_func_t)fft_radix32_bn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix32_bv
        },
        // Forward
        { 
            .n1 = (butterfly_n1_func_t)fft_radix32_fn1,
            .twiddle = (butterfly_twiddle_func_t)fft_radix32_fv
        }
    }
};

/**
 * @brief Map radix to table index
 */
static inline int radix_to_index(int radix)
{
    switch (radix) {
        case 2:  return 0;
        case 3:  return 1;
        case 4:  return 2;
        case 5:  return 3;
        case 7:  return 4;
        case 8:  return 5;
        case 11: return 6;
        case 13: return 7;
        case 16: return 8;
        case 32: return 9;
        default: return -1;
    }
}

//==============================================================================
// PUBLIC API
//==============================================================================

butterfly_pair_t get_butterfly_pair(int radix, int is_forward)
{
    butterfly_pair_t empty = { .n1 = NULL, .twiddle = NULL };
    
    int idx = radix_to_index(radix);
    if (idx < 0) {
        return empty;
    }
    
    return BUTTERFLY_TABLE[idx][is_forward ? 1 : 0];
}