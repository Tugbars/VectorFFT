/* vfftf.c -- the FP32 library: the core's translation unit (vfft.c) compiled a second time for
 * float data (common/vfft_prec.h). Built only when VFFT_ENABLE_FLOAT is on. Its public functions
 * are vfftf_create / vfftf_execute / vfftf_destroy / vfftf_info (include/vfftf.h); vfft_malloc,
 * vfft_free and vfft_error are the FP64 unit's and serve both. */
#define VFFT_FLOAT 1
#include "vfftf_names.h"
#include "vfft.c"
