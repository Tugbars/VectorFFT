/* kfr_arm.c -- the KFR comparator arm (see kfr_arm.h), over KFR's C API.
 *
 * KFR's release packages ship its DFT behind a C API (kfr/capi.h, the
 * kfr_capi shared library, built by KFR's own Clang build). The bench stays
 * one gcc build and calls that library: no C++ runtime and no second
 * compiler on our side, and KFR runs the code its authors built.
 *
 * kfr_c64 is two interleaved doubles (C99 double _Complex, or a plain double
 * pair without complex support): the bench's buffers are passed through a
 * cast, never copied. */
#include "kfr_arm.h"

#include <kfr/capi.h>

void *kfr_c2c_create(int N)
{
    return N < 1 ? NULL : (void *)kfr_dft_create_plan_f64((size_t)N);
}

size_t kfr_c2c_temp_size(const void *plan)
{
    return plan ? kfr_dft_get_temp_size_f64((KFR_DFT_PLAN_F64 *)plan) : 0;
}

void kfr_c2c_forward(const void *plan, const double *in, double *out, unsigned char *temp)
{
    kfr_dft_execute_f64((KFR_DFT_PLAN_F64 *)plan, (kfr_c64 *)out, (const kfr_c64 *)in, temp);
}

void kfr_c2c_forward_inplace(const void *plan, double *inout, unsigned char *temp)
{
    kfr_dft_execute_f64((KFR_DFT_PLAN_F64 *)plan, (kfr_c64 *)inout, (const kfr_c64 *)inout, temp);
}

void kfr_c2c_destroy(void *plan)
{
    if (plan)
        kfr_dft_delete_plan_f64((KFR_DFT_PLAN_F64 *)plan);
}

const char *kfr_arm_version(void)
{
    return kfr_version_string();
}
