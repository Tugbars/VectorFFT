/* kfr_arm.h -- the KFR comparator arm of the gauntlet bench, behind a C
 * interface (KFR is a C++ library and wants Clang; the bench is C).
 *
 * Wired 2026-09-25 for a future run, UNTESTED: no Clang and no KFR were
 * installed on the calibration host when it was written. The arm compiles
 * only with `build.py --kfr` (which requires a Clang toolchain, CC=clang) and
 * links the user's own KFR checkout; nothing of KFR ships with this tree.
 *
 * Contract, the same as the MKL arm's 1D c2c cell: one double-precision
 * interleaved complex transform of length N, forward, out of place or in
 * place, single thread (KFR's DFT does not thread). */
#ifndef VFFT_GAUNTLET_KFR_ARM_H
#define VFFT_GAUNTLET_KFR_ARM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* a plan for length N; NULL when KFR refuses the length */
void  *kfr_c2c_create(int N);
/* the scratch KFR wants per execute, in bytes (0 = none) */
size_t kfr_c2c_temp_size(const void *plan);
/* forward, out of place: in and out are 2N doubles, interleaved */
void   kfr_c2c_forward(const void *plan, const double *in, double *out, unsigned char *temp);
/* forward, in place on inout (2N doubles) */
void   kfr_c2c_forward_inplace(const void *plan, double *inout, unsigned char *temp);
void   kfr_c2c_destroy(void *plan);
/* the library's version string, for the report */
const char *kfr_arm_version(void);

#ifdef __cplusplus
}
#endif

#endif
