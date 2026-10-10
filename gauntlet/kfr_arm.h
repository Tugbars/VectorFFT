/* kfr_arm.h -- the KFR comparator arm of the gauntlet bench.
 *
 * kfr_arm.c implements it over KFR's C API (kfr/capi.h, the kfr_capi shared
 * library of KFR's release package). It compiles only with `build.py --kfr`,
 * which links the user's own KFR package; nothing of KFR ships with this
 * tree.
 *
 * Contract, the same as the MKL arm's 1D c2c cell: one double-precision
 * interleaved complex transform of length N, forward, out of place or in
 * place, single thread (KFR's DFT does not thread). The 1D r2c cell
 * (2026-10-10): N reals in, the N/2+1 CCE bins out (KFR's CCs format, the
 * layout our r2c writes), out of place, forward; KFR's real DFT takes an
 * even N only. The 1D c2r cell (2026-10-10): the same real plan backward,
 * the N/2+1 CCE bins in, N reals out, unnormalized. The 2D and 3D c2c cells
 * (2026-10-10): one N1 x N2 plane or N1 x N2 x N3 volume, row-major (the
 * last axis contiguous), forward, out of place. */
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
/* the 2D c2c cell: a plan for an N1 x N2 plane, row-major (N2 contiguous); KFR's 2D plan
 * is the same plan type, so kfr_c2c_temp_size / kfr_c2c_forward / kfr_c2c_destroy serve it */
void  *kfr_c2c2d_create(int N1, int N2);
/* the 3D c2c cell: a plan for an N1 x N2 x N3 volume, row-major (N3 contiguous); served the same way */
void  *kfr_c2c3d_create(int N1, int N2, int N3);
/* the 1D r2c cell: a plan for an even N (NULL for an odd N: KFR's real DFT is even-only) */
void  *kfr_r2c_create(int N);
size_t kfr_r2c_temp_size(const void *plan);
/* forward, out of place: N doubles in, N/2+1 interleaved complex out */
void   kfr_r2c_forward(const void *plan, const double *in, double *out, unsigned char *temp);
void   kfr_r2c_destroy(void *plan);
/* the 1D c2r cell: the same real plan (kfr_r2c_create / temp_size / destroy serve it);
 * backward, out of place: N/2+1 interleaved complex in, N doubles out, unnormalized */
void   kfr_c2r_backward(const void *plan, const double *in, double *out, unsigned char *temp);
/* the library's version string, for the report */
const char *kfr_arm_version(void);

#ifdef __cplusplus
}
#endif

#endif
