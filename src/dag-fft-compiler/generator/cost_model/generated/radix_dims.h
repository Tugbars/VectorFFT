/* radix_dims.h — radix-array bound for the CPE and memboundness tables.
 *
 * This is the one symbol that outlived radix_profile.h. The static
 * op-count profile was removed (the compiler rewrites op counts at
 * codegen, so a per-radix op table predicts something it cannot
 * predict; the cost model now uses measured CPE plus a coarse analytic
 * prior, see _radix_coarse_cpe in factorizer.h). The name is kept
 * unchanged so the existing `[STRIDE_RADIX_PROFILE_MAX_R]` array
 * declarations in radix_cpe.h / radix_memboundness.h need no edits.
 *
 * Value left at 1025 deliberately: it is only an array bound, it
 * generates and selects nothing, and keeping it generous means the
 * measured CPE/memboundness tables (which still carry historical
 * 128/256/512 rows) stay in-bounds. The trimmed registry is what
 * actually gates radix selection to <= 64. */
#ifndef STRIDE_RADIX_DIMS_H
#define STRIDE_RADIX_DIMS_H

#define STRIDE_RADIX_PROFILE_MAX_R 1025

#endif /* STRIDE_RADIX_DIMS_H */
