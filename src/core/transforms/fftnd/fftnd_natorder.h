/**
 * fftnd_natorder.h -- per-axis natural-order MAPS for fftnd plans, plus the
 * gather/scatter utilities that make them useful. The transform itself stays
 * scrambled (roundtrip-definitive, and the fusion architecture's contract);
 * these are the plan-time tools for the consumers that need addressing:
 *
 *   maps[m][k]  -- natural bin k of axis m lives at row maps[m][k] of the
 *                  scrambled output. O(1) bin addressing:
 *                      X_natural(k0,..,kd) = S[ flat(maps[0][k0], ...) ]
 *   gather      -- one out-of-place sweep scrambled -> natural (export /
 *                  cross-library elementwise comparison).
 *   scatter     -- natural -> scrambled placement: permute a multiplier
 *                  table (ik-vectors, Green's function, transfer function)
 *                  ONCE at setup so every subsequent pointwise op runs on
 *                  the scrambled spectrum directly, zero per-solve cost.
 *
 * MAP EXTRACTION IS CHAIN-FREE (phase probing): with an impulse at
 * position n0=1 on axis m (zeros elsewhere), the true spectrum is
 * X[k] = exp(-2*pi*i * k_m / N_m) -- constant across every other axis. So a
 * single pencil of the scrambled output along axis m (other indices 0)
 * reads  S[j] = exp(-2*pi*i * q(j) / N_m)  where q is the inverse map, and
 * q falls out of the phases:  q(j) = round(-angle(S[j]) * N_m / 2*pi) mod N.
 * Phases are separated by 2*pi/N_m while the values are accurate to
 * ~1e-15, so the rounding has astronomical margin; the probe validates
 * unit magnitude, integer residual, and bijectivity, and REFUSES (returns
 * 0) rather than ever returning a silently-wrong map -- the same fail-safe
 * posture as the 1D natorder machinery. Works uniformly for DIT, DIF, and
 * override (Rader/Bluestein) axes; the latter emit natural order already
 * and probe to the identity with no special-casing.
 *
 * Cost: one forward transform per axis, plan-time only.
 */
#ifndef STRIDE_FFTND_NATORDER_H
#define STRIDE_FFTND_NATORDER_H

#include <math.h>
#include "fftnd.h"

#ifndef FFTND_NAT_PI
#define FFTND_NAT_PI 3.14159265358979323846
#endif

/* Probe one axis of an fftnd plan. Returns malloc'd map (natural[k] at row
 * map[k]) or NULL. re/im are caller-provided cube scratch (total doubles). */
static int *_fftnd_nat_probe_axis(stride_plan_t *plan,
                                  const stride_fftnd_data_t *d, int m,
                                  double *re, double *im) {
    const int N = d->N[m];
    const size_t Km = d->K[m];
    memset(re, 0, d->total * sizeof(double));
    memset(im, 0, d->total * sizeof(double));
    re[Km] = 1.0;                       /* impulse at axis-m position 1 */
    stride_execute_fwd(plan, re, im);

    int *q = (int *)malloc((size_t)N * sizeof(int));
    int *map = (int *)malloc((size_t)N * sizeof(int));
    char *seen = (char *)calloc((size_t)N, 1);
    if (!q || !map || !seen) { free(q); free(map); free(seen); return NULL; }

    int ok = 1;
    for (int j = 0; j < N && ok; j++) {
        double sr = re[(size_t)j * Km], si = im[(size_t)j * Km];
        double mag = sqrt(sr * sr + si * si);
        if (fabs(mag - 1.0) > 1e-6) { ok = 0; break; }
        double qf = -atan2(si, sr) * (double)N / (2.0 * FFTND_NAT_PI);
        long qi = lround(qf);
        if (fabs(qf - (double)qi) > 0.01) { ok = 0; break; }
        int qm = (int)(((qi % N) + N) % N);
        if (seen[qm]) { ok = 0; break; }        /* bijectivity */
        seen[qm] = 1;
        q[j] = qm;
    }
    if (ok)
        for (int j = 0; j < N; j++) map[q[j]] = j;
    free(q); free(seen);
    if (!ok) { free(map); return NULL; }
    return map;
}

/** Build all per-axis maps for an fftnd plan (the plan returned by
 *  stride_plan_nd / _nd_from / _nd_wise). maps[m] receives a malloc'd
 *  int[N[m]] (caller frees); on failure all are freed and 0 is returned.
 *  Runs `rank` forward transforms on internal scratch -- plan-time only. */
static int fftnd_natorder_maps(stride_plan_t *plan,
                               int *maps[FFTND_MAX_RANK]) {
    if (!plan || !plan->override_data) return 0;
    const stride_fftnd_data_t *d =
        (const stride_fftnd_data_t *)plan->override_data;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, d->total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, d->total * sizeof(double));
    if (!re || !im) {
        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        return 0;
    }
    int ok = 1;
    for (int m = 0; m < d->rank; m++) maps[m] = NULL;
    for (int m = 0; m < d->rank && ok; m++) {
#ifdef VFFT_STRIDED_ROWS
        /* strided rows are plan-time VERIFIED natural (strided_rows.h), so
         * the last-axis map is identity by contract -- skip the probe
         * transform for that axis. */
        if (m == d->rank - 1 && d->srow_fwd) {
            maps[m] = (int *)malloc((size_t)d->N[m] * sizeof(int));
            if (!maps[m]) { ok = 0; break; }
            for (int k = 0; k < d->N[m]; k++) maps[m][k] = k;
            continue;
        }
#endif
        maps[m] = _fftnd_nat_probe_axis(plan, d, m, re, im);
        if (!maps[m]) ok = 0;
    }
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    if (!ok)
        for (int m = 0; m < d->rank; m++) { free(maps[m]); maps[m] = NULL; }
    return ok;
}

/* ── flat-index walkers ─────────────────────────────────────────── */

/** Gather scrambled -> natural (out-of-place):
 *  nat[k0,..] = scr[maps[0][k0], ..]. One sweep; export / comparison. */
static void fftnd_natorder_gather(const stride_fftnd_data_t *d,
                                  int *const maps[FFTND_MAX_RANK],
                                  const double *scr_re, const double *scr_im,
                                  double *nat_re, double *nat_im) {
    const int r = d->rank;
    int idx[FFTND_MAX_RANK] = { 0 };
    for (size_t kf = 0; kf < d->total; kf++) {
        size_t sf = 0;
        for (int m = 0; m < r; m++)
            sf += (size_t)maps[m][idx[m]] * d->K[m];
        nat_re[kf] = scr_re[sf];
        nat_im[kf] = scr_im[sf];
        for (int m = r - 1; m >= 0; m--) {      /* increment multi-index */
            if (++idx[m] < d->N[m]) break;
            idx[m] = 0;
        }
    }
}

/** Scatter natural -> scrambled placement (out-of-place):
 *  dst[maps[0][k0], ..] = nat[k0, ..]. Permute a pointwise multiplier
 *  table once at setup; thereafter multiply the scrambled spectrum
 *  directly (conv.h-style), zero per-use cost. */
static void fftnd_natorder_scatter(const stride_fftnd_data_t *d,
                                   int *const maps[FFTND_MAX_RANK],
                                   const double *nat_re, const double *nat_im,
                                   double *dst_re, double *dst_im) {
    const int r = d->rank;
    int idx[FFTND_MAX_RANK] = { 0 };
    for (size_t kf = 0; kf < d->total; kf++) {
        size_t sf = 0;
        for (int m = 0; m < r; m++)
            sf += (size_t)maps[m][idx[m]] * d->K[m];
        dst_re[sf] = nat_re[kf];
        dst_im[sf] = nat_im[kf];
        for (int m = r - 1; m >= 0; m--) {
            if (++idx[m] < d->N[m]) break;
            idx[m] = 0;
        }
    }
}

#endif /* STRIDE_FFTND_NATORDER_H */
