/* rfft_calibrate.h — measured calibration of the rfft (real-FFT forward) axis.
 *
 * Brute-forces every coverable factorization × per-stage variant (FLAT / LOG3 /
 * T1S), gates each candidate vs a reference (the seed factorization's output on
 * the roundtrip-validated engine), times best-of-5, and keeps the fastest. Fills
 * a `vfft_proto_wisdom_entry_t` (factors + variants + best_ns) destined for
 * rfft_wisdom.txt.
 *
 * This is the rfft analogue of `_calibrate_c2c` (the DP planner) — it lets
 * vfft_create calibrate the *rfft's own axis* on a wisdom miss, instead of
 * falling back to the fewest-stage heuristic. Lifted from the standalone
 * calibrator build_tuned/benches/calibrator/calibrate_r2c.c.
 *
 * The rfft search space is small (a handful of factorizations × 3^stages
 * variants), so the sweep is exhaustive and fast regardless of the requested
 * rigor; the CALLER decides whether to invoke it (rigor-gated in vfft.c).
 */
#ifndef VFFT_RFFT_CALIBRATE_H
#define VFFT_RFFT_CALIBRATE_H

#include "rfft.h"            /* rfft_codelets_t, rfft_plan_create_ex, execute   */
#include "dp_planner.h"      /* vfft_proto_now_ns                               */
#include "wisdom_reader.h"   /* vfft_proto_wisdom_entry_t + STRIDE_MAX_STAGES   */
#include "proto_stride_compat.h" /* vfft_proto_posix_memalign / aligned_free    */
#include <math.h>
#include <string.h>

#ifndef VFFT_RFFT_CAL_MAX_FACZ
#define VFFT_RFFT_CAL_MAX_FACZ 1024   /* cap on enumerated factorizations */
#endif
/* Cap the factor count. The optimum is always shallow (doc-60 U-curve: the (8,32)
 * and (4,4,16) winners are nf=2-3; "fewest stages wins"), and deeper factorizations
 * explode the per-stage variant axis (3^(nf-1) combos × timing) without ever winning.
 * Capping kills the blowup — e.g. N=256=2^8 has an 8-factor path with 3^7=2187 variants. */
#ifndef VFFT_RFFT_CAL_MAX_STAGES
#define VFFT_RFFT_CAL_MAX_STAGES 5
#endif

typedef struct { int nf; int factors[STRIDE_MAX_STAGES]; } _vfft_rfft_facz_t;

/* ordered stage multisets of `rem` (product == rem) over stage-coverable
 * radixes, each terminated by `leaf`. Appends to fz[]/*nfz (capped at `cap`). */
static void _vfft_rfft_enum(int rem, int *pre, int depth, int leaf, int maxnf,
                            const int *stage_ok, _vfft_rfft_facz_t *fz, int *nfz, int cap)
{
    if (depth + 1 >= maxnf) return;                 /* leave room for the leaf */
    if (rem == 1) {
        if (*nfz >= cap) return;
        _vfft_rfft_facz_t *f = &fz[(*nfz)++];
        f->nf = depth + 1;
        for (int i = 0; i < depth; i++) f->factors[i] = pre[i];
        f->factors[depth] = leaf;
        return;
    }
    for (int r = 2; r <= VFFT_RFFT_MAX_RADIX; r++) {
        if (!stage_ok[r] || rem % r != 0) continue;
        pre[depth] = r;
        _vfft_rfft_enum(rem / r, pre, depth + 1, leaf, maxnf, stage_ok, fz, nfz, cap);
    }
}

/* Calibrate the rfft cell (N, K) over `reg`'s codelet coverage. On success
 * returns 0 and fills `out` (N,K,nf,factors,variants,best_ns,use_dif_forward=0);
 * returns -1 if N is not rfft-coverable, allocation fails, or nothing survives
 * the consistency gate. Pure (no I/O); the caller persists `out` to wisdom. */
static int vfft_rfft_calibrate(int N, size_t K, const rfft_codelets_t *reg,
                               vfft_proto_wisdom_entry_t *out)
{
    int leaf_ok[VFFT_RFFT_MAX_RADIX + 1], stage_ok[VFFT_RFFT_MAX_RADIX + 1];
    for (int r = 0; r <= VFFT_RFFT_MAX_RADIX; r++) {
        leaf_ok[r]  = (r >= 2 && reg->r2cf[r] != NULL);                        /* leaf needs r2cf  */
        stage_ok[r] = (r >= 2 && reg->r2cf[r] != NULL && reg->hc2hc[r] != NULL);/* stage needs both */
    }

    _vfft_rfft_facz_t fz[VFFT_RFFT_CAL_MAX_FACZ]; int nfz = 0;
    int pre[STRIDE_MAX_STAGES];
    for (int leaf = 2; leaf <= VFFT_RFFT_MAX_RADIX; leaf++) {
        if (!leaf_ok[leaf] || N % leaf != 0) continue;
        _vfft_rfft_enum(N / leaf, pre, 0, leaf, VFFT_RFFT_CAL_MAX_STAGES, stage_ok, fz, &nfz, VFFT_RFFT_CAL_MAX_FACZ);
    }
    if (nfz == 0) return -1;

    size_t total = (size_t)N * K;
    double *x = NULL, *ref = NULL, *buf = NULL;
    if (vfft_proto_posix_memalign((void **)&x,   64, total * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&ref, 64, total * sizeof(double)) ||
        vfft_proto_posix_memalign((void **)&buf, 64, total * sizeof(double))) {
        vfft_proto_aligned_free(x); vfft_proto_aligned_free(ref); vfft_proto_aligned_free(buf);
        return -1;
    }
    for (size_t i = 0; i < total; i++)
        x[i] = (double)((i * 2654435761u) & 0xffff) / 65536.0 - 0.5;   /* deterministic */

    /* reference = the seed factorization on the (roundtrip-validated) engine. */
    rfft_plan_t *dp = rfft_plan_create_ex(N, K, fz[0].factors, fz[0].nf, NULL, reg);
    if (!dp) { vfft_proto_aligned_free(x); vfft_proto_aligned_free(ref); vfft_proto_aligned_free(buf); return -1; }
    memset(ref, 0, total * sizeof(double));
    rfft_execute_fwd_packed(dp, x, ref);
    rfft_plan_destroy(dp);

    int best_nf = 0, best_f[STRIDE_MAX_STAGES], best_v[STRIDE_MAX_STAGES];
    double best_ns = 1e18;

    for (int fi = 0; fi < nfz; fi++) {
        const _vfft_rfft_facz_t *f = &fz[fi];
        int ncomb = f->nf - 1;                 /* hc2hc combine stages d=0..nf-2 */
        int nvar = 1; for (int i = 0; i < ncomb; i++) nvar *= 3;
        for (int vc = 0; vc < nvar; vc++) {
            int variant[STRIDE_MAX_STAGES]; memset(variant, 0, sizeof variant);
            int t = vc; for (int d = 0; d < ncomb; d++) { variant[d] = t % 3; t /= 3; }
            int ok = 1;                         /* skip absent optional variants */
            for (int d = 0; d < ncomb; d++) {
                int r = f->factors[d];
                if (variant[d] == 1 && !reg->hc2hc_log3[r]) { ok = 0; break; }
                if (variant[d] == 2 && !reg->hc2hc_rng[r])  { ok = 0; break; }
            }
            if (!ok) continue;

            rfft_plan_t *p = rfft_plan_create_ex(N, K, f->factors, f->nf, variant, reg);
            if (!p) continue;

            memset(buf, 0, total * sizeof(double));        /* consistency gate */
            rfft_execute_fwd_packed(p, x, buf);
            double md = 0; for (size_t i = 0; i < total; i++) { double e = fabs(buf[i] - ref[i]); if (e > md) md = e; }
            if (md > 1e-9) { rfft_plan_destroy(p); continue; }

            int reps = (int)(2e6 / (double)(total + 1)); if (reps < 20) reps = 20; if (reps > 100000) reps = 100000;
            for (int w = 0; w < 10; w++) rfft_execute_fwd_packed(p, x, buf);
            double ns = 1e18;
            for (int tr = 0; tr < 5; tr++) {
                double t0 = vfft_proto_now_ns();
                for (int i = 0; i < reps; i++) rfft_execute_fwd_packed(p, x, buf);
                double e = (vfft_proto_now_ns() - t0) / reps; if (e < ns) ns = e;
            }
            rfft_plan_destroy(p);
            if (ns < best_ns) {
                best_ns = ns; best_nf = f->nf;
                memcpy(best_f, f->factors, sizeof best_f);
                memcpy(best_v, variant, sizeof best_v);
            }
        }
    }

    vfft_proto_aligned_free(x); vfft_proto_aligned_free(ref); vfft_proto_aligned_free(buf);
    if (best_nf == 0) return -1;

    memset(out, 0, sizeof *out);
    out->N = N; out->K = K; out->nf = best_nf; out->best_ns = best_ns; out->use_dif_forward = 0;
    for (int s = 0; s < best_nf; s++) { out->factors[s] = best_f[s]; out->variants[s] = best_v[s]; }
    return 0;
}

#endif /* VFFT_RFFT_CALIBRATE_H */
