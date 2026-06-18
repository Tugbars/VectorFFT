/* bench_r2c_opt_inner.c — DECISIVE PROTOTYPE: does running the inner c2c with the
 * WISDOM-BEST factorization (any shape) + a CORRECT general Hermitian recombine
 * (reorder digit-reversed Z -> natural, then contiguous fold) beat the EXISTING
 * stride-r2c path (which forces the inner to (16,8)) and close the gap to MKL at
 * high K?  N=256, K in {32,64,128,256}, single thread (MKL pinned), core-2 affin.
 *
 * Three engines, all REAL forward (r2c), transform-major real input x[n*K+lane]:
 *   1) opt_inner : pack reals -> inner c2c(128,K) via WISDOM-BEST factorization
 *                  (vfft_proto_auto_plan, e.g. (4,4,8) at K=256) -> REORDER the
 *                  digit-reversed Z to natural order -> CONTIGUOUS Hermitian fold.
 *                  This is correct for ANY inner factorization (the new variant).
 *   2) existing  : the constrained stride-r2c path (core/r2c.h), inner forced to
 *                  (16,8) (the only guard-whitelisted multi-stage N=128 shape).
 *   3) MKL r2c   : DFTI_REAL, COMPLEX_COMPLEX, NOT_INPLACE, transform-major.
 *
 * CORRECTNESS GATE per cell: opt_inner out_re[k*K]/out_im[k*K] (lane 0), k=0..N/2,
 * vs a direct reference DFT.  Abort the cell if max abs err >= 1e-9.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_opt_inner.c --mkl --compile
 * Run  : PATH += MKL bin + C:\mingw152\mingw64\bin, then run the .exe.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

#include "core/executor.h"
#include "core/env.h"                  /* stride_env_init + stride_pin_thread */
#include "core/planner.h"
#include "core/dp_planner.h"           /* vfft_proto_now_ns */
#include "core/proto_stride_compat.h"  /* threads pool + STRIDE_ALIGNED_ALLOC, before r2c.h */
#include "core/r2c.h"                  /* existing stride-r2c path */
#include "generator/generated/registry.h"

#define PIN_CORE 2
#define BEST_OF  15

static double *alloc_d(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static int reps_for(size_t total) {
    int r = (int)(4e6 / (total + 1));
    if (r < 30) r = 30; if (r > 200000) r = 200000;
    return r;
}

/* ── digit-reversal permutation (replica of core/r2c.h _r2c_compute_perm) ──
 * For DIT forward: DFT freq f lives at scratch slot perm[f]. */
static void compute_perm(const int *factors, int nf, int N, int *perm)
{
    for (int n = 0; n < N; n++) {
        int idx = n, rev = 0, radix_product = 1;
        for (int s = 0; s < nf; s++) {
            int R = factors[s];
            int digit = idx % R;
            idx /= R;
            rev += digit * (N / (radix_product * R));
            radix_product *= R;
        }
        perm[n] = rev;
    }
}

/* ── the NEW variant: pack + inner c2c + reorder + contiguous recombine ──
 * zre/zim: split-complex inner scratch, layout [row*K + lane], 128 complex rows.
 * Zn_re/Zn_im: natural-order Z (reorder target), same layout.
 * out_re/out_im: real-FFT output bins 0..128, layout [bin*K + lane].
 * If ro_ns / re_ns non-NULL, the inner-exec / (reorder+recombine) costs are
 * measured separately (best-of) for the time breakdown. */
static void opt_inner_run(stride_plan_t *inner, const double *x,
                          double *zre, double *zim,
                          double *Zn_re, double *Zn_im,
                          double *out_re, double *out_im,
                          const int *perm, const double *tw_re, const double *tw_im,
                          int N, int halfN, size_t K)
{
    /* (a) pack: z[j] = x[2j] + i*x[2j+1], split-complex, lane-batched. */
    for (int j = 0; j < halfN; j++) {
        const double *xe = x + (size_t)(2*j)   * K;
        const double *xo = x + (size_t)(2*j+1) * K;
        double *zr = zre + (size_t)j * K;
        double *zi = zim + (size_t)j * K;
        for (size_t l = 0; l < K; l++) { zr[l] = xe[l]; zi[l] = xo[l]; }
    }

    /* (b) inner c2c forward, in place. Output is DIGIT-REVERSED. */
    vfft_proto_execute_fwd(inner, zre, zim, K);

    /* (c) reorder to natural: Zn[f row] = Z[perm[f] row]. */
    for (int f = 0; f < halfN; f++) {
        const double *sr = zre + (size_t)perm[f] * K;
        const double *si = zim + (size_t)perm[f] * K;
        double *dr = Zn_re + (size_t)f * K;
        double *di = Zn_im + (size_t)f * K;
        for (size_t l = 0; l < K; l++) { dr[l] = sr[l]; di[l] = si[l]; }
    }

    /* (d) CONTIGUOUS Hermitian recombine, natural-order Z, bins 0..N/2.
     * z[j]=x[2j]+i*x[2j+1], Z=FFT_128(z).  W=exp(-2pi i k/256), c=cos,s=sin:
     *   Er=0.5(Zr[k]+Zr[hf-k]); Ei=0.5(Zi[k]-Zi[hf-k]);
     *   Or=0.5(Zr[k]-Zr[hf-k]); Oi=0.5(Zi[k]+Zi[hf-k]);
     * Classic identity X[k]=E + W_N^k*(-i*O), W=c - i*s (exp(-i th)), -i*O=(Oi,-Or):
     *   X[k].re = Er + (c*Oi - s*Or);  X[k].im = Ei + (-c*Or - s*Oi);
     * (verified vs reference DFT — candidate A from the task brief was wrong-signed).
     * tw_re[k]=cos(-2pi k/N)=c, tw_im[k]=sin(-2pi k/N)=-s. */
    {
        /* DC + Nyquist (k=0): X[0]=Zr[0]+Zi[0] (re), 0 (im);
         *                     X[hf]=Zr[0]-Zi[0] (re), 0 (im). */
        const double *Z0r = Zn_re, *Z0i = Zn_im;
        double *o0r = out_re, *o0i = out_im;
        double *onr = out_re + (size_t)halfN * K, *oni = out_im + (size_t)halfN * K;
        for (size_t l = 0; l < K; l++) {
            o0r[l] = Z0r[l] + Z0i[l]; o0i[l] = 0.0;
            onr[l] = Z0r[l] - Z0i[l]; oni[l] = 0.0;
        }
    }
    for (int k = 1; k < halfN; k++) {
        int mk = halfN - k;
        const double *Zk_r = Zn_re + (size_t)k  * K, *Zk_i = Zn_im + (size_t)k  * K;
        const double *Zm_r = Zn_re + (size_t)mk * K, *Zm_i = Zn_im + (size_t)mk * K;
        double *or_ = out_re + (size_t)k * K, *oi_ = out_im + (size_t)k * K;
        double c = tw_re[k];            /* cos(-2pi k/N) = cos(2pi k/N) */
        double s = -tw_im[k];           /* tw_im[k]=sin(-2pi k/N)=-sin -> s=sin(2pi k/N) */
        for (size_t l = 0; l < K; l++) {
            double zr = Zk_r[l], zi = Zk_i[l], mr = Zm_r[l], mi = Zm_i[l];
            double Er = 0.5*(zr + mr), Ei = 0.5*(zi - mi);
            double Or = 0.5*(zr - mr), Oi = 0.5*(zi + mi);
            or_[l] = Er + (c*Oi - s*Or);
            oi_[l] = Ei + (-c*Or - s*Oi);
        }
    }
}

/* FUSED variant: NO separate reorder buffer. Contiguous-output recombine that
 * reads the digit-reversed Z DIRECTLY via perm[] (scattered read), eliminating
 * the extra natural-order copy pass. Z[f] is at scratch row perm[f]. */
static void opt_recombine_fused(const double *zre, const double *zim,
                                double *out_re, double *out_im,
                                const int *perm, const double *tw_re, const double *tw_im,
                                int halfN, size_t K)
{
    {
        const double *Z0r = zre + (size_t)perm[0]*K, *Z0i = zim + (size_t)perm[0]*K;
        double *o0r = out_re, *o0i = out_im;
        double *onr = out_re + (size_t)halfN*K, *oni = out_im + (size_t)halfN*K;
        for (size_t l = 0; l < K; l++) {
            o0r[l] = Z0r[l] + Z0i[l]; o0i[l] = 0.0;
            onr[l] = Z0r[l] - Z0i[l]; oni[l] = 0.0;
        }
    }
    for (int k = 1; k < halfN; k++) {
        int mk = halfN - k;
        const double *Zk_r = zre + (size_t)perm[k]  * K, *Zk_i = zim + (size_t)perm[k]  * K;
        const double *Zm_r = zre + (size_t)perm[mk] * K, *Zm_i = zim + (size_t)perm[mk] * K;
        double *or_ = out_re + (size_t)k * K, *oi_ = out_im + (size_t)k * K;
        double c = tw_re[k], s = -tw_im[k];
        for (size_t l = 0; l < K; l++) {
            double zr = Zk_r[l], zi = Zk_i[l], mr = Zm_r[l], mi = Zm_i[l];
            double Er = 0.5*(zr + mr), Ei = 0.5*(zi - mi);
            double Or = 0.5*(zr - mr), Oi = 0.5*(zi + mi);
            or_[l] = Er + (c*Oi - s*Or);
            oi_[l] = Ei + (-c*Or - s*Oi);
        }
    }
}

/* AVX2-VECTORIZED fused recombine. Same perm-driven natural-frequency reads as
 * opt_recombine_fused (Z[k] at scratch row perm[k]), but vectorized OVER THE K
 * LANES (4 doubles per __m256d). The twiddle c,s for a given frequency k is a
 * SCALAR broadcast hoisted outside the lane loop. Math identical to the scalar:
 *   Er=0.5(zr+mr); Ei=0.5(zi-mi); Or=0.5(zr-mr); Oi=0.5(zi+mi);
 *   c=tw_re[k]; s=-tw_im[k];
 *   X[k].re = Er + (c*Oi - s*Or);  X[k].im = Ei + (-c*Or - s*Oi);
 * Structure mirrors core/r2c.h:272-314 (lane-vectorized recombine, hoisted twiddle
 * broadcasts) but keeps the per-frequency perm read instead of the slot-pair walk.
 * K is a multiple of 8, so the K-lane loop is fully vectorized (no scalar tail). */
static void opt_recombine_fused_avx2(const double *zre, const double *zim,
                                     double *out_re, double *out_im,
                                     const int *perm, const double *tw_re, const double *tw_im,
                                     int halfN, size_t K)
{
    /* DC + Nyquist (k=0). */
    {
        const double *Z0r = zre + (size_t)perm[0]*K, *Z0i = zim + (size_t)perm[0]*K;
        double *o0r = out_re, *o0i = out_im;
        double *onr = out_re + (size_t)halfN*K, *oni = out_im + (size_t)halfN*K;
        const __m256d zero = _mm256_setzero_pd();
        size_t l = 0;
        for (; l + 4 <= K; l += 4) {
            __m256d zr = _mm256_load_pd(Z0r + l), zi = _mm256_load_pd(Z0i + l);
            _mm256_store_pd(o0r + l, _mm256_add_pd(zr, zi));
            _mm256_store_pd(o0i + l, zero);
            _mm256_store_pd(onr + l, _mm256_sub_pd(zr, zi));
            _mm256_store_pd(oni + l, zero);
        }
        for (; l < K; l++) { o0r[l]=Z0r[l]+Z0i[l]; o0i[l]=0.0; onr[l]=Z0r[l]-Z0i[l]; oni[l]=0.0; }
    }
    const __m256d half_v = _mm256_set1_pd(0.5);
    const __m256d sign   = _mm256_set1_pd(-0.0);
    for (int k = 1; k < halfN; k++) {
        int mk = halfN - k;
        const double *Zk_r = zre + (size_t)perm[k]  * K, *Zk_i = zim + (size_t)perm[k]  * K;
        const double *Zm_r = zre + (size_t)perm[mk] * K, *Zm_i = zim + (size_t)perm[mk] * K;
        double *or_ = out_re + (size_t)k * K, *oi_ = out_im + (size_t)k * K;
        double c = tw_re[k], s = -tw_im[k];                 /* hoisted scalar twiddle */
        const __m256d vc = _mm256_set1_pd(c);
        const __m256d vs = _mm256_set1_pd(s);
        size_t l = 0;
        for (; l + 4 <= K; l += 4) {
            __m256d zr = _mm256_load_pd(Zk_r + l), zi = _mm256_load_pd(Zk_i + l);
            __m256d mr = _mm256_load_pd(Zm_r + l), mi = _mm256_load_pd(Zm_i + l);
            __m256d Er = _mm256_mul_pd(_mm256_add_pd(zr, mr), half_v);
            __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(zi, mi), half_v);
            __m256d Or = _mm256_mul_pd(_mm256_sub_pd(zr, mr), half_v);
            __m256d Oi = _mm256_mul_pd(_mm256_add_pd(zi, mi), half_v);
            /* Tr = c*Oi - s*Or ; Ti = -(c*Or + s*Oi) */
            __m256d Tr = _mm256_fmsub_pd(vc, Oi, _mm256_mul_pd(vs, Or));
            __m256d Ti = _mm256_xor_pd(sign, _mm256_fmadd_pd(vc, Or, _mm256_mul_pd(vs, Oi)));
            _mm256_store_pd(or_ + l, _mm256_add_pd(Er, Tr));
            _mm256_store_pd(oi_ + l, _mm256_add_pd(Ei, Ti));
        }
        for (; l < K; l++) {
            double zr = Zk_r[l], zi = Zk_i[l], mr = Zm_r[l], mi = Zm_i[l];
            double Er = 0.5*(zr + mr), Ei = 0.5*(zi - mi);
            double Or = 0.5*(zr - mr), Oi = 0.5*(zi + mi);
            or_[l] = Er + (c*Oi - s*Or);
            oi_[l] = Ei + (-c*Or - s*Oi);
        }
    }
}

/* (c)+(d) only: reorder + contiguous recombine (inner already executed into zre/zim). */
static void opt_reorder_recombine(const double *zre, const double *zim,
                                  double *Zn_re, double *Zn_im,
                                  double *out_re, double *out_im,
                                  const int *perm, const double *tw_re, const double *tw_im,
                                  int halfN, size_t K)
{
    for (int f = 0; f < halfN; f++) {
        const double *sr = zre + (size_t)perm[f] * K;
        const double *si = zim + (size_t)perm[f] * K;
        double *dr = Zn_re + (size_t)f * K;
        double *di = Zn_im + (size_t)f * K;
        for (size_t l = 0; l < K; l++) { dr[l] = sr[l]; di[l] = si[l]; }
    }
    {
        const double *Z0r = Zn_re, *Z0i = Zn_im;
        double *o0r = out_re, *o0i = out_im;
        double *onr = out_re + (size_t)halfN * K, *oni = out_im + (size_t)halfN * K;
        for (size_t l = 0; l < K; l++) {
            o0r[l] = Z0r[l] + Z0i[l]; o0i[l] = 0.0;
            onr[l] = Z0r[l] - Z0i[l]; oni[l] = 0.0;
        }
    }
    for (int k = 1; k < halfN; k++) {
        int mk = halfN - k;
        const double *Zk_r = Zn_re + (size_t)k  * K, *Zk_i = Zn_im + (size_t)k  * K;
        const double *Zm_r = Zn_re + (size_t)mk * K, *Zm_i = Zn_im + (size_t)mk * K;
        double *or_ = out_re + (size_t)k * K, *oi_ = out_im + (size_t)k * K;
        double c = tw_re[k];
        double s = -tw_im[k];
        for (size_t l = 0; l < K; l++) {
            double zr = Zk_r[l], zi = Zk_i[l], mr = Zm_r[l], mi = Zm_i[l];
            double Er = 0.5*(zr + mr), Ei = 0.5*(zi - mi);
            double Or = 0.5*(zr - mr), Oi = 0.5*(zi + mi);
            or_[l] = Er + (c*Oi - s*Or);
            oi_[l] = Ei + (-c*Or - s*Oi);
        }
    }
}

static double bench_opt(stride_plan_t *inner, const double *x,
                        double *zre, double *zim, double *Zn_re, double *Zn_im,
                        double *out_re, double *out_im,
                        const int *perm, const double *tw_re, const double *tw_im,
                        int N, int halfN, size_t K, size_t total) {
    for (int w = 0; w < 10; w++)
        opt_inner_run(inner, x, zre, zim, Zn_re, Zn_im, out_re, out_im,
                      perm, tw_re, tw_im, N, halfN, K);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            opt_inner_run(inner, x, zre, zim, Zn_re, Zn_im, out_re, out_im,
                          perm, tw_re, tw_im, N, halfN, K);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}
/* FUSED total: pack + inner c2c + perm-driven contiguous recombine (NO reorder buffer). */
static void opt_inner_run_fused(stride_plan_t *inner, const double *x,
                                double *zre, double *zim,
                                double *out_re, double *out_im,
                                const int *perm, const double *tw_re, const double *tw_im,
                                int halfN, size_t K) {
    for (int j = 0; j < halfN; j++) {
        const double *xe = x + (size_t)(2*j)*K, *xo = x + (size_t)(2*j+1)*K;
        double *zr = zre + (size_t)j*K, *zi = zim + (size_t)j*K;
        for (size_t l=0;l<K;l++){zr[l]=xe[l];zi[l]=xo[l];}
    }
    vfft_proto_execute_fwd(inner, zre, zim, K);
    opt_recombine_fused(zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
}
static double bench_opt_fused(stride_plan_t *inner, const double *x,
                              double *zre, double *zim, double *out_re, double *out_im,
                              const int *perm, const double *tw_re, const double *tw_im,
                              int halfN, size_t K, size_t total) {
    for (int w = 0; w < 10; w++)
        opt_inner_run_fused(inner, x, zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            opt_inner_run_fused(inner, x, zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}

/* FUSED-AVX2 total: pack + inner c2c + AVX2 perm-driven contiguous recombine. */
static void opt_inner_run_fused_avx2(stride_plan_t *inner, const double *x,
                                     double *zre, double *zim,
                                     double *out_re, double *out_im,
                                     const int *perm, const double *tw_re, const double *tw_im,
                                     int halfN, size_t K) {
    for (int j = 0; j < halfN; j++) {
        const double *xe = x + (size_t)(2*j)*K, *xo = x + (size_t)(2*j+1)*K;
        double *zr = zre + (size_t)j*K, *zi = zim + (size_t)j*K;
        for (size_t l=0;l<K;l++){zr[l]=xe[l];zi[l]=xo[l];}
    }
    vfft_proto_execute_fwd(inner, zre, zim, K);
    opt_recombine_fused_avx2(zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
}
static double bench_opt_fused_avx2(stride_plan_t *inner, const double *x,
                                   double *zre, double *zim, double *out_re, double *out_im,
                                   const int *perm, const double *tw_re, const double *tw_im,
                                   int halfN, size_t K, size_t total) {
    for (int w = 0; w < 10; w++)
        opt_inner_run_fused_avx2(inner, x, zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            opt_inner_run_fused_avx2(inner, x, zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}

/* recombine-ONLY timing (zre/zim primed with inner FFT output): scalar fused. */
static double bench_recombine_fused_scalar(const double *zre, const double *zim,
                                           double *out_re, double *out_im,
                                           const int *perm, const double *tw_re, const double *tw_im,
                                           int halfN, size_t K, size_t total) {
    for (int w = 0; w < 10; w++)
        opt_recombine_fused(zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            opt_recombine_fused(zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}
/* recombine-ONLY timing: AVX2 fused. */
static double bench_recombine_fused_avx2(const double *zre, const double *zim,
                                         double *out_re, double *out_im,
                                         const int *perm, const double *tw_re, const double *tw_im,
                                         int halfN, size_t K, size_t total) {
    for (int w = 0; w < 10; w++)
        opt_recombine_fused_avx2(zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            opt_recombine_fused_avx2(zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}

/* pure c2c only (execute, NO pack) — to split inner-only into pack vs FFT */
static double bench_c2c_only(stride_plan_t *inner, double *zre, double *zim,
                             size_t K, size_t total) {
    for (int w = 0; w < 10; w++) vfft_proto_execute_fwd(inner, zre, zim, K);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(inner, zre, zim, K);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}
/* inner c2c only (pack + execute) — for the breakdown */
static double bench_opt_inner_only(stride_plan_t *inner, const double *x,
                                   double *zre, double *zim,
                                   int halfN, size_t K, size_t total) {
    for (int w = 0; w < 10; w++) {
        for (int j = 0; j < halfN; j++) {
            const double *xe = x + (size_t)(2*j)*K, *xo = x + (size_t)(2*j+1)*K;
            double *zr = zre + (size_t)j*K, *zi = zim + (size_t)j*K;
            for (size_t l=0;l<K;l++){zr[l]=xe[l];zi[l]=xo[l];}
        }
        vfft_proto_execute_fwd(inner, zre, zim, K);
    }
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) {
            for (int j = 0; j < halfN; j++) {
                const double *xe = x + (size_t)(2*j)*K, *xo = x + (size_t)(2*j+1)*K;
                double *zr = zre + (size_t)j*K, *zi = zim + (size_t)j*K;
                for (size_t l=0;l<K;l++){zr[l]=xe[l];zi[l]=xo[l];}
            }
            vfft_proto_execute_fwd(inner, zre, zim, K);
        }
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}
/* reorder+recombine only — for the breakdown (zre/zim primed with FFT output) */
static double bench_opt_rr_only(const double *zre, const double *zim,
                                double *Zn_re, double *Zn_im,
                                double *out_re, double *out_im,
                                const int *perm, const double *tw_re, const double *tw_im,
                                int halfN, size_t K, size_t total) {
    for (int w = 0; w < 10; w++)
        opt_reorder_recombine(zre, zim, Zn_re, Zn_im, out_re, out_im, perm, tw_re, tw_im, halfN, K);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            opt_reorder_recombine(zre, zim, Zn_re, Zn_im, out_re, out_im, perm, tw_re, tw_im, halfN, K);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}

static double bench_stride_r2c(const stride_plan_t *p, const double *in,
                               double *re, double *im, size_t total) {
    for (int w = 0; w < 10; w++) stride_execute_r2c(p, in, re, im);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) stride_execute_r2c(p, in, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE h, const double *xin, double *cce, size_t total) {
    for (int w = 0; w < 10; w++) DftiComputeForward(h, (void *)xin, cce);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) DftiComputeForward(h, (void *)xin, cce);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    stride_env_init();
    if (stride_pin_thread(PIN_CORE) != 0)
        fprintf(stderr, "warn: pin cpu%d failed\n", PIN_CORE);
    mkl_set_num_threads(1);

    const int N = 256, halfN = N / 2;
    const size_t Ks[] = {32, 64, 128, 256};
    const int nK = (int)(sizeof Ks / sizeof Ks[0]);

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    /* c2c wisdom for the inner (N=128). */
    vfft_proto_wisdom_t wis; int have_wis =
        (vfft_proto_wisdom_load(&wis, "../src/dag-fft-compiler/generator/generated/spike_wisdom.txt") == 0);
    if (!have_wis)
        have_wis = (vfft_proto_wisdom_load(&wis,
                    "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt") == 0);
    printf("# c2c wisdom load: %s\n", have_wis ? "OK" : "FAILED (factorizer-default inner)");

    /* N-point fold twiddle table: tw_re[k]=cos(-2pi k/N), tw_im[k]=sin(-2pi k/N). */
    double *tw_re = alloc_d(halfN), *tw_im = alloc_d(halfN);
    _r2c_init_twiddles(N, tw_re, tw_im);

    printf("=== r2c opt-inner (wisdom-best + reorder + contiguous recombine) "
           "vs existing stride-r2c (16,8) vs MKL  (N=256, ST, cpu%d, best-of-%d) ===\n",
           PIN_CORE, BEST_OF);
    printf("# opt_fused      = wisdom-inner + perm-driven SCALAR recombine (NO reorder buffer)\n");
    printf("# opt_fused_AVX2 = wisdom-inner + perm-driven AVX2 lane-vectorized recombine (DECISIVE)\n");
    printf("%-5s %14s %14s %12s %11s %10s %10s   %s\n",
           "K", "fused_avx2_ns", "fused_scl_ns", "mkl_ns", "avx2/mkl",
           "recomb_scl", "recomb_avx2", "breakdown(inner-only ns | recomb scl->avx2 speedup)");
    printf("------+--------------+--------------+------------+-----------+----------+-----------+----------------------\n");

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki];
        size_t total = (size_t)N * K;

        /* input: lane-batched real x[n*K+lane], shared by all engines. */
        double *x = alloc_d(total);
        srand(7 + (int)K);
        for (size_t i = 0; i < total; i++) x[i] = (double)rand() / RAND_MAX * 2 - 1;

        /* ===== opt_inner: WISDOM-BEST inner c2c(128,K) ===== */
        stride_plan_t *inner = vfft_proto_auto_plan(halfN, K, &reg, have_wis ? &wis : NULL);
        if (!inner) { printf("%-5zu  auto_plan(128,%zu) NULL\n", K, K); vfft_proto_aligned_free(x); continue; }

        /* report the chosen inner factorization */
        char fs[64]; size_t pp = 0;
        for (int s = 0; s < inner->num_stages; s++)
            pp += (size_t)snprintf(fs+pp, sizeof fs-pp, "%s%d", s?",":"", inner->factors[s]);
        printf("# K=%-4zu inner c2c(128) wisdom-best = (%s)  [%d stage(s)]%s\n",
               K, fs, inner->num_stages,
               (inner->num_stages==2 && inner->factors[0]==16 && inner->factors[1]==8)
                 ? "  <-- (16,8): same as existing" : "");

        int perm[256];
        compute_perm(inner->factors, inner->num_stages, halfN, perm);

        double *zre = alloc_d((size_t)halfN * K), *zim = alloc_d((size_t)halfN * K);
        double *Znr = alloc_d((size_t)halfN * K), *Zni = alloc_d((size_t)halfN * K);
        double *oor = alloc_d((size_t)(halfN+1) * K), *ooi = alloc_d((size_t)(halfN+1) * K);

        /* ===== correctness gate: lane 0 vs reference DFT, k=0..N/2 ===== */
        memset(oor, 0, (size_t)(halfN+1)*K*sizeof(double));
        memset(ooi, 0, (size_t)(halfN+1)*K*sizeof(double));
        opt_inner_run(inner, x, zre, zim, Znr, Zni, oor, ooi, perm, tw_re, tw_im, N, halfN, K);
        double maxerr = 0.0;
        for (int k = 0; k <= halfN; k++) {
            double ref_re = 0.0, ref_im = 0.0;
            for (int n = 0; n < N; n++) {
                double xn = x[(size_t)n * K + 0];
                double ang = -2.0 * M_PI * (double)k * (double)n / (double)N;
                ref_re += xn * cos(ang);
                ref_im += xn * sin(ang);
            }
            double gr = oor[(size_t)k * K + 0], gi = ooi[(size_t)k * K + 0];
            double er = fabs(gr - ref_re), ei = fabs(gi - ref_im);
            if (er > maxerr) maxerr = er;
            if (ei > maxerr) maxerr = ei;
        }
        /* also gate the FUSED (perm-driven, no reorder buffer) variant */
        memset(oor, 0, (size_t)(halfN+1)*K*sizeof(double));
        memset(ooi, 0, (size_t)(halfN+1)*K*sizeof(double));
        opt_inner_run_fused(inner, x, zre, zim, oor, ooi, perm, tw_re, tw_im, halfN, K);
        double maxerr_f = 0.0;
        for (int k = 0; k <= halfN; k++) {
            double ref_re = 0.0, ref_im = 0.0;
            for (int n = 0; n < N; n++) {
                double xn = x[(size_t)n * K + 0];
                double ang = -2.0 * M_PI * (double)k * (double)n / (double)N;
                ref_re += xn * cos(ang); ref_im += xn * sin(ang);
            }
            double er = fabs(oor[(size_t)k*K] - ref_re), ei = fabs(ooi[(size_t)k*K] - ref_im);
            if (er > maxerr_f) maxerr_f = er; if (ei > maxerr_f) maxerr_f = ei;
        }
        /* gate the FUSED-AVX2 variant (the decisive one) */
        memset(oor, 0, (size_t)(halfN+1)*K*sizeof(double));
        memset(ooi, 0, (size_t)(halfN+1)*K*sizeof(double));
        opt_inner_run_fused_avx2(inner, x, zre, zim, oor, ooi, perm, tw_re, tw_im, halfN, K);
        double maxerr_a = 0.0;
        for (int k = 0; k <= halfN; k++) {
            double ref_re = 0.0, ref_im = 0.0;
            for (int n = 0; n < N; n++) {
                double xn = x[(size_t)n * K + 0];
                double ang = -2.0 * M_PI * (double)k * (double)n / (double)N;
                ref_re += xn * cos(ang); ref_im += xn * sin(ang);
            }
            double er = fabs(oor[(size_t)k*K] - ref_re), ei = fabs(ooi[(size_t)k*K] - ref_im);
            if (er > maxerr_a) maxerr_a = er; if (ei > maxerr_a) maxerr_a = ei;
        }
        if (maxerr >= 1e-9 || maxerr_f >= 1e-9 || maxerr_a >= 1e-9) {
            printf("%-5zu  *** CORRECTNESS FAIL: reord err=%.3e fused err=%.3e avx2 err=%.3e (ABORT cell) ***\n",
                   K, maxerr, maxerr_f, maxerr_a);
            vfft_proto_aligned_free(x);
            vfft_proto_aligned_free(zre); vfft_proto_aligned_free(zim);
            vfft_proto_aligned_free(Znr); vfft_proto_aligned_free(Zni);
            vfft_proto_aligned_free(oor); vfft_proto_aligned_free(ooi);
            stride_plan_destroy(inner);
            continue;
        }

        /* ===== existing stride-r2c path: inner forced to (16,8) ===== */
        int ef[2] = {16, 8}, ev[2] = {2, 2};
        stride_plan_t *einner = vfft_proto_plan_create_ex(halfN, K, ef, ev, 2, 0, &reg);
        stride_plan_t *ep = einner ? stride_r2c_plan(N, K, K, einner) : NULL;
        double *esr = alloc_d(total), *esi = alloc_d(total);
        int existing_ok = (ep != NULL);

        /* ===== MKL r2c (transform-major) ===== */
        DFTI_DESCRIPTOR_HANDLE h = 0; int mkl_ok = 0;
        double *xin = alloc_d(total), *cce = alloc_d((size_t)(halfN + 1) * K * 2);
        for (size_t t = 0; t < K; t++)
            for (int n = 0; n < N; n++) xin[t * N + n] = x[(size_t)n * K + t];
        DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N);
        DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)(halfN + 1));
        mkl_ok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);

        (void)existing_ok; (void)esr; (void)esi;
        /* ===== time everything ===== */
        double fa_ns = bench_opt_fused_avx2(inner, x, zre, zim, oor, ooi,
                                            perm, tw_re, tw_im, halfN, K, total);
        double fs_ns = bench_opt_fused(inner, x, zre, zim, oor, ooi,
                                       perm, tw_re, tw_im, halfN, K, total);
        /* breakdown: inner-only (pack+exec) leaves a valid FFT output in
         * zre/zim, then recombine-only (scalar vs AVX2) reads that. */
        double oi_ns = bench_opt_inner_only(inner, x, zre, zim, halfN, K, total);
        double c2c_ns = bench_c2c_only(inner, zre, zim, K, total);  /* FFT, no pack */
        double rs_ns = bench_recombine_fused_scalar(zre, zim, oor, ooi,
                                                    perm, tw_re, tw_im, halfN, K, total);
        double ra_ns = bench_recombine_fused_avx2(zre, zim, oor, ooi,
                                                  perm, tw_re, tw_im, halfN, K, total);
        double m_ns  = mkl_ok ? bench_mkl(h, xin, cce, total) : 0;

        double avx2_over_m = (m_ns > 0 && fa_ns > 0) ? m_ns / fa_ns : 0;
        double recomb_speedup = (ra_ns > 0) ? rs_ns / ra_ns : 0;
        double pack_ns = oi_ns - c2c_ns;
        printf("%-5zu %14.1f %14.1f %12.1f %10.3fx %10.1f %10.1f   pack=%.1f c2c=%.1f recomb=%.1f->%.1f(%.2fx)\n",
               K, fa_ns, fs_ns, m_ns, avx2_over_m, rs_ns, ra_ns,
               pack_ns, c2c_ns, rs_ns, ra_ns, recomb_speedup);

        if (h) DftiFreeDescriptor(&h);
        vfft_proto_aligned_free(x);
        vfft_proto_aligned_free(zre); vfft_proto_aligned_free(zim);
        vfft_proto_aligned_free(Znr); vfft_proto_aligned_free(Zni);
        vfft_proto_aligned_free(oor); vfft_proto_aligned_free(ooi);
        vfft_proto_aligned_free(esr); vfft_proto_aligned_free(esi);
        vfft_proto_aligned_free(xin); vfft_proto_aligned_free(cce);
        if (ep) stride_plan_destroy(ep);   /* frees einner via override_destroy */
        stride_plan_destroy(inner);
    }

    vfft_proto_aligned_free(tw_re); vfft_proto_aligned_free(tw_im);
    if (have_wis) vfft_proto_wisdom_free(&wis);
    printf("\n# avx2/mkl = MKL_ns / fused_avx2_ns  (ratio>1 = WE BEAT MKL).\n");
    printf("# recomb_scl / recomb_avx2 = recombine-ONLY time (zre/zim primed with inner FFT output).\n");
    printf("# breakdown = inner-only(pack+c2c) ns | recombine scalar->avx2 ns (speedup).\n");
    return 0;
}
