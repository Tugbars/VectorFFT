/* bench_r2c_packfuse.c — PACK-FUSION test for the decoupled r2c.
 *
 * Hypothesis: the ~12µs pre-pack memory pass (deinterleave real input x into
 * split-complex z) can be ELIMINATED by having the inner c2c read the raw reals
 * x DIRECTLY, with the right addressing, at zero extra memory traffic.
 *
 * ARCHITECTURAL FINDING (see report): the IN-PLACE n1 codelet
 * radix{R}_n1_fwd_avx2(rio_re, rio_im, tw, tw, ios, me) uses ONE stride `ios`
 * for BOTH load and store, and a SINGLE base index `k` into both rio_re/rio_im.
 * It therefore CANNOT read re-leg at x[(2n)K+l] and im-leg at x[(2n+1)K+l] while
 * writing the packed z @ stride K — load and store strides differ (2K vs K).
 *
 * The cheapest correct way is the OOP leaf codelet
 * radix{R}_n1_oop_fwd_avx2_UG_UG, which has SEPARATE in/out base pointers AND
 * SEPARATE in/out leg+group strides:
 *   in_re  = x,          in_im  = x + K       (the +1 real -> +K)
 *   in_leg_stride  = 2 * stride0              (read x interleaved: leg n at 2n)
 *   in_group_stride = 1                       (lanes contiguous in x)
 *   out_re = zre,        out_im = zim
 *   out_leg_stride = stride0                  (normal packed split-complex)
 *   out_group_stride = 1
 *   me = K
 * One call per stage-0 group; in/out per-group bases are 2*group_base[g] (input,
 * because x is double-rate) and group_base[g] (output). Stages 1..nf-1 then run
 * normally via the generic executor (started at stage 1). No pre-pack pass.
 *
 * We include the radix4 OOP n1 codelet source directly (the inner for N=128
 * K=256 wisdom factorization is (4,4,8), so stage 0 = radix 4). For other
 * stage-0 radixes we fall back to the pre-pack path for that cell.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_packfuse.c --mkl --compile
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
#include "core/env.h"
#include "core/planner.h"
#include "core/dp_planner.h"           /* vfft_proto_now_ns */
#include "core/proto_stride_compat.h"
#include "core/r2c.h"                  /* _r2c_init_twiddles */
#include "generator/generated/registry.h"

/* The OOP pack-fused stage-0 leaf. Included directly so the symbol links
 * without modifying build.py's cached codelet lib (which only covers
 * inplace/rfft/c2r dirs, not oop/). Self-contained TU (only immintrin). */
#include "../src/dag-fft-compiler/codelets/oop/avx2/radix4_n1_oop_avx2.c"

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

static void compute_perm(const int *factors, int nf, int N, int *perm) {
    for (int n = 0; n < N; n++) {
        int idx = n, rev = 0, radix_product = 1;
        for (int s = 0; s < nf; s++) {
            int R = factors[s];
            int digit = idx % R; idx /= R;
            rev += digit * (N / (radix_product * R));
            radix_product *= R;
        }
        perm[n] = rev;
    }
}

/* ── pre-pack reference path (the EXISTING decoupled r2c), for A/B + gate ── */
static void prepack_stage0(const double *x, double *zre, double *zim,
                           int halfN, size_t K) {
    for (int j = 0; j < halfN; j++) {
        const double *xe = x + (size_t)(2*j)*K, *xo = x + (size_t)(2*j+1)*K;
        double *zr = zre + (size_t)j*K, *zi = zim + (size_t)j*K;
        for (size_t l=0;l<K;l++){zr[l]=xe[l];zi[l]=xo[l];}
    }
}

/* ── PACK-FUSED stage 0: read x directly via the OOP leaf, write packed z.
 * Replaces (prepack + in-place stage-0 n1) with a single OOP pass that does
 * NO separate memory copy of x. Only valid when stage-0 radix == 4 (the
 * codelet we linked). Returns 1 if applied, 0 if unsupported. */
static int packfuse_stage0(const stride_plan_t *plan, const double *x,
                           double *zre, double *zim, size_t K) {
    const stride_stage_t *st0 = &plan->stages[0];
    /* Pack-fusion is a DIT-leaf technique: the no-twiddle leaf is stage 0
     * and reads the input directly. DIF puts the no-twiddle stage LAST, so
     * the technique doesn't apply — fall back. (Also requires an OOP leaf;
     * we linked radix-4.) */
    if (plan->use_dif_forward) return 0;
    if (st0->radix != 4) return 0;
    const size_t s0 = st0->stride;
    for (int g = 0; g < st0->num_groups; g++) {
        size_t gb = st0->group_base[g];
        radix4_n1_oop_fwd_avx2_UG_UG(
            x + 2*gb,           /* in_re  = x  + double-rate group base */
            x + K + 2*gb,       /* in_im  = x+K + double-rate group base */
            zre + gb, zim + gb, /* out_re/out_im = packed split-complex */
            NULL, NULL,         /* n1: no twiddle */
            2*s0,               /* in_leg_stride  = doubled (interleaved x) */
            1,                  /* in_group_stride (lanes contiguous in x) */
            s0,                 /* out_leg_stride = normal packed */
            1,                  /* out_group_stride */
            K);                 /* me = lanes */
    }
    return 1;
}

/* Run plan stages [s_start .. nf-1] via the generic per-stage loop (a copy of
 * vfft_proto_execute_fwd_generic with a start index — the library generic has
 * no start_stage knob; the Tier-1 specialized path does, but we want the
 * cold-cell-correct path for any wisdom factorization). DIT, T1S/FLAT/LOG3. */
static void exec_from_stage(const stride_plan_t *plan, double *re, double *im,
                            size_t slice_K, int s_start) {
    for (int s = s_start; s < plan->num_stages; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups, R = st->radix;
        for (int g = 0; g < G; g++) {
            double *base_re = re + st->group_base[g];
            double *base_im = im + st->group_base[g];
            if (!st->needs_tw[g]) {
                st->n1_fwd(base_re, base_im, base_re, base_im,
                           st->stride, st->stride, slice_K);
                continue;
            }
            double cfr = st->cf0_re[g], cfi = st->cf0_im[g];
            if (st->use_log3) {
                if (cfr != 1.0 || cfi != 0.0)
                    for (int j = 0; j < R; j++) {
                        double *lr = base_re + (size_t)j*st->stride;
                        double *li = base_im + (size_t)j*st->stride;
                        _stride_cmul_scalar_inplace(lr, li, slice_K, cfr, cfi);
                    }
                st->t1_fwd(base_re, base_im, st->grp_tw_re[g], st->grp_tw_im[g],
                           st->stride, slice_K);
                continue;
            }
            if (st->t1s_fwd) {
                if (cfr != 1.0 || cfi != 0.0)
                    _stride_cmul_scalar_inplace(base_re, base_im, slice_K, cfr, cfi);
                st->t1s_fwd(base_re, base_im,
                            st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
                continue;
            }
            /* FLAT */
            if (cfr != 1.0 || cfi != 0.0)
                _stride_cmul_scalar_inplace(base_re, base_im, slice_K, cfr, cfi);
            const int Rm1 = R - 1;
            const double *stw_r = st->tw_scalar_re[g];
            const double *stw_i = st->tw_scalar_im[g];
            double tw_buf_re[63 * VFFT_PROTO_TW_BLOCK_K];
            double tw_buf_im[63 * VFFT_PROTO_TW_BLOCK_K];
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                size_t this_K = slice_K - kb;
                if (this_K > VFFT_PROTO_TW_BLOCK_K) this_K = VFFT_PROTO_TW_BLOCK_K;
                for (int j = 0; j < Rm1; j++) {
                    size_t off = (size_t)j*this_K;
                    _stride_broadcast_2(tw_buf_re+off, tw_buf_im+off, this_K,
                                        stw_r[j], stw_i[j]);
                }
                st->t1_fwd(base_re+kb, base_im+kb, tw_buf_re, tw_buf_im,
                           st->stride, this_K);
            }
        }
    }
}

/* AVX2 perm-driven contiguous Hermitian recombine (lifted from
 * bench_r2c_opt_inner.c — math verified vs reference DFT there). */
static void recombine_avx2(const double *zre, const double *zim,
                           double *out_re, double *out_im,
                           const int *perm, const double *tw_re, const double *tw_im,
                           int halfN, size_t K) {
    {
        const double *Z0r = zre + (size_t)perm[0]*K, *Z0i = zim + (size_t)perm[0]*K;
        double *o0r = out_re, *o0i = out_im;
        double *onr = out_re + (size_t)halfN*K, *oni = out_im + (size_t)halfN*K;
        const __m256d zero = _mm256_setzero_pd();
        size_t l = 0;
        for (; l + 4 <= K; l += 4) {
            __m256d zr = _mm256_load_pd(Z0r+l), zi = _mm256_load_pd(Z0i+l);
            _mm256_store_pd(o0r+l, _mm256_add_pd(zr,zi));
            _mm256_store_pd(o0i+l, zero);
            _mm256_store_pd(onr+l, _mm256_sub_pd(zr,zi));
            _mm256_store_pd(oni+l, zero);
        }
        for (; l < K; l++){o0r[l]=Z0r[l]+Z0i[l];o0i[l]=0;onr[l]=Z0r[l]-Z0i[l];oni[l]=0;}
    }
    const __m256d half_v = _mm256_set1_pd(0.5);
    const __m256d sign   = _mm256_set1_pd(-0.0);
    for (int k = 1; k < halfN; k++) {
        int mk = halfN - k;
        const double *Zk_r = zre + (size_t)perm[k]*K,  *Zk_i = zim + (size_t)perm[k]*K;
        const double *Zm_r = zre + (size_t)perm[mk]*K, *Zm_i = zim + (size_t)perm[mk]*K;
        double *or_ = out_re + (size_t)k*K, *oi_ = out_im + (size_t)k*K;
        double c = tw_re[k], s = -tw_im[k];
        const __m256d vc = _mm256_set1_pd(c), vs = _mm256_set1_pd(s);
        size_t l = 0;
        for (; l + 4 <= K; l += 4) {
            __m256d zr = _mm256_load_pd(Zk_r+l), zi = _mm256_load_pd(Zk_i+l);
            __m256d mr = _mm256_load_pd(Zm_r+l), mi = _mm256_load_pd(Zm_i+l);
            __m256d Er = _mm256_mul_pd(_mm256_add_pd(zr,mr), half_v);
            __m256d Ei = _mm256_mul_pd(_mm256_sub_pd(zi,mi), half_v);
            __m256d Or = _mm256_mul_pd(_mm256_sub_pd(zr,mr), half_v);
            __m256d Oi = _mm256_mul_pd(_mm256_add_pd(zi,mi), half_v);
            __m256d Tr = _mm256_fmsub_pd(vc, Oi, _mm256_mul_pd(vs, Or));
            __m256d Ti = _mm256_xor_pd(sign, _mm256_fmadd_pd(vc, Or, _mm256_mul_pd(vs, Oi)));
            _mm256_store_pd(or_+l, _mm256_add_pd(Er, Tr));
            _mm256_store_pd(oi_+l, _mm256_add_pd(Ei, Ti));
        }
        for (; l < K; l++) {
            double zr=Zk_r[l],zi=Zk_i[l],mr=Zm_r[l],mi=Zm_i[l];
            double Er=0.5*(zr+mr),Ei=0.5*(zi-mi),Or=0.5*(zr-mr),Oi=0.5*(zi+mi);
            or_[l]=Er+(c*Oi-s*Or); oi_[l]=Ei+(-c*Or-s*Oi);
        }
    }
}

/* full PRE-PACK path: prepack + in-place stage0 n1 + stages 1.. + recombine. */
static void run_prepack(const stride_plan_t *inner, const double *x,
                        double *zre, double *zim, double *out_re, double *out_im,
                        const int *perm, const double *tw_re, const double *tw_im,
                        int halfN, size_t K) {
    prepack_stage0(x, zre, zim, halfN, K);
    vfft_proto_execute_fwd(inner, zre, zim, K);  /* full inner FFT in place */
    recombine_avx2(zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
}

/* full PACK-FUSED path: OOP stage0 reads x directly + stages 1.. + recombine.
 * Returns 1 if pack-fusion applied, 0 if it fell back to prepack. */
static int run_packfuse(const stride_plan_t *inner, const double *x,
                        double *zre, double *zim, double *out_re, double *out_im,
                        const int *perm, const double *tw_re, const double *tw_im,
                        int halfN, size_t K) {
    if (!packfuse_stage0(inner, x, zre, zim, K)) {
        run_prepack(inner, x, zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
        return 0;
    }
    exec_from_stage(inner, zre, zim, K, /*s_start=*/1);
    recombine_avx2(zre, zim, out_re, out_im, perm, tw_re, tw_im, halfN, K);
    return 1;
}

/* timers */
static double bench_prepack(const stride_plan_t *inner, const double *x,
                            double *zre, double *zim, double *oor, double *ooi,
                            const int *perm, const double *tw_re, const double *tw_im,
                            int halfN, size_t K, size_t total) {
    for (int w=0;w<10;w++) run_prepack(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
    int reps=reps_for(total); double best=1e18;
    for (int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) run_prepack(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
static double bench_packfuse(const stride_plan_t *inner, const double *x,
                             double *zre, double *zim, double *oor, double *ooi,
                             const int *perm, const double *tw_re, const double *tw_im,
                             int halfN, size_t K, size_t total) {
    for (int w=0;w<10;w++) run_packfuse(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
    int reps=reps_for(total); double best=1e18;
    for (int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) run_packfuse(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
/* pack-fused stage0 ONLY (for the breakdown: replaces pack+stage0-n1). */
static double bench_pf_stage0(const stride_plan_t *inner, const double *x,
                              double *zre, double *zim, size_t K, size_t total) {
    for (int w=0;w<10;w++) packfuse_stage0(inner,x,zre,zim,K);
    int reps=reps_for(total); double best=1e18;
    for (int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) packfuse_stage0(inner,x,zre,zim,K);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
/* prepack pass ONLY (the ~12µs we're trying to kill). */
static double bench_prepack_only(const double *x, double *zre, double *zim,
                                 int halfN, size_t K, size_t total) {
    for (int w=0;w<10;w++) prepack_stage0(x,zre,zim,halfN,K);
    int reps=reps_for(total); double best=1e18;
    for (int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) prepack_stage0(x,zre,zim,halfN,K);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
/* pack-fused INNER only (OOP s0 + stages 1.., NO recombine). */
static double bench_pf_inner(const stride_plan_t *inner, const double *x,
                             double *zre, double *zim, size_t K, size_t total) {
    for (int w=0;w<10;w++){packfuse_stage0(inner,x,zre,zim,K);exec_from_stage(inner,zre,zim,K,1);}
    int reps=reps_for(total); double best=1e18;
    for (int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++){packfuse_stage0(inner,x,zre,zim,K);exec_from_stage(inner,zre,zim,K,1);}
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
/* pre-pack INNER only (prepack + full execute_fwd, NO recombine). */
static double bench_pp_inner(const stride_plan_t *inner, const double *x,
                             double *zre, double *zim, size_t K, size_t total) {
    for (int w=0;w<10;w++){prepack_stage0(x,zre,zim,inner->N,K);vfft_proto_execute_fwd(inner,zre,zim,K);}
    int reps=reps_for(total); double best=1e18;
    for (int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++){prepack_stage0(x,zre,zim,inner->N,K);vfft_proto_execute_fwd(inner,zre,zim,K);}
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE h, const double *xin, double *cce, size_t total) {
    for (int w=0;w<10;w++) DftiComputeForward(h,(void*)xin,cce);
    int reps=reps_for(total); double best=1e18;
    for (int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) DftiComputeForward(h,(void*)xin,cce);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}

static double gate_vs_dft(const double *oor, const double *ooi,
                          const double *x, int N, int halfN, size_t K) {
    double maxerr = 0.0;
    for (int k = 0; k <= halfN; k++) {
        double rr=0, ri=0;
        for (int n=0;n<N;n++){
            double xn=x[(size_t)n*K+0];
            double ang=-2.0*M_PI*(double)k*(double)n/(double)N;
            rr+=xn*cos(ang); ri+=xn*sin(ang);
        }
        double er=fabs(oor[(size_t)k*K]-rr), ei=fabs(ooi[(size_t)k*K]-ri);
        if(er>maxerr)maxerr=er; if(ei>maxerr)maxerr=ei;
    }
    return maxerr;
}

int main(void) {
    stride_env_init();
    if (stride_pin_thread(PIN_CORE) != 0)
        fprintf(stderr, "warn: pin cpu%d failed\n", PIN_CORE);
    mkl_set_num_threads(1);

    const int N = 256, halfN = N/2;
    const size_t Ks[] = {32, 64, 128, 256};
    const int nK = (int)(sizeof Ks/sizeof Ks[0]);

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t wis; int have_wis =
        (vfft_proto_wisdom_load(&wis, "../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if (!have_wis)
        have_wis = (vfft_proto_wisdom_load(&wis, "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    printf("# c2c wisdom load: %s\n", have_wis ? "OK" : "FAILED (factorizer-default inner)");

    double *tw_re = alloc_d(halfN), *tw_im = alloc_d(halfN);
    _r2c_init_twiddles(N, tw_re, tw_im);

    printf("=== r2c PACK-FUSION (OOP stage-0 reads x directly, NO pre-pack pass) "
           "vs pre-pack vs MKL  (N=256, ST, cpu%d, best-of-%d) ===\n", PIN_CORE, BEST_OF);
    printf("%-5s %12s %12s %12s %10s %10s   %s\n",
           "K", "packfuse_ns", "prepack_ns", "mkl_ns", "pf/mkl", "pf/prepk",
           "breakdown(prepack_only | pf_stage0 | saved)");
    printf("------+------------+------------+------------+----------+----------+----------------------------\n");

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki];
        size_t total = (size_t)N*K;
        double *x = alloc_d(total);
        srand(7 + (int)K);
        for (size_t i=0;i<total;i++) x[i] = (double)rand()/RAND_MAX*2-1;

        stride_plan_t *inner = vfft_proto_auto_plan(halfN, K, &reg, have_wis ? &wis : NULL);
        if (!inner) { printf("%-5zu  auto_plan(128,%zu) NULL\n", K, K); vfft_proto_aligned_free(x); continue; }

        char fs[64]; size_t pp=0;
        for (int s=0;s<inner->num_stages;s++)
            pp += (size_t)snprintf(fs+pp,sizeof fs-pp,"%s%d",s?",":"",inner->factors[s]);
        int pf_supported = (inner->stages[0].radix == 4);
        printf("# K=%-4zu inner c2c(128) = (%s)  stage0_radix=%d  pack-fusion=%s\n",
               K, fs, inner->stages[0].radix, pf_supported ? "APPLIED" : "FALLBACK(prepack)");

        int perm[256];
        compute_perm(inner->factors, inner->num_stages, halfN, perm);

        double *zre=alloc_d((size_t)halfN*K), *zim=alloc_d((size_t)halfN*K);
        double *oor=alloc_d((size_t)(halfN+1)*K), *ooi=alloc_d((size_t)(halfN+1)*K);

        /* ---- correctness gate: both paths vs reference DFT (<1e-9) ---- */
        memset(oor,0,(size_t)(halfN+1)*K*sizeof(double));
        memset(ooi,0,(size_t)(halfN+1)*K*sizeof(double));
        run_prepack(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
        double err_pp = gate_vs_dft(oor,ooi,x,N,halfN,K);

        memset(oor,0,(size_t)(halfN+1)*K*sizeof(double));
        memset(ooi,0,(size_t)(halfN+1)*K*sizeof(double));
        int applied = run_packfuse(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K);
        double err_pf = gate_vs_dft(oor,ooi,x,N,halfN,K);

        if (err_pp >= 1e-9 || err_pf >= 1e-9) {
            printf("%-5zu  *** CORRECTNESS FAIL: prepack=%.3e packfuse=%.3e (applied=%d) ***\n",
                   K, err_pp, err_pf, applied);
            vfft_proto_aligned_free(x); vfft_proto_aligned_free(zre); vfft_proto_aligned_free(zim);
            vfft_proto_aligned_free(oor); vfft_proto_aligned_free(ooi);
            stride_plan_destroy(inner);
            continue;
        }

        /* ---- MKL r2c (transform-major) ---- */
        DFTI_DESCRIPTOR_HANDLE h=0; int mkl_ok=0;
        double *xin=alloc_d(total), *cce=alloc_d((size_t)(halfN+1)*K*2);
        for (size_t t=0;t<K;t++) for(int n=0;n<N;n++) xin[t*N+n]=x[(size_t)n*K+t];
        DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
        DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
        DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
        DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)(halfN+1));
        mkl_ok = (DftiCommitDescriptor(h)==DFTI_NO_ERROR);

        double pf_ns = bench_packfuse(inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K,total);
        double pp_ns = bench_prepack (inner,x,zre,zim,oor,ooi,perm,tw_re,tw_im,halfN,K,total);
        double prepk_only = bench_prepack_only(x,zre,zim,halfN,K,total);
        double pf_s0 = pf_supported ? bench_pf_stage0(inner,x,zre,zim,K,total) : 0;
        double m_ns = mkl_ok ? bench_mkl(h,xin,cce,total) : 0;

        double pf_inner = pf_supported ? bench_pf_inner(inner,x,zre,zim,K,total) : 0;
        double pp_inner = bench_pp_inner(inner,x,zre,zim,K,total);

        double pf_over_m = (m_ns>0 && pf_ns>0) ? m_ns/pf_ns : 0;
        double pf_over_pp = (pf_ns>0) ? pp_ns/pf_ns : 0;
        double saved = pp_ns - pf_ns;
        printf("%-5zu %12.1f %12.1f %12.1f %9.3fx %9.3fx   prepack_only=%.1f pf_stage0=%.1f saved=%.1f\n",
               K, pf_ns, pp_ns, m_ns, pf_over_m, pf_over_pp, prepk_only, pf_s0, saved);
        printf("#       inner-only: pf_inner=%.1f  pp_inner=%.1f  inner_saved=%.1f  (recombine~=pf_total-pf_inner=%.1f)\n",
               pf_inner, pp_inner, pp_inner-pf_inner, pf_ns-pf_inner);

        if (h) DftiFreeDescriptor(&h);
        vfft_proto_aligned_free(x); vfft_proto_aligned_free(zre); vfft_proto_aligned_free(zim);
        vfft_proto_aligned_free(oor); vfft_proto_aligned_free(ooi);
        vfft_proto_aligned_free(xin); vfft_proto_aligned_free(cce);
        stride_plan_destroy(inner);
    }

    vfft_proto_aligned_free(tw_re); vfft_proto_aligned_free(tw_im);
    if (have_wis) vfft_proto_wisdom_free(&wis);
    printf("\n# pf/mkl = MKL_ns / packfuse_ns (>1 = WE BEAT MKL).\n");
    printf("# pf/prepk = prepack_ns / packfuse_ns (>1 = pack-fusion faster than pre-pack path).\n");
    printf("# saved = prepack_ns - packfuse_ns (ns eliminated by killing the pre-pack pass).\n");
    return 0;
}
