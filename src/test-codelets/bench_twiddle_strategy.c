/**
 * bench_twiddle_strategy.c — Head-to-head: cf arrays (method A) vs t1 folded (method B)
 *
 * Method A (current): per-element cf_re/cf_im multiply, then n1 butterfly. 2 data passes.
 * Method B (t1 folded): for k_prev=0: n1 only. for k_prev>0: common-factor on all legs
 *                        + t1_dit codelet (fuses per-leg twiddle with butterfly).
 *
 * Tests: N=60 (3x4x5), N=1000 (10x4x25), N=200 (10x20), N=4096 (64x64)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <fftw3.h>
#include "bench_compat.h"

/* n1 codelets */
#include "fft_radix3_avx2_ct_n1.h"
#include "fft_radix4_avx2.h"
#include "fft_radix5_avx2_ct_n1.h"
#include "fft_radix10_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix20_avx2_ct_n1.h"
#include "fft_radix25_avx2_ct_n1.h"
#include "fft_radix32_avx2_ct_n1.h"
#include "fft_radix64_avx2_ct_n1.h"

/* t1_dit codelets */
#include "fft_radix3_avx2_ct_t1_dit.h"
#include "fft_radix5_avx2_ct_t1_dit.h"
#include "fft_radix10_avx2_ct_t1_dit.h"
#include "fft_radix16_avx2_ct_t1_dit.h"
#include "fft_radix20_avx2_ct_t1_dit.h"
#include "fft_radix25_avx2_ct_t1_dit.h"
#include "fft_radix32_avx2_ct_t1_dit.h"
#include "fft_radix64_avx2_ct_t1_dit.h"

/* R=4 stride n1 */
__attribute__((target("avx2,fma")))
static void radix4_n1_stride_fwd(const double *a,const double *b,double *c,double *d,size_t is,size_t os,size_t vl){
    (void)a;(void)b;
    for(size_t k=0;k<vl;k+=4){
        __m256d r0=_mm256_load_pd(&c[k+0*os]),i0=_mm256_load_pd(&d[k+0*os]);
        __m256d r1=_mm256_load_pd(&c[k+1*os]),i1=_mm256_load_pd(&d[k+1*os]);
        __m256d r2=_mm256_load_pd(&c[k+2*os]),i2=_mm256_load_pd(&d[k+2*os]);
        __m256d r3=_mm256_load_pd(&c[k+3*os]),i3=_mm256_load_pd(&d[k+3*os]);
        __m256d sr=_mm256_add_pd(r0,r2),si=_mm256_add_pd(i0,i2);
        __m256d dr=_mm256_sub_pd(r0,r2),di=_mm256_sub_pd(i0,i2);
        __m256d tr=_mm256_add_pd(r1,r3),ti=_mm256_add_pd(i1,i3);
        __m256d ur=_mm256_sub_pd(r1,r3),ui=_mm256_sub_pd(i1,i3);
        _mm256_store_pd(&c[k+0*os],_mm256_add_pd(sr,tr));_mm256_store_pd(&d[k+0*os],_mm256_add_pd(si,ti));
        _mm256_store_pd(&c[k+2*os],_mm256_sub_pd(sr,tr));_mm256_store_pd(&d[k+2*os],_mm256_sub_pd(si,ti));
        _mm256_store_pd(&c[k+1*os],_mm256_add_pd(dr,ui));_mm256_store_pd(&d[k+1*os],_mm256_sub_pd(di,ur));
        _mm256_store_pd(&c[k+3*os],_mm256_sub_pd(dr,ui));_mm256_store_pd(&d[k+3*os],_mm256_add_pd(di,ur));
    }
}

static void null_t1(double*a,double*b,const double*c,const double*d,size_t e,size_t f){
    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;}

#include "stride_executor.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * METHOD B: t1 folded executor
 *
 * Decompose the per-element twiddle into:
 *   1. Common factor: W_N^{k_prev * ow_prev * lower_data_pos} — same for all legs
 *   2. Per-leg t1 twiddle: W_{N_partial}^{j * k_prev} — handled by t1_dit codelet
 *
 * Groups with k_prev=0: use n1 (no twiddle).
 * Groups with k_prev>0: apply common factor to ALL legs, then call t1_dit.
 *
 * The t1 twiddle tables are shared across all groups with the same k_prev.
 * Layout: tw_re[(j-1)*K + kk] for j=1..R-1.
 * ═══════════════════════════════════════════════════════════════ */

typedef void (*t1_dit_fn)(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me);

typedef void (*n1_fn)(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl);

typedef struct {
    int N;
    int num_stages;
    size_t K;
    int factors[STRIDE_MAX_STAGES];

    /* Per-stage */
    int radix[STRIDE_MAX_STAGES];
    size_t stride[STRIDE_MAX_STAGES];    /* distance between butterfly legs */
    int num_groups[STRIDE_MAX_STAGES];
    size_t *group_base[STRIDE_MAX_STAGES];

    /* Per-stage codelet */
    n1_fn n1_fwd[STRIDE_MAX_STAGES];
    t1_dit_fn t1_fwd[STRIDE_MAX_STAGES];

    /* Per-stage twiddle: t1 table per k_prev value */
    /* tw_re[s][(k_prev-1) * (R-1) * K + (j-1) * K + kk] */
    int R_prev[STRIDE_MAX_STAGES];       /* R of stage s-1 (number of k_prev values) */
    double *tw_re[STRIDE_MAX_STAGES];
    double *tw_im[STRIDE_MAX_STAGES];

    /* Per-group: k_prev value and common-factor twiddle (scalar, broadcast to all K) */
    int *k_prev_per_group[STRIDE_MAX_STAGES];
    double *cf_re_scalar[STRIDE_MAX_STAGES]; /* cf per group (single complex value) */
    double *cf_im_scalar[STRIDE_MAX_STAGES];
} folded_plan_t;

static folded_plan_t *folded_plan_create(int N, size_t K, const int *factors, int nf,
                                          n1_fn *n1_fwd_table, t1_dit_fn *t1_fwd_table) {
    folded_plan_t *p = (folded_plan_t*)calloc(1, sizeof(*p));
    p->N = N; p->K = K; p->num_stages = nf;
    memcpy(p->factors, factors, nf * sizeof(int));

    /* Compute strides and groups (same as stride_executor) */
    size_t dim_stride[STRIDE_MAX_STAGES];
    {
        size_t acc = K;
        for (int d = nf-1; d >= 0; d--) { dim_stride[d] = acc; acc *= factors[d]; }
    }

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        p->radix[s] = R;
        p->stride[s] = dim_stride[s];
        p->num_groups[s] = N / R;
        p->n1_fwd[s] = n1_fwd_table[s];
        p->t1_fwd[s] = t1_fwd_table[s];

        /* Enumerate group bases (same algorithm as stride_executor) */
        p->group_base[s] = (size_t*)malloc(p->num_groups[s] * sizeof(size_t));
        {
            int other_sizes[STRIDE_MAX_STAGES], n_other = 0;
            size_t other_strides[STRIDE_MAX_STAGES];
            for (int d = 0; d < nf; d++) {
                if (d != s) {
                    other_sizes[n_other] = factors[d];
                    other_strides[n_other] = dim_stride[d];
                    n_other++;
                }
            }
            int counter[STRIDE_MAX_STAGES]; memset(counter, 0, sizeof(counter));
            for (int g = 0; g < p->num_groups[s]; g++) {
                size_t base = 0;
                for (int d = 0; d < n_other; d++) base += (size_t)counter[d] * other_strides[d];
                p->group_base[s][g] = base;
                for (int d = n_other-1; d >= 0; d--) {
                    counter[d]++; if (counter[d] < other_sizes[d]) break; counter[d] = 0;
                }
            }
        }

        if (s == 0) {
            p->R_prev[s] = 0;
            p->tw_re[s] = p->tw_im[s] = NULL;
            p->k_prev_per_group[s] = NULL;
            p->cf_re_scalar[s] = p->cf_im_scalar[s] = NULL;
            continue;
        }

        /* Compute t1 twiddle tables and per-group k_prev + common factor */
        int R_prev_val = factors[s-1];
        p->R_prev[s] = R_prev_val;

        /* S_s = product of factors[s+1..nf-1] */
        int S_s = 1;
        for (int d = s+1; d < nf; d++) S_s *= factors[d];

        /* ow_prev = product of factors[0..s-2] */
        int ow_prev = 1;
        for (int d = 0; d < s-1; d++) ow_prev *= factors[d];

        /* N_partial = product of factors[0..s] */
        int N_partial = 1;
        for (int d = 0; d <= s; d++) N_partial *= factors[d];

        /* t1 twiddle table: (R_prev-1) sets, each (R-1)*K entries
         * For k_prev value kp (1..R_prev-1), leg j (1..R-1):
         *   W_{N_partial}^{j * kp * ow_prev * S_s}
         * But this simplifies: the t1 codelet expects W[(j-1)*me + m]
         * where me = K. So tw[(kp-1)*(R-1)*K + (j-1)*K + kk] = W^{j * kp * ow_prev * S_s / N}
         * Actually: tw_exp = (j * kp * ow_prev * S_s) % N
         */
        size_t tw_size = (size_t)(R_prev_val - 1) * (R - 1) * K;
        p->tw_re[s] = (double*)aligned_alloc(64, (tw_size ? tw_size : 1) * sizeof(double));
        p->tw_im[s] = (double*)aligned_alloc(64, (tw_size ? tw_size : 1) * sizeof(double));

        for (int kp = 1; kp < R_prev_val; kp++) {
            for (int j = 1; j < R; j++) {
                int tw_exp = ((long long)j * kp * ow_prev * S_s) % N;
                double angle = -2.0 * M_PI * (double)tw_exp / (double)N;
                double wr = cos(angle), wi = sin(angle);
                size_t base_idx = (size_t)(kp-1) * (R-1) * K + (size_t)(j-1) * K;
                for (size_t kk = 0; kk < K; kk++) {
                    p->tw_re[s][base_idx + kk] = wr;
                    p->tw_im[s][base_idx + kk] = wi;
                }
            }
        }

        /* Per-group: compute k_prev and common factor */
        p->k_prev_per_group[s] = (int*)malloc(p->num_groups[s] * sizeof(int));
        p->cf_re_scalar[s] = (double*)malloc(p->num_groups[s] * sizeof(double));
        p->cf_im_scalar[s] = (double*)malloc(p->num_groups[s] * sizeof(double));

        {
            int other_sizes[STRIDE_MAX_STAGES]; int n_other = 0;
            for (int d = 0; d < nf; d++) if (d != s) other_sizes[n_other++] = factors[d];

            int counter[STRIDE_MAX_STAGES]; memset(counter, 0, sizeof(counter));
            for (int g = 0; g < p->num_groups[s]; g++) {
                int k_prev = 0, lower_data_pos = 0;
                {
                    int ci = 0;
                    for (int d = 0; d < nf; d++) {
                        if (d == s) continue;
                        if (d == s-1) k_prev = counter[ci];
                        if (d > s) {
                            int w = 1;
                            for (int d2 = d+1; d2 < nf; d2++) w *= factors[d2];
                            lower_data_pos += counter[ci] * w;
                        }
                        ci++;
                    }
                }
                p->k_prev_per_group[s][g] = k_prev;

                if (k_prev == 0 || lower_data_pos == 0) {
                    p->cf_re_scalar[s][g] = 1.0;
                    p->cf_im_scalar[s][g] = 0.0;
                } else {
                    int cf_exp = ((long long)k_prev * ow_prev * lower_data_pos) % N;
                    double angle = -2.0 * M_PI * (double)cf_exp / (double)N;
                    p->cf_re_scalar[s][g] = cos(angle);
                    p->cf_im_scalar[s][g] = sin(angle);
                }

                for (int d = n_other-1; d >= 0; d--) {
                    counter[d]++; if (counter[d] < other_sizes[d]) break; counter[d] = 0;
                }
            }
        }
    }
    return p;
}

static void folded_plan_destroy(folded_plan_t *p) {
    for (int s = 0; s < p->num_stages; s++) {
        free(p->group_base[s]);
        if (p->tw_re[s]) aligned_free(p->tw_re[s]);
        if (p->tw_im[s]) aligned_free(p->tw_im[s]);
        free(p->k_prev_per_group[s]);
        free(p->cf_re_scalar[s]);
        free(p->cf_im_scalar[s]);
    }
    free(p);
}

static inline void folded_execute_fwd(const folded_plan_t *p, double *re, double *im) {
    const size_t K = p->K;

    for (int s = 0; s < p->num_stages; s++) {
        const int R = p->radix[s];
        const size_t ios = p->stride[s];

        for (int g = 0; g < p->num_groups[s]; g++) {
            double *base_re = re + p->group_base[s][g];
            double *base_im = im + p->group_base[s][g];

            if (s == 0 || p->k_prev_per_group[s][g] == 0) {
                /* No twiddle: n1 codelet only */
                p->n1_fwd[s](base_re, base_im, base_re, base_im, ios, ios, K);
            } else if (p->t1_fwd[s] == NULL) {
                /* No t1 codelet: fall back to cf multiply + n1 (method A behavior) */
                int kp = p->k_prev_per_group[s][g];
                double cfr = p->cf_re_scalar[s][g];
                double cfi = p->cf_im_scalar[s][g];
                /* Compute and apply combined twiddle (cf + per-leg) for each leg */
                int S_s = 1;
                for (int d = s+1; d < p->num_stages; d++) S_s *= p->factors[d];
                int ow_prev = 1;
                for (int d = 0; d < s-1; d++) ow_prev *= p->factors[d];
                int lower_data_pos = 0;
                if (cfr != 1.0 || cfi != 0.0) {
                    /* Reverse-engineer lower_data_pos from cf angle */
                    /* Actually just apply per-leg combined twiddle directly */
                }
                for (int j = 0; j < R; j++) {
                    double *lr = base_re + (size_t)j * ios;
                    double *li = base_im + (size_t)j * ios;
                    /* Combined twiddle: W_N^{kp * ow_prev * (j*S_s + lower_data_pos)} */
                    int tw_exp = ((long long)kp * ow_prev * j * S_s) % p->N;
                    double angle = -2.0 * M_PI * (double)tw_exp / (double)p->N;
                    double pleg_r = cos(angle), pleg_i = sin(angle);
                    /* Combine with common factor */
                    double wr = pleg_r * cfr - pleg_i * cfi;
                    double wi = pleg_r * cfi + pleg_i * cfr;
                    for (size_t kk = 0; kk < K; kk++) {
                        double tr = lr[kk];
                        lr[kk] = tr * wr - li[kk] * wi;
                        li[kk] = tr * wi + li[kk] * wr;
                    }
                }
                p->n1_fwd[s](base_re, base_im, base_re, base_im, ios, ios, K);
            } else {
                int kp = p->k_prev_per_group[s][g];
                double cfr = p->cf_re_scalar[s][g];
                double cfi = p->cf_im_scalar[s][g];

                /* Apply common factor to ALL legs (if non-trivial) */
                if (cfr != 1.0 || cfi != 0.0) {
                    for (int j = 0; j < R; j++) {
                        double *lr = base_re + (size_t)j * ios;
                        double *li = base_im + (size_t)j * ios;
                        for (size_t kk = 0; kk < K; kk++) {
                            double tr = lr[kk];
                            lr[kk] = tr * cfr - li[kk] * cfi;
                            li[kk] = tr * cfi + li[kk] * cfr;
                        }
                    }
                }

                /* t1_dit codelet: fused per-leg twiddle + butterfly */
                const double *tw_r = p->tw_re[s] + (size_t)(kp-1) * (R-1) * K;
                const double *tw_i = p->tw_im[s] + (size_t)(kp-1) * (R-1) * K;
                p->t1_fwd[s](base_re, base_im, tw_r, tw_i, ios, K);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════ */

static void build_digit_rev_perm(int *perm, const int *factors, int nf) {
    int Nv = 1; for (int i = 0; i < nf; i++) Nv *= factors[i];
    int ow[STRIDE_MAX_STAGES]; ow[0] = 1;
    for (int i = 1; i < nf; i++) ow[i] = ow[i-1] * factors[i-1];
    int sw[STRIDE_MAX_STAGES]; sw[nf-1] = 1;
    for (int i = nf-2; i >= 0; i--) sw[i] = sw[i+1] * factors[i+1];
    int cnt[STRIDE_MAX_STAGES]; memset(cnt, 0, sizeof(cnt));
    for (int i = 0; i < Nv; i++) {
        int pos = 0, idx = 0;
        for (int d = 0; d < nf; d++) { pos += cnt[d]*sw[d]; idx += cnt[d]*ow[d]; }
        perm[idx] = pos;
        for (int d = nf-1; d >= 0; d--) { cnt[d]++; if (cnt[d] < factors[d]) break; cnt[d] = 0; }
    }
}

/* Brute-force DFT reference */
static void bruteforce_dft(const double *xr, const double *xi,
                           double *Xr, double *Xi, int N, size_t K) {
    for (int k = 0; k < N; k++) {
        for (size_t b = 0; b < K; b++) {
            double sr = 0, si = 0;
            for (int n = 0; n < N; n++) {
                double angle = -2.0 * M_PI * (double)n * (double)k / (double)N;
                double wr = cos(angle), wi = sin(angle);
                sr += xr[n*K + b] * wr - xi[n*K + b] * wi;
                si += xr[n*K + b] * wi + xi[n*K + b] * wr;
            }
            Xr[k*K + b] = sr;
            Xi[k*K + b] = si;
        }
    }
}

typedef struct {
    const char *label;
    int N;
    int nf;
    int factors[STRIDE_MAX_STAGES];
    stride_n1_fn n1f[STRIDE_MAX_STAGES];
    n1_fn n1f_typed[STRIDE_MAX_STAGES];
    t1_dit_fn t1f_typed[STRIDE_MAX_STAGES];
} test_case_t;

static void bench_case(const test_case_t *tc) {
    const int Nv = tc->N;
    const int nf = tc->nf;
    const int *factors = tc->factors;

    printf("\n== %s  N=%d ==\n", tc->label, Nv);

    /* Correctness check for method B (K=4) */
    {
        const size_t K = 4;
        const size_t total = (size_t)Nv * K;
        double *ref_re = (double*)malloc(total*8), *ref_im = (double*)malloc(total*8);
        double *data_re = aligned_alloc(64, total*8), *data_im = aligned_alloc(64, total*8);
        double *orig_re = (double*)malloc(total*8), *orig_im = (double*)malloc(total*8);

        srand(12345);
        for (size_t i = 0; i < total; i++) {
            orig_re[i] = (double)rand()/RAND_MAX - 0.5;
            orig_im[i] = (double)rand()/RAND_MAX - 0.5;
        }
        bruteforce_dft(orig_re, orig_im, ref_re, ref_im, Nv, K);

        folded_plan_t *fp = folded_plan_create(Nv, K, factors, nf, tc->n1f_typed, tc->t1f_typed);
        memcpy(data_re, orig_re, total*8);
        memcpy(data_im, orig_im, total*8);
        folded_execute_fwd(fp, data_re, data_im);

        int *perm = (int*)malloc(Nv * sizeof(int));
        double *sr = (double*)malloc(total*8), *si = (double*)malloc(total*8);
        build_digit_rev_perm(perm, factors, nf);
        for (int m = 0; m < Nv; m++) {
            memcpy(sr + (size_t)m*K, data_re + (size_t)perm[m]*K, K*8);
            memcpy(si + (size_t)m*K, data_im + (size_t)perm[m]*K, K*8);
        }
        double me = 0;
        for (size_t i = 0; i < total; i++) {
            double e = fabs(sr[i] - ref_re[i]) + fabs(si[i] - ref_im[i]);
            if (e > me) me = e;
        }
        printf("  Method B correctness: err=%.2e %s\n", me, me < 1e-9 ? "OK" : "FAIL");
        if (me >= 1e-9) { printf("  *** SKIP BENCH ***\n"); }

        free(perm); free(sr); free(si);
        free(ref_re); free(ref_im); free(orig_re); free(orig_im);
        aligned_free(data_re); aligned_free(data_im);
        folded_plan_destroy(fp);
        if (me >= 1e-9) return;
    }

    /* Performance comparison */
    printf("\n  %-5s %-8s %10s %10s %10s %8s\n", "K", "N*K", "A(cf+n1)", "B(t1fold)", "FFTW_M", "B/A");
    printf("  %-5s %-8s %10s %10s %10s %8s\n", "-----", "--------", "----------", "----------", "----------", "--------");

    stride_t1_fn null_t1s[STRIDE_MAX_STAGES];
    stride_n1_fn null_n1b[STRIDE_MAX_STAGES];
    for (int i = 0; i < nf; i++) { null_t1s[i] = null_t1; null_n1b[i] = (stride_n1_fn)tc->n1f[i]; }

    size_t bench_Ks[] = {4, 8, 16, 32, 64, 128, 256, 512};
    for (int bi = 0; bi < 8; bi++) {
        size_t K = bench_Ks[bi];
        size_t total = (size_t)Nv * K;
        int reps = (int)(5e5 / (total + 1));
        if (reps < 50) reps = 50; if (reps > 500000) reps = 500000;

        /* Method A: cf + n1 (existing stride executor) */
        stride_plan_t *planA = stride_plan_create(Nv, K, factors, nf,
            tc->n1f, null_n1b, null_t1s, null_t1s);
        double *reA = aligned_alloc(64, total*8), *imA = aligned_alloc(64, total*8);
        for (size_t i = 0; i < total; i++) { reA[i] = (double)rand()/RAND_MAX - 0.5; imA[i] = (double)rand()/RAND_MAX - 0.5; }
        for (int i = 0; i < 20; i++) stride_execute_fwd(planA, reA, imA);
        double bestA = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int i = 0; i < reps; i++) stride_execute_fwd(planA, reA, imA);
            double ns = (now_ns() - t0) / reps;
            if (ns < bestA) bestA = ns;
        }
        aligned_free(reA); aligned_free(imA);
        stride_plan_destroy(planA);

        /* Method B: t1 folded */
        folded_plan_t *planB = folded_plan_create(Nv, K, factors, nf, tc->n1f_typed, tc->t1f_typed);
        double *reB = aligned_alloc(64, total*8), *imB = aligned_alloc(64, total*8);
        for (size_t i = 0; i < total; i++) { reB[i] = (double)rand()/RAND_MAX - 0.5; imB[i] = (double)rand()/RAND_MAX - 0.5; }
        for (int i = 0; i < 20; i++) folded_execute_fwd(planB, reB, imB);
        double bestB = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int i = 0; i < reps; i++) folded_execute_fwd(planB, reB, imB);
            double ns = (now_ns() - t0) / reps;
            if (ns < bestB) bestB = ns;
        }
        aligned_free(reB); aligned_free(imB);
        folded_plan_destroy(planB);

        /* FFTW reference */
        double *fr = fftw_malloc(total*8), *fi = fftw_malloc(total*8);
        double *fo = fftw_malloc(total*8), *fo2 = fftw_malloc(total*8);
        for (size_t i = 0; i < total; i++) { fr[i] = (double)rand()/RAND_MAX - 0.5; fi[i] = (double)rand()/RAND_MAX - 0.5; }
        fftw_iodim dim = {.n = Nv, .is = (int)K, .os = (int)K};
        fftw_iodim howm = {.n = (int)K, .is = 1, .os = 1};
        fftw_plan fp = fftw_plan_guru_split_dft(1, &dim, 1, &howm, fr, fi, fo, fo2, FFTW_MEASURE);
        for (size_t i = 0; i < total; i++) { fr[i] = (double)rand()/RAND_MAX - 0.5; fi[i] = (double)rand()/RAND_MAX - 0.5; }
        for (int i = 0; i < 20; i++) fftw_execute(fp);
        double bestF = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int i = 0; i < reps; i++) fftw_execute_split_dft(fp, fr, fi, fo, fo2);
            double ns = (now_ns() - t0) / reps;
            if (ns < bestF) bestF = ns;
        }
        fftw_destroy_plan(fp); fftw_free(fr); fftw_free(fi); fftw_free(fo); fftw_free(fo2);

        double ratio = bestA > 0 ? bestA / bestB : 0;
        printf("  %-5zu %-8zu %10.1f %10.1f %10.1f %7.2fx\n",
               K, total, bestA, bestB, bestF, ratio);
    }
}

int main(void) {
    srand(42);
    printf("VectorFFT Twiddle Strategy Benchmark: Method A (cf+n1) vs Method B (t1 folded)\n");
    printf("================================================================================\n");

    /* N=60 = 3x4x5 (the original folded test case) */
    {
        test_case_t tc = {
            .label = "3x4x5", .N = 60, .nf = 3,
            .factors = {3, 4, 5},
            .n1f = {(stride_n1_fn)radix3_n1_fwd_avx2, (stride_n1_fn)radix4_n1_stride_fwd, (stride_n1_fn)radix5_n1_fwd_avx2},
            .n1f_typed = {(n1_fn)radix3_n1_fwd_avx2, (n1_fn)radix4_n1_stride_fwd, (n1_fn)radix5_n1_fwd_avx2},
            .t1f_typed = {NULL, (t1_dit_fn)radix4_t1_dit_fwd_avx2, (t1_dit_fn)radix5_t1_dit_fwd_avx2},
        };
        bench_case(&tc);
    }

    /* N=200 = 10x20 (2 stages, both composites) */
    {
        test_case_t tc = {
            .label = "10x20", .N = 200, .nf = 2,
            .factors = {10, 20},
            .n1f = {(stride_n1_fn)radix10_n1_fwd_avx2, (stride_n1_fn)radix20_n1_fwd_avx2},
            .n1f_typed = {(n1_fn)radix10_n1_fwd_avx2, (n1_fn)radix20_n1_fwd_avx2},
            .t1f_typed = {NULL, (t1_dit_fn)radix20_t1_dit_fwd_avx2},
        };
        bench_case(&tc);
    }

    /* N=1000 = 10x4x25 (3 stages) */
    {
        test_case_t tc = {
            .label = "10x4x25", .N = 1000, .nf = 3,
            .factors = {10, 4, 25},
            .n1f = {(stride_n1_fn)radix10_n1_fwd_avx2, (stride_n1_fn)radix4_n1_stride_fwd, (stride_n1_fn)radix25_n1_fwd_avx2},
            .n1f_typed = {(n1_fn)radix10_n1_fwd_avx2, (n1_fn)radix4_n1_stride_fwd, (n1_fn)radix25_n1_fwd_avx2},
            .t1f_typed = {NULL, (t1_dit_fn)radix4_t1_dit_fwd_avx2, (t1_dit_fn)radix25_t1_dit_fwd_avx2},
        };
        bench_case(&tc);
    }

    /* N=4096 = 64x64 (2 stages, pow2) */
    {
        test_case_t tc = {
            .label = "64x64", .N = 4096, .nf = 2,
            .factors = {64, 64},
            .n1f = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix64_n1_fwd_avx2},
            .n1f_typed = {(n1_fn)radix64_n1_fwd_avx2, (n1_fn)radix64_n1_fwd_avx2},
            .t1f_typed = {NULL, (t1_dit_fn)radix64_t1_dit_fwd_avx2},
        };
        bench_case(&tc);
    }

    /* N=1024 = 64x16 (2 stages) */
    {
        test_case_t tc = {
            .label = "64x16", .N = 1024, .nf = 2,
            .factors = {64, 16},
            .n1f = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix16_n1_fwd_avx2},
            .n1f_typed = {(n1_fn)radix64_n1_fwd_avx2, (n1_fn)radix16_n1_fwd_avx2},
            .t1f_typed = {NULL, (t1_dit_fn)radix16_t1_dit_fwd_avx2},
        };
        bench_case(&tc);
    }

    /* N=2048 = 64x32 (2 stages) */
    {
        test_case_t tc = {
            .label = "64x32", .N = 2048, .nf = 2,
            .factors = {64, 32},
            .n1f = {(stride_n1_fn)radix64_n1_fwd_avx2, (stride_n1_fn)radix32_n1_fwd_avx2},
            .n1f_typed = {(n1_fn)radix64_n1_fwd_avx2, (n1_fn)radix32_n1_fwd_avx2},
            .t1f_typed = {NULL, (t1_dit_fn)radix32_t1_dit_fwd_avx2},
        };
        bench_case(&tc);
    }

    printf("\nDone.\n");
    return 0;
}
