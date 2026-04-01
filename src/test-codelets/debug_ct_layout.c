/**
 * debug_ct_layout.c -- Trace the CT data flow for N=12 = 3*4
 * Tests various calling conventions to find the one that works.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_radix3_avx2_ct_n1.h"
#include "fft_radix3_avx2_ct_t1_dit.h"
#include "fft_radix4_avx2.h"

#define N 12
#define R_N1 3   /* child: DFT-3 */
#define R_T1 4   /* parent: DFT-4 */

static void init_tw(double *W_re, double *W_im, size_t R, size_t me, size_t Ntotal) {
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)Ntotal;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

static void print_arr(const char *label, const double *re, const double *im, size_t n) {
    printf("  %s:", label);
    for (size_t i = 0; i < n; i++) printf(" [%zu]=%+.4f%+.4fi", i, re[i], im[i]);
    printf("\n");
}

static double check_fftw(const double *ir, const double *ii,
                         const double *our_re, const double *our_im) {
    double *fre=fftw_malloc(N*8),*fim=fftw_malloc(N*8);
    double *fro=fftw_malloc(N*8),*fio=fftw_malloc(N*8);
    memcpy(fre,ir,N*8); memcpy(fim,ii,N*8);
    fftw_iodim d={.n=N,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=1,.os=1};
    fftw_plan fp=fftw_plan_guru_split_dft(1,&d,1,&h,fre,fim,fro,fio,FFTW_ESTIMATE);
    fftw_execute(fp);
    double maxe=0;
    for(size_t i=0;i<N;i++){
        double e=fabs(our_re[i]-fro[i])+fabs(our_im[i]-fio[i]);
        if(e>maxe)maxe=e;
    }
    fftw_destroy_plan(fp);fftw_free(fre);fftw_free(fim);fftw_free(fro);fftw_free(fio);
    return maxe;
}

int main(void) {
    srand(42);
    double ir[N], ii[N];
    for (int i=0;i<N;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}

    printf("N=%d = R_n1(%d) x R_t1(%d)\n\n", N, R_N1, R_T1);
    print_arr("input", ir, ii, N);
    printf("\n");

    /* The CT DIT for N=12 with child=DFT-3, parent=DFT-4:
     *
     * View input as 3x4 matrix (3 rows, 4 cols), row-major:
     *   row 0: ir[0..3]   = x[0], x[1], x[2], x[3]
     *   row 1: ir[4..7]   = x[4], x[5], x[6], x[7]
     *   row 2: ir[8..11]  = x[8], x[9], x[10], x[11]
     *
     * Step 1: 4 child DFT-3 on each column
     *   Column m: elements at ir[m], ir[m+4], ir[m+8]  (stride=4=R_T1)
     *   The n1_ovs_R3 does DFT-3 on 3 elements at stride is=4
     *
     * Step 2: Twiddle by W_12^{n*m} where n=row, m=col
     *
     * Step 3: 3 parent DFT-4 on each row
     *   Row n: 4 contiguous elements  (stride=1, or ios=R_T1 if viewing as columns)
     */

    /* Test various calling conventions */
    double out_re[64], out_im[64];
    double tmp_re[64], tmp_im[64];
    double W_re[64], W_im[64];

    /* Convention A: n1_R3(is=R_T1=4, os=1, vl=R_T1=4, ovs=R_T1=4) + t1_R4(ios=R_T1=4, me=R_T1=4)
     * Twiddles: W_12^{n*m} for n=1..3, m=0..3 (R_T1 as me)
     * But this makes N_tw = R_T1 * R_T1 = 16 -- WRONG. Use N=12. */
    printf("=== Conv A: n1_R3(is=%d,vl=%d,ovs=%d) + t1_R4(ios=%d,me=%d), tw N=%d ===\n",
           R_T1, R_T1, R_T1, R_T1, R_T1, N);
    memset(tmp_re, 0, 64*8); memset(tmp_im, 0, 64*8);
    init_tw(W_re, W_im, R_T1, R_T1, N);
    radix3_n1_ovs_fwd_avx2(ir, ii, tmp_re, tmp_im, R_T1, 1, R_T1, R_T1);
    printf("  after n1_ovs_R3:\n");
    print_arr("  tmp", tmp_re, tmp_im, 16);
    radix4_t1_dit_fwd_avx2(tmp_re, tmp_im, W_re, W_im, R_T1, R_T1);
    memcpy(out_re, tmp_re, N*8); memcpy(out_im, tmp_im, N*8);
    printf("  after t1_R4:\n");
    print_arr("  out", out_re, out_im, N);
    printf("  err vs FFTW: %.2e\n\n", check_fftw(ir, ii, out_re, out_im));

    /* Convention B: n1_R3(is=R_N1=3, os=1, vl=R_N1=3, ovs=R_T1=4) -- like the old bench_ct_factor
     * But vl=3 doesn't work with VL=4 SIMD... skip this. */

    /* Convention C: Use scalar n1 instead of n1_ovs to avoid SIMD alignment issues.
     * scalar n1_R3(is=R_T1=4, os=1, vl=R_T1=4) -- scalar has no VL constraint */
    printf("=== Conv C: scalar n1_R3(is=%d,os=1,vl=%d) + avx2 t1_R4(ios=%d,me=%d), tw N=%d ===\n",
           R_T1, R_T1, R_T1, R_T1, N);
    memset(out_re, 0, 64*8); memset(out_im, 0, 64*8);
    init_tw(W_re, W_im, R_T1, R_T1, N);
    radix3_n1_fwd_avx2(ir, ii, out_re, out_im, R_T1, 1, R_T1);
    printf("  after n1_R3 (non-ovs):\n");
    print_arr("  out", out_re, out_im, N);
    radix4_t1_dit_fwd_avx2(out_re, out_im, W_re, W_im, R_T1, R_T1);
    printf("  after t1_R4:\n");
    print_arr("  out", out_re, out_im, N);
    printf("  err vs FFTW: %.2e\n\n", check_fftw(ir, ii, out_re, out_im));

    /* Convention D: Flip roles -- n1_R4(is=R_N1=3, os=1, vl=R_N1=3, ovs=R_N1=3) + t1_R3
     * But vl=3 doesn't work... try with buffer padding */

    /* Convention E: n1_R4 as child, t1_R3 as parent
     * n1_R4(is=R_N1=3, os=1, vl=R_N1=3, ovs=R_N1=3) -- vl=3, won't work SIMD
     * Skip. */

    /* Convention F: Manual scalar DFT-3 on columns + twiddle + DFT-4 on rows */
    printf("=== Conv F: Manual scalar column DFT-3 + twiddle + manual row DFT-4 ===\n");
    {
        /* View as 3×4 row-major. Column m: ir[m], ir[m+4], ir[m+8] */
        double mid_re[12], mid_im[12];
        /* DFT-3 on each column */
        for (int m = 0; m < 4; m++) {
            double x0r=ir[m], x0i=ii[m];
            double x1r=ir[m+4], x1i=ii[m+4];
            double x2r=ir[m+8], x2i=ii[m+8];
            /* DFT-3: y[0]=x0+x1+x2, y[1]=x0+W3^1*x1+W3^2*x2, y[2]=x0+W3^2*x1+W3^4*x2 */
            for (int k = 0; k < 3; k++) {
                double yr=0, yi=0;
                double xr[3]={x0r,x1r,x2r}, xi[3]={x0i,x1i,x2i};
                for (int n = 0; n < 3; n++) {
                    double a = -2.0*M_PI*k*n/3.0;
                    yr += xr[n]*cos(a) + xi[n]*sin(a);
                    yi += xi[n]*cos(a) - xr[n]*sin(a);
                }
                mid_re[k*4+m] = yr;  /* store in row-major: row k, col m */
                mid_im[k*4+m] = yi;
            }
        }
        print_arr("  after col DFT-3", mid_re, mid_im, 12);

        /* Twiddle: mid[k][m] *= W_12^{k*m} */
        for (int k = 0; k < 3; k++)
            for (int m = 0; m < 4; m++) {
                double a = -2.0*M_PI*k*m/12.0;
                double wr=cos(a), wi=sin(a);
                double tr = mid_re[k*4+m];
                mid_re[k*4+m] = tr*wr - mid_im[k*4+m]*wi;
                mid_im[k*4+m] = tr*wi + mid_im[k*4+m]*wr;
            }
        print_arr("  after twiddle", mid_re, mid_im, 12);

        /* DFT-4 on each row */
        for (int k = 0; k < 3; k++) {
            double yr[4]={0}, yi[4]={0};
            for (int j = 0; j < 4; j++)
                for (int m = 0; m < 4; m++) {
                    double a = -2.0*M_PI*j*m/4.0;
                    yr[j] += mid_re[k*4+m]*cos(a) + mid_im[k*4+m]*sin(a);
                    yi[j] += mid_im[k*4+m]*cos(a) - mid_re[k*4+m]*sin(a);
                }
            /* Output: X[k + 3*j] for j=0..3 (DIT output permutation) */
            for (int j = 0; j < 4; j++) {
                out_re[k + 3*j] = yr[j];
                out_im[k + 3*j] = yi[j];
            }
        }
        print_arr("  final", out_re, out_im, 12);
        printf("  err vs FFTW: %.2e\n\n", check_fftw(ir, ii, out_re, out_im));
    }

    return 0;
}
