#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define K 8
#define MAX_N 11
#define MAX_W 10

#define DECL_ALL(R) \
extern void radix##R##_t1_dit_fwd_avx512_gen_inplace_su(double*,double*,const double*,const double*,size_t,size_t); \
extern void radix##R##_t1_dif_fwd_avx512_gen_inplace_su(double*,double*,const double*,const double*,size_t,size_t); \
extern void radix##R##_t1_dit_log3_fwd_avx512_gen_inplace_su(double*,double*,const double*,const double*,size_t,size_t); \
extern void radix##R##_t1_dif_log3_fwd_avx512_gen_inplace_su(double*,double*,const double*,const double*,size_t,size_t); \
extern void radix##R##_t1s_dit_fwd_avx512_gen_inplace_su(double*,double*,const double*,const double*,size_t,size_t); \
extern void radix##R##_t1s_dif_fwd_avx512_gen_inplace_su(double*,double*,const double*,const double*,size_t,size_t); \
extern void radix##R##_t1s_dit_log3_fwd_avx512_gen_inplace_su(double*,double*,const double*,const double*,size_t,size_t); \
extern void radix##R##_t1s_dif_log3_fwd_avx512_gen_inplace_su(double*,double*,const double*,const double*,size_t,size_t);

DECL_ALL(2) DECL_ALL(5) DECL_ALL(7) DECL_ALL(11)

typedef void (*fn_t)(double*,double*,const double*,const double*,size_t,size_t);

static double max_rel(const double *a, const double *b, int n) {
    double mx=0;
    for(int i=0;i<n;i++){
        double d=fabs(a[i]-b[i]); double s=fabs(a[i])+fabs(b[i])+1e-30;
        double r=d/s; if(r>mx) mx=r;
    }
    return mx;
}

static void brute(int dif, int N, const double *xr, const double *xi,
                   const double *wr, const double *wi, double *yr, double *yi) {
    double txr[MAX_N], txi[MAX_N];
    if (!dif) {
        txr[0]=xr[0]; txi[0]=xi[0];
        for(int n=1;n<N;n++){
            txr[n]=xr[n]*wr[n-1]-xi[n]*wi[n-1];
            txi[n]=xr[n]*wi[n-1]+xi[n]*wr[n-1];
        }
    } else { for(int n=0;n<N;n++){txr[n]=xr[n]; txi[n]=xi[n];} }
    for(int k=0;k<N;k++){
        double sr=0,si=0;
        for(int n=0;n<N;n++){
            double th=-2.0*M_PI*k*n/N;
            double c=cos(th), s=sin(th);
            sr+=txr[n]*c-txi[n]*s;
            si+=txr[n]*s+txi[n]*c;
        }
        yr[k]=sr; yi[k]=si;
    }
    if (dif) for(int k=1;k<N;k++){
        double r=yr[k]*wr[k-1]-yi[k]*wi[k-1];
        double i=yr[k]*wi[k-1]+yi[k]*wr[k-1];
        yr[k]=r; yi[k]=i;
    }
}

static int test_one(int R, const char *label, int dif, int strided, fn_t fn) {
    int W = R-1;
    static double rio_re[MAX_N*K] __attribute__((aligned(64)));
    static double rio_im[MAX_N*K] __attribute__((aligned(64)));
    static double tw_strided_re[MAX_W*K] __attribute__((aligned(64)));
    static double tw_strided_im[MAX_W*K] __attribute__((aligned(64)));
    static double tw_scalar_re[MAX_W] __attribute__((aligned(64)));
    static double tw_scalar_im[MAX_W] __attribute__((aligned(64)));
    static double in_re[MAX_N], in_im[MAX_N], ref_re[MAX_N], ref_im[MAX_N];

    srand(42 + R);
    for(int n=0;n<R;n++){
        in_re[n]=((double)rand()/RAND_MAX)*2-1;
        in_im[n]=((double)rand()/RAND_MAX)*2-1;
    }
    double base = ((double)rand()/RAND_MAX)*2*M_PI;
    for(int j=1;j<=W;j++){
        double th = -base*j;
        tw_scalar_re[j-1]=cos(th); tw_scalar_im[j-1]=sin(th);
    }
    for(int n=0;n<R;n++) for(int k=0;k<K;k++){
        rio_re[n*K+k]=in_re[n]; rio_im[n*K+k]=in_im[n];
    }
    for(int j=0;j<W;j++) for(int k=0;k<K;k++){
        tw_strided_re[j*K+k]=tw_scalar_re[j]; tw_strided_im[j*K+k]=tw_scalar_im[j];
    }
    if (strided) fn(rio_re, rio_im, tw_scalar_re, tw_scalar_im, K, K);
    else         fn(rio_re, rio_im, tw_strided_re, tw_strided_im, K, K);

    brute(dif, R, in_re, in_im, tw_scalar_re, tw_scalar_im, ref_re, ref_im);
    double rf[2*MAX_N], cf[2*MAX_N];
    for(int n=0;n<R;n++){
        rf[n]=ref_re[n]; rf[R+n]=ref_im[n];
        cf[n]=rio_re[n*K]; cf[R+n]=rio_im[n*K];
    }
    double err=max_rel(rf, cf, 2*R);
    int ok = err < 1e-10;
    printf("  R=%-2d %-22s  err=%.3e  %s\n", R, label, err, ok?"PASS":"FAIL");
    return ok;
}

#define RUN8(R) \
    fails += !test_one(R, "t1_dit",        0, 0, radix##R##_t1_dit_fwd_avx512_gen_inplace_su); \
    fails += !test_one(R, "t1_dif",        1, 0, radix##R##_t1_dif_fwd_avx512_gen_inplace_su); \
    fails += !test_one(R, "t1_dit_log3",   0, 0, radix##R##_t1_dit_log3_fwd_avx512_gen_inplace_su); \
    fails += !test_one(R, "t1_dif_log3",   1, 0, radix##R##_t1_dif_log3_fwd_avx512_gen_inplace_su); \
    fails += !test_one(R, "t1s_dit",       0, 1, radix##R##_t1s_dit_fwd_avx512_gen_inplace_su); \
    fails += !test_one(R, "t1s_dif",       1, 1, radix##R##_t1s_dif_fwd_avx512_gen_inplace_su); \
    fails += !test_one(R, "t1s_dit_log3",  0, 1, radix##R##_t1s_dit_log3_fwd_avx512_gen_inplace_su); \
    fails += !test_one(R, "t1s_dif_log3",  1, 1, radix##R##_t1s_dif_log3_fwd_avx512_gen_inplace_su);

int main(void) {
    int fails=0;
    printf("=== Runtime correctness: 8 variants × 4 radixes ===\n");
    RUN8(2); RUN8(5); RUN8(7); RUN8(11);
    if (fails) { printf("\n%d FAILS\n", fails); return 1; }
    printf("\n%d/32 PASS — all 8 variants × R={2,5,7,11} verified.\n", 32);
    return 0;
}
