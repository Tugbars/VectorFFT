#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#include "fft_n1_k1.h"
#include "fft_n1_k1_simd.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_notw.h"
#include "fft_radix16_avx2_dit_tw.h"

#define N_FFT 32768
#define N1 256
#define N2 128
#define TILE 32

static inline double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}

static void init_tw(double *twr,double *twi,size_t R,size_t K){
    for(size_t n=1;n<R;n++)for(size_t k=0;k<K;k++){
        double a=-2*M_PI*(double)(n*k)/(double)(R*K);twr[(n-1)*K+k]=cos(a);twi[(n-1)*K+k]=sin(a);}}

static size_t *build_perm(const size_t *rx,size_t ns,size_t N){
    size_t *p=malloc(N*8);for(size_t i=0;i<N;i++){size_t rem=i,rev=0,str=N;
    for(size_t s=0;s<ns;s++){str/=rx[s];rev+=(rem%rx[s])*str;rem/=rx[s];}p[i]=rev;}return p;}

static void r16_k1_wrap(const double*ir,const double*ii,double*or_,double*oi,size_t K){(void)K;dft16_k1_fwd_avx2(ir,ii,or_,oi);}

/* Leaf DFT-256 */
static double *tw256r,*tw256i; static size_t *perm256;
static void leaf256_init(void){tw256r=aligned_alloc(32,15*16*8);tw256i=aligned_alloc(32,15*16*8);init_tw(tw256r,tw256i,16,16);size_t rx[]={16,16};perm256=build_perm(rx,2,256);}
static void leaf256(const double*ir,const double*ii,double*or_,double*oi,double*br,double*bi){
    for(size_t i=0;i<256;i++){br[i]=ir[perm256[i]];bi[i]=ii[perm256[i]];}
    for(size_t g=0;g<16;g++){r16_k1_wrap(br+g*16,bi+g*16,or_+g*16,oi+g*16,1);}
    radix16_tw_flat_dit_kernel_fwd_avx2(or_,oi,br,bi,tw256r,tw256i,16);
    memcpy(or_,br,256*8);memcpy(oi,bi,256*8);}

/* Leaf DFT-128 */
static double *tw128r,*tw128i; static size_t *perm128;
static void leaf128_init(void){tw128r=aligned_alloc(32,7*16*8);tw128i=aligned_alloc(32,7*16*8);init_tw(tw128r,tw128i,8,16);size_t rx[]={16,8};perm128=build_perm(rx,2,128);}
static void leaf128(const double*ir,const double*ii,double*or_,double*oi,double*br,double*bi){
    for(size_t i=0;i<128;i++){br[i]=ir[perm128[i]];bi[i]=ii[perm128[i]];}
    for(size_t g=0;g<8;g++){r16_k1_wrap(br+g*16,bi+g*16,or_+g*16,oi+g*16,1);}
    radix8_tw_dit_kernel_fwd_avx2(or_,oi,br,bi,tw128r,tw128i,16);
    memcpy(or_,br,128*8);memcpy(oi,bi,128*8);}

static void do_transpose(const double*__restrict__ s,double*__restrict__ d,size_t rows,size_t cols){
    for(size_t i0=0;i0<rows;i0+=TILE){size_t i1=i0+TILE;if(i1>rows)i1=rows;
    for(size_t j0=0;j0<cols;j0+=TILE){size_t j1=j0+TILE;if(j1>cols)j1=cols;
    for(size_t i=i0;i<i1;i++)for(size_t j=j0;j<j1;j++)d[j*rows+i]=s[i*cols+j];}}}

__attribute__((target("avx2,fma")))
static void tw_apply(double*__restrict__ re,double*__restrict__ im,
    const double*__restrict__ wr,const double*__restrict__ wi,size_t N){
    for(size_t i=0;i+4<=N;i+=4){
        __m256d xr=_mm256_load_pd(&re[i]),xi=_mm256_load_pd(&im[i]);
        __m256d tr=_mm256_load_pd(&wr[i]),ti=_mm256_load_pd(&wi[i]);
        _mm256_store_pd(&re[i],_mm256_fmsub_pd(xr,tr,_mm256_mul_pd(xi,ti)));
        _mm256_store_pd(&im[i],_mm256_fmadd_pd(xr,ti,_mm256_mul_pd(xi,tr)));}}

int main(void){
    srand(42);
    leaf256_init(); leaf128_init();

    double *in_re=aligned_alloc(32,N_FFT*8),*in_im=aligned_alloc(32,N_FFT*8);
    double *out_re=aligned_alloc(32,N_FFT*8),*out_im=aligned_alloc(32,N_FFT*8);
    double *scr_re=aligned_alloc(32,N_FFT*8),*scr_im=aligned_alloc(32,N_FFT*8);
    double *scr2_re=aligned_alloc(32,N_FFT*8),*scr2_im=aligned_alloc(32,N_FFT*8);
    double *lbr=aligned_alloc(32,256*8),*lbi=aligned_alloc(32,256*8);
    double *twr=aligned_alloc(32,N_FFT*8),*twi=aligned_alloc(32,N_FFT*8);

    for(size_t i=0;i<N_FFT;i++){in_re[i]=(double)rand()/RAND_MAX-.5;in_im[i]=(double)rand()/RAND_MAX-.5;}
    for(size_t n1=0;n1<N1;n1++)for(size_t k2=0;k2<N2;k2++){
        double a=-2*M_PI*(double)(n1*k2)/(double)N_FFT;
        twr[n1*N2+k2]=cos(a);twi[n1*N2+k2]=sin(a);}

    int REPS=500;
    double t_s1=0,t_s2=0,t_s3=0,t_s4=0,t_s5=0,t_s6=0,t_tot=0;

    for(int w=0;w<20;w++){
        do_transpose(in_re,scr_re,N2,N1);do_transpose(in_im,scr_im,N2,N1);
        for(size_t n1=0;n1<N1;n1++)leaf128(scr_re+n1*N2,scr_im+n1*N2,out_re+n1*N2,out_im+n1*N2,lbr,lbi);
        tw_apply(out_re,out_im,twr,twi,N_FFT);
        do_transpose(out_re,scr_re,N1,N2);do_transpose(out_im,scr_im,N1,N2);
        for(size_t k2=0;k2<N2;k2++)leaf256(scr_re+k2*N1,scr_im+k2*N1,scr2_re+k2*N1,scr2_im+k2*N1,lbr,lbi);
        do_transpose(scr2_re,out_re,N2,N1);do_transpose(scr2_im,out_im,N2,N1);
    }

    for(int r=0;r<REPS;r++){
        double t0,t1;

        t0=now_ns();
        do_transpose(in_re,scr_re,N2,N1); do_transpose(in_im,scr_im,N2,N1);
        t1=now_ns(); t_s1+=t1-t0;

        t0=now_ns();
        for(size_t n1=0;n1<N1;n1++)leaf128(scr_re+n1*N2,scr_im+n1*N2,out_re+n1*N2,out_im+n1*N2,lbr,lbi);
        t1=now_ns(); t_s2+=t1-t0;

        t0=now_ns();
        tw_apply(out_re,out_im,twr,twi,N_FFT);
        t1=now_ns(); t_s3+=t1-t0;

        t0=now_ns();
        do_transpose(out_re,scr_re,N1,N2); do_transpose(out_im,scr_im,N1,N2);
        t1=now_ns(); t_s4+=t1-t0;

        t0=now_ns();
        for(size_t k2=0;k2<N2;k2++)leaf256(scr_re+k2*N1,scr_im+k2*N1,scr2_re+k2*N1,scr2_im+k2*N1,lbr,lbi);
        t1=now_ns(); t_s5+=t1-t0;

        t0=now_ns();
        do_transpose(scr2_re,out_re,N2,N1); do_transpose(scr2_im,out_im,N2,N1);
        t1=now_ns(); t_s6+=t1-t0;
    }

    t_tot=t_s1+t_s2+t_s3+t_s4+t_s5+t_s6;
    printf("N=%d, split %dx%d, %d reps\n\n",N_FFT,N1,N2,REPS);
    printf("Step 1 (transpose N2xN1→N1xN2):  %7.0f ns  %4.1f%%\n",t_s1/REPS,100*t_s1/t_tot);
    printf("Step 2 (%d DFT-%d):              %7.0f ns  %4.1f%%\n",(int)N1,(int)N2,t_s2/REPS,100*t_s2/t_tot);
    printf("Step 3 (twiddle):                 %7.0f ns  %4.1f%%\n",t_s3/REPS,100*t_s3/t_tot);
    printf("Step 4 (transpose N1xN2→N2xN1):  %7.0f ns  %4.1f%%\n",t_s4/REPS,100*t_s4/t_tot);
    printf("Step 5 (%d DFT-%d):              %7.0f ns  %4.1f%%\n",(int)N2,(int)N1,t_s5/REPS,100*t_s5/t_tot);
    printf("Step 6 (final transpose):         %7.0f ns  %4.1f%%\n",t_s6/REPS,100*t_s6/t_tot);
    printf("────────────────────────────────────────────\n");
    printf("Total:                            %7.0f ns\n",t_tot/REPS);
    printf("\nTranspose total:                  %7.0f ns  %4.1f%%\n",(t_s1+t_s4+t_s6)/REPS,100*(t_s1+t_s4+t_s6)/t_tot);
    printf("DFT total:                        %7.0f ns  %4.1f%%\n",(t_s2+t_s5)/REPS,100*(t_s2+t_s5)/t_tot);
    return 0;
}
