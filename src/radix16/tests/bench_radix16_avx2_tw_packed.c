/*
 * bench_radix16_avx2_tw_packed.c — DFT-16 AVX2: strided vs packed vs FFTW
 */
#include "vfft_test_utils.h"
#include <fftw3.h>

#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
#ifndef ALIGNAS_32
#ifdef _MSC_VER
#define ALIGNAS_32 __declspec(align(32))
#else
#define ALIGNAS_32 __attribute__((aligned(32)))
#endif
#endif

#include "fft_radix16_avx2_tw.h"
#include "fft_radix16_tw_packed.h"

static void naive_tw_dft16(size_t K, size_t k,
    const double *ir, const double *ii,
    const double *twr, const double *twi,
    double *or_, double *oi) {
    double xr[16], xi[16];
    xr[0]=ir[k]; xi[0]=ii[k];
    for (int n=1;n<16;n++) {
        double wr=twr[(n-1)*K+k],wi=twi[(n-1)*K+k];
        xr[n]=ir[n*K+k]*wr-ii[n*K+k]*wi; xi[n]=ir[n*K+k]*wi+ii[n*K+k]*wr;
    }
    for (int m=0;m<16;m++) {
        double sr=0,si=0;
        for (int n=0;n<16;n++) {
            double a=-2.0*M_PI*m*n/16.0;
            sr+=xr[n]*cos(a)-xi[n]*sin(a); si+=xr[n]*sin(a)+xi[n]*cos(a);
        }
        or_[m*K+k]=sr; oi[m*K+k]=si;
    }
}

/* Correctness: packed vs naive */
static int test_packed(size_t K) {
    size_t T=4, N=16*K;
    if (K < T || K % T != 0) return 1;

    double *ir=aa64(N),*ii_=aa64(N),*nr=aa64(N),*ni=aa64(N);
    double *ftwr=aa64(15*K),*ftwi=aa64(15*K);
    double *pir=aa64(N),*pii=aa64(N),*por=aa64(N),*poi=aa64(N);
    double *ptwr=aa64(15*K),*ptwi=aa64(15*K);
    double *sor=aa64(N),*soi=aa64(N);

    fill_rand(ir,N,1000+(unsigned)K); fill_rand(ii_,N,2000+(unsigned)K);
    r16_build_flat_twiddles(K,-1,ftwr,ftwi);

    /* Pack data + twiddles */
    r16_pack_input(ir,ii_,pir,pii,K,T);
    r16_pack_twiddles(K,T,ftwr,ftwi,ptwr,ptwi);

    /* Packed DFT */
    r16_tw_packed_fwd_avx2(pir,pii,por,poi,ptwr,ptwi,K,T);

    /* Unpack result */
    r16_unpack_output(por,poi,sor,soi,K,T);

    /* Naive reference */
    for (size_t k=0;k<K;k++) naive_tw_dft16(K,k,ir,ii_,ftwr,ftwi,nr,ni);

    double err=0;
    for (size_t i=0;i<N;i++){
        double e=fmax(fabs(sor[i]-nr[i]),fabs(soi[i]-ni[i]));if(e>err)err=e;}
    double mag=fmax(max_abs(nr,N),max_abs(ni,N));
    double rel=mag>0?err/mag:err;
    int pass=rel<5e-13;
    printf("  packed T=%zu K=%-5zu rel=%.2e  %s\n",T,K,rel,pass?"PASS":"FAIL");

    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(nr);r32_aligned_free(ni);
    r32_aligned_free(ftwr);r32_aligned_free(ftwi);
    r32_aligned_free(pir);r32_aligned_free(pii);r32_aligned_free(por);r32_aligned_free(poi);
    r32_aligned_free(ptwr);r32_aligned_free(ptwi);r32_aligned_free(sor);r32_aligned_free(soi);
    return pass;
}

/* Benchmark */
__attribute__((target("avx2,fma")))
static void run_bench(size_t K, int warm, int trials) {
    size_t N=16*K, T=4;
    double *ir=aa64(N),*ii_=aa64(N),*or_=aa64(N),*oi=aa64(N);
    double *ftwr=aa64(15*K),*ftwi=aa64(15*K);
    fill_rand(ir,N,9000+(unsigned)K); fill_rand(ii_,N,9500+(unsigned)K);
    r16_build_flat_twiddles(K,-1,ftwr,ftwi);

    /* FFTW */
    fftw_complex *fin=fftw_alloc_complex(N),*fout=fftw_alloc_complex(N);
    for(size_t k=0;k<K;k++) for(int n=0;n<16;n++){
        fin[k*16+n][0]=ir[n*K+k]; fin[k*16+n][1]=ii_[n*K+k];}
    int na[1]={16};
    fftw_plan plan=fftw_plan_many_dft(1,na,(int)K,
        fin,NULL,1,16,fout,NULL,1,16,FFTW_FORWARD,FFTW_MEASURE);
    for(int i=0;i<warm;i++) fftw_execute(plan);
    double bfw=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns();fftw_execute(plan);
        double dt=get_ns()-t0;if(dt<bfw)bfw=dt;}

    /* Strided */
    for(int i=0;i<warm;i++)
        radix16_tw_flat_dit_kernel_fwd_avx2(ir,ii_,or_,oi,ftwr,ftwi,K);
    double ns_str=1e18;
    for(int t=0;t<trials;t++){double t0=get_ns();
        radix16_tw_flat_dit_kernel_fwd_avx2(ir,ii_,or_,oi,ftwr,ftwi,K);
        double dt=get_ns()-t0;if(dt<ns_str)ns_str=dt;}

    /* Packed (DFT-only, no repack) */
    double ns_pk=1e18;
    if (K>=T && K%T==0) {
        double *pir=aa64(N),*pii=aa64(N),*por=aa64(N),*poi=aa64(N);
        double *ptwr=aa64(15*K),*ptwi=aa64(15*K);
        r16_pack_input(ir,ii_,pir,pii,K,T);
        r16_pack_twiddles(K,T,ftwr,ftwi,ptwr,ptwi);
        for(int i=0;i<warm;i++)
            r16_tw_packed_fwd_avx2(pir,pii,por,poi,ptwr,ptwi,K,T);
        for(int t=0;t<trials;t++){double t0=get_ns();
            r16_tw_packed_fwd_avx2(pir,pii,por,poi,ptwr,ptwi,K,T);
            double dt=get_ns()-t0;if(dt<ns_pk)ns_pk=dt;}
        r32_aligned_free(pir);r32_aligned_free(pii);
        r32_aligned_free(por);r32_aligned_free(poi);
        r32_aligned_free(ptwr);r32_aligned_free(ptwi);
    }

    printf("  K=%-5zu  FFTW=%7.0f  str=%7.0f(%5.2fx)  pkd=%7.0f(%5.2fx)\n",
           K,bfw,ns_str,bfw/ns_str,ns_pk,bfw/ns_pk);

    fftw_destroy_plan(plan);fftw_free(fin);fftw_free(fout);
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(or_);r32_aligned_free(oi);
    r32_aligned_free(ftwr);r32_aligned_free(ftwi);
}

int main(void) {
    R32_REQUIRE_AVX2();

    printf("====================================================================\n");
    printf("  DFT-16 AVX2: strided vs packed (T=4) vs FFTW\n");
    printf("  Both do fused twiddle+DFT-16. FFTW baseline = no twiddles.\n");
    printf("====================================================================\n\n");

    int p=0,t=0;
    printf("-- Correctness: packed vs naive --\n");
    {size_t Ks[]={4,8,16,32,64,128,256,512};
     for(int i=0;i<8;i++){t++;p+=test_packed(Ks[i]);}}

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n",p,t,p==t?"ALL PASSED":"FAILURES");
    printf("======================================\n");
    if(p!=t) return 1;

    printf("\n-- BENCHMARK (ns, forward, DFT-only for packed) --\n\n");
    run_bench(4,    500,5000);
    run_bench(8,    500,5000);
    run_bench(16,   500,3000);
    run_bench(32,   500,3000);
    run_bench(64,   200,2000);
    run_bench(128,  200,2000);
    run_bench(256,  100,1000);
    run_bench(512,  100,1000);
    run_bench(1024,  50,500);
    run_bench(2048,  50,500);
    run_bench(4096,  20,300);

    fftw_cleanup();
    return 0;
}
