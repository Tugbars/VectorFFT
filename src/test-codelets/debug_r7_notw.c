#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"
#include "fft_radix7_avx2_notw.h"
int main(void) {
    size_t R=7, K=4, N=R*K;
    double *ir=aligned_alloc(32,N*8), *ii=aligned_alloc(32,N*8);
    double *or_=aligned_alloc(32,N*8), *oi=aligned_alloc(32,N*8);
    double *fr=fftw_malloc(N*8), *fi=fftw_malloc(N*8);
    srand(42);
    for(size_t i=0;i<N;i++){ir[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}
    radix7_n1_dit_kernel_fwd_avx2(ir,ii,or_,oi,K);
    double *ir2=fftw_malloc(N*8),*ii2=fftw_malloc(N*8);
    memcpy(ir2,ir,N*8);memcpy(ii2,ii,N*8);
    fftw_iodim dim={.n=R,.is=(int)K,.os=(int)K};
    fftw_iodim howm={.n=(int)K,.is=1,.os=1};
    fftw_plan p=fftw_plan_guru_split_dft(1,&dim,1,&howm,ir2,ii2,fr,fi,FFTW_ESTIMATE);
    fftw_execute_split_dft(p,ir,ii,fr,fi);
    fftw_destroy_plan(p);
    printf("idx   ours_re         fftw_re         err_re          ours_im         fftw_im         err_im\n");
    double maxe=0;
    for(size_t i=0;i<N;i++){
        double er=fabs(or_[i]-fr[i]),ei=fabs(oi[i]-fi[i]);
        if(er>maxe)maxe=er;if(ei>maxe)maxe=ei;
        if(er>1e-10||ei>1e-10)
            printf("[%2zu] %+15.10f %+15.10f %+.2e  %+15.10f %+15.10f %+.2e\n",
                   i,or_[i],fr[i],or_[i]-fr[i],oi[i],fi[i],oi[i]-fi[i]);
    }
    printf("\nmax error: %.2e  %s\n",maxe,maxe<1e-10?"PASS":"FAIL");
    return 0;
}
