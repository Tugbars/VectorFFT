/* test_conv.c — conv.h correctness vs direct O(N^2) references.
 *
 * Cells:
 *   1D circular conv + correlate vs direct, N in {64, 60, 61(prime), 128}
 *   1D batched (N=64, K=8): 8 independent per-lane kernels in one call
 *   impulse identity: h = delta -> conv(x) == x (bit-tight)
 *   shift theorem: h = delta_shift(s) -> conv(x)[m] == x[(m-s) mod N]
 *   2D circular conv via fftnd rank-2 (32x48) vs direct
 *   3D circular conv via fft3d (16x12x20) vs direct
 *   linear convolution pattern: next_fast_n + zero-pad vs direct linear
 *   MT: pointwise sweep at T=4 on a large 1D case
 *
 * Build: python build.py --src benches/test_conv.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft3d.h"
#include "fftnd.h"
#include "conv.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif

static double frand(void){ return 2.0*((double)rand()/RAND_MAX)-1.0; }
static int g_fail = 0;

/* direct circular conv/corr, arbitrary rank (row-major), O(n^2) */
static void direct_circ(int rank, const int *N, int conj_h,
                        const double *xr, const double *xi,
                        const double *hr, const double *hi,
                        double *or_, double *oi) {
    size_t n = 1; for (int m=0;m<rank;m++) n *= (size_t)N[m];
    for (size_t o = 0; o < n; o++) {
        int oc[4]={0,0,0,0}; { size_t t=o;
            for (int m=rank-1;m>=0;m--){ oc[m]=(int)(t%N[m]); t/=N[m]; } }
        double sr=0, si=0;
        for (size_t k = 0; k < n; k++) {
            int kc[4]={0,0,0,0}; { size_t t=k;
                for (int m=rank-1;m>=0;m--){ kc[m]=(int)(t%N[m]); t/=N[m]; } }
            size_t j = 0;
            if (!conj_h) {            /* conv: x[k] * h[o-k mod N] */
                for (int m=0;m<rank;m++){
                    int d=(oc[m]-kc[m])%N[m]; if(d<0)d+=N[m];
                    j = j*(size_t)N[m] + (size_t)d;
                }
                double a=xr[k],b=xi[k],c=hr[j],d2=hi[j];
                sr += a*c - b*d2;
                si += a*d2 + b*c;
            } else {                  /* corr: x[k+o mod N] * conj(h[k]) */
                for (int m=0;m<rank;m++){
                    int d=(kc[m]+oc[m])%N[m];
                    j = j*(size_t)N[m] + (size_t)d;
                }
                double a=xr[j],b=xi[j],c=hr[k],d2=hi[k];
                sr += a*c + b*d2;
                si += b*c - a*d2;
            }
        }
        or_[o]=sr; oi[o]=si;
    }
}

static double max_rel(const double *ar,const double *ai,
                      const double *br,const double *bi,size_t n){
    double mx=0, sc=0;
    for(size_t i=0;i<n;i++){ double m=fabs(br[i])+fabs(bi[i]); if(m>sc)sc=m; }
    if (sc==0) sc=1;
    for(size_t i=0;i<n;i++){
        double e=(fabs(ar[i]-br[i])+fabs(ai[i]-bi[i]))/sc;
        if(e>mx)mx=e;
    }
    return mx;
}

static void check(const char *tag, double err, double tol){
    int ok = err < tol;
    if (!ok) g_fail++;
    printf("  %-38s err=%.2e  %s\n", tag, err, ok?"OK":"**FAIL**");
}

/* wrap a 1D plan at (N, K) */
static stride_conv_t *conv1d(int N, size_t K, const vfft_proto_registry_t *reg){
    stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, reg, NULL);
    return stride_conv_wrap(p, 1);
}

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    printf("conv.h correctness\n");

    /* ── 1D circular conv + correlate, incl. prime N (K=8, lane-wise) ── */
    int ns[4] = { 64, 60, 61, 128 };
    for (int c8 = 0; c8 < 4; c8++) {
        int N = ns[c8]; size_t K = 8, n = (size_t)N*K;
        stride_conv_t *cv = conv1d(N, K, &reg);
        double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*hr=AALLOC(n*8),*hi=AALLOC(n*8);
        double *rr=AALLOC(n*8),*ri=AALLOC(n*8),*er=AALLOC(n*8),*ei=AALLOC(n*8);
        srand(100+N);
        for(size_t i=0;i<n;i++){ xr[i]=frand(); xi[i]=frand();
                                 hr[i]=frand(); hi[i]=frand(); }
        stride_conv_set_kernel(cv, hr, hi);

        /* conv: per-lane direct reference (lane l = stride-K slice) */
        memcpy(rr,xr,n*8); memcpy(ri,xi,n*8);
        stride_conv_execute(cv, rr, ri);
        for (size_t l=0;l<K;l++){
            double LX[128],LY[128],LH[128],LG[128],OR[128],OI[128];
            for(int t=0;t<N;t++){ LX[t]=xr[(size_t)t*K+l]; LY[t]=xi[(size_t)t*K+l];
                                  LH[t]=hr[(size_t)t*K+l]; LG[t]=hi[(size_t)t*K+l]; }
            direct_circ(1,&N,0,LX,LY,LH,LG,OR,OI);
            for(int t=0;t<N;t++){ er[(size_t)t*K+l]=OR[t]; ei[(size_t)t*K+l]=OI[t]; }
        }
        char tag[64]; snprintf(tag,64,"1D conv N=%d K=8%s",N,N==61?" (prime)":"");
        check(tag, max_rel(rr,ri,er,ei,n), 1e-12);

        /* correlate */
        memcpy(rr,xr,n*8); memcpy(ri,xi,n*8);
        stride_conv_correlate(cv, rr, ri);
        for (size_t l=0;l<K;l++){
            double LX[128],LY[128],LH[128],LG[128],OR[128],OI[128];
            for(int t=0;t<N;t++){ LX[t]=xr[(size_t)t*K+l]; LY[t]=xi[(size_t)t*K+l];
                                  LH[t]=hr[(size_t)t*K+l]; LG[t]=hi[(size_t)t*K+l]; }
            direct_circ(1,&N,1,LX,LY,LH,LG,OR,OI);
            for(int t=0;t<N;t++){ er[(size_t)t*K+l]=OR[t]; ei[(size_t)t*K+l]=OI[t]; }
        }
        snprintf(tag,64,"1D corr N=%d K=8%s",N,N==61?" (prime)":"");
        check(tag, max_rel(rr,ri,er,ei,n), 1e-12);

        AFREE(xr);AFREE(xi);AFREE(hr);AFREE(hi);
        AFREE(rr);AFREE(ri);AFREE(er);AFREE(ei);
        stride_conv_destroy(cv);
    }

    /* ── impulse identity + shift theorem (N=96, K=8) ── */
    {
        int N=96; size_t K=8, n=(size_t)N*K;
        stride_conv_t *cv = conv1d(N,K,&reg);
        double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*hr=AALLOC(n*8),*hi=AALLOC(n*8);
        double *rr=AALLOC(n*8),*ri=AALLOC(n*8);
        srand(7);
        for(size_t i=0;i<n;i++){ xr[i]=frand(); xi[i]=frand(); hr[i]=0; hi[i]=0; }
        for(size_t l=0;l<K;l++) hr[0*K+l]=1.0;         /* delta */
        stride_conv_set_kernel(cv,hr,hi);
        memcpy(rr,xr,n*8); memcpy(ri,xi,n*8);
        stride_conv_execute(cv,rr,ri);
        check("impulse identity N=96", max_rel(rr,ri,xr,xi,n), 1e-13);

        int s=17;
        for(size_t i=0;i<n;i++){ hr[i]=0; hi[i]=0; }
        for(size_t l=0;l<K;l++) hr[(size_t)s*K+l]=1.0; /* delta shifted */
        stride_conv_set_kernel(cv,hr,hi);
        memcpy(rr,xr,n*8); memcpy(ri,xi,n*8);
        stride_conv_execute(cv,rr,ri);
        double *er=AALLOC(n*8),*ei=AALLOC(n*8);
        for(int t=0;t<N;t++){ int j=(t-s+N)%N;
            for(size_t l=0;l<K;l++){ er[(size_t)t*K+l]=xr[(size_t)j*K+l];
                                     ei[(size_t)t*K+l]=xi[(size_t)j*K+l]; } }
        check("shift theorem s=17 N=96", max_rel(rr,ri,er,ei,n), 1e-13);
        AFREE(xr);AFREE(xi);AFREE(hr);AFREE(hi);AFREE(rr);AFREE(ri);AFREE(er);AFREE(ei);
        stride_conv_destroy(cv);
    }

    /* ── 2D circular conv via fftnd rank-2 (32x48) ── */
    {
        int N2v[2]={32,48}; size_t n=(size_t)32*48;
        stride_fftnd_data_t tmp; memset(&tmp,0,sizeof tmp);
        tmp.rank=2; tmp.N[0]=32; tmp.N[1]=48; _fftnd_fill_ok(&tmp);
        size_t B=_fftnd_choose_tile(48,tmp.O[1]);
        stride_plan_t *pl[FFTND_MAX_RANK]={0};
        pl[0]=vfft_proto_auto_plan_dispatch(32,tmp.K[0],&reg,NULL);
        pl[1]=vfft_proto_auto_plan_dispatch(48,B,&reg,NULL);
        stride_conv_t *cv = stride_conv_wrap(
            stride_plan_nd_from(2,N2v,B,1,NULL,pl), 1);
        double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*hr=AALLOC(n*8),*hi=AALLOC(n*8);
        double *rr=AALLOC(n*8),*ri=AALLOC(n*8),*er=AALLOC(n*8),*ei=AALLOC(n*8);
        srand(22);
        for(size_t i=0;i<n;i++){ xr[i]=frand(); xi[i]=frand();
                                 hr[i]=frand(); hi[i]=frand(); }
        stride_conv_set_kernel(cv,hr,hi);
        memcpy(rr,xr,n*8); memcpy(ri,xi,n*8);
        stride_conv_execute(cv,rr,ri);
        direct_circ(2,N2v,0,xr,xi,hr,hi,er,ei);
        check("2D conv 32x48 (fftnd r2)", max_rel(rr,ri,er,ei,n), 1e-12);
        AFREE(xr);AFREE(xi);AFREE(hr);AFREE(hi);
        AFREE(rr);AFREE(ri);AFREE(er);AFREE(ei);
        stride_conv_destroy(cv);
    }

    /* ── 3D circular conv via fft3d (16x12x20) ── */
    {
        int N3v[3]={16,12,20}; size_t n=(size_t)16*12*20;
        size_t K0=(size_t)12*20, NR=(size_t)16*12, B=_fft3d_choose_tile(20,NR);
        stride_plan_t *p0=vfft_proto_auto_plan_dispatch(16,K0,&reg,NULL);
        stride_plan_t *p1=vfft_proto_auto_plan_dispatch(12,(size_t)20,&reg,NULL);
        stride_plan_t *pr=vfft_proto_auto_plan_dispatch(20,B,&reg,NULL);
        stride_conv_t *cv = stride_conv_wrap(
            stride_plan_3d_from(16,12,20,B,0,p0,p1,pr), 1);
        double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*hr=AALLOC(n*8),*hi=AALLOC(n*8);
        double *rr=AALLOC(n*8),*ri=AALLOC(n*8),*er=AALLOC(n*8),*ei=AALLOC(n*8);
        srand(33);
        for(size_t i=0;i<n;i++){ xr[i]=frand(); xi[i]=frand();
                                 hr[i]=frand(); hi[i]=frand(); }
        stride_conv_set_kernel(cv,hr,hi);
        memcpy(rr,xr,n*8); memcpy(ri,xi,n*8);
        stride_conv_execute(cv,rr,ri);
        direct_circ(3,N3v,0,xr,xi,hr,hi,er,ei);
        check("3D conv 16x12x20 (fft3d)", max_rel(rr,ri,er,ei,n), 1e-12);
        AFREE(xr);AFREE(xi);AFREE(hr);AFREE(hi);
        AFREE(rr);AFREE(ri);AFREE(er);AFREE(ei);
        stride_conv_destroy(cv);
    }

    /* ── linear conv pattern: Lx=100, Lh=37 -> next_fast_n(136)=? pad ── */
    {
        int Lx=100, Lh=37, L=Lx+Lh-1;
        int N=(int)stride_conv_next_fast_n((size_t)L);
        size_t K=8, n=(size_t)N*K;
        stride_conv_t *cv=conv1d(N,K,&reg);
        double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*hr=AALLOC(n*8),*hi=AALLOC(n*8);
        double *rr=AALLOC(n*8),*ri=AALLOC(n*8);
        memset(xr,0,n*8); memset(xi,0,n*8); memset(hr,0,n*8); memset(hi,0,n*8);
        srand(44);
        double LX[100],LY[100],LH[37],LG[37];
        for(int t=0;t<Lx;t++){ LX[t]=frand(); LY[t]=frand();
            for(size_t l=0;l<K;l++){ xr[(size_t)t*K+l]=LX[t]; xi[(size_t)t*K+l]=LY[t]; } }
        for(int t=0;t<Lh;t++){ LH[t]=frand(); LG[t]=frand();
            for(size_t l=0;l<K;l++){ hr[(size_t)t*K+l]=LH[t]; hi[(size_t)t*K+l]=LG[t]; } }
        stride_conv_set_kernel(cv,hr,hi);
        memcpy(rr,xr,n*8); memcpy(ri,xi,n*8);
        stride_conv_execute(cv,rr,ri);
        /* direct linear */
        double mx=0, sc=0;
        for(int m=0;m<L;m++){
            double sr=0,si=0;
            for(int k=0;k<Lh;k++){ int j=m-k; if(j<0||j>=Lx) continue;
                sr += LX[j]*LH[k]-LY[j]*LG[k];
                si += LX[j]*LG[k]+LY[j]*LH[k]; }
            double m0=fabs(sr)+fabs(si); if(m0>sc)sc=m0;
            double e=fabs(rr[(size_t)m*K]-sr)+fabs(ri[(size_t)m*K]-si);
            if(e>mx)mx=e;
        }
        char tag[64]; snprintf(tag,64,"linear conv 100*37 -> N=%d",N);
        check(tag, mx/(sc?sc:1), 1e-13);
        AFREE(xr);AFREE(xi);AFREE(hr);AFREE(hi);AFREE(rr);AFREE(ri);
        stride_conv_destroy(cv);
    }

    /* ── MT pointwise sweep (N=4096, K=64 -> 262144 elems > threshold) ── */
    {
        int N=4096; size_t K=64, n=(size_t)N*K;
        stride_conv_t *cv=conv1d(N,K,&reg);
        double *xr=AALLOC(n*8),*xi=AALLOC(n*8),*hr=AALLOC(n*8),*hi=AALLOC(n*8);
        double *r1=AALLOC(n*8),*i1=AALLOC(n*8),*r4=AALLOC(n*8),*i4=AALLOC(n*8);
        srand(55);
        for(size_t i=0;i<n;i++){ xr[i]=frand(); xi[i]=frand();
                                 hr[i]=frand(); hi[i]=frand(); }
        stride_conv_set_kernel(cv,hr,hi);
        stride_set_num_threads(1);
        memcpy(r1,xr,n*8); memcpy(i1,xi,n*8);
        stride_conv_execute(cv,r1,i1);
        stride_set_num_threads(4);
        memcpy(r4,xr,n*8); memcpy(i4,xi,n*8);
        stride_conv_execute(cv,r4,i4);
        stride_set_num_threads(1);
        int eq = !memcmp(r1,r4,n*8) && !memcmp(i1,i4,n*8);
        if(!eq) g_fail++;
        printf("  %-38s %s\n","MT pointwise T=4 vs T=1 (bit)",
               eq?"EXACT OK":"**MISMATCH**");
        AFREE(xr);AFREE(xi);AFREE(hr);AFREE(hi);
        AFREE(r1);AFREE(i1);AFREE(r4);AFREE(i4);
        stride_conv_destroy(cv);
    }

    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
