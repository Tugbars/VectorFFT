/* natorder_pf_probe.c — is the software prefetch in the cycle kernel actually helping?
 * T7 bundled prefetch+lists+AVX and never isolated; T10 found addresses are list-known (OoO can
 * already issue them early) -> prefetch may be redundant/harmful on this memory-bound kernel.
 * Times the cycle pass on REAL digit-reversal perms (chains from wisdom) with prefetch OFF vs
 * distance 2/4/8, paced-averaged (T8). Distances beat OFF only if prefetch genuinely helps.
 * Build: python build.py --src test/natorder_pf_probe.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>
#include "planner.h"          /* STRIDE_MAX_STAGES */
#include "natorder_perm.h"    /* mk_perm, cycle list + offsets */

static double qpc_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

static inline void row_mov(double *dr,double *di,const double *sr,const double *si,size_t K){
    size_t c=0; for(;c+4<=K;c+=4){ _mm256_storeu_pd(dr+c,_mm256_loadu_pd(sr+c));
        _mm256_storeu_pd(di+c,_mm256_loadu_pd(si+c)); }
    for(;c<K;c++){ dr[c]=sr[c]; di[c]=si[c]; } }

/* pfd = prefetch distance (0 = OFF). Fixed runtime param => identical branch overhead across variants. */
static void cyc(double *re,double *im,size_t K,const int *list,const int *off,int ncyc,double *tmp,int pfd){
    double *tr=tmp,*ti=tmp+K;
    for(int c=0;c<ncyc;c++){ const int *s=list+off[c]; int len=0; while(s[len]!=-1) len++;
        memcpy(tr,re+(size_t)s[0]*K,K*8); memcpy(ti,im+(size_t)s[0]*K,K*8);
        for(int i=0;i<len-1;i++){
            if(pfd && i+pfd<len){ _mm_prefetch((const char*)(re+(size_t)s[i+pfd]*K),_MM_HINT_T0);
                                  _mm_prefetch((const char*)(im+(size_t)s[i+pfd]*K),_MM_HINT_T0); }
            row_mov(re+(size_t)s[i]*K,im+(size_t)s[i]*K,re+(size_t)s[i+1]*K,im+(size_t)s[i+1]*K,K);
        }
        memcpy(re+(size_t)s[len-1]*K,tr,K*8); memcpy(im+(size_t)s[len-1]*K,ti,K*8);
    } }

static double timeit(double *re,double *im,size_t n,size_t K,const int *list,const int *off,int ncyc,double *tmp,int pfd){
    for(size_t i=0;i<n;i++){ re[i]=(double)(i&255); im[i]=(double)(i&127); }
    cyc(re,im,K,list,off,ncyc,tmp,pfd);  /* warm-up */
    double sum=0; int inner=200;
    for(int o=0;o<5;o++){ Sleep(120); double t0=qpc_ns();
        for(int r=0;r<inner;r++) cyc(re,im,K,list,off,ncyc,tmp,pfd);
        sum+=(qpc_ns()-t0)/inner; }
    return sum/5.0; }

static void cell(int N,size_t K,const int *f,int nf){
    size_t n=(size_t)N*K;
    double *re=_aligned_malloc(n*8,64),*im=_aligned_malloc(n*8,64),*tmp=_aligned_malloc(2*K*8,64);
    int *M=malloc(N*4); vfft_natorder_mk_perm(N,f,nf,M);
    int *list=vfft_natorder_mk_cycles(N,M); int ncyc; int *off=vfft_natorder_cycle_offsets(list,&ncyc);
    double off0=timeit(re,im,n,K,list,off,ncyc,tmp,0);
    double d2=timeit(re,im,n,K,list,off,ncyc,tmp,2);
    double d4=timeit(re,im,n,K,list,off,ncyc,tmp,4);
    double d8=timeit(re,im,n,K,list,off,ncyc,tmp,8);
    printf("N=%-5d K=%-3zu ncyc=%-5d | OFF %8.0f | d2 %8.0f (%.3f) | d4 %8.0f (%.3f) | d8 %8.0f (%.3f)\n",
        N,K,ncyc,off0,d2,d2/off0,d4,d4/off0,d8,d8/off0);
    _aligned_free(re);_aligned_free(im);_aligned_free(tmp); free(M); free(list); free(off);
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    printf("# prefetch probe: cycle pass, real digit-reversal perms. ratio = dist/OFF (<1 helps, >1 hurts)\n");
    int a[]={4,4,8,8};    cell(1024,32,a,4);
    int b[]={4,4,4,8,8};  cell(4096,32,b,5);
    int c[]={4,4,16};     cell(256,256,c,3);
    int d[]={4,4,8,32};   cell(4096,4,d,4);
    int e[]={64,16};      cell(1024,4,e,2);
    return 0;
}
