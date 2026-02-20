/**
 * @file bench_fft_radix2_bv.c
 * @brief Performance benchmark for radix-2 FFT butterfly (twiddle + N1)
 *
 * Reports: wall-clock ns, cycles/butterfly (rdtscp), GB/s, N1 speedup
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <x86intrin.h>
#include "fft_radix2.h"

#define MIN_NS  200000000.0   /* 200ms per benchmark point */
#define TRIALS  7
#define WARMUP  3

static double *aalloc(size_t n) {
    void *p; if(posix_memalign(&p,64,n*sizeof(double))) abort();
    return p;
}
static fft_twiddles_soa *tw_alloc(int h) {
    fft_twiddles_soa *t=malloc(sizeof(*t)); t->re=aalloc(h); t->im=aalloc(h); return t;
}
static void tw_free(fft_twiddles_soa *t) { free((void*)t->re); free((void*)t->im); free(t); }

static void fill(double *b, int n) { for(int i=0;i<n;i++) b[i]=sin(0.7*i+0.3); }
static void tw_fwd(fft_twiddles_soa *t, int h) {
    int N=h*2; for(int k=0;k<h;k++){double a=-2.0*M_PI*k/(double)N; t->re[k]=cos(a); t->im[k]=sin(a);}
}

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e9 + ts.tv_nsec;
}
static int dcmp(const void *a, const void *b) {
    double x=*(double*)a, y=*(double*)b; return (x>y)-(x<y);
}
static double median(double *a, int n) { qsort(a,n,sizeof(double),dcmp); return a[n/2]; }

typedef struct { double ns; double cyc; double gbs; } result_t;

static result_t bench_tw(int N) {
    int h=N/2;
    double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N);
    fft_twiddles_soa *tw=tw_alloc(h); fill(ir,N); fill(ii,N); tw_fwd(tw,h);
    for(int w=0;w<WARMUP;w++) fft_radix2_bv(or_,oi,ir,ii,tw,h);

    double tns[TRIALS], tcyc[TRIALS];
    for(int t=0;t<TRIALS;t++){
        long it=1; double el=0;
        while(1){double t0=now_ns(); for(long i=0;i<it;i++) fft_radix2_bv(or_,oi,ir,ii,tw,h);
            el=now_ns()-t0; if(el>=MIN_NS) break; it*=2;}
        double t0=now_ns(); unsigned aux; unsigned long long c0=__rdtscp(&aux);
        for(long i=0;i<it;i++) fft_radix2_bv(or_,oi,ir,ii,tw,h);
        unsigned long long c1=__rdtscp(&aux); _mm_lfence(); double t1=now_ns();
        tns[t]=(t1-t0)/(double)it; tcyc[t]=(double)(c1-c0)/((double)it*h);
    }
    result_t r; r.ns=median(tns,TRIALS); r.cyc=median(tcyc,TRIALS);
    r.gbs=(double)h*80.0/r.ns;  /* 80 bytes/bf: 6 reads + 4 writes */
    free(ir);free(ii);free(or_);free(oi); tw_free(tw); return r;
}

static result_t bench_n1(int N) {
    int h=N/2;
    double *ir=aalloc(N),*ii=aalloc(N),*or_=aalloc(N),*oi=aalloc(N);
    fill(ir,N); fill(ii,N);
    for(int w=0;w<WARMUP;w++) fft_radix2_bv_n1(or_,oi,ir,ii,h);

    double tns[TRIALS], tcyc[TRIALS];
    for(int t=0;t<TRIALS;t++){
        long it=1; double el=0;
        while(1){double t0=now_ns(); for(long i=0;i<it;i++) fft_radix2_bv_n1(or_,oi,ir,ii,h);
            el=now_ns()-t0; if(el>=MIN_NS) break; it*=2;}
        double t0=now_ns(); unsigned aux; unsigned long long c0=__rdtscp(&aux);
        for(long i=0;i<it;i++) fft_radix2_bv_n1(or_,oi,ir,ii,h);
        unsigned long long c1=__rdtscp(&aux); _mm_lfence(); double t1=now_ns();
        tns[t]=(t1-t0)/(double)it; tcyc[t]=(double)(c1-c0)/((double)it*h);
    }
    result_t r; r.ns=median(tns,TRIALS); r.cyc=median(tcyc,TRIALS);
    r.gbs=(double)h*64.0/r.ns;  /* 64 bytes/bf: 4 reads + 4 writes */
    free(ir);free(ii);free(or_);free(oi); return r;
}

int main(void) {
    printf("================================================================\n");
    printf(" Radix-2 Butterfly Benchmark\n");
    printf(" SIMD: %s | vw=%d | align=%zu\n",
           radix2_get_simd_capabilities(), radix2_get_vector_width(),
           radix2_get_alignment_requirement());
    printf(" Trials=%d (median), min=%.0fms\n", TRIALS, MIN_NS/1e6);
    printf("================================================================\n\n");

    printf("%-10s | %-10s %-8s %-8s | %-10s %-8s %-8s | %s\n",
           "N","tw(ns)","cyc/bf","GB/s","n1(ns)","cyc/bf","GB/s","speedup");
    printf("───────────┼─────────────────────────────────┼"
           "─────────────────────────────────┼────────\n");

    int sizes[]={16,64,256,1024,4096,16384,65536,262144,1048576};
    for(int i=0;i<(int)(sizeof(sizes)/sizeof(sizes[0]));i++){
        int N=sizes[i];
        result_t tw=bench_tw(N), n1=bench_n1(N);
        printf("%-10d | %9.1f %7.2f %7.2f  | %9.1f %7.2f %7.2f  | %.2fx\n",
               N, tw.ns, tw.cyc, tw.gbs, n1.ns, n1.cyc, n1.gbs, tw.ns/n1.ns);
        fflush(stdout);
    }

    printf("\n── Detail: N=4096 ──\n");
    result_t tw=bench_tw(4096), n1=bench_n1(4096);
    printf("  twiddle:  %.1f ns, %.2f cyc/bf, %.2f GB/s, %.1f M-bf/s\n",
           tw.ns, tw.cyc, tw.gbs, 2048.0/tw.ns*1e3);
    printf("  N1:       %.1f ns, %.2f cyc/bf, %.2f GB/s, %.1f M-bf/s\n",
           n1.ns, n1.cyc, n1.gbs, 2048.0/n1.ns*1e3);
    printf("  speedup:  %.2fx\n", tw.ns/n1.ns);
    printf("================================================================\n");
    return 0;
}
