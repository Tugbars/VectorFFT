/* ============================================================================
 * VectorFFT - recursive blocked out-of-place engine (natural-order)
 * ----------------------------------------------------------------------------
 * Reference engine for N = 1024 complex forward DFT, batched over K transforms,
 * out-of-place (input preserved), output in NATURAL order.
 *
 * This is the composition layer that the original flat executor lacked: it
 * matches FFTW's *method* (a recursive Cooley-Tukey step with a cache-resident
 * intermediate) using VectorFFT's own AVX-512 OOP codelets as the leaves.
 *
 * DECOMPOSITION (N = N1 x N2 = 64 x 16, DIT):
 *   index map  n = N2*n1 + n2          (n1 in [0,64), n2 in [0,16))
 *   stage 1 (inner, NO twiddle):  N2 sub-DFTs of size N1=64 over n1
 *   twiddle:                      W_N^{n2*k1}
 *   stage 2 (outer, twiddle):     N1 groups, size-N2=16 DFT over n2
 *   output:                       X[k1 + N1*k2] at element (k1 + N1*k2)
 *
 * BLOCKING (the single biggest performance lever, ~3x over the naive recursion):
 *   The K transforms are processed in blocks of V=8 (one AVX-512 double vector).
 *   Layout is block-local element-major:  buf[block*N*V + element*V + lane].
 *   A block is V*N*16 = 128 KB, which stays resident in L2 (1 MB here), so the
 *   inner write / outer read of the intermediate never touch DRAM. The only DRAM
 *   traffic is reading the input block and writing the output block, each once.
 *
 * NATURAL ORDER:
 *   The outer stage writes its legs at out_leg_stride = N1*V to place results
 *   in natural order directly (no separate digit-reversal pass). NOTE: that
 *   stride is exactly 4096 bytes, which equals the L1d set stride on this CPU,
 *   so all legs of a butterfly alias into one L1 set. That aliasing only shows
 *   up when the output is cache-resident (small K); at scale the output write
 *   is DRAM-bandwidth-bound and the aliasing is fully masked. Writing to a
 *   non-power-of-2 padded scratch + a de-pad copy removes the aliasing in
 *   isolation but the extra serialized pass costs more than it saves, so this
 *   direct-write version is the one to ship. See benchmarks/02 and 08.
 *
 * MEASURED (virtualized Sapphire/Emerald Rapids Xeon, gcc-13, FFTW PATIENT):
 *   K=2048 (DRAM-bound):  ~0.93-0.99x of FFTW-oop  (parity)
 *   K=512  (L3-resident): ~0.77-0.82x of FFTW-oop
 *   The individual leaf codelets are FASTER than FFTW's (see benchmarks/04).
 *   The residual K=512 gap is FFTW's recursive-composition advantage in the
 *   cache-resident regime, not any single fixable component (benchmarks/01-06).
 *
 * This is an N=1024-specific reference, not a general planner. It demonstrates
 * the architecture and pins the performance story; generalizing means a real
 * recursive planner (ct calling ct, tensor/iodim, leaf selection).
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"   /* only for verification + benchmark reference */

/* OOP leaves emitted by the OCaml DAG compiler (see codelets/) */
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,
        const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix16_t1s_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,
        const double*,const double*,size_t,size_t,size_t,size_t,size_t);

#define N1 64
#define N2 16
#define N  1024
#define V  8     /* transforms per block; 8 = one ZMM, block = 128 KB (L2-resident). Sweet spot, see benchmarks/06. */

static double twr[N1*(N2-1)], twi[N1*(N2-1)];  /* per-group twiddles W_N^{n2*k1}, leg j in [1,N2) */
static double WR[N*V], WI[N*V];                /* block-local intermediate (L2) */

static void engine_init(void){
    for (int k1 = 0; k1 < N1; k1++)
        for (int n2 = 1; n2 < N2; n2++){
            double a = -2.0*M_PI*(double)(n2*k1)/(double)N;
            twr[k1*(N2-1)+(n2-1)] = cos(a);
            twi[k1*(N2-1)+(n2-1)] = sin(a);
        }
}

/* one block = V transforms, fully cache-resident except the input read / output write */
static inline void block(const double* ir, const double* ii, double* orr, double* oi){
    /* inner: N2 sub-DFTs of size 64, no twiddle, block -> work (block-local strides) */
    radix64_n1_oop_fwd_avx512_UG_UG(ir, ii, WR, WI, 0, 0,
        (size_t)N2*V, 1, (size_t)N2*V, 1, (size_t)N2*V);
    /* outer: N1 groups, size-16 + twiddle, work -> out, natural order via out_leg_stride=N1*V */
    for (int k1 = 0; k1 < N1; k1++)
        radix16_t1s_oop_fwd_avx512_UG_UG(
            WR + (size_t)N2*k1*V, WI + (size_t)N2*k1*V,
            orr + (size_t)k1*V,   oi  + (size_t)k1*V,
            twr + (size_t)k1*(N2-1), twi + (size_t)k1*(N2-1),
            (size_t)V, 1, (size_t)N1*V, 1, (size_t)V);
}

/* engine: vector-rank loop over batch-blocks. Input/output in block-local
 * element-major layout  buf[block*N*V + element*V + lane]. */
void engine(const double* ir, const double* ii, double* orr, double* oi, size_t K){
    for (size_t b = 0; b < K/V; b++)
        block(ir + b*N*V, ii + b*N*V, orr + b*N*V, oi + b*N*V);
}

/* -------------------------- verify + benchmark ----------------------------- */
static unsigned long long mn(unsigned long long* a, int c){
    unsigned long long b = ~0ULL; for (int i=0;i<c;i++) if (a[i]<b) b=a[i]; return b;
}

int main(int argc, char** argv){
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    engine_init();
    size_t K = argc>1 ? atol(argv[1]) : 512;
    size_t TOT = (size_t)N*K;

    double *ir=aligned_alloc(64,TOT*8), *ii=aligned_alloc(64,TOT*8);
    double *orr=aligned_alloc(64,TOT*8), *oi=aligned_alloc(64,TOT*8);
    double *xr=malloc(TOT*8), *xi=malloc(TOT*8);
    srand(11);
    for (size_t t=0;t<K;t++) for (int e=0;e<N;e++){
        double vr=(double)rand()/RAND_MAX-0.5, vi=(double)rand()/RAND_MAX-0.5;
        size_t blk=t/V, lane=t%V;
        xr[t*N+e]=vr; xi[t*N+e]=vi;
        ir[blk*N*V+(size_t)e*V+lane]=vr; ii[blk*N*V+(size_t)e*V+lane]=vi;
    }
    engine(ir,ii,orr,oi,K);

    /* correctness vs FFTW on a few transforms */
    fftw_complex *fin=fftw_malloc(sizeof(fftw_complex)*N), *fout=fftw_malloc(sizeof(fftw_complex)*N);
    fftw_plan p=fftw_plan_dft_1d(N,fin,fout,FFTW_FORWARD,FFTW_ESTIMATE);
    double maxrel=0;
    for (size_t t=0;t<K;t+=(K>3?K/3:1)){ size_t blk=t/V, lane=t%V;
        for (int e=0;e<N;e++){fin[e][0]=xr[t*N+e]; fin[e][1]=xi[t*N+e];}
        fftw_execute(p);
        double me=0, mm=0;
        for (int Kk=0;Kk<N;Kk++){
            double dr=orr[blk*N*V+(size_t)Kk*V+lane]-fout[Kk][0];
            double di=oi [blk*N*V+(size_t)Kk*V+lane]-fout[Kk][1];
            double e=sqrt(dr*dr+di*di), m=hypot(fout[Kk][0],fout[Kk][1]);
            if (e>me) me=e; if (m>mm) mm=m;
        }
        if (me/mm>maxrel) maxrel=me/mm;
    }
    printf("VectorFFT natural-order OOP  N=1024 K=%zu  rel err %.2e  %s\n",
           K, maxrel, maxrel<1e-9 ? "OK" : "BAD");

    /* benchmark vs FFTW-oop (transform-major, PATIENT plan) */
    fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT), *go=fftw_malloc(sizeof(fftw_complex)*TOT);
    for (size_t i=0;i<TOT;i++){ gi[i][0]=sin(0.1*i); gi[i][1]=cos(0.07*i); }
    int nn[1]={N};
    fftw_plan pe=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
    unsigned long long va[64], fa[64];
    for (int w=0;w<6;w++){ engine(ir,ii,orr,oi,K); fftw_execute(pe); }
    for (int r=0;r<60;r++){ unsigned long long c0=__rdtsc(); engine(ir,ii,orr,oi,K); va[r%64]=__rdtsc()-c0; }
    for (int r=0;r<60;r++){ unsigned long long c0=__rdtsc(); fftw_execute(pe);          fa[r%64]=__rdtsc()-c0; }
    unsigned long long E=mn(va,60), F=mn(fa,60);
    printf("  engine %llu cyc | FFTW-oop %llu cyc | %.2fx\n", E, F, (double)F/E);
    return 0;
}
