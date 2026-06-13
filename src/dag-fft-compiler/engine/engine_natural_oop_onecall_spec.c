/* ============================================================================
 * engine_natural_oop_onecall_spec.c
 *
 * Stride-specialized + M-project variant of the 32x32 one-call engine.
 * Identical algorithm and data layout to engine_natural_oop_onecall.c, but
 * links the *_spec radix-32 codelets, whose four strides are baked as
 * compile-time constants (--oop-strides) and whose PASS-2 trailing sub-DFTs
 * are kept register-resident (--fuse 8, M-project). See:
 *   docs/oop_stride_specialization.md   (what / how / why)
 *   docs/OOP_DESIGN.md section 6.x      (summary + numbers)
 *
 * The specialized codelets take no stride arguments (7-arg ABI ending in me);
 * the strides are fixed to this engine's call sites:
 *   leaf n1 : in_leg=256 in_grp=1 out_leg=8   out_grp=32
 *   t1p     : in_leg=256 in_grp=1 out_leg=256 out_grp=1
 *
 * Measured (AVX-512, noisy VM, interleaved vs the general codelets, FFTW
 * PATIENT, rdtsc min-of-120): specialized is ~6-10% faster than general at
 * every K; log3_spec beats FFTW by ~1.15-1.23x. relerr ~8e-15 (machine
 * precision, identical to general). Re-confirm magnitude on quiet hardware.
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <fftw3.h>

/* Specialized codelets: strides baked, 7-arg (last arg = me). */
extern void radix32_n1_oop_fwd_avx512_UG_UG_spec(const double*,const double*,double*,double*,const double*,const double*,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG_spec(const double*,const double*,double*,double*,const double*,const double*,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG_log3_spec(const double*,const double*,double*,double*,const double*,const double*,size_t);

#define N 1024
#define V 8
static double Q2r[31*32],Q2i[31*32];   /* tw[(l2-1)*32+k2] = W_1024^{l2 k2}; same table for flat and log3 */
static void initw(void){ for(int l2=1;l2<32;l2++)for(int k2=0;k2<32;k2++){double a=-2.0*M_PI*(double)(l2*k2)/1024.0; Q2r[(l2-1)*32+k2]=cos(a);Q2i[(l2-1)*32+k2]=sin(a);} }

/* 32x32, flat t1p twiddle stage (specialized). */
static inline void blkFs(const double*ir,const double*ii,double*o,double*oi){
  radix32_n1_oop_fwd_avx512_UG_UG_spec(ir,ii,o,oi,0,0,(size_t)32*V);
  radix32_t1p_oop_fwd_avx512_UG_UG_spec(o,oi,o,oi,Q2r,Q2i,(size_t)32*V);
}
/* 32x32, log3 t1p twiddle stage (specialized, RECOMMENDED). */
static inline void blkLs(const double*ir,const double*ii,double*o,double*oi){
  radix32_n1_oop_fwd_avx512_UG_UG_spec(ir,ii,o,oi,0,0,(size_t)32*V);
  radix32_t1p_oop_fwd_avx512_UG_UG_log3_spec(o,oi,o,oi,Q2r,Q2i,(size_t)32*V);
}
static void eFs(const double*ir,const double*ii,double*o,double*oi,size_t K){for(size_t b=0;b<K/V;b++)blkFs(ir+b*N*V,ii+b*N*V,o+b*N*V,oi+b*N*V);}
static void eLs(const double*ir,const double*ii,double*o,double*oi,size_t K){for(size_t b=0;b<K/V;b++)blkLs(ir+b*N*V,ii+b*N*V,o+b*N*V,oi+b*N*V);}

static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
static double verify(void(*eng)(const double*,const double*,double*,double*,size_t),const double*ir,const double*ii,double*o,double*oi,const double*xr,const double*xi,size_t K){
  eng(ir,ii,o,oi,K); fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N),*fo=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan p=fftw_plan_dft_1d(N,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE); double mr=0;
  for(size_t t=0;t<K;t+=(K>3?K/3:1)){size_t bk=t/V,l=t%V;for(int e=0;e<N;e++){fi[e][0]=xr[t*N+e];fi[e][1]=xi[t*N+e];}fftw_execute(p);
    double me=0,mm=0;for(int k=0;k<N;k++){double dr=o[bk*N*V+(size_t)k*V+l]-fo[k][0],di=oi[bk*N*V+(size_t)k*V+l]-fo[k][1];double e=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);if(e>me)me=e;if(m>mm)mm=m;}if(me/mm>mr)mr=me/mm;}
  fftw_destroy_plan(p);fftw_free(fi);fftw_free(fo);return mr;
}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw(); size_t K=argc>1?atol(argv[1]):512, TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8); srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;size_t bk=t/V,l=t%V;xr[t*N+e]=vr;xi[t*N+e]=vi;ir[bk*N*V+(size_t)e*V+l]=vr;ii[bk*N*V+(size_t)e*V+l]=vi;}
  double rFs=verify(eFs,ir,ii,o,oi,xr,xi,K), rLs=verify(eLs,ir,ii,o,oi,xr,xi,K);
  printf("K=%zu  flat_spec relerr %.2e %s   log3_spec relerr %.2e %s\n",K,rFs,rFs<1e-9?"OK":"BAD",rLs,rLs<1e-9?"OK":"BAD");
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  for(size_t i=0;i<TOT;i++){gi[i][0]=sin(0.1*i);gi[i][1]=cos(0.07*i);}
  int nn[1]={N};fftw_plan pe=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
  unsigned long long aF[128],aL[128],ap[128];
  for(int w=0;w<8;w++){eFs(ir,ii,o,oi,K);eLs(ir,ii,o,oi,K);fftw_execute(pe);}
  for(int r=0;r<120;r++){
    unsigned long long c;
    c=__rdtsc(); eFs(ir,ii,o,oi,K); aF[r%128]=__rdtsc()-c;
    c=__rdtsc(); eLs(ir,ii,o,oi,K); aL[r%128]=__rdtsc()-c;
    c=__rdtsc(); fftw_execute(pe);  ap[r%128]=__rdtsc()-c;
  }
  unsigned long long F=mn(aF,120),L=mn(aL,120),P=mn(ap,120);
  printf("  32x32 flat_spec %llu (%.3fx)  |  32x32 log3_spec %llu (%.3fx)  |  FFTW %llu  [log3/flat=%.3f]\n",
         F,(double)P/F,L,(double)P/L,P,(double)L/F);
  return 0;
}
