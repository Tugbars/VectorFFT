/* bench_headline_3way.c — section 59: the most important transforms,
 * vfft vs FFTW PATIENT vs MKL DFTI, K=256 batched, single thread.
 *
 * Cells: c2c pow2 N=64/256/1024, r2c N=64/256, prime c2c N=127/257
 * (both Rader). All libraries on their HOME layouts: vfft split
 * batched in-place; FFTW interleaved contiguous-per-transform;
 * MKL interleaved, NUMBER_OF_TRANSFORMS=K. DCT rows: FFTW-only
 * elsewhere (bench_trig_vs_fftw) — MKL has no batched DCT.
 * Correctness: vfft vs MKL elementwise at N=64 (c2c, r2c).
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "executor.h"
#include "planner.h"
#include "dp_planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "bluestein.h"
#include "rader.h"
#include "bluestein_wisdom.h"
#include "exhaustive_patient.h"
#include "r2c.h"
#include <fftw3.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cachebust(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}

#define KB 256

static stride_registry_t reg;
static vfft_proto_wisdom_t wis;
static int use_exh = 0;   /* argv[1] == "exh": exhaustive-patient everywhere */

static int is_smooth(int n){ static const int pr[]={2,3,5,7,11,13,17,19,0};
  for(const int*p=pr;*p;p++) while(n%*p==0)n/=*p; return n==1; }

static stride_plan_t *plan_one(int N, size_t K){
  if (!use_exh) return vfft_proto_auto_plan(N, K, &reg, &wis);
  int f[16], nf = 0; double ns = 0;
  stride_plan_t *p = vfft_proto_patient_exhaustive_plan_verbose(
      N, (size_t)K, &reg, f, &nf, &ns, 0);
  if (p){
    printf("# exh N=%-5d K=%zu ->", N, K);
    for (int i = 0; i < nf; i++) printf(" %d", f[i]);
    printf("   (search est %.0f ns)\n", ns);
  }
  return p;
}

static stride_plan_t *build_c2c(int N, size_t K){
  if (is_smooth(N)) return plan_one(N, K);
  stride_plan_t *rin = plan_one(N-1, K);
  return rin ? stride_rader_plan(N, K, K, rin) : NULL;
}

typedef void (*runner)(void *ctx);
static double bench_run(runner f, void *ctx){
  for (int w=0; w<8; w++) f(ctx);
  double t0=now_ns(); f(ctx); double one=now_ns()-t0;
  int reps=(int)(3e7/(one+1)); if(reps<10)reps=10; if(reps>100000)reps=100000;
  double best=1e30;
  for(int t=0;t<5;t++){ double a=now_ns();
    for(int i=0;i<reps;i++) f(ctx);
    double ns=(now_ns()-a)/reps; if(ns<best)best=ns; }
  return best;
}

/* runner contexts */
typedef struct { stride_plan_t *p; double *re, *im; } vctx_t;
static void run_v(void *c){ vctx_t *x=c; stride_execute_fwd(x->p, x->re, x->im); }
typedef struct { stride_plan_t *p; double *in, *or_, *oi; } vrctx_t;
static void run_vr(void *c){ vrctx_t *x=c; stride_execute_r2c(x->p, x->in, x->or_, x->oi); }
typedef struct { fftw_plan p; } fctx_t;
static void run_f(void *c){ fftw_execute(((fctx_t*)c)->p); }
typedef struct { DFTI_DESCRIPTOR_HANDLE h; double *buf; } mctx_t;
static void run_m(void *c){ mctx_t *x=c; DftiComputeForward(x->h, x->buf); }
typedef struct { DFTI_DESCRIPTOR_HANDLE h; double *in, *out; } mrctx_t;
static void run_mr(void *c){ mrctx_t *x=c; DftiComputeForward(x->h, x->in, x->out); }

static void row(const char *kind, int N, double v, double f, double m){
  printf("%-6s %-5d %10.0f %12.0f %12.0f %8.2fx %8.2fx\n",
         kind, N, v, f, m, f/v, m/v);
}

int main(int argc, char **argv){
  if (argc > 1 && strcmp(argv[1], "exh") == 0) use_exh = 1;
  mkl_set_num_threads(1);
  printf("# vfft planning mode: %s\n", use_exh ? "EXHAUSTIVE-PATIENT" : "wisdom auto");
  vfft_proto_registry_init(&reg);
  vfft_proto_wisdom_load(&wis, "core/vfft_wisdom_tuned.txt");

  printf("%-6s %-5s %10s %12s %12s %8s %8s\n",
         "kind","N","vfft_ns","fftw_ns","mkl_ns","fftw/v","mkl/v");

  /* ---------- c2c pow2 + primes ---------- */
  int c2cN[] = {64, 256, 1024, 127, 257};
  for (size_t i = 0; i < 5; i++){
    int N = c2cN[i];
    size_t NK = (size_t)N * KB;
    stride_plan_t *vp = build_c2c(N, KB);
    if (!vp){ printf("%-6s %-5d plan FAIL\n", N==127||N==257?"prime":"c2c", N); continue; }
    double *vre=aligned_alloc(64,NK*8), *vim=aligned_alloc(64,NK*8);
    for(size_t q=0;q<NK;q++){ vre[q]=sin(0.37*(double)q); vim[q]=0.2*cos(2.1*(double)q); }

    fftw_complex *fb = fftw_malloc(NK*sizeof(fftw_complex));
    for (int n=0;n<N;n++) for (int k=0;k<KB;k++){
      fb[(size_t)k*N+n][0]=vre[(size_t)n*KB+k]; fb[(size_t)k*N+n][1]=vim[(size_t)n*KB+k]; }
    int na=N;
    fftw_plan fp = fftw_plan_many_dft(1,&na,KB, fb,NULL,1,N, fb,NULL,1,N,
                                      FFTW_FORWARD, FFTW_PATIENT);

    double *mb = (double*)mkl_malloc(NK*2*8, 64);
    for (int n=0;n<N;n++) for (int k=0;k<KB;k++){
      mb[2*((size_t)k*N+n)]   = vre[(size_t)n*KB+k];
      mb[2*((size_t)k*N+n)+1] = vim[(size_t)n*KB+k]; }
    DFTI_DESCRIPTOR_HANDLE mh;
    DftiCreateDescriptor(&mh, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiSetValue(mh, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)KB);
    DftiSetValue(mh, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
    DftiSetValue(mh, DFTI_OUTPUT_DISTANCE, (MKL_LONG)N);
    DftiCommitDescriptor(mh);

    /* correctness spot at N=64: vfft vs MKL elementwise. Raw stride
     * output is digit-reversed (the planner's native order); compare
     * vfft row perm[n] against MKL natural bin n. */
    if (N == 64 && vp->num_stages > 0){
      vctx_t tv={vp,vre,vim};
      double *sr=malloc(NK*8), *si=malloc(NK*8);
      memcpy(sr,vre,NK*8); memcpy(si,vim,NK*8);
      run_v(&tv); run_m(&(mctx_t){mh,mb});
      int *perm=malloc(N*sizeof(int)), *iperm=malloc(N*sizeof(int));
      _r2c_compute_perm(vp->factors, vp->num_stages, N, perm, iperm);
      double xc=0;
      for (int n=0;n<N;n++) for (int k=0;k<KB;k++){
        double dr=fabs(vre[(size_t)perm[n]*KB+k]-mb[2*((size_t)k*N+n)]);
        double di=fabs(vim[(size_t)perm[n]*KB+k]-mb[2*((size_t)k*N+n)+1]);
        if(dr>xc)xc=dr; if(di>xc)xc=di; }
      printf("# c2c N=64 vfft-vs-MKL xcheck (perm-aware): %.1e %s\n",
             xc, xc<1e-9?"PASS":"FAIL");
      free(perm); free(iperm);
      memcpy(vre,sr,NK*8); memcpy(vim,si,NK*8); free(sr); free(si);
      for (int n=0;n<N;n++) for (int k=0;k<KB;k++){
        mb[2*((size_t)k*N+n)]   = vre[(size_t)n*KB+k];
        mb[2*((size_t)k*N+n)+1] = vim[(size_t)n*KB+k]; }
    }

    vctx_t vc={vp,vre,vim}; fctx_t fc={fp}; mctx_t mc={mh,mb};
    cachebust(); double v=bench_run(run_v,&vc);
    cachebust(); double f=bench_run(run_f,&fc);
    cachebust(); double m=bench_run(run_m,&mc);
    row((N==127||N==257)?"prime":"c2c", N, v, f, m);
    stride_plan_destroy(vp); fftw_destroy_plan(fp);
    DftiFreeDescriptor(&mh); mkl_free(mb); fftw_free(fb);
    free(vre); free(vim);
  }

  /* ---------- r2c even ---------- */
  int rN[] = {64, 256};
  for (size_t i = 0; i < 2; i++){
    int N = rN[i], H = N/2+1;
    size_t NK=(size_t)N*KB;
    stride_plan_t *inner = plan_one(N/2, KB);
    stride_plan_t *vp = inner ? stride_r2c_plan(N, KB, KB, inner) : NULL;
    if (!vp){ printf("r2c    %-5d plan FAIL\n", N); continue; }
    double *x=aligned_alloc(64,NK*8), *yr=aligned_alloc(64,NK*8), *yi=aligned_alloc(64,NK*8);
    for(size_t q=0;q<NK;q++) x[q]=sin(0.37*(double)q)+0.2*cos(2.1*(double)q);

    double *fin=fftw_malloc(NK*8);
    fftw_complex *fout=fftw_malloc((size_t)H*KB*sizeof(fftw_complex));
    for (int n=0;n<N;n++) for (int k=0;k<KB;k++)
      fin[(size_t)k*N+n]=x[(size_t)n*KB+k];
    int na=N;
    fftw_plan fp=fftw_plan_many_dft_r2c(1,&na,KB, fin,NULL,1,N,
                                        fout,NULL,1,H, FFTW_PATIENT);

    double *min_=(double*)mkl_malloc(NK*8,64);
    double *mout=(double*)mkl_malloc((size_t)H*KB*2*8,64);
    for (int n=0;n<N;n++) for (int k=0;k<KB;k++)
      min_[(size_t)k*N+n]=x[(size_t)n*KB+k];
    DFTI_DESCRIPTOR_HANDLE mh;
    DftiCreateDescriptor(&mh, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N);
    DftiSetValue(mh, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)KB);
    DftiSetValue(mh, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mh, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(mh, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
    DftiSetValue(mh, DFTI_OUTPUT_DISTANCE, (MKL_LONG)H);
    DftiCommitDescriptor(mh);

    if (N == 64){
      vrctx_t tv={vp,x,yr,yi}; run_vr(&tv);
      mrctx_t tm={mh,min_,mout}; run_mr(&tm);
      double xc=0;
      for (int h=0;h<H;h++) for (int k=0;k<KB;k++){
        double dr=fabs(yr[(size_t)h*KB+k]-mout[2*((size_t)k*H+h)]);
        double di=fabs(yi[(size_t)h*KB+k]-mout[2*((size_t)k*H+h)+1]);
        if(dr>xc)xc=dr; if(di>xc)xc=di; }
      printf("# r2c N=64 vfft-vs-MKL xcheck: %.1e %s\n", xc, xc<1e-9?"PASS":"FAIL");
    }

    vrctx_t vc={vp,x,yr,yi}; fctx_t fc={fp}; mrctx_t mc={mh,min_,mout};
    cachebust(); double v=bench_run(run_vr,&vc);
    cachebust(); double f=bench_run(run_f,&fc);
    cachebust(); double m=bench_run(run_mr,&mc);
    row("r2c", N, v, f, m);
    stride_plan_destroy(vp); fftw_destroy_plan(fp);
    DftiFreeDescriptor(&mh); mkl_free(min_); mkl_free(mout);
    fftw_free(fin); fftw_free(fout); free(x); free(yr); free(yi);
  }
  return 0;
}
