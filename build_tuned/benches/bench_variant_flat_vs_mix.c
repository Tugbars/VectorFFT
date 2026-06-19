/* bench_variant_flat_vs_mix.c — per-stage twiddle-variant mixing payoff.
 *
 * For three high-N cells, build the SAME factorization two ways and compare:
 *   (1) ALL-FLAT  — every twiddled stage uses variant 0 (t1, full table)
 *   (2) WISDOM MIX — the per-stage variant assignment from spike_wisdom.txt
 * Variant codes: 0=FLAT(t1), 1=LOG3, 2=T1S. (Stage 0 is n1 regardless.)
 *
 * Single-thread, caller pinned core 0. PACING (sleep) between every measured
 * phase so the CPU isn't mid-ramp / thermal-drifting across the A/B — cleaner
 * flat-vs-mix delta. Correctness: mix output == flat output (same math, only
 * the twiddle rendering differs).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_variant_flat_vs_mix.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(_WIN32)
#include <windows.h>
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
static void pace(int ms){ Sleep(ms); }
static double now_ns(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart; }
#else
#include <time.h>
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
static void pace(int ms){ struct timespec t={ms/1000,(long)(ms%1000)*1000000L}; nanosleep(&t,NULL); }
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
#endif

#include "executor.h"
#include "planner.h"
#include "threads.h"
#include "env.h"
#include "generator/generated/registry.h"

#define BEST_OF 15
#define PACE_MS 150

typedef struct { int N; size_t K; int nf; int factors[9]; int mix[9]; const char *tag; } cell_t;

static double maxdiff(const double*a,const double*b,size_t n){double e=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);if(d>e)e=d;}return e;}

/* best-of wall-ns for an in-place plan; re-inits input each rep. */
static double bench_plan(const stride_plan_t*p,double*re,double*im,
                         const double*x0,const double*xi0,size_t NK){
    for(int w=0;w<3;w++){memcpy(re,x0,NK*8);memcpy(im,xi0,NK*8);vfft_proto_execute_fwd(p,re,im,p->K);}
    double best=1e30;
    for(int r=0;r<BEST_OF;r++){
        memcpy(re,x0,NK*8);memcpy(im,xi0,NK*8);
        double t0=now_ns(); vfft_proto_execute_fwd(p,re,im,p->K); double v=now_ns()-t0;
        if(v<best)best=v;
    }
    return best;
}

static void run(const cell_t*c, vfft_proto_registry_t*reg){
    size_t NK=(size_t)c->N*c->K;
    int flat[9]; for(int i=0;i<c->nf;i++) flat[i]=0;   /* all-FLAT */
    stride_plan_t*pf=vfft_proto_plan_create(c->N,c->K,c->factors,flat,c->nf,reg);
    stride_plan_t*pm=vfft_proto_plan_create(c->N,c->K,c->factors,c->mix,c->nf,reg);
    if(!pf||!pm){printf("%-7d K=%-3zu  plan NULL (flat=%p mix=%p)\n",c->N,c->K,(void*)pf,(void*)pm);return;}

    double*re=AALLOC(NK*8),*im=AALLOC(NK*8),*x=AALLOC(NK*8),*xi=AALLOC(NK*8),*of=AALLOC(NK*8),*oi_f=AALLOC(NK*8);
    srand(11+c->N); for(size_t i=0;i<NK;i++){x[i]=(double)rand()/RAND_MAX-0.5;xi[i]=(double)rand()/RAND_MAX-0.5;}

    /* correctness: flat vs mix output (same math) */
    memcpy(re,x,NK*8);memcpy(im,xi,NK*8);vfft_proto_execute_fwd(pf,re,im,c->K);memcpy(of,re,NK*8);memcpy(oi_f,im,NK*8);
    memcpy(re,x,NK*8);memcpy(im,xi,NK*8);vfft_proto_execute_fwd(pm,re,im,c->K);
    double err=maxdiff(re,of,NK); double e2=maxdiff(im,oi_f,NK); if(e2>err)err=e2;

    /* paced A/B */
    pace(PACE_MS);
    double tf=bench_plan(pf,re,im,x,xi,NK);
    pace(PACE_MS);
    double tm=bench_plan(pm,re,im,x,xi,NK);

    /* variant string for the mix */
    char vs[32]={0}; for(int i=0;i<c->nf;i++){char b[3]; b[0]="FLT"[0]; /*unused*/
        const char*nm=(c->mix[i]==0?"F":c->mix[i]==1?"L":"S"); strcat(vs,nm);} (void)vs;
    char fs[40]={0}; for(int i=0;i<c->nf;i++){char t[8]; snprintf(t,sizeof t,"%d%s",c->factors[i],i+1<c->nf?".":""); strcat(fs,t);}

    printf("%-7d K=%-3zu %-18s mix[%s]  flat=%9.0f ns  mix=%9.0f ns  mix %.3fx %s  (err %.0e)\n",
           c->N,c->K,fs,vs, tf, tm, tf/tm, (tm<tf?"FASTER":"slower"), err);
    fflush(stdout);

    AFREE(re);AFREE(im);AFREE(x);AFREE(xi);AFREE(of);AFREE(oi_f);
    vfft_proto_plan_destroy(pf); vfft_proto_plan_destroy(pm);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(0)!=0) fprintf(stderr,"warn pin\n");
    stride_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    printf("== twiddle-variant MIX vs ALL-FLAT (single-thread, core0-pinned, paced) ==\n");
    printf("# variants: F=FLAT(t1)  S=T1S  L=LOG3. stage0 is n1. mix = spike_wisdom assignment.\n");
    printf("# mix>1.0x => the per-stage mix beats uniform all-FLAT.\n\n");

    cell_t cells[] = {
        { 100000, 4,  4, {10,16,25,25},      {0,2,2,1},        "low-K high-N" },
        {  65536, 32, 7, {4,4,4,4,4,4,16},   {0,2,2,2,2,2,2},  "high-K high-N" },
        {  60060, 4,  5, {12,11,13,5,7},     {0,2,2,2,1},      "low-K high-N composite" },
    };
    for(size_t i=0;i<sizeof cells/sizeof cells[0];i++){
        run(&cells[i],&reg);
        pace(300);   /* inter-cell pacing */
    }
    stride_set_num_threads(1);
    return 0;
}
