/* bench_inplace_oddk.c — in-place 1D C2C arbitrary-K vs MKL, using the CALIBRATED
 * spike_wisdom plans, compared to vfft_perf_tuned_1d_avg.csv (the v1.0 reference).
 *
 * Mirrors bench_oop_oddk methodology (measure_ab): per cell, take the wisdom
 * factorization (the per-(N,K) calibrated plan — NOT a hardcoded one), build the
 * in-place plan, time it vs MKL DFTI_INPLACE split (REAL_REAL), best-of-5 min,
 * cachebust + cool between engines, order-flip (both flips reported). Bench at the
 * reference even K (matches the CSV) + its odd neighbours; print the CSV ratio
 * inline for the even cell so the regression is visible and the odd-K margin is
 * compared to the calibrated baseline. Correctness = roundtrip fwd+bwd == N*x.
 *
 * Wisdom factorization is K-dependent, so for an odd K we use the same N's
 * NEAREST-K wisdom entry (the closest calibrated plan).
 *
 * Build: python build.py --src benches/bench_inplace_oddk.c --mkl
 * Run  : MKL on PATH + MKL_THREADING_LAYER=SEQUENTIAL, run from benches/.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "executor.h"
#include "planner.h"
#include "dp_planner.h"   /* vfft_proto_now_ns (static inline) */
#include <mkl_dfti.h>
#include <mkl_service.h>

#define MAXF 12
typedef struct { int N; int K; int nf; int f[MAXF]; int v[MAXF]; int use_dif; } went_t;
static went_t WIS[4096]; static int NWIS = 0;

static void load_wisdom(const char *path)
{
    FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "no wisdom %s\n", path); exit(1); }
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        if (line[0]=='#' || line[0]=='@' || line[0]=='\n') continue;
        char *s, *t = strtok_r(line, " \t\n", &s);
        if (!t) continue;
        went_t e; memset(&e, 0, sizeof e);
        e.N = atoi(t);
        t = strtok_r(NULL," \t\n",&s); if(!t) continue; e.K = atoi(t);
        t = strtok_r(NULL," \t\n",&s); if(!t) continue; e.nf = atoi(t);
        if (e.nf < 1 || e.nf > MAXF) continue;
        int ok = 1;
        for (int i=0;i<e.nf;i++){ t=strtok_r(NULL," \t\n",&s); if(!t){ok=0;break;} e.f[i]=atoi(t); }
        if(!ok) continue;
        t = strtok_r(NULL," \t\n",&s); if(!t) continue; /* best_ns (skip) */
        /* 4 flags: use_blocked split_stage block_groups use_dif */
        int flags[4];
        for (int i=0;i<4;i++){ t=strtok_r(NULL," \t\n",&s); if(!t){ok=0;break;} flags[i]=atoi(t); }
        if(!ok) continue;
        e.use_dif = flags[3];
        for (int i=0;i<e.nf;i++){ t=strtok_r(NULL," \t\n",&s); e.v[i]= t?atoi(t):0; }
        if (NWIS < 4096) WIS[NWIS++] = e;
    }
    fclose(f);
}
/* nearest-K wisdom entry for N (factorization is N-dominant, K-tuned). */
static const went_t *wis_lookup(int N, int K)
{
    const went_t *best = NULL; int bestd = 1<<30;
    for (int i=0;i<NWIS;i++) if (WIS[i].N==N) {
        int d = abs(WIS[i].K - K);
        if (d < bestd) { bestd = d; best = &WIS[i]; }
    }
    return best;
}

/* reference ratios from vfft_perf_tuned_1d_avg.csv: (N,K)->ratio. */
typedef struct { int N,K; double ratio; char plan[48]; } ref_t;
static ref_t REF[4096]; static int NREF=0;
static void load_ref(const char *path)
{
    FILE *f=fopen(path,"r"); if(!f) return;
    char line[2048]; int first=1;
    while (fgets(line,sizeof line,f)){
        if(first){first=0;continue;}
        char *s,*t=strtok_r(line,",\n",&s); if(!t)continue; ref_t r; r.N=atoi(t);
        t=strtok_r(NULL,",\n",&s); if(!t)continue; r.K=atoi(t);
        t=strtok_r(NULL,",\n",&s); if(!t)continue; snprintf(r.plan,sizeof r.plan,"%s",t); /* plan */
        t=strtok_r(NULL,",\n",&s); /* path */
        t=strtok_r(NULL,",\n",&s); /* vfft_ns */
        t=strtok_r(NULL,",\n",&s); /* mkl_ns */
        t=strtok_r(NULL,",\n",&s); /* gflops */
        t=strtok_r(NULL,",\n",&s); if(!t)continue; r.ratio=atof(t);
        if(NREF<4096) REF[NREF++]=r;
    }
    fclose(f);
}
static const ref_t *ref_lookup(int N,int K){ for(int i=0;i<NREF;i++) if(REF[i].N==N&&REF[i].K==K) return &REF[i]; return NULL; }

static double *ad(size_t n){ double*p=NULL; if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){exit(1);} return p; }
static void afree(double*p){ vfft_proto_aligned_free(p); }
static void pace(int ms){ if(ms>0){ struct timespec ts={ms/1000,(long)(ms%1000)*1000000L}; nanosleep(&ts,NULL);} }
static void cachebust(void){ size_t s=32*1024*1024/sizeof(double); double*j=ad(s); for(size_t i=0;i<s;i++)j[i]=(double)i; volatile double a=0; for(size_t i=0;i<s;i++)a+=j[i]; (void)a; afree(j); }
static int reps_for(size_t t){ int r=(int)(50000000ull/(t?t:1)); return r<200?200:r; }

static DFTI_DESCRIPTOR_HANDLE mkl_make(int N,size_t K){
    DFTI_DESCRIPTOR_HANDLE d=NULL; MKL_LONG str[2]={0,(MKL_LONG)K};
    if(DftiCreateDescriptor(&d,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N)!=DFTI_NO_ERROR) return NULL;
    DftiSetValue(d,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL);
    DftiSetValue(d,DFTI_PLACEMENT,DFTI_INPLACE);
    DftiSetValue(d,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(d,DFTI_INPUT_DISTANCE,1); DftiSetValue(d,DFTI_OUTPUT_DISTANCE,1);
    DftiSetValue(d,DFTI_INPUT_STRIDES,str); DftiSetValue(d,DFTI_OUTPUT_STRIDES,str);
    if(DftiCommitDescriptor(d)!=DFTI_NO_ERROR){ DftiFreeDescriptor(&d); return NULL; }
    return d;
}
static double best_mkl(int N,size_t K,const double*sr,const double*si,size_t total){
    DFTI_DESCRIPTOR_HANDLE d=mkl_make(N,K); if(!d) return 0;
    double*re=ad(total),*im=ad(total);
    for(int w=0;w<10;w++){ memcpy(re,sr,total*8); memcpy(im,si,total*8); DftiComputeForward(d,re,im); }
    int reps=reps_for(total); double best=1e18;
    for(int t=0;t<5;t++){ memcpy(re,sr,total*8); memcpy(im,si,total*8); double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) DftiComputeForward(d,re,im);
        double e=(vfft_proto_now_ns()-t0)/reps; if(e<best)best=e; }
    afree(re); afree(im); DftiFreeDescriptor(&d); return best;
}
static double best_vfft(stride_plan_t*p,const double*sr,const double*si,double*re,double*im,size_t K,size_t total){
    for(int w=0;w<10;w++){ memcpy(re,sr,total*8); memcpy(im,si,total*8); vfft_proto_execute_fwd(p,re,im,K); }
    int reps=reps_for(total); double best=1e18;
    for(int t=0;t<5;t++){ memcpy(re,sr,total*8); memcpy(im,si,total*8); double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) vfft_proto_execute_fwd(p,re,im,K);
        double e=(vfft_proto_now_ns()-t0)/reps; if(e<best)best=e; }
    return best;
}

static void cell(int N,size_t K,vfft_proto_registry_t*reg,int cool){
    const went_t *w = wis_lookup(N,(int)K);
    if(!w){ printf("  N=%-6d K=%-4zu  no wisdom\n",N,K); return; }
    stride_plan_t *p = vfft_proto_plan_create_ex(N,K,(int*)w->f,(int*)w->v,w->nf,w->use_dif,reg);
    if(!p){ printf("  N=%-6d K=%-4zu  plan NULL\n",N,K); return; }
    char fs[64]; int o=0; fs[0]=0;
    for(int i=0;i<w->nf;i++) o+=snprintf(fs+o,sizeof fs-o,"%s%d",i?"x":"",w->f[i]);
    snprintf(fs+o,sizeof fs-o,"%s", w->use_dif?"/DIF":"/DIT");
    size_t total=(size_t)N*K;
    double *sr=ad(total),*si=ad(total),*re=ad(total),*im=ad(total),*er=ad(total),*ei=ad(total);
    srand(42+N+(int)K);
    for(size_t i=0;i<total;i++){ sr[i]=(double)rand()/RAND_MAX-0.5; si[i]=(double)rand()/RAND_MAX-0.5; }
    memcpy(re,sr,total*8); memcpy(im,si,total*8);
    vfft_proto_execute_fwd(p,re,im,K); vfft_proto_execute_bwd(p,re,im,K);
    double rt=0; for(size_t i=0;i<total;i++){ double a=fabs(re[i]/N-sr[i]),b=fabs(im[i]/N-si[i]); if(a>rt)rt=a; if(b>rt)rt=b; }
    double r0,r1;
    { double m=best_mkl(N,K,sr,si,total); cachebust(); pace(cool); double v=best_vfft(p,sr,si,re,im,K,total); r1=m/v; }
    { double v=best_vfft(p,sr,si,re,im,K,total); cachebust(); pace(cool); double m=best_mkl(N,K,sr,si,total); r0=m/v; }
    const ref_t *rf = ref_lookup(N,(int)K);
    char refbuf[48]; if(rf) snprintf(refbuf,sizeof refbuf,"  <- CSV %.2fx (%s)",rf->ratio,rf->plan); else refbuf[0]=0;
    printf("  N=%-6d K=%-4zu rem%zu %-16s rt=%.0e  flip0=%.2fx flip1=%.2fx avg=%.2fx%s\n",
           N,K,K%4,fs,rt,r0,r1,(r0+r1)/2,refbuf);
    afree(sr);afree(si);afree(re);afree(im);afree(er);afree(ei);
    vfft_proto_plan_destroy(p);
}

int main(void)
{
    setvbuf(stdout,NULL,_IONBF,0);
    mkl_set_num_threads(1);
    load_wisdom("../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt");
    load_ref("vfft_perf_tuned_1d_avg.csv");
    printf("in-place 1D C2C arbitrary-K vs MKL (DFTI_INPLACE split, measure_ab). wisdom=%d ref=%d\n",NWIS,NREF);
    printf("plan = calibrated spike_wisdom factorization (nearest-K for odd cells). CSV = vfft_perf_tuned_1d_avg.csv reference.\n");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    int Ns[] = {64, 128, 256, 512, 1024, 4096};
    for (int i=0;i<(int)(sizeof(Ns)/sizeof(Ns[0]));i++){
        int N=Ns[i];
        printf("== N=%d ==\n", N);
        cell(N,32,&reg,80);  /* reference even K (matches CSV) */
        cell(N,31,&reg,80);  /* odd */
        cell(N,33,&reg,80);  /* odd */
    }
    return 0;
}
