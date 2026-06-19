/* Decompose the 161us: per-phase timing + factor-order sweep. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "rfft.h"
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DECL(2) DECL(4) DECL(8) DECL(16)
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cb(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}
#define KB 256
static double x[256*KB] __attribute__((aligned(64)));
static double out[256*KB] __attribute__((aligned(64)));
static void reg_fill(rfft_codelets_t*c){ memset(c,0,sizeof *c);
#define R(r) c->r2cf[r]=radix##r##_r2cf_avx512; c->hc2hc[r]=radix##r##_hc2hc_dit_fwd_avx512;
  R(2) R(4) R(8) R(16) }

static void run_stage(const rfft_plan_t*p,int d,const double*cur,double*nxt){
  const rfft_stage_t*st=&p->st[d];
  const int r=st->radix,m=st->m,np=st->np; const size_t Q=st->Q,vl=st->vl,K=p->K;
  const int N=p->N;
  st->k0(cur,nxt,nxt+(size_t)N*K,(ptrdiff_t)vl,(ptrdiff_t)((size_t)m*vl),-(ptrdiff_t)((size_t)m*vl),vl);
  for(int k=1;k<=st->kmax;k++){
    const double*tr=st->tw_re+(size_t)(k-1)*(size_t)r*vl;
    const double*ti=st->tw_im+(size_t)(k-1)*(size_t)r*vl;
    st->hc(cur+(Q*(size_t)(r*k))*K,cur+(Q*(size_t)(r*(m-k)))*K,
           nxt+(Q*(size_t)k)*K,nxt+(Q*(size_t)(m-k))*K,
           tr,ti,(ptrdiff_t)vl,(ptrdiff_t)((size_t)m*vl),vl);
  }
  if(st->has_mid){
    const double*mc=st->mid_c,*ms=st->mid_s;
    for(size_t q=0;q<Q;q++) for(size_t lane=0;lane<K;lane++){
      double Xr[16],Xi[16];
      for(int s=0;s<r;s++){ double sr=0,si=0;
        for(int j=0;j<r;j++){ double c=cur[(q+Q*(size_t)(j+r*(m/2)))*K+lane];
          sr+=c*mc[s*r+j]; si+=c*ms[s*r+j]; } Xr[s]=sr; Xi[s]=si; }
      for(int s=0;s<r;s++){ int pp=m/2+s*m;
        nxt[(q+Q*(size_t)pp)*K+lane]=(pp<=np/2)?Xr[s]:Xi[r-1-s]; }
    }
  }
}
static double tmed(void(*f)(void*),void*a){
  for(int w=0;w<5;w++)f(a);
  double b=1e30; for(int t=0;t<5;t++){cb();double s=now_ns();for(int i=0;i<200;i++)f(a);double n=(now_ns()-s)/200;if(n<b)b=n;} return b;}
typedef struct{rfft_plan_t*p;int d;const double*in;double*o;}arg_t;
static void f_leaf(void*v){arg_t*a=v; rfft_plan_t*p=a->p;
  p->leaf(x,p->planeA,p->planeA+(size_t)p->N*p->K,(ptrdiff_t)(p->S*p->K),(ptrdiff_t)(p->S*p->K),-(ptrdiff_t)(p->S*p->K),p->S*p->K);}
static void f_stage(void*v){arg_t*a=v; run_stage(a->p,a->d,a->in,a->o);}
static void f_full(void*v){arg_t*a=v; rfft_execute_fwd_packed(a->p,x,out);}
int main(void){
  rfft_codelets_t reg; reg_fill(&reg);
  for(size_t i=0;i<256*KB;i++) x[i]=sin(0.37*(double)i);

  /* phase decomposition of (4,4,16) */
  int fa[]={4,4,16};
  rfft_plan_t*p=rfft_plan_create(256,KB,fa,3,&reg);
  arg_t aL={p,0,0,0};
  double tl=tmed(f_leaf,&aL);
  arg_t a1={p,1,p->planeA,p->planeB}; double t1=tmed(f_stage,&a1);
  arg_t a0={p,0,p->planeB,out};       double t0=tmed(f_stage,&a0);
  arg_t af={p,0,0,0};                 double tf=tmed(f_full,&af);
  printf("(4,4,16) phases: leaf %.0f | d=1 %.0f | d=0 %.0f | sum %.0f | full %.0f ns\n",
         tl,t1,t0,tl+t1+t0,tf);
  printf("  leaf share of full: %.0f%%\n", 100.0*tl/tf);
  rfft_plan_destroy(p);

  /* factor-order sweep */
  struct { int f[4]; int nf; } cells[] = {
    {{4,4,16},3},{{4,16,4},3},{{16,4,4},3},
    {{8,8,4},3},{{4,8,8},3},{{8,4,8},3},
    {{2,4,4,8},4},{{8,4,4,2},4},{{4,4,8,2},4},{{16,16},2},
  };
  for(unsigned c=0;c<sizeof cells/sizeof cells[0];c++){
    rfft_plan_t*q=rfft_plan_create(256,KB,cells[c].f,cells[c].nf,&reg);
    if(!q){printf("plan fail\n");continue;}
    arg_t a={q,0,0,0}; double b=tmed(f_full,&a);
    char fs[32]; int o=0;
    for(int i=0;i<cells[c].nf;i++) o+=snprintf(fs+o,32-o,"%d%s",cells[c].f[i],i<cells[c].nf-1?",":"");
    printf("  (%-9s): %8.0f ns\n",fs,b);
    rfft_plan_destroy(q);
  }
  return 0;
}
