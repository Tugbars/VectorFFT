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

static double tmed(void(*f)(void*),void*a){
  for(int w=0;w<5;w++)f(a);
  double b=1e30; for(int t=0;t<5;t++){cb();double s=now_ns();for(int i=0;i<200;i++)f(a);double n=(now_ns()-s)/200;if(n<b)b=n;} return b;}
typedef struct{rfft_plan_t*p;int d;const double*in;double*o;}arg_t;
static void f_full(void*v){arg_t*a=v; rfft_execute_fwd_packed(a->p,x,out);}
int main(void){
  rfft_codelets_t reg; reg_fill(&reg);
  for(size_t i=0;i<256*KB;i++) x[i]=sin(0.37*(double)i);

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
