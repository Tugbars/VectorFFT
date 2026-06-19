/* test_oop_execute.c — gates for core/oop_execute.h (Mode B OOP).
 *
 * Gate 1 (fwd): vfft_proto_execute_fwd_oop output must be BIT-IDENTICAL to
 *   vfft_proto_execute_fwd_generic run in-place on a copy (same codelets,
 *   same arithmetic, different memory), and src must be untouched.
 *   The generic loop is the reference because the tier-1 plan-shaped
 *   executors are not guaranteed bit-identical to it.
 * Gate 2 (bwd): vfft_proto_execute_bwd_oop must be bit-identical to the
 *   equivalent in-place swap dataflow (copy with re/im swapped, generic
 *   fwd, swap back).
 * Timing: same-binary round-robin: generic in-place vs public fwd (tier-1
 *   dispatch) vs OOP. Expect oop/generic ~ 1.00.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>
#include "../core/executor.h"
#include "../core/planner.h"
#include "../core/oop_execute.h"

static unsigned long long mn2(unsigned long long a,unsigned long long b){return a<b?a:b;}

static int run_cell(int N,size_t K,const int*factors,int nf,vfft_proto_registry_t*reg){
  stride_plan_t*plan=vfft_proto_plan_create(N,K,factors,NULL,nf,reg);
  if(!plan){printf("  N=%-6d plan FAIL\n",N);return 1;}
  size_t T=(size_t)N*K;
  double*sr=aligned_alloc(64,T*8),*si=aligned_alloc(64,T*8);
  double*s0r=malloc(T*8),*s0i=malloc(T*8);
  double*rr=aligned_alloc(64,T*8),*ri=aligned_alloc(64,T*8);
  double*dr=aligned_alloc(64,T*8),*di=aligned_alloc(64,T*8);
  srand(11+N);
  for(size_t i=0;i<T;i++){sr[i]=(double)rand()/RAND_MAX-0.5;si[i]=(double)rand()/RAND_MAX-0.5;}
  memcpy(s0r,sr,T*8);memcpy(s0i,si,T*8);

  /* Gate 1: fwd vs generic in-place */
  memcpy(rr,sr,T*8);memcpy(ri,si,T*8);
  vfft_proto_execute_fwd_generic(plan,rr,ri,K);
  memset(dr,0xCD,T*8);memset(di,0xCD,T*8);
  int rc=vfft_proto_execute_fwd_oop(plan,sr,si,dr,di,K);
  int g1 = rc==0 && memcmp(dr,rr,T*8)==0 && memcmp(di,ri,T*8)==0;
  int gp = memcmp(sr,s0r,T*8)==0 && memcmp(si,s0i,T*8)==0;

  /* Gate 2: bwd via swap == in-place swap dataflow */
  memcpy(rr,si,T*8);memcpy(ri,sr,T*8);          /* swapped copy */
  vfft_proto_execute_fwd_generic(plan,rr,ri,K);
  memset(dr,0xCD,T*8);memset(di,0xCD,T*8);
  int rcb=vfft_proto_execute_bwd_oop(plan,sr,si,dr,di,K);
  int g2 = rcb==0 && memcmp(di,rr,T*8)==0 && memcmp(dr,ri,T*8)==0;

  /* Timing, same binary, round-robin. Three contenders with matched
   * dataflow semantics:
   *   pure    : in-place generic on a hot 2-array working set (lower bound,
   *             different semantics: destroys input)
   *   cp+ip   : memcpy src->dst then in-place on dst — the workflow OOP
   *             replaces for an input-preserving caller
   *   oop     : vfft_proto_execute_fwd_oop direct                       */
  enum{ROUNDS=20};
  unsigned long long mg=~0ULL,mc=~0ULL,mo=~0ULL,c;
  for(int w=0;w<3;w++){
    vfft_proto_execute_fwd_generic(plan,rr,ri,K);
    memcpy(dr,sr,T*8);memcpy(di,si,T*8);vfft_proto_execute_fwd_generic(plan,dr,di,K);
    vfft_proto_execute_fwd_oop(plan,sr,si,dr,di,K);
  }
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc();vfft_proto_execute_fwd_generic(plan,rr,ri,K);mg=mn2(mg,__rdtsc()-c);
    c=__rdtsc();
    memcpy(dr,sr,T*8);memcpy(di,si,T*8);
    vfft_proto_execute_fwd_generic(plan,dr,di,K);
    mc=mn2(mc,__rdtsc()-c);
    c=__rdtsc();vfft_proto_execute_fwd_oop(plan,sr,si,dr,di,K);mo=mn2(mo,__rdtsc()-c);
  }
  printf("  N=%-5d K=%-4zu nf=%d  fwd %s  preserve %s  bwd-swap %s | speed vs cp+ip: oop %.3f pure-ip %.3f\n",
    N,K,nf,g1?"BITEXACT":"FAIL",gp?"OK":"FAIL",g2?"BITEXACT":"FAIL",
    (double)mc/mo,(double)mc/mg);
  free(sr);free(si);free(s0r);free(s0i);free(rr);free(ri);free(dr);free(di);
  return !(g1&&gp&&g2);
}

int main(void){
  vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
  int bad=0;
  { int f[]={8,8,16};      bad+=run_cell(1024,64,f,3,&reg); }
  { int f[]={16,16,16};    bad+=run_cell(4096,64,f,3,&reg); }
  { int f[]={32,32};       bad+=run_cell(1024,64,f,2,&reg); }
  { int f[]={2,3,5,7,11};  bad+=run_cell(2310,32,f,5,&reg); }
  { int f[]={20,10,10};    bad+=run_cell(2000,32,f,3,&reg); }
  { int f[]={13,13};       bad+=run_cell(169,256,f,2,&reg); }
  printf(bad?"SOME GATES FAILED\n":"ALL GATES PASS\n");
  return bad;
}
