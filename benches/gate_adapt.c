/* Adapter-level gate: 6a16 boundary folds vs generic executors, bit paths. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "prime_dispatch.h"
#include "plan_orchestrator.h"
#include "il_layout.h"
#include "il_execute.h"
#include "oop_execute.h"
#include "jit_runtime.h"
#include "generator/generated/registry.h"

static int ulp_ok(double a,double b){
    if(a==b) return 1;
    double m=fabs(a)>fabs(b)?fabs(a):fabs(b); if(m<1.0)m=1.0;
    if(fabs(a-b)<=4.0*2.220446049250313e-16*m) return 1;
    long long x,y; memcpy(&x,&a,8); memcpy(&y,&b,8);
    if((x<0)!=(y<0)) return 0;
    long long d=x-y; if(d<0)d=-d; return d<=4; }
static int cmpv(const double*a,const double*b,size_t n,const char*lbl){
    size_t bit=0,ok=0,bad=0;
    for(size_t i=0;i<n;i++){ if(a[i]==b[i]){bit++;continue;}
        if(ulp_ok(a[i],b[i]))ok++; else bad++; }
    printf("    %-14s bit=%zu ulp=%zu BAD=%zu %s\n",lbl,bit,ok,bad,bad?"**FAIL**":(ok?"ULP":"BIT"));
    return bad==0; }
static void fillr(double*p,size_t n,int seed){srand(seed);
    for(size_t i=0;i<n;i++)p[i]=2.0*rand()/RAND_MAX-1;}

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t *w=calloc(1,sizeof *w);
    if(vfft_proto_wisdom_load(w,"/tmp/wad.txt")){puts("wload fail");return 1;}
    struct { int N; size_t K; const char*lbl; } cs[] = {
        {1024,4,"DIT T1S-exit"},{1024,8,"DIT LOG3-exit"},{1024,16,"DIT FLAT-exit"},
        {1024,67,"DIT T1S K=67"},{100,4,"DIF LOG3"},{100,8,"DIF FLAT"},{100,67,"DIF L3 K=67"}};
    int all=1;
    for(int c=0;c<7;c++){
        int N=cs[c].N; size_t K=cs[c].K, NK=(size_t)N*K;
        vfft_proto_handle_t h;
        if(vfft_proto_plan(&h,N,K,VFFT_PROTO_WISDOM_ONLY,&reg,w,NULL)){
            printf("  %s: plan fail\n",cs[c].lbl); all=0; continue; }
        int dif = h.plan->use_dif_forward;
        vfft_proto_exec_fn cff = vfft_proto_plan_jit_fwd(h.plan);
        vfft_proto_exec_fn cfb = vfft_proto_plan_jit_bwd(h.plan);
        printf("  [%d,%zu] %s (dif=%d, stages=%d)\n",N,K,cs[c].lbl,dif,h.plan->num_stages);
        double *sr=aligned_alloc(64,NK*8),*si=aligned_alloc(64,NK*8);
        double *zi=aligned_alloc(64,2*NK*8);
        double *r1=aligned_alloc(64,NK*8),*i1=aligned_alloc(64,NK*8);
        double *r2=aligned_alloc(64,NK*8),*i2=aligned_alloc(64,NK*8);
        double *z1=aligned_alloc(64,2*NK*8),*z2=aligned_alloc(64,2*NK*8);
        fillr(sr,NK,1000+c); fillr(si,NK,2000+c); fillr(zi,2*NK,3000+c);
        /* T1 fwd_ilout */
        memcpy(r1,sr,NK*8); memcpy(i1,si,NK*8); memset(z1,0xA5,2*NK*8);
        int rc=vfft_proto_execute_fwd_ilout_core(h.plan,r1,i1,z1,K);
        if(rc){printf("    fwd_ilout      rc=%d **REJECT**\n",rc);all=0;}
        else { memcpy(r2,sr,NK*8); memcpy(i2,si,NK*8);
            if(dif) vfft_proto_execute_fwd_generic_dif(h.plan,r2,i2,K);
            else    vfft_proto_execute_fwd_generic_from(h.plan,r2,i2,K,0);
            for(size_t i=0;i<NK;i++){z2[2*i]=r2[i];z2[2*i+1]=i2[i];}
            all &= cmpv(z1,z2,2*NK,"fwd_ilout"); }
        /* T2 fwd_ilin generic-remainder + jit-remainder */
        memset(r1,0,NK*8); memset(i1,0,NK*8);
        rc=vfft_proto_execute_fwd_ilin_core(h.plan,zi,r1,i1,K,NULL);
        if(rc){printf("    fwd_ilin(gen)  rc=%d **REJECT**\n",rc);all=0;}
        else { for(size_t i=0;i<NK;i++){r2[i]=zi[2*i];i2[i]=zi[2*i+1];}
            if(dif) vfft_proto_execute_fwd_generic_dif(h.plan,r2,i2,K);
            else    vfft_proto_execute_fwd_generic_from(h.plan,r2,i2,K,0);
            all &= cmpv(r1,r2,NK,"fwd_ilin(gen)R") & cmpv(i1,i2,NK,"fwd_ilin(gen)I"); }
        if(dif && cff){
            memset(r1,0,NK*8); memset(i1,0,NK*8);
            rc=vfft_proto_execute_fwd_ilin_core(h.plan,zi,r1,i1,K,cff);
            if(rc){printf("    fwd_ilin(jit)  rc=%d **REJECT**\n",rc);all=0;}
            else all &= cmpv(r1,r2,NK,"fwd_ilin(jit)R") & cmpv(i1,i2,NK,"fwd_ilin(jit)I"); }
        /* T3 bwd_ilout generic + jit */
        memcpy(r1,sr,NK*8); memcpy(i1,si,NK*8); memset(z1,0xA5,2*NK*8);
        rc=vfft_proto_execute_bwd_ilout_core(h.plan,r1,i1,z1,K,NULL);
        if(rc){printf("    bwd_ilout(gen) rc=%d **REJECT**\n",rc);all=0;}
        else { memcpy(r2,sr,NK*8); memcpy(i2,si,NK*8);
            if(dif) vfft_proto_execute_bwd_generic_dif(h.plan,r2,i2,K);
            else    vfft_proto_execute_bwd_generic(h.plan,r2,i2,K);
            for(size_t i=0;i<NK;i++){z2[2*i]=r2[i];z2[2*i+1]=i2[i];}
            all &= cmpv(z1,z2,2*NK,"bwd_ilout(gen)"); }
        if(dif && cfb){
            memcpy(r1,sr,NK*8); memcpy(i1,si,NK*8); memset(z1,0xA5,2*NK*8);
            rc=vfft_proto_execute_bwd_ilout_core(h.plan,r1,i1,z1,K,cfb);
            if(rc){printf("    bwd_ilout(jit) rc=%d **REJECT**\n",rc);all=0;}
            else all &= cmpv(z1,z2,2*NK,"bwd_ilout(jit)"); }
        /* T5 fwd_il2il */
        memset(r1,0,NK*8); memset(i1,0,NK*8); memset(z1,0xA5,2*NK*8);
        rc=vfft_proto_execute_fwd_il2il_core(h.plan,zi,r1,i1,z1,K);
        if(rc){printf("    fwd_il2il      rc=%d **REJECT**\n",rc);all=0;}
        else { for(size_t i=0;i<NK;i++){r2[i]=zi[2*i];i2[i]=zi[2*i+1];}
            if(dif) vfft_proto_execute_fwd_generic_dif(h.plan,r2,i2,K);
            else    vfft_proto_execute_fwd_generic_from(h.plan,r2,i2,K,0);
            for(size_t i=0;i<NK;i++){z2[2*i]=r2[i];z2[2*i+1]=i2[i];}
            all &= cmpv(z1,z2,2*NK,"fwd_il2il"); }
        /* T5b fwd_il2il in-place z */
        memcpy(z1,zi,2*NK*8); memset(r1,0,NK*8); memset(i1,0,NK*8);
        rc=vfft_proto_execute_fwd_il2il_core(h.plan,z1,r1,i1,z1,K);
        if(rc){printf("    fwd_il2il(ip)  rc=%d **REJECT**\n",rc);all=0;}
        else all &= cmpv(z1,z2,2*NK,"fwd_il2il(ip)");
        /* T6 bwd_il2il */
        memset(r1,0,NK*8); memset(i1,0,NK*8); memset(z1,0xA5,2*NK*8);
        rc=vfft_proto_execute_bwd_il2il_core(h.plan,zi,r1,i1,z1,K);
        if(rc){printf("    bwd_il2il      rc=%d **REJECT**\n",rc);all=0;}
        else { for(size_t i=0;i<NK;i++){r2[i]=zi[2*i];i2[i]=zi[2*i+1];}
            if(dif) vfft_proto_execute_bwd_generic_dif(h.plan,r2,i2,K);
            else    vfft_proto_execute_bwd_generic(h.plan,r2,i2,K);
            for(size_t i=0;i<NK;i++){z2[2*i]=r2[i];z2[2*i+1]=i2[i];}
            all &= cmpv(z1,z2,2*NK,"bwd_il2il"); }
        /* T4 bwd_ilin */
        memset(r1,0,NK*8); memset(i1,0,NK*8);
        rc=vfft_proto_execute_bwd_ilin_core(h.plan,zi,r1,i1,K);
        if(rc){printf("    bwd_ilin       rc=%d **REJECT**\n",rc);all=0;}
        else { for(size_t i=0;i<NK;i++){r2[i]=zi[2*i];i2[i]=zi[2*i+1];}
            if(dif) vfft_proto_execute_bwd_generic_dif(h.plan,r2,i2,K);
            else    vfft_proto_execute_bwd_generic(h.plan,r2,i2,K);
            all &= cmpv(r1,r2,NK,"bwd_ilin R") & cmpv(i1,i2,NK,"bwd_ilin I"); }
        /* T11 oop jit-inner symmetry: {fwd,bwd}_oop_jit vs generic inner */
        if(!dif){
            if(!cff){ printf("    oop_jit        cff nil **RESOLVE FAIL**\n"); all=0; }
            else {
                memcpy(z1,sr,NK*8); memcpy(z1+NK,si,NK*8); /* src snapshot */
                memset(r1,0,NK*8); memset(i1,0,NK*8); memset(r2,0,NK*8); memset(i2,0,NK*8);
                int ra=vfft_proto_execute_fwd_oop(h.plan,sr,si,r2,i2,K);
                int rb=vfft_proto_execute_fwd_oop_jit(h.plan,sr,si,r1,i1,K,cff);
                if(ra||rb){printf("    fwd_oop        rc=%d/%d **REJECT**\n",ra,rb);all=0;}
                else all &= cmpv(r1,r2,NK,"fwd_oop(jit)R") & cmpv(i1,i2,NK,"fwd_oop(jit)I");
                ra=vfft_proto_execute_bwd_oop(h.plan,sr,si,r2,i2,K);
                rb=vfft_proto_execute_bwd_oop_jit(h.plan,sr,si,r1,i1,K,cff);
                if(ra||rb){printf("    bwd_oop        rc=%d/%d **REJECT**\n",ra,rb);all=0;}
                else all &= cmpv(r1,r2,NK,"bwd_oop(jit)R") & cmpv(i1,i2,NK,"bwd_oop(jit)I");
                int sp = !memcmp(z1,sr,NK*8) && !memcmp(z1+NK,si,NK*8);
                printf("    oop src-keep   %s\n", sp?"OK":"**CLOBBERED**"); all &= sp;
            }
        } else {
            int ra=vfft_proto_execute_fwd_oop_jit(h.plan,sr,si,r1,i1,K,cff);
            int rb=vfft_proto_execute_bwd_oop_jit(h.plan,sr,si,r1,i1,K,cff);
            printf("    oop DIF-reject %s\n",(ra==-1&&rb==-1)?"OK":"**MISSED**");
            all &= (ra==-1&&rb==-1);
        }
        /* ── 6a17 jit-tier arms: adapter(jit) vs full-jit reference ── */
        vfft_proto_exec_range_fn rff = vfft_proto_plan_jit_fwd_range(h.plan);
        vfft_proto_exec_range_fn rfb = vfft_proto_plan_jit_bwd_range(h.plan);
        if(!rff||!rfb){ printf("    jit range      fwd=%s bwd=%s **RESOLVE FAIL**\n",
                               rff?"ok":"nil", rfb?"ok":"nil"); all=0; }
        else {
            const int BIG=0x7fffffff;
            /* T7 fwd_ilout_jit vs full-jit fwd */
            memcpy(r1,sr,NK*8); memcpy(i1,si,NK*8); memset(z1,0xA5,2*NK*8);
            rc=vfft_proto_execute_fwd_ilout_jit(h.plan,r1,i1,z1,K,rff);
            if(rc){printf("    fwd_ilout(jit) rc=%d **REJECT**\n",rc);all=0;}
            else { memcpy(r2,sr,NK*8); memcpy(i2,si,NK*8);
                rff(h.plan,r2,i2,K,h.plan->K,0,BIG);
                for(size_t i=0;i<NK;i++){z2[2*i]=r2[i];z2[2*i+1]=i2[i];}
                all &= cmpv(z1,z2,2*NK,"fwd_ilout(jitT)"); }
            /* T8 bwd_ilin_jit2 vs full-jit bwd */
            for(size_t i=0;i<NK;i++){r2[i]=zi[2*i];i2[i]=zi[2*i+1];}
            rfb(h.plan,r2,i2,K,h.plan->K,0,BIG);
            memset(r1,0,NK*8); memset(i1,0,NK*8);
            rc=vfft_proto_execute_bwd_ilin_jit2(h.plan,zi,r1,i1,K,rfb);
            if(rc){printf("    bwd_ilin(jit)  rc=%d **REJECT**\n",rc);all=0;}
            else all &= cmpv(r1,r2,NK,"bwd_ilin(jitT)R") & cmpv(i1,i2,NK,"bwd_ilin(jitT)I");
            /* T9 fwd_il2il_jit vs full-jit fwd */
            memset(r1,0,NK*8); memset(i1,0,NK*8); memset(z1,0xA5,2*NK*8);
            rc=vfft_proto_execute_fwd_il2il_jit(h.plan,zi,r1,i1,z1,K,rff);
            if(rc){printf("    fwd_il2il(jit) rc=%d **REJECT**\n",rc);all=0;}
            else { for(size_t i=0;i<NK;i++){r2[i]=zi[2*i];i2[i]=zi[2*i+1];}
                rff(h.plan,r2,i2,K,h.plan->K,0,BIG);
                for(size_t i=0;i<NK;i++){z2[2*i]=r2[i];z2[2*i+1]=i2[i];}
                all &= cmpv(z1,z2,2*NK,"fwd_il2il(jitT)"); }
            /* T10 bwd_il2il_jit vs full-jit bwd (+ in-place z) */
            memset(r1,0,NK*8); memset(i1,0,NK*8); memset(z1,0xA5,2*NK*8);
            rc=vfft_proto_execute_bwd_il2il_jit(h.plan,zi,r1,i1,z1,K,rfb);
            if(rc){printf("    bwd_il2il(jit) rc=%d **REJECT**\n",rc);all=0;}
            else { for(size_t i=0;i<NK;i++){r2[i]=zi[2*i];i2[i]=zi[2*i+1];}
                rfb(h.plan,r2,i2,K,h.plan->K,0,BIG);
                for(size_t i=0;i<NK;i++){z2[2*i]=r2[i];z2[2*i+1]=i2[i];}
                all &= cmpv(z1,z2,2*NK,"bwd_il2il(jitT)");
                memcpy(z1,zi,2*NK*8); memset(r1,0,NK*8); memset(i1,0,NK*8);
                rc=vfft_proto_execute_bwd_il2il_jit(h.plan,z1,r1,i1,z1,K,rfb);
                if(rc){printf("    bwd_il2il(jip) rc=%d **REJECT**\n",rc);all=0;}
                else all &= cmpv(z1,z2,2*NK,"bwd_il2il(jip)"); }
        }
        free(sr);free(si);free(zi);free(r1);free(i1);free(r2);free(i2);free(z1);free(z2);
    }
    puts(all?"ADAPTER GATE: ALL PASS":"ADAPTER GATE: FAILURES");
    return all?0:1;
}
