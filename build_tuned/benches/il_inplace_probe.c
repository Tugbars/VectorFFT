/* il_inplace_probe.c — A3: are the Bailey-tier IL routes alias-safe in==out?
 *
 * STRUCTURAL READ (2026-08-02): il2p is zin->mid->zout (input read only by
 * pass 1, output written only by pass 2, internal scratch between) and il3p
 * is zin->mid1->mid2->zout — both the same shadow-plane shape as zturn and
 * MKL's in-place. MONO is the opposite: a single monolithic codelet whose
 * emitted signature carries __restrict__ on the inputs, so an aliased call is
 * UB — the natorder program's LEAF-IP/T2/T3 lesson, verdict REFUSE, not
 * probe. This probe verifies the two structural claims empirically; nothing
 * ships on a structure read any more than on a header comment.
 *
 * memcmp aliased-vs-OOP, both directions. bwd uses the shipped composition
 * (t2t, F-DIAG fallback). Correctness probe, no timing.
 * Build: python build_tuned/build.py --src build_tuned/benches/il_inplace_probe.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "il2p.h"

static double *az(size_t n){
#ifdef _WIN32
    return (double*)_aligned_malloc(2*n*sizeof(double),64);
#else
    void*p=NULL; if(posix_memalign(&p,64,2*n*sizeof(double)))p=NULL; return (double*)p;
#endif
}
static void fz(double*p){
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
static int g_fail=0;
static const char* chk(const double*w,const double*g,int N){
    if(!memcmp(w,g,2*(size_t)N*sizeof(double))) return "EXACT";
    g_fail++; return "*** DIFFER ***";
}

int main(void){
    printf("\n=== A3: Bailey-tier IL alias-safety (plan level, memcmp) ===\n");
    printf("%-7s %-14s %-14s %-14s\n","N","route","fwd","bwd");

    /* il2p pairs */
    static const int P2[][3]={{256,16,16},{1024,16,64},{2048,32,64},{4096,64,64}};
    for(size_t i=0;i<sizeof P2/sizeof P2[0];i++){
        const int N=P2[i][0],R1=P2[i][1],R2=P2[i][2];
        vfft_il2p_plan_t*p=vfft_il2p_create(N,R1,R2);
        if(!p){printf("%-7d il2p %dx%d create REFUSED\n",N,R1,R2);continue;}
        double*x=az((size_t)N),*r=az((size_t)N),*a=az((size_t)N),*rb=az((size_t)N);
        srand(3+N); for(int k=0;k<2*N;k++)x[k]=(double)rand()/RAND_MAX-0.5;
        vfft_il2p_execute_fwd(p,x,r);
        memcpy(a,x,2*(size_t)N*sizeof(double));
        vfft_il2p_execute_fwd(p,a,a);
        const char*vf=chk(r,a,N);
        vfft_il2p_execute_bwd(p,r,rb);
        memcpy(a,r,2*(size_t)N*sizeof(double));
        vfft_il2p_execute_bwd(p,a,a);
        const char*vb=chk(rb,a,N);
        char lab[20]; snprintf(lab,sizeof lab,"il2p %dx%d",R1,R2);
        printf("%-7d %-14s %-14s %-14s\n",N,lab,vf,vb);
        fz(x);fz(r);fz(a);fz(rb); vfft_il2p_destroy(p);
    }

    /* il3p via its default chain */
    static const int P3[]={512,1024,2048,3072};
    for(size_t i=0;i<sizeof P3/sizeof P3[0];i++){
        const int N=P3[i];
        int R2=0,A=0,B=0;
        if(!vfft_il3p_default_chain(N,&R2,&A,&B)){
            printf("%-7d il3p (no default chain — SKIP)\n",N);continue;}
        vfft_il3p_plan_t*p=vfft_il3p_create(N,R2,A,B);
        if(!p){printf("%-7d il3p %d.%d.%d create REFUSED\n",N,R2,A,B);continue;}
        double*x=az((size_t)N),*r=az((size_t)N),*a=az((size_t)N),*rb=az((size_t)N);
        srand(7+N); for(int k=0;k<2*N;k++)x[k]=(double)rand()/RAND_MAX-0.5;
        vfft_il3p_execute_fwd(p,x,r);
        memcpy(a,x,2*(size_t)N*sizeof(double));
        vfft_il3p_execute_fwd(p,a,a);
        const char*vf=chk(r,a,N);
        vfft_il3p_execute_bwd(p,r,rb);
        memcpy(a,r,2*(size_t)N*sizeof(double));
        vfft_il3p_execute_bwd(p,a,a);
        const char*vb=chk(rb,a,N);
        char lab[20]; snprintf(lab,sizeof lab,"il3p %d.%d.%d",R2,A,B);
        printf("%-7d %-14s %-14s %-14s\n",N,lab,vf,vb);
        fz(x);fz(r);fz(a);fz(rb); vfft_il3p_destroy(p);
    }

    printf("\n=== %s ===\nmono: NOT probed — emitted __restrict__ makes aliasing UB; "
           "verdict is REFUSE\n(the classic path keeps serving in-place <=64, "
           "which is trivially cheap there).\n",
           g_fail?"*** FAIL ***":"ALL EXACT");
    return g_fail?1:0;
}
